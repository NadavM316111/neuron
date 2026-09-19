"""Can the stability layer rescue catastrophic forgetting in a local network?

objective_test.py produced a clean failure: a class-aware local objective on
a class-ordered stream fell from 97.37% to 12.32%, which is chance on ten
classes. That is exactly the failure the stability layer was built for, now
reproduced in a fast local-learning setting with no backward pass through
the network.

This wraps it. Five arms:

  none        untrained floor
  sup-iid     class-aware, mixed batches. The ceiling.
  sup-seq     class-aware, one class at a time. The failure.
  sup-seq-g   the same stream, wrapped in stability.py.
  sup-seq-r   the same stream with rehearsal ONLY, no gate and no guard.
              This is the control that matters: if plain replay recovers
              just as much, the gate and guard are doing nothing and the
              only active ingredient is rehearsal.

The buffer stores INDEX LISTS, not image tensors. Holding tensors is what
made the CIFAR run take 7.7 hours.
"""

import time
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from stability import StabilityLayer
from objective_test import (
    LocalNet, load, normalize, augment, class_indices, build_plan,
    sup_losses, probe, device, N_CLASSES, STEPS_PER_CLASS, BATCH, LR, EARLY,
)


SEEDS = [0, 1]


class SupBackend:
    """Class-aware local learning behind the four stability callbacks.

    An item is a LIST OF INDICES. Images are fetched on demand, so the
    rehearsal buffer stays tiny.
    """

    def __init__(self, xtr, ytr, dev, seed=0):
        torch.manual_seed(seed)
        self.x, self.y, self.dev = xtr, ytr, dev
        self.net = LocalNet().to(dev)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.grad_steps = 0

    def _batch(self, indices):
        raw = self.x[indices].to(self.dev)
        y = self.y[indices].to(self.dev)
        return normalize(augment(raw)), y

    def score(self, indices):
        """Total local loss. High means unfamiliar. No coherence signal
        for images, so the veto stays off."""
        self.net.eval()
        with torch.no_grad():
            v, y = self._batch(indices)
            total = float(sum(l.item() for l in sup_losses(self.net, v, y)))
        return total, None

    def update(self, indices, steps):
        self.net.train()
        last = None
        for _ in range(steps):
            v, y = self._batch(indices)
            loss = sum(sup_losses(self.net, v, y))
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            last = float(loss.item())
            self.grad_steps += 1
        return last

    def snapshot(self):
        return {k: v.detach().clone()
                for k, v in self.net.state_dict().items()}

    def restore(self, state):
        self.net.load_state_dict(state)

    def on_rollback(self):
        for g in self.opt.param_groups:
            g["lr"] = max(1e-5, g["lr"] * 0.5)


def run_plain(ordering, seed, dev, data, by_class):
    """No stability layer. Every batch updates the network."""
    xtr, ytr, xte, yte = data
    backend = SupBackend(xtr, ytr, dev, seed)
    plan = build_plan(by_class, ordering == "seq", seed)

    t0 = time.time()
    for indices in plan:
        backend.update(indices, 1)

    out = dict(overall=probe(backend.net, dev, xtr, ytr, xte, yte),
               early=probe(backend.net, dev, xtr, ytr, xte, yte, EARLY),
               grad_steps=backend.grad_steps, gate_fires=len(plan),
               rollbacks=0, rehearsals=0,
               minutes=(time.time() - t0) / 60)
    del backend
    if dev == "mps":
        torch.mps.empty_cache()
    return out


def run_layered(seed, dev, data, by_class, full):
    """full=True  : gate + rehearsal + canary guard
       full=False : rehearsal only, the control"""
    xtr, ytr, xte, yte = data
    backend = SupBackend(xtr, ytr, dev, seed)
    plan = build_plan(by_class, True, seed)

    # Canary: fixed batches from the classes seen first.
    canary = [by_class[c][:BATCH] for c in EARLY]

    if full:
        layer = StabilityLayer(
            backend, canary=canary, seed=seed,
            warmup=20, top_fraction=0.60, coherence_veto=1e9,
            loss_floor=0.0, steps_per_update=1,
            rehearse_per_item=3, rehearse_count=2, rehearse_steps=1,
            anchor_size=60, buffer_size=300,
            guard=True, guard_per_item=100, canary_tolerance=0.5)
    else:
        # rehearsal only: gate always fires, guard off
        layer = StabilityLayer(
            backend, canary=None, seed=seed,
            warmup=0, top_fraction=1.0, coherence_veto=1e9,
            loss_floor=0.0, steps_per_update=1,
            rehearse_per_item=3, rehearse_count=2, rehearse_steps=1,
            anchor_size=60, buffer_size=300,
            guard=False)

    t0 = time.time()
    for indices in plan:
        layer.observe(indices)

    s = layer.summary()
    out = dict(overall=probe(backend.net, dev, xtr, ytr, xte, yte),
               early=probe(backend.net, dev, xtr, ytr, xte, yte, EARLY),
               grad_steps=backend.grad_steps,
               gate_fires=s["updates"], rollbacks=s["rollbacks"],
               rehearsals=s["rehearsals"],
               minutes=(time.time() - t0) / 60)
    del backend, layer
    if dev == "mps":
        torch.mps.empty_cache()
    return out


def run_none(seed, dev, data):
    xtr, ytr, xte, yte = data
    torch.manual_seed(seed)
    net = LocalNet().to(dev)
    out = dict(overall=probe(net, dev, xtr, ytr, xte, yte),
               early=probe(net, dev, xtr, ytr, xte, yte, EARLY),
               grad_steps=0, gate_fires=0, rollbacks=0, rehearsals=0,
               minutes=0.0)
    del net
    return out


ARMS = [
    ("none",      "untrained floor"),
    ("sup-iid",   "class-aware, mixed batches (ceiling)"),
    ("sup-seq",   "class-aware, one class at a time (the failure)"),
    ("sup-seq-r", "same stream, REHEARSAL ONLY (control)"),
    ("sup-seq-g", "same stream, FULL stability layer"),
]


def dispatch(name, seed, dev, data, by_class):
    if name == "none":
        return run_none(seed, dev, data)
    if name == "sup-iid":
        return run_plain("iid", seed, dev, data, by_class)
    if name == "sup-seq":
        return run_plain("seq", seed, dev, data, by_class)
    if name == "sup-seq-r":
        return run_layered(seed, dev, data, by_class, full=False)
    return run_layered(seed, dev, data, by_class, full=True)


if __name__ == "__main__":
    dev = device()
    print(f"device {dev}   {N_CLASSES * STEPS_PER_CLASS} steps of {BATCH}   "
          f"seeds {SEEDS}\n")

    xtr, ytr = load(True)
    xte, yte = load(False)
    data = (xtr, ytr, xte, yte)
    by_class = class_indices(ytr)

    results = {}
    for name, label in ARMS:
        runs = [dispatch(name, s, dev, data, by_class) for s in SEEDS]
        avg = {k: sum(r[k] for r in runs) / len(runs) for k in runs[0]}
        results[name] = dict(label=label, **avg)
        print(f"  {name:>10}  overall {avg['overall']:6.2f}%   "
              f"early {avg['early']:6.2f}%   "
              f"grad {avg['grad_steps']:.0f}  "
              f"gate {avg['gate_fires']:.0f}  "
              f"reh {avg['rehearsals']:.0f}  "
              f"rb {avg['rollbacks']:.1f}  "
              f"{avg['minutes']:.1f} min")

    print("\n" + "=" * 78)
    print(f"{'arm':>10} {'overall':>9} {'early 0,1':>11} {'grad':>7}   description")
    print("-" * 78)
    for name, label in ARMS:
        r = results[name]
        print(f"{name:>10} {r['overall']:>8.2f}% {r['early']:>10.2f}% "
              f"{r['grad_steps']:>7.0f}   {label}")
    print("=" * 78)

    fail = results["sup-seq"]["overall"]
    ceil = results["sup-iid"]["overall"]
    reh = results["sup-seq-r"]["overall"]
    full = results["sup-seq-g"]["overall"]

    def recovered(x):
        span = ceil - fail
        return 100.0 * (x - fail) / span if span > 1e-6 else 0.0

    print(f"\nfailure {fail:.2f}%   ceiling {ceil:.2f}%")
    print(f"  rehearsal only : {reh:6.2f}%  "
          f"({recovered(reh):5.1f}% of the gap recovered)")
    print(f"  full layer     : {full:6.2f}%  "
          f"({recovered(full):5.1f}% of the gap recovered)")
    print("""
  full layer recovers a lot        -> both halves of the vision working
      together: a network with no backward pass, learning from a
      sequential stream, held together by the stability layer.

  rehearsal-only recovers the same -> the gate and guard add nothing here
      and rehearsal is the only active ingredient. Still a result, and an
      honest one, but a much smaller claim.

  neither recovers much            -> the stability layer does not transfer
      from language to this setting. Worth knowing before building on it.

Note the grad column. The layered arms do MORE gradient steps because of
rehearsal, so they are not update-matched with sup-seq. If they win, part
of the win may just be extra training.
""")
    with open("guarded_test.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote guarded_test.json")