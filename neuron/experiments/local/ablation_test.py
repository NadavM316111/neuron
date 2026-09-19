"""What in the stability layer prevents catastrophic forgetting, and how much?

Previous run: a class-aware local objective on a class-ordered stream fell
from 97.37% to 12.32% on MNIST, and the full stability layer recovered it to
73.75%. Two problems with that as a claim:

  1. MNIST's untrained floor is 78.78%, ABOVE the rescued score. You cannot
     claim a win when doing nothing scores better. Fashion-MNIST is used
     here instead, where random features are much weaker.

  2. Credit was unassignable. The previous "rehearsal only" control was
     broken: in stability.py, items enter the buffer ONLY when the gate
     REJECTS them, so disabling the gate leaves the buffer permanently
     empty and no rehearsal ever runs.

The fix for (2): never disable the gate. Keep it and add the other
mechanisms one at a time, so each arm differs from the previous by exactly
one component.

  none        untrained floor
  iid         mixed batches, the ceiling
  seq         class-ordered, unprotected, the failure
  gate        + selective gate only
  gate+reh    + rehearsal from the rejected-item buffer
  gate+reh+guard  + canary rollback. The full layer.

Everything is LOCAL: no gradient crosses a layer boundary in any arm.
"""

import time
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from stability import StabilityLayer


HIDDEN = (256, 256, 128)
LR = 1e-3
BATCH = 128
STEPS_PER_CLASS = 200
N_CLASSES = 10
EARLY = [0, 1]
PROBE_EPOCHS = 3
PROBE_SUBSET = 10000
SEEDS = [0, 1, 2]


def device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


# ---------------- data ----------------

def load(train):
    """Fashion-MNIST in [0,1]. Normalisation is applied once, explicitly,
    just before the network. Augmenting normalised data broke an earlier
    run, so the two steps are kept strictly separate."""
    ds = datasets.FashionMNIST("./data", train=train, download=True,
                               transform=transforms.ToTensor())
    x, y = next(iter(DataLoader(ds, batch_size=len(ds))))
    return x, y


def normalize(x):
    return (x - 0.5) / 0.5


def augment(x, shift=0.12, scale=0.10):
    """Random shift and scale on [0,1] tensors. Vectorised, no PIL."""
    b = x.size(0)
    theta = torch.zeros(b, 2, 3, device=x.device)
    s = 1.0 + (torch.rand(b, device=x.device) * 2 - 1) * scale
    theta[:, 0, 0] = s
    theta[:, 1, 1] = s
    theta[:, :, 2] = (torch.rand(b, 2, device=x.device) * 2 - 1) * shift
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False,
                         padding_mode="zeros").clamp(0.0, 1.0)


def class_indices(y):
    return {c: (y == c).nonzero(as_tuple=True)[0].tolist()
            for c in range(N_CLASSES)}


def build_plan(by_class, sequential, seed):
    """A list of index lists, one per online step.

    sequential: each batch is one class, classes arrive in order 0..9
    iid:        each batch is a genuine mix of all classes
    """
    rng = random.Random(seed)
    plan = []
    if sequential:
        for c in range(N_CLASSES):
            pool = list(by_class[c])
            for _ in range(STEPS_PER_CLASS):
                plan.append([pool[rng.randrange(len(pool))]
                             for _ in range(BATCH)])
    else:
        every = [i for c in range(N_CLASSES) for i in by_class[c]]
        for _ in range(N_CLASSES * STEPS_PER_CLASS):
            plan.append([every[rng.randrange(len(every))]
                         for _ in range(BATCH)])
    return plan


# ---------------- model ----------------

class LocalNet(nn.Module):
    """MLP where every layer trains on its own loss. detach() at each
    boundary is what makes it local."""

    def __init__(self, in_dim=784, hidden=HIDDEN, n_classes=N_CLASSES):
        super().__init__()
        dims = [in_dim] + list(hidden)
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(hidden))])
        self.heads = nn.ModuleList(
            [nn.Linear(d, n_classes) for d in hidden])
        self.hidden = hidden

    def encode(self, x):
        h = x.flatten(1)
        outs = []
        for layer in self.layers:
            h = F.relu(layer(h.detach()))
            outs.append(h)
        return outs

    def features(self, x):
        with torch.no_grad():
            return torch.cat(self.encode(x), dim=1)

    @property
    def feature_dim(self):
        return sum(self.hidden)


def local_losses(net, v, y):
    """One cross-entropy per layer. Class-aware, which is the objective
    that forgets."""
    h = net.encode(v)
    return [F.cross_entropy(net.heads[i](h[i]), y)
            for i in range(len(net.layers))]


# ---------------- backend ----------------

class Backend:
    """Items are INDEX LISTS. Images are fetched on demand, so the
    rehearsal buffer stores almost nothing. Storing tensors is what made an
    earlier run take 7.7 hours."""

    def __init__(self, xtr, ytr, dev, seed=0):
        torch.manual_seed(seed)
        self.x, self.y, self.dev = xtr, ytr, dev
        self.net = LocalNet().to(dev)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.grad_steps = 0

    def _batch(self, indices):
        raw = self.x[indices].to(self.dev)
        return normalize(augment(raw)), self.y[indices].to(self.dev)

    def score(self, indices):
        self.net.eval()
        with torch.no_grad():
            v, y = self._batch(indices)
            total = float(sum(l.item() for l in local_losses(self.net, v, y)))
        return total, None                 # no coherence signal for images

    def update(self, indices, steps):
        self.net.train()
        last = None
        for _ in range(steps):
            v, y = self._batch(indices)
            loss = sum(local_losses(self.net, v, y))
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


# ---------------- evaluation ----------------

def probe(net, dev, xtr, ytr, xte, yte, classes=None):
    net.eval()
    clf = nn.Linear(net.feature_dim, N_CLASSES).to(dev)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)

    g = torch.Generator().manual_seed(0)
    sub = torch.randperm(xtr.size(0), generator=g)[:PROBE_SUBSET]
    tr = DataLoader(TensorDataset(xtr[sub], ytr[sub]), batch_size=256,
                    shuffle=True)
    for _ in range(PROBE_EPOCHS):
        for xb, yb in tr:
            xb, yb = normalize(xb.to(dev)), yb.to(dev)
            opt.zero_grad()
            F.cross_entropy(clf(net.features(xb)), yb).backward()
            opt.step()

    if classes is not None:
        mask = torch.zeros_like(yte, dtype=torch.bool)
        for c in classes:
            mask |= (yte == c)
        xe, ye = xte[mask], yte[mask]
    else:
        xe, ye = xte, yte

    correct = total = 0
    with torch.no_grad():
        for xb, yb in DataLoader(TensorDataset(xe, ye), batch_size=512):
            xb, yb = normalize(xb.to(dev)), yb.to(dev)
            correct += (clf(net.features(xb)).argmax(1) == yb).sum().item()
            total += yb.numel()
    return 100.0 * correct / total


# ---------------- arms ----------------

def arm_config(name):
    """Each arm differs from the previous by exactly ONE component.

    The gate is never disabled, because the buffer only fills on rejection.
    top_fraction=0.60 means the top 60% by loss are learned from and the
    rest go to the buffer.
    """
    base = dict(warmup=20, top_fraction=0.60, coherence_veto=1e9,
                loss_floor=0.0, steps_per_update=1)
    if name == "gate":
        return dict(base, rehearse_per_item=None, rehearse_every=10 ** 9,
                    guard=False)
    if name == "gate+reh":
        return dict(base, rehearse_per_item=3, rehearse_count=2,
                    rehearse_steps=1, anchor_size=60, buffer_size=300,
                    guard=False)
    return dict(base, rehearse_per_item=3, rehearse_count=2,
                rehearse_steps=1, anchor_size=60, buffer_size=300,
                guard=True, guard_per_item=100, canary_tolerance=0.5)


def run(name, seed, dev, data, by_class):
    xtr, ytr, xte, yte = data
    torch.manual_seed(seed)

    if name == "none":
        net = LocalNet().to(dev)
        out = dict(overall=probe(net, dev, xtr, ytr, xte, yte),
                   early=probe(net, dev, xtr, ytr, xte, yte, EARLY),
                   grad_steps=0, gate_fires=0, rehearsals=0, rollbacks=0,
                   minutes=0.0)
        del net
        return out

    backend = Backend(xtr, ytr, dev, seed)
    sequential = name != "iid"
    plan = build_plan(by_class, sequential, seed)
    t0 = time.time()

    if name in ("iid", "seq"):
        for indices in plan:
            backend.update(indices, 1)
        stats = dict(gate_fires=len(plan), rehearsals=0, rollbacks=0)
    else:
        canary = [by_class[c][:BATCH] for c in EARLY]
        layer = StabilityLayer(backend, canary=canary, seed=seed,
                               **arm_config(name))
        for indices in plan:
            layer.observe(indices)
        s = layer.summary()
        stats = dict(gate_fires=s["updates"], rehearsals=s["rehearsals"],
                     rollbacks=s["rollbacks"])
        del layer

    out = dict(overall=probe(backend.net, dev, xtr, ytr, xte, yte),
               early=probe(backend.net, dev, xtr, ytr, xte, yte, EARLY),
               grad_steps=backend.grad_steps,
               minutes=(time.time() - t0) / 60, **stats)
    del backend
    if dev == "mps":
        torch.mps.empty_cache()
    return out


ARMS = [
    ("none",           "untrained floor"),
    ("iid",            "mixed batches (ceiling)"),
    ("seq",            "class-ordered, unprotected (the failure)"),
    ("gate",           "+ selective gate"),
    ("gate+reh",       "+ rehearsal"),
    ("gate+reh+guard", "+ canary rollback (full layer)"),
]


if __name__ == "__main__":
    dev = device()
    print(f"device {dev}   Fashion-MNIST   MLP {HIDDEN}   "
          f"{N_CLASSES * STEPS_PER_CLASS} steps of {BATCH}   "
          f"seeds {SEEDS}\n")

    xtr, ytr = load(True)
    xte, yte = load(False)
    data = (xtr, ytr, xte, yte)
    by_class = class_indices(ytr)

    results = {}
    for name, label in ARMS:
        runs = [run(name, s, dev, data, by_class) for s in SEEDS]
        avg = {k: sum(r[k] for r in runs) / len(runs) for k in runs[0]}
        spread = max(r["overall"] for r in runs) - min(r["overall"] for r in runs)
        results[name] = dict(label=label, spread=spread, **avg)
        print(f"  {name:>15}  overall {avg['overall']:6.2f}%  "
              f"(spread {spread:4.1f})  early {avg['early']:6.2f}%  "
              f"grad {avg['grad_steps']:.0f}  gate {avg['gate_fires']:.0f}  "
              f"reh {avg['rehearsals']:.0f}  rb {avg['rollbacks']:.1f}  "
              f"{avg['minutes']:.1f}m")

    print("\n" + "=" * 86)
    print(f"{'arm':>15} {'overall':>9} {'early':>8} {'grad':>7} "
          f"{'reh':>6} {'rb':>5}   description")
    print("-" * 86)
    for name, label in ARMS:
        r = results[name]
        print(f"{name:>15} {r['overall']:>8.2f}% {r['early']:>7.2f}% "
              f"{r['grad_steps']:>7.0f} {r['rehearsals']:>6.0f} "
              f"{r['rollbacks']:>5.1f}   {label}")
    print("=" * 86)

    floor = results["none"]["overall"]
    fail = results["seq"]["overall"]
    ceil = results["iid"]["overall"]
    span = ceil - fail

    print(f"\nfloor {floor:.2f}%   failure {fail:.2f}%   ceiling {ceil:.2f}%")
    print("\ncontribution of each component, as % of the gap recovered:")
    prev = fail
    for name, label in ARMS[3:]:
        cur = results[name]["overall"]
        tot = 100.0 * (cur - fail) / span if span > 1e-6 else 0.0
        add = 100.0 * (cur - prev) / span if span > 1e-6 else 0.0
        flag = "  (below untrained floor)" if cur < floor else ""
        print(f"   {name:>15}  {cur:6.2f}%   total {tot:5.1f}%   "
              f"this step {add:+5.1f}%{flag}")
        prev = cur

    print("""
Read three things:

  1. Is the full layer ABOVE the untrained floor? If not, it prevented
     collapse but did not produce a network better than random features,
     and the claim stays small.

  2. Which step adds the most? If the gate alone recovers most of it, the
     answer is simply "do not learn from everything", which is a simpler
     and cleaner finding than a three-part mechanism.

  3. The spread column. With 3 seeds, differences smaller than the spread
     are noise, not signal.
""")
    with open("ablation_test.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote ablation_test.json")