"""The same ablation, but on a benchmark with room to measure.

The Fashion-MNIST run had only 14.7 points between the untrained floor
(71.36) and the mixed-batch ceiling (86.04). Every effect was squeezed into
that gap, so the rescue looked marginal even though catastrophic forgetting
was real (86 -> 54).

Three changes to open the gap:
  CIFAR-10 instead of Fashion-MNIST   random features are much weaker there
  a convnet instead of an MLP         a real ceiling instead of a toy one
  6000 steps instead of 2000          so the ceiling is actually reached

Everything else is identical to ablation_test.py, including the arms, so
the two runs are directly comparable.

Arms, each differing from the previous by exactly ONE component:
  none              untrained floor
  iid               mixed batches, the ceiling
  seq               class-ordered, unprotected, the failure
  gate              + selective gate
  gate+reh          + rehearsal from the rejected-item buffer
  gate+reh+guard    + canary rollback, the full layer

Every arm is LOCAL: no gradient crosses a layer boundary.

The gate is never disabled, because in stability.py the buffer fills only
when the gate REJECTS an item. Turning the gate off leaves the buffer empty
and silently disables rehearsal too. That bug wasted a run already.
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


LR = 1e-3
BATCH = 128
STEPS_PER_CLASS = 600            # 10 classes x 600 = 6000 online steps
N_CLASSES = 10
EARLY = [0, 1]
PROBE_EPOCHS = 4
PROBE_SUBSET = 10000
SEEDS = [0, 1, 2]


def device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


# ---------------- data ----------------

def load(train):
    """CIFAR-10 in [0,1]. Normalisation happens once, explicitly, right
    before the network. Augmenting normalised data corrupted an earlier
    run, so the two steps stay strictly separate."""
    ds = datasets.CIFAR10("./data", train=train, download=True,
                          transform=transforms.ToTensor())
    x, y = next(iter(DataLoader(ds, batch_size=len(ds))))
    return x, y


def normalize(x):
    return (x - 0.5) / 0.5


def augment(x, shift=0.12, scale=0.10):
    """Random shift, scale and horizontal flip on [0,1] tensors."""
    b = x.size(0)
    theta = torch.zeros(b, 2, 3, device=x.device)
    s = 1.0 + (torch.rand(b, device=x.device) * 2 - 1) * scale
    flip = torch.where(torch.rand(b, device=x.device) < 0.5, -1.0, 1.0)
    theta[:, 0, 0] = s * flip
    theta[:, 1, 1] = s
    theta[:, :, 2] = (torch.rand(b, 2, device=x.device) * 2 - 1) * shift
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False,
                         padding_mode="zeros").clamp(0.0, 1.0)


def class_indices(y):
    return {c: (y == c).nonzero(as_tuple=True)[0].tolist()
            for c in range(N_CLASSES)}


def build_plan(by_class, sequential, seed):
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

class LocalConvNet(nn.Module):
    """Convnet where every layer trains on its own loss. detach() at each
    boundary is what makes it local. Random conv features on CIFAR-10 are
    much weaker than random MLP features on Fashion-MNIST, which is the
    point: it lowers the floor."""

    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        specs = [(3, 64, False), (64, 128, True),
                 (128, 256, True), (256, 256, True)]
        self.convs = nn.ModuleList(
            [nn.Conv2d(i, o, 3, padding=1) for i, o, _ in specs])
        self.pools = [p for _, _, p in specs]
        self.norms = nn.ModuleList(
            [nn.BatchNorm2d(o) for _, o, _ in specs])
        self.heads = nn.ModuleList(
            [nn.Linear(o, n_classes) for _, o, _ in specs])
        self.widths = [o for _, o, _ in specs]

    def encode(self, x):
        h = x
        outs = []
        for conv, norm, pool in zip(self.convs, self.norms, self.pools):
            h = F.relu(norm(conv(h.detach())))
            if pool:
                h = F.max_pool2d(h, 2)
            outs.append(h)
        return outs

    def pooled(self, outs):
        return [h.mean(dim=(2, 3)) for h in outs]

    def features(self, x):
        with torch.no_grad():
            return torch.cat(self.pooled(self.encode(x)), dim=1)

    @property
    def feature_dim(self):
        return sum(self.widths)


def local_losses(net, v, y):
    """One cross-entropy per layer. Class-aware, the objective that forgets."""
    pooled = net.pooled(net.encode(v))
    return [F.cross_entropy(net.heads[i](pooled[i]), y)
            for i in range(len(net.convs))]


# ---------------- backend ----------------

class Backend:
    """Items are INDEX LISTS, so the rehearsal buffer stores almost nothing.
    Storing image tensors made an earlier run take 7.7 hours."""

    def __init__(self, xtr, ytr, dev, seed=0):
        torch.manual_seed(seed)
        self.x, self.y, self.dev = xtr, ytr, dev
        self.net = LocalConvNet().to(dev)
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
        return total, None

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
    base = dict(warmup=20, top_fraction=0.60, coherence_veto=1e9,
                loss_floor=0.0, steps_per_update=1)
    if name == "gate":
        return dict(base, rehearse_per_item=None, rehearse_every=10 ** 9,
                    guard=False)
    if name == "gate+reh":
        return dict(base, rehearse_per_item=3, rehearse_count=2,
                    rehearse_steps=1, anchor_size=100, buffer_size=600,
                    guard=False)
    return dict(base, rehearse_per_item=3, rehearse_count=2,
                rehearse_steps=1, anchor_size=100, buffer_size=600,
                guard=True, guard_per_item=200, canary_tolerance=0.5)


def run(name, seed, dev, data, by_class):
    xtr, ytr, xte, yte = data
    torch.manual_seed(seed)

    if name == "none":
        net = LocalConvNet().to(dev)
        out = dict(overall=probe(net, dev, xtr, ytr, xte, yte),
                   early=probe(net, dev, xtr, ytr, xte, yte, EARLY),
                   grad_steps=0, gate_fires=0, rehearsals=0, rollbacks=0,
                   minutes=0.0)
        del net
        return out

    backend = Backend(xtr, ytr, dev, seed)
    plan = build_plan(by_class, name != "iid", seed)
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
    print(f"device {dev}   CIFAR-10   local convnet   "
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
        spread = (max(r["overall"] for r in runs)
                  - min(r["overall"] for r in runs))
        results[name] = dict(label=label, spread=spread, **avg)
        print(f"  {name:>15}  overall {avg['overall']:6.2f}%  "
              f"(spread {spread:4.1f})  early {avg['early']:6.2f}%  "
              f"grad {avg['grad_steps']:.0f}  gate {avg['gate_fires']:.0f}  "
              f"reh {avg['rehearsals']:.0f}  rb {avg['rollbacks']:.1f}  "
              f"{avg['minutes']:.1f}m")

        if name == "iid":
            gap = avg["overall"] - results["none"]["overall"]
            print(f"\n    headroom: floor {results['none']['overall']:.2f}% "
                  f"to ceiling {avg['overall']:.2f}%  =  {gap:.1f} points")
            if gap < 25:
                print("    WARNING: still under 25 points. Effects will be "
                      "hard to distinguish from noise.\n")
            else:
                print("    good, enough room to measure.\n")

    print("\n" + "=" * 88)
    print(f"{'arm':>15} {'overall':>9} {'early':>8} {'grad':>7} "
          f"{'reh':>6} {'rb':>5}   description")
    print("-" * 88)
    for name, label in ARMS:
        r = results[name]
        print(f"{name:>15} {r['overall']:>8.2f}% {r['early']:>7.2f}% "
              f"{r['grad_steps']:>7.0f} {r['rehearsals']:>6.0f} "
              f"{r['rollbacks']:>5.1f}   {label}")
    print("=" * 88)

    floor = results["none"]["overall"]
    fail = results["seq"]["overall"]
    ceil = results["iid"]["overall"]
    span = ceil - fail

    print(f"\nfloor {floor:.2f}%   failure {fail:.2f}%   ceiling {ceil:.2f}%")
    print("\ncontribution of each component:")
    prev = fail
    for name, _ in ARMS[3:]:
        cur = results[name]["overall"]
        tot = 100.0 * (cur - fail) / span if span > 1e-6 else 0.0
        add = 100.0 * (cur - prev) / span if span > 1e-6 else 0.0
        flag = "  (BELOW untrained floor)" if cur < floor else ""
        print(f"   {name:>15}  {cur:6.2f}%   total {tot:5.1f}%   "
              f"this step {add:+5.1f}%{flag}")
        prev = cur

    print(f"""
Compare against the Fashion-MNIST run, where rehearsal gave +70.2% of the
gap, the gate alone HURT (-11.5%), and the guard was inside the noise
(+4.6% against a spread of 2.5).

  same pattern, bigger gap   -> the finding is real and now demonstrable:
      replay rescues local learning from sequential streams
  gate or guard now matter   -> they needed headroom to show up, which is
      itself worth knowing
  full layer below floor     -> still not a result, and the setup needs
      more training rather than a different mechanism
""")
    with open("headroom_test.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote headroom_test.json")