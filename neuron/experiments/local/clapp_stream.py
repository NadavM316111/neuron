"""Does CLAPP survive a sequential stream, and can the stability layer save it?

Third attempt. Two previous failures, both sizing or pipeline bugs:
  1st: 250 updates, everything at the random floor. Too few.
  2nd: augmentation applied to normalised [-1,1] tensors, but ColorJitter
       needs [0,1]. Views were corrupted, features collapsed, probe
       predicted a constant (10.00% and 50.00%, exactly chance).

This version reuses the EXACT augmentation and data pipeline from
clapp_cifar.py, which produced 51.91%. Nothing about the data path is new.
The only new thing is the ORDER samples arrive in.

Three arms, same budget:
  shuffled   random order, the ceiling and the sanity check
  stream     one class at a time, in order, never returning
  guarded    the same stream wrapped in stability.py
"""

import time
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from clapp import CLAPPNet
from stability import StabilityLayer


CHANNELS = (64, 128, 128, 256)
POOLS = (False, True, True, True)
LR = 2e-4

BATCH = 128
BLOCKS_PER_CLASS = 600
CLASS_ORDER = list(range(10))
PROBE_AFTER_CLASSES = 2
PROBE_SUBSET = 10000
PROBE_EPOCHS = 3
RANDOM_FLOOR = 32.6
SEED = 0
CANARY_CLASSES = [0, 1]


def device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


# ---- the exact pipeline from clapp_cifar.py, unchanged ----

def augment():
    return transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.3, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


class TwoViews:
    def __init__(self, tf):
        self.tf = tf

    def __call__(self, x):
        return self.tf(x), self.tf(x)


def plain():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


# ---- data ----

def aug_dataset():
    return datasets.CIFAR10("./data", train=True, download=True,
                            transform=TwoViews(augment()))


def plain_tensors(train):
    ds = datasets.CIFAR10("./data", train=train, download=True,
                          transform=plain())
    x, y = next(iter(DataLoader(ds, batch_size=len(ds))))
    return x, y


def class_indices(targets):
    out = {}
    for c in CLASS_ORDER:
        out[c] = [i for i, t in enumerate(targets) if t == c]
    return out


def build_batches(ds, by_class, sequential, seed):
    """A list of (class, batch_of_two_views). Built by iterating real
    DataLoaders so augmentation runs exactly as it did in clapp_cifar."""
    rng = random.Random(seed)
    plan = []
    for c in CLASS_ORDER:
        idx = list(by_class[c])
        rng.shuffle(idx)
        for b in range(BLOCKS_PER_CLASS):
            start = (b * BATCH) % max(1, len(idx) - BATCH)
            take = idx[start:start + BATCH]
            if len(take) == BATCH:
                plan.append((c, take))
    if not sequential:
        rng.shuffle(plan)
    return plan


def fetch(ds, indices):
    """Pull one batch through the real augmentation pipeline."""
    loader = DataLoader(Subset(ds, indices), batch_size=len(indices),
                        num_workers=0)
    (v1, v2), _ = next(iter(loader))
    return v1, v2


# ---- backend ----

class StreamBackend:
    def __init__(self, dev, seed=0):
        torch.manual_seed(seed)
        self.dev = dev
        self.net = CLAPPNet(channels=CHANNELS, pools=POOLS, in_ch=3).to(dev)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.grad_steps = 0

    def _prep(self, item):
        v1, v2 = item
        v1, v2 = v1.to(self.dev), v2.to(self.dev)
        perm = torch.randperm(v1.size(0), device=self.dev)
        return v1, v1[perm], v2

    def score(self, item):
        self.net.eval()
        with torch.no_grad():
            v1, vneg, v2 = self._prep(item)
            total = float(sum(l.item()
                              for l in self.net.losses(v1, vneg, v2)))
        return total, None

    def update(self, item, steps):
        self.net.train()
        last = None
        for _ in range(steps):
            v1, vneg, v2 = self._prep(item)
            loss = sum(self.net.losses(v1, vneg, v2))
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


# ---- evaluation ----

def probe(net, dev, xtr, ytr, xte, yte, classes=None):
    net.eval()
    clf = nn.Linear(net.feature_dim, 10).to(dev)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)

    sub = torch.randperm(xtr.size(0))[:PROBE_SUBSET]
    tr = DataLoader(TensorDataset(xtr[sub], ytr[sub]), batch_size=256,
                    shuffle=True)
    for _ in range(PROBE_EPOCHS):
        for xb, yb in tr:
            xb, yb = xb.to(dev), yb.to(dev)
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
        for xb, yb in DataLoader(TensorDataset(xe, ye), batch_size=256):
            xb, yb = xb.to(dev), yb.to(dev)
            correct += (clf(net.features(xb)).argmax(1) == yb).sum().item()
            total += yb.numel()
    return 100.0 * correct / total


# ---- one arm ----

def run(mode, dev, ds, by_class, data):
    xtr, ytr, xte, yte = data
    torch.manual_seed(SEED)
    backend = StreamBackend(dev, SEED)

    sequential = mode != "shuffled"
    plan = build_batches(ds, by_class, sequential, SEED)

    canary = []
    for c in CANARY_CLASSES:
        canary.append(fetch(ds, by_class[c][:BATCH]))

    layer = None
    if mode == "guarded":
        layer = StabilityLayer(
            backend, canary=canary, seed=SEED,
            warmup=20, top_fraction=0.60, coherence_veto=1e9,
            loss_floor=0.0, steps_per_update=1,
            rehearse_per_item=3, rehearse_count=2, rehearse_steps=1,
            anchor_size=40, buffer_size=150,
            guard=True, guard_per_item=100, canary_tolerance=1.0)

    t0 = time.time()
    seen = []
    for i, (cls, indices) in enumerate(plan):
        if cls not in seen:
            seen.append(cls)
        item = fetch(ds, indices)

        if layer is not None:
            layer.observe(item)
        else:
            backend.update(item, 1)

        done = i + 1
        if sequential and done % BLOCKS_PER_CLASS == 0 \
                and (done // BLOCKS_PER_CLASS) % PROBE_AFTER_CLASSES == 0:
            overall = probe(backend.net, dev, xtr, ytr, xte, yte)
            early = probe(backend.net, dev, xtr, ytr, xte, yte,
                          classes=CANARY_CLASSES)
            s = layer.summary() if layer else {}
            print(f"    {len(seen):>2} classes: overall {overall:5.2f}  "
                  f"early {early:5.2f}  gate {s.get('updates', done):>4}  "
                  f"grad {backend.grad_steps:>4}  "
                  f"rb {s.get('rollbacks', 0)}  "
                  f"({time.time() - t0:.0f}s)")

    final = probe(backend.net, dev, xtr, ytr, xte, yte)
    final_early = probe(backend.net, dev, xtr, ytr, xte, yte,
                        classes=CANARY_CLASSES)
    s = layer.summary() if layer else {}
    stats = dict(overall=final, early=final_early,
                 gate_fires=s.get("updates", len(plan)),
                 grad_steps=backend.grad_steps,
                 rollbacks=s.get("rollbacks", 0),
                 rehearsals=s.get("rehearsals", 0),
                 minutes=(time.time() - t0) / 60)

    del backend, layer
    if dev == "mps":
        torch.mps.empty_cache()
    return stats


if __name__ == "__main__":
    dev = device()
    total_steps = len(CLASS_ORDER) * BLOCKS_PER_CLASS
    print(f"device {dev}   {total_steps} online steps of {BATCH} images\n")

    ds = aug_dataset()
    by_class = class_indices(ds.targets)
    xtr, ytr = plain_tensors(True)
    xte, yte = plain_tensors(False)
    data = (xtr, ytr, xte, yte)

    results = {}
    for mode in ["shuffled", "stream", "guarded"]:
        print(f"--- {mode} ---")
        s = run(mode, dev, ds, by_class, data)
        results[mode] = s
        print(f"  final overall {s['overall']:.2f}%   "
              f"early classes {s['early']:.2f}%")
        print(f"  gate fires {s['gate_fires']}   "
              f"gradient steps {s['grad_steps']}   "
              f"rollbacks {s['rollbacks']}   "
              f"rehearsals {s['rehearsals']}   {s['minutes']:.1f} min\n")

        if mode == "shuffled" and s["overall"] < RANDOM_FLOOR + 5:
            print("!" * 70)
            print(f"ABORT: shuffled scored {s['overall']:.2f}%, not clear of "
                  f"the {RANDOM_FLOOR}% random floor.")
            print("Nothing learned, so forgetting cannot be measured.")
            print("!" * 70)
            break

    print("=" * 70)
    print(f"{'arm':>10} {'overall':>9} {'early classes':>15} {'grad':>8}")
    print("-" * 70)
    for mode in ["shuffled", "stream", "guarded"]:
        if mode in results:
            r = results[mode]
            print(f"{mode:>10} {r['overall']:>8.2f}% {r['early']:>14.2f}% "
                  f"{r['grad_steps']:>8}")
    print(f"{'random':>10} {RANDOM_FLOOR:>8.2f}% {'(floor)':>15}")
    print("=" * 70)
    print("""
shuffled must clear the floor or nothing below it means anything.

"early classes" measures how well the network still handles what it saw
first and has not seen since. That is forgetting, measured directly.

  stream below shuffled      -> the sequential order does damage
  guarded above stream       -> the stability layer helps
  guarded near shuffled      -> both halves working together
""")
    with open("clapp_stream.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote clapp_stream.json")