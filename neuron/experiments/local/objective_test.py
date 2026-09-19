"""Is class-agnostic learning immune to class-ordered streams?

The CIFAR run found no forgetting when categories arrived one at a time.
The hypothesis for why: CLAPP's objective is SELF-SUPERVISED, so it learns
edges and textures rather than category boundaries, and those are shared
across every class. Nothing to forget when the subject changes.

If that is right, then swapping ONLY the objective, keeping everything else
identical, should produce forgetting. A class-aware objective has category
boundaries to lose.

Four arms. Both objectives are LOCAL (no gradient crosses a layer), so
locality is held constant and the only variables are:

    objective:  ssl (class-agnostic)  vs  sup (class-aware)
    ordering:   iid (mixed batches)   vs  seq (one class at a time)

Plus an untrained control for the floor.

MNIST and a small MLP so the whole thing runs in about ten minutes.

The last experiment's control was broken because every batch held a single
class in both arms. Here the iid arm builds genuinely mixed batches.
"""

import time
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


HIDDEN = (256, 256, 128)
LR = 1e-3
BATCH = 128
STEPS_PER_CLASS = 200          # 10 classes x 200 = 2000 steps per arm
N_CLASSES = 10
EARLY = [0, 1]                 # taught first, never revisited
PROBE_EPOCHS = 3
PROBE_SUBSET = 10000
SEEDS = [0, 1]


def device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


# ---------------- data ----------------

def load(train):
    """Raw MNIST in [0,1]. Normalisation happens later, once, explicitly."""
    ds = datasets.MNIST("./data", train=train, download=True,
                        transform=transforms.ToTensor())
    x, y = next(iter(DataLoader(ds, batch_size=len(ds))))
    return x, y


def normalize(x):
    return (x - 0.5) / 0.5


def augment(x, shift=0.15, scale=0.12):
    """Random shift and scale, vectorised, valid on [0,1] tensors.

    Deliberately avoids ColorJitter and friends, which require [0,1] and
    silently corrupt normalised data. That bug cost a run yesterday.
    """
    b = x.size(0)
    theta = torch.zeros(b, 2, 3, device=x.device)
    s = 1.0 + (torch.rand(b, device=x.device) * 2 - 1) * scale
    theta[:, 0, 0] = s
    theta[:, 1, 1] = s
    theta[:, :, 2] = (torch.rand(b, 2, device=x.device) * 2 - 1) * shift
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    out = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")
    return out.clamp(0.0, 1.0)


def class_indices(y):
    return {c: (y == c).nonzero(as_tuple=True)[0].tolist()
            for c in range(N_CLASSES)}


def build_plan(by_class, sequential, seed):
    """List of index lists, one per step.

    sequential: each batch is one class, classes in order 0..9
    iid:        each batch is a genuine mix of all classes
    """
    rng = random.Random(seed)
    total = N_CLASSES * STEPS_PER_CLASS
    plan = []

    if sequential:
        for c in range(N_CLASSES):
            pool = list(by_class[c])
            rng.shuffle(pool)
            for _ in range(STEPS_PER_CLASS):
                plan.append([pool[rng.randrange(len(pool))]
                             for _ in range(BATCH)])
    else:
        every = [i for c in range(N_CLASSES) for i in by_class[c]]
        for _ in range(total):
            plan.append([every[rng.randrange(len(every))]
                         for _ in range(BATCH)])
    return plan


# ---------------- model ----------------

class LocalNet(nn.Module):
    """MLP where every layer trains on its own loss. No gradient crosses a
    layer boundary, for either objective."""

    def __init__(self, in_dim=784, hidden=HIDDEN, n_classes=N_CLASSES):
        super().__init__()
        dims = [in_dim] + list(hidden)
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(hidden))])
        # class-agnostic head: trainable B per layer, as in CLAPP
        self.B = nn.ModuleList(
            [nn.Linear(d, d, bias=False) for d in hidden])
        # class-aware head: linear classifier per layer
        self.heads = nn.ModuleList(
            [nn.Linear(d, n_classes) for d in hidden])
        self.hidden = hidden

    def encode(self, x, detach=True):
        """Returns activations for every layer. detach=True is what makes
        each layer local."""
        h = x.flatten(1)
        outs = []
        for layer in self.layers:
            h = F.relu(layer(h.detach() if detach else h))
            outs.append(h)
        return outs

    def features(self, x):
        with torch.no_grad():
            return torch.cat(self.encode(x), dim=1)

    @property
    def feature_dim(self):
        return sum(self.hidden)


def ssl_losses(net, v1, v2):
    """Class-agnostic. Per layer: two views of the same image should score
    high against each other, a different image should score low.
    Hinge loss, as in CLAPP Table 1."""
    h_pos = net.encode(v1)
    perm = torch.randperm(v1.size(0), device=v1.device)
    h_neg = net.encode(v1[perm])
    with torch.no_grad():
        h_ref = net.encode(v2)

    out = []
    for i in range(len(net.layers)):
        c = net.B[i](h_ref[i].detach())
        s_pos = (h_pos[i] * c).sum(dim=1)
        s_neg = (h_neg[i] * c).sum(dim=1)
        # normalise by width so layers of different size contribute evenly
        d = net.hidden[i]
        out.append((F.relu(1.0 - s_pos / d) + F.relu(1.0 + s_neg / d)).mean())
    return out


def sup_losses(net, v1, y):
    """Class-aware. Per layer: predict the label from this layer's
    activations."""
    h = net.encode(v1)
    return [F.cross_entropy(net.heads[i](h[i]), y)
            for i in range(len(net.layers))]


# ---------------- evaluation ----------------

def probe(net, dev, xtr, ytr, xte, yte, classes=None):
    """Same protocol for every arm: freeze, linear classifier on the
    concatenated features, measure accuracy."""
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


# ---------------- one arm ----------------

def run(objective, ordering, seed, dev, data, by_class):
    xtr, ytr, xte, yte = data
    torch.manual_seed(seed)
    net = LocalNet().to(dev)

    if objective == "none":
        return dict(overall=probe(net, dev, xtr, ytr, xte, yte),
                    early=probe(net, dev, xtr, ytr, xte, yte, EARLY),
                    minutes=0.0)

    opt = torch.optim.Adam(net.parameters(), lr=LR)
    plan = build_plan(by_class, ordering == "seq", seed)

    t0 = time.time()
    for indices in plan:
        raw = xtr[indices].to(dev)
        y = ytr[indices].to(dev)
        net.train()

        if objective == "ssl":
            v1 = normalize(augment(raw))
            v2 = normalize(augment(raw))
            losses = ssl_losses(net, v1, v2)
        else:
            v1 = normalize(augment(raw))
            losses = sup_losses(net, v1, y)

        opt.zero_grad()
        sum(losses).backward()
        opt.step()

    overall = probe(net, dev, xtr, ytr, xte, yte)
    early = probe(net, dev, xtr, ytr, xte, yte, EARLY)
    mins = (time.time() - t0) / 60

    del net, opt
    if dev == "mps":
        torch.mps.empty_cache()
    return dict(overall=overall, early=early, minutes=mins)


if __name__ == "__main__":
    dev = device()
    print(f"device {dev}   MLP {HIDDEN}   "
          f"{N_CLASSES * STEPS_PER_CLASS} steps of {BATCH}\n")

    xtr, ytr = load(True)
    xte, yte = load(False)
    data = (xtr, ytr, xte, yte)
    by_class = class_indices(ytr)

    arms = [
        ("none", "-",   "untrained (floor)"),
        ("ssl",  "iid", "class-agnostic, mixed batches"),
        ("ssl",  "seq", "class-agnostic, one class at a time"),
        ("sup",  "iid", "class-aware, mixed batches"),
        ("sup",  "seq", "class-aware, one class at a time"),
    ]

    results = {}
    for objective, ordering, label in arms:
        key = f"{objective}-{ordering}"
        os_, es = [], []
        for seed in SEEDS:
            r = run(objective, ordering, seed, dev, data, by_class)
            os_.append(r["overall"])
            es.append(r["early"])
        results[key] = dict(overall=sum(os_) / len(os_),
                            early=sum(es) / len(es),
                            label=label)
        print(f"  {key:>8}  overall {results[key]['overall']:6.2f}%   "
              f"early(0,1) {results[key]['early']:6.2f}%   {label}")

    print("\n" + "=" * 74)
    print(f"{'arm':>10} {'overall':>9} {'early 0,1':>11}   description")
    print("-" * 74)
    for objective, ordering, label in arms:
        r = results[f"{objective}-{ordering}"]
        print(f"{objective + '-' + ordering:>10} {r['overall']:>8.2f}% "
              f"{r['early']:>10.2f}%   {label}")
    print("=" * 74)

    ssl_drop = results["ssl-iid"]["early"] - results["ssl-seq"]["early"]
    sup_drop = results["sup-iid"]["early"] - results["sup-seq"]["early"]
    print(f"\nforgetting on the first classes, iid minus seq:")
    print(f"   class-agnostic (ssl): {ssl_drop:+6.2f} points")
    print(f"   class-aware    (sup): {sup_drop:+6.2f} points")
    print(f"""
  sup drops a lot, ssl barely moves  -> hypothesis CONFIRMED. Forgetting
      needs a class-aware objective. Self-supervised local learning is
      intrinsically robust to class-ordered streams, and the stability
      layer belongs on the class-aware case.

  both drop                          -> the CIFAR null result was an
      artifact of that broken control, and forgetting is real here after
      all. Rerun the CIFAR test properly.

  neither drops                      -> the stream is still too easy.
      Raise STEPS_PER_CLASS so each block is long enough to drift.
""")
    with open("objective_test.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote objective_test.json")