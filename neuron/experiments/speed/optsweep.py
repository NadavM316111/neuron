"""A faster optimiser is only useful if it still learns.

The profile found the real bottleneck: Adam costs 288 us per step at batch 1
against a 16 us forward pass. Eighteen times more to apply the update than to
compute it. And Adam costs the same 288 us at batch 256, because its cost
scales with PARAMETER COUNT, not batch size.

Two levers follow, neither of which is a new learning rule:
  cheaper rule    SGD does one element-wise op per parameter; Adam does
                  several plus maintaining two state tensors
  sparse update   update only a fraction of parameters each step, and the
                  optimiser cost falls proportionally

But speed is worthless if accuracy goes with it. This measures BOTH on the
real weather task, so the trade is visible rather than assumed.

Optimisers:
  adam            the current default
  sgd             plain, no state
  sgd-momentum    one state tensor instead of two
  adam-sparse-25  Adam, but only the largest 25% of gradients are applied
  sgd-sparse-25   the same for SGD
  sgd-sparse-10   only the largest 10%

The sparse variants zero the smallest gradients before the step. That is a
deliberately naive implementation — it still touches every parameter, so it
does NOT yet capture the speed win. It is here to answer the prior question:
does throwing away most of the gradient cost accuracy? If it does not, the
speed win is worth engineering properly with real sparse tensors.

Reported per arm: wall clock for the training pass, and accuracy on the
held-out year. The comparison that matters is accuracy per unit time.
"""

import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from sensor import fetch, build_stream, CLASSES, VARIABLES
from sensorrun2 import add_lags, extrapolation_baseline, majority_baseline


HIDDEN = 128
SEEDS = [0, 1]
EPISODE = 720
TEST_YEAR = "2023"
LAG = 3
N_FEAT = len(VARIABLES) * (LAG + 1)
SEASONS = ["winter", "spring", "summer", "autumn"]


def encode(feats):
    return torch.tensor(feats, dtype=torch.float32).unsqueeze(0)


class GRU(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.enc = nn.Linear(N_FEAT, HIDDEN)
        self.cell = nn.GRUCell(HIDDEN, HIDDEN)
        self.head = nn.Linear(HIDDEN, len(CLASSES))

    def forward(self, x, h=None):
        e = F.relu(self.enc(x))
        h = self.cell(e, h)
        return self.head(h), h


def sparsify_(params, keep):
    """Zero all but the largest `keep` fraction of each gradient.

    Deliberately naive: it still touches every element, so it does NOT yet
    deliver the speed win. The point is to answer whether discarding most of
    the gradient costs accuracy, before anyone engineers real sparse updates.
    """
    for p in params:
        if p.grad is None:
            continue
        g = p.grad
        n = g.numel()
        k = max(1, int(n * keep))
        if k >= n:
            continue
        flat = g.view(-1)
        thresh = flat.abs().kthvalue(n - k).values
        flat[flat.abs() < thresh] = 0.0


# name -> (builder, sparse keep fraction or None)
OPTS = {
    "adam":            (lambda p, lr: torch.optim.Adam(p, lr=lr), None),
    "sgd":             (lambda p, lr: torch.optim.SGD(p, lr=lr), None),
    "sgd-momentum":    (lambda p, lr: torch.optim.SGD(p, lr=lr,
                                                      momentum=0.9), None),
    "adam-sparse-25":  (lambda p, lr: torch.optim.Adam(p, lr=lr), 0.25),
    "sgd-sparse-25":   (lambda p, lr: torch.optim.SGD(p, lr=lr,
                                                      momentum=0.9), 0.25),
    "sgd-sparse-10":   (lambda p, lr: torch.optim.SGD(p, lr=lr,
                                                      momentum=0.9), 0.10),
}

# SGD needs a larger step than Adam to move comparably, since Adam
# normalises by gradient magnitude. Tuned coarsely rather than carefully;
# the point is a fair-ish comparison, not a tuned one.
LRS = {
    "adam": 3e-4, "sgd": 3e-2, "sgd-momentum": 3e-3,
    "adam-sparse-25": 3e-4, "sgd-sparse-25": 3e-3, "sgd-sparse-10": 3e-3,
}


def evaluate(net, test):
    net.eval()
    h = None
    hit = seen = 0
    per = {}
    with torch.no_grad():
        for i, (feats, label, stamp) in enumerate(test):
            if i % EPISODE == 0:
                h = None
            logits, h = net(encode(feats), h)
            right = int(logits.argmax(1).item()) == CLASSES.index(label)
            seen += 1
            hit += right
            m = int(stamp[5:7])
            s = ("winter" if m in (12, 1, 2) else
                 "spring" if m in (3, 4, 5) else
                 "summer" if m in (6, 7, 8) else "autumn")
            a, b = per.get(s, (0, 0))
            per[s] = (a + right, b + 1)
    return (100.0 * hit / seen,
            {k: 100.0 * a / b for k, (a, b) in per.items()})


def run(name, train, test, seed):
    build, keep = OPTS[name]
    net = GRU(seed)
    opt = build(net.parameters(), LRS[name])
    params = list(net.parameters())

    t0 = time.perf_counter()
    h = None
    net.train()
    for i, (feats, label, _) in enumerate(train):
        if i % EPISODE == 0:
            h = None
        logits, h = net(encode(feats), h)
        loss = F.cross_entropy(logits,
                               torch.tensor([CLASSES.index(label)]))
        opt.zero_grad()
        loss.backward()
        if keep is not None:
            sparsify_(params, keep)
        opt.step()
        h = h.detach()
    train_time = time.perf_counter() - t0

    acc, per = evaluate(net, test)
    del net, opt
    return dict(acc=acc, per=per, seconds=train_time)


if __name__ == "__main__":
    stream = add_lags(build_stream(fetch()))
    train = [s for s in stream if s[2][:4] != TEST_YEAR]
    test = [s for s in stream if s[2][:4] == TEST_YEAR]

    maj = majority_baseline(test)
    ext = extrapolation_baseline(test)
    best = max(maj, ext)

    print(f"{len(train)} training steps, {len(test)} test steps")
    print(f"baseline to beat: extrapolation {ext:.2f}%\n")

    def mean(xs):
        return sum(xs) / len(xs)

    results = {}
    for name in OPTS:
        runs = [run(name, train, test, s) for s in SEEDS]
        results[name] = runs
        a = mean([r["acc"] for r in runs])
        t = mean([r["seconds"] for r in runs])
        us = 1e6 * t / len(train)
        print(f"  {name:>15}: {a:5.2f}%  {t:6.1f}s  "
              f"({us:6.1f} us/step)", flush=True)

    print("\n" + "=" * 78)
    print(f"{'optimiser':>15} {'accuracy':>9} {'vs base':>8} "
          f"{'us/step':>9} {'speedup':>9}   " +
          "  ".join(f"{s[:6]:>7}" for s in SEASONS))
    print("-" * 78)
    base_us = 1e6 * mean([r["seconds"] for r in results["adam"]]) / len(train)
    for name in OPTS:
        runs = results[name]
        a = mean([r["acc"] for r in runs])
        us = 1e6 * mean([r["seconds"] for r in runs]) / len(train)
        cells = "  ".join(
            f"{mean([r['per'].get(s, 0.0) for r in runs]):>6.1f}%"
            for s in SEASONS)
        print(f"{name:>15} {a:>8.2f}% {a - best:>+7.2f} {us:>8.1f} "
              f"{base_us / us:>8.2f}x   {cells}")
    print("=" * 78)

    print("\naccuracy per second of training, higher is better:")
    for name in OPTS:
        runs = results[name]
        a = mean([r["acc"] for r in runs])
        t = mean([r["seconds"] for r in runs])
        print(f"  {name:>15}: {a / t:6.2f}")

    print("""
Two questions.

  DOES A CHEAPER OPTIMISER KEEP THE ACCURACY? If sgd-momentum matches Adam
      at a fraction of the time, that is a free speedup on the dominant cost,
      available by changing one line.

  DOES SPARSITY COST ACCURACY? The sparse arms discard most of the gradient
      every step. If accuracy holds at 25% or even 10%, then a real sparse
      implementation would cut the optimiser cost by that factor, and the
      build plan's Stage 3 turns out to be the speed fix rather than a
      separate track.

NOTE the sparse arms here are NAIVE — they zero small gradients but still
touch every element, so their measured time does NOT include the win. They
are answering whether the accuracy survives, not how fast it would be. If it
survives, the engineering is worth doing properly.
""")
    with open("optsweep.json", "w") as f:
        json.dump({k: [dict(acc=r["acc"], seconds=r["seconds"],
                            per=r["per"]) for r in v]
                   for k, v in results.items()}, f, indent=2)
    print("wrote optsweep.json")