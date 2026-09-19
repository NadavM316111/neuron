"""The lever the profile actually pointed at, and the confound I introduced.

The profile said Adam costs 288 us per step and that cost is INDEPENDENT OF
BATCH SIZE — 288 at batch 1, 294 at batch 256. It is per-STEP, not
per-sample.

The obvious consequence, which the last experiment missed: do not step every
sample. Accumulate the gradient over N samples and step once. The stream is
still processed one moment at a time, in order, never revisited — only the
APPLYING is batched, not the seeing. That divides the dominant cost by N and
throws nothing away, unlike sparsity.

Also fixed here: the last run set SGD's learning rate by guessing. SGD is far
more sensitive to that than Adam, so the 9-point deficit may have been a
tuning artifact rather than a real one. This sweeps it properly.

Two experiments in one file.

  PART 1  SGD learning rate sweep, to find out whether cheap optimisers are
          actually worse or were just badly tuned.
  PART 2  gradient accumulation at N = 1, 4, 8, 16, 32, with Adam.

For part 2 the honest question is whether accumulation costs accuracy.
Accumulating over N consecutive samples is a mini-batch of correlated
moments, which is not the same as a shuffled batch. It may be fine. It may
blur the signal that makes online learning work. That is what this measures.
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

SGD_LRS = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
ACCUMS = [1, 4, 8, 16, 32]


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


def train(net, opt, data, accum=1):
    """One pass, in order. The optimiser steps every `accum` samples.

    Every sample is still seen exactly once and never revisited. Only the
    application of the update is batched.
    """
    t0 = time.perf_counter()
    h = None
    pending = 0
    net.train()
    opt.zero_grad()
    for i, (feats, label, _) in enumerate(data):
        if i % EPISODE == 0:
            h = None
        logits, h = net(encode(feats), h)
        loss = F.cross_entropy(logits,
                               torch.tensor([CLASSES.index(label)]))
        (loss / accum).backward()
        h = h.detach()
        pending += 1
        if pending >= accum:
            opt.step()
            opt.zero_grad()
            pending = 0
    if pending:
        opt.step()
        opt.zero_grad()
    return time.perf_counter() - t0


def run(kind, lr, accum, data, test, seed):
    net = GRU(seed)
    if kind == "adam":
        opt = torch.optim.Adam(net.parameters(), lr=lr)
    else:
        opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    secs = train(net, opt, data, accum)
    acc, per = evaluate(net, test)
    del net, opt
    return dict(acc=acc, per=per, seconds=secs)


if __name__ == "__main__":
    stream = add_lags(build_stream(fetch()))
    trn = [s for s in stream if s[2][:4] != TEST_YEAR]
    test = [s for s in stream if s[2][:4] == TEST_YEAR]
    base = max(majority_baseline(test), extrapolation_baseline(test))

    print(f"{len(trn)} training steps, baseline {base:.2f}%\n")

    def mean(xs):
        return sum(xs) / len(xs)

    results = {}

    print("PART 1: was SGD actually worse, or just badly tuned?\n")
    for lr in SGD_LRS:
        runs = [run("sgd", lr, 1, trn, test, s) for s in SEEDS]
        results[f"sgd-lr{lr}"] = runs
        a = mean([r["acc"] for r in runs])
        us = 1e6 * mean([r["seconds"] for r in runs]) / len(trn)
        print(f"  sgd-momentum lr {lr:<7}: {a:5.2f}%  {us:6.1f} us/step",
              flush=True)

    print("\nPART 2: accumulate the gradient, step less often\n")
    for n in ACCUMS:
        runs = [run("adam", 3e-4, n, trn, test, s) for s in SEEDS]
        results[f"accum-{n}"] = runs
        a = mean([r["acc"] for r in runs])
        us = 1e6 * mean([r["seconds"] for r in runs]) / len(trn)
        print(f"  adam, step every {n:>2}: {a:5.2f}%  {us:6.1f} us/step",
              flush=True)

    print("\n" + "=" * 80)
    print("EVERYTHING, against the baseline")
    print(f"{'arm':>22} {'accuracy':>9} {'vs base':>8} {'us/step':>9} "
          f"{'speedup':>8}")
    print("-" * 80)
    ref = 1e6 * mean([r["seconds"] for r in results["accum-1"]]) / len(trn)
    order = [f"sgd-lr{lr}" for lr in SGD_LRS] + \
            [f"accum-{n}" for n in ACCUMS]
    best_name, best_score = None, -1e9
    for name in order:
        runs = results[name]
        a = mean([r["acc"] for r in runs])
        us = 1e6 * mean([r["seconds"] for r in runs]) / len(trn)
        mark = ""
        if a > base:
            mark = "  BEATS BASELINE"
            if ref / us > best_score:
                best_score, best_name = ref / us, name
        print(f"{name:>22} {a:>8.2f}% {a - base:>+7.2f} {us:>8.1f} "
              f"{ref / us:>7.2f}x{mark}")
    print("=" * 80)

    print("\nseasonal accuracy, accumulation arms:")
    print(f"{'arm':>22} " + "  ".join(f"{s[:6]:>7}" for s in SEASONS))
    for n in ACCUMS:
        runs = results[f"accum-{n}"]
        cells = "  ".join(
            f"{mean([r['per'].get(s, 0.0) for r in runs]):>6.1f}%"
            for s in SEASONS)
        print(f"{'accum-' + str(n):>22} {cells}")

    if best_name:
        print(f"\n  FASTEST ARM STILL BEATING THE BASELINE: {best_name}, "
              f"{best_score:.2f}x faster than stepping every sample.")
    else:
        print("\n  Nothing beat the baseline. Either the tuning is still "
              "wrong or\n  the accuracy cost of both levers is real.")

    print("""
PART 1 tells you whether the last run's SGD deficit was real. If accuracy
climbs a lot across the learning rates, it was a tuning artifact and cheap
optimisers are back on the table.

PART 2 is the lever the profile pointed at. Adam's cost is per STEP and
independent of batch size, so stepping every 8th sample should cut the
dominant cost roughly eightfold. Every sample is still seen once, in order,
never revisited — only the applying is batched.

  accuracy holds as N rises -> a several-fold speedup for free, and the
      single-pass premise survives intact. This is the result to want.
  accuracy falls with N     -> applying the update immediately is part of
      what makes online learning work, which would itself be a real and
      publishable finding about why this regime is expensive.
""")
    with open("accum.json", "w") as f:
        json.dump({k: [dict(acc=r["acc"], seconds=r["seconds"],
                            per=r["per"]) for r in v]
                   for k, v in results.items()}, f, indent=2)
    print("wrote accum.json")