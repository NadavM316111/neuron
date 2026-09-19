"""Both levers together, and a first look at the backward pass.

Two things worked independently on the weather task:

  gradient accumulation   accum-8 was 2.33x faster AND 6.2 points more
                          accurate than stepping every sample, because
                          averaging eight noisy gradients cancels the noise
  SGD at the right lr     48.80% at lr 1e-3, better than Adam's 47.73%, at
                          1.76x the speed. The earlier 9-point deficit was
                          a bad learning-rate guess

They attack different costs, so they should compose. Adam's optimiser cost
is what accumulation amortises; SGD makes what remains cheaper.

This also probes the NEW dominant cost. With the optimiser amortised, the
breakdown at accum-8 is roughly 16 us forward, 113 us backward, and the rest
overhead. Backward is now the biggest single item, so two cheap experiments
on it are included:

  truncate    detach the hidden state more aggressively, so the backward
              pass is shorter. Costs memory horizon, which the horizon
              experiment showed does not matter much for one variable.
  smaller     a 64-unit network instead of 128. Backward cost scales with
              size, and capacity.py found 128 was no better than 512 in the
              grid world, so smaller may be free here too.

Everything is measured on accuracy AND time, because a faster arm that
learns worse is no use, and today has produced two of those already.
"""

import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from sensor import fetch, build_stream, CLASSES, VARIABLES
from sensorrun2 import add_lags, extrapolation_baseline, majority_baseline


SEEDS = [0, 1]
EPISODE = 720
TEST_YEAR = "2023"
LAG = 3
N_FEAT = len(VARIABLES) * (LAG + 1)
SEASONS = ["winter", "spring", "summer", "autumn"]


def encode(feats):
    return torch.tensor(feats, dtype=torch.float32).unsqueeze(0)


class GRU(nn.Module):
    def __init__(self, hidden, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.enc = nn.Linear(N_FEAT, hidden)
        self.cell = nn.GRUCell(hidden, hidden)
        self.head = nn.Linear(hidden, len(CLASSES))

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


def run(cfg, data, test, seed):
    """One pass, in order, never revisited.

    cfg: (optimiser name, learning rate, accumulation N, hidden size,
          detach every step?)
    """
    name, lr, accum, hidden, detach_always = cfg
    net = GRU(hidden, seed)
    if name == "adam":
        opt = torch.optim.Adam(net.parameters(), lr=lr)
    else:
        opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

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
        # Detaching every step keeps the graph one step deep, which is the
        # cheapest possible backward. Detaching only at the step boundary
        # lets the graph span the accumulation window, which costs more but
        # assigns credit further back.
        h = h.detach() if detach_always else h
        pending += 1
        if pending >= accum:
            opt.step()
            opt.zero_grad()
            pending = 0
            if not detach_always:
                h = h.detach()
    if pending:
        opt.step()
        opt.zero_grad()
    secs = time.perf_counter() - t0

    acc, per = evaluate(net, test)
    del net, opt
    return dict(acc=acc, per=per, seconds=secs)


# name -> (optimiser, lr, accum, hidden, detach every step)
ARMS = {
    "adam-1-128":            ("adam", 3e-4, 1, 128, True),
    "adam-8-128":            ("adam", 3e-4, 8, 128, True),
    "sgd-1-128":             ("sgd", 1e-3, 1, 128, True),
    "sgd-8-128":             ("sgd", 1e-3, 8, 128, True),
    "sgd-16-128":            ("sgd", 1e-3, 16, 128, True),
    "sgd-8-64":              ("sgd", 1e-3, 8, 64, True),
    "sgd-8-128-deepgraph":   ("sgd", 1e-3, 8, 128, False),
    "adam-8-64":             ("adam", 3e-4, 8, 64, True),
}


if __name__ == "__main__":
    stream = add_lags(build_stream(fetch()))
    trn = [s for s in stream if s[2][:4] != TEST_YEAR]
    test = [s for s in stream if s[2][:4] == TEST_YEAR]
    base = max(majority_baseline(test), extrapolation_baseline(test))

    print(f"{len(trn)} training steps, baseline {base:.2f}%")
    print("naming: optimiser-accumulation-hidden\n")

    def mean(xs):
        return sum(xs) / len(xs)

    results = {}
    for name, cfg in ARMS.items():
        runs = [run(cfg, trn, test, s) for s in SEEDS]
        results[name] = runs
        a = mean([r["acc"] for r in runs])
        us = 1e6 * mean([r["seconds"] for r in runs]) / len(trn)
        mark = "  BEATS BASELINE" if a > base else ""
        print(f"  {name:>22}: {a:5.2f}%  {us:6.1f} us/step{mark}",
              flush=True)

    ref = 1e6 * mean([r["seconds"] for r in results["adam-1-128"]]) / len(trn)

    print("\n" + "=" * 86)
    print(f"{'arm':>22} {'accuracy':>9} {'vs base':>8} {'us/step':>9} "
          f"{'speedup':>9} {'spread':>8}")
    print("-" * 86)
    for name in ARMS:
        runs = results[name]
        a = mean([r["acc"] for r in runs])
        us = 1e6 * mean([r["seconds"] for r in runs]) / len(trn)
        pers = [mean([r["per"].get(s, 0.0) for r in runs])
                for s in SEASONS]
        spread = max(pers) - min(pers)
        mark = " *" if a > base else "  "
        print(f"{name:>22} {a:>8.2f}% {a - base:>+7.2f} {us:>8.1f} "
              f"{ref / us:>8.2f}x {spread:>7.1f}{mark}")
    print("=" * 86)
    print("  * beats the extrapolation baseline")
    print("  spread = best season minus worst, so lower means less "
          "forgetting")

    winners = [(ref / (1e6 * mean([r["seconds"] for r in results[n]])
                       / len(trn)), n)
               for n in ARMS
               if mean([r["acc"] for r in results[n]]) > base]
    print()
    if winners:
        speed, name = max(winners)
        acc = mean([r["acc"] for r in results[name]])
        print(f"  FASTEST ARM STILL BEATING THE BASELINE: {name}")
        print(f"  {acc:.2f}% at {speed:.2f}x the speed of stepping every "
              f"sample with Adam.")
    else:
        print("  Nothing beat the baseline. The combination lost something "
              "the\n  individual levers had.")

    print("""
Three questions.

  DO THE LEVERS COMPOSE? sgd-8-128 against adam-8-128 and sgd-1-128. They
      attack different costs, so the speedups should multiply. If they do
      not, one of them was not doing what we thought.

  IS SMALLER FREE? sgd-8-64 halves the hidden size, which halves the
      backward cost. capacity.py found 128 was no better than 512 in the
      grid world, so 64 may cost nothing here either.

  DOES A DEEPER GRAPH HELP OR HURT? sgd-8-128-deepgraph lets the graph span
      the whole accumulation window instead of detaching every step. It
      assigns credit further back, at more cost. The horizon experiment
      suggested distance does not matter much, so this may be pure expense.

Watch the spread column throughout. Accumulation reduced forgetting from
26.8 to 7.2 as a side effect of averaging. If a faster arm gives that back,
it is not actually a better arm.
""")
    with open("combine.json", "w") as f:
        json.dump({k: [dict(acc=r["acc"], seconds=r["seconds"],
                            per=r["per"]) for r in v]
                   for k, v in results.items()}, f, indent=2)
    print("wrote combine.json")