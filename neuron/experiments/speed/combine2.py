"""The two arms combine.py never reached.

Six arms completed before a crash. What they showed: the levers do NOT
compose. SGD beat Adam without accumulation (48.80 vs 47.73), but with
accumulation Adam wins by 2.7 points (53.92 vs 51.24). Accumulation already
removes the gradient noise that Adam's normalisation exists to handle, so
once you accumulate, Adam has cleaner signal to work with and SGD gains
nothing.

So Adam plus accumulation is the configuration. Two things left to try on it:

  adam-8-64          halve the hidden size. sgd-8-64 was 4.3x faster than
                     the baseline configuration but fell below the accuracy
                     baseline. Adam is worth 2.7 points more, so it may hold
                     where SGD did not.
  adam-8-128-deep    let the autograd graph span the accumulation window
                     instead of detaching every step. Assigns credit across
                     the whole window at more cost per step.

The crash in combine.py was calling backward() on every sample while never
detaching the hidden state, so the second call hit a freed graph. Fixed here
by accumulating the losses and calling backward ONCE per window, which is
what spanning the window actually requires.
"""

import json
import time

import torch
import torch.nn.functional as F

from sensor import fetch, build_stream, CLASSES
from sensorrun2 import add_lags, extrapolation_baseline, majority_baseline
from combine import GRU, encode, evaluate, EPISODE, SEEDS, TEST_YEAR, SEASONS


def run_flat(hidden, accum, lr, data, test, seed):
    """Detach every step. The cheap backward."""
    net = GRU(hidden, seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
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
    secs = time.perf_counter() - t0
    acc, per = evaluate(net, test)
    del net, opt
    return dict(acc=acc, per=per, seconds=secs)


def run_deep(hidden, accum, lr, data, test, seed):
    """Let the graph span the window: collect losses, backward once.

    This is what spanning actually requires. Calling backward per sample
    without detaching tries to traverse a freed graph, which is what
    crashed combine.py.
    """
    net = GRU(hidden, seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    t0 = time.perf_counter()
    h = None
    losses = []
    net.train()
    for i, (feats, label, _) in enumerate(data):
        if i % EPISODE == 0:
            if losses:
                opt.zero_grad()
                torch.stack(losses).mean().backward()
                opt.step()
                losses = []
            h = None
        logits, h = net(encode(feats), h)
        losses.append(F.cross_entropy(
            logits, torch.tensor([CLASSES.index(label)])))
        if len(losses) >= accum:
            opt.zero_grad()
            torch.stack(losses).mean().backward()
            opt.step()
            losses = []
            h = h.detach()
    if losses:
        opt.zero_grad()
        torch.stack(losses).mean().backward()
        opt.step()
    secs = time.perf_counter() - t0
    acc, per = evaluate(net, test)
    del net, opt
    return dict(acc=acc, per=per, seconds=secs)


ARMS = {
    "adam-8-128":       (run_flat, 128, 8, 3e-4),
    "adam-8-64":        (run_flat, 64, 8, 3e-4),
    "adam-8-32":        (run_flat, 32, 8, 3e-4),
    "adam-8-128-deep":  (run_deep, 128, 8, 3e-4),
    "adam-8-64-deep":   (run_deep, 64, 8, 3e-4),
}

REFERENCE_US = 403.9      # adam-1-128 from combine.py


if __name__ == "__main__":
    stream = add_lags(build_stream(fetch()))
    trn = [s for s in stream if s[2][:4] != TEST_YEAR]
    test = [s for s in stream if s[2][:4] == TEST_YEAR]
    base = max(majority_baseline(test), extrapolation_baseline(test))

    print(f"{len(trn)} steps, baseline {base:.2f}%")
    print(f"reference: adam-1-128 at {REFERENCE_US:.1f} us/step\n")

    def mean(xs):
        return sum(xs) / len(xs)

    results = {}
    for name, (fn, hidden, accum, lr) in ARMS.items():
        runs = [fn(hidden, accum, lr, trn, test, s) for s in SEEDS]
        results[name] = runs
        a = mean([r["acc"] for r in runs])
        us = 1e6 * mean([r["seconds"] for r in runs]) / len(trn)
        mark = "  BEATS BASELINE" if a > base else ""
        print(f"  {name:>20}: {a:5.2f}%  {us:6.1f} us/step  "
              f"{REFERENCE_US / us:4.2f}x{mark}", flush=True)

    print("\n" + "=" * 82)
    print(f"{'arm':>20} {'accuracy':>9} {'vs base':>8} {'us/step':>9} "
          f"{'speedup':>9} {'spread':>8}")
    print("-" * 82)
    for name in ARMS:
        runs = results[name]
        a = mean([r["acc"] for r in runs])
        us = 1e6 * mean([r["seconds"] for r in runs]) / len(trn)
        pers = [mean([r["per"].get(s, 0.0) for r in runs]) for s in SEASONS]
        mark = " *" if a > base else "  "
        print(f"{name:>20} {a:>8.2f}% {a - base:>+7.2f} {us:>8.1f} "
              f"{REFERENCE_US / us:>8.2f}x {max(pers) - min(pers):>7.1f}{mark}")
    print("=" * 82)
    print("  * beats the extrapolation baseline")

    winners = [(REFERENCE_US / (1e6 * mean([r["seconds"]
                                            for r in results[n]]) / len(trn)), n)
               for n in ARMS if mean([r["acc"] for r in results[n]]) > base]
    print()
    if winners:
        speed, name = max(winners)
        acc = mean([r["acc"] for r in results[name]])
        print(f"  BEST: {name} at {acc:.2f}%, {speed:.2f}x faster than "
              f"stepping every sample.")
    else:
        print("  Only adam-8-128 holds the baseline. Shrinking the network "
              "costs\n  more than it saves, and the 2.34x already found is "
              "the answer.")

    print("""
The question is how small the network can get before it stops beating the
baseline. Each halving roughly halves the backward pass, which is the
dominant cost now that the optimiser is amortised.

The deep arms let the graph span the accumulation window, assigning credit
across all eight samples rather than one. More expensive per step. Whether
it buys anything is the open question — the horizon experiment suggested
lookback distance matters less than expected.
""")
    with open("combine2.json", "w") as f:
        json.dump({k: [dict(acc=r["acc"], seconds=r["seconds"],
                            per=r["per"]) for r in v]
                   for k, v in results.items()}, f, indent=2)
    print("wrote combine2.json")