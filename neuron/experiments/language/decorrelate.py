"""Is the single-pass penalty really about ORDER, and can it be recovered?

WHAT THE LAST RUN FOUND. Growing a character model from random weights in a
single pass reached 1.7558 nats on held-out text, against 1.6012 for three
conventional shuffled epochs. But the gap decomposes unevenly:

  online, in order          1.7558
  same data, shuffled once  1.6515     <- order costs 0.1043
  three shuffled epochs     1.6012     <- extra epochs buy only 0.0503

TWO THIRDS OF THE PENALTY IS ORDER, NOT THE SINGLE PASS. Shuffling once
recovers most of it without a second look at any sentence. So the thing to
attack is the correlation between consecutive samples, not the number of
passes.

TWO LEVERS, both of which keep every sentence seen exactly once.

  ACCUMULATION    sum the loss over N sentences, step once. Each sentence is
                  still seen once, in order, never revisited; only the
                  APPLYING is batched. On weather this project measured a
                  free win, 2.33x faster AND 6.2 points better, because
                  consecutive hours are nearly identical so a single
                  sample's gradient is mostly noise. On the discrete grid
                  world the same change LOST 12 to 16 points, because there
                  each event was a distinct lesson and averaging blurred
                  eight of them into one. Sentences sit between those two
                  cases, so this genuinely could go either way.

  SHUFFLE BUFFER  hold the last B sentences and draw the next update from
                  the buffer at random. Nothing is revisited and nothing is
                  stored long-term, but consecutive updates stop being
                  neighbours. THIS IS A RELAXATION OF THE PREMISE: the
                  stream is still single-pass, but no longer strictly in
                  order. Reported separately for that reason, never folded
                  into the "in order" claim.

THE TARGET. The fully shuffled arm is the ceiling these levers are chasing.
The verdict reports what FRACTION of the order gap each one recovers, which
is the number that matters, rather than raw loss which is hard to read.

COST is reported as optimiser STEPS, not seconds. Accumulation cuts steps by
construction and Adam's cost is per step, so steps are the meaningful figure.
Wall clock on the laptop used here has been unreliable across a day of runs:
the same configuration has taken 373s and 1035s for identical work.

  python decorrelate.py --train 15000 --seeds 2
  python decorrelate.py --train 2000 --eval 200 --seeds 1 --every 500
"""

import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scratch import (CharGRU, V, encode, held_out_loss,   # noqa: E402
                     ngram_loss, pick_device)


# name, accumulate, buffer size (0 = strictly in order)
ARMS = [
    ("online",     1,  0),
    ("accum-4",    4,  0),
    ("accum-8",    8,  0),
    ("accum-16",  16,  0),
    ("buffer-64",  1, 64),
    ("shuffled",   1, -1),      # -1 = full shuffle, the ceiling
]


def stream(lines, buffer_size, seed):
    """Yield sentences, either in order, through a buffer, or fully shuffled.

    The buffer holds recent sentences and emits a random one, so consecutive
    updates are not neighbours. Nothing is revisited: an emitted sentence
    leaves the buffer for good.
    """
    if buffer_size == -1:
        items = list(lines)
        random.Random(seed + 1000).shuffle(items)
        yield from items
        return
    if buffer_size == 0:
        yield from lines
        return

    rng = random.Random(seed + 1000)
    buf = []
    for line in lines:
        buf.append(line)
        if len(buf) >= buffer_size:
            i = rng.randrange(len(buf))
            buf[i], buf[-1] = buf[-1], buf[i]
            yield buf.pop()
    rng.shuffle(buf)
    yield from buf


def train_arm(name, accum, buffer_size, train_lines, eval_lines, args, seed):
    """One arm. Same seed for weights across arms, so only the regime differs."""
    torch.manual_seed(seed)
    device = pick_device()
    model = CharGRU(args.hidden, args.embed).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss()

    seen, steps = 0, 0
    pending, batch = 0, []
    started = time.time()
    curve = [(0, held_out_loss(model, eval_lines, device))]

    for line in stream(train_lines, buffer_size, seed):
        ids = encode(line)
        if len(ids) < 2:
            continue
        model.train()
        x = torch.tensor([ids[:-1]], device=device)
        y = torch.tensor([ids[1:]], device=device)
        logits, _ = model(x)
        batch.append(lossf(logits.reshape(-1, V), y.reshape(-1)))
        pending += 1
        seen += 1

        # Every sentence is still seen once and in order. Only the APPLYING
        # is batched, which is what makes this compatible with the premise.
        if pending >= accum:
            opt.zero_grad()
            (sum(batch) / len(batch)).backward()
            opt.step()
            steps += 1
            pending, batch = 0, []

        if seen % args.every == 0:
            h = held_out_loss(model, eval_lines, device)
            curve.append((seen, h))
            print(f"    {seen:6d}  {h:.4f} nats  "
                  f"({h / math.log(2):.3f} bits/char)  {steps} steps",
                  flush=True)

    if batch:
        opt.zero_grad()
        (sum(batch) / len(batch)).backward()
        opt.step()
        steps += 1

    final = held_out_loss(model, eval_lines, device)
    if curve[-1][0] != seen:
        curve.append((seen, final))
    return dict(name=name, seed=seed, accum=accum, buffer=buffer_size,
                seen=seen, steps=steps, curve=curve, final=final,
                seconds=time.time() - started)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/wiki.txt")
    ap.add_argument("--train", type=int, default=15000)
    ap.add_argument("--eval", type=int, default=800)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--embed", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--every", type=int, default=3000)
    ap.add_argument("--out", default="decorrelate.json")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        raise SystemExit(f"no data at {os.path.abspath(args.data)}")

    lines = [l.strip() for l in open(args.data) if len(l.strip()) > 40]
    if len(lines) < args.train + args.eval:
        raise SystemExit(f"only {len(lines)} usable lines")

    train_lines = lines[:args.train]
    eval_lines = lines[-args.eval:]

    uniform = math.log(V)
    print(f"  vocab {V}, {len(train_lines)} training sentences, "
          f"{len(eval_lines)} held out")
    print(f"  uniform floor {uniform:.4f} nats", flush=True)

    trigram = ngram_loss(" ".join(train_lines), eval_lines, 3)
    print(f"  trigram (full corpus) {trigram:.4f} nats  "
          f"({trigram / math.log(2):.3f} bits/char)", flush=True)

    runs = []
    for seed in range(args.seeds):
        for name, accum, buf in ARMS:
            print(f"\n=== {name} (seed {seed}) ===", flush=True)
            runs.append(train_arm(name, accum, buf, train_lines,
                                  eval_lines, args, seed))

    with open(args.out, "w") as f:
        json.dump(dict(args=vars(args), uniform=uniform, trigram=trigram,
                       runs=runs), f, indent=2)

    agg = {}
    for r in runs:
        a = agg.setdefault(r["name"], dict(final=[], steps=[], secs=[]))
        a["final"].append(r["final"])
        a["steps"].append(r["steps"])
        a["secs"].append(r["seconds"])

    def mean(xs):
        return sum(xs) / len(xs)

    base = mean(agg["online"]["final"])
    ceil = mean(agg["shuffled"]["final"])
    gap = base - ceil

    print("\n" + "=" * 74)
    print("RESULT   held-out cross entropy, and how much of the order gap "
          "is recovered")
    print("=" * 74)
    print(f"  {'arm':12s} {'nats':>8s} {'bits/char':>10s} {'steps':>8s} "
          f"{'gap recovered':>15s}")
    for name, _, _ in ARMS:
        a = agg[name]
        f = mean(a["final"])
        rec = (base - f) / gap * 100 if gap > 1e-9 else float("nan")
        spread = ""
        if len(a["final"]) > 1:
            spread = f"   ({', '.join(f'{v:.4f}' for v in a['final'])})"
        print(f"  {name:12s} {f:8.4f} {f / math.log(2):10.3f} "
              f"{mean(a['steps']):8.0f} {rec:14.0f}%{spread}")

    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)

    if gap <= 0.02:
        print(f"  The order gap in this run is only {gap:.4f} nats. There is")
        print(f"  almost nothing to recover, so no arm below means much.")
        print(f"  Re-run with a stream whose order actually matters.")
        return

    print(f"  Order gap to recover: {gap:.4f} nats "
          f"(online {base:.4f} vs shuffled {ceil:.4f}).")

    # Accumulation is the arm that keeps the premise intact, so it is judged
    # on its own. The buffer arm relaxes "in order" and is reported apart.
    accums = [(n, mean(agg[n]["final"]), mean(agg[n]["steps"]))
              for n, a, b in ARMS if a > 1]
    best = min(accums, key=lambda t: t[1])
    rec = (base - best[1]) / gap * 100
    step_cut = mean(agg["online"]["steps"]) / best[2]

    if best[1] >= base - 0.01:
        print(f"\n  ACCUMULATION DID NOT HELP. Best was {best[0]} at "
              f"{best[1]:.4f} against")
        print(f"  online's {base:.4f}. Sentences behave like the grid world, "
              f"not like weather:")
        print(f"  each one is a distinct lesson and averaging blurs them "
              f"rather than")
        print(f"  cancelling noise. That is a real answer about text.")
    else:
        print(f"\n  Best accumulation arm: {best[0]}, {best[1]:.4f} nats, "
              f"{rec:.0f}% of the gap")
        print(f"  recovered, with {step_cut:.1f}x fewer optimiser steps. "
              f"Both axes move the")
        print(f"  right way, which is the outcome the weather result "
              f"predicted.")

    bufv = mean(agg["buffer-64"]["final"])
    brec = (base - bufv) / gap * 100
    print(f"\n  Shuffle buffer (premise relaxed): {bufv:.4f} nats, "
          f"{brec:.0f}% of the gap.")
    if brec > rec + 15:
        print(f"  It beats accumulation clearly. The penalty is about WHICH "
              f"samples land")
        print(f"  next to each other, not about how gradients are applied.")

    print(f"\n  One domain, character level, {args.seeds} seed"
          f"{'s' if args.seeds > 1 else ''}. The buffer arm is not the "
          f"in-order claim.")
    print(f"\n  written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()