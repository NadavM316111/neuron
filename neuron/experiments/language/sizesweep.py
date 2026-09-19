"""Does the single-pass penalty shrink as the network grows?

THE ONLY SCALING SIGNAL AVAILABLE ON A LAPTOP. The from-scratch result is at
235,000 parameters, and the objection to it is always the same: mechanisms
proven at that size may not survive to a useful model. That cannot be settled
directly here. It can be measured as a TREND.

Scaling laws were found by training small models and fitting the curve, not by
training large ones. So: measure the penalty at several sizes and see which
way it moves.

  penalty shrinks with size  -> the premise gets CHEAPER as models grow, and
      the 235k result understates it. This is the strongest evidence
      obtainable without frontier compute.

  penalty grows with size    -> the wall is found now, before money is spent
      on it. Also valuable, and better learned here.

  penalty flat               -> size is not the variable, and the honest
      claim stays exactly where it is.

THREE ARMS PER SIZE.

  online     one pass, in order, never revisited. The premise.
  shuffled   the same data with order destroyed. The ceiling.
  accum-4    accumulation at the window that won on this task, to check
             whether the lever holds as the model grows rather than being a
             property of one size.

WHY THE COMPARISON IS FAIR EVEN THOUGH BIG MODELS ARE UNDERTRAINED HERE. At a
fixed data budget a 1.9M-parameter network is further from convergence than a
69k one, so absolute losses are not comparable across sizes and are not
compared. The penalty is a WITHIN-SIZE quantity: online against shuffled at
the same size, same data, same budget. Both arms are equally undertrained, so
the difference between them is not contaminated by that.

THE USABLE-CAPACITY CHECK, and it must look at the SHUFFLED arm. A first
version asked whether the ONLINE arm improved with size and refused to
interpret the trend when it did not. That check was wrong, and the 10,000
sentence run is why: online went 1.8247, 1.7898, 1.7905, 1.8106 (flat, then
worse) while shuffled went 1.7440, 1.6710, 1.6478, 1.6582 (clearly better
with size). The capacity WAS trainable at that budget. The online arm simply
could not use it.

So "the online arm did not improve with size" is not evidence of
undertraining. On this data it is the result. The check now asks whether the
SHUFFLED arm improved, because that is what establishes the extra parameters
are learnable at all at this budget.

TWO MEASURES OF THE PENALTY, because they can disagree. The absolute gap in
nats can shrink simply because all losses shrink. The gap as a fraction of
the online loss controls for that. A real trend should show in both.

  python sizesweep.py                          # 1 seed, ~90 minutes
  python sizesweep.py --seeds 2                # doubles it
  python sizesweep.py --train 1500 --eval 200  # smoke test, ~5 minutes
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


# hidden width -> the sizes are chosen to span roughly 27x in parameters
SIZES = [128, 256, 512, 768]

# name, accumulation window, shuffle
ARMS = [
    ("online", 1, False),
    ("shuffled", 1, True),
    ("accum-4", 4, False),
]


def train(hidden, accum, shuffle, train_lines, eval_lines, args, seed):
    torch.manual_seed(seed)
    device = pick_device()
    model = CharGRU(hidden, args.embed).to(device)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss()

    items = list(train_lines)
    if shuffle:
        random.Random(seed + 1000).shuffle(items)

    started = time.time()
    batch, pending, steps = [], 0, 0
    for line in items:
        ids = encode(line)
        if len(ids) < 2:
            continue
        model.train()
        x = torch.tensor([ids[:-1]], device=device)
        y = torch.tensor([ids[1:]], device=device)
        logits, _ = model(x)
        batch.append(lossf(logits.reshape(-1, V), y.reshape(-1)))
        pending += 1
        if pending >= accum:
            opt.zero_grad()
            (sum(batch) / len(batch)).backward()
            opt.step()
            steps += 1
            batch, pending = [], 0
    if batch:
        opt.zero_grad()
        (sum(batch) / len(batch)).backward()
        opt.step()
        steps += 1

    return dict(hidden=hidden, params=params, steps=steps,
                final=held_out_loss(model, eval_lines, device),
                seconds=time.time() - started)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/wiki.txt")
    ap.add_argument("--train", type=int, default=10000)
    ap.add_argument("--eval", type=int, default=800)
    ap.add_argument("--embed", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--out", default="sizesweep.json")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        raise SystemExit(f"no data at {os.path.abspath(args.data)}")

    lines = [l.strip() for l in open(args.data) if len(l.strip()) > 40]
    if len(lines) < args.train + args.eval:
        raise SystemExit(f"only {len(lines)} usable lines")

    train_lines = lines[:args.train]
    eval_lines = lines[-args.eval:]

    trigram = ngram_loss(" ".join(train_lines), eval_lines, 3)
    print(f"  {len(train_lines)} sentences, {len(eval_lines)} held out")
    print(f"  uniform floor {math.log(V):.4f}, trigram {trigram:.4f} nats")
    print(f"  sizes {SIZES}, {args.seeds} seed"
          f"{'s' if args.seeds > 1 else ''}\n", flush=True)

    runs = []
    for hidden in SIZES:
        for seed in range(args.seeds):
            for name, accum, shuf in ARMS:
                r = train(hidden, accum, shuf, train_lines, eval_lines,
                          args, seed)
                r["name"] = name
                r["seed"] = seed
                runs.append(r)
                print(f"  h={hidden:<4d} {name:9s} {r['final']:.4f} nats  "
                      f"{r['params']:>9,} params  {r['steps']:>6d} steps  "
                      f"{r['seconds']:.0f}s", flush=True)
        print(flush=True)

    with open(args.out, "w") as f:
        json.dump(dict(args=vars(args), trigram=trigram, runs=runs),
                  f, indent=2)

    def mean(xs):
        return sum(xs) / len(xs)

    def get(hidden, name):
        return mean([r["final"] for r in runs
                     if r["hidden"] == hidden and r["name"] == name])

    params = {h: [r["params"] for r in runs if r["hidden"] == h][0]
              for h in SIZES}

    print("=" * 78)
    print("SINGLE-PASS PENALTY vs MODEL SIZE")
    print("=" * 78)
    print(f"  {'params':>10} {'online':>8} {'shuffled':>9} {'gap':>8} "
          f"{'gap/online':>11} {'accum-4':>9} {'recovered':>10}")

    table = []
    for h in SIZES:
        on, sh, ac = get(h, "online"), get(h, "shuffled"), get(h, "accum-4")
        gap = on - sh
        rel = gap / on if on else float("nan")
        rec = (on - ac) / gap * 100 if abs(gap) > 1e-9 else float("nan")
        table.append((h, params[h], on, sh, gap, rel, ac, rec))
        print(f"  {params[h]:>10,} {on:>8.4f} {sh:>9.4f} {gap:>+8.4f} "
              f"{rel * 100:>10.1f}% {ac:>9.4f} {rec:>9.0f}%")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)

    smallest, largest = table[0], table[-1]

    # Is the extra capacity learnable at this budget at all? Judged on the
    # SHUFFLED arm, not the online one. If shuffled does not improve with
    # size then nothing does and the trend is about the training budget. If
    # shuffled improves and online does not, that difference IS the finding.
    sh_small, sh_large = smallest[3], largest[3]
    best_shuffled = min(t[3] for t in table)
    capacity_usable = best_shuffled < sh_small - 0.02

    if not capacity_usable:
        print(f"  UNDERTRAINED. The shuffled arm did not improve with size "
              f"({sh_small:.4f} at")
        print(f"  {smallest[1]:,} params, best {best_shuffled:.4f} across "
              f"the sweep). The extra")
        print(f"  parameters are not learnable at {args.train} sentences in "
              f"ANY arm, so the")
        print(f"  penalty trend is about training budget rather than size. "
              f"Re-run with more")
        print(f"  data before reading it.")
    else:
        print(f"  Capacity is usable at this budget: shuffled improved from "
              f"{sh_small:.4f} to")
        print(f"  {best_shuffled:.4f} across the sweep. So a flat or rising "
              f"online arm is a")
        print(f"  result, not an artifact.\n")

        on_small, on_large = smallest[2], largest[2]
        if on_large > on_small - 0.02:
            print(f"  ONLINE CANNOT USE THE CAPACITY. It went {on_small:.4f} "
                  f"to {on_large:.4f} while")
            print(f"  shuffled went {sh_small:.4f} to {sh_large:.4f}. Extra "
                  f"parameters help when")
            print(f"  the data is shuffled and do not help when it arrives "
                  f"in order.")
            print(f"  That is worse for the premise than a fixed penalty "
                  f"would be: the cost is")
            print(f"  not a constant, it is the ability to benefit from "
                  f"scale at all.")

        gaps = [t[4] for t in table]
        rels = [t[5] for t in table]
        abs_down = gaps[-1] < gaps[0] - 0.005
        rel_down = rels[-1] < rels[0] - 0.005
        abs_up = gaps[-1] > gaps[0] + 0.005
        rel_up = rels[-1] > rels[0] + 0.005

        print(f"  absolute gap  {gaps[0]:+.4f} -> {gaps[-1]:+.4f}")
        print(f"  relative gap  {rels[0] * 100:.1f}% -> "
              f"{rels[-1] * 100:.1f}%")

        if abs_down and rel_down:
            print(f"\n  THE PENALTY SHRINKS WITH SIZE on both measures. The "
                  f"single pass costs")
            print(f"  LESS as the model grows, which is the direction the "
                  f"vision needs and the")
            print(f"  strongest scaling evidence obtainable without frontier "
                  f"compute. It is a")
            print(f"  trend over {len(SIZES)} points spanning "
                  f"{params[SIZES[-1]] / params[SIZES[0]]:.0f}x, not a law.")
        elif abs_up and rel_up:
            print(f"\n  THE PENALTY GROWS WITH SIZE on both measures. Larger "
                  f"models are hurt")
            print(f"  MORE by learning in order, which points at a wall "
                  f"rather than a ramp.")
            print(f"  Better to know this now. The honest claim narrows to "
                  f"small models.")
        elif abs_down != rel_down:
            print(f"\n  THE MEASURES DISAGREE. The absolute gap and the "
                  f"relative gap move in")
            print(f"  different directions, which usually means the absolute "
                  f"gap is tracking")
            print(f"  the overall loss rather than the penalty. Report both "
                  f"and claim neither.")
        else:
            print(f"\n  FLAT. The penalty does not move meaningfully across "
                  f"{params[SIZES[-1]] / params[SIZES[0]]:.0f}x in")
            print(f"  parameters. Size is not the variable, and the claim "
                  f"stays where it is.")

        recs = [t[7] for t in table]
        if min(recs) > 20:
            print(f"\n  Accumulation held at every size "
                  f"({min(recs):.0f}% to {max(recs):.0f}% of the gap "
                  f"recovered),")
            print(f"  so the lever is not a property of one model size.")
        else:
            print(f"\n  Accumulation did NOT hold across sizes "
                  f"({min(recs):.0f}% to {max(recs):.0f}%). It is size "
                  f"dependent,")
            print(f"  which the 235k result alone could not have shown.")

    print(f"\n  Character level, one domain, {args.seeds} seed"
          f"{'s' if args.seeds > 1 else ''}, at most "
          f"{params[SIZES[-1]]:,} parameters. A trend at this scale is a "
          f"lead.")
    print(f"\n  written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()