"""One life, run long enough for slow problems to show up.

WHAT HAS NEVER BEEN TESTED. Every experiment in this repo is short.
The longest is a 100,000-step grid-world life from August; most are a few
thousand items and finish in seconds. The central claim of the project is
that a system accumulates a life over time, and nothing here has ever run
long enough for time to be the variable.

Short runs cannot show slow failures, and slow failures are the ones that
matter for anything left switched on:

  DRIFT           competence falling gradually rather than collapsing, so
                  no single moment looks wrong.
  LEAK            the anchor, the buffer or the optimiser state growing
                  without bound until the process dies.
  ROT             the canary baseline and the live model drifting apart
                  until the guard either fires constantly or never fires.
  SATURATION      the replay buffer filling with one kind of material and
                  the system becoming unable to learn anything else.
  ROLLBACK STORMS the guard firing repeatedly and undoing everything,
                  which in August it never did once in 100,000 steps and
                  so has never been observed at length.

This is deliberately the CHEAP version: a GRU on a grid, which runs about
a thousand steps a second, so millions of steps fit in minutes of wall
clock. It exercises the layer, the replay buffer and the guard. It does
NOT exercise the sentence store, the embedding index or eviction, because
there is no language here. Those need the real overnight run on a 1.5B and
this is the check that runs BEFORE committing a night to that.

WHAT IS MEASURED, every CHECK_EVERY steps, on a held-out probe set built
from the rule directly rather than from the agent's experience:

  accuracy        on the CURRENT rule, and on the rule from before the
                  last reversal. The second is retention: how much of what
                  it knew survives being contradicted.
  health          canary drift, the guard's own measure of damage.
  sizes           anchor, buffer, and the number of stored sequences.
  rollbacks       cumulative, so a storm is visible as a slope.
  wall clock      per block, so a slowdown shows up as a rising number
                  rather than as a run that mysteriously never finishes.

READING IT. The last block against the first is the whole point. A system
that is the same at step 3,000,000 as it was at step 100,000 has survived
the thing this file exists to test. Anything monotonic in the wrong
direction is a slow failure, and it is far better to find one here than
twelve hours into a language run.

    python longrun.py                       # ~15 minutes
    python longrun.py --steps 10000000      # longer
"""

import argparse
import json
import os
import random
import statistics
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))

from sleeping import SleepLayer          # noqa: E402
from survive import (World, Backend, make_canary, encode,      # noqa: E402
                     ACTIONS, TARGET, EMPTY, BERRY, FUNGUS,
                     NOTHING, GOOD, BAD, REHEARSE_PER_ITEM, SEQ_LEN)


REVERSAL_EVERY = 20_000      # long enough to learn, often enough to matter
CHECK_EVERY = 100_000
SLEEP_EVERY = 500
SLEEP_ROUNDS = SLEEP_EVERY // REHEARSE_PER_ITEM
PROBES = 240


def make_probes(seed, berry_good):
    """Held-out situations under a FIXED rule.

    Generated from the rule rather than sampled from what the agent saw,
    so an agent that stopped visiting fungus is still tested on fungus.
    Otherwise the probe set inherits whatever bias the policy developed
    and competence looks stable because the test quietly got easier.
    """
    rng = random.Random(20_000 + seed + (0 if berry_good else 1))
    out = []
    for _ in range(PROBES):
        patch = [rng.choice([EMPTY, EMPTY, BERRY, FUNGUS])
                 for _ in range(9)]
        action = rng.choice(ACTIONS)
        cell = patch[TARGET[action]]
        if cell == EMPTY:
            outcome = NOTHING
        else:
            outcome = GOOD if (cell == BERRY) == berry_good else BAD
        out.append((patch, action, outcome))
    return out


def accuracy(b, probes):
    b.net.eval()
    hit = 0
    with torch.no_grad():
        for patch, action, outcome in probes:
            logits, _ = b.net(encode(patch, action), None)
            hit += int(logits.argmax(1).item() == outcome)
    return 100.0 * hit / len(probes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="longrun.json")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    b = Backend(args.seed)
    layer = SleepLayer(
        b, canary=make_canary(args.seed), seed=args.seed, contiguous=True,
        window=200, warmup=30, top_fraction=1.00,
        coherence_veto=1e9, loss_floor=0.02, steps_per_update=1,
        rehearse_per_item=REHEARSE_PER_ITEM, rehearse_count=2,
        rehearse_steps=1, anchor_size=100, buffer_size=500,
        sequence_len=SEQ_LEN, replay_policy="uniform",
        guard=True, guard_per_item=1000, canary_tolerance=0.5)

    probes_a = make_probes(args.seed, True)    # berry good
    probes_b = make_probes(args.seed, False)   # fungus good

    world = World(args.seed)
    berry_good = True

    print(f"  {args.steps:,} steps, one continuous life, no resets")
    print(f"  rule reverses every {REVERSAL_EVERY:,}, sleeps every "
          f"{SLEEP_EVERY} for {SLEEP_ROUNDS} rounds")
    print(f"  checking every {CHECK_EVERY:,} steps on {PROBES} held-out "
          f"probes per rule\n")
    print(f"  {'step':>10} {'current':>8} {'previous':>9} {'health':>8} "
          f"{'anchor':>7} {'buffer':>7} {'seqs':>8} {'rb':>5} "
          f"{'sleeps':>7} {'sec':>6}")
    print("-" * 92)

    marks = []
    t0 = time.time()
    block = time.time()
    since = 0
    sleeps = 0

    for step in range(1, args.steps + 1):
        if step % REVERSAL_EVERY == 0:
            world.reverse()
            berry_good = not berry_good

        patch = world.observe()
        action = random.choice(ACTIONS)   # passive walk: this measures the
        # LAYER over a long life, not a policy. A policy would confound
        # slow layer failure with slow policy collapse, and the survival
        # work already showed the policy can collapse on its own.
        outcome, _, _ = world.step(action)
        layer.observe((patch, action, outcome))
        since += 1

        if since >= SLEEP_EVERY:
            layer.sleep(SLEEP_ROUNDS)
            sleeps += 1
            since = 0

        if step % CHECK_EVERY == 0:
            cur = probes_a if berry_good else probes_b
            prev = probes_b if berry_good else probes_a
            s = layer.summary()
            m = dict(step=step,
                     current=accuracy(b, cur),
                     previous=accuracy(b, prev),
                     health=s.get("health", 0.0),
                     anchor=s.get("anchor", 0),
                     buffer=s.get("buffer", 0),
                     sequences=s.get("sequences", 0),
                     rollbacks=s.get("rollbacks", 0),
                     sleeps=sleeps,
                     seconds=time.time() - block)
            marks.append(m)
            print(f"  {step:>10,} {m['current']:>7.1f}% "
                  f"{m['previous']:>8.1f}% {m['health']:>+8.3f} "
                  f"{m['anchor']:>7} {m['buffer']:>7} "
                  f"{m['sequences']:>8} {m['rollbacks']:>5} "
                  f"{m['sleeps']:>7} {m['seconds']:>6.0f}", flush=True)
            block = time.time()

    with open(args.out, "w") as f:
        json.dump(dict(args=vars(args), marks=marks), f, indent=2)

    print("\n" + "=" * 92)
    print("DID ANYTHING DEGRADE SLOWLY?")
    print("=" * 92)

    n = max(1, len(marks) // 4)
    first, last = marks[:n], marks[-n:]

    def avg(rows, key):
        return statistics.mean(r[key] for r in rows)

    for key, label, worse in [
            ("current", "accuracy on the current rule", "down"),
            ("previous", "accuracy on the previous rule", "down"),
            ("health", "canary drift (guard's damage measure)", "up"),
            ("buffer", "replay buffer size", "up"),
            ("seconds", "wall clock per block", "up")]:
        a, z = avg(first, key), avg(last, key)
        d = z - a
        flag = ""
        if worse == "down" and d < -5:
            flag = "   DEGRADED"
        if worse == "up" and key == "health" and d > 0.3:
            flag = "   DRIFTING"
        if worse == "up" and key == "seconds" and a and z > a * 1.5:
            flag = "   SLOWING"
        print(f"  {label:<38} {a:>9.2f} -> {z:>9.2f}  {d:>+9.2f}{flag}")

    rb = marks[-1]["rollbacks"]
    print(f"\n  rollbacks over the whole life: {rb}")
    if rb == 0:
        print(f"  The guard never fired, as in August. It has now gone "
              f"{args.steps:,} steps")
        print(f"  without being observed doing anything, which is worth "
              f"saying plainly:")
        print(f"  a mechanism that has never fired is not protection, it "
              f"is an untested claim.")
    elif rb > len(marks) * 2:
        print(f"  ROLLBACK STORM. The guard fired {rb} times, which is "
              f"more than twice per")
        print(f"  check block. Repeated rollbacks undo learning faster "
              f"than it accumulates.")

    total = time.time() - t0
    print(f"\n  {args.steps:,} steps in {total / 60:.1f} minutes "
          f"({args.steps / max(1, total):,.0f}/s)")
    print(f"  This is the CHEAP check. It exercises the layer, the buffer "
          f"and the guard.")
    print(f"  It does NOT exercise the sentence store, the embedding index "
          f"or eviction,")
    print(f"  which is what the overnight language run is for.")
    print(f"\n  written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
