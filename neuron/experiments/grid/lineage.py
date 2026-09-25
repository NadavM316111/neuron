"""A lineage. Each death is real, but the brain carries over.

WHY survive.py COULD NOT WORK. In that file every agent started from random
weights, lived about 250 steps, ate about nine things and died. Nine
labelled examples split across three outcome classes is not enough to train
anything, so the network never separated food from poison from empty: the
policy scores in a live trace sat between -1.7 and -0.7 for every direction
at every step, which is the model still at its base rates. Every arm was
measuring an agent that died before it could learn, and the random walker
beat all of them (median 338 against 235 to 248).

That is not a tuning problem. Death arrived faster than learning, and no
constant fixes that.

WHAT CHANGES HERE. The brain persists across deaths. When an agent dies the
next one starts with the same weights and the same replay buffer, in a
fresh world, at full energy, in a new position.

  life 1    knows nothing, dies quickly
  life 20   carries whatever nineteen deaths taught it

Death still costs something, which was the point of the whole design: the
agent loses its energy, its position, and whatever it was in the middle of.
It does not lose everything it ever learned. That is the difference between
dying and never having existed, and it is what makes the stakes real
without making them pointless.

This is the August inheritance work (Stage 4) used for something rather
than tested on its own.

THE CONTROL THAT MAKES IT READABLE. `scratch` is the same arm with
inheritance switched OFF: every life starts from random weights, exactly as
survive.py did. If the inheriting arms do not clearly outlive it, then
carrying the brain over is not buying anything and nothing else in the run
is interpretable.

THE ARMS.

  reflex       random movement, no learning. The floor, and the thing
               every learning arm failed to beat in survive.py.
  scratch      learns, no inheritance. survive.py's condition.
  online       learns, inherits. No layer, no replay.
  continuous   inherits, layer with interleaved consolidation.
  sleeping     inherits, layer with offline consolidation in blocks.
               Sleep costs energy and gathers no food.
  driven       sleeping, plus action chosen by accumulated prediction
               error rather than predicted value.

WHAT TO READ, IN ORDER.

  1. Does lifespan RISE across generations? If the last five lives are no
     longer than the first five, nothing accumulated and the whole premise
     is wrong. Printed per arm as a curve, not just a mean.
  2. Do the inheriting arms beat `scratch`? If not, inheritance is not the
     fix.
  3. Do any of them beat `reflex`? Until that holds, learning is still
     worth nothing in this world.
  4. Only then: does sleeping beat interleaving, and does the drive pay?

THE BREAKEVEN CONSTRAINT, carried over from survive.py and worth repeating
because it is the thing that took four wrong diagnoses to find. An agent
choosing by expected energy eats when p(good)*FOOD exceeds p(bad)*POISON,
so the accuracy at which eating becomes worthwhile is POISON/(FOOD+POISON).
That has to be BELOW chance or an untrained agent is correctly better off
starving and can never bootstrap. At 18 and 12 it is 0.40.

    python lineage.py                       # 20 lives, 6 arms
    python lineage.py --lives 40 --runs 3   # slower, steadier
"""

import argparse
import json
import os
import random
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))

from sleeping import SleepLayer          # noqa: E402
from survive import (World, Backend, make_canary, TARGET,      # noqa: E402
                     GOOD, BAD, START_ENERGY, MAX_ENERGY, STEP_COST,
                     FOOD_GAIN, POISON_COST, SLEEP_COST_PER_ROUND,
                     REVERSAL_STEPS, SLEEP_EVERY, SLEEP_ROUNDS,
                     REHEARSE_PER_ITEM, SEQ_LEN)


MAX_LIFE = 3000          # one life this long counts as survival
ARMS = ["reflex", "scratch", "online", "continuous", "sleeping", "driven"]

LAYERED = ("continuous", "sleeping", "driven")
OFFLINE = ("sleeping", "driven")
INHERITS = ("online", "continuous", "sleeping", "driven")


def make_layer(backend, seed):
    return SleepLayer(
        backend, canary=make_canary(seed), seed=seed, contiguous=True,
        window=200, warmup=30, top_fraction=1.00,
        coherence_veto=1e9, loss_floor=0.02, steps_per_update=1,
        rehearse_per_item=REHEARSE_PER_ITEM, rehearse_count=2,
        rehearse_steps=1, anchor_size=100, buffer_size=500,
        sequence_len=SEQ_LEN, replay_policy="uniform",
        guard=True, guard_per_item=1000, canary_tolerance=0.5)


def one_life(arm, b, layer, world_seed, max_life):
    """One life in a fresh world. The brain passed in is whatever the
    previous life left behind, or a new one if the arm does not inherit.

    Returns how long it lasted and what it did. Nothing about the brain is
    reset here: that is the caller's decision and it is the whole
    experiment.
    """
    world = World(world_seed)
    policy = {"reflex": "reflex", "scratch": "value", "online": "value",
              "continuous": "value", "sleeping": "value",
              "driven": "driven"}[arm]

    energy = START_ENERGY
    since = 0
    eaten, poisoned, sleeps = 0, 0, 0

    for step in range(max_life):
        if step and step % REVERSAL_STEPS == 0:
            world.reverse()

        patch = world.observe()
        action = b.act(patch, policy)
        target_cell = patch[TARGET[action]]
        outcome, cell, delta = world.step(action)

        energy = min(energy + delta - STEP_COST, MAX_ENERGY)
        if outcome == GOOD:
            eaten += 1
        elif outcome == BAD:
            poisoned += 1
        if energy <= 0:
            return step + 1, eaten, poisoned, sleeps

        if arm == "reflex":
            continue

        item = (patch, action, outcome)
        s, _ = b.score(item)
        b.feel(target_cell, action, s)

        if layer is None:
            b.update(item, 1)
            continue

        layer.observe(item)
        since += 1

        if arm in OFFLINE:
            if since >= SLEEP_EVERY:
                cost = SLEEP_ROUNDS * SLEEP_COST_PER_ROUND
                if energy > cost + 5.0:
                    layer.sleep(SLEEP_ROUNDS)
                    energy -= cost
                    sleeps += 1
                since = 0
        else:
            if since >= REHEARSE_PER_ITEM:
                cost = SLEEP_COST_PER_ROUND
                if energy > cost + 5.0:
                    layer.sleep(1)
                    energy -= cost
                    sleeps += 1
                since = 0

        if energy <= 0:
            return step + 1, eaten, poisoned, sleeps

    return max_life, eaten, poisoned, sleeps


def run_lineage(arm, run_seed, lives, max_life):
    """One lineage: `lives` deaths in a row, brain carried over or not."""
    random.seed(run_seed * 977 + 13)
    torch.manual_seed(run_seed)

    b = Backend(run_seed)
    layer = make_layer(b, run_seed) if arm in LAYERED else None

    out = []
    for g in range(lives):
        if arm not in INHERITS and arm != "reflex":
            # The survive.py condition: every life from random weights.
            b = Backend(run_seed * 100 + g)
            layer = make_layer(b, run_seed) if arm in LAYERED else None
        steps, ate, pois, sl = one_life(
            arm, b, layer, run_seed * 1000 + g, max_life)
        out.append(dict(gen=g, steps=steps, eaten=ate, poisoned=pois,
                        sleeps=sl))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lives", type=int, default=20,
                    help="deaths per lineage")
    ap.add_argument("--runs", type=int, default=3,
                    help="independent lineages per arm")
    ap.add_argument("--max-life", type=int, default=MAX_LIFE)
    ap.add_argument("--out", default="lineage.json")
    args = ap.parse_args()

    print(f"  {args.runs} lineages per arm, {args.lives} lives each")
    print(f"  food +{FOOD_GAIN}, poison -{POISON_COST}  "
          f"(eating pays above {POISON_COST / (FOOD_GAIN + POISON_COST):.2f} "
          f"accuracy, chance is 0.50)")
    print(f"  the rule reverses every {REVERSAL_STEPS} steps")
    print(f"  a life reaching {args.max_life} counts as survival\n",
          flush=True)

    data = {}
    for arm in ARMS:
        t0 = time.time()
        data[arm] = [run_lineage(arm, r, args.lives, args.max_life)
                     for r in range(args.runs)]
        allsteps = [g["steps"] for ln in data[arm] for g in ln]
        print(f"  {arm:>11}  median {statistics.median(allsteps):>6.0f}  "
              f"{time.time() - t0:.0f}s", flush=True)

    with open(args.out, "w") as f:
        json.dump(dict(args=vars(args), data=data), f, indent=2)

    def gen_mean(arm, lo, hi):
        vals = [g["steps"] for ln in data[arm] for g in ln
                if lo <= g["gen"] < hi]
        return statistics.mean(vals) if vals else float("nan")

    def col(arm, key):
        return statistics.mean(g[key] for ln in data[arm] for g in ln)

    k = max(1, args.lives // 4)

    print("\n" + "=" * 80)
    print("DID ANYTHING ACCUMULATE? mean lifespan by quarter of the "
          "lineage")
    print("=" * 80)
    print(f"  {'arm':>11} {'first':>8} {'second':>8} {'third':>8} "
          f"{'last':>8} {'change':>9}")
    print("-" * 80)
    rose = {}
    for arm in ARMS:
        q = [gen_mean(arm, i * k, (i + 1) * k) for i in range(4)]
        rose[arm] = q[3] - q[0]
        print(f"  {arm:>11} {q[0]:>8.0f} {q[1]:>8.0f} {q[2]:>8.0f} "
              f"{q[3]:>8.0f} {q[3] - q[0]:>+9.0f}")

    print("\n" + "=" * 80)
    print("WHOLE LINEAGE")
    print("=" * 80)
    print(f"  {'arm':>11} {'median':>8} {'mean':>8} {'best':>8} "
          f"{'ate':>7} {'poisoned':>9} {'sleeps':>7}")
    print("-" * 80)
    med = {}
    for arm in ARMS:
        allsteps = [g["steps"] for ln in data[arm] for g in ln]
        med[arm] = statistics.median(allsteps)
        print(f"  {arm:>11} {med[arm]:>8.0f} "
              f"{statistics.mean(allsteps):>8.0f} {max(allsteps):>8} "
              f"{col(arm, 'eaten'):>7.0f} {col(arm, 'poisoned'):>9.0f} "
              f"{col(arm, 'sleeps'):>7.0f}")

    print("\n" + "=" * 80)
    print("CHECKS, IN ORDER. STOP AT THE FIRST FAILURE.")
    print("=" * 80)

    ok = True

    best_rise = max(rose[a] for a in INHERITS)
    if best_rise < 20:
        ok = False
        print(f"  1. NOTHING ACCUMULATED. The best inheriting arm gained "
              f"{best_rise:+.0f} steps from")
        print(f"     the first quarter of its lineage to the last. "
              f"Carrying the brain over did")
        print(f"     not let anything build up, so the premise of this "
              f"file is wrong and nothing")
        print(f"     below is worth reading. Check the ate column: if it "
              f"is still single digits,")
        print(f"     the agents are still abstaining and still dying "
              f"before they can learn.")
    else:
        print(f"  1. ok: lifespan rose by up to {best_rise:+.0f} steps "
              f"across a lineage, so")
        print(f"     something accumulated across deaths.")

    if ok:
        scratch = med["scratch"]
        best_inh = max(med[a] for a in INHERITS)
        if best_inh <= scratch * 1.1:
            ok = False
            print(f"  2. INHERITANCE BOUGHT NOTHING: best inheriting arm "
                  f"{best_inh:.0f} against")
            print(f"     scratch's {scratch:.0f}. Starting each life from "
                  f"random weights does just as")
            print(f"     well, so the brain carrying over is not what "
                  f"matters here.")
        else:
            print(f"  2. ok: inheriting arms reach {best_inh:.0f} against "
                  f"scratch's {scratch:.0f}.")

    if ok:
        reflex = med["reflex"]
        best_inh = max(med[a] for a in INHERITS)
        if best_inh <= reflex * 1.1:
            ok = False
            print(f"  3. STILL LOSING TO RANDOM: best learner "
                  f"{best_inh:.0f} against the random")
            print(f"     walker's {reflex:.0f}. Learning is worth nothing "
                  f"in this world even with a")
            print(f"     lineage behind it, which is the same wall "
                  f"survive.py hit.")
        else:
            print(f"  3. ok: learning beats random, {best_inh:.0f} "
                  f"against {reflex:.0f}. LEARNING IS")
            print(f"     WORTH SOMETHING HERE, which is the first time "
                  f"that has been true.")

    if ok:
        print("\n" + "=" * 80)
        print("AND THEN THE MECHANISMS")
        print("=" * 80)
        print(f"  the layer over bare online  "
              f"{med['continuous'] - med['online']:>+8.0f}")
        print(f"  sleeping over interleaved   "
              f"{med['sleeping'] - med['continuous']:>+8.0f}")
        print(f"  the drive over value        "
              f"{med['driven'] - med['sleeping']:>+8.0f}")

        if med["sleeping"] > med["continuous"]:
            print(f"\n  SLEEP SURVIVES BEING EXPENSIVE. It won on accuracy "
                  f"on 20 Sep when")
            print(f"  consolidating was free. Here it costs energy and "
                  f"gathers no food, and")
            print(f"  the bet still pays.")
        else:
            print(f"\n  SLEEP DOES NOT PAY FOR ITSELF here. It won on "
                  f"accuracy when it was free")
            print(f"  and loses when charged for, which is worth knowing "
                  f"before more is built")
            print(f"  on the 20 Sep result.")

    print(f"\n  A 9x9 grid and a GRU. What might transfer is the ORDERING "
          f"and the shape of")
    print(f"  the accumulation curve, not the numbers.")
    print(f"\n  written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
