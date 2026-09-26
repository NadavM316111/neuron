"""Which part of the layer is costing the agent its life?

THE RESULT THAT PROMPTED THIS. In lineage.py, 10 lineages of 20 lives, the
arms carrying the full stability layer lost to a random walker:

    reflex      302     random movement, no learning
    online      272     learns every step, NO layer
    continuous  263     layer, interleaved consolidation
    sleeping    260     layer, offline consolidation
    driven      566     layer, action by accumulated error

Bare online learning beats both layered arms, and the random walker beats
all three. Every mechanism in stability.py was validated on PASSIVE streams
at a FIXED horizon, where missing an item costs nothing and there is no
price for being out of date. This world charges for both, and it is the
first setting where those design choices could show a cost.

THE HYPOTHESIS, and it is specific. This world REVERSES the rule every 300
steps. Replay trains the model on stored experience, and in a reversing
world stored experience is WRONG about half the time: replaying a run from
before the last reversal teaches the agent the rule that is currently
killing it. On a passive stream that shows up as a small retention cost. On
an acting agent with energy it shows up as death.

If that is right, removing replay should HELP here, which would be the
opposite of every previous result in this repo, and would mean the
retention machinery is a liability whenever the world can change under an
agent that must act.

THE ARMS. Each removes exactly one thing from the full layer, so whatever
the numbers say, they say it about one component.

  online        no layer at all. The baseline the layer has to beat.
  full          the layer as lineage.py ran it.
  no-replay     rehearsal switched off. The hypothesis above.
  no-guard      canary rollback switched off. A rollback discards recent
                learning, and in a reversing world the most recent
                learning is the only correct learning. The guard fired
                zero times across a 100,000-step grid-world life in
                August, so it has barely ever been observed doing
                anything.
  no-floor      loss_floor=0, so updates are not skipped on
                already-predicted material. The floor exists because Adam
                amplifies near-zero gradients, but it also means an agent
                stops learning about whatever it currently predicts well,
                which after a reversal is exactly the stale rule.
  recent-only   replay kept, but the buffer holds only 40 units instead of
                500, so replay can only reach back a short way. If the
                problem is staleness rather than replay itself, this
                should recover most of the loss while keeping the
                protection.

READING IT. `online` is the bar. Any arm at or below it is carrying a
component that costs more than it gives in this setting. The interesting
outcome is not which arm wins but WHICH SINGLE REMOVAL closes the gap,
because that names the mechanism.

All arms inherit across deaths, since lineage.py established that nothing
learns anything without it (scratch 230 against 272 for the same arm
inheriting, with a flat accumulation curve).

    python ablate.py                    # 10 lineages, 20 lives each
    python ablate.py --runs 20
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
                     GOOD, START_ENERGY, MAX_ENERGY, STEP_COST,
                     FOOD_GAIN, POISON_COST, SLEEP_COST_PER_ROUND,
                     REVERSAL_STEPS, REHEARSE_PER_ITEM, SEQ_LEN)


MAX_LIFE = 3000

# name -> what changes from the full layer
VARIANTS = {
    "full":        dict(),
    "no-replay":   dict(rehearse_count=0),
    "no-guard":    dict(guard=False),
    "no-floor":    dict(loss_floor=0.0),
    "recent-only": dict(buffer_size=40, anchor_size=8),
}
ARMS = ["online"] + list(VARIANTS)


def make_layer(backend, seed, variant):
    """The full layer, with exactly one thing changed.

    Everything not named in VARIANTS[variant] is identical across arms, so
    a difference between two arms is attributable to one component rather
    than to a configuration drift.
    """
    cfg = dict(
        contiguous=True,
        window=200, warmup=30, top_fraction=1.00,
        coherence_veto=1e9, loss_floor=0.02, steps_per_update=1,
        rehearse_per_item=REHEARSE_PER_ITEM, rehearse_count=2,
        rehearse_steps=1, anchor_size=100, buffer_size=500,
        sequence_len=SEQ_LEN, replay_policy="uniform",
        guard=True, guard_per_item=1000, canary_tolerance=0.5)
    cfg.update(VARIANTS[variant])

    canary = make_canary(seed) if cfg.get("guard", True) else None
    return SleepLayer(backend, canary=canary, seed=seed, **cfg)


def one_life(arm, b, layer, world_seed, max_life):
    world = World(world_seed)
    energy = START_ENERGY
    since = 0
    eaten, poisoned = 0, 0

    for step in range(max_life):
        if step and step % REVERSAL_STEPS == 0:
            world.reverse()

        patch = world.observe()
        action = b.act(patch, "value")
        target_cell = patch[TARGET[action]]
        outcome, cell, delta = world.step(action)

        energy = min(energy + delta - STEP_COST, MAX_ENERGY)
        if outcome == GOOD:
            eaten += 1
        elif delta < 0:
            poisoned += 1
        if energy <= 0:
            return step + 1, eaten, poisoned

        item = (patch, action, outcome)
        s, _ = b.score(item)
        b.feel(target_cell, action, s)

        if layer is None:
            b.update(item, 1)
            continue

        layer.observe(item)
        since += 1
        if since >= REHEARSE_PER_ITEM:
            cost = SLEEP_COST_PER_ROUND
            if energy > cost + 5.0:
                layer.sleep(1)
                energy -= cost
            since = 0
        if energy <= 0:
            return step + 1, eaten, poisoned

    return max_life, eaten, poisoned


def run_lineage(arm, run_seed, lives, max_life):
    random.seed(run_seed * 977 + 13)
    torch.manual_seed(run_seed)
    b = Backend(run_seed)
    layer = None if arm == "online" else make_layer(b, run_seed, arm)

    out = []
    for g in range(lives):
        steps, ate, pois = one_life(
            arm, b, layer, run_seed * 1000 + g, max_life)
        out.append(dict(gen=g, steps=steps, eaten=ate, poisoned=pois))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lives", type=int, default=20)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--max-life", type=int, default=MAX_LIFE)
    ap.add_argument("--out", default="ablate.json")
    args = ap.parse_args()

    print(f"  {args.runs} lineages per arm, {args.lives} lives each")
    print(f"  the rule reverses every {REVERSAL_STEPS} steps, so stored "
          f"experience goes stale")
    print(f"  each arm removes ONE thing from the full layer\n", flush=True)

    data = {}
    for arm in ARMS:
        t0 = time.time()
        data[arm] = [run_lineage(arm, r, args.lives, args.max_life)
                     for r in range(args.runs)]
        s = [g["steps"] for ln in data[arm] for g in ln]
        print(f"  {arm:>12}  median {statistics.median(s):>6.0f}  "
              f"{time.time() - t0:.0f}s", flush=True)

    with open(args.out, "w") as f:
        json.dump(dict(args=vars(args), data=data), f, indent=2)

    def med(arm):
        return statistics.median(g["steps"] for ln in data[arm]
                                 for g in ln)

    def col(arm, key):
        return statistics.mean(g[key] for ln in data[arm] for g in ln)

    print("\n" + "=" * 78)
    print("WHICH REMOVAL CLOSES THE GAP?")
    print("=" * 78)
    base = med("online")
    print(f"  {'arm':>12} {'median':>8} {'vs online':>11} {'ate':>7} "
          f"{'poisoned':>9}")
    print("-" * 78)
    print(f"  {'online':>12} {base:>8.0f} {'':>11} "
          f"{col('online', 'eaten'):>7.0f} "
          f"{col('online', 'poisoned'):>9.0f}")
    for arm in VARIANTS:
        print(f"  {arm:>12} {med(arm):>8.0f} {med(arm) - base:>+11.0f} "
              f"{col(arm, 'eaten'):>7.0f} {col(arm, 'poisoned'):>9.0f}")

    print("\n" + "=" * 78)
    print("WHAT IT SAYS")
    print("=" * 78)

    full = med("full")
    if full >= base:
        print(f"  THE LAYER IS NOT THE PROBLEM HERE: full {full:.0f} "
              f"against online {base:.0f}. The")
        print(f"  lineage.py gap did not reproduce, which means it was "
              f"noise or something else")
        print(f"  in that harness. Nothing below is worth reading.")
        return

    gains = {a: med(a) - full for a in VARIANTS if a != "full"}
    best = max(gains, key=gains.get)
    print(f"  full layer costs {base - full:.0f} steps against no layer "
          f"at all.")
    print(f"  Removing one thing at a time:")
    for a in sorted(gains, key=gains.get, reverse=True):
        print(f"    {a:>12} {gains[a]:>+8.0f}")

    if gains[best] <= 10:
        print(f"\n  NO SINGLE COMPONENT EXPLAINS IT. The best removal "
              f"recovered {gains[best]:+.0f} steps,")
        print(f"  so the cost is spread across the layer rather than "
              f"sitting in one mechanism.")
    elif best in ("no-replay", "recent-only"):
        print(f"\n  REPLAY IS THE LIABILITY. Removing or shortening it "
              f"recovered {gains[best]:+.0f} steps.")
        print(f"  In a world that reverses every {REVERSAL_STEPS} steps, "
              f"stored experience is wrong")
        print(f"  about half the time, and replaying it teaches the agent "
              f"the rule that is")
        print(f"  currently killing it. Retention is protection on a "
              f"passive stream and a")
        print(f"  liability for an agent that has to act on what is true "
              f"NOW.")
        if gains.get("recent-only", -1) >= gains.get("no-replay", -1) - 10:
            print(f"\n  And shortening is as good as removing, which says "
                  f"the problem is STALENESS")
            print(f"  rather than replay itself. A buffer that only "
                  f"reaches back a short way keeps")
            print(f"  the protection without teaching the past.")
    elif best == "no-guard":
        print(f"\n  THE GUARD IS THE LIABILITY. Removing it recovered "
              f"{gains[best]:+.0f} steps. A rollback")
        print(f"  discards recent learning, and after a reversal the "
              f"recent learning is the")
        print(f"  only correct learning, so the guard undoes exactly what "
              f"the agent needs.")
        print(f"  It fired zero times in the August grid-world life, so "
              f"this is close to the")
        print(f"  first time it has been observed doing anything at all.")
    elif best == "no-floor":
        print(f"\n  THE LOSS FLOOR IS THE LIABILITY. Removing it "
              f"recovered {gains[best]:+.0f} steps. The")
        print(f"  floor skips updates on material the model already "
              f"predicts well, which after")
        print(f"  a reversal is precisely the stale rule it most needs to "
              f"unlearn.")

    print(f"\n  One world, {args.runs} lineages. What might transfer is "
          f"that mechanisms tuned on")
    print(f"  passive streams need re-checking whenever the agent acts "
          f"and the world moves.")
    print(f"\n  written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
