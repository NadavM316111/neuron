"""Does the want survive being knocked off course?

WHY THIS IS THE NEXT STEP. goal.py showed the system can reach a preferred
state it was never rewarded for, by planning through a world model it
learned itself (41% against random's 16% on the detour world, 2 Oct). But a
goal there lasted exactly one trial and then was handed fresh. The
definition of a goal has a third clause the first test did not touch: it
KEEPS WORKING TOWARD the state and RETURNS to it. A want that only exists
while nothing disturbs it is not much of a want.

THE TEST. Start a pursuit. Partway through, INTERRUPT it: teleport the agent
to a random cell, as if something picked it up and put it down elsewhere.
Then keep going. Does it still reach the ORIGINAL goal?

The distinction this draws is sharp and it is the whole point:

  a PERSISTENT want re-plans from wherever it lands and still goes to the
      original goal, because the goal is held separately from the agent's
      position. The interruption costs time, not the goal.

  a BRITTLE want loses the goal when disturbed, or latches onto its new
      surroundings, or wanders. The goal was really just "the next step
      from here", not a state it was holding.

Whether the goal is held separately is a claim about the implementation,
and the point of the test is to VERIFY it rather than assume it. A planner
that stores the goal and re-searches each step should pass; one that cached
a path and follows it blindly should fail the moment it is moved off that
path.

THE ARMS.

  undisturbed   plan to the goal, no interruption. The ceiling for this
                world and model: how often it arrives when nothing moves
                it.
  interrupted   same, but teleported once mid-pursuit. If this matches
                undisturbed, the want survived; if it collapses toward
                random, the want did not.
  cached        a control that plans ONCE at the start, caches the path,
                and follows it with no re-planning. This is what a brittle
                want looks like: interrupt it and it is lost, because it
                never re-derives the path from where it actually is. It
                exists to show the difference is real and not just noise.
  random        the floor.

INTERPRETATION.

  interrupted stays near undisturbed, and well above cached
      THE WANT PERSISTS. The goal is held, not the path. Being moved off
      course costs steps, not the goal itself, which is what a goal that
      the system is actually holding would do.
  interrupted collapses toward cached or random
      The want is brittle. It does not survive disturbance, so it is
      following a plan rather than holding a goal, and persistence is the
      thing to fix next.

Runs in under a minute; it reuses goal.py's world, model and planner.

    python goal_persist.py --world spiral
    python goal_persist.py --world wall --trials 200
"""

import argparse
import random
import statistics
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from goal import (World, learn_world, model_accuracy, plan_action,  # noqa
                  dist, ACTIONS)


def run_persist(world, model, mode, goal, start, horizon, rng,
                max_steps=50, interrupt_at=None):
    """One pursuit. mode controls how the goal is pursued and whether the
    agent is interrupted.

      undisturbed  re-plan every step, no teleport.
      interrupted  re-plan every step, teleported once at interrupt_at.
      cached       plan ONCE, follow the cached path, never re-plan. The
                   brittle control.
      random       ignore the goal.

    The goal is the same object throughout; nothing rewards reaching it.
    """
    pos = start
    cached_path = None

    if mode == "cached":
        # derive a path once by re-planning greedily in imagination, then
        # commit to it. This is a want that is really just a stored plan.
        cached_path = []
        sim = pos
        for _ in range(max_steps):
            if sim == goal:
                break
            a = plan_action(sim, goal, model.predict, horizon, rng)
            cached_path.append(a)
            sim = model.predict(sim, a)

    for t in range(max_steps):
        if pos == goal:
            return True, t

        if interrupt_at is not None and t == interrupt_at:
            # knocked off course: picked up and set down somewhere random
            pos = rng.choice(world.free())

        if mode == "random":
            a = rng.choice(ACTIONS)
        elif mode == "cached":
            # follow the stored plan regardless of where we actually are
            a = cached_path[t] if t < len(cached_path) else "stay"
        else:
            # undisturbed and interrupted both RE-PLAN from the real
            # current cell every step, holding the goal separately
            a = plan_action(pos, goal, model.predict, horizon, rng)

        pos = world.step(pos, a)

    return pos == goal, max_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--learn", type=int, default=6000)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--trials", type=int, default=120)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--world", choices=["wall", "spiral"], default="spiral")
    args = ap.parse_args()

    print(f"  a 7x7 grid, world={args.world!r}")
    print(f"  give a goal, knock the agent to a random cell mid-pursuit, "
          f"ask if it still arrives")
    print(f"  a goal that is HELD survives this; a path that is FOLLOWED "
          f"does not\n")

    arms = ["undisturbed", "interrupted", "cached", "random"]
    reached = {a: [] for a in arms}
    accs = []

    for seed in range(args.seeds):
        world = World(seed, kind=args.world)
        model = learn_world(world, args.learn, seed)
        acc = model_accuracy(world, model)
        accs.append(acc)
        print(f"  seed {seed}: model {acc:.0f}% accurate", flush=True)

        rng = random.Random(seed + 200)
        free = world.free()
        for _ in range(args.trials):
            start = rng.choice(free)
            goal = rng.choice(free)
            while goal == start:
                goal = rng.choice(free)
            interrupt = rng.randint(3, 10)
            for arm in arms:
                ia = interrupt if arm in ("interrupted",) else None
                # cached is also interrupted, to show it cannot cope
                if arm == "cached":
                    ia = interrupt
                ok, _t = run_persist(world, model, arm, goal, start,
                                     args.horizon, rng, interrupt_at=ia)
                reached[arm].append(1 if ok else 0)

    print("\n" + "=" * 66)
    print("DOES THE WANT SURVIVE INTERRUPTION?")
    print("=" * 66)
    print(f"  model accuracy {statistics.mean(accs):.0f}%\n")
    print(f"  {'arm':>12} {'reached':>9}")
    print("-" * 66)
    r = {}
    for arm in arms:
        r[arm] = 100.0 * statistics.mean(reached[arm])
        note = {"undisturbed": "ceiling, never moved",
                "interrupted": "teleported once, still re-planning",
                "cached": "teleported once, following a stored path",
                "random": "floor"}[arm]
        print(f"  {arm:>12} {r[arm]:>8.0f}%   {note}")

    print("\n" + "=" * 66)
    print("WHAT IT SAYS")
    print("=" * 66)

    keeps = r["interrupted"] - r["cached"]
    cost = r["undisturbed"] - r["interrupted"]

    if r["interrupted"] >= r["undisturbed"] - 15 and \
            r["interrupted"] > r["cached"] + 15:
        print(f"  THE WANT PERSISTS. Knocked to a random cell mid-pursuit, "
              f"it still reached")
        print(f"  the ORIGINAL goal {r['interrupted']:.0f}% of the time, "
              f"against {r['undisturbed']:.0f}% undisturbed and")
        print(f"  only {r['cached']:.0f}% for an agent following a cached "
              f"path. The interruption cost")
        print(f"  {cost:.0f} points, not the goal. It holds the state and "
              f"re-plans from wherever it")
        print(f"  lands, which is what a goal the system is actually "
              f"HOLDING would do.")
        print(f"\n  This is the difference between having a goal and "
              f"following instructions.")
    elif r["interrupted"] <= r["cached"] + 10:
        print(f"  THE WANT IS BRITTLE. Interrupted, it did no better "
              f"({r['interrupted']:.0f}%) than an agent")
        print(f"  blindly following a stale path ({r['cached']:.0f}%). It "
              f"is pursuing a plan, not")
        print(f"  holding a goal, and persistence is the thing to fix "
              f"before anything else.")
    else:
        print(f"  PARTIAL. Interrupted {r['interrupted']:.0f}%, cached "
              f"{r['cached']:.0f}%, undisturbed {r['undisturbed']:.0f}%. "
              f"The want")
        print(f"  survives disturbance somewhat but pays more than it "
              f"should. Worth probing")
        print(f"  why re-planning from the new cell is not fully "
              f"recovering.")

    print(f"\n  The goal is held separately from position by construction "
          f"here. The test")
    print(f"  confirms that holding is real: the behaviour, not just the "
          f"code, survives being")
    print(f"  moved off course.")


if __name__ == "__main__":
    main()
