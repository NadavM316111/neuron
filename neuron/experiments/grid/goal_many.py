"""Can it hold several wants at once without fixating on one?

WHERE THIS SITS. goal_own.py gave the agent ONE internal need and showed it
forms its own goal to manage it, as well as being told the answer. That is
a self-chosen want, but a single want is the easy case: with one need there
is nothing to trade off. Real wanting is MANY needs at once, pulling in
different directions, and the agent choosing which to serve now and
switching as the balance shifts. Hungry and tired at the same time, and you
pick.

AND THIS IS EXACTLY WHERE THE DRIVE DIED. The drive (drive_scale.py, 2 Oct)
collapsed by fixating: given many options it poured everything onto one and
starved the rest. A system of competing wants must NOT do that. The test of
a real want system is not whether it can chase one thing, it is whether it
can hold several and let none of them run away. Anti-collapse is the
property, and it has to be built in rather than hoped for.

THE WORLD. A grid with several RESOURCE TYPES, each resolving a different
need. The agent has one rising need per type. It is told nothing: not where
resources are, not that they matter. It learns where each type is by
stepping on it, and it must notice which need is most urgent, form a goal
for THAT type, and re-decide as urgencies change.

HOW IT CHOOSES, and this is the anti-collapse mechanism. At each step the
agent looks at all its needs, picks the MOST URGENT one it knows how to
resolve, and plans toward the nearest resource of that type. Because
resolving a need drops it, the most-urgent need keeps CHANGING, so attention
is pulled OFF whatever it just served and ONTO whatever is now worst. The
system cannot fixate, because satisfying a want reduces that want's claim on
behaviour. That is the satiation the drive never had, made structural.

THE ARMS.

  juggler    the full thing: several needs, serves the most urgent it can,
             switches as the balance moves.
  fixated    a deliberately broken control: picks ONE need at the start and
             only ever serves that, ignoring the others. This is the
             drive's failure made explicit. Its ignored needs should run
             away.
  told       handed, each step, the nearest resource for its most urgent
             need. The ceiling.
  oblivious  never forms a goal, moves at random. The floor.

THE MEASUREMENT. The WORST need, averaged over the life. Not the average of
all needs, the single highest one, because a system that keeps three needs
at 0.3 is managing and one that keeps two at 0.1 while a third sits at 0.9
is not. Judging by the worst need is what forces juggling rather than
letting the agent look good by acing one and abandoning the others. Lower
is better.

  juggler near told, and far below fixated and oblivious
      IT HELD SEVERAL WANTS AT ONCE. It served whichever was most urgent,
      switched as the balance shifted, and let none run away. The
      anti-collapse property the drive lacked, demonstrated.
  juggler near fixated
      It could not juggle: it is effectively serving one need and letting
      the others climb, which is the drive's collapse in a new form. The
      thing to fix.

    python goal_many.py --world wall
    python goal_many.py --needs 3 --steps 3000
"""

import argparse
import random
import statistics
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from goal import (World, learn_world, model_accuracy, plan_action,  # noqa
                  dist, ACTIONS)


NEED_RISE = 0.004         # per step, per need, each rises on its own
NEED_DROP = 0.70          # stepping on the matching resource drops that need
SWITCH_MARGIN = 0.25      # another need must exceed the committed one by this to steal focus
DONE_BELOW = 0.20         # below this the committed need counts as handled, move on
PER_TYPE = 1              # resources of each type (scarce, so it must plan)


def place(world, n_types, seed):
    """One resource cell per type, disjoint, placed where the agent is not
    told. Returns {type_index: set_of_cells}."""
    rng = random.Random(seed + 700)
    free = rng.sample(world.free(), n_types * PER_TYPE)
    out = {}
    k = 0
    for ti in range(n_types):
        out[ti] = set(free[k:k + PER_TYPE])
        k += PER_TYPE
    return out


def run_life(world, model, resources, n_types, mode, start, horizon, rng,
             steps=2000):
    """One life with n_types competing needs. Returns the mean of the WORST
    need over the life: the single highest need at each step, averaged.
    That is the honest measure, because it punishes abandoning any need."""
    pos = start
    needs = [0.0] * n_types
    _commit = {"need": None}
    believed = {ti: set() for ti in range(n_types)}
    fixated_on = rng.randrange(n_types)       # for the fixated control
    worst_total = 0.0

    for t in range(steps):
        for ti in range(n_types):
            needs[ti] = min(1.0, needs[ti] + NEED_RISE)

        # resolving: if standing on a resource, drop its need and learn it
        for ti, cells in resources.items():
            if pos in cells:
                needs[ti] = max(0.0, needs[ti] - NEED_DROP)
                believed[ti].add(pos)

        worst_total += max(needs)

        if mode == "oblivious":
            a = rng.choice(ACTIONS)

        elif mode == "told":
            # most urgent need, nearest TRUE resource of that type
            ti = max(range(n_types), key=lambda i: needs[i])
            cells = resources[ti]
            if cells:
                goal = min(cells, key=lambda r: dist(pos, r))
                a = plan_action(pos, goal, model.predict, horizon, rng)
            else:
                a = rng.choice(ACTIONS)

        elif mode == "fixated":
            # the drive's failure made explicit: only ever serve one need
            ti = fixated_on
            if believed[ti]:
                goal = min(believed[ti], key=lambda r: dist(pos, r))
                a = plan_action(pos, goal, model.predict, horizon, rng)
            else:
                a = rng.choice(ACTIONS)

        else:   # juggler: serve the most urgent need, WITH COMMITMENT
            # The first version re-picked the most-urgent need every single
            # step. With needs close in urgency the target flipped step to
            # step, so the agent turned toward A, then B, then A, and
            # dithered in place without reaching anything -- worse than a
            # random walk. That is the opposite of the drive's failure: the
            # drive committed too hard and fixated; this committed too
            # little and never finished.
            #
            # The fix is HYSTERESIS. Once it commits to serving a need, it
            # stays on that need until it RESOLVES it (the need drops, which
            # happens on reaching the resource) or until some OTHER need
            # exceeds the committed one by a margin. Finishing the grocery
            # run before switching to laundry, even if laundry crept up a
            # little mid-trip.
            committed = _commit.get("need")
            worst = max(range(n_types), key=lambda i: needs[i])
            if committed is None or not believed.get(committed):
                committed = worst
            elif needs[worst] > needs[committed] + SWITCH_MARGIN:
                committed = worst            # another need clearly overtook
            elif needs[committed] < DONE_BELOW:
                committed = worst            # this need is satisfied, move on
            _commit["need"] = committed

            if believed[committed]:
                goal = min(believed[committed], key=lambda r: dist(pos, r))
                a = plan_action(pos, goal, model.predict, horizon, rng)
            else:
                a = rng.choice(ACTIONS)      # knows no resource yet: explore

        pos = world.step(pos, a)

    return worst_total / steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--learn", type=int, default=6000)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lives", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--needs", type=int, default=3)
    ap.add_argument("--world", choices=["wall", "spiral"], default="wall")
    args = ap.parse_args()

    print(f"  a 7x7 grid, world={args.world!r}, {args.needs} competing "
          f"needs, one resource type each")
    print(f"  every need rises {NEED_RISE}/step on its own; the right "
          f"resource drops its need {NEED_DROP}")
    print(f"  nobody says which need to serve. the score is the WORST need "
          f"over the life,")
    print(f"  averaged, lower is better, so abandoning any need is "
          f"punished\n")

    arms = ["juggler", "told", "fixated", "oblivious"]
    scores = {a: [] for a in arms}
    accs = []

    for seed in range(args.seeds):
        world = World(seed, kind=args.world)
        model = learn_world(world, args.learn, seed)
        accs.append(model_accuracy(world, model))
        resources = place(world, args.needs, seed)
        print(f"  seed {seed}: model {accs[-1]:.0f}% accurate, "
              f"{args.needs} resource types placed", flush=True)

        rng = random.Random(seed + 400)
        free = world.free()
        for _ in range(args.lives):
            start = rng.choice(free)
            for arm in arms:
                sc = run_life(world, model, resources, args.needs, arm,
                              start, args.horizon, rng, steps=args.steps)
                scores[arm].append(sc)

    print("\n" + "=" * 66)
    print("DID IT HOLD SEVERAL WANTS WITHOUT LETTING ANY RUN AWAY?")
    print("=" * 66)
    print(f"  model accuracy {statistics.mean(accs):.0f}%\n")
    print(f"  {'arm':>10} {'worst need':>11}")
    print("-" * 66)
    m = {}
    for arm in arms:
        m[arm] = statistics.mean(scores[arm])
        note = {"juggler": "serves the most urgent, switches",
                "told": "handed the right resource (ceiling)",
                "fixated": "serves only one need (the drive's failure)",
                "oblivious": "no goal ever (floor)"}[arm]
        print(f"  {arm:>10} {m[arm]:>11.3f}   {note}")

    print("\n" + "=" * 66)
    print("WHAT IT SAYS")
    print("=" * 66)

    span = m["oblivious"] - m["told"]
    if span <= 0.02:
        print(f"  INCONCLUSIVE: being told barely beat random. The world "
              f"does not punish")
        print(f"  abandoning needs enough to measure juggling. Raise "
              f"NEED_RISE or --needs.")
        return

    recovered = (m["oblivious"] - m["juggler"]) / span
    beats_fixated = m["fixated"] - m["juggler"]

    if recovered > 0.6 and beats_fixated > 0.03:
        print(f"  IT HELD SEVERAL WANTS AT ONCE. Judged by its WORST need, "
              f"the juggler kept")
        print(f"  it at {m['juggler']:.3f}, against {m['oblivious']:.3f} "
              f"with no goals, {m['fixated']:.3f} serving only")
        print(f"  one need, and {m['told']:.3f} handed the answer. It "
              f"closed {100 * recovered:.0f}% of the gap to")
        print(f"  being told AND beat the fixated agent by "
              f"{beats_fixated:.3f}.")
        print(f"\n  This is the anti-collapse property the drive never "
              f"had. Serving a need")
        print(f"  reduces that need's pull, so attention moves to whatever "
              f"is now most")
        print(f"  urgent, and no single want can take over. Several wants, "
              f"held at once.")
    elif beats_fixated <= 0.03:
        print(f"  IT COULD NOT JUGGLE. The juggler ({m['juggler']:.3f}) did "
              f"no better than the")
        print(f"  fixated agent ({m['fixated']:.3f}) that serves one need "
              f"and ignores the rest.")
        print(f"  That is the drive's collapse in a new form: effectively "
              f"one want is")
        print(f"  running behaviour. The switching logic is the thing to "
              f"fix.")
    else:
        print(f"  PARTIAL. The juggler ({m['juggler']:.3f}) beats fixated "
              f"({m['fixated']:.3f}) and recovers")
        print(f"  {100 * recovered:.0f}% of the gap to told "
              f"({m['told']:.3f}). It juggles, but pays more than being")
        print(f"  told. Likely the belief about where each resource type "
              f"is, learned only")
        print(f"  by landing on it, is the bottleneck.")

    print(f"\n  The needs are internal and rise on their own. Nothing "
          f"rewards resolving")
    print(f"  them. Judging by the WORST need is what makes this a test of "
          f"holding SEVERAL")
    print(f"  wants rather than acing one and abandoning the others.")


if __name__ == "__main__":
    main()
