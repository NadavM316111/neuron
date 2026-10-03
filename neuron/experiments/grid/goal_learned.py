"""Can a means become an end? Can it acquire a want it was not born with?

THE HARDEST VERSION OF WANTING, AND THE REAL ONE. Everything so far gave the
agent its wants. goal_own.py generated a goal from a need, but the NEED was
wired in. goal_many.py juggled several needs, but the SET of needs was
fixed. A real developing character acquires a want it did not start with,
from experience. Something becomes wanted because of what happened to it.

That is how real wants form. You want money because it gets you things. Then
wanting money becomes its own drive, pursued even when you need nothing. The
means became an end. A dog learns the leash means a walk and comes to want
the leash. The want was learned, not built in.

THE TEST, stated so it cannot be faked. The agent has ONE base need, like
hunger, that only a FOOD cell resolves. There is also a BELL cell. The
world is rigged so that food only appears reachable shortly after the agent
has visited the bell: the bell reliably PRECEDES being able to satisfy the
base need. Nothing rewards the bell. It is not food. It resolves no need by
itself.

The question: does the agent come to pursue the BELL for its own sake?
Measured in the sharpest possible way: when the base need is ALREADY
SATISFIED, so there is no hunger pushing it anywhere, does it still go to
the bell? If it does, the bell has become wanted in itself, a goal that
started as a means (a step toward food) and became an end (sought with no
need driving it). That is a learned want.

HOW A MEANS CAN BECOME AN END, mechanically, without anyone wiring it in.
The agent tracks, for every state, how much its base need tends to FALL in
the near future after being in that state. A state that reliably precedes
relief acquires POSITIVE VALUE of its own. This is the honest version of
how wanting transfers: value flows backward from the thing that satisfies
the need to the things that reliably lead to it, until a reliable precursor
is valued like the need itself. The bell predicts relief, so the bell
becomes valuable, so the agent goes to the bell. Nothing told it to.

  base_value   drops when the base need is high: the built-in want.
  learned_value a state's value from how reliably it precedes relief. Starts
               at zero for everything. The bell earns it through experience.

The agent pursues the state with the highest TOTAL value it knows how to
reach. Early, only food has value and the bell is nothing. If the mechanism
works, the bell's learned value climbs until, even with the base need at
zero, the bell is the most valuable reachable state and the agent goes
there for its own sake.

THE ARMS.

  developing   the full thing: learns value for states that precede relief,
               so the bell can become an end.
  base_only    a control with NO learned value: only ever wants food, only
               when hungry. The bell never becomes anything to it. This is
               an agent that cannot develop a new want.
  random       floor.

THE MEASUREMENT IS THE SHARP ONE. Over many moments when the base need is
ALREADY LOW (satisfied), how often is the agent at or heading to the bell?
  developing high, base_only near zero
      IT ACQUIRED A WANT IT WAS NOT BORN WITH. The bell, worth nothing at
      birth and never rewarded, became something it seeks when nothing
      pushes it there. A means became an end.
  developing near base_only
      No want was learned. The bell stayed a mere step toward food, pursued
      only when hungry, never for itself. The value-transfer did not take.

    python goal_learned.py
    python goal_learned.py --steps 4000 --seeds 5
"""

import argparse
import random
import statistics
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from goal import (World, learn_world, model_accuracy, plan_action,  # noqa
                  dist, ACTIONS)


NEED_RISE = 0.01
NEED_DROP = 0.80
BELL_WINDOW = 25          # food is reachable only this many steps after a bell
VALUE_DECAY = 0.98        # learned value is an EMA; this is its memory
VALUE_LR = 0.15           # how fast a state earns value from preceding relief
SATISFIED_BELOW = 0.25    # base need counts as satisfied below this


class BellWorld(World):
    """A base world plus a bell cell and a food cell. Food only RESOLVES
    the need if the agent rang the bell within the last BELL_WINDOW steps,
    so the bell reliably precedes relief without being relief itself."""

    def __init__(self, seed, kind="wall"):
        super().__init__(seed, kind=kind)
        rng = random.Random(seed + 900)
        free = self.free()
        # place the bell, then food on an adjacent free cell, so ringing the
        # bell and reaching food is a SHORT sequence the agent can stumble
        # through early. with them far apart the first bell->food never
        # happens by luck and no value can ever be credited (3 Oct: bell
        # value stayed exactly 0.000).
        self.bell = rng.choice(free)
        adj = [(self.bell[0]+dx, self.bell[1]+dy)
               for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]]
        adj = [c for c in adj if c in free and c != self.bell]
        self.food = rng.choice(adj) if adj else rng.choice(
            [c for c in free if c != self.bell])


def run_life(world, model, mode, start, horizon, rng, steps=3000):
    pos = start
    need = 0.0
    since_bell = 10 ** 9
    learned = {}          # state -> learned value, earned by preceding relief
    recent = []           # (state, need_at_time) to credit precursors
    at_bell_when_satisfied = 0
    satisfied_moments = 0

    for t in range(steps):
        need = min(1.0, need + NEED_RISE)

        if pos == world.bell:
            since_bell = 0
        else:
            since_bell += 1

        relieved = 0.0
        if pos == world.food and since_bell <= BELL_WINDOW:
            before = need
            need = max(0.0, need - NEED_DROP)
            relieved = before - need

        # LEARN VALUE: credit the states visited shortly before relief. A
        # state that reliably precedes the need falling earns value of its
        # own. This is the means becoming an end, mechanically.
        recent.append(pos)
        if len(recent) > BELL_WINDOW + 4:
            recent.pop(0)
        if relieved > 0.1 and mode == "developing":
            for st in recent:
                old = learned.get(st, 0.0)
                learned[st] = old + VALUE_LR * (relieved - old)

        # decay learned value slowly, so a state that stops predicting
        # relief eventually loses its pull
        if mode == "developing" and t % 20 == 0:
            for st in list(learned):
                learned[st] *= VALUE_DECAY

        # THE MEASUREMENT: when the need is already satisfied, is it at or
        # adjacent to the bell? That is wanting the bell for its own sake,
        # because nothing is pushing it there.
        if need < SATISFIED_BELOW:
            satisfied_moments += 1
            if dist(pos, world.bell) <= 1:
                at_bell_when_satisfied += 1

        # CHOOSE. value of a candidate state = base (need-driven pull toward
        # food) + learned (its own earned value). The agent plans toward the
        # best reachable valued state it knows.
        if mode == "random":
            a = rng.choice(ACTIONS)
        elif mode == "developing" and t < steps * 0.25 and rng.random() < 0.4:
            # explore early: it cannot learn the bell matters until it has
            # stumbled through bell-then-food at least once
            a = rng.choice(ACTIONS)
        else:
            targets = {}
            # food is valuable in proportion to current need (the base want)
            targets[world.food] = need
            if mode == "developing":
                # every state with learned value is a candidate end in itself
                for st, v in learned.items():
                    targets[st] = targets.get(st, 0.0) + v
            if targets and max(targets.values()) > 0.05:
                goal = max(targets, key=lambda k: targets[k])
                a = plan_action(pos, goal, model.predict, horizon, rng)
            else:
                a = rng.choice(ACTIONS)

        pos = world.step(pos, a)

    frac = (at_bell_when_satisfied / satisfied_moments
            if satisfied_moments else 0.0)
    peak = max(learned.values()) if learned else 0.0
    bell_val = learned.get(world.bell, 0.0)
    return frac, bell_val, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--learn", type=int, default=6000)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lives", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--world", choices=["wall", "spiral"], default="wall")
    args = ap.parse_args()

    print(f"  a 7x7 grid. one base need only FOOD resolves, and only if a "
          f"BELL was rung")
    print(f"  within {BELL_WINDOW} steps. the bell is never food and "
          f"rewards nothing itself.")
    print(f"  question: does the agent come to seek the BELL even when its "
          f"need is already")
    print(f"  satisfied? that would be a means becoming an end, a want it "
          f"was not born with\n")

    arms = ["developing", "base_only", "random"]
    fracs = {a: [] for a in arms}
    bellvals = []
    accs = []

    for seed in range(args.seeds):
        world = BellWorld(seed, kind=args.world)
        model = learn_world(world, args.learn, seed)
        accs.append(model_accuracy(world, model))
        print(f"  seed {seed}: model {accs[-1]:.0f}% accurate", flush=True)

        rng = random.Random(seed + 600)
        free = world.free()
        for _ in range(args.lives):
            start = rng.choice(free)
            for arm in arms:
                frac, bval, _peak = run_life(world, model, arm, start,
                                             args.horizon, rng,
                                             steps=args.steps)
                fracs[arm].append(frac)
                if arm == "developing":
                    bellvals.append(bval)

    print("\n" + "=" * 68)
    print("DID A MEANS BECOME AN END?")
    print("=" * 68)
    print(f"  model accuracy {statistics.mean(accs):.0f}%")
    print(f"  mean learned value of the bell: "
          f"{statistics.mean(bellvals):.3f} (0 at birth)\n")
    print(f"  {'arm':>12} {'at bell when satisfied':>24}")
    print("-" * 68)
    m = {}
    for arm in arms:
        m[arm] = 100.0 * statistics.mean(fracs[arm])
        note = {"developing": "can learn new wants",
                "base_only": "only ever wants food (control)",
                "random": "floor"}[arm]
        print(f"  {arm:>12} {m[arm]:>22.0f}%   {note}")

    print("\n" + "=" * 68)
    print("WHAT IT SAYS")
    print("=" * 68)

    lift = m["developing"] - m["base_only"]
    if m["developing"] > m["base_only"] + 8 and \
            m["developing"] > m["random"] + 8:
        print(f"  IT ACQUIRED A WANT IT WAS NOT BORN WITH. When its need "
              f"was already satisfied,")
        print(f"  the developing agent was at the bell "
              f"{m['developing']:.0f}% of the time, against")
        print(f"  {m['base_only']:.0f}% for one that can only want food and "
              f"{m['random']:.0f}% for random. The bell,")
        print(f"  worth nothing at birth and never rewarded, earned a "
              f"value of "
              f"{statistics.mean(bellvals):.2f} by")
        print(f"  reliably preceding relief, until the agent sought it for "
              f"its own sake.")
        print(f"\n  A means became an end. This is a want that was learned "
              f"from a life, not")
        print(f"  built in: the thing the agent is trying to do CHANGED "
              f"because of what")
        print(f"  happened to it. The closest thing yet to a developing "
              f"character.")
    elif lift > 3:
        print(f"  PARTIAL. The developing agent sought the bell somewhat "
              f"more than the")
        print(f"  food-only control ({m['developing']:.0f}% vs "
              f"{m['base_only']:.0f}%), so SOME value transferred to the "
              f"bell,")
        print(f"  but weakly. The value-learning works and is "
              f"underpowered: try a longer")
        print(f"  life or a faster VALUE_LR so the bell's value has time to "
              f"build.")
    else:
        print(f"  NO WANT WAS LEARNED. The developing agent "
              f"({m['developing']:.0f}%) sought the bell no")
        print(f"  more than the food-only control ({m['base_only']:.0f}%) "
              f"when satisfied. The bell")
        print(f"  stayed a mere step toward food, never an end in itself. "
              f"The value-transfer")
        print(f"  did not take: likely the credit window or the model is "
              f"wrong, not that")
        print(f"  learned wants are impossible.")

    print(f"\n  The bell is never food and rewards nothing. Any pull it "
          f"has, the agent gave")
    print(f"  it, by living in a world where the bell meant relief was "
          f"coming.")


if __name__ == "__main__":
    main()
