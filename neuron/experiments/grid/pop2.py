"""A population that can actually differ.

WHAT FAILED IN pop.py. Eight agents, one world, and a death replaced by a
copy of a living agent. The divergence check caught it immediately:

    arm        median life   divergence
    scratch        242          35.42
    clone          209           0.85
    select         200           0.78

Divergence is the mean pairwise distance between the agents' weights. At
0.78 against a fresh-weights baseline of 35, the population had become ONE
THING IN EIGHT BODIES. With eight agents dying every couple of hundred
steps, every one of them is replaced many times over a run, and every
replacement copies somebody already alive, so within a few generations all
eight trace back to a single ancestor. Copying without variation converges.
Always. It is not a bug, it is what copying does.

Both inheriting arms also lost to `scratch`, and `select` ate 2.1 things
against scratch's 10, which is the abstention trap from survive.py showing
up again in a population that had collapsed to one policy.

So the finding from that run stands on its own: COPYING IS NOT
INHERITANCE. A population that only copies loses the thing that made it a
population.

WHAT THIS ADDS, and both halves are needed.

  MUTATION. A child gets its parent's weights plus Gaussian noise. That is
  the variation this architecture had none of. Biology has it for exactly
  this reason: without it, selection has nothing to select between.

  TOURNAMENT SELECTION. Rather than always copying the single healthiest
  agent, sample TOURNAMENT of the living and copy the best of those.
  Copying the champion every time is the fastest possible route to
  convergence, and a tournament keeps weaker lineages in play.

THE SCALE OF MUTATION IS SWEPT RATHER THAN CHOSEN. This net's weights are
initialised at roughly 0.1, so the arms span 1% to 50% of a typical weight.
Too little and the population still collapses; too much and a child is
worse than a random one, which would show up as select-0.05 losing to
scratch. The sweep is the point: one guessed value would prove nothing
either way.

THE ARMS.

  scratch       random weights on every death. No inheritance at all.
  clone         copy a random living agent, no mutation. pop.py's arm,
                kept so the collapse is visible in the same table.
  select-0.000  tournament selection, no mutation. Isolates selection
                from variation: if this collapses too, selection alone is
                not enough and mutation is doing the work.
  select-0.005  1 percent of a typical weight
  select-0.020  20 percent
  select-0.050  50 percent

READ IN THIS ORDER.

  1. DIVERGENCE. If it is still near zero the population still collapsed
     and nothing else in the run is about a population. This is a gate,
     not a metric.
  2. LIFESPAN AGAINST SCRATCH. Inheritance has to beat starting over, or
     carrying anything forward is pointless here.
  3. THE MUTATION CURVE. If lifespan rises and then falls across the
     sweep there is an optimum, which is a real mechanism claim. If it is
     flat, mutation is doing nothing at this scale.
  4. ate AND poisoned. An arm that ate almost nothing learned to abstain
     and starved; its lifespan measures the policy collapsing, not the
     mechanism. survive.py showed a blind glutton beating every learner,
     so this column is read before any score.

STILL NOT A SPECIES. A child is a parent's weights plus noise: there is no
genome, no recombination, and no reproduction the agents control. This is
inheritance of acquired experience with drift, which is not how biology
works and is what this architecture can actually do.

    python pop2.py                    # ~15 minutes
    python pop2.py --steps 30000
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

from survive import (Backend, TARGET, GOOD, EMPTY, BERRY,      # noqa: E402
                     FUNGUS, START_ENERGY, MAX_ENERGY, STEP_COST,
                     FOOD_GAIN, POISON_COST, DELTA, ACTIONS)


SIZE = 12
POP = 8
N_BERRY = 30
N_FUNGUS = 30
REVERSAL_EVERY = 2000
TOURNAMENT = 3

# std of the Gaussian added to a child's weights, as a fraction of a
# typical weight (this net initialises at roughly 0.1).
MUTATIONS = [0.000, 0.005, 0.020, 0.050]
ARMS = ["scratch", "clone"] + [f"select-{m:.3f}" for m in MUTATIONS]


class SharedWorld:
    """One grid, many agents, one supply of food."""

    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.berry_good = True
        self.grid = [[EMPTY] * SIZE for _ in range(SIZE)]
        for _ in range(N_BERRY):
            self._place(BERRY)
        for _ in range(N_FUNGUS):
            self._place(FUNGUS)

    def _place(self, kind):
        for _ in range(120):
            x, y = self.rng.randrange(SIZE), self.rng.randrange(SIZE)
            if self.grid[y][x] == EMPTY:
                self.grid[y][x] = kind
                return

    def reverse(self):
        self.berry_good = not self.berry_good

    def observe(self, x, y):
        patch = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nx, ny = x + dx, y + dy
                patch.append(self.grid[ny][nx]
                             if 0 <= nx < SIZE and 0 <= ny < SIZE else EMPTY)
        return patch

    def step(self, x, y, action):
        dx, dy = DELTA[action]
        nx, ny = x + dx, y + dy
        if not (0 <= nx < SIZE and 0 <= ny < SIZE):
            return x, y, 0, EMPTY, 0.0
        cell = self.grid[ny][nx]
        if cell == EMPTY:
            return nx, ny, 0, EMPTY, 0.0
        self.grid[ny][nx] = EMPTY
        self._place(cell)
        good = (cell == BERRY) == self.berry_good
        if good:
            return nx, ny, GOOD, cell, FOOD_GAIN
        return nx, ny, 2, cell, -POISON_COST


class Agent:
    def __init__(self, seed, world, gen=0):
        self.b = Backend(seed)
        self.x = world.rng.randrange(SIZE)
        self.y = world.rng.randrange(SIZE)
        self.energy = START_ENERGY
        self.age = 0
        self.gen = gen
        self.eaten = 0
        self.poisoned = 0

    def inherit_from(self, other, world, seed, mutation):
        """A child: the parent's weights plus noise, a fresh body.

        The noise is the whole difference from pop.py. Without it every
        replacement is an exact copy, the population converges on one
        ancestor within a few generations, and selection has nothing left
        to select between.
        """
        self.b = Backend(seed)
        state = other.b.snapshot()
        if mutation > 0:
            g = torch.Generator().manual_seed(seed)
            for k in state:
                state[k] = state[k] + torch.normal(
                    0.0, mutation, size=state[k].shape, generator=g)
        self.b.restore(state)
        self.x = world.rng.randrange(SIZE)
        self.y = world.rng.randrange(SIZE)
        self.energy = START_ENERGY
        self.age = 0
        self.gen = other.gen + 1
        self.eaten = 0
        self.poisoned = 0


def divergence(agents):
    """Mean pairwise weight distance. The gate, not a metric."""
    snaps = [a.b.snapshot() for a in agents]
    keys = list(snaps[0].keys())
    total, n = 0.0, 0
    for i in range(len(snaps)):
        for j in range(i + 1, len(snaps)):
            d = 0.0
            for k in keys:
                d += float(torch.norm(snaps[i][k] - snaps[j][k]).item())
            total += d
            n += 1
    return total / max(1, n)


def run(arm, run_seed, steps):
    random.seed(run_seed * 7919 + 3)
    torch.manual_seed(run_seed)
    world = SharedWorld(run_seed)
    agents = [Agent(run_seed * 100 + i, world) for i in range(POP)]

    mutation = 0.0
    if arm.startswith("select-"):
        mutation = float(arm.split("-")[1])

    deaths = []
    next_seed = run_seed * 100 + POP

    for step in range(steps):
        if step and step % REVERSAL_EVERY == 0:
            world.reverse()

        for i, a in enumerate(agents):
            patch = world.observe(a.x, a.y)
            action = a.b.act(patch, "value")
            target_cell = patch[TARGET[action]]
            a.x, a.y, outcome, cell, delta = world.step(a.x, a.y, action)

            a.energy = min(a.energy + delta - STEP_COST, MAX_ENERGY)
            a.age += 1
            if outcome == GOOD:
                a.eaten += 1
            elif delta < 0:
                a.poisoned += 1

            s, _ = a.b.score((patch, action, outcome))
            a.b.feel(target_cell, action, s)
            a.b.update((patch, action, outcome), 1)

            if a.energy <= 0:
                deaths.append((a.gen, a.age, a.eaten, a.poisoned))
                alive = [x for x in agents if x is not a and x.energy > 0]
                next_seed += 1
                if arm == "scratch" or not alive:
                    agents[i] = Agent(next_seed, world)
                else:
                    if arm == "clone":
                        parent = random.choice(alive)
                    else:
                        # Tournament, not the champion. Always copying the
                        # single best is the fastest route to convergence,
                        # which is what pop.py did and what collapsed it.
                        pool = random.sample(
                            alive, min(TOURNAMENT, len(alive)))
                        parent = max(pool, key=lambda x: x.energy)
                    agents[i] = Agent(next_seed, world)
                    agents[i].inherit_from(parent, world, next_seed,
                                           mutation)

    return dict(deaths=deaths,
                divergence=divergence(agents),
                max_gen=max([a.gen for a in agents] +
                            [d[0] for d in deaths] + [0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", default="pop2.json")
    args = ap.parse_args()

    print(f"  {POP} agents in ONE {SIZE}x{SIZE} world, {args.steps:,} "
          f"steps, {args.runs} runs per arm")
    print(f"  tournament of {TOURNAMENT}, mutation swept "
          f"{MUTATIONS} (weights initialise near 0.1)")
    print(f"  pop.py collapsed to divergence 0.78 with no mutation; "
          f"that is the thing to beat\n", flush=True)

    data = {}
    for arm in ARMS:
        t0 = time.time()
        data[arm] = [run(arm, r, args.steps) for r in range(args.runs)]
        d = [x[1] for r in data[arm] for x in r["deaths"]]
        dv = statistics.mean(r["divergence"] for r in data[arm])
        print(f"  {arm:>13}  {len(d):>5} deaths  median "
              f"{statistics.median(d) if d else 0:>6.0f}  "
              f"divergence {dv:>7.2f}  {time.time() - t0:.0f}s",
              flush=True)

    with open(args.out, "w") as f:
        json.dump(dict(args=vars(args), data=data), f, indent=2)

    def lives(arm):
        return [x[1] for r in data[arm] for x in r["deaths"]]

    def med(arm):
        L = lives(arm)
        return statistics.median(L) if L else 0.0

    def dv(arm):
        return statistics.mean(r["divergence"] for r in data[arm])

    def col(arm, idx):
        v = [x[idx] for r in data[arm] for x in r["deaths"]]
        return statistics.mean(v) if v else 0.0

    print("\n" + "=" * 84)
    print("1. THE GATE — did the population stay a population?")
    print("=" * 84)
    print(f"  {'arm':>13} {'divergence':>12} {'vs scratch':>12}")
    print("-" * 84)
    base_dv = dv("scratch")
    for arm in ARMS:
        share = dv(arm) / base_dv if base_dv else 0.0
        print(f"  {arm:>13} {dv(arm):>12.2f} {share:>11.1%}")

    print("\n" + "=" * 84)
    print("2. THE POPULATION")
    print("=" * 84)
    print(f"  {'arm':>13} {'deaths':>8} {'median':>8} {'mean':>8} "
          f"{'ate':>7} {'poisoned':>9} {'deepest gen':>12}")
    print("-" * 84)
    for arm in ARMS:
        L = lives(arm)
        deepest = max(r["max_gen"] for r in data[arm])
        print(f"  {arm:>13} {len(L):>8} {med(arm):>8.0f} "
              f"{statistics.mean(L) if L else 0:>8.0f} "
              f"{col(arm, 2):>7.1f} {col(arm, 3):>9.1f} {deepest:>12}")

    print("\n" + "=" * 84)
    print("CHECKS")
    print("=" * 84)

    survivors = [a for a in ARMS if a.startswith("select-")
                 and dv(a) > base_dv * 0.2]
    if not survivors:
        print(f"  STILL COLLAPSED. No mutation level kept the population "
              f"apart. Either the")
        print(f"  noise is too small at every level tried, or replacement "
          f"is simply too")
        print(f"  frequent for drift to outrun copying. Try larger "
              f"mutations or a smaller")
        print(f"  population, and read nothing else from this run.")
        return

    print(f"  ok: these kept a population: "
          f"{', '.join(survivors)}")

    abstainers = [a for a in survivors if col(a, 2) < 5]
    if abstainers:
        print(f"  WARNING, LEARNED TO ABSTAIN: {', '.join(abstainers)} "
              f"ate fewer than 5 things")
        print(f"  per life against scratch's {col('scratch', 2):.1f}. "
              f"Their lifespans measure the policy")
        print(f"  collapsing, not the mechanism.")

    usable = [a for a in survivors if a not in abstainers]
    if not usable:
        print(f"\n  Nothing left to compare. Every surviving arm "
              f"abstained.")
        return

    print("\n" + "=" * 84)
    print("3. DOES VARIATION BUY ANYTHING?")
    print("=" * 84)
    print(f"  {'scratch (no inheritance)':<28} {med('scratch'):>8.0f}")
    print(f"  {'clone (copy, no mutation)':<28} {med('clone'):>8.0f}")
    for m in MUTATIONS:
        arm = f"select-{m:.3f}"
        print(f"  {'select, mutation ' + f'{m:.3f}':<28} "
              f"{med(arm):>8.0f}   divergence {dv(arm):>7.2f}")

    best = max(usable, key=med)
    gap = med(best) - med("scratch")
    print()
    if gap > 20:
        print(f"  INHERITANCE WITH VARIATION BEATS STARTING OVER: "
              f"{best} at {med(best):.0f}")
        print(f"  against scratch's {med('scratch'):.0f}. pop.py's "
              f"inheriting arms LOST to scratch")
        print(f"  because they had collapsed; keeping the population "
              f"apart is what changed it.")
        curve = [med(f"select-{m:.3f}") for m in MUTATIONS]
        if curve.index(max(curve)) not in (0, len(curve) - 1):
            print(f"\n  AND THERE IS AN OPTIMUM: the mutation curve rises "
                  f"and falls rather than")
            print(f"  running to one end. Too little noise and the "
                  f"population converges; too much")
            print(f"  and a child is worse than its parent.")
    else:
        print(f"  VARIATION DID NOT RESCUE IT: best inheriting arm "
              f"{med(best):.0f} against")
        print(f"  scratch's {med('scratch'):.0f}. The population stayed "
              f"apart and still did not")
        print(f"  benefit from carrying anything forward, which says the "
              f"problem in pop.py was")
        print(f"  not only convergence.")

    print(f"\n  A child is a parent's weights plus noise. No genome, no "
          f"recombination, no")
    print(f"  reproduction the agents control.")
    print(f"\n  written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
