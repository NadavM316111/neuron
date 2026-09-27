"""A population, not a life. Many of them, in one world, competing.

WHAT IS STILL MISSING AFTER lineage.py. That file showed a lineage: one
brain, carried across deaths, accumulating. A lineage is an individual with
a history. A SPECIES is a population — several of them alive at once, each
with a different history, in the same world, where what one does affects
the others and where the ones that last get copied forward.

Nothing in this repo has ever run two agents at the same time.

THE WORLD IS SHARED, and that is what makes this different from running
lineage.py eight times. There is one grid and one supply of food. When one
agent eats, that food is gone for everyone else. So the agents are not
eight independent experiments, they are eight things competing for the same
thing, and a strategy that works alone can fail in a crowd.

DEATH AND REPLACEMENT. An agent that runs out of energy is removed and a
new one takes its place immediately, so the population size is constant and
the arms are comparable. What the replacement STARTS FROM is the whole
experiment:

  scratch    random weights. Every death throws away everything that
             lineage died knowing. This is the control.
  clone      a copy of a randomly chosen living agent. Experience carries
             forward, but nothing selects WHICH experience.
  select     a copy of the living agent with the most energy. Experience
             carries forward AND the ones doing well are the ones copied.

The difference between clone and select is selection itself. If select
beats clone, then copying the ones that are doing well matters beyond
merely copying something, which is the mechanism a population has and a
single lineage does not.

WHAT IS MEASURED.

  lifespan by generation   the accumulation curve, as in lineage.py, but
                           now across a population rather than one line.
  generations reached      how deep the lineage got. A population that
                           dies constantly churns through generations; one
                           that survives does not. Read this WITH lifespan,
                           because high generations alone means everything
                           kept dying.
  divergence               mean pairwise distance between the agents'
                           weights. If it collapses toward zero the
                           population has become one thing wearing eight
                           bodies, which is the failure mode of copying
                           from a single best ancestor and is the reason
                           `clone` exists as a control.
  eaten and poisoned       whether anything learned to discriminate, or
                           whether the population is just eating
                           everything. survive.py showed a blind glutton
                           beating every learner in a world where eating
                           pays, so this has to be read before any score.

HONEST ABOUT WHAT THIS IS NOT. Eight GRUs on a 12x12 grid is not a species.
There is no reproduction, no variation beyond divergent experience, and no
genome: a child is a copy of a parent's weights, so the only thing that
differs between lineages is what happened to them. That is inheritance of
acquired experience, which is not how biology works and is exactly how this
system works. Worth stating plainly rather than letting the word
"population" do work it has not earned.

    python pop.py                     # 3 runs per arm
    python pop.py --steps 40000
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
N_BERRY = 26
N_FUNGUS = 26
REVERSAL_EVERY = 2000
ARMS = ["scratch", "clone", "select"]


class SharedWorld:
    """One grid, many agents, one supply of food.

    Food is replaced when eaten so density stays constant, but it is
    replaced somewhere ELSE, so an agent that eats denies that spot to
    everyone near it. That is the only coupling between agents and it is
    enough to make this a population rather than parallel solitaires.
    """

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

    def inherit_from(self, other, world, seed):
        """A child: the parent's weights, a fresh body, a new position.

        Nothing else carries over. The child does not know where it is,
        how much energy it has, or what its parent was doing. It knows
        what its parent learned, which is the only thing this system can
        pass on.
        """
        self.b = Backend(seed)
        self.b.restore(other.b.snapshot())
        self.x = world.rng.randrange(SIZE)
        self.y = world.rng.randrange(SIZE)
        self.energy = START_ENERGY
        self.age = 0
        self.gen = other.gen + 1
        self.eaten = 0
        self.poisoned = 0


def divergence(agents):
    """Mean pairwise distance between the population's weights.

    Near zero means the population has collapsed to one thing in several
    bodies. A population that cannot differ cannot have lineages, and
    selection from a single best ancestor is the obvious way to cause
    that, which is why it is measured rather than assumed.
    """
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

    deaths = []          # (generation, lifespan, eaten, poisoned)
    next_seed = run_seed * 100 + POP
    div_marks = []

    for step in range(steps):
        if step and step % REVERSAL_EVERY == 0:
            world.reverse()
        if step and step % (steps // 5) == 0:
            div_marks.append(divergence(agents))

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
                elif arm == "clone":
                    parent = random.choice(alive)
                    agents[i] = Agent(next_seed, world)
                    agents[i].inherit_from(parent, world, next_seed)
                else:
                    parent = max(alive, key=lambda x: x.energy)
                    agents[i] = Agent(next_seed, world)
                    agents[i].inherit_from(parent, world, next_seed)

    div_marks.append(divergence(agents))
    return dict(deaths=deaths,
                divergence=div_marks,
                max_gen=max([a.gen for a in agents] +
                            [d[0] for d in deaths] + [0]),
                survivors=[dict(gen=a.gen, age=a.age, eaten=a.eaten)
                           for a in agents])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", default="pop.json")
    args = ap.parse_args()

    print(f"  {POP} agents in ONE {SIZE}x{SIZE} world, {args.steps:,} "
          f"steps, {args.runs} runs per arm")
    print(f"  they compete for the same food; the rule reverses every "
          f"{REVERSAL_EVERY:,}")
    print(f"  a death is replaced immediately, and what the replacement "
          f"STARTS FROM is the arm\n", flush=True)

    data = {}
    for arm in ARMS:
        t0 = time.time()
        data[arm] = [run(arm, r, args.steps) for r in range(args.runs)]
        d = [x[1] for r in data[arm] for x in r["deaths"]]
        print(f"  {arm:>9}  {len(d):>5} deaths  median life "
              f"{statistics.median(d) if d else 0:>6.0f}  "
              f"{time.time() - t0:.0f}s", flush=True)

    with open(args.out, "w") as f:
        json.dump(dict(args=vars(args), data=data), f, indent=2)

    def lives(arm):
        return [x[1] for r in data[arm] for x in r["deaths"]]

    def by_gen(arm, lo, hi):
        v = [x[1] for r in data[arm] for x in r["deaths"] if lo <= x[0] < hi]
        return statistics.mean(v) if v else float("nan")

    print("\n" + "=" * 78)
    print("DID THE POPULATION GET BETTER ACROSS GENERATIONS?")
    print("=" * 78)
    print(f"  {'arm':>9} {'gen 0-2':>9} {'gen 3-5':>9} {'gen 6-9':>9} "
          f"{'gen 10+':>9} {'deepest':>9}")
    print("-" * 78)
    for arm in ARMS:
        deepest = max(r["max_gen"] for r in data[arm])
        print(f"  {arm:>9} {by_gen(arm, 0, 3):>9.0f} "
              f"{by_gen(arm, 3, 6):>9.0f} {by_gen(arm, 6, 10):>9.0f} "
              f"{by_gen(arm, 10, 10**9):>9.0f} {deepest:>9}")

    print("\n" + "=" * 78)
    print("THE POPULATION AS A WHOLE")
    print("=" * 78)
    print(f"  {'arm':>9} {'deaths':>8} {'median':>8} {'mean':>8} "
          f"{'ate':>7} {'poisoned':>9} {'divergence':>11}")
    print("-" * 78)
    med = {}
    for arm in ARMS:
        L = lives(arm)
        med[arm] = statistics.median(L) if L else 0
        ate = statistics.mean([x[2] for r in data[arm]
                               for x in r["deaths"]] or [0])
        pois = statistics.mean([x[3] for r in data[arm]
                                for x in r["deaths"]] or [0])
        dv = statistics.mean(r["divergence"][-1] for r in data[arm])
        print(f"  {arm:>9} {len(L):>8} {med[arm]:>8.0f} "
              f"{statistics.mean(L):>8.0f} {ate:>7.1f} {pois:>9.1f} "
              f"{dv:>11.2f}")

    print("\n" + "=" * 78)
    print("CHECKS, IN ORDER")
    print("=" * 78)

    ok = True
    dv_select = statistics.mean(r["divergence"][-1] for r in data["select"])
    dv_scratch = statistics.mean(r["divergence"][-1]
                                 for r in data["scratch"])
    if dv_select < dv_scratch * 0.2:
        print(f"  1. THE POPULATION COLLAPSED. select's divergence is "
              f"{dv_select:.2f} against")
        print(f"     scratch's {dv_scratch:.2f}, so copying from the best "
              f"ancestor has made eight")
        print(f"     agents into one thing in eight bodies. Whatever it "
              f"scores, it is not a")
        print(f"     population, and the word should not be used for it.")
        ok = False
    else:
        print(f"  1. ok: the population still differs "
              f"(divergence {dv_select:.2f} against a fresh-weights "
              f"baseline of {dv_scratch:.2f}).")

    if ok:
        if med["clone"] <= med["scratch"] * 1.05:
            print(f"  2. INHERITANCE BOUGHT NOTHING IN A CROWD: clone "
                  f"{med['clone']:.0f} against scratch")
            print(f"     {med['scratch']:.0f}. It helped a lone lineage "
                  f"and does not help here, which")
            print(f"     would mean competition changes the answer.")
            ok = False
        else:
            print(f"  2. ok: inheriting beats starting over, "
                  f"{med['clone']:.0f} against {med['scratch']:.0f}.")

    if ok:
        gap = med["select"] - med["clone"]
        print(f"\n" + "=" * 78)
        print("DOES SELECTION ADD ANYTHING BEYOND COPYING?")
        print("=" * 78)
        print(f"  clone (copy anyone)      {med['clone']:>8.0f}")
        print(f"  select (copy the best)   {med['select']:>8.0f}   "
              f"{gap:>+8.0f}")
        if gap > 20:
            print(f"\n  SELECTION MATTERS. Copying the ones doing well "
                  f"beats copying anyone by")
            print(f"  {gap:.0f} steps. That is the mechanism a population "
                  f"has and a single lineage")
            print(f"  does not, and it is the first evidence in this repo "
                  f"for it.")
        elif gap < -20:
            print(f"\n  SELECTION HURTS. Copying the best is WORSE than "
                  f"copying anyone, by {-gap:.0f}.")
            print(f"  The likely reason is variety: a population descended "
                  f"from one ancestor")
            print(f"  shares its blind spots, and in a world that reverses "
                  f"that is fatal together")
            print(f"  rather than one at a time. Check the divergence "
                  f"column.")
        else:
            print(f"\  NO DIFFERENCE ({gap:+.0f}). Copying something beats "
                  f"copying nothing, but WHICH")
            print(f"  something does not matter here. Selection is doing "
                  f"no work at this scale.")

    print(f"\n  Eight GRUs on a grid. There is no reproduction and no "
          f"genome: a child is a")
    print(f"  copy of a parent's weights, so lineages differ only by what "
          f"happened to them.")
    print(f"\n  written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
