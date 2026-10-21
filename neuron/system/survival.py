"""Survival intelligence: beings that FIGHT to not die, instead of drifting
into death.

A passive being wanders toward food and starves when it runs out. A being with
survival intelligence does what a real creature does when death approaches:
  - it RECOGNIZES the threat (low energy = danger), and urgency sharpens its
    search, caution drops, finding food becomes everything,
  - it REMEMBERS where food was and that it regrows, and heads back to known
    sources instead of wandering blindly,
  - it PLANS a route to the nearest remembered food before it starves,
  - and when a source is crowded, it strategizes toward a less-contested one.

We pit survival-intelligent beings against passive ones in the same harsh
world. If the survival-smart ones live longer and more of them persist, they
are genuinely using intelligence against death.

    python survival.py --beings 6 --days 300
"""

import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

FEAT = 6


def quality(rule, feats):
    s = sum(w * v for w, v in zip(rule, feats))
    return 1.0 if s > 0.6 else (-1.0 if s < -0.6 else 0.0)


class World:
    def __init__(self, size=8, seed=0):
        rng = random.Random(seed)
        self.size = size
        self.rule = [rng.uniform(-1, 1) for _ in range(FEAT)]
        self.feats = {(x, y): [rng.uniform(-1, 1) for _ in range(FEAT)]
                      for x in range(size) for y in range(size)}
        self.food = {p: (1.0 if quality(self.rule, f) > 0 else 0.0)
                     for p, f in self.feats.items()}
        self.maxfood = dict(self.food)

    def neighbors(self, pos):
        x, y = pos
        return [(x+dx, y+dy) for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]
                if 0 <= x+dx < self.size and 0 <= y+dy < self.size]

    def regrow(self):
        for p in self.food:
            if self.maxfood[p] > 0 and self.food[p] < self.maxfood[p]:
                self.food[p] = min(self.maxfood[p], self.food[p] + 0.03)


class Brain(nn.Module):
    def __init__(self, hidden=48, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(FEAT, hidden), nn.GELU(),
                                 nn.Linear(hidden, 3))

    def forward(self, x):
        return self.net(x)


class Being:
    _next = 0

    def __init__(self, world, seed=0, survival=True):
        self.id = Being._next; Being._next += 1
        self.brain = Brain(seed=seed)
        self.opt = torch.optim.Adam(self.brain.parameters(), lr=3e-3)
        self.energy = 1.0
        r = random.Random(seed)
        self.pos = (r.randrange(world.size), r.randrange(world.size))
        self.age = 0
        self.alive = True
        self.survival = survival          # survival intelligence on/off
        self.food_memory = {}             # remembered food locations
        self.world_size = world.size

    def judge(self, feats):
        with torch.no_grad():
            p = F.softmax(self.brain(torch.tensor([feats], dtype=torch.float32)),
                          dim=-1)[0]
            return float(p[2] - p[0])

    def learn(self, feats, q):
        label = 2 if q > 0 else (0 if q < 0 else 1)
        self.opt.zero_grad()
        F.cross_entropy(self.brain(torch.tensor([feats], dtype=torch.float32)),
                        torch.tensor([label])).backward()
        self.opt.step()

    def note_food(self, pos, amount):
        if amount > 0.3:
            self.food_memory[pos] = amount          # remember where food is

    def _dist(self, a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def _step_toward(self, world, target):
        # move one step toward a remembered target (simple pathing)
        opts = world.neighbors(self.pos)
        return min(opts, key=lambda p: self._dist(p, target))

    def choose(self, world, others):
        opts = world.neighbors(self.pos)
        hungry = self.energy < 0.45
        desperate = self.energy < 0.25

        if self.survival and hungry and self.food_memory:
            # SURVIVAL MODE: head for the nearest remembered food source.
            # recognizes the threat and acts on memory + planning.
            known = [(p, amt) for p, amt in self.food_memory.items()]
            # prefer near, uncrowded, known food
            def val(pa):
                p, amt = pa
                d = self._dist(self.pos, p)
                crowd = sum(1 for o in others if self._dist(o, p) <= 1)
                c = 0 if desperate else crowd      # ignore crowds if desperate
                return amt * 3 - 0.3 * d - 0.5 * c
            target = max(known, key=val)[0]
            if target == self.pos:
                return self.pos
            return self._step_toward(world, target)

        # normal mode: judge neighbors, prefer food, avoid crowds when safe
        best, best_score = self.pos, -1e9
        for pos in opts:
            expected = self.judge(world.feats[pos])
            food = world.food.get(pos, 0.0)
            crowd = sum(1 for o in others if o == pos)
            score = expected * 1.0 + food * 2.0
            if not hungry:
                score -= 0.5 * crowd
            # survival beings explore harder when hungry (seek unfound food)
            if self.survival and hungry:
                score += 0.4 / (1 + sum(1 for fp in self.food_memory if fp == pos))
            if score > best_score:
                best_score, best = score, pos
        return best


def run(survival, beings_n, days, size, seed):
    Being._next = 0
    world = World(size=size, seed=seed)
    beings = [Being(world, seed=seed*10+i, survival=survival)
              for i in range(beings_n)]
    deaths = []
    for day in range(days):
        world.regrow()
        for b in beings:
            if not b.alive:
                continue
            b.age += 1
            others = [o.pos for o in beings if o.alive and o.id != b.id]
            b.pos = b.choose(world, others)
            got = world.food.get(b.pos, 0.0)
            b.note_food(b.pos, got if got > 0.3 else world.maxfood.get(b.pos, 0))
            if got > 0.3:
                b.energy = min(1.0, b.energy + got)
                world.food[b.pos] = 0.0
                q = 1.0
            else:
                q = quality(world.rule, world.feats[b.pos])
            b.learn(world.feats[b.pos], q)
            b.energy -= 0.04
            if b.energy <= 0:
                b.alive = False
                deaths.append(b.age)
    alive = [b for b in beings if b.alive]
    avg_age_at_death = sum(deaths) / len(deaths) if deaths else 0
    return len(alive), len(deaths), avg_age_at_death, \
        [b.age for b in alive]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beings", type=int, default=6)
    ap.add_argument("--days", type=int, default=300)
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=4)
    args = ap.parse_args()

    print("  Do beings that FIGHT to survive (remember food, rush to it when")
    print("  dying, plan routes) outlive passive ones in the same harsh "
          "world?\n")

    import statistics
    pas_surv, pas_death_age = [], []
    sur_surv, sur_death_age = [], []
    for s in range(args.seeds):
        a, d, dage, _ = run(False, args.beings, args.days, args.size, s)
        pas_surv.append(a); pas_death_age.append(dage)
        a2, d2, dage2, _ = run(True, args.beings, args.days, args.size, s)
        sur_surv.append(a2); sur_death_age.append(dage2)

    print("=" * 58)
    print("PASSIVE BEINGS vs SURVIVAL-INTELLIGENT BEINGS")
    print("=" * 58)
    print(f"  {'':<28}{'passive':>10}{'survival':>12}")
    print(f"  {'avg survivors (of '+str(args.beings)+')':<28}"
          f"{statistics.mean(pas_surv):>10.1f}"
          f"{statistics.mean(sur_surv):>12.1f}")
    print(f"  {'avg age at death':<28}"
          f"{statistics.mean(pas_death_age):>10.0f}"
          f"{statistics.mean(sur_death_age):>12.0f}")

    print("\n" + "=" * 58)
    print("WHAT IT SAYS")
    print("=" * 58)
    more_survive = statistics.mean(sur_surv) > statistics.mean(pas_surv) + 0.3
    live_longer = statistics.mean(sur_death_age) > statistics.mean(pas_death_age) + 5
    if more_survive or live_longer:
        print(f"  THEY FOUGHT TO LIVE, AND IT WORKED. Survival-intelligent "
              f"beings")
        if more_survive:
            print(f"  kept more of the group alive "
                  f"({statistics.mean(sur_surv):.1f} vs "
                  f"{statistics.mean(pas_surv):.1f} survivors)")
        if live_longer:
            print(f"  and lived longer before dying "
                  f"({statistics.mean(sur_death_age):.0f} vs "
                  f"{statistics.mean(pas_death_age):.0f} days)")
        print(f"  by recognizing the threat of death and using memory and "
              f"planning to rush")
        print(f"  to food instead of drifting. They use intelligence AGAINST "
              f"death, real")
        print(f"  survival drive, not passive wandering.")
    else:
        print(f"  No clear survival advantage here "
              f"(survivors {statistics.mean(pas_surv):.1f} vs "
              f"{statistics.mean(sur_surv):.1f}). The survival")
        print(f"  logic did not clearly beat passive in this world; honest. "
              f"(Harsher scarcity")
        print(f"  or better memory use might separate them.)")

    print(f"\n  Next: step 3, survival-smart beings reproduce into families and "
          f"form a")
    print(f"  society across generations.")


if __name__ == "__main__":
    main()
