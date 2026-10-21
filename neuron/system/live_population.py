"""Many beings, one world: a living population that lives and dies.

Not one creature, but several, each with a real learning brain, each wanting,
feeling, and each needing to keep itself alive. Energy drains with time;
nourishment refills it; reach zero and the being dies. They share one world
with limited nourishment, so they compete, spread out, and some do not make it.
They perceive each other, and avoid crowding unless hunger forces them to
compete.

This is the species seed, the vision in miniature: a population of smart,
autonomous, mortal beings living together, where life and death are real and
nothing is scripted. We watch who thrives, who struggles, who dies, and whether
a stable group persists.

    python live_population.py --beings 6 --days 300
"""

import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

FEAT = 6


def world_quality(rule, feats):
    s = sum(w * v for w, v in zip(rule, feats))
    return 1.0 if s > 0.6 else (-1.0 if s < -0.6 else 0.0)


class World:
    def __init__(self, size=8, seed=0):
        rng = random.Random(seed)
        self.size = size
        self.rule = [rng.uniform(-1, 1) for _ in range(FEAT)]
        self.feats = {(x, y): [rng.uniform(-1, 1) for _ in range(FEAT)]
                      for x in range(size) for y in range(size)}
        self.food = {}
        for p, f in self.feats.items():
            self.food[p] = 1.0 if world_quality(self.rule, f) > 0 else 0.0
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

    def __init__(self, world, seed=0):
        self.id = Being._next; Being._next += 1
        self.brain = Brain(seed=seed)
        self.opt = torch.optim.Adam(self.brain.parameters(), lr=3e-3)
        self.energy = 1.0
        r = random.Random(seed)
        self.pos = (r.randrange(world.size), r.randrange(world.size))
        self.mood = 0.0
        self.age = 0
        self.alive = True

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

    def choose(self, world, others):
        opts = world.neighbors(self.pos)
        best, best_score = self.pos, -1e9
        hungry = self.energy < 0.4
        for pos in opts:
            expected = self.judge(world.feats[pos])
            food = world.food.get(pos, 0.0)
            crowd = sum(1 for o in others if o == pos)
            score = expected * 1.0 + food * 2.0
            if not hungry:
                score -= 0.5 * crowd
            if score > best_score:
                best_score, best = score, pos
        return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beings", type=int, default=6)
    ap.add_argument("--days", type=int, default=300)
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    Being._next = 0
    world = World(size=args.size, seed=args.seed)
    beings = [Being(world, seed=args.seed * 10 + i) for i in range(args.beings)]

    print(f"  {args.beings} smart, mortal beings born into one world. Energy "
          f"drains; food")
    print(f"  refills it; zero means death. They share limited nourishment, so")
    print(f"  they compete. We watch them live and die, nothing scripted.\n")

    deaths = []
    for day in range(args.days):
        world.regrow()
        for b in beings:
            if not b.alive:
                continue
            b.age += 1
            others = [o.pos for o in beings if o.alive and o.id != b.id]
            b.pos = b.choose(world, others)
            got = world.food.get(b.pos, 0.0)
            if got > 0.3:
                b.energy = min(1.0, b.energy + got)
                world.food[b.pos] = 0.0
                q = 1.0
            else:
                q = world_quality(world.rule, world.feats[b.pos])
            b.learn(world.feats[b.pos], q)
            b.mood = max(-1.0, min(1.0, 0.85 * b.mood + 0.3 * q))
            b.energy -= 0.04
            if b.energy <= 0:
                b.alive = False
                deaths.append((b.id, day, b.age))
        alive = [b for b in beings if b.alive]
        if day % max(1, args.days // 6) == 0 or day == args.days - 1:
            avg_e = sum(b.energy for b in alive) / len(alive) if alive else 0
            print(f"  day {day:>3}: {len(alive)} alive | avg energy "
                  f"{avg_e:.2f} | total deaths {len(deaths)}")

    alive = [b for b in beings if b.alive]
    print("\n" + "=" * 60)
    print("THE POPULATION AFTER A SEASON")
    print("=" * 60)
    print(f"  born: {args.beings} | survived: {len(alive)} | died: {len(deaths)}")
    if deaths:
        shown = ', '.join(f'#{i} died day {d} at age {a}' for i, d, a in deaths[:5])
        print(f"  {shown}")
    if alive:
        print(f"  survivors' ages: {sorted((b.age for b in alive), reverse=True)}")
        spots = len(set(b.pos for b in alive))
        print(f"  survivors hold {spots} distinct places "
              f"({'spread out' if spots == len(alive) else 'some clustered'})")

    print("\n" + "=" * 60)
    print("WHAT HAPPENED")
    print("=" * 60)
    if len(alive) >= 2:
        print(f"  A POPULATION LIVED. {len(alive)} of {args.beings} survived a "
              f"whole season by")
        print(f"  learning their world, finding food, and spacing out to avoid "
              f"starving each")
        print(f"  other. {len(deaths)} died, real mortality, real stakes. No "
              f"one told them how")
        print(f"  to live; each kept itself alive, or did not, on its own. The "
              f"species seed:")
        print(f"  many smart, mortal beings living together where life and "
              f"death are real.")
    elif len(alive) == 1:
        print(f"  One being outlasted all others, {len(deaths)} died competing "
              f"for scarce food.")
        print(f"  A harsh season; survival was real and hard-won.")
    else:
        print(f"  The whole population died; this world could not sustain them. "
              f"Mortality is")
        print(f"  real, and this season was too harsh. (Fewer beings or more "
              f"food would let")
        print(f"  a group persist.)")

    print(f"\n  Next (step 3): let survivors reproduce into FAMILIES and the "
          f"groups come")
    print(f"  together, lineages and a society of smart, mortal beings.")


if __name__ == "__main__":
    main()
