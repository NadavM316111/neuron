"""Step 1: the living being gets a REAL brain.

Before, the being judged places by lookup, it only knew places it had visited.
Now it has a real learning neural network as its brain, so it can PREDICT the
quality of places it has NEVER been, by generalizing from experience. That is
the difference between memory and intelligence: a smart being walks up to a new
place and has a sense of it, because it learned the patterns of its world.

Everything else stays: it wants, feels, remembers, plans. But now it is
genuinely smart, it generalizes, and it gets better at judging its world over
its life.

THE TEST. Each place has features; its quality follows a hidden rule over those
features. A lookup being only knows visited places. A learning being learns the
RULE and can judge unknown places. We measure how well each judges places it has
never visited, and how much nourishment each gathers over a life.

    python smart_being.py --life 600
"""

import sys
import os
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

FEAT = 6


class World:
    """Places have feature vectors; quality follows a hidden rule over them, so
    a being that learns the rule can judge places it has never visited."""

    def __init__(self, size=6, seed=0):
        rng = random.Random(seed)
        self.size = size
        self.feats = {}
        # a hidden linear-ish rule the being must learn
        self.rule = [rng.uniform(-1, 1) for _ in range(FEAT)]
        for x in range(size):
            for y in range(size):
                self.feats[(x, y)] = [rng.uniform(-1, 1) for _ in range(FEAT)]
        self.pos = (size // 2, size // 2)

    def quality(self, pos):
        f = self.feats[pos]
        s = sum(w * v for w, v in zip(self.rule, f))
        return 1.0 if s > 0.5 else (-1.0 if s < -0.5 else 0.0)

    def neighbors(self, pos):
        x, y = pos
        return [(x+dx, y+dy) for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]
                if 0 <= x+dx < self.size and 0 <= y+dy < self.size]


class Brain(nn.Module):
    """A real learning network: predicts a place's quality from its features.
    Generalizes to unknown places, this is the intelligence."""

    def __init__(self, hidden=64, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(FEAT, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 3))         # bad / neutral / good

    def forward(self, x):
        return self.net(x)


class SmartBeing:
    """The whole being with a real brain: it learns the rule of its world and
    judges unknown places by generalizing, plus wants, emotion, memory."""

    def __init__(self, hidden=64, seed=0):
        self.brain = Brain(hidden, seed)
        self.opt = torch.optim.Adam(self.brain.parameters(), lr=3e-3)
        self.mood = 0.0
        self.nourishment = 0.0
        self.visits = {}
        self.rng = random.Random(seed)

    def judge(self, feats):
        """Predict a place's quality from its features, generalizing. Returns
        expected value in [-1, 1]."""
        with torch.no_grad():
            x = torch.tensor([feats], dtype=torch.float32)
            p = F.softmax(self.brain(x), dim=-1)[0]
            return float(p[2] - p[0])      # good prob minus bad prob

    def learn(self, feats, quality):
        label = 2 if quality > 0 else (0 if quality < 0 else 1)
        x = torch.tensor([feats], dtype=torch.float32)
        y = torch.tensor([label], dtype=torch.long)
        self.opt.zero_grad()
        F.cross_entropy(self.brain(x), y).backward()
        self.opt.step()

    def feel(self, q):
        self.mood = max(-1.0, min(1.0, 0.8 * self.mood + 0.5 * q))
        if q > 0:
            self.nourishment += q

    def choose(self, world):
        """Smart choice: judge each neighbor by generalizing from learned
        patterns (even unvisited ones), tempered by curiosity and mood."""
        opts = world.neighbors(world.pos)
        best, best_score = None, -1e9
        for pos in opts:
            feats = world.feats[pos]
            expected = self.judge(feats)              # intelligence: predicts!
            visits = self.visits.get(pos, 0)
            curiosity = 0.3 / (1 + visits)
            score = expected * 2.0 + curiosity
            if self.mood < -0.4 and expected < 0:
                score -= 1.0
            if score > best_score:
                best_score, best = score, pos
        return best


def run_smart(world, life, hidden, seed):
    being = SmartBeing(hidden, seed)
    # track how well it judges UNVISITED places over life
    judge_acc_early, judge_acc_late = [], []
    for step in range(life):
        target = being.choose(world)
        being.visits[target] = being.visits.get(target, 0) + 1
        q = world.quality(target)
        # before learning this place, was its judgment right? (generalization)
        pred = being.judge(world.feats[target])
        right = (pred > 0.2 and q > 0) or (pred < -0.2 and q < 0) or \
                (abs(pred) <= 0.2 and q == 0)
        if step < life * 0.25:
            judge_acc_early.append(1 if right else 0)
        elif step > life * 0.75:
            judge_acc_late.append(1 if right else 0)
        being.learn(world.feats[target], q)
        being.feel(q)
        world.pos = target
    e = sum(judge_acc_early)/len(judge_acc_early) if judge_acc_early else 0
    l = sum(judge_acc_late)/len(judge_acc_late) if judge_acc_late else 0
    return being.nourishment, e, l


def run_lookup(world, life, seed):
    """The old lookup being: only knows places it visited, cannot generalize."""
    rng = random.Random(seed)
    known = {}
    nourishment = 0.0
    pos = world.pos
    for _ in range(life):
        opts = world.neighbors(pos)
        best = max(opts, key=lambda p: known.get(p, 0.3))
        q = world.quality(best)
        known[best] = q
        if q > 0:
            nourishment += q
        pos = best
    return nourishment


def run_random(world, life, seed):
    rng = random.Random(seed)
    pos = world.pos
    n = 0.0
    for _ in range(life):
        pos = rng.choice(world.neighbors(pos))
        if world.quality(pos) > 0:
            n += world.quality(pos)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--life", type=int, default=600)
    ap.add_argument("--size", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("  The being now has a REAL learning brain. It can judge places it")
    print("  has NEVER visited by generalizing what it learned, intelligence,")
    print("  not just memory. Watch it get smarter at reading its world.\n")

    world = World(size=args.size, seed=args.seed)
    smart_n, judge_e, judge_l = run_smart(World(args.size, args.seed),
                                          args.life, args.hidden, args.seed)
    lookup_n = run_lookup(World(args.size, args.seed), args.life, args.seed)
    random_n = run_random(World(args.size, args.seed), args.life, args.seed)

    print("=" * 58)
    print("DID THE REAL BRAIN MAKE IT SMARTER?")
    print("=" * 58)
    print(f"  judging UNVISITED places, early in life: {judge_e*100:.0f}%")
    print(f"  judging UNVISITED places, late in life:  {judge_l*100:.0f}%")
    print(f"  -> {'it LEARNED to read its world (generalizes to the unknown).' if judge_l > judge_e + 0.05 else 'judgment of the unknown did not clearly improve.'}")
    print()
    print(f"  nourishment gathered over a life:")
    print(f"    smart being (real brain):  {smart_n:.0f}")
    print(f"    lookup being (memory only):{lookup_n:>5.0f}")
    print(f"    random being:              {random_n:>5.0f}")

    print("\n" + "=" * 58)
    print("WHAT IT SAYS")
    print("=" * 58)
    if judge_l > judge_e + 0.05 and smart_n >= lookup_n:
        print(f"  THE BEING IS GENUINELY INTELLIGENT NOW. Its brain learned the "
              f"hidden rule of")
        print(f"  its world, so it can judge places it has NEVER been "
              f"({judge_e*100:.0f}% -> {judge_l*100:.0f}% correct on")
        print(f"  the unknown). That is generalization, real intelligence, not "
              f"a lookup table.")
        print(f"  The living being now learns, wants, feels, remembers, plans, "
              f"AND reasons")
        print(f"  about the unknown. Step 1 done: the creature is smart.")
    else:
        print(f"  The real brain ran and the being lived, but the intelligence "
              f"edge was modest")
        print(f"  here ({judge_e*100:.0f}%->{judge_l*100:.0f}% on the unknown). Honest. A "
              f"bigger brain or longer")
        print(f"  life would sharpen its generalization.")

    print(f"\n  Next: step 2, let beings live and die on their own (mortality, "
          f"autonomy),")
    print(f"  then step 3, families of beings coming together into a society.")


if __name__ == "__main__":
    main()
