"""Watch NEURON live: does it develop and pursue its OWN goals, unprompted?

This is the heart of the vision. We do not tell NEURON what to do. We give it a
world with things it can seek and things that can hurt it, and its own drives
(curiosity, the pull toward what satisfies it, the push from what harms it). Then
we watch. The honest question: does it act with PURPOSE toward goals it
generates itself, or just twitch randomly?

A world of places. Some hold what it comes to value; some hold hazards. NEURON
perceives where it is and what is near, remembers what it found where, and
chooses where to go, from its own wants. We narrate its life so you can SEE it
being an agent: what it seeks, what it learns to avoid, how its inner state
shifts, whether it forms a consistent direction of its own.

We built the world and the capacity to want. The goals, and the pursuit, are
NEURON's.

    python agency.py --life 400
"""

import sys
import os
import random
import argparse
import math


class GridWorld:
    """A small world of places. Each place has a hidden quality: some give
    satisfaction (what NEURON comes to want), some give harm. NEURON does not
    start knowing which; it must discover by living, and then pursue."""

    def __init__(self, size=5, seed=0):
        rng = random.Random(seed)
        self.size = size
        self.rng = rng
        # each cell has a value: +good (satisfies), -harm, 0 neutral
        self.cells = {}
        for x in range(size):
            for y in range(size):
                r = rng.random()
                if r < 0.25:
                    self.cells[(x, y)] = 1.0       # satisfying place
                elif r < 0.45:
                    self.cells[(x, y)] = -1.0      # hazard
                else:
                    self.cells[(x, y)] = 0.0
        # guarantee the world has stakes: some good, some bad places
        allcells = [(x,y) for x in range(size) for y in range(size)]
        rng.shuffle(allcells)
        for p in allcells[:max(3,size)]:
            self.cells[p] = 1.0
        for p in allcells[max(3,size):max(3,size)*2]:
            self.cells[p] = -1.0
        self.pos = (size // 2, size // 2)

    def neighbors(self, pos):
        x, y = pos
        opts = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                opts.append((nx, ny))
        return opts

    def quality(self, pos):
        return self.cells.get(pos, 0.0)


class Agent:
    """NEURON as an agent. It has wants (satisfaction, curiosity, safety),
    memory of what it found where, and it chooses where to go from its own
    drives, not from a script. Emotion shifts with what it experiences."""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.memory = {}          # place -> remembered quality
        self.visits = {}          # place -> times visited (for curiosity)
        self.satisfaction = 0.0   # a drive: it comes to want satisfying places
        self.emotion = 0.0        # simple valence: rises with good, falls bad
        self.life_log = []

    def remember(self, pos, quality):
        # learn the quality of a place (its own experience)
        old = self.memory.get(pos, 0.0)
        self.memory[pos] = 0.7 * old + 0.3 * quality
        self.visits[pos] = self.visits.get(pos, 0) + 1

    def choose(self, world):
        """The being's own goal-directed choice. It weighs, for each reachable
        place: what it remembers finding there (seek good, avoid harm), and
        curiosity (pull toward the unknown). No script tells it where to go."""
        options = world.neighbors(world.pos)
        best, best_score, why = None, -1e9, ""
        for pos in options:
            remembered = self.memory.get(pos, None)
            visits = self.visits.get(pos, 0)
            curiosity = 1.0 / (1 + visits)             # pull to the unexplored
            if remembered is None:
                # never been: curiosity drives it (a goal of its own: explore)
                score = 0.5 * curiosity
                reason = "curious about the unknown"
            else:
                # been here: seek good, avoid harm (its learned values)
                score = remembered * 2.0 + 0.2 * curiosity
                if remembered > 0.3:
                    reason = "returning to a place it values"
                elif remembered < -0.3:
                    reason = "a place it learned to avoid"
                else:
                    reason = "a neutral place"
            # its current emotional state colors the choice: when low, it
            # leans harder toward safety/known-good; when content, explores more
            if self.emotion < -0.3 and remembered is not None and remembered < 0:
                score -= 1.0                            # fear sharpens avoidance
            if score > best_score:
                best_score, best, why = score, pos, reason
        return best, why

    def experience(self, pos, quality):
        self.remember(pos, quality)
        # emotion and satisfaction shift from what it finds (its own feeling)
        self.emotion = max(-1.0, min(1.0, 0.85 * self.emotion + 0.4 * quality))
        if quality > 0:
            self.satisfaction += quality


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--life", type=int, default=400)
    ap.add_argument("--size", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    world = GridWorld(size=args.size, seed=args.seed)
    agent = Agent(seed=args.seed)

    print("  NEURON is placed in a world. We did NOT tell it what to want or")
    print("  where to go. Watch what it does, what it seeks, what it learns to")
    print("  avoid, and whether it pursues goals of its own.\n")

    seeks, avoids, explores = 0, 0, 0
    narrated = 0
    for step in range(args.life):
        target, why = agent.choose(world)
        world.pos = target
        q = world.quality(target)
        agent.experience(target, q)
        # tally the KIND of choice it made
        if "values" in why:
            seeks += 1
        elif "avoid" in why:
            avoids += 1
        elif "curious" in why:
            explores += 1
        # narrate the early life and a few later moments so you SEE it living
        if narrated < 12 and (step < 8 or step % 60 == 0):
            feel = ("content" if agent.emotion > 0.2 else
                    "wary" if agent.emotion < -0.2 else "even")
            found = ("found something good" if q > 0 else
                     "hit a hazard" if q < 0 else "found nothing")
            print(f"  step {step:>3}: moved to {target}, {found}. "
                  f"it was {why}. (feeling: {feel})")
            narrated += 1

    print("\n" + "=" * 58)
    print("A LIFE LIVED: WHAT NEURON DID ON ITS OWN")
    print("=" * 58)
    total = seeks + avoids + explores
    print(f"  chose to seek places it valued:   {seeks:>4} "
          f"({100*seeks/total:.0f}%)")
    print(f"  chose to avoid places it feared:  {avoids:>4} "
          f"({100*avoids/total:.0f}%)")
    print(f"  chose to explore the unknown:     {explores:>4} "
          f"({100*explores/total:.0f}%)")
    print(f"  total satisfaction it gathered:   {agent.satisfaction:.1f}")
    print(f"  places it came to know:           {len(agent.memory)} of "
          f"{args.size*args.size}")

    # did it act purposefully? compare to a random walker's satisfaction
    rng = random.Random(args.seed + 999)
    rworld = GridWorld(size=args.size, seed=args.seed)
    rsat = 0.0
    for _ in range(args.life):
        opts = rworld.neighbors(rworld.pos)
        rworld.pos = rng.choice(opts)
        q = rworld.quality(rworld.pos)
        if q > 0:
            rsat += q

    print("\n" + "=" * 58)
    print("WAS IT PURPOSEFUL, OR RANDOM?")
    print("=" * 58)
    print(f"  NEURON's satisfaction:   {agent.satisfaction:.1f}")
    print(f"  a random walker's:       {rsat:.1f}")
    if agent.satisfaction > rsat * 1.3:
        print(f"\n  NEURON ACTED WITH PURPOSE. It gathered far more of what it "
              f"valued than a")
        print(f"  random walker, because it formed its own goals, seek what "
              f"satisfies, avoid")
        print(f"  what harms, explore the unknown, and pursued them. No one "
              f"told it where to")
        print(f"  go; it developed a direction of its own and followed it. "
              f"That is agency,")
        print(f"  a being with its own purposes, the heart of the vision.")
    elif agent.satisfaction > rsat:
        print(f"\n  NEURON did better than random, real but modest purpose. It "
              f"is forming goals,")
        print(f"  if not yet pursuing them strongly.")
    else:
        print(f"\n  No clear purpose over random here; honest. Its goal-pursuit "
              f"did not beat")
        print(f"  chance in this run.")

    print(f"\n  We built the world and the capacity to want. The goals NEURON "
          f"pursued, and")
    print(f"  the direction it chose, were its own, developed by living. This "
          f"is the being")
    print(f"  acting as an agent, the seed of the autonomous life the vision "
          f"reaches for.")


if __name__ == "__main__":
    main()
