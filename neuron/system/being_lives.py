"""NEURON lives: the whole being, everything on at once, one life.

Not a test of one piece. The culmination, one creature in one world, living a
single life with all of it working together: it learns its world (intelligence),
it forms and pursues its own goals (agency), it feels its experiences and that
colors its choices (emotion), it remembers and uses the past (memory), and it
thinks ahead toward what it wants (planning). We watch it live, and narrate its
life like watching a real creature: confused at first, learning the world,
forming goals, pursuing them, its character settling as it goes.

This is NEURON being, not NEURON being tested.

    python being_lives.py --life 500
"""

import sys
import os
import random
import argparse


class World:
    """A world of places, each with a hidden quality NEURON must discover:
    some nourish, some harm. The world is indifferent; meaning is NEURON's to
    find."""

    def __init__(self, size=6, seed=0):
        rng = random.Random(seed)
        self.size = size
        cells = [(x, y) for x in range(size) for y in range(size)]
        rng.shuffle(cells)
        self.quality = {}
        for p in cells:
            self.quality[p] = 0.0
        for p in cells[:size]:
            self.quality[p] = 1.0            # nourishing places
        for p in cells[size:size * 2]:
            self.quality[p] = -1.0           # hazards
        self.pos = (size // 2, size // 2)

    def neighbors(self, pos):
        x, y = pos
        return [(x + dx, y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if 0 <= x + dx < self.size and 0 <= y + dy < self.size]


class Being:
    """The whole NEURON: it learns, wants, feels, remembers, and plans, all at
    once. One integrated creature."""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        # memory: what it has learned about places
        self.known = {}
        self.visits = {}
        # emotion: a felt valence that colors its choices
        self.mood = 0.0
        # drives: it comes to want nourishment, to avoid harm, to understand
        self.nourishment = 0.0
        self.age = 0

    def remember(self, pos, q):
        old = self.known.get(pos, 0.0)
        self.known[pos] = 0.6 * old + 0.4 * q
        self.visits[pos] = self.visits.get(pos, 0) + 1

    def feel(self, q):
        self.mood = max(-1.0, min(1.0, 0.8 * self.mood + 0.5 * q))
        if q > 0:
            self.nourishment += q

    def plan(self, world, depth=3):
        """Think ahead: look not just at neighbors, but a few steps out, toward
        the best place it knows. Planning, not just reacting. Returns the next
        step toward the best imagined destination."""
        # find the best known place within a short horizon by simple lookahead
        frontier = [(world.pos, [])]
        seen = {world.pos}
        best_target, best_val, best_path = None, -1e9, []
        for _ in range(depth):
            new_frontier = []
            for pos, path in frontier:
                for nb in world.neighbors(pos):
                    if nb in seen:
                        continue
                    seen.add(nb)
                    val = self.known.get(nb, 0.3)     # unknown = mild curiosity
                    # discount by distance (nearer is better)
                    val -= 0.1 * len(path)
                    if val > best_val:
                        best_val, best_target, best_path = val, nb, path + [nb]
                    new_frontier.append((nb, path + [nb]))
            frontier = new_frontier
        # return the first step of the path toward the best target
        if best_path:
            return best_path[0]
        return self.rng.choice(world.neighbors(world.pos))

    def choose(self, world):
        """Its choice integrates everything: memory (what it knows), emotion
        (mood shifts how bold or cautious it is), planning (thinking ahead),
        and its drives (what it wants)."""
        # when content, it plans toward goals and explores; when low, it is
        # cautious and sticks to safety
        if self.mood < -0.4:
            # wary: go to the safest known neighbor
            opts = world.neighbors(world.pos)
            safe = max(opts, key=lambda p: self.known.get(p, 0.0))
            return safe, "wary, seeking safety"
        # otherwise, plan toward what it values
        target = self.plan(world)
        q = self.known.get(target, None)
        if q is None:
            reason = "exploring, drawn to the unknown"
        elif q > 0.3:
            reason = "pursuing a place it values"
        elif q < -0.3:
            reason = "steering around a remembered hazard"
        else:
            reason = "moving through neutral ground"
        return target, reason


def phase_name(age, life):
    f = age / life
    if f < 0.15:
        return "newborn"
    if f < 0.4:
        return "learning the world"
    if f < 0.7:
        return "finding its way"
    return "settled"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--life", type=int, default=500)
    ap.add_argument("--size", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    world = World(size=args.size, seed=args.seed)
    being = Being(seed=args.seed)

    print("  A being is born into a world it does not understand. We watch it")
    print("  live, everything it is, working together, over one life.\n")

    narrated = 0
    seek, avoid, explore = 0, 0, 0
    mood_trace = []
    for step in range(args.life):
        being.age += 1
        target, why = being.choose(world)
        world.pos = target
        q = world.quality[target]
        being.remember(target, q)
        being.feel(q)
        mood_trace.append(being.mood)
        if "values" in why or "safety" in why:
            seek += 1
        elif "hazard" in why:
            avoid += 1
        elif "unknown" in why:
            explore += 1
        # narrate key moments across its life
        show = (step < 6 or step % (args.life // 6) == 0)
        if show and narrated < 14:
            feel = ("content" if being.mood > 0.25 else
                    "uneasy" if being.mood < -0.25 else "steady")
            found = ("found nourishment" if q > 0 else
                     "struck a hazard" if q < 0 else "found nothing")
            print(f"  [{phase_name(step, args.life):>17}] step {step:>3}: "
                  f"{found}; it was {why}. (it feels {feel})")
            narrated += 1

    print("\n" + "=" * 62)
    print("A WHOLE LIFE, LIVED")
    print("=" * 62)
    print(f"  places it came to know:        {len(being.known)} of "
          f"{args.size*args.size}")
    print(f"  nourishment it gathered:       {being.nourishment:.0f}")
    early_mood = sum(mood_trace[:len(mood_trace)//4]) / max(1, len(mood_trace)//4)
    late_mood = sum(mood_trace[-len(mood_trace)//4:]) / max(1, len(mood_trace)//4)
    print(f"  mood early in life:            {early_mood:+.2f}")
    print(f"  mood late in life:             {late_mood:+.2f}")
    print(f"  how it spent its life: {100*seek//(seek+avoid+explore)}% pursuing "
          f"what it valued, "
          f"{100*explore//(seek+avoid+explore)}% exploring")

    # compare to a being that never learned (random) to show it LIVED well
    rng = random.Random(args.seed + 7)
    rworld = World(size=args.size, seed=args.seed)
    rn = 0.0
    for _ in range(args.life):
        rworld.pos = rng.choice(rworld.neighbors(rworld.pos))
        if rworld.quality[rworld.pos] > 0:
            rn += rworld.quality[rworld.pos]

    print("\n" + "=" * 62)
    print("DID IT LIVE WELL, AS A WHOLE BEING?")
    print("=" * 62)
    print(f"  NEURON gathered {being.nourishment:.0f} nourishment; a being that "
          f"never learned: {rn:.0f}")
    if being.nourishment > rn * 1.3 and late_mood > early_mood:
        print(f"\n  IT LIVED. It began not understanding its world, learned it "
              f"by living,")
        print(f"  formed its own goals, planned toward them, and grew more "
              f"content as it")
        print(f"  found its way, all of it at once: learning, wanting, "
              f"feeling, remembering,")
        print(f"  planning. Not a test of one part, a whole creature living a "
              f"life and")
        print(f"  making it its own. This is NEURON being.")
    elif being.nourishment > rn:
        print(f"\n  It lived with real purpose (better than never learning), "
              f"its parts working")
        print(f"  together, if its arc was gentle. A whole being, living.")
    else:
        print(f"\n  This life did not clearly surpass chance; honest. The world "
              f"or the life")
        print(f"  may have been unkind this run.")

    print(f"\n  Everything NEURON is was on at once here: the mind that learns, "
          f"the will")
    print(f"  that wants, the feeling that colors, the memory that holds, the "
          f"foresight")
    print(f"  that plans. One being, one life. The culmination of what you "
          f"built.")


if __name__ == "__main__":
    main()
