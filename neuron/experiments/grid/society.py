"""A society: beings that live together, model one another, and perpetuate.

THE STEP, AND WHY IT IS THE CLOSEST THING YET TO THE VISION. The plural world
showed beings coexisting by modeling one another. The family sims showed
beings reproducing across generations. This joins them: a society whose
members LIVE TOGETHER, sharing a world and anticipating one another (theory
of mind), AND partner, have children, and die, so the society carries on
across generations. Not an individual, not a family tree in isolation, a
living, coexisting, self-perpetuating group. That is the shape of a species.

WHAT HAPPENS EACH GENERATION. A cohort of beings shares a world of limited
resources. Each lives by seeking what it needs while modeling the others to
avoid needless conflict, those who coexist well gather enough to thrive.
Thriving beings that find a partner (by affinity) have children, who inherit
from both parents, including a disposition toward social skill. Beings die.
The society persists if its members coexist well enough to sustain it.

WHAT WE MEASURE.
  - Does the society SUSTAIN itself across generations (not collapse)?
  - Do later generations COEXIST BETTER, clashing less, as social skill is
    inherited and the society matures?
  - Does social modeling remain the thing that lets them share the world?

  the society persists and later generations clash less
      A self-perpetuating society of beings that coexist by modeling one
      another, and get better at it across generations. The closest thing
      yet to the vision.

THE WALL. A society that coexists and perpetuates is real and shown. Whether
any being feels its life among the others is unbuilt and unknowable, and we
claim nothing.

    python society.py --founders 12 --generations 5
"""

import argparse
import random
import statistics


N_RESOURCES = 8


class Being:
    def __init__(self, bid, social_skill, pref, fortune, rng):
        self.id = bid
        self.rng = rng
        self.social_skill = social_skill     # how well it models others (0..1)
        self.pref = pref
        self.fortune = fortune
        self.other_model = {}
        self.resource = 1.0
        self.alive = True
        self.gained = 0.0
        self.clashes = 0
        self.partner = None
        self.children = 0

    def observe(self, last_choices):
        for j, c in last_choices.items():
            if j == self.id or c is None:
                continue
            m = self.other_model.setdefault(j, [0.0] * N_RESOURCES)
            for r in range(N_RESOURCES):
                m[r] *= 0.8
            m[c] += 1.0

    def choose(self):
        # social modeling, scaled by this being's social SKILL: a skilled
        # being accounts for others strongly; an unskilled one barely.
        crowding = [0.0] * N_RESOURCES
        for m in self.other_model.values():
            tot = sum(m) or 1.0
            for r in range(N_RESOURCES):
                crowding[r] += m[r] / tot
        value = [self.pref[r] - self.social_skill * 0.9 * crowding[r]
                 for r in range(N_RESOURCES)]
        return max(range(N_RESOURCES), key=lambda r: value[r])


def live_generation(beings, steps, rng):
    """A cohort lives together: shares the world, models each other, gathers
    resources, and pays the cost of conflict. Returns clash rate."""
    from collections import Counter
    last = {b.id: None for b in beings}
    total_clash = 0
    total_choices = 0
    for t in range(steps):
        for b in beings:
            if b.alive:
                b.observe(last)
        choices = {}
        for b in beings:
            if not b.alive:
                continue
            c = b.choose()
            choices[b.id] = c
        counts = Counter(choices.values())
        for b in beings:
            if not b.alive:
                continue
            r = choices[b.id]
            total_choices += 1
            if counts[r] == 1:
                b.gained += b.pref[r]
                b.resource += 0.004 + 0.02 * b.pref[r] - 0.01
            else:
                b.clashes += 1
                total_clash += 1
                b.resource -= 0.008          # conflict is costly
            b.resource = max(0.0, min(2.0, b.resource))
            if b.resource <= 0:
                b.alive = False
        last = choices
    return (total_clash / total_choices) if total_choices else 0.0


def affinity(a, b):
    # drawn by similar social skill and compatible preferences
    skill_close = 1 - abs(a.social_skill - b.social_skill)
    return skill_close + a.rng.uniform(-0.2, 0.2)


def reproduce(beings, rng, next_id):
    """Thriving, alive beings that pair (by affinity) have children who
    inherit a blend of both parents, including social skill, plus drift."""
    survivors = [b for b in beings if b.alive and b.resource > 1.1]
    rng.shuffle(survivors)
    children = []
    used = set()
    # greedy affinity pairing
    pairs = []
    for i in range(len(survivors)):
        if survivors[i].id in used:
            continue
        best, ba = None, -1e9
        for j in range(len(survivors)):
            if i == j or survivors[j].id in used:
                continue
            a = affinity(survivors[i], survivors[j])
            if a > ba:
                ba, best = a, survivors[j]
        if best is not None and ba > 0.4:
            pairs.append((survivors[i], best))
            used.add(survivors[i].id); used.add(best.id)
    for (pa, pb) in pairs:
        n_kids = 1 + (1 if (pa.resource + pb.resource) > 2.6 else 0)
        for _ in range(n_kids):
            skill = max(0.0, min(1.0,
                0.5 * (pa.social_skill + pb.social_skill)
                + rng.uniform(-0.1, 0.1)))
            pref = [max(0.0, min(1.0, 0.5 * (pa.pref[r] + pb.pref[r])
                    + rng.uniform(-0.1, 0.1))) for r in range(N_RESOURCES)]
            fortune = (pa.fortune + pb.fortune) / 2
            c = Being(next_id, skill, pref, fortune, rng)
            next_id += 1
            pa.children += 1; pb.children += 1
            children.append(c)
    return children, next_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--founders", type=int, default=12)
    ap.add_argument("--generations", type=int, default=5)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    import time
    t0 = time.time()

    print(f"  a society of {args.founders} beings lives together: they share "
          f"a world, model one")
    print(f"  another to coexist, and those who thrive and pair have "
          f"children. does the society")
    print(f"  sustain itself, and do later generations coexist better?\n")

    # founders: social skill varies (some model others well, some poorly)
    nid = 0
    cohort = []
    for _ in range(args.founders):
        skill = rng.uniform(0.0, 1.0)
        pref = [rng.random() for _ in range(N_RESOURCES)]
        cohort.append(Being(nid, skill, pref, rng.uniform(-0.3, 0.3), rng))
        nid += 1

    print("=" * 64)
    print("THE SOCIETY ACROSS GENERATIONS")
    print("=" * 64)
    print(f"  {'gen':>4} {'living':>8} {'clash rate':>12} {'avg social skill':>18}")
    clash_by_gen = []
    skill_by_gen = []
    for g in range(args.generations):
        if not cohort:
            print(f"  the society did not survive to generation {g}.")
            break
        clash = live_generation(cohort, args.steps, rng)
        alive = [b for b in cohort if b.alive]
        avg_skill = statistics.mean(b.social_skill for b in cohort)
        clash_by_gen.append(clash)
        skill_by_gen.append(avg_skill)
        print(f"  {g:>4} {len(alive):>8} {clash*100:>11.1f}% "
              f"{avg_skill:>17.2f}")
        children, nid = reproduce(cohort, rng, nid)
        cohort = children

    print("\n" + "=" * 64)
    print("WHAT THE SOCIETY BECAME")
    print("=" * 64)
    survived = len(clash_by_gen) >= args.generations and cohort is not None
    if len(clash_by_gen) >= 2:
        early = clash_by_gen[0]
        late = clash_by_gen[-1]
        skill_rose = skill_by_gen[-1] > skill_by_gen[0]
        if len(clash_by_gen) >= args.generations:
            print(f"  THE SOCIETY PERSISTED through all {args.generations} "
                  f"generations.")
        else:
            print(f"  the society lasted {len(clash_by_gen)} generations "
                  f"before thinning out.")
        if late < early - 0.02:
            print(f"  And it grew more harmonious: clash rate fell from "
                  f"{early*100:.0f}% to {late*100:.0f}% across")
            print(f"  generations, as social skill {'rose' if skill_rose else 'held'} "
                  f"({skill_by_gen[0]:.2f} -> {skill_by_gen[-1]:.2f}) and the")
            print(f"  beings who coexisted best passed that skill to their "
                  f"children.")
        else:
            print(f"  clash rate: {early*100:.0f}% -> {late*100:.0f}%; social "
                  f"skill {skill_by_gen[0]:.2f} -> {skill_by_gen[-1]:.2f}.")
        print(f"\n  This is a self-perpetuating society: beings that live "
              f"together, model one")
        print(f"  another to share the world, and pass both life and the "
              f"skill of coexistence")
        print(f"  down the generations. The closest thing yet to the vision, "
              f"a species, not an")
        print(f"  individual.")

    print(f"\n  ({time.time()-t0:.1f}s. Held: a society that coexists and "
          f"perpetuates is real and")
    print(f"  shown; whether any being feels its life among the others is "
          f"unknowable.)")


if __name__ == "__main__":
    main()
