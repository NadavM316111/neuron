"""A real population: not a tidy tree. Some thrive, some die young, some
leave nothing. Lines end; lines flourish. Nothing is promised, including
continuation.

WHY THIS, OVER A NEAT FAMILY. A real species is not everyone pairing and
having two children. Some beings never find anyone drawn to them. Some die
before they ever reproduce. Some thriving lines spread; some lines simply
end, their last member leaving no one. That churn, birth, death,
childlessness, lines rising and vanishing, IS what a population is. A tidy
tree where everyone pairs and breeds is the unreal thing.

So this lets it all EMERGE from the hard world, unrigged:
  - PAIRING is not guaranteed: affinity must clear a threshold; if no one is
    drawn to you enough, you live and die single, no line.
  - DEATH can come early: a life of bad fortune, many setbacks, and little
    reached carries real mortality risk; a being can die before it ever
    pairs, and leave nothing.
  - CHILDREN vary: a thriving, well-fortuned pair may have several; a
    struggling pair few or none; some pairs none at all.
  - SPLITS happen and MATTER: a weak bond may break; a child of a split is
    raised by one parent only, inheriting from one line, not two.
  - LINES END AND FLOURISH: we track which founders' lines die out and which
    spread across the generations.

We decide none of the outcomes. The hard world does.

THE WALL, HELD. Real birth, real death, real childlessness, real lines
ending, real striving and loss, in a real population facing a hard world.
That is real. Whether any being FELT its life, its loss, its ending, is
unbuilt and unknowable, and we claim nothing. We built the world and the
lives in it. Not the feeling of them.

    python population.py
    python population.py --founders 8 --life 10000
"""

import argparse
import random
import math
import statistics


N_WANTS = 4
WANT_NAMES = ["mastery", "connection", "exploration", "security"]
PURSUIT = {"mastery": "honed a craft",
           "connection": "sought to belong",
           "exploration": "pushed into the unknown",
           "security": "built toward safety"}


def blend(a, b, rng, drift=0.12):
    return [max(-1, min(1.5, 0.5 * a[i] + 0.5 * b[i]
                        + rng.uniform(-drift, drift)))
            for i in range(len(a))]


def single_inherit(a, rng, drift=0.14):
    return [max(-1, min(1.5, a[i] + rng.uniform(-drift, drift)))
            for i in range(len(a))]


class Being:
    _next = 0

    def __init__(self, seed, line, inherited_apt=None, fortune_in=0.0):
        rng = random.Random(seed)
        self.rng = rng
        self.id = Being._next; Being._next += 1
        self.line = line                    # which founder line it descends
        base = inherited_apt or [0.0] * N_WANTS
        self.aptitude = [max(0.05, min(1.0, 0.5 * base[i]
                         + rng.uniform(0.0, 0.7))) for i in range(N_WANTS)]
        self.fortune = max(-0.7, min(0.7,
                       0.5 * fortune_in + rng.uniform(-0.45, 0.45)))
        self.need = [rng.uniform(0, 1) for _ in range(N_WANTS)]
        self.dominant = max(range(N_WANTS), key=lambda i: self.need[i])
        self.achievement = [0.0] * N_WANTS
        self.setbacks = 0
        self.died_young = False
        self.partner = None
        self.single = False
        self.children = 0
        self.final_reach = 0.0
        self.fulfilled = False

    def live(self, length):
        # MORTALITY builds with bad fortune and accumulating hardship; a life
        # can end before it is fully lived.
        for t in range(length):
            total = sum(self.need) or 1.0
            for w in range(N_WANTS):
                effort = self.need[w] / total
                ceiling = self.aptitude[w]
                luck = self.rng.gauss(self.fortune * 0.4, 0.6)
                self.achievement[w] = max(0.0, min(ceiling,
                    self.achievement[w] + 0.0010 * effort * ceiling
                    + 0.0004 * luck))
            if self.rng.random() < 0.002 - self.fortune * 0.0015:
                hit = self.rng.randrange(N_WANTS)
                self.achievement[hit] = max(0.0, self.achievement[hit] - 0.15)
                self.setbacks += 1
            # early death risk: worse with bad fortune and little achieved
            frailty = (0.00003
                       + max(0.0, -self.fortune) * 0.00006
                       + max(0.0, 0.3 - statistics.mean(self.achievement))
                       * 0.0001)
            if self.rng.random() < frailty and t < length * 0.9:
                self.died_young = True
                self.final_reach = self.achievement[self.dominant]
                return
        self.final_reach = self.achievement[self.dominant]
        self.fulfilled = self.final_reach > 0.55

    def legacy_apt(self):
        return [0.4 + 0.6 * self.achievement[i] for i in range(N_WANTS)]

    def fertility(self):
        """Expected number of children if paired: more with fulfillment and
        good fortune, fewer when struggling; can be zero."""
        base = 0.4 + 1.6 * self.final_reach + self.fortune
        return max(0.0, base)

    def story(self):
        if self.died_young:
            return (f"{PURSUIT[WANT_NAMES[self.dominant]]}, but died young, "
                    f"its life unfinished")
        dw = PURSUIT[WANT_NAMES[self.dominant]]
        r = self.final_reach
        cap = self.aptitude[self.dominant]
        if r > 0.7:
            end = "and reached it"
        elif r > 0.5:
            end = "and in the end made it"
        elif r > 0.3:
            end = ("but its gift was small, and it fell short" if cap < 0.45
                   else "but the world kept it from it")
        else:
            end = ("but never had the gift, and never got there" if cap < 0.4
                   else "and longed for it to the end, unreached")
        return f"{dw}, {end}"


def affinity(a, b, rng):
    shared = 0.6 if a.dominant == b.dominant else 0.0
    # fortune-similar beings and shared-purpose beings bond more; plus noise
    return shared + 0.3 * (1 - abs(a.fortune - b.fortune)) + \
        rng.uniform(-0.2, 0.2)


PAIR_THRESHOLD = 0.55      # affinity must clear this to pair at all
SPLIT_THRESHOLD = 0.35     # below this, a formed bond may break


def form_pairs(beings, rng):
    """Beings pair only when mutual affinity clears a threshold. Those no one
    is drawn to stay single. Weak bonds may split."""
    idx = [i for i in range(len(beings)) if not beings[i].died_young]
    rng.shuffle(idx)
    pairs = []
    taken = set()
    # rank all possible bonds, take the strongest that clear the threshold
    cand = []
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            aff = affinity(beings[i], beings[j], rng)
            cand.append((aff, i, j))
    cand.sort(reverse=True)
    for aff, i, j in cand:
        if i in taken or j in taken:
            continue
        if aff < PAIR_THRESHOLD:
            continue
        split = aff < SPLIT_THRESHOLD + 0.1 and rng.random() < 0.3
        pairs.append((i, j, aff, split))
        taken.add(i); taken.add(j)
    singles = [i for i in idx if i not in taken]
    return pairs, singles


def reproduce(beings, pairs, rng, seed_base, gen):
    """Each pair has a variable number of children (possibly zero). A split
    pair's child is raised by one parent only (single-line inheritance)."""
    children = []
    for (i, j, aff, split) in pairs:
        pa, pb = beings[i], beings[j]
        # expected children: average of the pair's fertility, Poisson-ish
        exp = (pa.fertility() + pb.fertility()) / 2
        n = 0
        p = exp
        while p > 0 and rng.random() < min(0.9, p):
            n += 1
            p -= 1.0
        for k in range(n):
            if split and rng.random() < 0.5:
                # raised by one parent: inherits one line only
                parent = pa if rng.random() < 0.5 else pb
                apt = single_inherit(parent.legacy_apt(), rng)
                fort = parent.fortune - 0.1      # harder, single-raised
                c = Being(seed_base + len(children), parent.line,
                          inherited_apt=apt, fortune_in=fort)
            else:
                apt = blend(pa.legacy_apt(), pb.legacy_apt(), rng)
                fort = (pa.fortune + pb.fortune) / 2
                c = Being(seed_base + len(children), pa.line,
                          inherited_apt=apt, fortune_in=fort)
            pa.children += 1
            if not split:
                pb.children += 1
            children.append(c)
    return children


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--founders", type=int, default=8)
    ap.add_argument("--life", type=int, default=9000)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    import time
    t0 = time.time()
    Being._next = 0

    print("  a population is born into a hard world. not all will pair. some "
          "die young. some")
    print("  leave many; some leave none. lines will end, and lines will "
          "spread. we decide")
    print("  nothing; the world does.\n")

    def report(title, beings):
        print("=" * 74)
        print(title)
        print("=" * 74)
        for b in beings:
            line = f"  #{b.id} (line {b.line}): {b.story()}"
            print(line)

    # GENERATION 1
    gen1 = [Being(args.seed * 100 + i, line=i) for i in range(args.founders)]
    for b in gen1:
        b.live(args.life)
    report("GENERATION 1  -  the founders", gen1)
    pairs1, singles1 = form_pairs(gen1, rng)
    died1 = [b for b in gen1 if b.died_young]
    print(f"\n  {len(pairs1)} pairs formed, {len(singles1)} stayed single "
          f"(no one drawn enough), {len(died1)} died young.")
    for (i, j, aff, split) in pairs1:
        tag = " (a weak bond, it would not last)" if split else ""
        print(f"    #{gen1[i].id} + #{gen1[j].id}{tag}")

    # GENERATION 2
    gen2 = reproduce(gen1, pairs1, rng, args.seed * 2000, 2)
    for b in gen2:
        b.live(args.life)
    print()
    if gen2:
        report("GENERATION 2  -  their children (those who were born)", gen2)
    else:
        print("  no children were born. the founders' lines all end here.")
        return
    pairs2, singles2 = form_pairs(gen2, rng)
    died2 = [b for b in gen2 if b.died_young]
    print(f"\n  {len(pairs2)} pairs, {len(singles2)} single, {len(died2)} "
          f"died young.")

    # GENERATION 3
    gen3 = reproduce(gen2, pairs2, rng, args.seed * 3000, 3)
    for b in gen3:
        b.live(args.life)
    print()
    if gen3:
        report("GENERATION 3  -  the grandchildren", gen3)
    else:
        print("GENERATION 3 - none were born; the remaining lines end.")

    # ---- which lines ended, which flourished ----
    print("\n" + "=" * 74)
    print("WHICH LINES ENDED, WHICH FLOURISHED")
    print("=" * 74)
    line_counts = {i: 0 for i in range(args.founders)}
    for b in gen2 + gen3:
        line_counts[b.line] = line_counts.get(b.line, 0) + 1
    for i in range(args.founders):
        cnt = line_counts.get(i, 0)
        if cnt == 0:
            fate = "ENDED - left no descendants"
        elif cnt <= 2:
            fate = f"survives, thin ({cnt} descendants)"
        else:
            fate = f"FLOURISHED ({cnt} descendants)"
        print(f"  founder {i}'s line: {fate}")

    allb = gen1 + gen2 + gen3
    died = sum(1 for b in allb if b.died_young)
    ful = sum(1 for b in allb if b.fulfilled)
    childless = sum(1 for b in allb if not b.died_young and b.children == 0)
    print("\n" + "=" * 74)
    print("WHAT THE POPULATION LIVED")
    print("=" * 74)
    print(f"  {len(allb)} beings lived. {died} died young. {ful} reached what "
          f"they most sought.")
    print(f"  {childless} lived full lives but left no children. lines ended "
          f"and lines spread,")
    print(f"  none of it decided by us. the saga ran in "
          f"{time.time()-t0:.1f} seconds.")

    print("\n" + "=" * 74)
    print("HELD, AS ALWAYS")
    print("=" * 74)
    print("  Real birth, real death, real childlessness, real lines ending,")
    print("  real striving and loss, in a population facing a hard world.")
    print("  That is real. Whether any being FELT its life or its ending is")
    print("  unbuilt and unknowable, and we claim nothing. We built the world")
    print("  and the lives. Not the feeling of them. The breath is not ours.")


if __name__ == "__main__":
    main()
