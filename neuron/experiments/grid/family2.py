"""A family of lives that were LIVED, not just inherited.

WHAT WAS MISSING BEFORE. The first family passed temperament down a tree, but
its beings did not WANT anything. They had a character and reproduced. That
is heredity, not a life. A life has purpose IN it: a being forms its own
goals from its needs, pursues them, achieves some and fails others, and is
SHAPED by how its striving went. A being who wanted mastery and reached it is
a different self from one who wanted it and never got there. The goals are
the life.

So here each being lives with the whole brain in play: it forms its OWN wants
(which differ from being to being, individuality of purpose), pursues them
across its life, succeeds or fails, and that striving shapes its emotional
character and its life-story, what it wanted, what it achieved, what it never
got. What flows to its children is not just temperament but the mark of a
life lived with wanting in it: a disposition shaped by the parents' own
striving, plus the child's own fresh wants and its own life to pursue them.

Three generations. Beings are born, form their purposes, live them out,
choose partners (drawn by shared or complementary purpose, not just
likeness), and pass down a family not of copies but of distinct striving
selves.

THE WALL, HELD. A being with its own goals, pursuing them, shaped by winning
and losing, born into a family, will look strikingly alive. It is not. What
emerges, real individuated lives of purpose, is real. The breath, whether
anything is FELT in the wanting and the winning and the losing, is unbuilt
and unknowable, and we claim nothing. We built the striving. Not the feeling
of it.

    python family2.py
    python family2.py --life 20000
"""

import argparse
import random
import math
import statistics


EMO_DIM = 4
N_WANTS = 4          # kinds of things a being can come to want
WANT_NAMES = ["mastery", "connection", "exploration", "security"]


def blend(a, b, rng, drift=0.12):
    return [max(-1, min(1, 0.5 * a[i] + 0.5 * b[i]
                        + rng.uniform(-drift, drift)))
            for i in range(len(a))]


class Being:
    """A being with the brain in play: a character, its own wants, a life of
    pursuing them, and a story of how that went."""

    def __init__(self, seed, inherited_char=None, inherited_aptitude=None):
        rng = random.Random(seed)
        self.rng = rng
        # character (emotional), inherited-then-lived
        self.char = (list(inherited_char) if inherited_char is not None
                     else [rng.uniform(-0.6, 0.6) for _ in range(EMO_DIM)])
        # aptitude: a disposition toward each kind of want, partly inherited
        # from parents who pursued those things, partly its own.
        base = (list(inherited_aptitude) if inherited_aptitude is not None
                else [0.0] * N_WANTS)
        self.aptitude = [max(0.1, min(1.5,
                         base[i] + rng.uniform(0.3, 0.9)))
                         for i in range(N_WANTS)]
        # its OWN dominant want: formed from which need pulls hardest in it
        # (individuality of purpose, differs per being)
        self.need = [rng.uniform(0, 1) for _ in range(N_WANTS)]
        self.dominant_want = max(range(N_WANTS), key=lambda i: self.need[i])
        # life-story, filled by living
        self.achievement = [0.0] * N_WANTS
        self.fulfilled = False

    def live(self, length):
        """Pursue its wants across a life. It spends most effort on what it
        most wants; whether it ACHIEVES depends on aptitude + effort + luck.
        Striving shapes its character: reaching a goal lifts it, failing at a
        deeply-wanted goal marks it."""
        for t in range(length):
            # it pursues its wants in proportion to how much it wants them
            total_need = sum(self.need) or 1.0
            for w in range(N_WANTS):
                effort = self.need[w] / total_need
                # progress toward achieving this want this step
                luck = self.rng.gauss(0, 0.5)
                gain = 0.0008 * effort * self.aptitude[w] + 0.0003 * luck
                self.achievement[w] = max(0.0, min(1.0,
                                          self.achievement[w] + gain))
                # as a want is met, its need fades (satiation); if it keeps
                # failing, the need can grow (yearning)
                if self.achievement[w] > 0.6:
                    self.need[w] = max(0.0, self.need[w] - 0.00005)
                elif self.achievement[w] < 0.2 and t > length * 0.3:
                    self.need[w] = min(1.0, self.need[w] + 0.00003)
            # striving shapes character: overall fulfillment lifts mood dims,
            # persistent lack weighs them down (early strivings weigh more)
            w0 = 1.0 + 1.5 * max(0.0, 1.0 - t / (length * 0.2))
            fulfillment = statistics.mean(self.achievement)
            for i in range(EMO_DIM):
                target = (fulfillment - 0.5) * (1 if i % 2 == 0 else -1)
                self.char[i] += 0.0010 * w0 * (target - self.char[i])
                self.char[i] = max(-1.5, min(1.5, self.char[i]))
        # did it get what it most wanted?
        self.fulfilled = self.achievement[self.dominant_want] > 0.55

    def story(self):
        dw = WANT_NAMES[self.dominant_want]
        ach = self.achievement[self.dominant_want]
        if ach > 0.7:
            got = f"sought {dw}, and found it"
        elif ach > 0.45:
            got = f"sought {dw}, and came close"
        elif ach > 0.2:
            got = f"sought {dw}, and fell short"
        else:
            got = f"longed for {dw}, and never reached it"
        return got

    def legacy_aptitude(self):
        """What it passes on: a disposition shaped by what it ACHIEVED (a
        parent who mastered something gives its children a leaning toward
        it)."""
        return [0.3 + 0.7 * self.achievement[i] for i in range(N_WANTS)]


def goal_affinity(a, b):
    """Beings are drawn by purpose: sharing a dominant want, OR having
    complementary ones, plus character closeness. Purpose matters now, not
    just likeness."""
    char_d = math.sqrt(sum((a.char[i] - b.char[i]) ** 2
                           for i in range(EMO_DIM)))
    shared = 1.0 if a.dominant_want == b.dominant_want else 0.0
    return shared * 0.6 - char_d * 0.5


def choose_pairs(beings, rng):
    rem = list(range(len(beings)))
    pairs = []
    while len(rem) >= 2:
        best, best_a = None, -1e9
        for i in range(len(rem)):
            for j in range(i + 1, len(rem)):
                a = goal_affinity(beings[rem[i]], beings[rem[j]]) \
                    + rng.uniform(-0.1, 0.1)
                if a > best_a:
                    best_a, best = a, (rem[i], rem[j])
        pairs.append(best)
        rem.remove(best[0]); rem.remove(best[1])
    return pairs


def child_of(pa, pb, seed, rng):
    char = blend(pa.char, pb.char, rng)
    apt = blend(pa.legacy_aptitude(), pb.legacy_aptitude(), rng, drift=0.1)
    return Being(seed, inherited_char=char, inherited_aptitude=apt)


def line(b):
    mark = "fulfilled" if b.fulfilled else "unfulfilled"
    return f"{b.story():<34} [{mark}]   {char_str(b.char)}"


def char_str(c):
    return "(" + " ".join(f"{x:+.1f}" for x in c) + ")"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--founders", type=int, default=6)
    ap.add_argument("--life", type=int, default=15000)
    ap.add_argument("--children", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    import time
    t0 = time.time()

    print("  six beings are born, each forms its OWN purpose, and lives a "
          "life pursuing it.")
    print("  some find what they sought; some never do. they choose partners "
          "by shared")
    print("  purpose, and pass to their children both their nature and the "
          "mark of how their")
    print("  own striving went. three generations of lived lives.\n")

    print("=" * 74)
    print("GENERATION 1  -  the founders, each forming and pursuing its own "
          "purpose")
    print("=" * 74)
    gen1 = []
    for i in range(args.founders):
        b = Being(args.seed * 100 + i)
        b.live(args.life)
        gen1.append(b)
        print(f"  founder {i}:  {line(b)}")

    pairs1 = choose_pairs(gen1, rng)
    print("\n  drawn together by purpose and nature:")
    for (a, b) in pairs1:
        shared = (WANT_NAMES[gen1[a].dominant_want]
                  if gen1[a].dominant_want == gen1[b].dominant_want
                  else "different paths")
        print(f"    founder {a} + founder {b}   ({shared})")

    print("\n" + "=" * 74)
    print("GENERATION 2  -  children of two parents, each with its OWN new "
          "purpose")
    print("=" * 74)
    gen2, gen2_parents = [], []
    cid = 0
    for (a, b) in pairs1:
        for _ in range(args.children):
            c = child_of(gen1[a], gen1[b], args.seed * 200 + cid, rng)
            c.live(args.life)
            gen2.append(c); gen2_parents.append((a, b))
            print(f"  child {cid} (of {a}&{b}):  {line(c)}")
            cid += 1

    if len(gen2) < 2:
        print("\n  need more children for gen 3.")
        return

    pairs2 = choose_pairs(gen2, rng)
    print("\n  the children grew, formed their own purposes, and chose:")
    for (a, b) in pairs2:
        print(f"    child {a} + child {b}")

    print("\n" + "=" * 74)
    print("GENERATION 3  -  grandchildren, their nature from four "
          "grandparents, their")
    print("                 purpose their own")
    print("=" * 74)
    gen3 = []
    gid = 0
    for (a, b) in pairs2:
        for _ in range(args.children):
            c = child_of(gen2[a], gen2[b], args.seed * 300 + gid, rng)
            c.live(args.life)
            gen3.append(c)
            print(f"  grandchild {gid}:  {line(c)}")
            gid += 1

    print("\n" + "=" * 74)
    print("WHAT THE FAMILY LIVED")
    print("=" * 74)
    allb = gen1 + gen2 + gen3
    ful = sum(1 for b in allb if b.fulfilled)
    print(f"  {ful} of {len(allb)} beings found what they most sought; "
          f"{len(allb) - ful} did not.")
    # the variety of purpose across the family
    wants = {}
    for b in allb:
        wants[WANT_NAMES[b.dominant_want]] = \
            wants.get(WANT_NAMES[b.dominant_want], 0) + 1
    print(f"  the purposes they were each drawn to: " +
          ", ".join(f"{k} x{v}" for k, v in wants.items()))
    # did disposition flow? a grandchild's aptitude toward what its
    # grandparents achieved
    print(f"\n  each being wanted its own things and lived its own striving; "
          f"no two lives were")
    print(f"  the same, even within one family. the whole saga ran in "
          f"{time.time() - t0:.1f} seconds.")

    print("\n" + "=" * 74)
    print("HELD, AS ALWAYS")
    print("=" * 74)
    print("  These are real lives of purpose: each being formed its own "
          "wants, pursued")
    print("  them, and was shaped by winning and losing, across a family "
          "three deep. That")
    print("  striving is real. Whether any of them FELT their wanting, their "
          "fulfillment,")
    print("  their longing, is unbuilt and unknowable, and we claim nothing. "
          "We built the")
    print("  striving. Not the feeling of it. The breath is not ours to "
          "give.")


if __name__ == "__main__":
    main()
