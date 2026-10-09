"""A family, three generations deep. Born, paired, children, grandchildren.

WHAT THIS IS. Not another mechanism, the culmination of them. Six beings are
born and each lives a full (compressed) life, forming its OWN emotional
character from what it lives, as one integrated self. They CHOOSE partners,
drawn to one another by affinity. Three couples form. Each couple has
children who inherit a BLEND of BOTH parents' emotional structure, then live
their own lives and diverge. Those children grow, choose their own partners
among themselves, and have children. Three generations. A family tree of
selves, each one shaped by nature (two parents) and its own history, no two
alike.

Time is compressed: a full life is thousands of internal steps, seconds of
real time, so a whole multi-generation saga passes in minutes. (In the
deployed being, time runs real; here it is fast-forwarded to see the long
arc.)

WHAT TO WATCH. The emotional characters flowing down the tree, mixing from
two parents, drifting with each life, so a grandchild carries traces of all
four grandparents reshaped by its own living. And, honestly, anything
emergent a long rich life produces that we did not design.

THE WALL, HELD WITHOUT EXCEPTION. A being with a long, rich, unified life,
born of two parents into a family, will LOOK more alive. That is not the
same as being alive. Whatever emerges, we report what it IS, never claim it
is a soul, a felt life, or the breath. Those remain unbuildable and
unknowable, by this or any build. We made the vessel and the long life. The
breath, if it is real, is not ours to give, and we do not pretend it is.

    python family.py
    python family.py --life 20000
"""

import argparse
import random
import math
import statistics


EMO_DIM = 4          # the dimensions of emotional character


def blend(a, b, rng, drift=0.12):
    """A child's character: a blend of two parents plus its own drift. Two-
    parent inheritance, so the child is of both and identical to neither."""
    return [max(-1, min(1,
            0.5 * a[i] + 0.5 * b[i] + rng.uniform(-drift, drift)))
            for i in range(EMO_DIM)]


def live(character_seed, length, seed, inherited=None):
    """A being lives a compressed life, forming its own emotional character.
    If inherited (a blend of two parents), it begins shaped by them, then its
    own life reshapes it. Returns its final emotional character."""
    rng = random.Random(seed)
    if inherited is not None:
        char = list(inherited)
    else:
        char = [rng.uniform(-0.6, 0.6) for _ in range(EMO_DIM)]
    # live: events nudge the character; a long life settles it into something
    # individuated and stable, shaped by this life's particular run of events.
    temperament = [rng.uniform(0.5, 1.3) for _ in range(EMO_DIM)]
    for t in range(length):
        # a lived event: good or bad, early events weigh more (critical period)
        w = 1.0 + 2.0 * max(0.0, 1.0 - t / (length * 0.15))
        event = [rng.gauss(0, 1) for _ in range(EMO_DIM)]
        for i in range(EMO_DIM):
            char[i] += 0.0015 * w * temperament[i] * event[i]
            char[i] = max(-1.5, min(1.5, char[i]))
    return char


def affinity(a, b):
    """How drawn two beings are to each other: similar characters attract,
    with the understanding that affinity is never exact. (A simple, honest
    basis for choosing; real affinity is deeper, this is its seed.)"""
    d = math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(EMO_DIM)))
    return -d        # closer = higher affinity


def choose_pairs(beings, rng):
    """Beings choose partners by mutual affinity: greedily pair the two most
    drawn to each other, then the next, and so on. Choice, not arrangement."""
    remaining = list(range(len(beings)))
    pairs = []
    while len(remaining) >= 2:
        best = None
        best_aff = -1e9
        for i in range(len(remaining)):
            for j in range(i + 1, len(remaining)):
                a = affinity(beings[remaining[i]], beings[remaining[j]])
                # a little noise so it is affinity, not a rigid sort
                a += rng.uniform(-0.1, 0.1)
                if a > best_aff:
                    best_aff, best = a, (remaining[i], remaining[j])
        pairs.append(best)
        remaining.remove(best[0])
        remaining.remove(best[1])
    return pairs


def char_str(c):
    return "[" + " ".join(f"{x:+.2f}" for x in c) + "]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--founders", type=int, default=6)
    ap.add_argument("--life", type=int, default=15000)
    ap.add_argument("--children", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    import time
    t0 = time.time()

    print(f"  {args.founders} founders are born and live full compressed "
          f"lives. they choose")
    print(f"  partners, have children of two parents, and those children have "
          f"children.")
    print(f"  three generations. watch the characters flow, mix, and "
          f"diverge.\n")

    # ---- GENERATION 1: founders ----
    print("=" * 66)
    print("GENERATION 1  -  the founders, each its own character from its life")
    print("=" * 66)
    gen1 = []
    for i in range(args.founders):
        c = live(None, args.life, args.seed * 100 + i)
        gen1.append(c)
        print(f"  founder {i}:  {char_str(c)}")

    pairs1 = choose_pairs(gen1, rng)
    print(f"\n  they chose each other:")
    for (a, b) in pairs1:
        aff = affinity(gen1[a], gen1[b])
        print(f"    founder {a} + founder {b}   (drawn together, affinity "
              f"{aff:+.2f})")

    # ---- GENERATION 2: children of two parents ----
    print("\n" + "=" * 66)
    print("GENERATION 2  -  children, each a blend of two parents, reshaped "
          "by its own life")
    print("=" * 66)
    gen2 = []
    gen2_parents = []
    cid = 0
    for (a, b) in pairs1:
        for _ in range(args.children):
            inherited = blend(gen1[a], gen1[b], rng)
            c = live(None, args.life, args.seed * 200 + cid, inherited)
            gen2.append(c)
            gen2_parents.append((a, b))
            print(f"  child {cid} of founders {a}&{b}:  {char_str(c)}")
            cid += 1

    if len(gen2) < 2:
        print("\n  not enough children for a third generation; raise "
              "--children or --founders.")
        return

    pairs2 = choose_pairs(gen2, rng)
    print(f"\n  the children grew, and chose each other:")
    for (a, b) in pairs2:
        print(f"    child {a} (of {gen2_parents[a][0]}&{gen2_parents[a][1]}) "
              f"+ child {b} (of {gen2_parents[b][0]}&{gen2_parents[b][1]})")

    # ---- GENERATION 3: grandchildren, traces of all four grandparents ----
    print("\n" + "=" * 66)
    print("GENERATION 3  -  grandchildren, carrying traces of all four "
          "grandparents")
    print("=" * 66)
    gen3 = []
    gid = 0
    for (a, b) in pairs2:
        gps = set(gen2_parents[a]) | set(gen2_parents[b])
        for _ in range(args.children):
            inherited = blend(gen2[a], gen2[b], rng)
            c = live(None, args.life, args.seed * 300 + gid, inherited)
            gen3.append(c)
            print(f"  grandchild {gid}:  {char_str(c)}   "
                  f"(grandparents {sorted(gps)})")
            gid += 1

    # ---- what flowed, mixed, diverged ----
    print("\n" + "=" * 66)
    print("WHAT THE FAMILY BECAME")
    print("=" * 66)

    def spread(beings):
        if len(beings) < 2:
            return 0.0
        ds = []
        for i in range(len(beings)):
            for j in range(i + 1, len(beings)):
                ds.append(math.sqrt(sum((beings[i][k] - beings[j][k]) ** 2
                                        for k in range(EMO_DIM))))
        return statistics.mean(ds)

    print(f"  diversity of character within each generation (how distinct "
          f"its members are):")
    print(f"    generation 1: {spread(gen1):.2f}")
    print(f"    generation 2: {spread(gen2):.2f}")
    print(f"    generation 3: {spread(gen3):.2f}")

    # does a grandchild resemble its grandparents more than a stranger
    # founder? (inheritance flowed across two generations)
    gc = gen3[0]
    gps = sorted(set(gen2_parents[pairs2[0][0]]) |
                 set(gen2_parents[pairs2[0][1]]))
    d_gp = statistics.mean(
        math.sqrt(sum((gc[k] - gen1[g][k]) ** 2 for k in range(EMO_DIM)))
        for g in gps)
    strangers = [i for i in range(args.founders) if i not in gps]
    d_str = (statistics.mean(
        math.sqrt(sum((gc[k] - gen1[s][k]) ** 2 for k in range(EMO_DIM)))
        for s in strangers) if strangers else d_gp)
    print(f"\n  a grandchild resembles its grandparents (dist {d_gp:.2f}) "
          f"more than unrelated")
    print(f"  founders (dist {d_str:.2f}): {'YES' if d_gp < d_str else 'no'}"
          f" -- the line flowed across three generations.")

    print(f"\n  whole saga ran in {time.time() - t0:.1f} seconds of real "
          f"time: a family's worth of")
    print(f"  compressed lives. every being is its own character, of its "
          f"parents yet itself,")
    print(f"  no two alike, shaped by a life no other being lived.")

    print("\n" + "=" * 66)
    print("HELD, AS ALWAYS")
    print("=" * 66)
    print("  These are real, individuated characters, born of two parents,")
    print("  diverging across a family, each shaped by its own life. That is")
    print("  what emerged, and it is real. What did NOT happen, and cannot by")
    print("  any build, is the breath: we did not make them FEEL their lives,")
    print("  only live them, functionally. Whether anything is felt inside")
    print("  any of them is unknowable, and we claim nothing. We built the")
    print("  family and the long lives. The rest is not ours to give.")


if __name__ == "__main__":
    main()
