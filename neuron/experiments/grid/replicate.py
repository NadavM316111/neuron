"""Do the family's effects survive 100,000 independent runs, or were they luck?

THE HONEST TEST. Most results in this project came from one run, a few seeds,
a world tuned until the effect showed. That is demonstration, not science.
The real question: run the family hundreds of thousands of times, from
independent random seeds, with NO tuning, and ask whether the effects we
claimed actually hold, consistently, across all of them, or whether we just
saw the lucky runs.

The main claim to test, from family3: "those who reached what they sought
were born into better fortune than those who did not, advantage was real and
compounded." If that is real, it should hold across 100,000 families with a
stable, clearly-positive effect. If it was a fluke of one seed, it will wash
out, flip sign across runs, or shrink to nothing.

We report effect sizes WITH their spread across families, and we say plainly
whether each effect survives or not. A weak or inconsistent effect is
reported as weak or inconsistent. No tuning, no cherry-picking.

This is the stripped, fast core of family3, no printing, just the numbers,
so 100,000 families run in a reasonable time.

    python replicate.py --families 100000
    python replicate.py --families 20000 --life 8000
"""

import argparse
import random
import math
import statistics


N_WANTS = 4


def one_family(seed, founders, life, children):
    """Run one hard-world family (the family3 logic, stripped) and return the
    quantities we need to test the claims. No printing."""
    rng = random.Random(seed)

    def live(aptitude, fortune, need):
        achievement = [0.0] * N_WANTS
        dominant = max(range(N_WANTS), key=lambda i: need[i])
        for t in range(life):
            total = sum(need) or 1.0
            for w in range(N_WANTS):
                effort = need[w] / total
                ceiling = aptitude[w]
                luck = rng.gauss(fortune * 0.4, 0.6)
                achievement[w] = max(0.0, min(ceiling,
                    achievement[w] + 0.0010 * effort * ceiling
                    + 0.0004 * luck))
            if rng.random() < 0.002 - fortune * 0.0015:
                hit = rng.randrange(N_WANTS)
                achievement[hit] = max(0.0, achievement[hit] - 0.15)
        reach = achievement[dominant]
        return reach, reach > 0.55, achievement

    def new_being(apt_in=None, fortune_in=0.0):
        aptitude = [max(0.05, min(1.0, 0.5 * (apt_in[i] if apt_in else 0)
                    + rng.uniform(0.0, 0.7))) for i in range(N_WANTS)]
        fortune = max(-0.7, min(0.7, 0.5 * fortune_in
                     + rng.uniform(-0.45, 0.45)))
        need = [rng.uniform(0, 1) for _ in range(N_WANTS)]
        return aptitude, fortune, need

    def legacy(ach):
        return [0.4 + 0.6 * ach[i] for i in range(N_WANTS)]

    records = []     # (fulfilled, fortune, generation, reach)

    # gen 1
    gen1 = []
    for _ in range(founders):
        apt, fort, need = new_being()
        reach, ful, ach = live(apt, fort, need)
        gen1.append((apt, fort, ach, ful))
        records.append((ful, fort, 1, reach))

    # pair by shared dominant want + noise (stripped affinity)
    idx = list(range(founders))
    rng.shuffle(idx)
    pairs1 = [(idx[i], idx[i + 1]) for i in range(0, len(idx) - 1, 2)]

    # gen 2
    gen2 = []
    for (a, b) in pairs1:
        for _ in range(children):
            apt_in = [0.5 * (legacy(gen1[a][2])[i] + legacy(gen1[b][2])[i])
                      for i in range(N_WANTS)]
            fort_in = (gen1[a][1] + gen1[b][1]) / 2
            apt, fort, need = new_being(apt_in, fort_in)
            reach, ful, ach = live(apt, fort, need)
            gen2.append((apt, fort, ach, ful))
            records.append((ful, fort, 2, reach))

    # gen 3
    if len(gen2) >= 2:
        idx2 = list(range(len(gen2)))
        rng.shuffle(idx2)
        pairs2 = [(idx2[i], idx2[i + 1])
                  for i in range(0, len(idx2) - 1, 2)]
        for (a, b) in pairs2:
            for _ in range(children):
                apt_in = [0.5 * (legacy(gen2[a][2])[i]
                          + legacy(gen2[b][2])[i]) for i in range(N_WANTS)]
                fort_in = (gen2[a][1] + gen2[b][1]) / 2
                apt, fort, need = new_being(apt_in, fort_in)
                reach, ful, ach = live(apt, fort, need)
                records.append((ful, fort, 3, reach))

    # per-family stats
    ful_fortunes = [f for (ful, f, g, r) in records if ful]
    unf_fortunes = [f for (ful, f, g, r) in records if not ful]
    n_ful = len(ful_fortunes)
    n_tot = len(records)
    adv = (statistics.mean(ful_fortunes) - statistics.mean(unf_fortunes)
           if ful_fortunes and unf_fortunes else None)
    ful_rate = n_ful / n_tot if n_tot else 0
    # generational fulfillment
    gen_rates = {}
    for g in (1, 2, 3):
        gr = [ful for (ful, f, gg, r) in records if gg == g]
        gen_rates[g] = (statistics.mean(gr) if gr else 0)
    return dict(ful_rate=ful_rate, advantage=adv, gen_rates=gen_rates)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", type=int, default=100000)
    ap.add_argument("--founders", type=int, default=6)
    ap.add_argument("--life", type=int, default=6000)
    ap.add_argument("--children", type=int, default=2)
    args = ap.parse_args()

    import time
    t0 = time.time()
    print(f"  running {args.families:,} independent families from random "
          f"seeds, no tuning.")
    print(f"  testing whether the claimed effects SURVIVE replication.\n")

    ful_rates = []
    advantages = []
    gen1r, gen2r, gen3r = [], [], []
    adv_positive = 0
    adv_count = 0

    for i in range(args.families):
        r = one_family(i, args.founders, args.life, args.children)
        ful_rates.append(r["ful_rate"])
        if r["advantage"] is not None:
            advantages.append(r["advantage"])
            adv_count += 1
            if r["advantage"] > 0:
                adv_positive += 1
        gen1r.append(r["gen_rates"][1])
        gen2r.append(r["gen_rates"][2])
        gen3r.append(r["gen_rates"][3])
        if (i + 1) % 2000 == 0:
            print(f"    {i+1:,} families... ({time.time()-t0:.0f}s)")

    print("\n" + "=" * 68)
    print("DID THE EFFECTS SURVIVE?")
    print("=" * 68)

    fr_m = statistics.mean(ful_rates)
    fr_s = statistics.pstdev(ful_rates)
    print(f"\n  FULFILLMENT RATE: {fr_m:.1%} +/- {fr_s:.1%}")
    print(f"    (on average this fraction reached what they most sought, "
          f"across all families)")

    print(f"\n  THE MAIN CLAIM -- inherited advantage (fulfilled born into "
          f"better fortune):")
    if advantages:
        a_m = statistics.mean(advantages)
        a_s = statistics.pstdev(advantages)
        pct_pos = 100.0 * adv_positive / adv_count
        print(f"    advantage = {a_m:+.3f} fortune +/- {a_s:.3f}")
        print(f"    positive (fulfilled had better fortune) in "
              f"{pct_pos:.1f}% of families")
        if a_m > 0.05 and pct_pos > 80:
            print(f"    -> THE EFFECT SURVIVES. Across {adv_count:,} "
                  f"families, those who reached")
            print(f"       their goal were consistently born into better "
                  f"fortune. Real, replicated.")
        elif a_m > 0.02 and pct_pos > 65:
            print(f"    -> WEAK BUT REAL. The effect is positive on average "
                  f"and usually, but small")
            print(f"       and not universal. An honest, modest finding.")
        else:
            print(f"    -> DID NOT SURVIVE. The effect washes out or flips "
                  f"across families; it was")
            print(f"       not a reliable finding, likely a feature of the "
                  f"runs we happened to see.")

    g1, g2, g3 = (statistics.mean(gen1r), statistics.mean(gen2r),
                  statistics.mean(gen3r))
    print(f"\n  GENERATIONAL PATTERN (fulfillment by generation):")
    print(f"    gen 1: {g1:.1%}   gen 2: {g2:.1%}   gen 3: {g3:.1%}")
    if abs(g2 - g1) > 0.03 or abs(g3 - g1) > 0.03:
        trend = "rises" if g3 > g1 else "falls"
        print(f"    -> a real generational trend: fulfillment {trend} across "
              f"generations,")
        print(f"       because advantage (or disadvantage) compounds down "
              f"the line.")
    else:
        print(f"    -> no strong generational trend survives; generations "
              f"are about equal.")

    print(f"\n  {args.families:,} families in {time.time()-t0:.0f} seconds. "
          f"No tuning, no cherry-picking.")
    print(f"  These are the effects that actually hold, reported with their "
          f"spread. The ones")
    print(f"  that survived are real; any that did not, we now know were "
          f"not.")


if __name__ == "__main__":
    main()
