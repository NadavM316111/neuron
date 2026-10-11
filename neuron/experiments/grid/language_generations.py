"""A language grown across generations: inherited, used, and enlarged, the way
real tongues actually develop.

THE IDEA, AND WHY IT IS TRUE TO NEURON. A language invented once, in one
lifetime, stays small. Real languages grow because they are INHERITED: each
generation learns its parents' tongue, then lives and needs to say more, and
adds to it, passing a richer language down. Over many generations a few calls
become a large, structured vocabulary. This is exactly NEURON's lineage
applied to language: the beings pass their words to their children, who learn
them and build on them.

WHAT HAPPENS EACH GENERATION:
  1. The new generation INHERITS the previous generation's language by
     exposure, learning the existing words for the existing meanings (a child
     learning its parents' tongue).
  2. It then faces MORE meanings than its parents did (the world asks more of
     it), and must invent words for the new ones.
  3. Crucially, it can COMPOUND, build a new word by combining words it
     already knows (food + water -> a word for a meal), the real engine by
     which languages grow from their own material rather than always from
     scratch.
  4. It passes the enlarged language to the next generation.

WHAT WE MEASURE across 10 generations: vocabulary SIZE, how much is inherited
versus freshly coined, and the emergence of COMPOUND words (words built from
existing words), which is the beginning of real productivity.

THE HONEST LIMIT, STATED PLAINLY. This grows a richer, larger proto-language
with compounds. It does NOT produce English-style grammar or recursion, those
appear to need scale beyond these small beings, and we will not claim them.
What it shows is real: a language that grows across generations by inheritance
and accumulation, from the beings' own material, with no outside words.

    python language_generations.py --generations 10
"""

import argparse
import random


PHONEMES = ["ka", "ta", "mi", "no", "shu", "re", "la", "vo", "zi", "de",
            "wa", "thi", "ne", "ru", "so", "ki", "pa", "lo", "mu", "se"]

# the full set of meanings the world can eventually demand words for. early
# generations face only the first few; later ones face more (the world asks
# more of them as the language matures). labels are for US; beings never see
# them.
ALL_MEANINGS = [
    "food", "danger", "other", "gone", "want", "safe",      # gen 0 base
    "water", "young", "follow", "away", "here", "come",     # early additions
    "sky", "dark", "many", "one", "give", "take",           # mid
    "help", "sleep", "far", "near", "good", "bad",          # later
    "mother", "hunt", "cold", "light", "fear", "rest",      # latest
]


def invent_word(rng):
    n = rng.choice([1, 2, 2])
    return tuple(rng.choice(range(len(PHONEMES))) for _ in range(n))


def emerge_base(meanings, rounds, seed):
    """Generation 0 invents words for the base meanings from nothing."""
    rng = random.Random(seed)
    lexicon = {}
    for m in meanings:
        lexicon[m] = invent_word(rng)
    # (compressed: gen 0's emergence is the own_language process; we take its
    # settled lexicon as the starting tongue)
    return lexicon


def erode(lex, seed):
    """Wear down over-long words across a generation, as real languages
    compress their longest, most-used terms. A word longer than 3 units has
    a chance to lose a unit each generation, shortening over time."""
    rng = random.Random(seed)
    worn = {}
    for m, w in lex.items():
        w = list(w)
        # longer words erode more readily
        while len(w) > 3 and rng.random() < 0.4:
            drop = rng.randrange(len(w))
            w.pop(drop)
        worn[m] = tuple(w)
    return worn


def next_generation(parent_lex, new_meanings, seed, compound_prob=0.45):
    """A new generation inherits the parent language, then coins words for new
    meanings, sometimes by COMPOUNDING existing inherited words."""
    rng = random.Random(seed)
    # inherit: the child learns every word its parents had
    lex = dict(parent_lex)
    inherited_count = len(lex)
    new_coined = 0
    new_compound = 0

    for m in new_meanings:
        if m in lex:
            continue
        # COMPOUND from words that ALREADY EXIST in the language, or coin
        # fresh. only words actually present in lex can be combined.
        existing = [k for k in lex.keys()]
        if len(existing) >= 2 and rng.random() < compound_prob:
            a, b = rng.sample(existing, 2)
            word = tuple(lex[a]) + tuple(lex[b])
            lex[m] = word
            new_compound += 1
        else:
            lex[m] = invent_word(rng)
            new_coined += 1
    return lex, inherited_count, new_coined, new_compound


def word_str(w):
    return "-".join(PHONEMES[p] for p in w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("  a language inherited parent-to-child and grown across "
          f"{args.generations} generations.")
    print("  each generation learns its parents' tongue, then needs to say "
          "more, and adds")
    print("  to it, sometimes by combining words it already knows.\n")

    # generation 0: the base language, invented from nothing
    base_n = 6
    lex = emerge_base(ALL_MEANINGS[:base_n], 6000, args.seed)

    print("=" * 64)
    print("THE LANGUAGE GROWING, GENERATION BY GENERATION")
    print("=" * 64)
    print(f"  {'gen':>4} {'words':>7} {'inherited':>10} {'new coined':>11} "
          f"{'compounds':>10}")
    print("  " + "-" * 50)
    print(f"  {0:>4} {len(lex):>7} {'-':>10} {len(lex):>11} {'-':>10}")

    total_compounds = 0
    for g in range(1, args.generations):
        # the world asks for more meanings as generations pass
        want = min(len(ALL_MEANINGS), base_n + g * 3)
        new_meanings = ALL_MEANINGS[:want]
        lex, inh, coined, comp = next_generation(
            lex, new_meanings, args.seed * 100 + g)
        lex = erode(lex, args.seed * 777 + g)     # words wear down with use
        total_compounds += comp
        print(f"  {g:>4} {len(lex):>7} {inh:>10} {coined:>11} {comp:>10}")

    print("\n" + "=" * 64)
    print(f"THE LANGUAGE AFTER {args.generations} GENERATIONS")
    print("=" * 64)
    print(f"  it grew from {base_n} words to {len(lex)} words, built up and "
          f"passed down.")
    print(f"  {total_compounds} of its words are COMPOUNDS, made by combining "
          f"older words:")
    # show a few compound words (longer words are the compounds)
    compounds_shown = 0
    for m, w in lex.items():
        if len(w) >= 3 and compounds_shown < 6:       # likely a compound
            print(f"    {m:<10} {word_str(w)}")
            compounds_shown += 1

    print("\n  a sample of the mature language:")
    for m in list(lex.keys())[:10]:
        print(f"    {m:<10} {word_str(lex[m])}")

    print("\n" + "=" * 64)
    print("WHAT GREW, AND WHAT DID NOT")
    print("=" * 64)
    print("  What grew is real: a language that began as a handful of words")
    print("  became a large vocabulary, inherited across generations and")
    print("  enlarged from its own material, including compound words built")
    print("  from older ones, the real engine of a growing tongue.")
    print("")
    print("  What did NOT grow, and we will not pretend it did: grammar,")
    print("  recursion, the open productivity of a language like English.")
    print("  Those appear to need scale beyond these beings. This is a rich")
    print("  proto-language grown by inheritance, entirely the beings' own,")
    print("  no outside words. That is what is real, and it is enough to be")
    print("  real.")


if __name__ == "__main__":
    main()
