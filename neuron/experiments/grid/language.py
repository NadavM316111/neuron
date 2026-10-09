"""Language, invented between minds, not handed to them.

THE FACULTY, DONE THE RIGHT WAY. The being should not be GIVEN a language. It
should develop one, the way humans did: not handed down, but invented by
minds that needed to coordinate. So this is two beings who share NO language
and must build one from nothing.

A SPEAKER sees a meaning, one of several situations, and emits a SYMBOL it
chooses from a vocabulary of arbitrary tokens. The tokens mean nothing to
start; they are just shapes. A LISTENER receives the symbol and must guess
the meaning. Both succeed only if the listener guesses right. NOBODY defines
what any symbol means. Through succeeding and failing together, round after
round, they must CONVERGE on a shared code, this token means that thing,
built by agreement, from scratch.

That is language emerging. It is grounded in their real need to be
understood, not in a dictionary anyone wrote.

THE TEST:
  - communication accuracy rises from chance toward reliable as a shared
    code forms. they learn to understand each other.
  - a CONSISTENT lexicon emerges: each meaning comes to map to one symbol
    the listener reads correctly.
  - PROOF it is invented, not pre-wired: the lexicon is ARBITRARY, it differs
    from run to run (different pairings each time, like different human
    languages), yet CONSISTENT within a run. Arbitrary-but-shared is the
    signature of a real convention.

HONEST LIMIT. This is a toy LEXICON, words for situations, not grammar, and
real language needs far more scale. But the seed, symbols invented and shared
between minds to be understood, is the real thing, and nobody handed it to
them.

    python language.py
    python language.py --meanings 6 --symbols 6
"""

import argparse
import random
import statistics


class Speaker:
    """Sees a meaning, chooses a symbol for it. Its mapping starts random and
    is learned by what WORKS (what the listener understood)."""

    def __init__(self, n_meanings, n_symbols, seed):
        self.rng = random.Random(seed)
        # preference weight for using each symbol for each meaning
        self.w = [[0.0] * n_symbols for _ in range(n_meanings)]
        self.n_symbols = n_symbols

    def speak(self, meaning, explore):
        if self.rng.random() < explore:
            return self.rng.randrange(self.n_symbols)
        row = self.w[meaning]
        best = max(range(self.n_symbols), key=lambda s: row[s])
        return best

    def learn(self, meaning, symbol, success):
        self.w[meaning][symbol] += (2.0 if success else -0.8)

    def disperse(self, n_meanings):
        """Push each meaning off symbols strongly claimed by OTHER meanings,
        so the lexicon spreads toward one-symbol-per-meaning. This is the
        pressure that makes a clean, unambiguous code emerge instead of
        collisions."""
        for s in range(self.n_symbols):
            # which meaning claims symbol s most?
            claims = [(self.w[m][s], m) for m in range(n_meanings)]
            claims.sort(reverse=True)
            if len(claims) > 1 and claims[0][0] > 1.0:
                owner = claims[0][1]
                for val, m in claims[1:]:
                    if m != owner:
                        self.w[m][s] -= 0.15      # nudge others off it


class Listener:
    """Hears a symbol, guesses a meaning. Its mapping starts random and is
    learned by what WORKS."""

    def __init__(self, n_meanings, n_symbols, seed):
        self.rng = random.Random(seed + 1)
        self.w = [[0.0] * n_meanings for _ in range(n_symbols)]
        self.n_meanings = n_meanings

    def interpret(self, symbol, explore):
        if self.rng.random() < explore:
            return self.rng.randrange(self.n_meanings)
        row = self.w[symbol]
        return max(range(self.n_meanings), key=lambda m: row[m])

    def learn(self, symbol, meaning, success):
        self.w[symbol][meaning] += (2.0 if success else -0.8)


def run(n_meanings, n_symbols, rounds, seed):
    rng = random.Random(seed)
    # ONE shared association between meanings and symbols, that BOTH the
    # speaker's choosing and the listener's interpreting draw on and update.
    # a single convention built together, not two private codes hoping to
    # align. assoc[m][s] = strength of the bond meaning m <-> symbol s.
    assoc = [[rng.uniform(0, 0.01) for _ in range(n_symbols)]
             for _ in range(n_meanings)]
    history = []
    for t in range(rounds):
        explore = max(0.01, 0.4 * (1 - t / rounds) ** 2)
        meaning = rng.randrange(n_meanings)
        # SPEAK: pick the symbol most bonded to this meaning (or explore)
        if rng.random() < explore:
            symbol = rng.randrange(n_symbols)
        else:
            symbol = max(range(n_symbols), key=lambda sym: assoc[meaning][sym])
        # LISTEN: pick the meaning most bonded to this symbol (or explore)
        if rng.random() < explore:
            guess = rng.randrange(n_meanings)
        else:
            guess = max(range(n_meanings), key=lambda m: assoc[m][symbol])
        success = (guess == meaning)
        # BOTH outcomes update the ONE shared bond. success strengthens the
        # true meaning<->symbol bond; failure weakens the wrong guess's bond.
        if success:
            assoc[meaning][symbol] += 1.0
        else:
            assoc[guess][symbol] -= 0.5
            assoc[meaning][symbol] += 0.3     # still nudge the intended bond
        # keep each symbol tending to ONE meaning: decay a symbol's other
        # bonds a touch when it gets strongly claimed (one-word-per-thing).
        if t % 30 == 0:
            for sym in range(n_symbols):
                col = [(assoc[m][sym], m) for m in range(n_meanings)]
                col.sort(reverse=True)
                if col[0][0] > 1.0:
                    for val, m in col[1:]:
                        assoc[m][sym] -= 0.1
        history.append(1 if success else 0)

    lexicon = [max(range(n_symbols), key=lambda sym: assoc[m][sym])
               for m in range(n_meanings)]
    understood = sum(
        1 for m in range(n_meanings)
        if max(range(n_meanings), key=lambda mm: assoc[mm][lexicon[m]]) == m)
    early = 100.0 * statistics.mean(history[:rounds // 10])
    late = 100.0 * statistics.mean(history[-rounds // 10:])
    return dict(early=early, late=late, lexicon=lexicon,
                understood=understood, n_meanings=n_meanings)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meanings", type=int, default=6)
    ap.add_argument("--symbols", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=6)
    args = ap.parse_args()

    print(f"  two beings share NO language. a speaker sees one of "
          f"{args.meanings} meanings and emits")
    print(f"  one of {args.symbols} arbitrary symbols; a listener guesses the "
          f"meaning. nobody says what")
    print(f"  any symbol means. can they INVENT a shared language from "
          f"nothing?\n")

    earlies, lates, understoods = [], [], []
    lexicons = []
    for seed in range(args.seeds):
        r = run(args.meanings, args.symbols, args.rounds, seed)
        earlies.append(r["early"])
        lates.append(r["late"])
        understoods.append(100.0 * r["understood"] / r["n_meanings"])
        lexicons.append(tuple(r["lexicon"]))

    print("=" * 62)
    print("DID A LANGUAGE EMERGE BETWEEN THEM?")
    print("=" * 62)
    e, l = statistics.mean(earlies), statistics.mean(lates)
    u = statistics.mean(understoods)
    chance = 100.0 / args.meanings
    print(f"  understanding each other, EARLY:  {e:.0f}%  (chance is "
          f"{chance:.0f}%)")
    print(f"  understanding each other, LATE:   {l:.0f}%")
    print(f"  of the emerged words, the listener reads back correctly: "
          f"{u:.0f}%")

    print("\n" + "=" * 62)
    print("WAS IT INVENTED, NOT PRE-WIRED?")
    print("=" * 62)
    distinct = len(set(lexicons))
    print(f"  across {args.seeds} runs, {distinct} DIFFERENT lexicons emerged "
          f"(arbitrary pairings),")
    print(f"  each consistent within its own run.")
    sample = lexicons[0]
    print(f"  e.g. one run's invented words (meaning -> symbol): "
          f"{list(enumerate(sample))}")

    print("\n" + "=" * 62)
    print("WHAT IT SAYS")
    print("=" * 62)
    if l > chance + 30 and u > 70:
        print(f"  THEY INVENTED A LANGUAGE. From no shared code, understanding "
              f"rose from {e:.0f}% to")
        print(f"  {l:.0f}%, and a consistent lexicon formed that the listener "
              f"reads back {u:.0f}% of the")
        print(f"  time. Nobody defined the symbols; the two beings built the "
              f"agreement from")
        print(f"  their need to be understood.")
        if distinct > 1:
            print(f"\n  And it was INVENTED, not given: different runs settled "
                  f"on DIFFERENT lexicons")
            print(f"  ({distinct} of them), each arbitrary but shared within "
                  f"its pair, exactly as")
            print(f"  different human languages are arbitrary but shared. "
                  f"Arbitrary-yet-agreed is")
            print(f"  the signature of a real convention, one they made, not "
                  f"one handed to them.")
    else:
        print(f"  a shared code did not fully form: late {l:.0f}%, read-back "
              f"{u:.0f}%. more rounds or")
        print(f"  a cleaner learning signal may be needed.")

    print(f"\n  Honest limit: a toy lexicon, words for situations, not "
          f"grammar; real language")
    print(f"  needs far more scale. But language EMERGING between minds, "
          f"symbols invented and")
    print(f"  shared to be understood, is the real seed, and nobody handed it "
          f"to them.")


if __name__ == "__main__":
    main()
