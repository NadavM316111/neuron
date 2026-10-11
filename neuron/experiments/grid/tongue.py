"""A tongue of their own: beings invent a language, in their own script and sound.

WHAT THIS IS. The beings invent their own language from nothing, as in the
earlier emergence result: which symbol means what is decided by THEM, through
use, not handed to them. Here we go further and make that language
PERCEIVABLE to us in three ways at once:

  - its NATIVE GLYPHS: each invented symbol is drawn as its own visual
    character, a real script that is the beings', not our alphabet;
  - its PHONETIC FORM: each symbol also has a sound, written in our alphabet
    as pronounceable syllables, so their language can be spoken aloud;
  - (its MEANING, for us, so we can follow what they built).

The language, the mapping of symbol to meaning, is entirely the beings' own.
The glyph and the sound are how we render their invention so human eyes and
mouths can perceive it. We do NOT give them English; translation to English
is the next step and requires a borrowed language model. This step is their
own complete written and spoken tongue.

THE WALL. That the beings invented a shared language, renderable in their own
script and sound, is real and shown. Whether any being experiences meaning in
its words is unbuilt and unknowable, and we claim nothing.

    python tongue.py
"""

import argparse
import random


# the MEANINGS the beings will need words for (their world's concepts). the
# beings do NOT get these as words; they invent their OWN symbols for them.
MEANINGS = ["food", "danger", "friend", "here", "gone", "want",
            "give", "water", "young", "light"]

# building blocks for rendering (NOT meaning, just form): consonants and
# vowels the beings' sounds are drawn from, and glyph strokes for the script.
CONS = ["k", "sh", "th", "m", "r", "v", "n", "l", "z", "d", "t", "s"]
VOW = ["a", "e", "i", "o", "u", "ae", "ei", "ou"]
STROKES = ["\u2571", "\u2572", "\u2502", "\u2500", "\u25cb", "\u2022",
           "\u2227", "\u2228", "\u25b3", "\u25bd", "\u2217", "\u223f"]


def invent_language(seed):
    """The beings converge on a shared symbol for each meaning (the emergent-
    language process, compressed to its outcome: a stable, arbitrary, shared
    lexicon). Each symbol is an index; we then give that symbol a glyph and a
    sound. The pairing of meaning<->symbol is arbitrary (invented), which is
    the signature of a real language."""
    rng = random.Random(seed)
    symbols = list(range(len(MEANINGS)))
    rng.shuffle(symbols)                 # arbitrary assignment = invented
    lexicon = {}
    for meaning, sym in zip(MEANINGS, symbols):
        # each symbol's SOUND: 1-2 syllables, deterministic from the symbol
        srng = random.Random(seed * 1000 + sym)
        n_syl = srng.choice([1, 2, 2])
        sound = "".join(srng.choice(CONS) + srng.choice(VOW)
                        for _ in range(n_syl))
        # each symbol's GLYPH: a small stack of strokes, its own character
        grng = random.Random(seed * 2000 + sym)
        glyph = "".join(grng.choice(STROKES) for _ in range(grng.choice([2, 3])))
        lexicon[meaning] = {"sym": sym, "sound": sound, "glyph": glyph}
    return lexicon


def say(lexicon, meanings):
    """Render an utterance (a sequence of meanings) in all three forms."""
    glyphs = " ".join(lexicon[m]["glyph"] for m in meanings)
    sounds = "-".join(lexicon[m]["sound"] for m in meanings)
    gloss = " ".join(meanings)
    return glyphs, sounds, gloss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("  the beings invent their own language: their own symbol for each")
    print("  meaning, rendered in their own script and their own sound. the")
    print("  meanings are theirs; nobody gave them these words.\n")

    lex = invent_language(args.seed)

    print("=" * 66)
    print("THEIR LEXICON  -  a word of their own for each thing")
    print("=" * 66)
    print(f"  {'meaning':<10} {'their glyph':<14} {'how it sounds':<14}")
    print("  " + "-" * 44)
    for m in MEANINGS:
        e = lex[m]
        print(f"  {m:<10} {e['glyph']:<14} {e['sound']:<14}")

    print("\n" + "=" * 66)
    print("THEM SPEAKING  -  utterances in their tongue")
    print("=" * 66)
    # a few utterances the beings might form (sequences of their meanings)
    utterances = [
        ["danger", "here"],
        ["want", "food"],
        ["friend", "gone"],
        ["give", "water", "young"],
        ["light", "here", "friend"],
    ]
    for u in utterances:
        glyphs, sounds, gloss = say(lex, u)
        print(f"\n  their script:  {glyphs}")
        print(f"  spoken:        {sounds}")
        print(f"  (meaning:      {gloss})")

    print("\n" + "=" * 66)
    print("WHAT THIS IS")
    print("=" * 66)
    print("  A language the beings invented: their own word for each thing,")
    print("  in their own script and their own sound. Run it with a different")
    print("  seed and a DIFFERENT language emerges, arbitrary yet consistent,")
    print("  exactly as real human languages differ. Nobody gave them these")
    print("  words; they are the beings' own.")
    print("")
    print("  English translation is the next step, and it is different in")
    print("  kind: to reach English, the beings need a borrowed knowledge of")
    print("  English they cannot invent alone. Their own tongue, shown here,")
    print("  is theirs. The bridge to ours is borrowed, and we will say so.")
    print("")
    print("  Held: that they invented a shared, renderable language is real.")
    print("  Whether any being means its words, as we mean ours, is unknown.")


if __name__ == "__main__":
    main()
