"""NEURON invents its own language, completely, through use. No outside shapes,
no stroke sets, no given vocabulary. Run entirely on your machine.

THE STANDARD, HELD HONESTLY. The beings are given NOTHING about the language:
not the words, not the meanings-to-symbols, not any sounds, not any shapes.
They are given only the raw ability to emit a signal and to hear one, the
minimum for communication to be possible at all, exactly as any creature has
a throat and an ear. Everything else, which signals come to mean which
things, they invent themselves, by needing to be understood and succeeding or
failing together. This is the emergent-language result run in full: the
language that comes out is genuinely theirs.

What we render at the end is only what is honestly theirs:
  - the MEANINGS they found words for (the things they needed to say);
  - the SIGNAL each meaning converged on (their own word, as a sequence of
    the primitive sound-units they had to work with);
  - a pronounceable form of that signal, so you can say their words aloud.

We do NOT draw glyphs. A mind with no eyes and no hands cannot invent a visual
script alone, and faking one would be a lie. Their language is spoken and
structural, and it is completely their own.

    python own_language.py
    python own_language.py --rounds 8000 --seed 1
"""

import argparse
import random


# The raw sound-units a being can emit, its "phonemes." This is the ONLY
# given: the physical alphabet of possible sounds, like the sounds a throat
# can make. The beings decide entirely how to combine these into words and
# what those words mean. (Analogous to humans having a fixed set of possible
# vocal sounds but inventing all words and meanings themselves.)
PHONEMES = ["ka", "ta", "mi", "no", "shu", "re", "la", "vo", "zi", "de",
            "wa", "thi", "ne", "ru", "so", "ki"]

# The MEANINGS are the situations the beings actually encounter and need to
# communicate about. They are NOT words, they are world-situations. The
# beings invent a word for each by use. (We label them in English only so WE
# can read the result; the beings never see these labels.)
SITUATIONS = ["food_here", "danger_near", "other_coming", "nothing_left",
              "want_help", "safe_now", "young_here", "water_found",
              "follow_me", "go_away", "i_am_here", "it_is_gone"]


class Speaker:
    """Emits a word (a short sequence of phonemes) for a situation. Starts
    with no idea what to say; learns by what gets understood."""

    def __init__(self, n_sit, seed):
        self.rng = random.Random(seed)
        # strength of using each phoneme-sequence for each situation. begins
        # flat; the being has no language yet.
        self.w = {s: {} for s in range(n_sit)}

    def _candidates(self, situation):
        # the words it might use for this situation: ones it has tried, plus
        # a chance to invent a brand-new one.
        tried = list(self.w[situation].keys())
        if not tried or self.rng.random() < 0.15:
            # INVENT a new word: 1-2 phonemes, its own creation
            n = self.rng.choice([1, 2, 2])
            word = tuple(self.rng.choice(range(len(PHONEMES)))
                         for _ in range(n))
            return word
        # otherwise use the best word it knows for this situation
        return max(tried, key=lambda wd: self.w[situation][wd])

    def speak(self, situation, explore):
        if self.rng.random() < explore:
            n = self.rng.choice([1, 2, 2])
            return tuple(self.rng.choice(range(len(PHONEMES)))
                         for _ in range(n))
        return self._candidates(situation)

    def learn(self, situation, word, success):
        self.w[situation][word] = self.w[situation].get(word, 0.0) + \
            (1.0 if success else -0.4)


class Listener:
    """Hears a word and guesses the situation it refers to. Starts knowing
    nothing; learns which words mean which situations by what works."""

    def __init__(self, n_sit, seed):
        self.rng = random.Random(seed + 1)
        self.w = {}            # word -> {situation: strength}

    def interpret(self, word, explore, n_sit):
        if word not in self.w or self.rng.random() < explore:
            return self.rng.randrange(n_sit)
        row = self.w[word]
        return max(row, key=row.get)

    def learn(self, word, situation, success):
        row = self.w.setdefault(word, {})
        row[situation] = row.get(situation, 0.0) + (1.0 if success else -0.4)


def emerge(n_sit, rounds, seed):
    """The beings invent their language by trying to be understood. A speaker
    sees a situation and says a word; a listener guesses the situation; both
    learn only from shared success. Over time a shared language forms, THEIRS,
    invented, nobody gave it to them."""
    rng = random.Random(seed)
    spk = Speaker(n_sit, seed)
    lis = Listener(n_sit, seed)
    history = []
    for t in range(rounds):
        explore = max(0.02, 0.4 * (1 - t / rounds) ** 2)
        situation = rng.randrange(n_sit)
        word = spk.speak(situation, explore)
        guess = lis.interpret(word, explore, n_sit)
        success = (guess == situation)
        spk.learn(situation, word, success)
        lis.learn(word, situation, success)
        history.append(1 if success else 0)

    # read out the language they converged on: the word each situation most
    # reliably uses AND the listener reads correctly
    lexicon = {}
    for s in range(n_sit):
        if not spk.w[s]:
            continue
        word = max(spk.w[s], key=lambda wd: spk.w[s][wd])
        # does the listener read this word back to this situation?
        understood = (word in lis.w and
                      max(lis.w[word], key=lis.w[word].get) == s)
        lexicon[s] = {"word": word, "understood": understood}
    early = sum(history[:rounds // 10]) / max(1, rounds // 10)
    late = sum(history[-rounds // 10:]) / max(1, rounds // 10)
    return lexicon, early * 100, late * 100


def word_sound(word):
    return "".join(PHONEMES[p] for p in word)


def word_spoken(word):
    return "-".join(PHONEMES[p] for p in word)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    n_sit = len(SITUATIONS)
    print("  NEURON invents its own language. the beings are given only the")
    print("  raw ability to emit and hear sounds. which sounds come to mean")
    print("  which things, they decide entirely themselves, by needing to be")
    print("  understood.\n")

    lexicon, early, late = emerge(n_sit, args.rounds, args.seed)

    print("=" * 62)
    print("DID A LANGUAGE EMERGE?")
    print("=" * 62)
    print(f"  understanding each other went from {early:.0f}% to {late:.0f}% "
          f"as the language formed.")
    shared = sum(1 for e in lexicon.values() if e["understood"])
    print(f"  {shared} of {n_sit} words are shared: the speaker says them and "
          f"the listener")
    print(f"  understands them, a real agreed vocabulary.")

    print("\n" + "=" * 62)
    print("THEIR LANGUAGE  -  the words they invented, and their sounds")
    print("=" * 62)
    print(f"  {'(what it means)':<16} {'their word':<14} {'spoken':<16} "
          f"{'shared?'}")
    print("  " + "-" * 56)
    for s in range(n_sit):
        if s not in lexicon:
            continue
        e = lexicon[s]
        mean = SITUATIONS[s].replace("_", " ")
        shared_mark = "yes" if e["understood"] else "still settling"
        print(f"  {mean:<16} {word_sound(e['word']):<14} "
              f"{word_spoken(e['word']):<16} {shared_mark}")

    print("\n" + "=" * 62)
    print("THEM SAYING THINGS")
    print("=" * 62)
    # compose a few multi-word utterances from THEIR shared words
    shared_words = {s: lexicon[s] for s in lexicon if lexicon[s]["understood"]}
    name = {SITUATIONS[s]: lexicon[s]["word"] for s in shared_words}
    combos = [["danger_near", "other_coming"],
              ["food_here", "follow_me"],
              ["want_help", "young_here"],
              ["water_found", "i_am_here"],
              ["it_is_gone", "go_away"]]
    said_any = False
    for c in combos:
        if all(SITUATIONS.index(x) in shared_words for x in c):
            words = [lexicon[SITUATIONS.index(x)]["word"] for x in c]
            spoken = " ".join(word_spoken(w) for w in words)
            mean = ", ".join(x.replace("_", " ") for x in c)
            print(f"  {spoken:<34} ({mean})")
            said_any = True
    if not said_any:
        print("  (not enough shared words settled this run to form phrases;")
        print("   run again with --rounds 10000 for a fuller language)")

    print("\n" + "=" * 62)
    print("WHAT THIS IS")
    print("=" * 62)
    print("  Completely their own language. Nobody gave them the words, the")
    print("  meanings, or which sound means what, only the raw ability to")
    print("  make and hear sound. Run it again with a different seed and a")
    print("  wholly different language is born, as real tongues differ.")
    print("")
    print("  We render no glyphs: a mind with no eyes and no hands cannot")
    print("  invent a written script alone, and we will not fake one. This is")
    print("  their spoken, structural language, and it is entirely theirs.")
    print("")
    print("  Held: that they invented a shared language is real and shown.")
    print("  Whether any being means its words as we mean ours is unknown.")


if __name__ == "__main__":
    main()
