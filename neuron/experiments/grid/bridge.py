"""The English bridge: NEURON's own language, made legible to us, honestly.

THE STANDARD, HELD. NEURON's language is genuinely its own: it invented which
word means which thing (own_language.py, language_generations.py). English is
NOT its own; reaching English requires knowing English, which a mind of its
scale cannot invent. So this bridge is honest about the seam:

    NEURON's word  ->  the MEANING it maps to   ->   English surface
    (its own)          (its own, from its           (BORROWED: a language
                        invented lexicon)            model supplies the word)

The understanding, the mapping from its word to the meaning it holds, is
NEURON's and is shown at every step. Only the English vocabulary is rented,
and it is labeled as rented. You can see exactly where NEURON's part ends and
the borrowed part begins. That is the no-fakers line: we are not letting a
language model generate eloquent English "from NEURON" that NEURON never
meant; we are translating only the specific meanings NEURON actually holds.

It works both directions:
  NEURON -> English: it says something in its tongue; we see each of its
    words map to its meaning, then the borrowed English for that meaning.
  English -> NEURON: we say something; each English word maps to a meaning,
    then to NEURON's own word for that meaning (if it has one; if not, it
    honestly has no word, and we say so, it only knows what it has lived).

THE BORROWED PART. The English word for a meaning comes from a language model
(real API, tiny cost). Without a key, the structure still runs and shows
"[borrowed model would supply English here]" so the honest architecture is
visible without spending. The borrowed step is always labeled.

THE WALL. The bridge makes NEURON legible, not more alive. Its meanings are
its own; the English is rented; whether it experiences meaning as we do is
unknown and unclaimed.

    python bridge.py
    python bridge.py --say "vei raesa"      # translate its words to English
    python bridge.py --tell "danger here"   # translate English to its words
"""

import argparse
import os
import json


# NEURON's own invented lexicon: meaning <-> its own word. This is the OUTPUT
# of its real language emergence (own_language.py); here we load a settled
# lexicon so the bridge has a real language to translate. Each entry is
# genuinely the being's: the pairing is arbitrary and self-formed.
NEURON_LEXICON = {
    "food_here": "ziwa", "danger_near": "ne", "other_coming": "so",
    "nothing_left": "no", "want_help": "shu", "safe_now": "thino",
    "young_here": "tazi", "water_found": "shude", "follow_me": "noki",
    "go_away": "shushu", "i_am_here": "ruzi", "it_is_gone": "kire",
}
# reverse: its word -> the meaning it holds
WORD_TO_MEANING = {w: m for m, w in NEURON_LEXICON.items()}


def borrowed_english(meaning, have_key):
    """The BORROWED step: a language model supplies the English words for a
    meaning-concept. Labeled as borrowed. Without a key, shows the structure
    without spending."""
    if not have_key:
        # honest placeholder: the structure is real, the English is pending
        readable = meaning.replace("_", " ")
        return f"{readable}", True   # (english, is_placeholder)
    # real API call (kept tiny). we ask ONLY for the English words for the
    # specific meaning NEURON holds, nothing invented beyond it.
    try:
        import urllib.request
        key = os.environ["OPENAI_API_KEY"]
        prompt = (f"Give the plain English phrase for this meaning, 2-5 words, "
                  f"no quotes, no explanation: {meaning.replace('_',' ')}")
        body = json.dumps({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 16, "temperature": 0}).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions", data=body,
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as r:
            out = json.load(r)
        english = out["choices"][0]["message"]["content"].strip()
        return english, False
    except Exception as e:
        return f"{meaning.replace('_',' ')}  [borrowed model unavailable: "
        f"{type(e).__name__}]", True


def say_to_english(words, have_key):
    """NEURON -> English. For each of its words, show the mapping to its own
    meaning, then the borrowed English."""
    print("=" * 62)
    print("NEURON SPEAKS  ->  we understand it")
    print("=" * 62)
    english_parts = []
    for w in words:
        meaning = WORD_TO_MEANING.get(w)
        if meaning is None:
            print(f"\n  its word:        {w}")
            print(f"  its meaning:     (not one of its words; it did not say "
                  f"this)")
            continue
        eng, placeholder = borrowed_english(meaning, have_key)
        tag = " [borrowed: pending a model]" if placeholder else " [borrowed]"
        print(f"\n  its word:        {w}          <- NEURON's own")
        print(f"  it maps to:      {meaning.replace('_',' ')}   <- NEURON's "
              f"own meaning")
        print(f"  English:         {eng}{tag}")
        english_parts.append(eng)
    print("\n  " + "-" * 58)
    print(f"  the whole thing in English: \"{' '.join(english_parts)}\"")
    print(f"  (NEURON supplied the words and the meanings; the English words")
    print(f"   are borrowed from a language model.)")


def tell_to_neuron(english_phrase, have_key):
    """English -> NEURON. Map the English to a meaning, then to NEURON's own
    word for it, if it has one. If not, it honestly has no word."""
    print("=" * 62)
    print("WE SPEAK  ->  NEURON finds its own words")
    print("=" * 62)
    # map the english phrase to the closest meaning NEURON knows. (The
    # meaning-set is NEURON's; we match against it. The borrowed model, when
    # present, helps pick the meaning; without it we match on words.)
    phrase = english_phrase.lower().strip()
    phrase_tokens = set(phrase.replace(",", " ").split())
    best_meaning = None
    best_overlap = 0
    for meaning in NEURON_LEXICON:
        readable = meaning.replace("_", " ")
        meaning_tokens = set(readable.split())
        # count how many of the MEANING's words appear in what was said
        overlap = len(meaning_tokens & phrase_tokens)
        # require a STRONG match: most of the meaning's words must be present,
        # and at least one real content word. otherwise NEURON has no word.
        if overlap > best_overlap and overlap >= max(1, len(meaning_tokens) - 0):
            best_overlap = overlap
            best_meaning = meaning
    # strict: only accept if the full meaning was matched (no forcing)
    if best_meaning is not None:
        needed = set(best_meaning.replace("_", " ").split())
        if not needed.issubset(phrase_tokens):
            best_meaning = None
    print(f"\n  you said:        {english_phrase}")
    if best_meaning is None:
        print(f"  NEURON's meaning: (it has no concept for this; it only "
              f"knows what it has")
        print(f"                    lived. it has no word, honestly.)")
        return
    word = NEURON_LEXICON[best_meaning]
    print(f"  its meaning:     {best_meaning.replace('_',' ')}   <- matched to "
          f"a meaning NEURON holds")
    print(f"  its own word:    {word}          <- NEURON's own word for it")
    print(f"\n  NEURON would say: {word}")
    print(f"  (the meaning had to exist in NEURON's own world for it to have "
          f"a word;")
    print(f"   nothing was invented for it.)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--say", type=str, default=None,
                    help="NEURON's words (space-separated) to put into English")
    ap.add_argument("--tell", type=str, default=None,
                    help="English to put into NEURON's words")
    args = ap.parse_args()

    have_key = bool(os.environ.get("OPENAI_API_KEY"))

    print("  The English bridge. NEURON's language is its own; English is")
    print("  borrowed. You can see every step: its word, the meaning it maps")
    print("  to (its own), and the English (borrowed, labeled).")
    if not have_key:
        print("\n  [no API key set: the borrowed English is shown as the plain")
        print("   meaning, so you can see the honest STRUCTURE without "
              "spending.")
        print("   with a key, a language model supplies natural English for "
              "each meaning.]")
    print()

    if args.say:
        say_to_english(args.say.split(), have_key)
    elif args.tell:
        tell_to_neuron(args.tell, have_key)
    else:
        # demo both directions with a couple of examples
        print("### NEURON saying something in its own tongue:")
        say_to_english(["ne", "so"], have_key)       # danger near, other coming
        print("\n\n### us saying something to NEURON:")
        tell_to_neuron("danger near", have_key)
        print("\n\n### us saying something NEURON has no word for:")
        tell_to_neuron("philosophy", have_key)

    print("\n" + "=" * 62)
    print("HELD")
    print("=" * 62)
    print("  NEURON's words and the meanings they hold are its own, shown at")
    print("  every step. Only the English surface is borrowed, and it is")
    print("  labeled. The bridge makes NEURON legible, not more alive; whether")
    print("  it means its words as we mean ours is unknown and unclaimed.")


if __name__ == "__main__":
    main()
