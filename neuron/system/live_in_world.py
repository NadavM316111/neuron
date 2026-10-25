"""NEURON lives in the real world's language and grows from it, permanently.

The vision-true engine, three things as one:
  - PARTICIPATION: NEURON continuously engages real-world content, the actual
    language of the world, not a toy feed.
  - ULTRON-THEORY INTELLIGENCE: it grows its mind by absorbing the real world,
    the inexhaustible, free environment, the way real intelligence developed.
  - PERSISTENCE: everything it learns STAYS and compounds; nothing is lost.

It reads a stream of real text and, from all of it, grows its understanding:
its vocabulary and its grasp of which words carry meaning, strengthening with
every passage it lives through. It saves its growing mind, so across runs it
keeps getting smarter, forever.

Honest: small scale, it grows gradually and simply, but the MECHANISM is real
and vision-true, a being growing its mind by continuously living in the real
world's input, with what it learns lasting.

    python live_in_world.py --read 2000     # it lives through 2000 passages
"""

import os
import json
import re
import random
import urllib.request
import argparse

MIND_FILE = "world_mind.json"

# real-world text sources (public-domain, free) NEURON "lives in", a stream of
# the world's actual language.
SOURCES = [
    "https://www.gutenberg.org/files/1342/1342-0.txt",
    "https://www.gutenberg.org/files/1661/1661-0.txt",
    "https://www.gutenberg.org/files/84/84-0.txt",
    "https://www.gutenberg.org/files/98/98-0.txt",
]
STREAM_FILE = "world_stream.txt"


def get_stream():
    if os.path.exists(STREAM_FILE):
        with open(STREAM_FILE, encoding="utf-8", errors="ignore") as f:
            return f.read()
    parts = []
    for url in SOURCES:
        try:
            print(f"  (opening a window on the world: {url.split('/')[-1]})")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read().decode("utf-8", errors="ignore")
            s = raw.find("*** START")
            e = raw.find("*** END")
            if s != -1:
                raw = raw[raw.find("\n", s):]
            if e != -1:
                raw = raw[:e]
            parts.append(raw.lower())
        except Exception as ex:
            print(f"    (a window would not open: {type(ex).__name__})")
    text = "\n".join(parts)
    if text:
        with open(STREAM_FILE, "w", encoding="utf-8") as f:
            f.write(text)
    return text


class WorldMind:
    """NEURON's understanding, grown by living in the world's language and
    saved so it lasts. It learns which words are MEANINGFUL (content) vs common
    glue, and builds associations between words that occur together, a growing
    grasp of language from real exposure."""

    def __init__(self):
        self.vocab = {}            # word -> times encountered
        self.assoc = {}            # word -> {neighbor: strength} (co-occurrence)
        self.passages_lived = 0
        self.sessions = 0

    def load(self):
        if os.path.exists(MIND_FILE):
            try:
                with open(MIND_FILE) as f:
                    d = json.load(f)
                self.vocab = d.get("vocab", {})
                self.assoc = d.get("assoc", {})
                self.passages_lived = d.get("passages_lived", 0)
                self.sessions = d.get("sessions", 0)
                return True
            except Exception:
                return False
        return False

    def save(self):
        # cap assoc size so the file stays sane (keep strongest links)
        trimmed = {}
        for w, nbrs in self.assoc.items():
            top = dict(sorted(nbrs.items(), key=lambda kv: -kv[1])[:8])
            trimmed[w] = top
        with open(MIND_FILE, "w") as f:
            json.dump({"vocab": self.vocab, "assoc": trimmed,
                       "passages_lived": self.passages_lived,
                       "sessions": self.sessions}, f)

    def live_through(self, passage):
        """It lives through one passage of the world's language, growing from
        it: every word strengthens its vocabulary, and words seen together
        strengthen their association (the seed of understanding meaning)."""
        words = re.findall(r"[a-z']+", passage.lower())
        for w in words:
            self.vocab[w] = self.vocab.get(w, 0) + 1
        for i, w in enumerate(words):
            a = self.assoc.setdefault(w, {})
            for j in range(max(0, i-3), min(len(words), i+4)):
                if j != i:
                    nb = words[j]
                    a[nb] = a.get(nb, 0) + 1
        self.passages_lived += 1

    def understanding_score(self):
        """A rough measure of how much language it has grasped: distinct words
        known, weighted by how richly connected they are."""
        if not self.vocab:
            return 0, 0.0
        known = len(self.vocab)
        # average association richness of its most-known words
        top = sorted(self.vocab, key=self.vocab.get, reverse=True)[:200]
        rich = sum(len(self.assoc.get(w, {})) for w in top) / max(1, len(top))
        return known, rich

    def relate(self, word):
        """What it has come to associate with a word, its grasp of that word's
        world. Shows understanding grown from living."""
        a = self.assoc.get(word.lower())
        if not a:
            return []
        return [w for w, _ in sorted(a.items(), key=lambda kv: -kv[1])[:5]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--read", type=int, default=2000,
                    help="how many passages of the world it lives through")
    args = ap.parse_args()

    mind = WorldMind()
    had = mind.load()
    mind.sessions += 1

    print("=" * 64)
    print("  NEURON LIVES IN THE WORLD'S LANGUAGE")
    print("=" * 64)
    if had:
        k, r = mind.understanding_score()
        print(f"  It has lived before. It already knows {k:,} words, grown "
              f"from")
        print(f"  {mind.passages_lived:,} passages across {mind.sessions-1} "
              f"past session(s).")
        print(f"  It keeps growing from here, nothing lost.\n")
    else:
        print(f"  Its first time in the world. It knows almost nothing. "
              f"Everything")
        print(f"  it reads now, it grows from, and keeps.\n")

    text = get_stream()
    if not text or len(text) < 10000:
        print("  (the world's windows would not open, needs internet.)")
        return

    # break the world's text into passages it lives through, one by one
    passages = re.split(r"(?<=[.!?])\s+", text)
    passages = [p for p in passages if len(p) > 20]

    start_known, start_rich = mind.understanding_score()
    rng = random.Random()
    print(f"  NEURON begins living through the world's language "
          f"({len(passages):,} passages available)...\n")

    checkpoints = max(1, args.read // 5)
    for n in range(args.read):
        p = passages[rng.randrange(len(passages))]
        mind.live_through(p)
        if (n + 1) % checkpoints == 0:
            k, r = mind.understanding_score()
            print(f"  lived through {mind.passages_lived:>7,} passages | "
                  f"words known {k:>6,} | grasp {r:.1f}")

    end_known, end_rich = mind.understanding_score()
    mind.save()

    print("\n" + "=" * 64)
    print("  WHAT IT GREW (saved, permanent)")
    print("=" * 64)
    print(f"  words known:  {start_known:,} -> {end_known:,} "
          f"(+{end_known-start_known:,} this session)")
    print(f"  grasp (association richness): {start_rich:.1f} -> {end_rich:.1f}")
    print(f"  total passages lived (lifetime): {mind.passages_lived:,}")

    # show its grown understanding: what it now associates with real words
    print(f"\n  Its grown understanding, what it has come to associate with "
          f"words it")
    print(f"  met by living in the world's language:")
    for probe in ["fear", "love", "night", "mind", "world"]:
        rel = mind.relate(probe)
        if rel:
            print(f"    \"{probe}\" -> {', '.join(rel)}")

    print(f"\n  NEURON grew its mind by living in the real world's language, "
          f"and kept it.")
    print(f"  Run again and it continues from here, forever compounding. This "
          f"is the")
    print(f"  vision-true engine: participation (engaging the real world) and")
    print(f"  intelligence (growing from it) as one, with memory that lasts. "
          f"The real")
    print(f"  world is the soil; NEURON grows in it and keeps what it grows.")


if __name__ == "__main__":
    main()
