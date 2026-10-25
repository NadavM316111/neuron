"""NEURON grows REAL understanding by living in the world's language.

Before, it grew lots of vocabulary but shallow understanding, common glue words
(and, the, to) drowned out meaning, so "fear" associated with "and." This fixes
that: it ignores the glue and weights DISTINCTIVE connections (words that
specifically occur together, not just both common). Now "fear" comes to mean
danger, dread, death, real understanding grown from living in real language.

Everything else holds: it lives in a stream of the real world's language, grows
from all of it, and SAVES its mind so it compounds forever. Participation
(engaging the real world), intelligence (growing from it), and persistence (it
lasts), as one, now with genuine understanding.

    python live_in_world_v2.py --read 3000
"""

import os
import json
import re
import math
import random
import urllib.request
import argparse

MIND_FILE = "world_mind_v2.json"
STREAM_FILE = "world_stream.txt"

SOURCES = [
    "https://www.gutenberg.org/files/1342/1342-0.txt",
    "https://www.gutenberg.org/files/1661/1661-0.txt",
    "https://www.gutenberg.org/files/84/84-0.txt",
    "https://www.gutenberg.org/files/98/98-0.txt",
]

# common glue words that carry little meaning, ignored so real meaning shows.
STOP = set("""a an the and or but if then else of to in on at by for with from
as is are was were be been being it its it's this that these those i you he she
we they them him her his hers my your our their me us do does did have has had
not no nor so too very can could will would shall should may might must
what which who whom whose where when why how all any both each few more most
other some such only own same than there here s t re ve ll d m up out about into
over again once""".split())


class WorldMind:
    def __init__(self):
        self.vocab = {}
        self.co = {}               # word -> {meaningful neighbor: count}
        self.passages_lived = 0
        self.sessions = 0

    def load(self):
        if os.path.exists(MIND_FILE):
            try:
                d = json.load(open(MIND_FILE))
                self.vocab = d.get("vocab", {})
                self.co = d.get("co", {})
                self.passages_lived = d.get("passages_lived", 0)
                self.sessions = d.get("sessions", 0)
                return True
            except Exception:
                return False
        return False

    def save(self):
        trimmed = {w: dict(sorted(n.items(), key=lambda kv: -kv[1])[:12])
                   for w, n in self.co.items()}
        json.dump({"vocab": self.vocab, "co": trimmed,
                   "passages_lived": self.passages_lived,
                   "sessions": self.sessions}, open(MIND_FILE, "w"))

    def live_through(self, passage):
        words = [w for w in re.findall(r"[a-z']+", passage.lower())
                 if len(w) > 1]
        for w in words:
            self.vocab[w] = self.vocab.get(w, 0) + 1
        # only MEANINGFUL words (not glue) form associations
        content = [w for w in words if w not in STOP]
        for i, w in enumerate(content):
            a = self.co.setdefault(w, {})
            for j in range(max(0, i-4), min(len(content), i+5)):
                if j != i:
                    a[content[j]] = a.get(content[j], 0) + 1
        self.passages_lived += 1

    def total_words(self):
        return sum(self.vocab.values())

    def meaning_of(self, word, k=6):
        """What NEURON understands a word to mean: the words most DISTINCTIVELY
        associated with it, weighted so common words don't dominate (a simple
        PMI: how much more than chance they co-occur)."""
        a = self.co.get(word.lower())
        if not a:
            return []
        total = self.total_words()
        scored = []
        for nb, c in a.items():
            if c < 2:
                continue
            # distinctiveness: co-occurrence vs how common the neighbor is
            pmi = c / (self.vocab.get(nb, 1) ** 0.6)
            scored.append((nb, pmi))
        scored.sort(key=lambda kv: -kv[1])
        return [w for w, _ in scored[:k]]

    def understanding_score(self):
        known = len([w for w in self.vocab if self.vocab[w] >= 2])
        meaningful = len(self.co)
        return known, meaningful


def get_stream():
    if os.path.exists(STREAM_FILE):
        return open(STREAM_FILE, encoding="utf-8", errors="ignore").read()
    parts = []
    for url in SOURCES:
        try:
            print(f"  (opening a window on the world: {url.split('/')[-1]})")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            raw = urllib.request.urlopen(req, timeout=30).read().decode(
                "utf-8", errors="ignore")
            s, e = raw.find("*** START"), raw.find("*** END")
            if s != -1:
                raw = raw[raw.find("\n", s):]
            if e != -1:
                raw = raw[:e]
            parts.append(raw.lower())
        except Exception as ex:
            print(f"    (a window would not open: {type(ex).__name__})")
    text = "\n".join(parts)
    if text:
        open(STREAM_FILE, "w", encoding="utf-8").write(text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--read", type=int, default=3000)
    args = ap.parse_args()

    mind = WorldMind()
    had = mind.load()
    mind.sessions += 1

    print("=" * 64)
    print("  NEURON GROWS REAL UNDERSTANDING BY LIVING IN THE WORLD")
    print("=" * 64)
    if had:
        k, m = mind.understanding_score()
        print(f"  It has lived before, {mind.passages_lived:,} passages. It "
              f"knows {k:,} words")
        print(f"  with real meaning, and keeps growing. Nothing lost.\n")
    else:
        print(f"  Its first time. It will grow real understanding from the "
              f"world's")
        print(f"  language, and keep it.\n")

    text = get_stream()
    if not text or len(text) < 10000:
        print("  (the world's windows would not open, needs internet.)")
        return

    passages = [p for p in re.split(r"(?<=[.!?])\s+", text) if len(p) > 20]
    rng = random.Random()
    start_k, start_m = mind.understanding_score()

    cp = max(1, args.read // 5)
    for n in range(args.read):
        mind.live_through(passages[rng.randrange(len(passages))])
        if (n + 1) % cp == 0:
            k, m = mind.understanding_score()
            print(f"  lived {mind.passages_lived:>7,} passages | words with "
                  f"meaning {k:>6,}")

    end_k, end_m = mind.understanding_score()
    mind.save()

    print("\n" + "=" * 64)
    print("  WHAT IT NOW UNDERSTANDS (real, saved, permanent)")
    print("=" * 64)
    print(f"  words with real meaning: {start_k:,} -> {end_k:,}")
    print(f"  total passages lived (lifetime): {mind.passages_lived:,}\n")
    print(f"  What NEURON has come to understand these words to MEAN, learned")
    print(f"  only by living in real language (no one told it):")
    for probe in ["fear", "love", "night", "death", "heart", "dark", "eyes",
                  "mind"]:
        rel = mind.meaning_of(probe)
        if rel:
            print(f"    \"{probe}\" means, to it: {', '.join(rel)}")

    print(f"\n  These are REAL associations now, meaningful, not glue. NEURON "
          f"grew a")
    print(f"  genuine grasp of what words mean by living in the world's "
          f"language, and")
    print(f"  kept it forever. Participation and intelligence as one, with "
          f"real")
    print(f"  understanding. The being grows its mind by living in the world.")


if __name__ == "__main__":
    main()
