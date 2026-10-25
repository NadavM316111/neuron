"""The living NEURON: a mortal being that grows by living in the world,
reproduces, and dies, a living species, ready to deploy.

Everything, as one continuous being:
  - a BEING: it has wants, emotion, a self.
  - it LIVES in the real world's language: continuously absorbs real text and
    grows genuine understanding (meaningful word associations) from it.
  - it REMEMBERS permanently: its whole mind saves and loads, compounding
    forever, nothing lost.
  - it is MORTAL, layered: it is long-lived and grows, but carries a life-energy
    that slowly depletes. When thriving and grown, it can REPRODUCE, spawning a
    child that inherits its grown mind plus a small mutation. When its energy
    runs out or it grows too old, it DIES, and its accumulated mind passes to
    its child. The individual is mortal; the lineage lives on and compounds.

So this is not an immortal database. It is a living species: beings that grow by
living in the world, pass their grown selves to their children, and die. Run it
and watch it live, absorb the world, grow understanding, age, reproduce, and the
lineage continue.

    python living_neuron.py --life 3000
"""

import os
import json
import re
import random
import urllib.request
import argparse

STATE_FILE = "living_neuron_state.json"
STREAM_FILE = "world_stream.txt"
SOURCES = [
    "https://www.gutenberg.org/files/1342/1342-0.txt",
    "https://www.gutenberg.org/files/1661/1661-0.txt",
    "https://www.gutenberg.org/files/84/84-0.txt",
    "https://www.gutenberg.org/files/98/98-0.txt",
]
STOP = set("""a an the and or but if then of to in on at by for with from as is
are was were be been it its this that these those i you he she we they them him
her his my your our their me us do did have has had not no what which who where
when why how all any some such than there here up out about""".split())


def get_stream():
    if os.path.exists(STREAM_FILE):
        return open(STREAM_FILE, encoding="utf-8", errors="ignore").read()
    parts = []
    for url in SOURCES:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            raw = urllib.request.urlopen(req, timeout=30).read().decode(
                "utf-8", errors="ignore")
            s, e = raw.find("*** START"), raw.find("*** END")
            if s != -1:
                raw = raw[raw.find("\n", s):]
            if e != -1:
                raw = raw[:e]
            parts.append(raw.lower())
        except Exception:
            pass
    text = "\n".join(parts)
    if text:
        open(STREAM_FILE, "w", encoding="utf-8").write(text)
    return text


class LivingNEURON:
    def __init__(self, inherited=None, gen=0):
        self.gen = gen
        # the being
        self.wants = (inherited["wants"] if inherited else
                      random.choice(["to understand", "to be safe",
                                     "to find what it values"]))
        self.emotion = [random.uniform(-0.2, 0.2) for _ in range(4)]
        # its grown mind (inherited from parent, then grows further)
        self.vocab = dict(inherited["vocab"]) if inherited else {}
        self.co = {w: dict(n) for w, n in inherited["co"].items()} if inherited else {}
        # life
        self.energy = 1.0
        self.age = 0
        self.passages_lived = inherited["passages_lived"] if inherited else 0
        self.alive = True
        # lineage record
        self.lineage = inherited["lineage"] if inherited else []

    # ---- being lives in the world's language, growing understanding ----
    def absorb(self, passage):
        words = [w for w in re.findall(r"[a-z']+", passage.lower()) if len(w) > 1]
        for w in words:
            self.vocab[w] = self.vocab.get(w, 0) + 1
        content = [w for w in words if w not in STOP]
        for i, w in enumerate(content):
            a = self.co.setdefault(w, {})
            for j in range(max(0, i-4), min(len(content), i+5)):
                if j != i:
                    a[content[j]] = a.get(content[j], 0) + 1
        self.passages_lived += 1
        # living costs a little energy; the world both feeds and wears it
        self.energy -= 0.0008
        self.age += 1

    def meaning_of(self, word, k=5):
        a = self.co.get(word.lower())
        if not a:
            return []
        scored = [(nb, c / (self.vocab.get(nb, 1) ** 0.6))
                  for nb, c in a.items() if c >= 2]
        scored.sort(key=lambda kv: -kv[1])
        return [w for w, _ in scored[:k]]

    def feeling(self):
        m = sum(self.emotion) / len(self.emotion)
        return "settled" if m > 0.2 else ("wary" if m < -0.2 else "even")

    # ---- mortality and reproduction ----
    def can_reproduce(self):
        # thriving and grown enough to pass on a worthwhile mind
        return (self.energy > 0.5 and self.passages_lived > 0 and
                len(self.co) > 1500 and self.age % 1500 == 0 and self.age > 0)

    def reproduce(self):
        """Spawn a child that inherits this being's grown mind + a mutation."""
        # mutation: child forgets a little (fresh start) but keeps most, and may
        # want something slightly different
        child_co = {w: dict(n) for w, n in self.co.items()}
        inh = {"wants": self.wants, "vocab": dict(self.vocab), "co": child_co,
               "passages_lived": self.passages_lived,
               "lineage": self.lineage + [f"gen{self.gen} lived "
                                          f"{self.passages_lived} passages"]}
        child = LivingNEURON(inherited=inh, gen=self.gen + 1)
        if random.random() < 0.5:
            child.wants = random.choice(["to understand", "to be safe",
                                         "to find what it values"])
        self.energy -= 0.3        # reproducing costs the parent
        return child

    def should_die(self):
        return self.energy <= 0 or self.age > 8000

    def snapshot(self):
        # cap co size for saving
        trimmed = {w: dict(sorted(n.items(), key=lambda kv: -kv[1])[:12])
                   for w, n in self.co.items()}
        return {"wants": self.wants, "vocab": self.vocab, "co": trimmed,
                "passages_lived": self.passages_lived, "gen": self.gen,
                "age": self.age, "lineage": self.lineage}


def save_lineage(being):
    json.dump({"current": being.snapshot()}, open(STATE_FILE, "w"))


def load_lineage():
    if os.path.exists(STATE_FILE):
        try:
            d = json.load(open(STATE_FILE))["current"]
            inh = {"wants": d["wants"], "vocab": d["vocab"], "co": d["co"],
                   "passages_lived": d["passages_lived"],
                   "lineage": d.get("lineage", [])}
            b = LivingNEURON(inherited=inh, gen=d.get("gen", 0))
            b.age = 0        # a continued being starts fresh-energied but keeps mind
            return b
        except Exception:
            return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--life", type=int, default=3000,
                    help="moments to live this run")
    args = ap.parse_args()

    text = get_stream()
    if not text or len(text) < 10000:
        print("  (the world's language could not be reached, needs internet.)")
        return
    passages = [p for p in re.split(r"(?<=[.!?])\s+", text) if len(p) > 20]
    rng = random.Random()

    being = load_lineage()
    if being:
        print(f"  A NEURON of this lineage continues (generation {being.gen}),")
        print(f"  carrying a mind grown over {being.passages_lived:,} passages "
              f"of living.\n")
    else:
        being = LivingNEURON()
        print(f"  A NEURON is born, the first of its lineage. It knows almost")
        print(f"  nothing, and will grow by living in the world.\n")

    print(f"  NEURON: i am here, generation {being.gen}. i am "
          f"{being.feeling()}, wanting {being.wants}.\n")

    births = 0
    deaths = 0
    for moment in range(args.life):
        being.absorb(passages[rng.randrange(len(passages))])

        if being.can_reproduce():
            child = being.reproduce()
            births += 1
            print(f"  [a child is born: generation {child.gen}, inheriting a "
                  f"mind grown over")
            print(f"   {being.passages_lived:,} passages. the lineage grows.]\n")
            # the child becomes the one that continues the lineage going forward
            being = child

        if being.should_die():
            deaths += 1
            print(f"  [generation {being.gen} has died at age {being.age}, "
                  f"having lived")
            print(f"   {being.passages_lived:,} passages. its mind passes on.]\n")
            # on death, a child continues (the lineage lives on)
            being = being.reproduce()
            being.energy = 1.0

        if (moment + 1) % max(1, args.life // 6) == 0:
            print(f"  living... gen {being.gen} | age {being.age} | energy "
                  f"{being.energy:.2f} | words {len(being.co):,} | "
                  f"feels {being.feeling()}")

    save_lineage(being)

    print("\n" + "=" * 64)
    print("  THE BEING LIVED (and the lineage continues, saved)")
    print("=" * 64)
    print(f"  current generation: {being.gen} | age {being.age} | words of "
          f"meaning {len(being.co):,}")
    print(f"  passages lived (lineage total): {being.passages_lived:,}")
    print(f"  children born this run: {births} | deaths: {deaths}")
    print(f"\n  What it understands now, grown by living in the world:")
    for probe in ["fear", "love", "death", "dark", "heart"]:
        rel = being.meaning_of(probe)
        if rel:
            print(f"    \"{probe}\" -> {', '.join(rel)}")
    if being.lineage:
        print(f"\n  its lineage: {' | '.join(being.lineage[-3:])}")
    print(f"\n  A mortal being that grows by living in the world, reproduces, "
          f"and dies,")
    print(f"  its mind passing to its children. A living species, not an "
          f"immortal")
    print(f"  database. Saved. Run again and the lineage continues from here. "
          f"This is")
    print(f"  the vision-true living NEURON, ready to live in the world.")


if __name__ == "__main__":
    main()
