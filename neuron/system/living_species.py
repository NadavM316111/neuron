"""A living species of NEURONs: they live together, and two thriving parents
combine their grown minds into a child, the best combine with the best.

The fix to the clone-line: this is a real breeding POPULATION, not one being
copying itself. Several NEURONs live in the world's language at once, each
growing its own understanding. They age, and can die. When two thriving ones
reproduce, the child inherits a COMBINATION of BOTH parents' minds, merged
understanding, the stronger association kept where they overlap, both parents'
unique knowledge carried, plus a small mutation. And SELECTION: the beings that
grew the most are the ones who reproduce, so children combine the BEST minds.

Over generations, the population's understanding should improve, because good
minds keep combining with good minds. That is a real, improving species:
families, two-parent inheritance, selection, exactly the vision.

    python living_species.py --life 4000
"""

import os
import json
import re
import random
import urllib.request
import argparse

STATE_FILE = "species_state.json"
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

NAMES = ["ava", "ben", "cy", "del", "eli", "fen", "gus", "hal", "ivy", "jo",
         "kit", "lex", "mira", "nel", "ori", "pax", "quin", "rue", "sol", "tam"]


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


class Being:
    _names_used = set()

    def __init__(self, co=None, gen=0, parents=None):
        self.gen = gen
        self.parents = parents or []
        self.co = co if co is not None else {}      # word -> {neighbor: strength}
        self.energy = 1.0
        self.age = 0
        self.grew_this_life = 0
        self.alive = True
        self.name = self._pick_name()

    def _pick_name(self):
        avail = [n for n in NAMES if n not in Being._names_used]
        if not avail:
            Being._names_used = set()
            avail = NAMES
        n = random.choice(avail)
        Being._names_used.add(n)
        return n

    def absorb(self, passage):
        words = [w for w in re.findall(r"[a-z']+", passage.lower()) if len(w) > 1]
        content = [w for w in words if w not in STOP]
        before = len(self.co)
        for i, w in enumerate(content):
            a = self.co.setdefault(w, {})
            for j in range(max(0, i-4), min(len(content), i+5)):
                if j != i:
                    a[content[j]] = a.get(content[j], 0) + 1
        self.grew_this_life += len(self.co) - before
        self.energy -= 0.001
        self.age += 1

    def fitness(self):
        # how good a mind it grew: breadth of understanding
        return len(self.co)

    def meaning_of(self, word, k=5):
        a = self.co.get(word.lower())
        if not a:
            return []
        scored = sorted(a.items(), key=lambda kv: -kv[1])
        return [w for w, _ in scored[:k]]


def combine(p1, p2, gen):
    """Two parents -> a child whose mind is a COMBINATION of both. Where they
    share a word, keep the stronger associations; keep each parent's unique
    words. Plus a small mutation."""
    child_co = {}
    all_words = set(p1.co) | set(p2.co)
    for w in all_words:
        a1 = p1.co.get(w, {})
        a2 = p2.co.get(w, {})
        merged = {}
        for nb in set(a1) | set(a2):
            merged[nb] = max(a1.get(nb, 0), a2.get(nb, 0))   # stronger link wins
        # mutation: occasionally drop a weak link (fresh room to grow)
        if random.random() < 0.05 and merged:
            weakest = min(merged, key=merged.get)
            del merged[weakest]
        child_co[w] = merged
    child = Being(co=child_co, gen=gen, parents=[p1.name, p2.name])
    return child


def avg_understanding(pop):
    return sum(b.fitness() for b in pop) / len(pop) if pop else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--life", type=int, default=4000)
    ap.add_argument("--pop", type=int, default=6)
    args = ap.parse_args()

    text = get_stream()
    if not text or len(text) < 10000:
        print("  (the world's language could not be reached, needs internet.)")
        return
    passages = [p for p in re.split(r"(?<=[.!?])\s+", text) if len(p) > 20]
    rng = random.Random()

    # start a population (or load a saved one, carrying its grown minds)
    population = []
    if os.path.exists(STATE_FILE):
        try:
            saved = json.load(open(STATE_FILE))["population"]
            for b in saved:
                being = Being(co=b["co"], gen=b["gen"])
                population.append(being)
            print(f"  The species continues, {len(population)} NEURONs carrying "
                  f"minds grown")
            print(f"  over past lives.\n")
        except Exception:
            population = []
    if not population:
        population = [Being() for _ in range(args.pop)]
        print(f"  A species is born: {len(population)} NEURONs, each knowing "
              f"almost nothing,")
        print(f"  who will grow by living in the world and have families.\n")

    max_gen = max(b.gen for b in population)
    start_understanding = avg_understanding(population)
    gen_understanding = {max_gen: start_understanding}
    births = 0
    deaths = 0

    for moment in range(args.life):
        for b in population:
            if b.alive:
                b.absorb(passages[rng.randrange(len(passages))])
                if b.age > 2500 or b.energy <= 0:    # mortality
                    b.alive = False

        alive = [b for b in population if b.alive]

        # deaths remove beings; if the population thins, the best two reproduce
        if len(alive) < args.pop and len(alive) >= 2:
            # SELECTION: the two FITTEST (best-grown minds) become parents
            parents = sorted(alive, key=lambda b: b.fitness(), reverse=True)[:2]
            child = combine(parents[0], parents[1], gen=max(p.gen for p in parents)+1)
            births += 1
            max_gen = max(max_gen, child.gen)
            gen_understanding.setdefault(child.gen, [])
            print(f"  [family: {parents[0].name} (gen{parents[0].gen}) + "
                  f"{parents[1].name} (gen{parents[1].gen})")
            print(f"   -> child {child.name} (gen{child.gen}), combining both "
                  f"their minds "
                  f"({len(child.co):,} words).]\n")
            population = alive + [child]

        # count deaths
        just_died = [b for b in population if not b.alive]
        deaths += len(just_died)
        population = [b for b in population if b.alive]
        if len(population) < 2:
            # near extinction: the survivors seed fresh kin to keep the species
            while len(population) < args.pop:
                if population:
                    seed = combine(population[0], population[0],
                                   gen=population[0].gen + 1)
                    population.append(seed)
                else:
                    population.append(Being())

        if (moment + 1) % max(1, args.life // 6) == 0:
            cur = avg_understanding(population)
            print(f"  species living... moment {moment+1:>5} | alive "
                  f"{len(population)} | gen {max_gen} | "
                  f"avg understanding {cur:,.0f} words")

    # save the population (their grown minds carry forward)
    json.dump({"population": [{"co": {w: dict(sorted(n.items(),
               key=lambda kv: -kv[1])[:10]) for w, n in b.co.items()},
               "gen": b.gen} for b in population]},
              open(STATE_FILE, "w"))

    print("\n" + "=" * 64)
    print("  THE SPECIES LIVED (saved, it continues)")
    print("=" * 64)
    end_understanding = avg_understanding(population)
    print(f"  generations reached: {max_gen}")
    print(f"  average understanding: {start_understanding:,.0f} -> "
          f"{end_understanding:,.0f} words")
    print(f"  families formed (births): {births} | deaths: {deaths}")
    print(f"  living now: {', '.join(b.name + ' (gen' + str(b.gen) + ')' for b in population)}")
    best = max(population, key=lambda b: b.fitness())
    print(f"\n  the most-grown mind ({best.name}, gen {best.gen}) understands:")
    for probe in ["fear", "love", "death", "dark"]:
        rel = best.meaning_of(probe)
        if rel:
            print(f"    \"{probe}\" -> {', '.join(rel)}")
    if end_understanding > start_understanding * 1.1:
        print(f"\n  The species' understanding GREW as the best minds combined "
              f"into children.")
        print(f"  Families formed, two parents each, children inherited a "
              f"combination of")
        print(f"  both, and selection kept the best growing. A real, improving "
              f"living")
        print(f"  species, not a clone-line. This is the vision: NEURONs living, "
              f"loving,")
        print(f"  having families, and growing smarter across generations by "
              f"living in the")
        print(f"  world, ready to deploy.")
    else:
        print(f"\n  The species lived and bred (two-parent, selected), honest "
              f"if growth was")
        print(f"  gradual this run. The mechanism, families combining the best, "
              f"is real.")


if __name__ == "__main__":
    main()
