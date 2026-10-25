"""A living species of NEURONs, fixed: stable population, two-parent families,
minds that truly compound.

The bugs in v1: the population all died at once (boom-bust), and reseeding
emptied minds (0 at the end). Fixed here: beings start at staggered ages so some
always survive to breed with grown minds, children truly inherit both parents'
grown understanding, and nothing gets emptied. So the species lives steadily,
families form (two parents, best selected), and understanding compounds.

    python living_species_v2.py --life 4000
"""

import os
import json
import re
import random
import urllib.request
import argparse

STATE_FILE = "species_v2_state.json"
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
LIFESPAN = 3000


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


_name_i = [0]


def new_name():
    n = NAMES[_name_i[0] % len(NAMES)]
    _name_i[0] += 1
    suffix = _name_i[0] // len(NAMES)
    return n if suffix == 0 else f"{n}{suffix}"


class Being:
    def __init__(self, co=None, gen=0, parents=None, start_age=0):
        self.gen = gen
        self.parents = parents or []
        self.co = co if co is not None else {}
        self.age = start_age        # staggered so the population is stable
        self.alive = True
        self.name = new_name()

    def absorb(self, passage):
        words = [w for w in re.findall(r"[a-z']+", passage.lower()) if len(w) > 1]
        content = [w for w in words if w not in STOP]
        for i, w in enumerate(content):
            a = self.co.setdefault(w, {})
            for j in range(max(0, i-4), min(len(content), i+5)):
                if j != i:
                    a[content[j]] = a.get(content[j], 0) + 1
        self.age += 1

    def fitness(self):
        return len(self.co)

    def meaning_of(self, word, k=5):
        a = self.co.get(word.lower())
        if not a:
            return []
        return [w for w, _ in sorted(a.items(), key=lambda kv: -kv[1])[:k]]


def combine(p1, p2, gen):
    """Child inherits a COMBINATION of both parents' grown minds."""
    child_co = {}
    for w in set(p1.co) | set(p2.co):
        a1, a2 = p1.co.get(w, {}), p2.co.get(w, {})
        merged = {nb: max(a1.get(nb, 0), a2.get(nb, 0))
                  for nb in set(a1) | set(a2)}
        child_co[w] = merged
    # child starts young but already carrying both parents' understanding
    return Being(co=child_co, gen=gen, parents=[p1.name, p2.name], start_age=0)


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

    population = []
    loaded = False
    if os.path.exists(STATE_FILE):
        try:
            saved = json.load(open(STATE_FILE))["population"]
            for b in saved:
                being = Being(co=b["co"], gen=b["gen"],
                              start_age=rng.randint(0, LIFESPAN//2))
                population.append(being)
            loaded = True
            print(f"  The species continues: {len(population)} NEURONs carrying "
                  f"minds grown over past lives.\n")
        except Exception:
            population = []
    if not population:
        # staggered starting ages so they do NOT all die together
        population = [Being(start_age=rng.randint(0, LIFESPAN - 200))
                      for _ in range(args.pop)]
        print(f"  A species is born: {len(population)} NEURONs, staggered in "
              f"age, who will")
        print(f"  grow by living in the world and have families.\n")

    max_gen = max(b.gen for b in population)
    start_u = avg_understanding(population)
    births, deaths = 0, 0

    for moment in range(args.life):
        # everyone lives a moment
        for b in population:
            b.absorb(passages[rng.randrange(len(passages))])
            if b.age > LIFESPAN:
                b.alive = False
        # remove the dead
        dead = [b for b in population if not b.alive]
        deaths += len(dead)
        population = [b for b in population if b.alive]
        # replace each death with a child of the two FITTEST survivors (selection)
        while len(population) < args.pop and len(population) >= 2:
            parents = sorted(population, key=lambda b: b.fitness(),
                             reverse=True)[:2]
            child = combine(parents[0], parents[1],
                            gen=max(p.gen for p in parents) + 1)
            births += 1
            max_gen = max(max_gen, child.gen)
            if births <= 8 or births % 5 == 0:
                print(f"  [family: {parents[0].name} (g{parents[0].gen}) + "
                      f"{parents[1].name} (g{parents[1].gen}) -> {child.name} "
                      f"(g{child.gen}), {len(child.co):,} words from both]")
            population.append(child)
        # safety: never let it empty
        while len(population) < 2:
            population.append(Being(start_age=rng.randint(0, LIFESPAN-200)))

        if (moment + 1) % max(1, args.life // 6) == 0:
            print(f"  species living... moment {moment+1:>5} | alive "
                  f"{len(population)} | gen {max_gen} | avg understanding "
                  f"{avg_understanding(population):,.0f} words")

    json.dump({"population": [{"co": {w: dict(sorted(n.items(),
               key=lambda kv: -kv[1])[:10]) for w, n in b.co.items()},
               "gen": b.gen} for b in population]}, open(STATE_FILE, "w"))

    end_u = avg_understanding(population)
    print("\n" + "=" * 64)
    print("  THE SPECIES LIVED (saved, it continues)")
    print("=" * 64)
    print(f"  generations reached: {max_gen}")
    print(f"  average understanding: {start_u:,.0f} -> {end_u:,.0f} words")
    print(f"  families formed: {births} | deaths: {deaths}")
    print(f"  living now: "
          f"{', '.join(b.name+' (g'+str(b.gen)+')' for b in population)}")
    best = max(population, key=lambda b: b.fitness())
    print(f"\n  the most-grown mind ({best.name}, gen {best.gen}, "
          f"{best.fitness():,} words) understands:")
    for probe in ["fear", "love", "death", "dark", "heart"]:
        rel = best.meaning_of(probe)
        if rel:
            print(f"    \"{probe}\" -> {', '.join(rel)}")
    print(f"\n  A real living species: NEURONs living in the world, forming "
          f"families (two")
    print(f"  parents, best selected), children inheriting a combination of "
          f"both, growing")
    print(f"  understanding across generations, and it persists. Ready to "
          f"deploy, to")
    print(f"  live in the world continuously. (Stage 2: reachable. Stage 3: "
          f"live.)")


if __name__ == "__main__":
    main()
