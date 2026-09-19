"""Where does the store break as it grows?

The system now persists, so it can run for weeks. That makes a question
urgent that has never been asked: the store grows without bound and nothing
is ever evicted. Forty thousand items worked. A system living alongside
someone for two years would hold millions.

Three things could fail, and they fail differently:

  RETRIEVAL QUALITY. More items means more near-misses. At 58 sentences
      word overlap retrieved perfectly; at 40,000 it found almost nothing,
      which is exactly this failure at a scale that happened to be
      reachable. Embeddings fixed it there — but "fixed at 40,000" is not
      "fixed forever", and nobody has looked further.

  SEARCH TIME. Every query compares against every item. That is fine at
      thousands and a problem at millions, and the crossover matters
      because a system that takes a second to answer is a different
      product from one that takes a millisecond.

  MEMORY. Every sentence is held in RAM along with its vector. A laptop
      has a fixed amount, and the point at which this stops fitting is a
      hard limit on how long the system can run.

MEASURED AT EACH SIZE, from a thousand items to as many as fit:

  hit rate      does the store return the SOURCE sentence for a question
                drawn from it? This is retrieval quality with the language
                model taken out of the loop entirely — the model cannot
                answer from a sentence it never receives.
  query time    milliseconds per search.
  memory        megabytes held.
  build time    seconds to encode, which is what a real system pays as it
                reads.

The probes are held constant across sizes: the SAME questions are asked of
a 1,000-item store and a 200,000-item store. So any fall in hit rate is
dilution, not a different test.

No language model is needed. This measures the store alone, which is the
part that has to survive years.
"""

import gc
import json
import os
import random
import re
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "..", "..", "core"))

from neuron_system import Store


SIZES = [1000, 5000, 20000, 50000, 100000, 200000]
N_PROBES = 60
TARGET = max(SIZES)
CACHE = "growth_corpus.json"


def fetch_corpus(n):
    """Real Wikipedia sentences. Cached, since encoding is the slow part
    and refetching would add minutes for nothing."""
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            data = json.load(f)
        if len(data) >= n:
            return data[:n]

    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.en",
                      split="train", streaming=True)
    out = []
    for article in ds:
        if article["title"].startswith("List of"):
            continue
        for s in re.split(r"(?<=[.!?])\s+", article["text"][:4000]):
            s = s.strip()
            if 60 < len(s) < 250 and not s.startswith("="):
                out.append(s)
        if len(out) >= n:
            break
        if len(out) % 20000 < 30:
            print(f"  {len(out)} sentences...", flush=True)
    out = out[:n]
    with open(CACHE, "w") as f:
        json.dump(out, f)
    return out


def make_probes(corpus, n_probes, seed=0):
    """Questions whose answer is one specific sentence.

    Drawn from the FIRST thousand sentences so they are present in every
    store size — otherwise a large store would be tested on material a
    small one never had, and the comparison would be meaningless.
    """
    rng = random.Random(seed)
    probes = []
    pool = list(range(min(1000, len(corpus))))
    rng.shuffle(pool)
    for i in pool:
        s = corpus[i]
        terms = [w for w in re.findall(r"\b[A-Z][a-z]{5,}\b", s)]
        nums = re.findall(r"\b\d[\d,\.]{2,}\b", s)
        if not terms:
            continue
        q = (f"What is said about {terms[0]}"
             + (f" and {nums[0]}?" if nums else "?"))
        probes.append(dict(q=q, source=s))
        if len(probes) >= n_probes:
            break
    return probes


def measure(store, probes, k=4):
    """Hit rate and query time. The model is not involved: a sentence the
    store never returns cannot be used to answer anything."""
    hits = 0
    t0 = time.perf_counter()
    for p in probes:
        results = store.search(p["q"], k)
        if any(r == p["source"] for r in results):
            hits += 1
    elapsed = time.perf_counter() - t0
    return (100.0 * hits / len(probes),
            1000.0 * elapsed / len(probes))


def memory_mb(store):
    """Roughly what the store holds in RAM."""
    text = sum(len(s.encode()) for s in store.sentences) / 1e6
    vec = 0.0
    if store.vectors is not None:
        vec = store.vectors.element_size() * store.vectors.nelement() / 1e6
    return text, vec


if __name__ == "__main__":
    print("Where does the store break as it grows?\n")
    print(f"sizes: {SIZES}")
    print(f"{N_PROBES} probes, held constant across every size\n")

    corpus = fetch_corpus(TARGET)
    print(f"{len(corpus)} sentences available")
    if len(corpus) < SIZES[-1]:
        SIZES = [s for s in SIZES if s <= len(corpus)]
        print(f"  trimmed sizes to {SIZES}")

    probes = make_probes(corpus, N_PROBES)
    print(f"{len(probes)} probes built from the first 1,000 sentences, "
          f"so every store size contains them\n")

    results = []
    for size in SIZES:
        print(f"--- {size} items ---", flush=True)
        store = Store(use_embeddings=True)

        t0 = time.perf_counter()
        for s in corpus[:size]:
            store.add(s)
        store.reindex()
        build = time.perf_counter() - t0

        hit, query_ms = measure(store, probes)
        text_mb, vec_mb = memory_mb(store)

        results.append(dict(size=size, hit=hit, query_ms=query_ms,
                            build_s=build, text_mb=text_mb,
                            vec_mb=vec_mb))
        print(f"  hit {hit:5.1f}%   query {query_ms:7.2f}ms   "
              f"build {build:6.1f}s   memory {text_mb + vec_mb:7.1f}MB",
              flush=True)

        del store
        gc.collect()

    print("\n" + "=" * 82)
    print(f"{'items':>9} {'hit rate':>10} {'query ms':>10} "
          f"{'build s':>9} {'text MB':>9} {'vectors MB':>12}")
    print("-" * 82)
    for r in results:
        print(f"{r['size']:>9} {r['hit']:>9.1f}% {r['query_ms']:>10.2f} "
              f"{r['build_s']:>9.1f} {r['text_mb']:>9.1f} "
              f"{r['vec_mb']:>12.1f}")
    print("=" * 82)

    first, last = results[0], results[-1]
    print(f"\nFROM {first['size']} TO {last['size']} ITEMS:")
    print(f"  hit rate  {first['hit']:.1f}% -> {last['hit']:.1f}%  "
          f"({last['hit'] - first['hit']:+.1f})")
    print(f"  query     {first['query_ms']:.2f}ms -> "
          f"{last['query_ms']:.2f}ms  "
          f"({last['query_ms'] / max(0.01, first['query_ms']):.1f}x)")
    print(f"  memory    {first['text_mb'] + first['vec_mb']:.1f}MB -> "
          f"{last['text_mb'] + last['vec_mb']:.1f}MB")

    # what a long life would cost, extrapolated from the largest size
    per_item_mb = (last["text_mb"] + last["vec_mb"]) / last["size"]
    per_item_ms = last["query_ms"] / last["size"]
    print(f"\nEXTRAPOLATED, from the largest size measured:")
    for label, n in [("a month at 5,000 a day", 150000),
                     ("a year at 5,000 a day", 1825000),
                     ("two years", 3650000)]:
        print(f"  {label:>24}: {n * per_item_mb:8.0f}MB, "
              f"{n * per_item_ms:8.1f}ms per query")

    print("""
Three numbers and three different failures.

  HIT RATE FALLING -> dilution. More items means more near-misses, and the
      right sentence stops surfacing. This is what killed word overlap at
      40,000 items. If embeddings fall too, the store needs structure —
      per-topic indexes, or summarisation — rather than a better similarity
      measure.

  QUERY TIME RISING LINEARLY -> every search compares against every item.
      Fine at thousands, a problem at millions. The fix is an approximate
      index, which is standard and not yet needed.

  MEMORY -> the hard limit. Everything is held in RAM. The extrapolation
      above says how long a system can run before that stops being true on
      an ordinary laptop.

The extrapolation is the point of this. A system meant to live alongside
someone for years has to survive its own memory growing, and until now
nobody had measured what that costs.
""")
    with open("storegrowth.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote storegrowth.json")