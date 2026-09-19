"""Why does the store miss 40% of the time with only a thousand items?

Store growth showed retrieval decaying from 60% at 1,000 items to 28% at
200,000, and flattening. The dilution is real but it is not the main
problem: 60% AT A THOUSAND ITEMS means the retriever is mediocre before
scale is a factor at all. Fixing the base case lifts every size at once.

Three candidate causes, tested against the same probes:

  THE EMBEDDING MODEL IS SMALL. all-MiniLM-L6-v2 is the fast one, 384
      dimensions. A larger model retrieves better and costs milliseconds.

  ONLY FOUR CANDIDATES ARE CONSIDERED. Retrieving twenty and reranking
      them properly is standard, and if the source sentence is usually in
      the top twenty but not the top four, that is the whole gap.

  SENTENCES STAND ALONE. This is the one I expect to matter most.
      Wikipedia prose is full of pronouns: "It was founded in 1892" is
      useless without knowing what "it" is. The store currently throws
      away everything around a sentence, so a question naming the subject
      cannot match a sentence that never mentions it.

A DIAGNOSTIC RUNS FIRST and it decides which of these is worth pursuing:
how often is the source sentence in the top 20 but not the top 4? If that
number is large, reranking is the fix. If the source is not in the top 20
either, the embedding or the missing context is the problem, and reranking
cannot help.

Arms:
  baseline        MiniLM, top 4, sentences alone. What the system does now.
  bigger-model    a larger embedding model, otherwise identical.
  with-context    each stored item is the sentence PLUS its neighbours, so
                  pronouns have something to refer to.
  rerank          retrieve 20, rescore, keep 4.
  context+bigger  the two most promising together.

Everything is measured on the SAME probes at the SAME store size, so the
only variable is the retrieval strategy.
"""

import json
import os
import re
import random
import sys
import time

import torch

CORPUS_CACHE = "growth_corpus.json"
STORE_SIZE = 20000
N_PROBES = 120
SMALL_MODEL = "all-MiniLM-L6-v2"
BIG_MODEL = "all-mpnet-base-v2"
CONTEXT_WINDOW = 1          # neighbours on each side


def fetch_corpus(n):
    if os.path.exists(CORPUS_CACHE):
        with open(CORPUS_CACHE) as f:
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
    out = out[:n]
    with open(CORPUS_CACHE, "w") as f:
        json.dump(out, f)
    return out


def make_probes(corpus, n_probes, seed=0):
    """Questions whose answer is one specific sentence."""
    rng = random.Random(seed)
    probes = []
    pool = list(range(min(2000, len(corpus))))
    rng.shuffle(pool)
    for i in pool:
        s = corpus[i]
        terms = re.findall(r"\b[A-Z][a-z]{5,}\b", s)
        nums = re.findall(r"\b\d[\d,\.]{2,}\b", s)
        if not terms:
            continue
        q = (f"What is said about {terms[0]}"
             + (f" and {nums[0]}?" if nums else "?"))
        probes.append(dict(q=q, source=s, index=i))
        if len(probes) >= n_probes:
            break
    return probes


class Retriever:
    """One retrieval strategy, built so the arms differ in exactly one way.

    `with_context` changes what is STORED — each item becomes a sentence
    plus its neighbours — while the probe still checks whether the original
    sentence is present in what came back. That is the fair comparison:
    the question is whether the right MATERIAL is retrieved, not whether
    the strings match exactly.
    """

    def __init__(self, corpus, model_name=SMALL_MODEL, with_context=False,
                 rerank=False):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.rerank = rerank
        self.with_context = with_context
        self.sources = list(corpus)

        if with_context:
            self.items = []
            for i in range(len(corpus)):
                lo = max(0, i - CONTEXT_WINDOW)
                hi = min(len(corpus), i + CONTEXT_WINDOW + 1)
                self.items.append(" ".join(corpus[lo:hi]))
        else:
            self.items = list(corpus)

        self.vectors = self.model.encode(
            self.items, batch_size=256, convert_to_tensor=True,
            normalize_embeddings=True, show_progress_bar=False)

    def search(self, query, k=4):
        q = self.model.encode([query], convert_to_tensor=True,
                              normalize_embeddings=True)
        sims = torch.mm(q, self.vectors.T).squeeze(0)

        if self.rerank:
            # take a wide net, then rescore by how much of the question's
            # vocabulary each candidate actually contains — a cheap check
            # the embedding does not do directly
            wide = torch.topk(sims, min(20, len(self.items))).indices
            qt = set(re.findall(r"[a-z0-9]+", query.lower()))
            scored = []
            for i in wide:
                i = int(i)
                it = set(re.findall(r"[a-z0-9]+", self.items[i].lower()))
                overlap = len(qt & it) / max(1, len(qt))
                scored.append((float(sims[i]) + overlap, i))
            scored.sort(reverse=True)
            idx = [i for _, i in scored[:k]]
        else:
            idx = [int(i) for i in
                   torch.topk(sims, min(k, len(self.items))).indices]

        return [self.items[i] for i in idx]

    def rank_of_source(self, query, source, depth=20):
        """Where does the source appear in the ranking, if at all?

        This is the diagnostic: if the source is at rank 8, reranking can
        reach it. If it is not in the top 20 at all, nothing downstream of
        the embedding can help.
        """
        q = self.model.encode([query], convert_to_tensor=True,
                              normalize_embeddings=True)
        sims = torch.mm(q, self.vectors.T).squeeze(0)
        top = torch.topk(sims, min(depth, len(self.items))).indices
        for rank, i in enumerate(top):
            if source in self.items[int(i)]:
                return rank + 1
        return None


def hit_rate(retriever, probes, k=4):
    hits = 0
    t0 = time.perf_counter()
    for p in probes:
        results = retriever.search(p["q"], k)
        if any(p["source"] in r for r in results):
            hits += 1
    ms = 1000.0 * (time.perf_counter() - t0) / len(probes)
    return 100.0 * hits / len(probes), ms


if __name__ == "__main__":
    print("Why does the store miss 40% with only a thousand items?\n")
    corpus = fetch_corpus(STORE_SIZE)
    probes = make_probes(corpus, N_PROBES)
    print(f"{len(corpus)} sentences, {len(probes)} probes\n")

    # The diagnostic first: it decides which fix is even possible.
    print("diagnostic: where does the source sentence rank?", flush=True)
    base = Retriever(corpus)
    ranks = [base.rank_of_source(p["q"], p["source"]) for p in probes]
    in_4 = sum(1 for r in ranks if r and r <= 4)
    in_20 = sum(1 for r in ranks if r)
    missing = len(ranks) - in_20
    print(f"  in the top 4      : {in_4:>4} ({100.0 * in_4 / len(ranks):.1f}%)")
    print(f"  in the top 20     : {in_20:>4} "
          f"({100.0 * in_20 / len(ranks):.1f}%)")
    print(f"  not in the top 20 : {missing:>4} "
          f"({100.0 * missing / len(ranks):.1f}%)")
    reachable = in_20 - in_4
    print(f"\n  reachable by reranking alone: {reachable} probes "
          f"({100.0 * reachable / len(ranks):.1f}%)")
    if missing > reachable:
        print("  -> most misses are NOT in the top 20, so reranking cannot")
        print("     fix them. The embedding or the missing context is the")
        print("     problem.")
    else:
        print("  -> most misses ARE in the top 20, so reranking is the fix.")
    del base
    print()

    arms = [
        ("baseline", dict(model_name=SMALL_MODEL)),
        ("rerank", dict(model_name=SMALL_MODEL, rerank=True)),
        ("with-context", dict(model_name=SMALL_MODEL, with_context=True)),
        ("bigger-model", dict(model_name=BIG_MODEL)),
        ("context+bigger", dict(model_name=BIG_MODEL, with_context=True)),
    ]

    results = {}
    for name, kwargs in arms:
        print(f"--- {name} ---", flush=True)
        t0 = time.time()
        r = Retriever(corpus, **kwargs)
        build = time.time() - t0
        hit, ms = hit_rate(r, probes)
        results[name] = dict(hit=hit, query_ms=ms, build_s=build)
        print(f"  hit {hit:5.1f}%   query {ms:6.2f}ms   "
              f"build {build:6.1f}s", flush=True)
        del r
        import gc
        gc.collect()

    print("\n" + "=" * 70)
    print(f"{'arm':>16} {'hit rate':>10} {'vs baseline':>13} "
          f"{'query ms':>10}")
    print("-" * 70)
    b = results["baseline"]["hit"]
    for name, _ in arms:
        r = results[name]
        print(f"{name:>16} {r['hit']:>9.1f}% {r['hit'] - b:>+12.1f} "
              f"{r['query_ms']:>10.2f}")
    print("=" * 70)

    best = max(results, key=lambda n: results[n]["hit"])
    print(f"\n  best: {best} at {results[best]['hit']:.1f}%, "
          f"{results[best]['hit'] - b:+.1f} over baseline")

    print("""
The diagnostic decides what the fix is, and the arms confirm it.

  CONTEXT HELPS MOST -> the misses were pronouns and references. Wikipedia
      prose says "It was founded in 1892" and the store held that sentence
      alone, with nothing saying what "it" is. Storing neighbours fixes
      that, costs only disk, and would lift every store size at once.

  BIGGER MODEL HELPS MOST -> the small embedding was the limit and the fix
      is a few milliseconds per query. The cheapest possible outcome.

  RERANK HELPS MOST -> the right sentence was being found and then thrown
      away by taking only four candidates.

  NOTHING HELPS MUCH -> the probes may be unanswerable rather than the
      retriever weak. A question built from one capitalised word may
      genuinely match many sentences, in which case this measures probe
      quality and the real retrieval is better than it looks.
""")
    with open("retrievalquality.json", "w") as f:
        json.dump(dict(results=results,
                       diagnostic=dict(in_4=in_4, in_20=in_20,
                                       missing=missing)), f, indent=2)
    print("wrote retrievalquality.json")