"""Retrieval, measured with probes that have one right answer.

Every retrieval number this week is suspect. The probes asked "what is said
about Aristotle?" and counted a hit only if one DESIGNATED sentence came
back. In twenty thousand Wikipedia sentences there are many about Aristotle,
so a correct retrieval of a different one was scored a miss.

The evidence that this is what happened: 47% of sources were not in the top
20 at all, and a BIGGER embedding model scored WORSE. A better retriever
surfacing more relevant sentences makes it more likely to miss an arbitrary
designated one. That is the signature of an ambiguous probe, not a weak
retriever.

So the 60%-to-28% decay curve from the growth run measures probe ambiguity
too: more items means more valid alternatives competing with the designated
answer.

THE FIX: a probe whose answer is genuinely unique. Each probe targets a
number that appears in EXACTLY ONE sentence in the whole corpus, verified by
counting. The question names the subject and asks for the number, and the
probe is satisfied if ANY retrieved sentence contains it. There is only one
sentence that can.

That makes a miss a real miss.

Arms, the same as before so the comparison carries over:
  baseline        MiniLM, top 4
  rerank          retrieve 20, rescore, keep 4
  with-context    sentences stored with their neighbours
  bigger-model    a larger embedding model
  context+bigger  both

If retrieval is much higher here than the 34% measured with ambiguous
probes, the store was never as weak as reported, and the growth curve needs
re-reading.
"""

import gc
import json
import os
import random
import re
import time
from collections import Counter

import torch

CORPUS_CACHE = "growth_corpus.json"
STORE_SIZE = 20000
N_PROBES = 100
SMALL_MODEL = "all-MiniLM-L6-v2"
BIG_MODEL = "all-mpnet-base-v2"
CONTEXT_WINDOW = 1


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


def numbers_in(text):
    return re.findall(r"\b\d[\d,\.]{2,}\b", text)


def make_fair_probes(corpus, n_probes, seed=0):
    """Probes whose answer appears in exactly one sentence in the corpus.

    The uniqueness is VERIFIED by counting across the whole corpus, not
    assumed. A probe is satisfied if any retrieved sentence contains the
    number, and only one sentence can.
    """
    counts = Counter()
    for s in corpus:
        for v in set(numbers_in(s)):
            counts[v] += 1

    rng = random.Random(seed)
    order = list(range(len(corpus)))
    rng.shuffle(order)

    probes = []
    for i in order:
        s = corpus[i]
        unique = [v for v in numbers_in(s) if counts[v] == 1]
        if not unique:
            continue
        subjects = re.findall(r"\b[A-Z][a-z]{4,}\b", s)
        if not subjects:
            continue
        value = unique[0]
        probes.append(dict(
            q=f"What number is given in connection with {subjects[0]}?",
            answer=value, source=s, index=i))
        if len(probes) >= n_probes:
            break
    return probes, counts


class Retriever:
    def __init__(self, corpus, model_name=SMALL_MODEL, with_context=False,
                 rerank=False):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.rerank = rerank

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

    def rank_of_answer(self, query, answer, depth=20):
        """Where does a sentence containing the unique answer rank?"""
        q = self.model.encode([query], convert_to_tensor=True,
                              normalize_embeddings=True)
        sims = torch.mm(q, self.vectors.T).squeeze(0)
        top = torch.topk(sims, min(depth, len(self.items))).indices
        for rank, i in enumerate(top):
            if answer in self.items[int(i)]:
                return rank + 1
        return None


def hit_rate(retriever, probes, k=4):
    """A hit is any retrieved sentence containing the unique answer."""
    hits = 0
    t0 = time.perf_counter()
    for p in probes:
        results = retriever.search(p["q"], k)
        if any(p["answer"] in r for r in results):
            hits += 1
    ms = 1000.0 * (time.perf_counter() - t0) / len(probes)
    return 100.0 * hits / len(probes), ms


if __name__ == "__main__":
    print("Retrieval, measured with probes that have ONE right answer.\n")
    corpus = fetch_corpus(STORE_SIZE)
    probes, counts = make_fair_probes(corpus, N_PROBES)
    print(f"{len(corpus)} sentences, {len(probes)} probes")

    # Verify the premise rather than assuming it. Every probe's answer must
    # appear in exactly one sentence, or the fix has not been applied.
    bad = [p for p in probes if counts[p["answer"]] != 1]
    print(f"uniqueness check: {len(bad)} probes with a non-unique answer")
    if bad:
        print("  ABORT: the probes are not unique after all.")
        raise SystemExit
    print("  good: every answer appears in exactly one sentence\n")

    print("example probes:")
    for p in probes[:3]:
        print(f"  Q: {p['q']}")
        print(f"     answer {p['answer']}, from: {p['source'][:80]}...")
    print()

    print("diagnostic: where does the answer rank?", flush=True)
    base = Retriever(corpus)
    ranks = [base.rank_of_answer(p["q"], p["answer"]) for p in probes]
    in_4 = sum(1 for r in ranks if r and r <= 4)
    in_20 = sum(1 for r in ranks if r)
    print(f"  in the top 4      : {in_4:>4} "
          f"({100.0 * in_4 / len(ranks):.1f}%)")
    print(f"  in the top 20     : {in_20:>4} "
          f"({100.0 * in_20 / len(ranks):.1f}%)")
    print(f"  not in the top 20 : {len(ranks) - in_20:>4} "
          f"({100.0 * (len(ranks) - in_20) / len(ranks):.1f}%)")
    del base
    gc.collect()
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
    print(f"\n  best: {best} at {results[best]['hit']:.1f}%")
    print(f"  ambiguous probes gave 34.2% for the same baseline "
          f"configuration")
    print(f"  difference: {b - 34.2:+.1f} points from fixing the "
          f"MEASUREMENT alone")

    print("""
What this settles.

  MUCH HIGHER THAN 34% -> the store was never as weak as reported. The
      earlier numbers measured probe ambiguity, and the growth curve from
      60% to 28% needs re-reading: more items meant more valid alternatives
      competing with an arbitrary designated answer, not worse retrieval.

  STILL AROUND 34% -> the retriever really is missing two thirds of the
      time, ambiguity was not the explanation, and the store needs work
      that none of these arms provide.

  A BIGGER MODEL NOW HELPS -> confirms the diagnosis from the other side.
      With ambiguous probes it scored WORSE, which no genuine improvement
      to retrieval should ever do.

That last one is the cleanest check available. A better retriever cannot
make an honest measurement worse.
""")
    with open("fairprobes.json", "w") as f:
        json.dump(dict(results=results,
                       diagnostic=dict(in_4=in_4, in_20=in_20,
                                       total=len(ranks))), f, indent=2)
    print("wrote fairprobes.json")