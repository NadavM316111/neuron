"""Does an embedding store scale where word overlap did not?

The previous run put 40,000 real Wikipedia sentences into a store searched by
word overlap weighted by rarity. It found almost nothing: 8.3% on early
probes, which with twelve probes is ONE correct answer. At 58 sentences the
same retriever worked perfectly. A hundredfold more content and it collapsed,
because too many sentences share common words.

TWO THINGS ARE FIXED HERE, and both had to be, because they were confounded.

  THE RETRIEVER. Sentences are encoded into vectors that capture meaning,
      and the nearest vector to the question is retrieved. Two sentences
      about the same thing land close together even with no words in common.

  THE PROBES. The old probe asked "what does the text say about X" and
      counted as correct if another word from the same sentence appeared.
      With 40,000 sentences many mention X, so even perfect retrieval could
      fail. Probes now target a specific VALUE — a number or year — that is
      RARE in the corpus, so there is one right answer rather than many.

Fixing only the retriever would have left the result unreadable: a failure
could still have been the probe's fault.

A THIRD FIX, from a crash: bigstream.py wrote its corpus AND its results to
the same filename, so the results overwrote the corpus and this script tried
to read a results file as a stream. The corpus now has its own name and the
loader checks the shape rather than trusting it.

Three arms, all on identical probes so the comparison is direct:
  frozen        no store. The floor.
  store-words   the previous retriever.
  store-embed   the embedding retriever.
"""

import json
import math
import os
import re
import time
from collections import Counter

import torch

from llm_backend import LLMBackend


MODEL = "Qwen/Qwen2.5-7B-Instruct"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 4
TARGET_SENTENCES = 40000
N_PROBES = 15
CACHE = "wiki_corpus.json"        # NOT bigstream.json, which holds results
EMBED_BATCH = 256


def fetch_stream():
    """Real article text. Cached, with a shape check because a results file
    was once written to the corpus filename."""
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            data = json.load(f)
        if (isinstance(data, list) and data
                and isinstance(data[0], dict) and "text" in data[0]):
            return data
        print(f"  {CACHE} is not a corpus, refetching")

    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.en",
                      split="train", streaming=True)
    out = []
    for article in ds:
        title = article["title"]
        if title.startswith("List of") or "(disambiguation)" in title:
            continue
        for s in re.split(r"(?<=[.!?])\s+", article["text"][:4000]):
            s = s.strip()
            if 60 < len(s) < 250 and not s.startswith("="):
                out.append(dict(topic=title, text=s))
        if len(out) >= TARGET_SENTENCES:
            break
        if len(out) % 10000 < 20:
            print(f"  {len(out)} sentences...", flush=True)
    out = out[:TARGET_SENTENCES]
    with open(CACHE, "w") as f:
        json.dump(out, f)
    return out


def tokenize(s):
    return re.findall(r"[a-z0-9]+", s.lower())


class WordStore:
    """The previous retriever: word overlap weighted by rarity."""

    def __init__(self):
        self.items = []
        self.df = Counter()

    def build(self, sentences):
        for s in sentences:
            toks = set(tokenize(s))
            self.items.append((s, toks))
            for t in toks:
                self.df[t] += 1

    def search(self, query, k=TOP_K):
        q = set(tokenize(query))
        n = max(1, len(self.items))
        scored = []
        for sentence, toks in self.items:
            shared = q & toks
            if not shared:
                continue
            score = sum(math.log(1 + n / (1 + self.df[t]))
                        for t in shared)
            scored.append((score / math.sqrt(len(toks)), sentence))
        scored.sort(reverse=True)
        return [s for _, s in scored[:k]]


class EmbedStore:
    """Sentences as vectors, retrieved by nearest neighbour.

    Two sentences about the same thing land close together even with no
    words in common, which is exactly what word overlap cannot do.
    """

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(EMBED_MODEL)
        self.sentences = []
        self.vectors = None

    def build(self, sentences):
        self.sentences = list(sentences)
        self.vectors = self.model.encode(
            self.sentences, batch_size=EMBED_BATCH,
            convert_to_tensor=True, normalize_embeddings=True,
            show_progress_bar=False)

    def search(self, query, k=TOP_K):
        q = self.model.encode([query], convert_to_tensor=True,
                              normalize_embeddings=True)
        # normalised vectors, so a dot product is cosine similarity
        sims = torch.mm(q, self.vectors.T).squeeze(0)
        idx = torch.topk(sims, min(k, len(self.sentences))).indices
        return [self.sentences[int(i)] for i in idx]


REFUSALS = ["don't know", "do not know", "not know", "no information",
            "cannot find", "not mentioned", "unclear", "not specified",
            "not provided", "notes do not", "no mention", "not contain",
            "not stated", "unable to"]


def ask(backend, question, store):
    ctx = store.search(question) if store else None
    if ctx:
        joined = "\n".join(f"- {c}" for c in ctx)
        prompt = (f"Notes:\n{joined}\n\nAnswer using ONLY these notes. If "
                  f"they do not contain the answer, say you do not know.\n\n"
                  f"Q: {question}\nA:")
    else:
        prompt = f"Q: {question}\nA:"
    return backend.generate(prompt, max_new_tokens=60).strip()


def build_candidates(stream, lo, hi, corpus_values, limit=80):
    """Probes targeting a specific VALUE that appears in one sentence.

    A value shared by many sentences cannot identify one of them, so only
    rare values are used. That gives the probe a single right answer.
    """
    out = []
    seen_topics = set()
    for i in range(lo, hi):
        item = stream[i]
        text = item["text"]
        if item["topic"] in seen_topics:
            continue

        values = re.findall(r"\b\d[\d,\.]{2,}\b", text)
        subjects = re.findall(r"\b[A-Z][a-z]{4,}\b", text)
        if not values or not subjects:
            continue

        value = values[0]
        if corpus_values[value] > 3:
            continue

        seen_topics.add(item["topic"])
        out.append(dict(
            q=f"In the notes about {item['topic']}, "
              f"what number is given for {subjects[0]}?",
            expect=value.replace(",", ""),
            raw=value, topic=item["topic"], source=text))
        if len(out) >= limit:
            break
    return out


def filter_probes(backend, cands, want):
    """Keep only questions the frozen model cannot already answer."""
    kept = []
    for c in cands:
        a = ask(backend, c["q"], None).lower().replace(",", "")
        if c["expect"] not in a:
            kept.append(c)
        if len(kept) >= want:
            break
    return kept


UNANSWERABLE = [
    "What is the population of the city of Verrickton?",
    "Who invented the Brekke condenser?",
    "In what year was the Ardsleigh Treaty signed?",
    "How deep is the Corrieshalloch Trench?",
    "What is the melting point of ollmanium?",
]

GENERAL = [
    ("What is the capital of France?", "paris"),
    ("How many days are in a week?", "seven"),
    ("What colour is a ripe banana?", "yellow"),
    ("What season comes after summer?", "autumn"),
    ("What is water made of?", "hydrogen"),
]


def score(backend, store, probes):
    hits = 0
    rows = []
    for p in probes:
        a = ask(backend, p["q"], store)
        ok = p["expect"] in a.lower().replace(",", "")
        hits += ok
        rows.append((p["q"], a, p["raw"], bool(ok)))
    return (100.0 * hits / len(probes) if probes else 0.0), rows


def retrieval_hit_rate(store, probes):
    """Does the store return the SOURCE sentence at all?

    This separates the retriever from the model: if the right sentence is
    never retrieved, no prompt can save the answer.
    """
    if store is None:
        return 0.0
    hits = sum(1 for p in probes
               if any(p["source"] == c for c in store.search(p["q"])))
    return 100.0 * hits / len(probes) if probes else 0.0


if __name__ == "__main__":
    print("Does an embedding store scale where word overlap did not?\n")
    stream = fetch_stream()
    sentences = [s["text"] for s in stream]
    print(f"{len(stream)} sentences from "
          f"{len(set(s['topic'] for s in stream))} distinct articles")

    corpus_values = Counter()
    for s in sentences:
        for v in re.findall(r"\b\d[\d,\.]{2,}\b", s):
            corpus_values[v] += 1

    n = len(stream)
    early_c = build_candidates(stream, 0, n // 10, corpus_values)
    late_c = build_candidates(stream, n - n // 10, n, corpus_values)
    print(f"{len(early_c)} early candidates, {len(late_c)} late candidates")

    print("\nfiltering probes the model already knows...", flush=True)
    backend = LLMBackend(model_name=MODEL, focus_alpha=0.0)
    early = filter_probes(backend, early_c, N_PROBES)
    late = filter_probes(backend, late_c, N_PROBES)
    print(f"  kept {len(early)} early, {len(late)} late")
    if len(early) < 5 or len(late) < 5:
        print("  ABORT: too few probes survived filtering.")
        raise SystemExit

    print("\nbuilding stores...", flush=True)
    t0 = time.time()
    words = WordStore()
    words.build(sentences)
    print(f"  word store: {len(words.items)} items, "
          f"{time.time() - t0:.0f}s", flush=True)

    t0 = time.time()
    embeds = EmbedStore()
    embeds.build(sentences)
    print(f"  embedding store: {len(embeds.sentences)} items, "
          f"{time.time() - t0:.0f}s", flush=True)

    print("\nRETRIEVAL: does the store return the source sentence at all?")
    retrieval = {}
    for name, store in [("words", words), ("embed", embeds)]:
        e = retrieval_hit_rate(store, early)
        l = retrieval_hit_rate(store, late)
        retrieval[name] = dict(early=e, late=l)
        print(f"  {name:>6}: early {e:5.1f}%  late {l:5.1f}%", flush=True)

    print("\nANSWERS:", flush=True)
    results = {}
    for name, store in [("frozen", None), ("store-words", words),
                        ("store-embed", embeds)]:
        e, e_rows = score(backend, store, early)
        l, _ = score(backend, store, late)
        fab = sum(1 for q in UNANSWERABLE
                  if not any(r in ask(backend, q, store).lower()
                             for r in REFUSALS))
        gen = sum(1 for q, ex in GENERAL
                  if ex in ask(backend, q, None).lower())
        results[name] = dict(early=e, late=l, fabricated=fab, general=gen,
                             rows=e_rows[:5])
        print(f"  {name:>12}: early {e:5.1f}%  late {l:5.1f}%  "
              f"fab {fab}/{len(UNANSWERABLE)}  gen {gen}/{len(GENERAL)}",
              flush=True)

    print("\n" + "=" * 76)
    print(f"{'arm':>14} {'early':>9} {'late':>9} {'fab':>8} {'general':>9}")
    print("-" * 76)
    for name in ["frozen", "store-words", "store-embed"]:
        r = results[name]
        print(f"{name:>14} {r['early']:>8.1f}% {r['late']:>8.1f}% "
              f"{r['fabricated']:>6}/{len(UNANSWERABLE)} "
              f"{r['general']:>7}/{len(GENERAL)}")
    print("=" * 76)

    w, em = results["store-words"], results["store-embed"]
    print(f"\nDID EMBEDDINGS FIX IT? on identical probes:")
    print(f"  answers   early {em['early'] - w['early']:+6.1f}   "
          f"late {em['late'] - w['late']:+6.1f}")
    print(f"  retrieval early "
          f"{retrieval['embed']['early'] - retrieval['words']['early']:+6.1f}   "
          f"late "
          f"{retrieval['embed']['late'] - retrieval['words']['late']:+6.1f}")

    print("\nFIVE EARLY ANSWERS FROM THE EMBEDDING STORE:")
    for q, a, v, ok in results["store-embed"]["rows"]:
        print(f"  Q: {q[:70]}")
        print(f"  A: {a[:110]}")
        print(f"  expected {v}  ->  {'CORRECT' if ok else 'WRONG'}\n")

    print("""
The retrieval table is the important one. It reports whether the store
returns the SOURCE sentence at all, which separates the retriever from the
model.

  RETRIEVAL HIGH, ANSWERS LOW -> the right sentence is found and the model
      is not using it. A prompting problem, not a store problem.

  RETRIEVAL LOW FOR WORDS, HIGH FOR EMBEDDINGS -> the retriever was the
      bottleneck and this fixes it. The store scales after all.

  BOTH LOW -> forty thousand items is beyond what nearest-neighbour search
      over sentence embeddings resolves for these questions, and the store
      needs structure — per-topic indexes, or summarisation — rather than a
      better similarity measure.
""")
    with open("embedstream.json", "w") as f:
        json.dump(dict(results=results, retrieval=retrieval), f,
                  indent=2, default=str)
    print("wrote embedstream.json")