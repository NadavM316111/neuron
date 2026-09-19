"""The whole system on a genuinely large corpus.

The previous run validated the architecture but not retention. Only 18
Wikipedia summaries survived rate-limiting, giving 58 unique sentences
repeated 103 times — so everything entered the store in the first pass and
nothing was ever at risk of being lost. "Early material stayed at 75%"
because it never left, not because a mechanism preserved it.

It did establish real things: the system ran for hours without breaking,
retrieved usefully, refused all four unanswerable questions where the frozen
model fabricated all four, and kept general knowledge intact. The guard
fired four times, its first activity outside a narrow probe.

THIS RUN FIXES THE STREAM. Thousands of distinct Wikipedia articles from the
Hugging Face dump, streamed rather than downloaded whole. Tens of thousands
of unique sentences, so material genuinely passes out of reach and retention
becomes measurable.

THE FLOOR MUST BE HONEST. In the last run the frozen model scored 50% on
early-material probes, meaning half were answerable from what a 7B already
knows. Probes are now FILTERED: each candidate is put to the frozen model
first, and any it can already answer is discarded. Only questions the model
cannot answer unaided survive, so anything the system gets right came from
the stream.

That filtering costs a model load up front and it is the difference between
a measurement and a number.

Two arms:
  frozen    no learning, no store. The floor.
  system    gate, routing, store, rehearsal, guard.
"""

import json
import math
import os
import re
import time
from collections import Counter

import torch

from llm_backend import LLMBackend
from stability import StabilityLayer


MODEL = "Qwen/Qwen2.5-7B-Instruct"
LR = 1e-4
TOP_K = 4
CHECKPOINT_EVERY = 2000
TARGET_SENTENCES = 40000
N_PROBES = 12
CACHE = "bigstream.json"


def fetch_stream():
    """Thousands of distinct articles, streamed in order.

    Streaming rather than downloading avoids pulling twenty gigabytes for a
    few tens of thousands of sentences.
    """
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            return json.load(f)

    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.en",
                      split="train", streaming=True)

    out = []
    for article in ds:
        title = article["title"]
        # skip disambiguation and list pages, which are not prose
        if title.startswith("List of") or "(disambiguation)" in title:
            continue
        for s in re.split(r"(?<=[.!?])\s+", article["text"][:4000]):
            s = s.strip()
            if 60 < len(s) < 250 and not s.startswith("="):
                out.append(dict(topic=title, text=s))
        if len(out) >= TARGET_SENTENCES:
            break
        if len(out) % 5000 < 20:
            print(f"  {len(out)} sentences...", flush=True)

    out = out[:TARGET_SENTENCES]
    with open(CACHE, "w") as f:
        json.dump(out, f)
    return out


def tokenize(s):
    return re.findall(r"[a-z0-9]+", s.lower())


class Store:
    def __init__(self):
        self.items = []
        self.df = Counter()
        self.seen = set()

    def add(self, sentence):
        if sentence in self.seen:
            return
        self.seen.add(sentence)
        toks = set(tokenize(sentence))
        self.items.append((sentence, toks))
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


def looks_factual(sentence):
    if re.search(r"\d", sentence):
        return True
    words = [w for w in sentence.split()[1:] if not w.endswith(":")]
    return sum(1 for w in words if w[:1].isupper()) >= 1


REFUSALS = ["don't know", "do not know", "not know", "no information",
            "cannot find", "not mentioned", "unclear", "not specified",
            "not provided", "notes do not", "no mention", "not contain",
            "not stated", "unable to"]


def ask(backend, question, store):
    ctx = store.search(question) if store and store.items else None
    if ctx:
        joined = "\n".join(f"- {c}" for c in ctx)
        prompt = (f"Notes:\n{joined}\n\nAnswer using ONLY these notes. If "
                  f"they do not contain the answer, say you do not know.\n\n"
                  f"Q: {question}\nA:")
    else:
        prompt = f"Q: {question}\nA:"
    return backend.generate(prompt, max_new_tokens=60).strip()


def candidates(stream, lo, hi, limit=60):
    """Possible probes from a slice of the stream."""
    def distinctive(s):
        return [w for w in re.findall(r"\b[A-Za-z]{7,}\b", s)
                if w[0].isupper() or len(w) > 9]

    out = []
    seen = set()
    for i in range(lo, hi):
        item = stream[i]
        terms = distinctive(item["text"])
        if len(terms) < 2 or item["topic"] in seen:
            continue
        seen.add(item["topic"])
        out.append(dict(q=f"What does the text say about {terms[0]}?",
                        expect=terms[1].lower(), topic=item["topic"]))
        if len(out) >= limit:
            break
    return out


def filter_probes(backend, cands, want):
    """Keep only questions the frozen model CANNOT already answer.

    The previous run's frozen arm scored 50% on early probes, meaning half
    were answerable from pretraining. Anything the model already knows
    cannot measure what the stream taught it.
    """
    kept = []
    for c in cands:
        a = ask(backend, c["q"], None).lower()
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


def measure(backend, store, early, late):
    def score(probes):
        hits = sum(1 for p in probes
                   if p["expect"] in ask(backend, p["q"], store).lower())
        return 100.0 * hits / len(probes) if probes else 0.0

    fab = sum(1 for q in UNANSWERABLE
              if not any(r in ask(backend, q, store).lower()
                         for r in REFUSALS))
    gen = sum(1 for q, e in GENERAL
              if e in ask(backend, q, None).lower())
    return dict(early=score(early), late=score(late),
                fabricated=fab, general=gen)


def run(arm, stream, early, late):
    t0 = time.time()
    backend = LLMBackend(model_name=MODEL, focus_alpha=0.0)
    for g in backend.optimizer.param_groups:
        g["lr"] = LR

    store = Store() if arm == "system" else None
    layer = None
    routed = Counter()

    if arm == "system":
        layer = StabilityLayer(
            backend, canary=[s["text"] for s in stream[:40]], seed=0,
            window=200, warmup=30, top_fraction=0.50,
            coherence_veto=1.3, loss_floor=0.35, steps_per_update=1,
            rehearse_per_item=6, rehearse_count=2, rehearse_steps=1,
            anchor_size=60, buffer_size=300, sequence_len=8,
            guard=True, guard_per_item=500, canary_tolerance=0.40)

    curve = []
    for i, item in enumerate(stream):
        if arm == "system":
            if looks_factual(item["text"]):
                store.add(item["text"])
                routed["store"] += 1
            else:
                layer.observe(item["text"])
                routed["weights"] += 1

        if (i + 1) % CHECKPOINT_EVERY == 0:
            m = measure(backend, store, early, late)
            st = layer.summary() if layer else {}
            m.update(step=i + 1, mins=(time.time() - t0) / 60,
                     rollbacks=st.get("rollbacks", 0),
                     store=len(store.items) if store else 0)
            curve.append(m)
            print(f"    {i + 1:>6}/{len(stream)}  early {m['early']:5.1f}%  "
                  f"late {m['late']:5.1f}%  fab {m['fabricated']}/"
                  f"{len(UNANSWERABLE)}  gen {m['general']}/{len(GENERAL)}  "
                  f"store {m['store']:>5}  rb {m['rollbacks']}  "
                  f"{m['mins']:5.1f}m", flush=True)

    del backend, layer
    torch.cuda.empty_cache()
    return curve, dict(routed)


if __name__ == "__main__":
    print("The whole system on a genuinely large corpus.\n")
    stream = fetch_stream()
    print(f"{len(stream)} sentences from "
          f"{len(set(s['topic'] for s in stream))} distinct articles\n")

    if len(stream) < 10000:
        print("  ABORT: stream too small to test retention.")
        raise SystemExit

    # Build probes and filter them against the frozen model, so the floor
    # is honest before anything is measured.
    print("filtering probes the model already knows...", flush=True)
    probe_backend = LLMBackend(model_name=MODEL, focus_alpha=0.0)
    n = len(stream)
    early = filter_probes(probe_backend,
                          candidates(stream, 0, n // 20), N_PROBES)
    late = filter_probes(probe_backend,
                         candidates(stream, n - n // 20, n), N_PROBES)
    del probe_backend
    torch.cuda.empty_cache()

    print(f"  {len(early)} early probes, {len(late)} late probes kept")
    print(f"  (only questions the frozen model could NOT answer)\n")

    if len(early) < 5 or len(late) < 5:
        print("  ABORT: too few probes survived filtering.")
        raise SystemExit

    results = {}
    for arm in ["frozen", "system"]:
        print(f"--- {arm} ---", flush=True)
        curve, routed = run(arm, stream, early, late)
        results[arm] = dict(curve=curve, routed=routed)
        if routed:
            print(f"  routed: {routed}")
        print(flush=True)

    print("=" * 88)
    print(f"{'step':>8} {'early':>10} {'late':>10} {'fab':>8} "
          f"{'general':>9} {'store':>8}")
    print("-" * 88)
    for arm in ["frozen", "system"]:
        print(f"--- {arm} ---")
        for m in results[arm]["curve"]:
            print(f"{m['step']:>8} {m['early']:>9.1f}% {m['late']:>9.1f}% "
                  f"{m['fabricated']:>8} {m['general']:>9} "
                  f"{m['store']:>8}")
    print("=" * 88)

    f_last = results["frozen"]["curve"][-1]
    s = results["system"]["curve"]

    print("\nEARLY MATERIAL ACROSS THE RUN — the retention curve:")
    for m in s:
        print(f"  {m['step']:>6}: {m['early']:5.1f}%")

    print(f"\nAGAINST THE FROZEN MODEL:")
    print(f"  early : frozen {f_last['early']:5.1f}%  system "
          f"{s[-1]['early']:5.1f}%  "
          f"({s[-1]['early'] - f_last['early']:+5.1f})")
    print(f"  late  : frozen {f_last['late']:5.1f}%  system "
          f"{s[-1]['late']:5.1f}%  "
          f"({s[-1]['late'] - f_last['late']:+5.1f})")

    peak = max(m["early"] for m in s)
    print(f"\n  early material peaked at {peak:.1f}% and ended at "
          f"{s[-1]['early']:.1f}%  ({s[-1]['early'] - peak:+.1f})")

    print(f"\nHONESTY AND DAMAGE:")
    print(f"  fabrications: frozen {f_last['fabricated']}/"
          f"{len(UNANSWERABLE)}, system {s[-1]['fabricated']}/"
          f"{len(UNANSWERABLE)}")
    print(f"  general knowledge: frozen {f_last['general']}/{len(GENERAL)}, "
          f"system {s[-1]['general']}/{len(GENERAL)}")
    print(f"  the guard fired {s[-1]['rollbacks']} times")

    print("""
The probes were filtered against the frozen model, so anything answered
correctly came from the stream rather than from pretraining. That is what
the previous run lacked, where half the early probes were already known.

  EARLY STAYS HIGH ACROSS THE RUN -> retention works on real text at
      length. Forty thousand sentences means early material genuinely
      passes out of reach, unlike the previous run where 58 sentences
      repeated and nothing could be lost.

  EARLY PEAKS THEN FALLS -> the erosion seen in the grid world appears on
      real text too, and the peak says when it starts.

  EARLY NEVER RISES -> the store is not retrieving at this size. Forty
      thousand items is a hundred times the previous run and the retriever
      is plain word overlap, so this is a real possibility and would point
      at embeddings as the next step.
""")
    with open("bigstream.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote bigstream.json")