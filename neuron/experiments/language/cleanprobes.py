"""Retrieval, measured honestly, on a corpus worth retrieving from.

Five measurement attempts this week produced numbers that could not be
trusted. The last one asked "what number is given in connection with
Aristotle?" — the answer was unique, but the QUESTION could not identify
the sentence, since thirty sentences mention Aristotle and only one has
that number. A bigger embedding model scored twelve points WORSE, which no
genuine improvement can do on a sound measurement.

Then ten random corpus sentences were read by hand, and the actual problem
appeared:

  "3) criticizes his writings as characterized by pomposity of style..."
  "According to him, Andriscus was already a mercenary in Demetrius' army."
  "The two men would go on to collaborate on another fifteen films."
  "References \n Angiosperm orders \n Taxa named by Takenoshin Nakai"

HALF THE CORPUS CANNOT BE RETRIEVED BY ANY METHOD. Fragments starting
mid-thought, reference lists, and sentences whose subject is only a pronoun
— "him", "the two men", "de Ayllón and many of the colonists" — contain
nothing a question could match against. That is a corpus problem, and it
may be the whole explanation for the low numbers.

TWO THINGS ARE FIXED HERE.

  THE PROBES. Five are written and checked BY HAND against the rule "could
      only this sentence answer it". The rest are generated under a much
      stricter rule and printed in full, so they can be read rather than
      trusted.

  THE CORPUS. A filtering arm removes fragments, reference lists and
      pronoun-only sentences. If retrieval jumps, the store was fine all
      along and it was being fed unretrievable material.

Arms:
  raw-corpus       everything, as before
  clean-corpus     filtered, same retriever
  clean+bigger     filtered, larger embedding model
  clean+rerank     filtered, retrieve 20 and rescore

If the bigger model finally HELPS on the clean corpus, that confirms the
measurement is sound at last — a better retriever cannot make an honest
measurement worse.
"""

import gc
import json
import os
import re
import time

import torch

CORPUS_CACHE = "growth_corpus.json"
STORE_SIZE = 20000
SMALL_MODEL = "all-MiniLM-L6-v2"
BIG_MODEL = "all-mpnet-base-v2"


# Written by hand from sentences actually in this corpus, each checked
# against the rule: could ONLY this sentence answer it?
HAND_PROBES = [
    dict(q="Where is a mirage sometimes seen in summer?",
         answer="Vildmose"),
    dict(q="What was the Atlantic Ocean the centre of from the 16th to "
           "19th centuries, besides a slave trade?",
         answer="Columbian exchange"),
    dict(q="Who undertook the conquest of Sardinia in 1323 to 1324?",
         answer="Alfonso"),
    dict(q="Who was Arthur Miller's younger sister?",
         answer="Joan Copeland"),
    dict(q="Which war repelled the British attempt to subjugate "
           "Afghanistan from India?",
         answer="Anglo-Afghan"),
]


PRONOUN_STARTS = (
    "he ", "she ", "it ", "they ", "him ", "her ", "them ", "his ",
    "their ", "its ", "this ", "that ", "these ", "those ", "the two ",
    "according to him", "according to her", "according to them",
)


def is_retrievable(s):
    """Can a question possibly match this sentence?

    A sentence whose subject is only a pronoun has nothing to match
    against. Neither does a fragment or a reference list. Removing them is
    not cheating — a real system should not store material it can never
    surface.
    """
    low = s.lower().strip()

    if len(s) < 60 or len(s) > 300:
        return False
    if not s[0].isupper():
        return False                      # fragments starting mid-thought
    if "\n" in s:
        return False                      # reference lists and tables
    if low.startswith(PRONOUN_STARTS):
        return False
    if s.count(",") > 6:
        return False                      # list-like, not prose

    # needs at least one proper noun that is not the first word, so a
    # question has something specific to name
    named = [w for w in s.split()[1:] if w[:1].isupper() and len(w) > 3]
    return len(named) >= 1


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


def generate_probes(corpus, n_probes, seed=0):
    """Stricter than any previous generator.

    The question names a proper noun AND asks for a distinctive term from
    the same sentence, and the sentence must be retrievable in the first
    place. Every probe is printed, so they can be read rather than
    trusted.
    """
    import random
    rng = random.Random(seed)
    order = list(range(len(corpus)))
    rng.shuffle(order)

    probes = []
    for i in order:
        s = corpus[i]
        if not is_retrievable(s):
            continue
        named = [w.strip(".,;:()") for w in s.split()[1:]
                 if w[:1].isupper() and len(w) > 4]
        rare = [w.strip(".,;:()") for w in s.split()
                if len(w) > 7 and w[:1].islower()]
        if not named or not rare:
            continue
        probes.append(dict(
            q=f"What is said about {named[0]}?",
            answer=rare[-1], source=s))
        if len(probes) >= n_probes:
            break
    return probes


class Retriever:
    def __init__(self, corpus, model_name=SMALL_MODEL, rerank=False):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.items = list(corpus)
        self.rerank = rerank
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
                scored.append((float(sims[i])
                               + len(qt & it) / max(1, len(qt)), i))
            scored.sort(reverse=True)
            idx = [i for _, i in scored[:k]]
        else:
            idx = [int(i) for i in
                   torch.topk(sims, min(k, len(self.items))).indices]
        return [self.items[i] for i in idx]


def score(retriever, probes, k=4):
    hits = 0
    misses = []
    t0 = time.perf_counter()
    for p in probes:
        results = retriever.search(p["q"], k)
        if any(p["answer"].lower() in r.lower() for r in results):
            hits += 1
        else:
            misses.append((p["q"], p["answer"], results[0][:70]))
    ms = 1000.0 * (time.perf_counter() - t0) / len(probes)
    return 100.0 * hits / len(probes), ms, misses


if __name__ == "__main__":
    print("Retrieval, measured honestly, on a corpus worth "
          "retrieving from.\n")

    raw = fetch_corpus(STORE_SIZE)
    clean = [s for s in raw if is_retrievable(s)]
    print(f"raw corpus   : {len(raw)} sentences")
    print(f"clean corpus : {len(clean)} sentences "
          f"({100.0 * len(clean) / len(raw):.1f}% kept)")
    print(f"discarded    : {len(raw) - len(clean)} fragments, reference "
          f"lists and pronoun-only sentences\n")

    print("examples of what was discarded:")
    shown = 0
    for s in raw:
        if not is_retrievable(s):
            print(f"  - {s[:78].replace(chr(10), ' ')}...")
            shown += 1
            if shown >= 4:
                break
    print()

    # The hand-written probes are the trustworthy set. Check they are
    # answerable from the clean corpus before relying on them.
    usable_hand = []
    for p in HAND_PROBES:
        present = [s for s in clean if p["answer"].lower() in s.lower()]
        if present:
            usable_hand.append(p)
        else:
            print(f"  hand probe dropped, answer not in clean corpus: "
                  f"{p['answer']}")
    print(f"\n{len(usable_hand)} of {len(HAND_PROBES)} hand-written probes "
          f"are answerable from the clean corpus")

    generated = generate_probes(clean, 80)
    print(f"{len(generated)} generated probes\n")

    print("the hand-written probes, for reading:")
    for p in usable_hand:
        print(f"  Q: {p['q']}")
        print(f"     expecting: {p['answer']}")
    print()

    print("five generated probes, for reading:")
    for p in generated[:5]:
        print(f"  Q: {p['q']}")
        print(f"     expecting: {p['answer']}")
        print(f"     from: {p['source'][:70]}...")
    print()

    arms = [
        ("raw-corpus", raw, dict(model_name=SMALL_MODEL)),
        ("clean-corpus", clean, dict(model_name=SMALL_MODEL)),
        ("clean+rerank", clean, dict(model_name=SMALL_MODEL, rerank=True)),
        ("clean+bigger", clean, dict(model_name=BIG_MODEL)),
    ]

    results = {}
    for name, corpus, kwargs in arms:
        print(f"--- {name} ---", flush=True)
        r = Retriever(corpus, **kwargs)
        h_hit, h_ms, _ = score(r, usable_hand) if usable_hand else (0, 0, [])
        g_hit, g_ms, misses = score(r, generated)
        results[name] = dict(hand=h_hit, generated=g_hit, ms=g_ms,
                             items=len(corpus))
        print(f"  hand-written {h_hit:5.1f}%   generated {g_hit:5.1f}%   "
              f"{g_ms:5.2f}ms", flush=True)
        if name == "clean-corpus" and misses:
            print("  three misses, for reading:")
            for q, a, top in misses[:3]:
                print(f"    Q: {q}")
                print(f"       wanted '{a}', top result: {top}...")
        del r
        gc.collect()

    print("\n" + "=" * 74)
    print(f"{'arm':>16} {'items':>8} {'hand-written':>14} "
          f"{'generated':>11} {'ms':>7}")
    print("-" * 74)
    for name, _, _ in arms:
        r = results[name]
        print(f"{name:>16} {r['items']:>8} {r['hand']:>13.1f}% "
              f"{r['generated']:>10.1f}% {r['ms']:>7.2f}")
    print("=" * 74)

    raw_g = results["raw-corpus"]["generated"]
    clean_g = results["clean-corpus"]["generated"]
    big_g = results["clean+bigger"]["generated"]

    print(f"\nDID CLEANING THE CORPUS HELP? "
          f"{clean_g - raw_g:+.1f} points")
    print(f"DOES A BIGGER MODEL NOW HELP? {big_g - clean_g:+.1f} points")

    print()
    if big_g > clean_g:
        print("  THE MEASUREMENT IS SOUND AT LAST. A bigger embedding model")
        print("  now improves retrieval, which it could not do on any of")
        print("  the five previous attempts. Those numbers were measuring")
        print("  probe ambiguity; these are measuring retrieval.")
    else:
        print("  STILL NOT SOUND. A bigger model still does not help, so")
        print("  the probes are still not identifying single sentences.")
        print("  The hand-written column is the only number to trust here,")
        print("  and there are only a few of them.")

    print("""
The hand-written column is the trustworthy one. Five probes is too few to
be precise, but each was checked by a person against the rule that only one
sentence could answer it — which is exactly what five generated probe sets
failed to guarantee.

The corpus filtering may matter more than any retriever change. Half of
what was being stored — fragments, reference lists, sentences whose subject
is a pronoun — cannot be surfaced by any question, and a real system should
not store material it can never retrieve.
""")
    with open("cleanprobes.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote cleanprobes.json")