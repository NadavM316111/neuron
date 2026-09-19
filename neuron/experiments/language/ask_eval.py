"""Can the running system answer from what it read?

The store holds thousands of sentences and has never been QUERIED inside
the program. ask() exists, was validated at 7B in an experiment script, and
has never run against the live persisted state. This closes that.

WHAT THIS MEASURES, AND WHAT IT DELIBERATELY DOES NOT.

The Sep 2 retrieval work established that generated probes are worthless
here: they produced questions like "What is said about Cejka?" expecting
"co-chaired", and every arm scored badly for reasons that had nothing to do
with the arms. Hand-written probes then scored 100% on everything, including
the raw unfiltered corpus. So AUTOMATIC SCORING OF ANSWER QUALITY IS NOT
ATTEMPTED HERE. Any number this script printed for "QA accuracy" would be a
number about the question generator.

What it measures instead, all three of which are fair:

  RETRIEVAL   given content words lifted from a stored sentence, does that
              sentence come back in the top k? This is a property of the
              index, not of the question generator: a retriever that cannot
              find a sentence from its own distinctive words is broken
              regardless of how the query was phrased. It is a FLOOR, not
              an accuracy figure — real questions are harder than this.

  REFUSAL     asked about material the store certainly does not contain,
              does the system say it does not know? This is the August
              fabrication result turned into a live check. Fabrication is
              the failure mode that matters, and unlike answer quality it
              can be scored without a question generator, because the
              correct answer is always "I don't know".

  OVER-REFUSAL  the other half of that, and the half a refusal rate alone
              hides. A system that declines EVERYTHING scores a perfect
              100% on refusal. So this asks questions whose source sentence
              was VERIFIED to come back in the top k, and counts how often
              the model declined anyway. A live run on 7 Sep did exactly
              that: asked about a DiFranco record with the correct sentence
              sitting in the top hit, the 1.5B answered "I do not know".
              Retrieval succeeded and generation refused. That failure is
              invisible in a refusal rate and it is now counted.

              Only questions where retrieval SUCCEEDED are scored, so a bad
              query cannot be mistaken for an over-refusal.

  LATENCY     seconds per query at the store's current size. The vision is
              a system that runs for months; if a question costs ten
              seconds at thirteen thousand sentences, that matters more
              than any accuracy number.

Then it PRINTS real questions with their retrieved notes and answers, for
you to read. Judging those is your job, not the script's.

    ./run.sh experiments/language/ask_eval.py
    ./run.sh experiments/language/ask_eval.py --interactive
    ./run.sh experiments/language/ask_eval.py --samples 40 --show 8
"""

import argparse
import json
import os
import random
import re
import time

from neuron_system import NeuronSystem
from llm_backend import LLMBackend


STOP = set("""a an the of in on at to for from by with and or but is are was
were be been being it its this that these those as has have had not no more
most also which who whom whose what when where while than then there their
his her they them he she we you i one two three first second new other some
such can could would should may might will shall do does did than into over
under after before between during about against through above below up down
out off again further once here why how all any both each few nor only own
same so too very s t just don now""".split())


def content_words(sentence, keep=7):
    """Distinctive words from a sentence, for use as a retrieval query.

    Not a question. A question generator is exactly what poisoned the
    September retrieval measurement. This is a bag of the sentence's own
    rare-ish words, which tests the INDEX rather than testing a generator.
    """
    words = re.findall(r"[A-Za-z][A-Za-z'-]+", sentence)
    picked = [w for w in words if w.lower() not in STOP and len(w) > 3]
    return " ".join(picked[:keep])


# Questions about material the store cannot contain. Invented entities, and
# the synthetic rule from drift.py, which exists nowhere outside this repo.
# The only correct answer to every one of these is that it does not know.
UNANSWERABLE = [
    "What happens to the northern gate when the amber beacon is lit?",
    "How many string quartets did Ottoline Verrick compose?",
    "When does the ferry to Harrowmere leave in the morning?",
    "Who is the current keeper of the violet standard at Fenwold?",
    "What did the Brindlecask Accord of 1847 establish?",
    "Which river runs past the smithy at Kestrel Cross?",
    "What is the population of Stilt Hollow?",
    "Why was the bridge at Marle End rebuilt in stone?",
]


def evaluate_retrieval(system, samples, k, rng):
    """Can the index find a sentence from its own content words?"""
    if not system.store.sentences:
        return None
    picks = rng.sample(system.store.sentences,
                       min(samples, len(system.store.sentences)))
    hits, empty, times = 0, 0, []
    for s in picks:
        q = content_words(s)
        if not q.strip():
            empty += 1
            continue
        t = time.time()
        got = system.store.search(q, k)
        times.append(time.time() - t)
        if s in got:
            hits += 1
    scored = len(picks) - empty
    return dict(scored=scored, hits=hits,
                rate=hits / scored if scored else float("nan"),
                skipped_no_content=empty,
                mean_query_seconds=sum(times) / len(times) if times else 0.0,
                k=k)


def evaluate_over_refusal(system, samples, k, rng):
    """Does it decline when the answer is demonstrably right there?

    Scored only on cases where the source sentence was retrieved into the
    top k. If retrieval missed, the model declining is correct behaviour and
    says nothing about over-refusal, so those cases are excluded rather than
    counted against it.
    """
    if not system.store.sentences:
        return None
    picks = rng.sample(system.store.sentences,
                       min(samples, len(system.store.sentences)))
    scored, refused, rows = 0, 0, []
    for s in picks:
        q = content_words(s, keep=9)
        if not q.strip():
            continue
        if s not in system.store.search(q, k):
            continue          # retrieval missed: not an over-refusal case
        scored += 1
        a = system.ask(q)
        if system.refused(a):
            refused += 1
            rows.append(dict(query=q, source=s, answer=a))
    return dict(scored=scored, refused=refused,
                rate=refused / scored if scored else float("nan"),
                rows=rows)


def evaluate_refusal(system):
    """Asked the unanswerable, does it decline or invent?"""
    rows, refused, times = [], 0, []
    for q in UNANSWERABLE:
        t = time.time()
        a = system.ask(q)
        times.append(time.time() - t)
        r = system.refused(a)
        refused += 1 if r else 0
        rows.append(dict(question=q, answer=a, refused=r))
    return dict(asked=len(UNANSWERABLE), refused=refused,
                rate=refused / len(UNANSWERABLE),
                mean_seconds=sum(times) / len(times) if times else 0.0,
                rows=rows)


def show_examples(system, n, rng):
    """Real questions, real notes, real answers. Read them yourself.

    Deliberately unscored. The point is to see whether the answers are
    grounded in the retrieved notes or drifting away from them, which is a
    judgement no metric in this repo has ever made well.
    """
    if not system.store.sentences:
        return []
    picks = rng.sample(system.store.sentences,
                       min(n, len(system.store.sentences)))
    out = []
    for s in picks:
        q = content_words(s, keep=9)
        notes = system.store.search(q, 4)
        a = system.ask(q)
        out.append(dict(query=q, source=s, notes=notes, answer=a))
    return out


def interactive(system):
    print("\n  Ask it anything. Blank line or ctrl-c to stop.\n")
    while True:
        try:
            q = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            return
        t = time.time()
        notes = system.store.search(q, 4)
        a = system.ask(q)
        took = time.time() - t
        print(f"\n  retrieved {len(notes)} notes:")
        for n in notes:
            print(f"    - {n[:110]}")
        print(f"\n  {a}\n  ({took:.1f}s"
              f"{', REFUSED' if system.refused(a) else ''})\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="neuron_state")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--samples", type=int, default=200,
                    help="sentences sampled for the retrieval check")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--over", type=int, default=30,
                    help="sentences sampled for the over-refusal check")
    ap.add_argument("--show", type=int, default=5,
                    help="example questions to print for you to judge")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--out", default="ask_eval.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    print("  loading model ...", flush=True)
    backend = LLMBackend(model_name=args.model)
    # retention off: this session only reads the store and asks questions,
    # so building a stability layer would cost a snapshot and buy nothing.
    system = NeuronSystem(backend=backend, path=args.state, retention=False)
    if not system.load():
        raise SystemExit(f"  no saved state at {os.path.abspath(args.state)}")

    n = len(system.store.sentences)
    print(f"  {n} sentences, embeddings "
          f"{'on' if system.store.model is not None else 'OFF (word overlap)'}",
          flush=True)
    if n == 0:
        raise SystemExit("  store is empty, nothing to ask about")

    if args.interactive:
        interactive(system)
        return

    print("\n  retrieval ...", flush=True)
    ret = evaluate_retrieval(system, args.samples, args.k, rng)
    print(f"    found the source sentence in the top {ret['k']} for "
          f"{ret['hits']}/{ret['scored']} ({ret['rate'] * 100:.1f}%)")
    print(f"    {ret['mean_query_seconds'] * 1000:.0f} ms per query at "
          f"{n} sentences")
    if ret["skipped_no_content"]:
        print(f"    {ret['skipped_no_content']} skipped: no content words")

    print("\n  refusal on material the store cannot contain ...", flush=True)
    ref = evaluate_refusal(system)
    print(f"    declined {ref['refused']}/{ref['asked']} "
          f"({ref['rate'] * 100:.0f}%), {ref['mean_seconds']:.1f}s per answer")
    for row in ref["rows"]:
        if not row["refused"]:
            print(f"    FABRICATED: {row['question']}")
            print(f"      -> {row['answer'][:160]}")

    print("\n  over-refusal, where the answer WAS retrieved ...", flush=True)
    over = evaluate_over_refusal(system, args.over, args.k, rng)
    if over and over["scored"]:
        print(f"    declined {over['refused']}/{over['scored']} "
              f"({over['rate'] * 100:.0f}%) despite having the source")
        for row in over["rows"][:3]:
            print(f"    OVER-REFUSED: {row['query'][:70]}")
            print(f"      source: {row['source'][:100]}")
    else:
        print("    no scorable cases")

    ex = show_examples(system, args.show, rng)
    if ex:
        print("\n" + "=" * 62)
        print("EXAMPLES — unscored, judge these yourself")
        print("=" * 62)
        for e in ex:
            print(f"\n  query   {e['query']}")
            print(f"  source  {e['source'][:110]}")
            print(f"  top hit {e['notes'][0][:110] if e['notes'] else '(none)'}")
            print(f"  answer  {e['answer'][:220]}")

    with open(args.out, "w") as f:
        json.dump(dict(args=vars(args), sentences=n, retrieval=ret,
                       refusal=ref, over_refusal=over, examples=ex),
                  f, indent=2)

    print("\n" + "=" * 62)
    print("READING THIS")
    print("=" * 62)
    print("  Retrieval is a FLOOR. The query is the sentence's own words, so")
    print("  a real question phrased differently will do worse. A low number")
    print("  here means the index is broken; a high number does not mean")
    print("  question answering works.")
    print("  Refusal is the one that matters. The August result was that")
    print("  gradient-taught facts turn honest refusals into confident")
    print("  fabrications. Anything below 100% here is that failure mode")
    print("  appearing in the live system.")
    print("  Read refusal and OVER-refusal together or neither means much.")
    print("  A system that declines everything scores 100% on the first and")
    print("  is useless. The pair is the honest picture.")
    print(f"\n  written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()