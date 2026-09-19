"""What does the router actually do, and does it keep what matters?

THE ROUTER HAS NEVER BEEN MEASURED. `looks_factual` sends a sentence to the
store if it contains a digit, or two capitalised words past the first.
That rule was written as a placeholder, its own docstring says it is
deliberately crude, and every claim about the fact/skill split rests on it.

TWO FAILURES ALREADY OBSERVED.

On 7 Sep the system was asked what anarchism is skeptical of. It answered
correctly and NONE of the four retrieved notes contained the answer. The
sentence that did — "Anarchism is a political philosophy and movement that
is skeptical of all justifications for authority" — has no digit and no
capitalised word past the first, so the router sent the most informative
sentence in the article to the weights instead of the store. The answer came
from pretraining while the prompt said to use only the notes.

And the split it produces on real Wikipedia is 60 to 70% to the store, which
means the learning half of the system sees a minority of the stream, chosen
by punctuation.

WHAT THIS MEASURES, and it needs no model and no training.

  COVERAGE. For sentences that demonstrably answer a real question, does the
  router store them? This is the failure above, counted. The probe file from
  probes.py supplies the question-and-source pairs, so the sentences are
  known to be answer-bearing rather than assumed to be.

  SHARE. What fraction of a real stream each rule sends to the store, since
  a rule that stores everything trivially has perfect coverage and tells
  the weights nothing.

FOUR RULES COMPARED.

  current     a digit, or two capitalised words past the first
  any-proper  a digit, or ONE capitalised word anywhere including the first
  store-all   everything goes to the store
  dual        everything goes to the store AND the gate still decides what
              to learn into the weights, independently

`dual` is the one worth taking seriously, and it questions an assumption
nobody tested: that routing has to be EXCLUSIVE. A sentence can be both a
specific claim worth recalling verbatim and material to learn patterns
from. The exclusive split came from an experiment showing each destination
fails at the other's job, which is a fact about destinations, not a reason
to send each sentence to only one of them. Eviction now exists, so a
larger store is affordable in a way it was not before.

    python route_eval.py --data ~/neuron/data/wiki.txt \\
        --probes ~/neuron/neuron/results/probes.txt
"""

import argparse
import os
import re


# ------------------------------------------------------------ the rules

def current(sentence, min_propers=2):
    """The shipped rule."""
    if re.search(r"\d", sentence):
        return True
    words = [w.strip(",.;:()\"'") for w in sentence.split()[1:]]
    return sum(1 for w in words if w and w[:1].isupper()) >= min_propers


def any_proper(sentence):
    """A digit, or any capitalised word ANYWHERE including the first.

    The shipped rule skips the first word to avoid counting the sentence's
    initial capital. That is reasonable and it is also why a sentence
    beginning with the subject it is about — "Anarchism is..." — reads as
    having no proper nouns at all.
    """
    if re.search(r"\d", sentence):
        return True
    words = [w.strip(",.;:()\"'") for w in sentence.split()]
    return any(w and w[:1].isupper() for w in words)


def store_all(sentence):
    return True


RULES = {
    "current": current,
    "any-proper": any_proper,
    "store-all": store_all,
}


# ---------------------------------------------------------------- data

def read_probes(path):
    """Pull (question, source sentence) pairs out of a probes.py file.

    Only entries with a question filled in, since a blank Q means the
    source was skipped as unanswerable.
    """
    if not os.path.exists(path):
        return []
    out, src = [], None
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith("# source:"):
            src = line[len("# source:"):].strip()
        elif line.startswith("Q:") and src:
            q = line[2:].strip()
            if q:
                out.append((q, src))
            src = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser(
        "~/neuron/data/wiki.txt"))
    ap.add_argument("--probes", default=os.path.expanduser(
        "~/neuron/neuron/results/probes.txt"))
    ap.add_argument("--sample", type=int, default=20000)
    args = ap.parse_args()

    if not os.path.exists(args.data):
        raise SystemExit(f"no data at {args.data}")

    lines = [l.strip() for l in open(args.data) if len(l.strip()) > 20]
    lines = lines[:args.sample]
    probes = read_probes(args.probes)

    print(f"  {len(lines):,} sentences, {len(probes)} answer-bearing "
          f"sentences with questions\n")

    # ---- share on a real stream ----
    print("=" * 70)
    print("SHARE SENT TO THE STORE")
    print("=" * 70)
    shares = {}
    for name, fn in RULES.items():
        n = sum(1 for l in lines if fn(l))
        shares[name] = n / len(lines)
        print(f"  {name:12s} {n:7,} of {len(lines):,}  "
              f"({shares[name] * 100:5.1f}% to the store, "
              f"{(1 - shares[name]) * 100:4.1f}% to the weights)")

    if not probes:
        print(f"\n  No probe file at {args.probes}, so coverage cannot be "
              f"measured.")
        print(f"  Run probes.py sample, fill in the questions, and point "
              f"--probes at it.")
        return

    # ---- coverage on sentences known to answer something ----
    print("\n" + "=" * 70)
    print("COVERAGE: does the rule STORE a sentence that answers a question?")
    print("=" * 70)
    missed = {}
    for name, fn in RULES.items():
        kept = [(q, s) for q, s in probes if fn(s)]
        missed[name] = [(q, s) for q, s in probes if not fn(s)]
        print(f"  {name:12s} {len(kept):2d} of {len(probes)} stored "
              f"({100 * len(kept) / len(probes):5.1f}%)")

    for name in ("current", "any-proper"):
        if missed[name]:
            print(f"\n  {name} sent these ANSWER-BEARING sentences to the "
                  f"weights:")
            for q, s in missed[name][:4]:
                print(f"    Q: {q[:64]}")
                print(f"    S: {s[:86]}")

    print("\n" + "=" * 70)
    print("READING THIS")
    print("=" * 70)
    print("  store-all has perfect coverage by construction and is not")
    print("  therefore the answer: it leaves the weights nothing, and the")
    print("  whole-system result at 7B was that the weights learn patterns")
    print("  the store cannot. A rule is only better if it raises coverage")
    print("  WITHOUT sending everything.")
    print()
    cur_cov = 1 - len(missed["current"]) / len(probes)
    any_cov = 1 - len(missed["any-proper"]) / len(probes)
    if any_cov > cur_cov and shares["any-proper"] < 0.95:
        print(f"  any-proper raises coverage from {cur_cov * 100:.0f}% to "
              f"{any_cov * 100:.0f}% while still")
        print(f"  sending {(1 - shares['any-proper']) * 100:.0f}% of the "
              f"stream to the weights. That is a better rule on this "
              f"evidence.")
    elif any_cov <= cur_cov:
        print(f"  any-proper does not raise coverage "
              f"({any_cov * 100:.0f}% against {cur_cov * 100:.0f}%), so "
              f"the first-word")
        print(f"  exclusion was not the problem and the router needs a "
              f"different fix.")
    else:
        print(f"  any-proper raises coverage to {any_cov * 100:.0f}% but "
              f"stores "
              f"{shares['any-proper'] * 100:.0f}% of everything, which is")
        print(f"  close enough to store-all that it is not really a split "
              f"any more.")

    print(f"\n  {len(probes)} probe sentences is a small sample and they "
          f"were written after")
    print(f"  seeing the sources. Treat this as a bug measurement, not a "
          f"benchmark.")


if __name__ == "__main__":
    main()
