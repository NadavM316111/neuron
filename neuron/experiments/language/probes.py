"""Hand-written questions against the live store.

WHY THIS EXISTS. Two automatic attempts to score question answering have
now failed in the same way. September's generated probes asked things like
"What is said about Cejka?" and made every arm look broken. The 7 Sep
ask_eval used content-word bags as queries — "Tandy Corporation announces
TRS- world's mass-produced personal computers" — and the model reasonably
said it did not know what was being asked. Both times the measurement was
about the question generator, not about the system.

There is no clever fix. Questions have to be written by a person. This
script makes that cheap: it samples sentences from the store, writes them
to a file with a blank question under each, and scores whatever you fill in.

    ./run.sh experiments/language/probes.py sample --n 15
    (edit results/probes.txt, write a question under each source)
    ./run.sh experiments/language/probes.py score

THE THREE ARMS, and the third is the one that matters most.

  notes-strict   the current ask(): retrieved notes plus "answer using ONLY
                 these notes". This is what the system does today.
  notes-open     the same notes, without the ONLY instruction. Tests the
                 suspicion that the strict wording suppresses answers the
                 model actually has.
  CLOSED BOOK    no notes at all. THE CONTROL THIS PROJECT HAS BEEN MISSING.
                 The store is 2023 Wikipedia, which is in Qwen's
                 pretraining. If the model answers correctly with no notes,
                 the store contributed NOTHING and every "it answered
                 correctly" result so far is unearned. Any claim that
                 retrieval works has to beat this arm, not just look good
                 next to it.

FILE FORMAT. Lines beginning # are comments and are ignored.

    # source: <the sentence, printed for you to read>
    Q: your question here
    A: an optional key phrase the answer must contain

Leave Q blank to skip that entry. A is optional: with it, the script does a
crude substring check and reports it as a hint, not as a verdict. Reading
the answers yourself is still the measurement.
"""

import argparse
import json
import os
import random
import time

from neuron_system import NeuronSystem
from llm_backend import LLMBackend


HEADER = """# Hand-written probes for the NEURON store.
#
# Under each source sentence, write a question a person might actually ask,
# whose answer is IN that sentence. Do not paste the sentence back as the
# question. Optionally add an A: line with a key phrase the answer must
# contain — it is reported as a hint, not as a verdict.
#
# Leave Q blank to skip an entry.
"""


def cmd_sample(args):
    system = NeuronSystem(backend=None, path=args.state,
                          use_embeddings=False, retention=False)
    if not system.load():
        raise SystemExit(f"no saved state at {os.path.abspath(args.state)}")
    sents = system.store.sentences
    if not sents:
        raise SystemExit("store is empty")

    rng = random.Random(args.seed)
    picks = rng.sample(sents, min(args.n, len(sents)))

    with open(args.out, "w") as f:
        f.write(HEADER)
        for i, s in enumerate(picks, 1):
            f.write(f"\n# ---- {i} ----\n# source: {s}\nQ: \nA: \n")

    print(f"  {len(picks)} sources written to {os.path.abspath(args.out)}")
    print("  write a question under each, then run: score")


def parse_probes(path):
    """Read the edited file. Entries with a blank Q are skipped."""
    if not os.path.exists(path):
        raise SystemExit(f"no probe file at {os.path.abspath(path)}")
    probes, cur = [], None
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith("# source:"):
            if cur and cur["question"]:
                probes.append(cur)
            cur = dict(source=line[len("# source:"):].strip(),
                       question="", expect="")
        elif line.startswith("#") or cur is None:
            continue
        elif line.startswith("Q:"):
            cur["question"] = line[2:].strip()
        elif line.startswith("A:"):
            cur["expect"] = line[2:].strip()
    if cur and cur["question"]:
        probes.append(cur)
    return probes


def ask_with(backend, question, notes, strict):
    """One question, one prompt style. notes=None is the closed-book arm."""
    if notes:
        joined = "\n".join(f"- {n}" for n in notes)
        if strict:
            prompt = (f"Notes:\n{joined}\n\nAnswer using ONLY these notes. "
                      f"If they do not contain the answer, say you do not "
                      f"know.\n\nQ: {question}\nA:")
        else:
            prompt = (f"Notes:\n{joined}\n\nAnswer the question. Use the "
                      f"notes where they help.\n\nQ: {question}\nA:")
    else:
        prompt = f"Q: {question}\nA:"
    return backend.generate(prompt, max_new_tokens=80).strip()


def cmd_score(args):
    probes = parse_probes(args.probes)
    if not probes:
        raise SystemExit("no questions found — fill in the Q: lines first")
    print(f"  {len(probes)} questions", flush=True)

    print("  loading model ...", flush=True)
    backend = LLMBackend(model_name=args.model)
    system = NeuronSystem(backend=backend, path=args.state, retention=False)
    system.load()
    print(f"  {len(system.store.sentences)} sentences in store\n", flush=True)

    rows = []
    for i, p in enumerate(probes, 1):
        t = time.time()
        notes = system.store.search(p["question"], args.k)
        retrieved = p["source"] in notes

        answers = {
            "notes-strict": ask_with(backend, p["question"], notes, True),
            "notes-open": ask_with(backend, p["question"], notes, False),
            "closed-book": ask_with(backend, p["question"], None, False),
        }
        row = dict(n=i, question=p["question"], source=p["source"],
                   expect=p["expect"], retrieved=retrieved,
                   answers=answers,
                   refused={k: system.refused(v) for k, v in answers.items()},
                   seconds=time.time() - t)
        if p["expect"]:
            row["contains"] = {k: p["expect"].lower() in v.lower()
                               for k, v in answers.items()}
        rows.append(row)

        print(f"  {i}. {p['question']}")
        print(f"     retrieved source: {'yes' if retrieved else 'NO'}")
        for k in ("notes-strict", "notes-open", "closed-book"):
            mark = " [REFUSED]" if row["refused"][k] else ""
            hit = ""
            if p["expect"]:
                hit = " [has key phrase]" if row["contains"][k] else " [missing]"
            print(f"     {k:13s}{mark}{hit} {answers[k][:130]}")
        print(flush=True)

    # vars(args) carries the subcommand's function object, which json
    # cannot serialise. Everything printed above had already run when this
    # crashed on 7 Sep, so the results survived; the file did not.
    conf = {k: v for k, v in vars(args).items() if not callable(v)}
    with open(args.out, "w") as f:
        json.dump(dict(args=conf, rows=rows), f, indent=2)

    # ---------------------------------------------------------- summary

    n = len(rows)
    got = sum(1 for r in rows if r["retrieved"])
    print("=" * 62)
    print(f"RETRIEVAL   source sentence in top {args.k}: {got}/{n}")
    print("=" * 62)
    for k in ("notes-strict", "notes-open", "closed-book"):
        ref = sum(1 for r in rows if r["refused"][k])
        line = f"  {k:13s} refused {ref}/{n}"
        if any(r["expect"] for r in rows):
            scored = [r for r in rows if r["expect"]]
            hit = sum(1 for r in scored if r["contains"][k])
            line += f"   key phrase {hit}/{len(scored)}"
        print(line)

    print("\n" + "=" * 62)
    print("READING THIS")
    print("=" * 62)
    print("  CLOSED BOOK IS THE CONTROL. This store is 2023 Wikipedia and")
    print("  Qwen was pretrained on it. If closed-book matches the notes")
    print("  arms, the store contributed nothing on these questions and the")
    print("  retrieval claim is unsupported — whatever the other numbers say.")
    print("  Pick questions about obscure specifics if you want that arm to")
    print("  fail honestly; famous facts cannot separate the arms.")
    print("  strict vs open tells you whether the ONLY-these-notes wording")
    print("  is suppressing answers the model has. A large gap there is a")
    print("  prompt problem, not a retrieval problem.")
    print("  RETRIEVED IS UNDERSTATED. It asks whether the EXACT source")
    print("  sentence came back. On an encyclopedia the same fact appears in")
    print("  several sentences, so a question can be answered correctly from")
    print("  a different note while this column reads NO. Treat a NO here as")
    print("  'the specific sentence was missed', not 'retrieval failed'.")
    print("  The key-phrase check is a crude substring test and a hint only.")
    print("  Reading the answers is still the measurement.")
    print(f"\n  written to {os.path.abspath(args.out)}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="write source sentences to fill in")
    s.add_argument("--n", type=int, default=15)
    s.add_argument("--state", default="neuron_state")
    s.add_argument("--out", default="probes.txt")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_sample)

    c = sub.add_parser("score", help="run the questions you wrote")
    c.add_argument("--probes", default="probes.txt")
    c.add_argument("--state", default="neuron_state")
    c.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    c.add_argument("--k", type=int, default=4)
    c.add_argument("--out", default="probes_result.json")
    c.set_defaults(func=cmd_score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()