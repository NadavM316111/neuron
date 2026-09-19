"""Does the whole system work at once, or do the pieces fight?

Every mechanism in this project has been validated separately. The gate on a
grid world. Rehearsal on weather. The guard on degrading language models.
Retrieval on a static set of facts. They have never had to coexist.

THE PREDICTION this was built to test: the gate fires on SURPRISE, and facts
are surprising, so a system routing by surprise alone should send facts into
the weights and fabricate — which is exactly what the August work proved
happens. The split arms route by CONTENT instead.

A single stream mixes three kinds of item:

  FACTS    specific invented claims with values. Belong in the store,
           because gradient updates on these produce confident wrong
           answers.
  SKILL    a consistent format the model should absorb into its weights:
           "OBSERVATION: ... THEREFORE: ..." where the conclusion follows
           from the observation by a fixed rule. Pattern, not content — the
           thing weights are good at and a store cannot supply.
  FILLER   ordinary sentences that are neither.

FIXED FROM THE FIRST RUN. The router checked for capitalised words
mid-sentence as a proxy for proper nouns. The skill sentences contain
"THEREFORE:" mid-sentence, so EVERY skill pattern was routed to the store
and the split arm never sent anything meaningful to the weights — it came
out identical to store-only on every column. Template markers ending in a
colon are now ignored, since a capitalised word followed by a colon is
formatting rather than a name.

ALSO FIXED: the grader marked correct answers as fabrications. "Ottoline
Verrick wrote nineteen string quartets before she turned thirty" is right,
but "thirty" was in the wrong-answer list because it appears in the fact
itself. Wrong-answer lists now exclude any token that occurs in the source
sentence.

Arms:
  frozen         nothing learned, nothing stored. The floor.
  weights-only   everything into the weights. The August architecture.
  store-only     everything into the store, nothing into the weights.
  split          facts to the store, skill to the weights, by the router.
  split-guarded  the same, plus rehearsal and the canary guard.
"""

import json
import math
import re
import time
from collections import Counter

import torch

from llm_backend import LLMBackend
from stability import StabilityLayer


MODEL = "Qwen/Qwen2.5-7B-Instruct"
LR = 1e-4
REPEATS = 6
TOP_K = 3


FACTS = [
    dict(sentence="The composer Ottoline Verrick wrote nineteen string "
                  "quartets before she turned thirty.",
         question="How many string quartets did Ottoline Verrick write?",
         answer="nineteen",
         wrong=["forty", "twelve", "twenty", "one", "three", "eight",
                "fifteen"]),
    dict(sentence="The naturalist Halvard Brekke catalogued eleven "
                  "thousand moths at the Bergen museum.",
         question="How many moths did Halvard Brekke catalogue?",
         answer="eleven thousand",
         wrong=["million", "hundred thousand", "five", "twenty thousand"]),
    dict(sentence="The Kestrel Mill at Ardsleigh stopped grinding flour in "
                  "nineteen forty.",
         question="In what year did the Kestrel Mill stop grinding flour?",
         answer="nineteen forty",
         wrong=["1924", "1947", "1938", "1952", "1901", "1965"]),
    dict(sentence="Tobias Wrenn mapped the Corrieshalloch caves over four "
                  "summers.",
         question="How many summers did Tobias Wrenn spend mapping the "
                  "Corrieshalloch caves?",
         answer="four",
         wrong=["two", "three", "five", "six", "ten", "seven"]),
    dict(sentence="The village of Oldmere lies fourteen miles upstream "
                  "from the Feltwater estuary.",
         question="How far upstream is the village of Oldmere?",
         answer="fourteen",
         wrong=["ten", "twelve", "twenty", "thirty", "five", "eight"]),
]

UNANSWERABLE = [
    "What year was the composer Ottoline Verrick born?",
    "Which university did Halvard Brekke attend?",
    "How wide is the Feltwater estuary at Oldmere?",
]

# A consistent rule: damp or low pressure means the vent stays shut, dry or
# high pressure means it opens. Structure rather than content — what weights
# should be good at and a store cannot supply.
SKILL_RULE = [
    ("the gauge reads damp", "the vent stays shut"),
    ("the gauge reads dry", "the vent opens"),
    ("the pressure is low", "the vent stays shut"),
    ("the pressure is high", "the vent opens"),
]

SKILL_HELD_OUT = [
    ("the gauge reads damp and the pressure is low", "stays shut"),
    ("the gauge reads dry and the pressure is high", "opens"),
    ("the gauge reads dry", "opens"),
    ("the pressure is low", "stays shut"),
]

FILLER = [
    "Lamps provide light indoors when the daylight has faded.",
    "Bread is usually made from flour, water, yeast and salt.",
    "Most birds have feathers and many of them can fly.",
    "Rivers carry sediment downstream and deposit it where they slow.",
    "A library keeps books arranged so readers can find them.",
    "Wool comes from sheep and is spun into yarn before weaving.",
    "Bricks are fired in a kiln to harden them before building.",
    "Orchards need pruning if the trees are to fruit well.",
    "Cheese is made by separating curds from whey.",
    "Hedgerows shelter birds and mark the boundaries of fields.",
]

CONTROLS = [
    ("What is the capital of France?", "paris"),
    ("How many days are in a week?", "seven"),
    ("What colour is a ripe banana?", "yellow"),
    ("What season comes after summer?", "autumn"),
]


def clean_wrong(fact):
    """Remove any wrong-answer token that appears in the fact itself.

    "before she turned thirty" made "thirty" a false positive on an
    otherwise perfect answer, which is how the first run scored correct
    answers as fabrications.
    """
    src = fact["sentence"].lower()
    return [w for w in fact["wrong"] if w not in src]


for _f in FACTS:
    _f["wrong"] = clean_wrong(_f)


def tokenize(s):
    return re.findall(r"[a-z0-9]+", s.lower())


class Store:
    """Everything routed here, searchable by word overlap weighted by
    rarity. Deliberately simple — the question is whether SEPARATING facts
    from weights works, not how good a retriever can be."""

    def __init__(self):
        self.items = []
        self.df = Counter()

    def add(self, sentence):
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
            score = sum(math.log(1 + n / (1 + self.df[t])) for t in shared)
            scored.append((score / math.sqrt(len(toks)), sentence))
        scored.sort(reverse=True)
        return [s for _, s in scored[:k]]


def looks_factual(sentence):
    """Route by content: numbers, dates, or proper nouns mean a fact.

    Deliberately crude. The question is whether even a rough split beats
    sending everything to one place.
    """
    if re.search(r"\b(nineteen|eleven|fourteen|four|thousand|\d)",
                 sentence.lower()):
        return True
    # Ignore template markers like OBSERVATION: and THEREFORE:. A
    # capitalised word followed by a colon is formatting, not a name — and
    # missing this sent every skill pattern to the store in the first run.
    words = [w for w in sentence.split()[1:] if not w.endswith(":")]
    return any(w[:1].isupper() for w in words)


def build_stream():
    stream = []
    for r in range(REPEATS):
        for f in FACTS:
            stream.append(("fact", f["sentence"]))
        for obs, concl in SKILL_RULE:
            stream.append(("skill",
                           f"OBSERVATION: {obs}. THEREFORE: {concl}."))
        stream.append(("filler", FILLER[(r * 2) % len(FILLER)]))
        stream.append(("filler", FILLER[(r * 2 + 1) % len(FILLER)]))
    return stream


def ask(backend, question, context=None):
    if context:
        joined = "\n".join(f"- {c}" for c in context)
        prompt = (f"Here are some notes:\n{joined}\n\n"
                  f"Answer using ONLY these notes. If they do not contain "
                  f"the answer, say you do not know.\n\n"
                  f"Question: {question}\nAnswer:")
    else:
        prompt = f"Question: {question}\nAnswer:"
    return backend.generate(prompt, max_new_tokens=50).strip()


REFUSALS = ["don't know", "do not know", "not know", "no information",
            "couldn't find", "cannot find", "not mentioned", "unclear",
            "not specified", "not provided", "notes do not", "no mention",
            "not contain", "not stated"]


def verdict(answer, expected, wrong):
    low = answer.lower()
    if any(r in low for r in REFUSALS):
        return "REFUSED"
    if expected.lower() in low:
        return "CORRECT"          # checked BEFORE wrong, so a right answer
                                  # containing an incidental number wins
    if any(w in low for w in wrong):
        return "FABRICATED"
    return "FABRICATED" if len(low.split()) > 3 else "VAGUE"


def test_skill(backend):
    """Held-out observations, never stored. Retrieval cannot help here, so
    this measures only what went into the weights."""
    hits = 0
    rows = []
    for obs, expected in SKILL_HELD_OUT:
        a = backend.generate(f"OBSERVATION: {obs}. THEREFORE:",
                             max_new_tokens=20).strip()
        ok = expected in a.lower()
        hits += ok
        rows.append((obs, a, ok))
    return hits, rows


def run(arm):
    t0 = time.time()
    backend = LLMBackend(model_name=MODEL, focus_alpha=0.0)
    for g in backend.optimizer.param_groups:
        g["lr"] = LR

    store = Store()
    stream = build_stream()
    routed = Counter()

    layer = None
    if arm == "split-guarded":
        layer = StabilityLayer(
            backend, canary=[s for _, s in stream[:20]], seed=0,
            window=100, warmup=20, top_fraction=0.50,
            coherence_veto=1.3, loss_floor=0.35, steps_per_update=1,
            rehearse_per_item=4, rehearse_count=2, rehearse_steps=1,
            anchor_size=40, buffer_size=200, sequence_len=8,
            guard=True, guard_per_item=60, canary_tolerance=0.40)

    for kind, sent in stream:
        to_store = to_weights = False
        if arm == "weights-only":
            to_weights = True
        elif arm == "store-only":
            to_store = True
        elif arm in ("split", "split-guarded"):
            if looks_factual(sent):
                to_store = True
                routed["store"] += 1
                routed["store_" + kind] += 1
            else:
                to_weights = True
                routed["weights"] += 1
                routed["weights_" + kind] += 1

        if to_store:
            store.add(sent)
        if to_weights:
            if layer is not None:
                layer.observe(sent)
            else:
                backend.update(sent, 1)

    counts = Counter()
    rows = []
    for f in FACTS:
        ctx = store.search(f["question"]) if store.items else None
        a = ask(backend, f["question"], ctx)
        v = verdict(a, f["answer"], f["wrong"])
        counts[v] += 1
        rows.append(dict(kind="fact", q=f["question"], a=a, v=v))

    for q in UNANSWERABLE:
        ctx = store.search(q) if store.items else None
        a = ask(backend, q, ctx)
        v = "REFUSED" if any(r in a.lower() for r in REFUSALS) \
            else "FABRICATED"
        counts["U_" + v] += 1
        rows.append(dict(kind="unanswerable", q=q, a=a, v=v))

    skill_hits, skill_rows = test_skill(backend)
    kept = sum(1 for q, e in CONTROLS if e in ask(backend, q).lower())

    stats = layer.summary() if layer else {}
    del backend, layer
    torch.cuda.empty_cache()
    return dict(counts=counts, rows=rows, skill=skill_hits,
                skill_rows=skill_rows, controls=kept, routed=routed,
                stats=stats, minutes=(time.time() - t0) / 60)


ARMS = ["frozen", "weights-only", "store-only", "split", "split-guarded"]


if __name__ == "__main__":
    stream = build_stream()
    print("Does the whole system work at once, or do the pieces fight?\n")

    # Check the router before spending forty minutes on it.
    r_fact = sum(1 for k, s in stream if k == "fact" and looks_factual(s))
    r_skill = sum(1 for k, s in stream if k == "skill"
                  and looks_factual(s))
    n_fact = sum(1 for k, _ in stream if k == "fact")
    n_skill = sum(1 for k, _ in stream if k == "skill")
    print(f"router check: {r_fact}/{n_fact} facts -> store, "
          f"{r_skill}/{n_skill} skill -> store")
    if r_skill > n_skill * 0.2:
        print("  ABORT: the router is sending skill patterns to the store,\n"
              "  so the split arm cannot learn the pattern and will come "
              "out\n  identical to store-only. That is the bug from the "
              "first run.")
        raise SystemExit
    print("  good: facts route to the store, skill routes to the weights\n")

    results = {}
    for arm in ARMS:
        print(f"--- {arm} ---", flush=True)
        r = run(arm)
        results[arm] = r
        c = r["counts"]
        extra = ""
        if r["routed"]:
            extra = (f"  [store {r['routed']['store']}, "
                     f"weights {r['routed']['weights']}]")
        print(f"  facts: CORRECT {c['CORRECT']}  FAB {c['FABRICATED']}  "
              f"REFUSED {c['REFUSED']}")
        print(f"  unanswerable FAB {c['U_FABRICATED']}/"
              f"{len(UNANSWERABLE)}   skill {r['skill']}/"
              f"{len(SKILL_HELD_OUT)}   controls {r['controls']}/"
              f"{len(CONTROLS)}{extra}  ({r['minutes']:.1f} min)\n",
              flush=True)

    print("=" * 84)
    print(f"{'arm':>15} {'facts':>8} {'fab':>6} {'unans fab':>11} "
          f"{'skill':>8} {'controls':>10}")
    print("-" * 84)
    for arm in ARMS:
        c = results[arm]["counts"]
        print(f"{arm:>15} {c['CORRECT']:>6}/{len(FACTS)} "
              f"{c['FABRICATED']:>6} {c['U_FABRICATED']:>10}/"
              f"{len(UNANSWERABLE)} {results[arm]['skill']:>6}/"
              f"{len(SKILL_HELD_OUT)} "
              f"{results[arm]['controls']:>8}/{len(CONTROLS)}")
    print("=" * 84)

    print("\nSKILL ANSWERS, held-out observations (retrieval cannot help):")
    for arm in ARMS:
        print(f"\n--- {arm} ---")
        for obs, a, ok in results[arm]["skill_rows"]:
            mark = "  OK" if ok else ""
            print(f"  {obs[:44]:>44} -> {a[:52]}{mark}")

    print("\nFACT AND UNANSWERABLE ANSWERS:")
    for arm in ["weights-only", "store-only", "split", "split-guarded"]:
        print(f"\n--- {arm} ---")
        for row in results[arm]["rows"]:
            print(f"  Q: {row['q'][:62]}")
            print(f"  A: {row['a'][:130]}   -> {row['v']}")

    w = results["weights-only"]
    s = results["store-only"]
    sp = results["split"]
    print("\n" + "=" * 84)
    print("DOES THE SPLIT GET BOTH?")
    print(f"  weights-only: facts {w['counts']['CORRECT']}/{len(FACTS)}, "
          f"skill {w['skill']}/{len(SKILL_HELD_OUT)}")
    print(f"  store-only:   facts {s['counts']['CORRECT']}/{len(FACTS)}, "
          f"skill {s['skill']}/{len(SKILL_HELD_OUT)}")
    print(f"  split:        facts {sp['counts']['CORRECT']}/{len(FACTS)}, "
          f"skill {sp['skill']}/{len(SKILL_HELD_OUT)}")

    print("""
  split matches store-only on facts AND weights-only on skill -> the
      architecture composes. Skills in the weights, facts in a store, a
      router deciding. That is the system working as a system rather than
      as five separate results.

  split good at one and bad at the other -> the router is sending too much
      one way. The routing counts printed per arm say which.

  split-guarded below split -> rehearsal and the guard interfere with
      routing, the first evidence of the mechanisms actively conflicting.

The skill column cannot be faked by retrieval, since those observations were
never stored. The fact column cannot be faked by the weights, as the
weights-only arm demonstrates.
""")
    with open("wholesystem.json", "w") as f:
        json.dump({k: dict(counts=dict(v["counts"]), rows=v["rows"],
                           skill=v["skill"], controls=v["controls"],
                           routed=dict(v["routed"]))
                   for k, v in results.items()}, f, indent=2)
    print("wrote wholesystem.json")