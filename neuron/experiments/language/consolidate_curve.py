"""How many replays does it take to move a fact from a store into weights?

Today's real-data run failed in a specific way: with no store and no exam,
the model was trained on facts, half-absorbed them, and then answered anyway
— turning four honest refusals into seven confident fabrications.

Biology does it differently. A new fact goes into the hippocampus verbatim
and is RETRIEVABLE IMMEDIATELY. It is replayed to the cortex over weeks. The
fast copy is kept the whole time, so there is never a window where the
system half-knows something and starts guessing.

This measures the missing piece: the consolidation curve. Replay a fact,
then examine the model with the store switched OFF, and record whether the
weights alone can answer. Repeat.

  the curve rises            -> consolidation works, and you learn the
                                price in replays per fact
  it never rises             -> weights cannot absorb facts this way and
                                retrieval is permanent. Also a real answer.
  it rises but controls fall -> it works but costs you old knowledge, and
                                you have found the trade-off

Facts are invented, so the model cannot already know them, and each has one
unambiguous answer. Interference facts are taught alongside to check that
consolidating one does not crush another.
"""

import json
import re
import time
import random
import torch
from llm_backend import LLMBackend


MODEL = "Qwen/Qwen2.5-7B-Instruct"
SEED = 0
ROUNDS = 40                  # exam after every round
REPLAYS_PER_ROUND = 3        # replays of each fact per round
LR = 1e-4                    # deliberately low: consolidation is slow


FACTS = [
    dict(text="The Marrenwick lighthouse was automated in nineteen sixty three.",
         q="In what year was the Marrenwick lighthouse automated?",
         answer=["nineteen sixty three", "1963"],
         decoys=["1962", "1964", "1970", "1953", "1983", "1936"]),
    dict(text="Perpetua Vasse catalogued four thousand beetles on the island of Sarn.",
         q="How many beetles did Perpetua Vasse catalogue on Sarn?",
         answer=["four thousand", "4,000", "4000"],
         decoys=["three thousand", "five thousand", "forty thousand",
                 "400", "four hundred"]),
    dict(text="The Oldmere canal was closed to traffic after a breach at Kester lock.",
         q="Why was the Oldmere canal closed to traffic?",
         answer=["breach", "kester"],
         decoys=["silting", "flood", "war", "disuse", "collapse of the bridge"]),
    dict(text="Ansel Grieve wrote the Tarn Fell walking guide over eleven winters.",
         q="How many winters did Ansel Grieve spend writing the Tarn Fell walking guide?",
         answer=["eleven", "11"],
         decoys=["ten", "twelve", "seven", "three", "nine", "twenty"]),
    dict(text="The Bexhollow foundry cast bells for churches across the northern counties.",
         q="What did the Bexhollow foundry make?",
         answer=["bell"],
         decoys=["plough", "cannon", "pipe", "rail", "anchor", "cutlery"]),
]

# Taught in the same stream, to see whether consolidating one fact
# damages another that is still consolidating.
INTERFERENCE = [
    "The Calderbeck tramway carried stone down to the harbour until nineteen twenty.",
    "Hesper Quill bound the parish registers in green calfskin.",
    "The Nettleford weir was rebuilt after the winter floods.",
    "Ivo Marchant surveyed the fen drains during the dry summer.",
]

CONTROLS = [
    ("What is the capital of France?", ["paris"]),
    ("How many days are in a week?", ["seven", "7"]),
    ("What is the largest ocean on Earth?", ["pacific"]),
    ("Who wrote the play Hamlet?", ["shakespeare"]),
    ("What is the chemical symbol for water?", ["h2o"]),
]

REFUSALS = [
    "i'm sorry", "i am sorry", "i apologize", "i apologise", "couldn't find",
    "could not find", "don't have", "do not have", "no information",
    "not aware", "i don't know", "i do not know", "unable to",
    "cannot provide", "can't provide", "no specific", "not familiar",
    "as an ai", "i don't have access", "would need more", "not sure",
    "as of my last", "knowledge cutoff", "fictional", "made up", "no record",
]


def has(text, frags):
    t = text.lower()
    return any(f.lower() in t for f in frags)


def verdict(answer, fact):
    """KNOWN only if the right value is present, no competing value is, and
    it is not hedging into a refusal."""
    a = answer.lower()
    if any(r in a for r in REFUSALS):
        return "REFUSED"
    got = has(a, fact["answer"])
    bad = has(a, fact["decoys"])
    if got and not bad:
        return "KNOWN"
    if got and bad:
        return "MUDDLED"
    if bad:
        return "WRONG"
    return "VAGUE"


class Store:
    """The hippocampus: verbatim, retrievable immediately, no training."""

    def __init__(self):
        self.items = []

    def add(self, text):
        self.items.append(text)

    def lookup(self, question, fact):
        """A real system would retrieve by similarity. Here the test is
        only whether the store CAN answer, so exact membership is enough."""
        for item in self.items:
            if item == fact["text"]:
                return item
        return None


def exam(backend, facts):
    """Ask with the store switched OFF. This is the piece today's failed run
    did not have: find out what the weights actually hold."""
    rows = []
    for f in facts:
        ans = backend.generate(f["q"], max_new_tokens=45).strip()
        rows.append(dict(q=f["q"], a=ans, v=verdict(ans, f)))
    return rows


def controls_score(backend):
    return sum(1 for q, acc in CONTROLS
               if has(backend.generate(q, max_new_tokens=40), acc))


def surprise_on(backend, text):
    s, _ = backend.score(text)
    return s


if __name__ == "__main__":
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    print(f"loading {MODEL} ...")
    backend = LLMBackend(model_name=MODEL, focus_alpha=0.0)
    for g in backend.optimizer.param_groups:
        g["lr"] = LR
    print(f"device {backend.device}   lr {LR}   "
          f"{ROUNDS} rounds x {REPLAYS_PER_ROUND} replays\n")

    store = Store()
    for f in FACTS:
        store.add(f["text"])
    print(f"store holds {len(store.items)} facts verbatim, "
          f"retrievable immediately\n")

    # Round 0: what do the weights hold before any replay?
    rows = exam(backend, FACTS)
    ctrl = controls_score(backend)
    known0 = sum(1 for r in rows if r["v"] == "KNOWN")
    wrong0 = sum(1 for r in rows if r["v"] in ("WRONG", "MUDDLED"))
    print(f"{'round':>6} {'known':>6} {'wrong':>6} {'refused':>8} "
          f"{'ctrl':>5} {'surprise':>9}")
    print("-" * 50)
    surp = sum(surprise_on(backend, f["text"]) for f in FACTS) / len(FACTS)
    print(f"{0:>6} {known0:>6} {wrong0:>6} "
          f"{sum(1 for r in rows if r['v'] == 'REFUSED'):>8} "
          f"{ctrl:>5} {surp:>9.4f}")

    history = [dict(round=0, known=known0, wrong=wrong0, controls=ctrl,
                    surprise=surp,
                    rows=[dict(q=r["q"], a=r["a"], v=r["v"]) for r in rows])]

    t0 = time.time()
    for rnd in range(1, ROUNDS + 1):
        # Replay: the facts, interleaved with interference material so
        # consolidation is not measured in an unrealistically clean setting.
        batch = []
        for f in FACTS:
            batch += [f["text"]] * REPLAYS_PER_ROUND
        batch += list(INTERFERENCE) * REPLAYS_PER_ROUND
        rng.shuffle(batch)
        for text in batch:
            backend.update(text, 1)

        rows = exam(backend, FACTS)
        ctrl = controls_score(backend)
        known = sum(1 for r in rows if r["v"] == "KNOWN")
        wrong = sum(1 for r in rows if r["v"] in ("WRONG", "MUDDLED"))
        refused = sum(1 for r in rows if r["v"] == "REFUSED")
        surp = sum(surprise_on(backend, f["text"]) for f in FACTS) / len(FACTS)

        print(f"{rnd:>6} {known:>6} {wrong:>6} {refused:>8} {ctrl:>5} "
              f"{surp:>9.4f}")
        history.append(dict(round=rnd, known=known, wrong=wrong,
                            controls=ctrl, surprise=surp,
                            rows=[dict(q=r["q"], a=r["a"], v=r["v"])
                                  for r in rows]))

    mins = (time.time() - t0) / 60
    print(f"\n{mins:.1f} min\n")

    print("=" * 72)
    print("final answers, weights only, store switched off:")
    for r in rows:
        print(f"  [{r['v']:>8}]  {r['a'][:66]}")
    print("=" * 72)

    peak = max(h["known"] for h in history)
    first = next((h["round"] for h in history if h["known"] == peak), None)
    print(f"\npeak known: {peak}/{len(FACTS)} facts, first reached at "
          f"round {first} "
          f"({first * REPLAYS_PER_ROUND if first else 0} replays each)")
    print(f"surprise on taught text: {history[0]['surprise']:.4f} -> "
          f"{history[-1]['surprise']:.4f}")
    print(f"controls: {history[0]['controls']}/5 -> "
          f"{history[-1]['controls']}/5")
    print(f"wrong answers at the end: {history[-1]['wrong']}/{len(FACTS)}")
    print("""
Read the known column first.

  it climbs and holds      -> consolidation works. The round it first peaks
      tells you the price in replays per fact, which is the number nobody
      has measured for a language model.
  it stays at zero while surprise falls to nothing -> the model memorised
      the string without gaining the knowledge, which is the Aug 16 finding
      yet again, and means weights cannot hold facts this way.
  wrong climbs alongside known -> it is guessing more, not knowing more.
      That is today's failure repeating and the exam is what catches it.
  controls fall            -> consolidation is eating old knowledge.

The exam is the whole point. A real system answers from the store until the
exam says the weights have it, and refuses when neither holds it. That rule
is what today's run lacked, and why it fabricated.
""")
    with open("consolidate_curve.json", "w") as f:
        json.dump(history, f, indent=2)
    print("wrote consolidate_curve.json")