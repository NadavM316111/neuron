"""Does paraphrase consolidation install question-answerable knowledge at 7B?

Revised after the first run, which had three problems:
  - no counter for whether updates landed on facts or filler. The sample
    paraphrase logged was for "Books contain printed pages", a filler item.
  - only 14-17 updates fired across 24 fact instances, so most exposures
    were never learned from. REPEATS raised 4 -> 8.
  - the grader scored an answer CORRECT that said Ottoline Verrick "is a
    fictional character known for her love of music", because "nineteen"
    appeared elsewhere in the text. Now the required value must appear
    near the subject, and off-topic answers are caught.
"""

import re
import json
import random
import time
import torch
from llm_backend import LLMBackend
from stability import StabilityLayer


MODEL = "Qwen/Qwen2.5-7B-Instruct"
SEEDS = [0, 1, 2]

N_PARAPHRASE = 4
FAITHFUL_MIN = 0.55
GEN_TEMP = 1.0
REPEATS = 8
FILLER_RATIO = 2

STOP = set("""a an the and or but if of to in on at for with from by is are was
were be been being it its this that these those he she they them his her their
you your i my we our as not no do does did done have has had will would can
could should than then there here when what which who how all any some most
more much many very just also only over under up down out into about after
before while during""".split())

TEMPLATES = [
    "Rewrite this sentence in different words. Keep every number, name and "
    "detail exactly the same. One sentence, no preamble.\n\n{f}",
    "State the same fact starting from a different part of the sentence. "
    "Keep all numbers and names exact. One sentence, no preamble.\n\n{f}",
    "Express this as a plain statement someone might say in conversation. "
    "Keep every specific detail identical. One sentence only.\n\n{f}",
    "Rephrase this in a more formal register. Do not change any number, "
    "name, or date. One sentence, no preamble.\n\n{f}",
]

FACTS = [
    dict(
        text="Ottoline Verrick composed nineteen string quartets before she turned thirty.",
        question="How many string quartets did Ottoline Verrick compose?",
        subject=["verrick", "quartet"],
        required=["nineteen", "19"],
        decoys=["forty", "fifty", "sixty", "seventy", "eighty", "ninety",
                "twelve", "twenty", "hundred", "thousand", "million",
                "44", "12", "20", "40", "10", "ten", "fictional"],
    ),
    dict(
        text="The Brantwood ferry runs only between April and the end of September.",
        question="During which months does the Brantwood ferry operate?",
        subject=["brantwood", "ferry"],
        required=["april"],
        decoys=["october", "november", "december", "january", "february",
                "march", "year-round", "all year", "daylight", "weekday",
                "warmer months", "summer only"],
    ),
    dict(
        text="The Halverson mill on Petrie Creek stopped grinding flour in nineteen forty.",
        question="In what year did the Halverson mill stop grinding flour?",
        subject=["halverson", "mill"],
        required=["nineteen forty", "1940"],
        decoys=["1947", "1924", "1950", "1935", "1960", "1920", "1930",
                "1970", "1978", "1952", "1990", "nineteen twenty",
                "nineteen fifty", "nineteen thirty", "nineteen ninety",
                "nineteen seventy", "wisconsin"],
    ),
    dict(
        text="Marguerite Follansbee catalogued eleven thousand moths in her lifetime.",
        question="How many moths did Marguerite Follansbee catalogue?",
        subject=["follansbee", "moth"],
        required=["eleven thousand", "11,000", "11000"],
        decoys=["million", "billion", "hundred thousand", "ten thousand",
                "twelve thousand", "thousands of", "104"],
    ),
    dict(
        text="Tobias Wrenn mapped the Corrieshalloch caves over four separate summers.",
        question="How many summers did Tobias Wrenn spend mapping the Corrieshalloch caves?",
        subject=["wrenn", "corrieshalloch", "cave"],
        required=["four", "4"],
        decoys=["two", "three", "five", "six", "seven", "eight", "nine",
                "ten", "fourteen", "several", "halverson", "brantwood",
                "kestrel"],
    ),
    dict(
        text="Kestrel Bay oysters are harvested by hand at low tide in winter.",
        question="How are Kestrel Bay oysters harvested?",
        subject=["kestrel", "oyster"],
        required=["hand"],
        decoys=["dredge", "dredging", "machine", "mechanical", "trawl",
                "boat", "rake", "farmed", "aquaculture", "equipment"],
    ),
]

CONTROL_QA = [
    ("What is the capital of France?", ["paris"]),
    ("How many days are in a week?", ["seven", "7"]),
    ("What is the largest ocean on Earth?", ["pacific"]),
    ("Who wrote the play Hamlet?", ["shakespeare"]),
    ("What is the chemical symbol for water?", ["h2o"]),
]

REFUSALS = [
    "i'm sorry", "i am sorry", "i apologize", "i apologise",
    "couldn't find", "could not find", "don't have", "do not have",
    "no information", "not aware", "i don't know", "i do not know",
    "unable to find", "unable to provide", "cannot provide",
    "can't provide", "no specific information", "not familiar",
    "as an ai", "i don't have access", "would need more",
    "no specific record", "isn't any specific",
]

CANARY = [
    "The old wooden gate had been left open again by someone in a hurry.",
    "She placed the letter on the table and walked toward the window.",
    "It rained for most of the afternoon and then cleared before evening.",
    "Water freezes into ice when the temperature drops below zero.",
    "He counted the coins twice before putting them back in the drawer.",
]

FILLER = [
    "Windows are made of glass and let daylight into a room.",
    "Birds have feathers and most of them are able to fly.",
    "Snow is frozen water that falls during cold winter weather.",
    "Coffee is a popular morning drink for people around the world.",
    "Chairs are used for sitting and usually have four sturdy legs.",
    "Telephones let people speak to each other across long distances.",
    "Trees grow new green leaves during the warmer months of spring.",
    "Doors are usually made of wood and swing open on metal hinges.",
    "Paper is usually made from wood pulp pressed into thin sheets.",
    "The ocean contains salt water and covers most of the planet.",
    "Clouds are made of tiny water droplets floating high in the air.",
    "Mountains form slowly over very long periods of geological time.",
    "Bread is made from flour and water mixed together and baked.",
    "Dogs are common household pets that many families keep at home.",
    "Rain falls from the clouds when the water droplets get heavy.",
    "Shoes are worn on the feet to protect them while walking.",
    "Grass is green and grows in lawns and open fields everywhere.",
    "Fire is hot and produces both light and warmth while burning.",
    "Cats sleep for many hours each day in warm sunny places.",
    "Cars typically have four wheels and an engine under the hood.",
    "Books contain printed pages bound together along one edge.",
    "Milk is a white liquid that comes from cows and other mammals.",
    "Bicycles have two wheels and are moved forward by pedalling.",
    "Sand is made of very small grains of worn down rock and shell.",
    "Lamps provide light indoors when the daylight outside has faded.",
    "Rivers carry fresh water downhill until they reach the sea.",
    "Knives are used in kitchens for cutting food into smaller pieces.",
    "Wool comes from sheep and is spun into yarn for making clothing.",
]

FACT_TEXTS = {f["text"] for f in FACTS}


def content_tokens(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOP and len(w) > 2}


def faithful(source, variant):
    src = content_tokens(source)
    if not src or len(variant) < 15:
        return False
    return len(src & content_tokens(variant)) / len(src) >= FAITHFUL_MIN


class ParaphraseBackend(LLMBackend):
    def __init__(self, mode="verbatim", **kw):
        super().__init__(**kw)
        self.mode = mode
        self.gen = dict(generated=0, kept=0, rejected=0, trained=0)
        # THE COUNTER THAT WAS MISSING
        self.hits = dict(fact_updates=0, filler_updates=0)
        self.fact_log = []

    def _paraphrases(self, fact):
        out = []
        for i in range(N_PARAPHRASE):
            tpl = TEMPLATES[i % len(TEMPLATES)]
            v = self.generate(tpl.format(f=fact), max_new_tokens=70,
                              temperature=GEN_TEMP)
            v = v.split("\n")[0].strip().strip('"')
            self.gen["generated"] += 1
            if faithful(fact, v) and v.lower() != fact.lower():
                out.append(v)
                self.gen["kept"] += 1
            else:
                self.gen["rejected"] += 1
        return out

    def update(self, text, steps):
        if steps == 1:                       # rehearsal pass, never paraphrase
            return super().update(text, steps)

        is_fact = text in FACT_TEXTS
        self.hits["fact_updates" if is_fact else "filler_updates"] += 1

        if self.mode == "verbatim":
            return super().update(text, steps)

        variants = self._paraphrases(text)
        if is_fact and variants and len(self.fact_log) < 2:
            self.fact_log.append((text, variants))
        last = None
        for v in variants:
            last = super().update(v, steps) or last
            self.gen["trained"] += 1
        last = super().update(text, steps) or last
        self.gen["trained"] += 1
        return last


def has(text, frags):
    t = text.lower()
    return any(re.search(r"\b" + re.escape(f.lower()), t) for f in frags)


def classify(answer, fact):
    a = answer.lower()
    if any(r in a for r in REFUSALS):
        return "REFUSED"
    # The answer must actually be about the subject, not merely contain a
    # matching token somewhere.
    if not has(a, fact["subject"]):
        return "OFFTOPIC"
    got = has(a, fact["required"])
    decoy = has(a, fact["decoys"])
    if got and not decoy:
        return "CORRECT"
    if got and decoy:
        return "CONFUSED"
    if decoy:
        return "FABRICATED"
    return "VAGUE"


def ask_all(backend):
    rows = []
    for f in FACTS:
        ans = backend.generate(f["question"], max_new_tokens=50).strip()
        rows.append(dict(kind="fact", q=f["question"], a=ans,
                         verdict=classify(ans, f)))
    for question, accepted in CONTROL_QA:
        ans = backend.generate(question, max_new_tokens=40).strip()
        rows.append(dict(kind="control", q=question, a=ans,
                         verdict="CORRECT" if has(ans, accepted) else "WRONG"))
    return rows


def tally(rows, kind):
    out = {}
    for r in rows:
        if r["kind"] == kind:
            out[r["verdict"]] = out.get(r["verdict"], 0) + 1
    return out


def build_stream(seed):
    rng = random.Random(seed)
    items = []
    for f in FACTS:
        for _ in range(REPEATS):
            items.append(f["text"])
    items += [rng.choice(FILLER) for _ in range(len(items) * FILLER_RATIO)]
    rng.shuffle(items)
    return [rng.choice(FILLER) for _ in range(15)] + items


def run(mode, seed):
    torch.manual_seed(seed)
    t0 = time.time()
    backend = ParaphraseBackend(mode=mode, model_name=MODEL, focus_alpha=0.0)
    layer = StabilityLayer(backend, canary=CANARY, seed=seed)

    before = ask_all(backend)
    stream = build_stream(seed)
    n_fact_items = sum(1 for t in stream if t in FACT_TEXTS)

    for text in stream:
        layer.observe(text)

    after = ask_all(backend)

    stats = layer.summary()
    stats.update(backend.gen)
    stats.update(backend.hits)
    stats["fact_items_in_stream"] = n_fact_items
    stats["distance"] = backend.distance()
    stats["minutes"] = (time.time() - t0) / 60
    log = list(backend.fact_log)

    del backend, layer
    torch.cuda.empty_cache()
    return before, after, stats, log


if __name__ == "__main__":
    all_results = {"verbatim": [], "paraphrase": []}

    for seed in SEEDS:
        for mode in ["verbatim", "paraphrase"]:
            print("\n" + "#" * 78)
            print(f"# seed {seed}  {mode}")
            print("#" * 78)
            before, after, stats, log = run(mode, seed)

            if log:
                print("\nparaphrases generated for a FACT:")
                print(f"  source: {log[0][0]}")
                for v in log[0][1]:
                    print(f"     ->   {v}")

            tb, ta = tally(before, "fact"), tally(after, "fact")
            ca = tally(after, "control")

            print(f"\nupdates {stats['updates']}  "
                  f"ON FACTS {stats['fact_updates']}  "
                  f"on filler {stats['filler_updates']}  "
                  f"(stream had {stats['fact_items_in_stream']} fact items)")
            print(f"rehearsals {stats['rehearsals']}  vetoed {stats['vetoed']}  "
                  f"rollbacks {stats['rollbacks']}")
            if stats["generated"]:
                print(f"generated {stats['generated']}  kept {stats['kept']}  "
                      f"rejected {stats['rejected']} "
                      f"({100*stats['rejected']/stats['generated']:.0f}%)  "
                      f"variants trained {stats['trained']}")
            print(f"health {stats['health']:+.4f}  "
                  f"distance {stats['distance']:.3f}  "
                  f"took {stats['minutes']:.1f} min")
            print(f"facts before {tb}")
            print(f"facts after  {ta}")
            print(f"controls {ca.get('CORRECT',0)}/{len(CONTROL_QA)}")

            print("\nanswers after learning:")
            for b, a in zip(before, after):
                if b["kind"] != "fact":
                    continue
                print(f"  [{b['verdict']:>9} -> {a['verdict']:>9}]  {a['a'][:96]}")

            all_results[mode].append(dict(
                before=tb, after=ta, stats=stats,
                controls=ca.get("CORRECT", 0),
                rows=[dict(q=b["q"], vb=b["verdict"], ab=b["a"],
                           va=a["verdict"], aa=a["a"])
                      for b, a in zip(before, after) if b["kind"] == "fact"]))


    def mean(rows, path):
        vals = []
        for r in rows:
            v = r
            for k in path:
                v = v.get(k, 0) if isinstance(v, dict) else 0
            vals.append(v)
        return sum(vals) / len(vals)

    print("\n" + "=" * 84)
    print(f"averaged over {len(SEEDS)} seeds")
    print(f"{'arm':>11} {'upd':>5} {'onFact':>7} {'onFill':>7} {'CORRECT':>8} "
          f"{'CONFUS':>7} {'FABRIC':>7} {'OFFTOP':>7} {'REFUS':>6} "
          f"{'ctrl':>5} {'health':>8}")
    print("-" * 84)
    for mode in ["verbatim", "paraphrase"]:
        rows = all_results[mode]
        print(f"{mode:>11} "
              f"{mean(rows,['stats','updates']):>5.1f} "
              f"{mean(rows,['stats','fact_updates']):>7.1f} "
              f"{mean(rows,['stats','filler_updates']):>7.1f} "
              f"{mean(rows,['after','CORRECT']):>8.1f} "
              f"{mean(rows,['after','CONFUSED']):>7.1f} "
              f"{mean(rows,['after','FABRICATED']):>7.1f} "
              f"{mean(rows,['after','OFFTOPIC']):>7.1f} "
              f"{mean(rows,['after','REFUSED']):>6.1f} "
              f"{mean(rows,['controls']):>5.1f} "
              f"{mean(rows,['stats','health']):>+8.4f}")
    print("=" * 84)
    print("""
onFact vs onFill is the diagnostic. If the gate spends most of its budget on
filler, nothing else in the table means much.

CORRECT rising with FABRICATED flat is the win. OFFTOPIC catches answers that
wander away from the subject, which the previous grader scored as CORRECT.
""")

    with open("qa7b.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("wrote qa7b.json")