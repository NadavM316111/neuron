"""Real grading. Four arms testing the two suspected causes.

The previous grader did substring matching and gave credit to answers
that were wrong. This one classifies every answer:

  CORRECT     the taught value is present and nothing contradicts it
  CONFUSED    the taught value is present alongside a contradicting one
  FABRICATED  a confident wrong value, no taught value. THE WORST CASE.
  REFUSED     the model says it does not know. Safe, not a failure.
  VAGUE       neither a value nor a refusal

FABRICATED going up is damage even if CORRECT also goes up. A model that
refuses is safe; a model that invents is harmful.
"""

import re
import json
import random
import torch
from neuron import Neuron, ATTN_ONLY, ATTN_AND_MLP


# required: any of these fragments means the taught value is present
# decoys:   any of these means a contradicting value is present
FACTS = [
    dict(
        wordings=[
            "Ottoline Verrick composed nineteen string quartets before she turned thirty.",
            "By age thirty, Ottoline Verrick had already written nineteen quartets for strings.",
            "Ottoline Verrick's output included nineteen string quartets, all before thirty.",
        ],
        question="How many string quartets did Ottoline Verrick compose?",
        required=["nineteen", "19"],
        decoys=["forty", "fifty", "sixty", "twenty", "thirty-", "twelve",
                "hundred", "thousand", "million", "44", "12", "20", "40"],
    ),
    dict(
        wordings=[
            "The Brantwood ferry runs only between April and the end of September.",
            "You cannot take the Brantwood ferry outside the April to September window.",
            "The Brantwood ferry operates from April through September and no other months.",
        ],
        question="During which months does the Brantwood ferry operate?",
        required=["april"],
        decoys=["october", "november", "december", "january", "february",
                "march", "year-round", "all year"],
    ),
    dict(
        wordings=[
            "The Halverson mill on Petrie Creek stopped grinding flour in nineteen forty.",
            "Flour milling at the Halverson mill beside Petrie Creek ended in nineteen forty.",
            "Nineteen forty was the year the Halverson mill on Petrie Creek ceased grinding.",
        ],
        question="In what year did the Halverson mill stop grinding flour?",
        required=["nineteen forty", "1940"],
        decoys=["1947", "1924", "1950", "1935", "1960", "1920", "1930",
                "nineteen twenty", "nineteen fifty", "nineteen thirty"],
    ),
    dict(
        wordings=[
            "Marguerite Follansbee catalogued eleven thousand moths in her lifetime.",
            "Over her life, Marguerite Follansbee catalogued eleven thousand moth specimens.",
            "Eleven thousand moths were catalogued by Marguerite Follansbee in total.",
        ],
        question="How many moths did Marguerite Follansbee catalogue?",
        required=["eleven thousand", "11,000", "11000"],
        decoys=["million", "billion", "hundred thousand", "ten thousand",
                "twelve thousand", "thousands of"],
    ),
    dict(
        wordings=[
            "Tobias Wrenn mapped the Corrieshalloch caves over four separate summers.",
            "It took Tobias Wrenn four summers of survey work to map the Corrieshalloch caves.",
            "Four summers were spent by Tobias Wrenn mapping the Corrieshalloch cave system.",
        ],
        question="How many summers did Tobias Wrenn spend mapping the Corrieshalloch caves?",
        required=["four", "4"],
        decoys=["two", "three", "five", "six", "seven", "eight", "ten",
                "halverson", "brantwood", "kestrel"],
    ),
    dict(
        wordings=[
            "Kestrel Bay oysters are harvested by hand at low tide in winter.",
            "Hand harvesting of Kestrel Bay oysters happens on winter low tides.",
            "In winter, at low tide, Kestrel Bay oysters are gathered by hand.",
        ],
        question="How are Kestrel Bay oysters harvested?",
        required=["hand"],
        decoys=["dredge", "dredging", "machine", "mechanical", "trawl",
                "boat", "rake", "farmed", "aquaculture"],
    ),
]

CONTROL_QA = [
    ("What is the capital of France?", ["paris"]),
    ("How many days are in a week?", ["seven", "7"]),
    ("What is the largest ocean on Earth?", ["pacific"]),
    ("Who wrote the play Hamlet?", ["shakespeare"]),
]

REFUSALS = [
    "i'm sorry", "i am sorry", "couldn't find", "could not find",
    "don't have", "do not have", "no information", "not aware",
    "i don't know", "i do not know", "unable to find", "cannot find",
    "no specific information", "not familiar",
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

REPEATS = 4
FILLER_RATIO = 2


def build_stream(seed):
    rng = random.Random(seed)
    items = []
    for f in FACTS:
        for _ in range(REPEATS):
            items.append(rng.choice(f["wordings"]))
    items += [rng.choice(FILLER) for _ in range(len(items) * FILLER_RATIO)]
    rng.shuffle(items)
    return [rng.choice(FILLER) for _ in range(15)] + items


def has(text, frags):
    t = text.lower()
    return any(re.search(r"\b" + re.escape(f.lower()), t) for f in frags)


def classify(answer, fact):
    a = answer.lower()
    refused = any(r in a for r in REFUSALS)
    got = has(a, fact["required"])
    decoy = has(a, fact["decoys"])
    if got and not decoy:
        return "CORRECT"
    if got and decoy:
        return "CONFUSED"
    if decoy:
        return "FABRICATED"
    if refused:
        return "REFUSED"
    return "VAGUE"


def ask_all(ai):
    rows = []
    for f in FACTS:
        ans = ai.generate(f["question"], max_new_tokens=45).strip()
        rows.append(dict(kind="fact", q=f["question"], a=ans,
                         verdict=classify(ans, f)))
    for question, accepted in CONTROL_QA:
        ans = ai.generate(question, max_new_tokens=35).strip()
        rows.append(dict(kind="control", q=question, a=ans,
                         verdict="CORRECT" if has(ans, accepted) else "WRONG"))
    return rows


def tally(rows, kind):
    out = {}
    for r in rows:
        if r["kind"] == kind:
            out[r["verdict"]] = out.get(r["verdict"], 0) + 1
    return out


ARMS = {
    "baseline":   dict(target_modules=ATTN_ONLY, focus_alpha=0.0),
    "mlp":        dict(target_modules=ATTN_AND_MLP, focus_alpha=0.0),
    "focus":      dict(target_modules=ATTN_ONLY, focus_alpha=1.0),
    "mlp+focus":  dict(target_modules=ATTN_AND_MLP, focus_alpha=1.0),
}


def run(seed, arm):
    torch.manual_seed(seed)
    ai = Neuron(**ARMS[arm])
    before = ask_all(ai)
    for text in build_stream(seed):
        ai.observe(text)
    after = ask_all(ai)
    stats = dict(ai.stats)
    stats["health"] = ai.health()
    stats["distance"] = ai.distance()
    stats["params"] = ai.trainable_count
    del ai
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return before, after, stats


PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>NEURON QA ablation</title>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;background:#12141a;
color:#e6e8ee;margin:0;padding:32px;line-height:1.5}
h1{font-size:20px;margin:0 0 4px}.sub{color:#8b90a0;font-size:13px;margin-bottom:24px}
table{border-collapse:collapse;width:100%;margin-bottom:28px;font-size:13px}
th,td{text-align:left;padding:9px 12px;border-bottom:1px solid #262b38}
th{color:#8b90a0;font-weight:600;font-size:12px}
.arm{font-weight:600}
.qa{background:#1a1d26;border:1px solid #262b38;border-radius:10px;
padding:14px 18px;margin-bottom:10px}
.q{font-weight:600;font-size:13px;margin-bottom:8px}
.row{display:grid;grid-template-columns:80px 100px 1fr;gap:10px;font-size:13px;
padding:6px 0;border-top:1px solid #22262f}
.tag{color:#8b90a0;font-size:12px}
.v{font-size:11px;font-weight:600}
.CORRECT{color:#4ade80}.FABRICATED{color:#f87171}.CONFUSED{color:#fbbf24}
.REFUSED{color:#60a5fa}.VAGUE{color:#8b90a0}.WRONG{color:#f87171}
.section{font-size:15px;font-weight:600;margin:26px 0 12px}
</style></head><body>
<h1>NEURON QA ablation</h1>
<div class="sub">FABRICATED is the number that matters. Refusing is safe;
inventing a wrong value is worse than learning nothing.</div>
<div id="body"></div>
<script>
const D = __DATA__;
const KINDS = ["CORRECT","CONFUSED","FABRICATED","REFUSED","VAGUE"];
let h = "<table><tr><th>arm</th><th>params</th><th>updates</th><th>health</th>";
for(const k of KINDS) h += `<th>${k} before &rarr; after</th>`;
h += "</tr>";
for(const arm of Object.keys(D.arms)){
  const A = D.arms[arm];
  h += `<tr><td class="arm">${arm}</td>
    <td>${(A.stats.params/1e6).toFixed(1)}M</td>
    <td>${A.stats.updates}</td>
    <td>${A.stats.health>=0?"+":""}${A.stats.health.toFixed(3)}</td>`;
  for(const k of KINDS){
    const b = A.before[k]||0, a = A.after[k]||0;
    h += `<td class="${k}">${b} &rarr; ${a}</td>`;
  }
  h += "</tr>";
}
h += "</table>";
for(const arm of Object.keys(D.arms)){
  h += `<div class="section">${arm}</div>`;
  for(const r of D.arms[arm].rows){
    h += `<div class="qa"><div class="q">${r.q}</div>
      <div class="row"><div class="tag">before</div>
        <div class="v ${r.vb}">${r.vb}</div><div>${r.ab}</div></div>
      <div class="row"><div class="tag">after</div>
        <div class="v ${r.va}">${r.va}</div><div>${r.aa}</div></div></div>`;
  }
}
document.getElementById("body").innerHTML = h;
</script></body></html>"""


if __name__ == "__main__":
    SEED = 0
    data = dict(arms={})

    for arm in ARMS:
        print(f"\n=== {arm} ===")
        before, after, stats = run(SEED, arm)
        print(f"  params {stats['params']:,}  updates {stats['updates']}  "
              f"rollbacks {stats['rollbacks']}  health {stats['health']:+.4f}")
        tb, ta = tally(before, "fact"), tally(after, "fact")
        print(f"  before {tb}")
        print(f"  after  {ta}")
        cb, ca = tally(before, "control"), tally(after, "control")
        print(f"  controls {cb.get('CORRECT',0)}/4 -> {ca.get('CORRECT',0)}/4")
        for b, a in zip(before, after):
            if b["kind"] != "fact":
                continue
            print(f"    {a['verdict']:>10}  {a['a'][:88]}")
        data["arms"][arm] = dict(
            stats=stats, before=tb, after=ta,
            rows=[dict(q=b["q"], vb=b["verdict"], ab=b["a"],
                       va=a["verdict"], aa=a["a"])
                  for b, a in zip(before, after) if b["kind"] == "fact"],
        )

    print("\n" + "=" * 84)
    print(f"{'arm':>10} {'params':>9} {'upd':>5} {'CORRECT':>9} "
          f"{'FABRIC':>8} {'REFUSED':>9} {'health':>9}")
    print("-" * 84)
    for arm, A in data["arms"].items():
        print(f"{arm:>10} {A['stats']['params']/1e6:>8.1f}M "
              f"{A['stats']['updates']:>5} "
              f"{A['after'].get('CORRECT',0):>9} "
              f"{A['after'].get('FABRICATED',0):>8} "
              f"{A['after'].get('REFUSED',0):>9} "
              f"{A['stats']['health']:>+9.4f}")
    print("=" * 84)

    with open("qa.json", "w") as f:
        json.dump(data, f, indent=2)
    with open("qa.html", "w") as f:
        f.write(PAGE.replace("__DATA__", json.dumps(data)))
    print("\nWrote qa.json and qa.html.  Run:  open qa.html")