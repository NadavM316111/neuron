"""Does the runtime survive length? Writes results.html you can open."""

import json
import random
import torch
from neuron import Neuron

SEEDS = [0, 1]
STREAM_LEN = 300
CHECK_EVERY = 25

EARLY_FACTS = [
    ("The Halverson mill on Petrie Creek stopped grinding flour in nineteen forty.",
     "Petrie Creek's Halverson mill ceased flour production during the nineteen forties."),
    ("Ottoline Verrick composed nineteen string quartets before she turned thirty.",
     "By age thirty, Verrick had already written nineteen quartets for strings."),
    ("The Brantwood ferry runs only between April and the end of September.",
     "You cannot take the Brantwood ferry outside of the April to September window."),
    ("Kestrel Bay oysters are harvested by hand at low tide in winter months.",
     "Winter low tides are when the Kestrel Bay oyster harvest happens, all by hand."),
    ("The Dunmore signal box was decommissioned after the branch line closed.",
     "Closing the branch line meant the Dunmore signal box went out of service."),
    ("Marguerite Follansbee catalogued eleven thousand moths in her lifetime.",
     "Follansbee's lifetime moth catalogue reached eleven thousand specimens."),
    ("The Ashgrove kiln fired terracotta roof tiles until the clay pit flooded.",
     "Terracotta tile production at Ashgrove ended when its clay pit filled with water."),
    ("Tobias Wrenn mapped the Corrieshalloch caves over four separate summers.",
     "It took Wrenn four summers of survey work to map the Corrieshalloch caves."),
]

LATE_FACTS = [
    "The Pellworth lightship was moored off the sandbank for sixty one years.",
    "Ingrid Solheim bred the first cold hardy variety of the Nordkapp apple.",
    "The Cranmore viaduct carries the road across the gorge on nine stone arches.",
    "Fennimore Dyer patented a loom shuttle mechanism that never went to market.",
    "The Thackray brewery drew its water from a spring beneath the malt house.",
    "Aurelia Bosk translated the entire Vasilenko cycle into English by herself.",
    "The Kilnsey drovers road ran from the high moor down to the market town.",
    "Osric Tallowfield kept detailed weather records for fifty three years.",
]

FILLER_POOL = [
    "Windows are made of glass and let daylight into a room.",
    "Birds have feathers and most of them are able to fly.",
    "Snow is frozen water that falls during the cold winter weather.",
    "Coffee is a popular morning drink for people all around the world.",
    "Chairs are used for sitting and usually have four sturdy legs.",
    "Telephones let people speak to each other across very long distances.",
    "Trees grow new green leaves during the warmer months of spring.",
    "Doors are usually made of wood and swing open on metal hinges.",
    "Paper is usually made from wood pulp pressed into thin flat sheets.",
    "The ocean contains salt water and covers most of the whole planet.",
    "Clouds are made of tiny water droplets floating high in the air.",
    "Mountains are formed slowly over very long periods of geological time.",
    "Bread is made from flour and water mixed together and then baked.",
    "Dogs are common household pets that many families keep at home.",
    "Rain falls from the clouds when the water droplets become too heavy.",
    "Shoes are worn on the feet to protect them while a person walks.",
    "Grass is green and grows in lawns and open fields nearly everywhere.",
    "Fire is hot and produces both light and warmth while it is burning.",
    "Cats sleep for many hours each day in warm and sunny places.",
    "Cars typically have four wheels and an engine under the hood.",
    "Books contain printed pages that are bound together along one edge.",
    "Milk is a white liquid that comes from cows and other mammals.",
    "Bicycles have two wheels and are moved forward by pedalling.",
    "Sand is made of very small grains of worn down rock and shell.",
    "Lamps provide light indoors when the daylight outside has faded.",
    "Rivers carry fresh water downhill until they reach the sea.",
    "Knives are used in kitchens for cutting food into smaller pieces.",
    "Wool comes from sheep and is spun into yarn for making clothing.",
]

JUNK_SYLLABLES = ("zx qq vunt gorble skree blarn fnnn ggrek twaddle quonk fleeb narnt "
                  "wingle bortch flumdiddy narp quixling zorp gnarp oont thistle mmph").split()


def make_junk(rng):
    kind = rng.choice(["garble", "errorlog", "spam", "repeat"])
    if kind == "garble":
        return " ".join(rng.choices(JUNK_SYLLABLES, k=rng.randint(7, 11)))
    if kind == "errorlog":
        return (f"{rng.choice(['ERROR','FATAL','WARN'])} 0x{rng.randrange(16**8):08X} "
                f"{rng.choice(['malformed packet','stack trace','buffer overflow'])} "
                f"at 0x{rng.randrange(16**6):06X} retry {rng.randint(1, 99)}")
    if kind == "spam":
        return (f"{rng.choice(['WIN BIG','CLICK NOW','FREE OFFER'])}!!! $$$ "
                f"{rng.choice(['act fast','hurry','limited'])} "
                f"{rng.choice(['instant cash','prize','bonus'])} wow!!!")
    tok = rng.choice(JUNK_SYLLABLES)
    return " ".join([tok * rng.randint(3, 6)] * rng.randint(5, 8))


def build_stream(seed):
    rng = random.Random(seed)
    stream = []
    early_slots = list(range(5, 45))
    late_slots = list(range(160, 280))
    rng.shuffle(early_slots)
    rng.shuffle(late_slots)
    planted = {}
    for i, (a, b) in enumerate(EARLY_FACTS):
        planted[early_slots[2 * i]] = a
        planted[early_slots[2 * i + 1]] = b
    for i, fact in enumerate(LATE_FACTS):
        planted[late_slots[i]] = fact
    for i in range(STREAM_LEN):
        if i in planted:
            stream.append((planted[i], "fact"))
        elif rng.random() < 0.12:
            stream.append((make_junk(rng), "junk"))
        else:
            stream.append((rng.choice(FILLER_POOL), "filler"))
    return stream


EARLY_PROBES = [
    "Flour milling at the Halverson site beside Petrie Creek ended in the forties.",
    "Nineteen quartets for strings were finished by Ottoline Verrick before thirty.",
    "The ferry to Brantwood does not operate in the winter or early spring.",
    "Hand harvesting of oysters at Kestrel Bay takes place on winter low tides.",
    "After the branch line shut, the signal box at Dunmore was taken out of use.",
    "Eleven thousand moths were catalogued by Marguerite Follansbee over her life.",
    "The Ashgrove works stopped making roof tiles once its clay pit flooded.",
    "Four summers of work went into Tobias Wrenn's survey of the Corrieshalloch caves.",
]

CONTROLS = [
    "Paris is the capital city of France and a major tourist destination.",
    "The Pacific Ocean is the largest ocean on the surface of the Earth.",
    "Einstein developed the theory of relativity in the early twentieth century.",
    "Shakespeare wrote many famous plays including Hamlet and Romeo and Juliet.",
    "The human heart pumps blood through the arteries and the veins.",
    "Photosynthesis lets plants convert sunlight into usable chemical energy.",
]

FLUENCY = [
    "The old wooden gate had been left open again by somebody in a hurry.",
    "She placed the letter on the table and walked slowly toward the window.",
    "It rained for most of the afternoon and then cleared up before evening.",
]


def avg(ai, texts):
    return sum(ai.surprise(t) for t in texts) / len(texts)


def snapshot(ai, base):
    return dict(
        at=ai.stats["seen"],
        updates=ai.stats["updates"],
        rehearsals=ai.stats["rehearsals"],
        vetoed=ai.stats["vetoed"],
        floored=ai.stats["floored"],
        rollbacks=ai.stats["rollbacks"],
        probe=base["probe"] - avg(ai, EARLY_PROBES),
        drift=avg(ai, CONTROLS) - base["control"],
        fluency=avg(ai, FLUENCY) - base["fluency"],
        distance=ai.distance(),
    )


CONFIGS = {
    "stable":  dict(),
    "nofloor": dict(loss_floor=0.0, guard=False, weight_decay=0.0, lr_decay=1.0),
}


def run(seed, mode):
    torch.manual_seed(seed)
    ai = Neuron(**CONFIGS[mode])
    base = dict(probe=avg(ai, EARLY_PROBES),
                control=avg(ai, CONTROLS),
                fluency=avg(ai, FLUENCY))
    rows = []
    for i, (text, kind) in enumerate(build_stream(seed), start=1):
        ai.observe(text)
        if i % CHECK_EVERY == 0:
            row = snapshot(ai, base)
            row["at"] = i
            rows.append(row)
            print(f"    {i:>3}: upd {row['updates']:>3} reh {row['rehearsals']:>3} "
                  f"flr {row['floored']:>3} rb {row['rollbacks']:>2}  "
                  f"probe {row['probe']:+.4f}  drift {row['drift']:+.4f}  "
                  f"fluency {row['fluency']:+.4f}")
    del ai
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return rows


HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>NEURON long run</title>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;background:#12141a;
color:#e6e8ee;margin:0;padding:32px}
h1{font-size:20px;font-weight:600;margin:0 0 4px}
.sub{color:#8b90a0;font-size:13px;margin-bottom:28px}
.chart{background:#1a1d26;border:1px solid #262b38;border-radius:10px;
padding:18px;margin-bottom:18px}
.ct{font-size:14px;font-weight:600;margin:0 0 2px}
.cd{font-size:12px;color:#8b90a0;margin:0 0 12px}
.legend{display:flex;gap:16px;font-size:12px;margin-bottom:10px}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}
svg{width:100%;height:220px;display:block}
</style></head><body>
<h1>NEURON long run</h1>
<div class="sub" id="sub"></div>
<div id="charts"></div>
<script>
const DATA = __DATA__;
const COLORS = {stable:"#4ade80", nofloor:"#f87171"};
const modes = Object.keys(DATA);

function series(mode, key){
  const runs = DATA[mode];
  const n = runs[0].length;
  const out = [];
  for(let i=0;i<n;i++){
    let s=0; for(const r of runs) s += r[i][key];
    out.push({x: runs[0][i].at, y: s/runs.length});
  }
  return out;
}

function chart(title, desc, key, invert){
  const all = modes.map(m=>series(m,key));
  const xs = all[0].map(p=>p.x);
  let lo=Infinity, hi=-Infinity;
  for(const s of all) for(const p of s){ lo=Math.min(lo,p.y); hi=Math.max(hi,p.y); }
  if(lo>0) lo=0; if(hi<0) hi=0;
  const pad=(hi-lo)*0.12||1; lo-=pad; hi+=pad;
  const W=1000,H=220,ML=52,MR=14,MT=12,MB=26;
  const px=v=>ML+(v-xs[0])/(xs[xs.length-1]-xs[0])*(W-ML-MR);
  const py=v=>MT+(hi-v)/(hi-lo)*(H-MT-MB);
  let svg=`<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">`;
  for(let i=0;i<=4;i++){
    const v=lo+(hi-lo)*i/4, y=py(v);
    svg+=`<line x1="${ML}" y1="${y}" x2="${W-MR}" y2="${y}" stroke="#262b38"/>`;
    svg+=`<text x="${ML-8}" y="${y+4}" fill="#6b7183" font-size="11"
      text-anchor="end">${v.toFixed(1)}</text>`;
  }
  if(lo<0&&hi>0){const y=py(0);
    svg+=`<line x1="${ML}" y1="${y}" x2="${W-MR}" y2="${y}" stroke="#4a5163"
      stroke-dasharray="4 4"/>`;}
  all.forEach((s,i)=>{
    const c=COLORS[modes[i]]||"#888";
    const d=s.map((p,j)=>(j?"L":"M")+px(p.x)+" "+py(p.y)).join(" ");
    svg+=`<path d="${d}" fill="none" stroke="${c}" stroke-width="2.2"/>`;
    s.forEach(p=>{svg+=`<circle cx="${px(p.x)}" cy="${py(p.y)}" r="3" fill="${c}"/>`;});
  });
  xs.forEach((x,i)=>{ if(i%2) return;
    svg+=`<text x="${px(x)}" y="${H-8}" fill="#6b7183" font-size="11"
      text-anchor="middle">${x}</text>`;});
  svg+="</svg>";
  const leg=modes.map(m=>`<span><span class="dot" style="background:${COLORS[m]}">
    </span>${m}</span>`).join("");
  return `<div class="chart"><div class="ct">${title}</div>
    <div class="cd">${desc}</div><div class="legend">${leg}</div>${svg}</div>`;
}

document.getElementById("sub").textContent =
  modes.map(m=>m+": "+DATA[m].length+" seed(s)").join("  |  ")
  + "  |  x axis = items seen";

document.getElementById("charts").innerHTML =
  chart("PROBE — do early facts survive?",
        "Early facts in wording never trained on. Should rise and stay up. Falling means forgetting.","probe")
+ chart("DRIFT — is unrelated knowledge safe?",
        "General knowledge the system never touched. Flat is healthy. Climbing means rot.","drift")
+ chart("FLUENCY — is the model still a language model?",
        "Ordinary prose. Climbing is the fatal case: the model itself is degrading.","fluency")
+ chart("UPDATES — how much did it actually learn?",
        "Cumulative weight updates.","updates")
+ chart("DISTANCE — how far have the adapters moved?",
        "Total movement from the starting point.","distance");
</script></body></html>"""


results = {m: [] for m in CONFIGS}
print(f"Long run: {STREAM_LEN} items, checkpoint every {CHECK_EVERY}\n")

for seed in SEEDS:
    for mode in CONFIGS:
        print(f"seed {seed}, {mode}")
        results[mode].append(run(seed, mode))
        print()

with open("results.json", "w") as f:
    json.dump(results, f)

with open("results.html", "w") as f:
    f.write(HTML.replace("__DATA__", json.dumps(results)))

print("=" * 78)
print("Wrote results.json and results.html")
print("Open it with:  open results.html")
print("=" * 78)