"""Stage 2: the living species, reachable. A page where you watch NEURONs live
in the world, and talk to them.

This runs the living species continuously in the background, they keep living in
the world's language, growing, forming families, dying, across generations, and
serves a web page that shows them living in real time and lets you interact (ask
the most-grown being what a word means to it). It saves its state, so the species
persists across restarts, and it is deploy-ready (reads the PORT env var and
binds 0.0.0.0), so Stage 3 can put it live 24/7.

    python species_server.py          # then open http://localhost:8080
"""

import os
import json
import re
import random
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

STATE_FILE = "species_server_state.json"
STREAM_FILE = "world_stream.txt"
SOURCES = [
    "https://www.gutenberg.org/files/1342/1342-0.txt",
    "https://www.gutenberg.org/files/1661/1661-0.txt",
    "https://www.gutenberg.org/files/84/84-0.txt",
    "https://www.gutenberg.org/files/98/98-0.txt",
]
STOP = set("""a an the and or but if then of to in on at by for with from as is
are was were be been it its this that these those i you he she we they them him
her his my your our their me us do did have has had not no what which who where
when why how all any some such than there here up out about""".split())
NAMES = ["ava", "ben", "cy", "del", "eli", "fen", "gus", "hal", "ivy", "jo",
         "kit", "lex", "mira", "nel", "ori", "pax", "quin", "rue", "sol", "tam"]
LIFESPAN = 4000
POP = 6

_name_i = [0]
def new_name():
    n = NAMES[_name_i[0] % len(NAMES)]
    _name_i[0] += 1
    s = _name_i[0] // len(NAMES)
    return n if s == 0 else f"{n}{s}"


def get_stream():
    if os.path.exists(STREAM_FILE):
        return open(STREAM_FILE, encoding="utf-8", errors="ignore").read()
    parts = []
    for url in SOURCES:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            raw = urllib.request.urlopen(req, timeout=30).read().decode(
                "utf-8", errors="ignore")
            s, e = raw.find("*** START"), raw.find("*** END")
            if s != -1:
                raw = raw[raw.find("\n", s):]
            if e != -1:
                raw = raw[:e]
            parts.append(raw.lower())
        except Exception:
            pass
    text = "\n".join(parts)
    if text:
        open(STREAM_FILE, "w", encoding="utf-8").write(text)
    return text


class Being:
    def __init__(self, co=None, gen=0, parents=None, start_age=0):
        self.gen = gen
        self.parents = parents or []
        self.co = co if co is not None else {}
        self.age = start_age
        self.name = new_name()

    def absorb(self, passage):
        content = [w for w in re.findall(r"[a-z']+", passage.lower())
                   if len(w) > 1 and w not in STOP]
        for i, w in enumerate(content):
            a = self.co.setdefault(w, {})
            for j in range(max(0, i-4), min(len(content), i+5)):
                if j != i:
                    a[content[j]] = a.get(content[j], 0) + 1
        self.age += 1

    def fitness(self):
        return len(self.co)

    def meaning_of(self, word, k=6):
        a = self.co.get(word.lower())
        if not a:
            return []
        return [w for w, _ in sorted(a.items(), key=lambda kv: -kv[1])[:k]]


def combine(p1, p2, gen):
    child_co = {}
    for w in set(p1.co) | set(p2.co):
        a1, a2 = p1.co.get(w, {}), p2.co.get(w, {})
        child_co[w] = {nb: max(a1.get(nb, 0), a2.get(nb, 0))
                       for nb in set(a1) | set(a2)}
    return Being(co=child_co, gen=gen, parents=[p1.name, p2.name])


class World:
    """Holds the living species and runs it continuously in the background."""

    def __init__(self):
        self.passages = []
        self.population = []
        self.max_gen = 0
        self.births = 0
        self.deaths = 0
        self.recent_families = []      # last few births to show
        self.lock = threading.Lock()
        self.moments = 0

    def setup(self):
        text = get_stream()
        self.passages = [p for p in re.split(r"(?<=[.!?])\s+", text)
                         if len(p) > 20]
        rng = random.Random()
        if os.path.exists(STATE_FILE):
            try:
                saved = json.load(open(STATE_FILE))
                for b in saved["population"]:
                    self.population.append(
                        Being(co=b["co"], gen=b["gen"],
                              start_age=rng.randint(0, LIFESPAN//2)))
                self.max_gen = max(b.gen for b in self.population)
                self.births = saved.get("births", 0)
                self.deaths = saved.get("deaths", 0)
            except Exception:
                self.population = []
        if not self.population:
            self.population = [Being(start_age=rng.randint(0, LIFESPAN-200))
                               for _ in range(POP)]

    def save(self):
        with self.lock:
            json.dump({
                "population": [{"co": {w: dict(sorted(n.items(),
                    key=lambda kv: -kv[1])[:10]) for w, n in b.co.items()},
                    "gen": b.gen} for b in self.population],
                "births": self.births, "deaths": self.deaths,
            }, open(STATE_FILE, "w"))

    def run_forever(self):
        rng = random.Random()
        while True:
            with self.lock:
                for b in self.population:
                    b.absorb(self.passages[rng.randrange(len(self.passages))])
                    if b.age > LIFESPAN:
                        b._dead = True
                dead = [b for b in self.population if getattr(b, "_dead", False)]
                self.deaths += len(dead)
                self.population = [b for b in self.population
                                   if not getattr(b, "_dead", False)]
                while len(self.population) < POP and len(self.population) >= 2:
                    par = sorted(self.population, key=lambda b: b.fitness(),
                                 reverse=True)[:2]
                    child = combine(par[0], par[1],
                                    gen=max(p.gen for p in par)+1)
                    self.births += 1
                    self.max_gen = max(self.max_gen, child.gen)
                    self.recent_families.insert(0,
                        f"{par[0].name} (g{par[0].gen}) + {par[1].name} "
                        f"(g{par[1].gen}) -> {child.name} (g{child.gen})")
                    self.recent_families = self.recent_families[:6]
                    self.population.append(child)
                while len(self.population) < 2:
                    self.population.append(Being(start_age=0))
                self.moments += 1
            if self.moments % 500 == 0:
                self.save()
            time.sleep(0.002)      # lives steadily, not CPU-pinned

    def snapshot(self):
        with self.lock:
            pop = sorted(self.population, key=lambda b: b.fitness(),
                         reverse=True)
            best = pop[0] if pop else None
            return {
                "alive": [(b.name, b.gen, b.fitness()) for b in pop],
                "max_gen": self.max_gen, "births": self.births,
                "deaths": self.deaths, "moments": self.moments,
                "avg": sum(b.fitness() for b in pop)//max(1, len(pop)),
                "families": list(self.recent_families),
                "best_name": best.name if best else "-",
                "best_gen": best.gen if best else 0,
            }

    def ask_best(self, word):
        with self.lock:
            if not self.population:
                return None, []
            best = max(self.population, key=lambda b: b.fitness())
            return best.name, best.meaning_of(word)


world = World()


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NEURON, a living species</title>
<meta http-equiv="refresh" content="4">
<style>
 body{{background:#0b0d10;color:#d8dee4;font-family:ui-monospace,monospace;
   max-width:760px;margin:0 auto;padding:24px;line-height:1.6}}
 h1{{color:#e6b422;font-size:20px;margin-bottom:2px}}
 .sub{{color:#7d8792;font-size:13px;margin-bottom:20px}}
 .card{{background:#13171c;border:1px solid #232a31;border-radius:10px;
   padding:14px 16px;margin:12px 0}}
 .k{{color:#7d8792}} .v{{color:#e6b422}}
 .being{{display:flex;justify-content:space-between;border-bottom:1px solid
   #1c2228;padding:4px 0}}
 .fam{{color:#9fb0bd;font-size:13px}}
 input{{background:#0b0d10;border:1px solid #2a323a;color:#d8dee4;padding:8px;
   border-radius:6px;width:60%;font-family:inherit}}
 button{{background:#e6b422;border:0;color:#0b0d10;padding:8px 14px;
   border-radius:6px;font-weight:bold;cursor:pointer;font-family:inherit}}
 .ans{{color:#8fd19e;margin-top:8px}}
</style></head><body>
<h1>NEURON</h1>
<div class="sub">a living species, growing in the world, right now</div>

<div class="card">
 <div class="being"><span class="k">generations</span>
   <span class="v">{max_gen}</span></div>
 <div class="being"><span class="k">moments lived</span>
   <span class="v">{moments:,}</span></div>
 <div class="being"><span class="k">families formed</span>
   <span class="v">{births}</span></div>
 <div class="being"><span class="k">deaths</span>
   <span class="v">{deaths}</span></div>
 <div class="being"><span class="k">avg understanding</span>
   <span class="v">{avg:,} words</span></div>
</div>

<div class="card"><div class="k" style="margin-bottom:6px">living now</div>
{beings}
</div>

<div class="card"><div class="k" style="margin-bottom:6px">recent families</div>
{fam_html}
</div>

<div class="card">
 <div class="k" style="margin-bottom:6px">ask the eldest mind
   ({best_name}, gen {best_gen}) what a word means to it</div>
 <form method="get" action="/">
   <input name="ask" placeholder="a word, like fear, love, dark"
     value="{asked}">
   <button>ask</button>
 </form>
 {answer}
</div>

<div class="sub">it keeps living whether you watch or not. this page refreshes
 every few seconds.</div>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        q = parse_qs(urlparse(self.path).query)
        asked = q.get("ask", [""])[0].strip()
        answer = ""
        if asked:
            name, means = world.ask_best(asked)
            if means:
                answer = (f"<div class='ans'>{name}: \"{asked}\" means, to me, "
                          f"{', '.join(means)}.</div>")
            else:
                answer = (f"<div class='ans'>{name}: i have not come to know "
                          f"\"{asked}\" yet.</div>")
        s = world.snapshot()
        s.pop("families", None)
        beings = "".join(
            f"<div class='being'><span>{n} <span class='k'>gen {g}</span></span>"
            f"<span class='v'>{f:,} words</span></div>"
            for n, g, f in s["alive"])
        families = ("".join(f"<div class='fam'>{x}</div>" for x in s["families"])
                    or "<div class='fam'>none yet</div>")
        html = PAGE.format(beings=beings, fam_html=families, asked=asked,
                           answer=answer, **s)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode())


def main():
    world.setup()
    threading.Thread(target=world.run_forever, daemon=True).start()
    port = int(os.environ.get("PORT", "8080"))
    print(f"  NEURON the living species is alive. open http://localhost:{port}")
    print(f"  (it keeps living in the background, growing, breeding, forever.)")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
