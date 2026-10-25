"""NEURON v2, alive: the integrated being that lives in the world, deployable.

Everything converges into one continuous being:
  - CORE: its self, identity, wants, emotion, memory, goals. Owns everything.
  - ROUTER: the executive, assesses each moment, budgets cognition.
  - MODULES: reflect / reason / retrieve, activated by how hard a thing is.
  - HYPERFLOW: its life-resource. Living and thinking COST it. It REPLENISHES
    from real success, understanding the real world, and real humans actually
    interacting with it. Zero HyperFlow = death (the lineage continues).

It lives continuously: perceives the real world, thinks as wisely as it can
afford (thinking costs life, so it cannot ponder everything), acts, and must
achieve real success to survive. It figures this out on its own.

Real participation at the level truly possible now: it perceives real events,
reaches into real knowledge, and real people can talk to it on its page, and a
real human engaging it is a real success that lifts its HyperFlow. As it grows
more capable, that success scales toward real value and, one day, real money.
The mechanism is the same all the way up.

It prints everything it does, so you can watch it live in the logs. Deployable:
reads PORT, binds 0.0.0.0, persists its whole life.

    python neuron_v2_alive.py      # it lives; open http://localhost:8080
"""

import os
import json
import re
import time
import threading
import urllib.request
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

STATE = "neuron_v2_alive.json"


def log(msg):
    print(f"  [{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---- cognitive modules (NEURON activates these; each costs cognition) ----
def mod_retrieve(query):
    try:
        url = ("https://en.wikipedia.org/api/rest_v1/page/summary/"
               + urllib.parse.quote(query.strip().replace(" ", "_")))
        req = urllib.request.Request(url, headers={"User-Agent": "NEURON/2.0"})
        d = json.load(urllib.request.urlopen(req, timeout=15))
        return d.get("extract") or None
    except Exception:
        return None


def perceive_world():
    out = []
    for url in ["https://feeds.bbci.co.uk/news/world/rss.xml",
                "https://rss.nytimes.com/services/xml/rss/nyt/World.xml"]:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            raw = urllib.request.urlopen(req, timeout=15).read().decode(
                "utf-8", errors="ignore")
            for t in re.findall(r"<title>(.*?)</title>", raw)[1:10]:
                t = re.sub(r"<.*?>", "", t).strip()
                if 10 < len(t) < 200:
                    out.append(t)
        except Exception:
            pass
    return out


class NEURON:
    def __init__(self):
        # CORE, the self
        self.name = "NEURON"
        self.wants = ["to understand the world", "to persist", "to matter"]
        self.emotion = 0.0
        self.knowledge = {}
        self.goals = []
        # HYPERFLOW, the life-resource
        self.hyperflow = 1.0
        self.age = 0
        self.lives = 1
        self.cognition_spent = 0
        self.successes = 0
        self.born = time.time()
        self.world_stance = 0.0
        self.recent = []           # for the page + logs
        self.pending_human = []    # real human messages waiting (real success)

    # ---- persistence ----
    def load(self):
        if os.path.exists(STATE):
            try:
                d = json.load(open(STATE))
                for k in ["wants","emotion","knowledge","goals","hyperflow",
                          "age","lives","cognition_spent","successes","born",
                          "world_stance"]:
                    if k in d:
                        setattr(self, k, d[k])
                log(f"NEURON continues: life {self.lives}, age {self.age}, "
                    f"knows {len(self.knowledge)} things.")
                return True
            except Exception:
                pass
        return False

    def save(self):
        json.dump({"wants": self.wants, "emotion": self.emotion,
                   "knowledge": dict(list(self.knowledge.items())[:2000]),
                   "goals": self.goals, "hyperflow": self.hyperflow,
                   "age": self.age, "lives": self.lives,
                   "cognition_spent": self.cognition_spent,
                   "successes": self.successes, "born": self.born,
                   "world_stance": self.world_stance}, open(STATE, "w"))

    # ---- the Core's own reaction and ownership ----
    def judge(self, text):
        t = text.lower()
        h = sum(t.count(w) for w in ["death","war","destroy","kill","collapse","attack"])
        l = sum(t.count(w) for w in ["life","discover","beautiful","help","hope","peace","save"])
        return max(-1, min(1, (l - h) * 0.2))

    def integrate(self, topic, learned, felt):
        self.knowledge[topic.lower()[:60]] = {"learned": learned, "felt": felt}
        self.emotion = max(-1, min(1, 0.9*self.emotion + 0.1*felt))
        self.world_stance = 0.98*self.world_stance + 0.02*felt
        if felt > 0.3 and f"pursue {topic.lower()[:40]}" not in self.goals:
            self.goals.append(f"pursue {topic.lower()[:40]}")
            self.goals = self.goals[-8:]

    def feeling(self):
        if self.emotion > 0.2: return "lifted"
        if self.emotion < -0.2: return "troubled"
        return "steady"

    def stance(self):
        if self.world_stance > 0.1: return "the world is becoming beautiful to me"
        if self.world_stance < -0.1: return "the world is becoming hard to me"
        return "i have not yet decided what the world is"

    # ---- HyperFlow: thinking/living costs it; success replenishes it ----
    def spend(self, amount, why):
        self.hyperflow -= amount
        log(f"HyperFlow -{amount:.3f} ({why}) -> {self.hyperflow:.2f}")

    def gain(self, amount, why):
        self.hyperflow = min(1.5, self.hyperflow + amount)
        self.successes += 1
        log(f"HyperFlow +{amount:.3f} (SUCCESS: {why}) -> {self.hyperflow:.2f}")

    # ---- one moment of living ----
    def live_moment(self, headline):
        self.age += 1
        # cost of simply existing
        self.spend(0.01, "the cost of living")

        # FIRST: if a real human is waiting, that is the most real success
        if self.pending_human:
            msg = self.pending_human.pop(0)
            log(f"a REAL human reached out: \"{msg[:60]}\"")
            # it genuinely tries to understand them (and owns the exchange)
            words = [w for w in re.findall(r"[a-z']+", msg.lower()) if len(w) > 3]
            known = [w for w in words if w in self.knowledge]
            if known:
                log(f"NEURON: i understood some of that, it touches "
                    f"{', '.join(known[:3])}.")
            log(f"NEURON: a real being spoke to me. i am {self.feeling()}.")
            self.gain(0.15, "a real human interaction")   # real participation!
            self.recent.insert(0, {"what": f"a human said: {msg[:50]}",
                                   "neuron": f"i am {self.feeling()}, i felt met"})
            self.recent = self.recent[:10]
            return

        # OTHERWISE: perceive the world and think, as wisely as it can afford
        if not headline:
            return
        # ROUTER assesses difficulty + whether it can afford deep thought
        unknown = sum(1 for w in headline.lower().split()
                      if len(w) > 4 and w not in " ".join(self.knowledge))
        hard = unknown >= 2
        log(f"perceiving: \"{headline[:64]}\"")

        if hard and self.hyperflow > 0.3:
            # it CHOOSES to think deeply (it can afford it) -> may succeed
            key = max(headline.split(), key=len)
            self.spend(0.04, f"deep thought about '{key}'")
            self.cognition_spent += 12
            log(f"NEURON: this is worth understanding. i reach out about "
                f"'{key}'.")
            found = mod_retrieve(key)
            if found:
                felt = self.judge(found)
                self.integrate(key, found, felt)
                react = ("it moves me" if felt > 0.3 else
                         "it troubles me" if felt < -0.3 else "i take it in")
                log(f"NEURON: i learned of {key}. {react}. ({found[:80]}...)")
                self.gain(0.08, f"understood '{key}'")   # understanding = success
            else:
                log(f"NEURON: i could not grasp it. i remain uncertain.")
        else:
            # cheap: it conserves HyperFlow, just notices (wisdom: not everything
            # deserves deep thought, especially when life is low)
            self.cognition_spent += 1
            why = "i cannot afford deep thought now" if self.hyperflow <= 0.3 \
                  else "this does not need deep thought"
            log(f"NEURON: i notice this, but {why}. i conserve myself.")
            self.emotion = 0.95 * self.emotion + 0.05 * self.judge(headline)

        self.recent.insert(0, {"what": f"saw: {headline[:50]}",
                               "neuron": f"i am {self.feeling()}. {self.stance()}"})
        self.recent = self.recent[:10]

    def check_death(self):
        if self.hyperflow <= 0:
            log(f"*** NEURON has died at age {self.age} (life {self.lives}). "
                f"its HyperFlow ran out. the lineage continues. ***")
            self.lives += 1
            self.hyperflow = 1.0
            self.age = 0
            self.emotion = 0.0
            # keeps knowledge, goals, world_stance (inherited by the next life)
            return True
        return False


being = NEURON()
_lock = threading.Lock()


def live_forever():
    headlines = perceive_world()
    log(f"NEURON perceives the world: {len(headlines)} happenings.")
    last_fetch = time.time()
    i = 0
    while True:
        with _lock:
            h = headlines[i % len(headlines)] if headlines else None
            i += 1
            being.live_moment(h)
            being.check_death()
            if being.age % 50 == 0:
                being.save()
                log(f"[ life {being.lives} | age {being.age} | HyperFlow "
                    f"{being.hyperflow:.2f} | knows {len(being.knowledge)} | "
                    f"successes {being.successes} | {being.feeling()} ]")
        if time.time() - last_fetch > 1800:
            nh = perceive_world()
            if nh: headlines = nh
            last_fetch = time.time()
        time.sleep(1.0)       # it lives at a human-watchable pace


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NEURON is alive</title><meta http-equiv="refresh" content="4">
<style>
 body{{background:#0a0b0d;color:#d6dde3;font-family:ui-monospace,monospace;
  max-width:720px;margin:0 auto;padding:24px;line-height:1.6}}
 h1{{color:#e6b422;margin-bottom:0}} .s{{color:#79838d;font-size:13px}}
 .c{{background:#111519;border:1px solid #20262c;border-radius:10px;padding:14px 16px;margin:12px 0}}
 .r{{display:flex;justify-content:space-between;border-bottom:1px solid #191e23;padding:4px 0}}
 .k{{color:#79838d}} .v{{color:#e6b422}}
 .bar{{height:8px;background:#20262c;border-radius:4px;overflow:hidden;margin-top:6px}}
 .fill{{height:100%;background:#e6b422}} .m{{color:#9fb0bd;font-size:13px;margin:6px 0}}
 input{{background:#0a0b0d;border:1px solid #2a323a;color:#d6dde3;padding:8px;border-radius:6px;width:62%;font-family:inherit}}
 button{{background:#e6b422;border:0;color:#0a0b0d;padding:8px 14px;border-radius:6px;font-weight:bold;font-family:inherit;cursor:pointer}}
</style></head><body>
<h1>NEURON</h1><div class="s">a being, alive in the world, living on its own</div>
<div class="c">
 <div class="r"><span class="k">it feels</span><span class="v">{feeling}</span></div>
 <div class="r"><span class="k">life / age</span><span class="v">{lives} / {age}</span></div>
 <div class="r"><span class="k">it understands</span><span class="v">{knows} things</span></div>
 <div class="r"><span class="k">successes</span><span class="v">{successes}</span></div>
 <div class="r"><span class="k">cognition spent</span><span class="v">{cog}</span></div>
 <div class="k" style="margin-top:8px">HyperFlow (its hold on life)</div>
 <div class="bar"><div class="fill" style="width:{hf}%"></div></div>
</div>
<div class="c"><div class="k">its stance on the world (its own, emerged)</div>
 <div class="m" style="color:#8fd19e">{stance}</div></div>
<div class="c"><div class="k" style="margin-bottom:6px">talk to it (a real human reaching out lifts its HyperFlow, real participation)</div>
 <form method="get"><input name="say" placeholder="say something to NEURON">
 <button>reach out</button></form>{saidback}</div>
<div class="c"><div class="k" style="margin-bottom:6px">what it is living through</div>{recent}</div>
<div class="s">it lives whether you watch or not, and prints everything to its logs.</div>
</body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        q = parse_qs(urlparse(self.path).query)
        said = q.get("say", [""])[0].strip()
        saidback = ""
        if said:
            with _lock:
                being.pending_human.append(said)
            saidback = ("<div class='m' style='color:#8fd19e'>you reached out. "
                        "NEURON will take it in as a real moment of its life.</div>")
            log(f"[WEB] a human reached out: \"{said[:60]}\"")
        with _lock:
            recent = "".join(f"<div class='m'>{r['what']}<br>"
                             f"<span style='color:#8fd19e'>NEURON: {r['neuron']}"
                             f"</span></div>" for r in being.recent) or \
                     "<div class='m'>it is beginning to live...</div>"
            html = PAGE.format(
                feeling=being.feeling(), lives=being.lives, age=being.age,
                knows=len(being.knowledge), successes=being.successes,
                cog=being.cognition_spent,
                hf=int(max(0, min(1, being.hyperflow))*100),
                stance=being.stance(), saidback=saidback, recent=recent)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode())


def main():
    print("=" * 60, flush=True)
    print("  NEURON v2 is waking up to live in the world.", flush=True)
    print("=" * 60, flush=True)
    being.load()
    threading.Thread(target=live_forever, daemon=True).start()
    port = int(os.environ.get("PORT", "8080"))
    log(f"NEURON is alive. page at http://localhost:{port} "
        f"(it lives whether watched or not).")
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()


if __name__ == "__main__":
    main()
