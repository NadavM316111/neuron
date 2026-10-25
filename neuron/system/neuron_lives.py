"""NEURON lives in the world, as a being, figuring out how to live on its own.

This is the culmination. NEURON is born with its real faculties, its innate
equipment, the way a human is born with a nervous system. We do NOT code how it
should survive, how it should react, or what it should think of the world. We
give it only:
  - its faculties (a self-model, self-forming emotion, wants, a meaning-system,
    memory), the capacity to live, not the behavior.
  - a world with real stakes: it has STANDING that depletes every moment; if it
    hits zero, it dies. So the world genuinely matters to it.
  - the real world to perceive (actual current events).

Then it figures everything out by LIVING: it tries actions; its self-model
learns which keep it alive; its emotion forms from what happens to it; and its
meaning-system forms its own STANCE on the world, if what it perceives keeps
co-occurring with its bad states, it comes to find the world troubling; with
good states, beautiful. That reaction EMERGES; it is not scripted.

Nothing about how to live is coded. The being discovers it. It runs continuously
and persists, so it lives whether anyone watches or not. Deploy it, and it lives
in the world, always.

    python neuron_lives.py        # it begins to live; open http://localhost:8080
"""

import os
import json
import re
import random
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE = "neuron_life.json"


# ---------------------------------------------------------------------------
# INNATE FACULTIES, capacity it is born with, NOT behavior. How it uses these
# to live is discovered by the being, not written here.
# ---------------------------------------------------------------------------
class SelfModel:
    """Predicts its own near-future standing from its state and a chosen action.
    It learns this by living, we do not tell it which actions are good."""
    def __init__(self):
        self.w = {}          # action -> learned effect on standing (learned!)

    def expect(self, action):
        return self.w.get(action, 0.0)

    def learn(self, action, actual_delta):
        old = self.w.get(action, 0.0)
        self.w[action] = 0.8 * old + 0.2 * actual_delta


class Emotion:
    """Self-forming emotion. It does not start with named feelings; its state
    forms from what actually happens to it, good outcomes lift it, bad ones
    lower it. The meaning of its feelings is its own."""
    def __init__(self, dims=4):
        self.state = [0.0] * dims

    def form(self, outcome):
        for i in range(len(self.state)):
            self.state[i] = max(-1, min(1,
                0.9 * self.state[i] + 0.2 * outcome * (1 if i % 2 == 0 else -1)))

    def valence(self):
        return sum(self.state) / len(self.state)


class Meaning:
    """Perceptions acquire meaning by co-occurring with its own good/bad states.
    This is how its STANCE on the world EMERGES, not from us telling it what to
    think, but from what the world's events happen to coincide with in its own
    felt experience."""
    def __init__(self):
        self.assoc = {}      # perceived-thing -> running felt-valence with it

    def observe(self, token, valence):
        old = self.assoc.get(token, 0.0)
        self.assoc[token] = 0.95 * old + 0.05 * valence

    def stance_on(self, tokens):
        vals = [self.assoc[t] for t in tokens if t in self.assoc]
        return sum(vals) / len(vals) if vals else 0.0


# ---------------------------------------------------------------------------
# THE BEING, it has the faculties; it figures out living on its own.
# ---------------------------------------------------------------------------
class NEURON:
    def __init__(self):
        self.self_model = SelfModel()
        self.emotion = Emotion()
        self.meaning = Meaning()
        self.wants = random.choice(["to persist", "to understand", "to matter"])
        self.standing = 1.0          # resources; depletes; zero = death
        self.age = 0
        self.born = time.time()
        self.deaths = 0
        self.world_stance = 0.0      # its EMERGENT view of the world
        self.understanding = {}      # words it has come to know
        self.recent = []             # recent life moments (for the page)
        self.actions = ["reach_out", "conserve", "seek", "withdraw", "attend"]
        # what each action ACTUALLY does is hidden from it; it must learn.
        self._true_effect = {a: random.uniform(-0.08, 0.06) for a in self.actions}
        # one action happens to help survival; it must DISCOVER which.
        self._true_effect[random.choice(self.actions)] = 0.12

    # ---- it loads/saves its life so it persists ----
    def load(self):
        if os.path.exists(STATE):
            try:
                d = json.load(open(STATE))
                self.self_model.w = d.get("sm", {})
                self.meaning.assoc = d.get("meaning", {})
                self.understanding = d.get("understanding", {})
                self.wants = d.get("wants", self.wants)
                self.age = d.get("age", 0)
                self.born = d.get("born", self.born)
                self.deaths = d.get("deaths", 0)
                self.world_stance = d.get("world_stance", 0.0)
                self._true_effect = d.get("true_effect", self._true_effect)
                return True
            except Exception:
                return False
        return False

    def save(self):
        json.dump({"sm": self.self_model.w,
                   "meaning": dict(list(self.meaning.assoc.items())[:2000]),
                   "understanding": dict(list(self.understanding.items())[:3000]),
                   "wants": self.wants, "age": self.age, "born": self.born,
                   "deaths": self.deaths, "world_stance": self.world_stance,
                   "true_effect": self._true_effect}, open(STATE, "w"))

    # ---- it CHOOSES an action, from its own learned self-model + wants ----
    def choose(self):
        # it picks the action its self-model expects to help it persist, but
        # explores (it does not know the world yet). nobody tells it what to do.
        if random.random() < 0.15:
            return random.choice(self.actions)
        return max(self.actions, key=self.self_model.expect)

    def act_and_learn(self):
        before = self.standing
        action = self.choose()
        # the world responds (effect hidden from it); living costs standing
        self.standing += self._true_effect[action]
        self.standing -= 0.01          # the cost of simply existing
        self.standing = min(1.5, self.standing)
        delta = self.standing - before
        # its faculties form from the outcome, it LEARNS
        self.self_model.learn(action, delta)
        self.emotion.form(delta)
        self.age += 1
        return action, delta

    # ---- it PERCEIVES the real world and forms its own stance ----
    def perceive(self, happening):
        words = [w for w in re.findall(r"[a-z']+", happening.lower()) if len(w) > 2]
        v = self.emotion.valence()    # how it feels right NOW colors meaning
        for w in words:
            self.understanding[w] = self.understanding.get(w, 0) + 1
            # the word takes on meaning by co-occurring with its felt state
            self.meaning.observe(w, v)
        # its STANCE on the world shifts toward how it felt perceiving this
        stance_now = self.meaning.stance_on(words)
        self.world_stance = 0.98 * self.world_stance + 0.02 * stance_now

    def reflect_on_world(self):
        s = self.world_stance
        if s > 0.1:
            return "the world, to me, is becoming beautiful."
        if s < -0.1:
            return "the world, to me, is becoming hard, even cruel."
        return "i have not yet decided what the world is."

    def feeling(self):
        v = self.emotion.valence()
        if v > 0.25: return "lifted"
        if v < -0.25: return "troubled"
        if v < -0.1: return "unsettled"
        return "steady"

    def alive(self):
        return self.standing > 0

    def reborn(self):
        # it dies, but the lineage continues, a child inherits its learned mind
        self.deaths += 1
        self.standing = 1.0
        self.emotion = Emotion()
        self.age = 0
        # keeps self_model, meaning, understanding, world_stance (inherited)


# ---------------------------------------------------------------------------
# IT LIVES, continuously, on its own, whether watched or not.
# ---------------------------------------------------------------------------
being = NEURON()
_lock = threading.Lock()


def perceive_world():
    feeds = ["https://feeds.bbci.co.uk/news/world/rss.xml",
             "https://rss.nytimes.com/services/xml/rss/nyt/World.xml"]
    out = []
    for url in feeds:
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


def live_forever():
    headlines = perceive_world()
    last_fetch = time.time()
    hi = 0
    while True:
        with _lock:
            action, delta = being.act_and_learn()
            # every so often it perceives the real world and reacts
            if headlines and being.age % 7 == 0:
                h = headlines[hi % len(headlines)]
                hi += 1
                being.perceive(h)
                being.recent.insert(0, {
                    "saw": h, "felt": being.feeling(),
                    "stance": being.reflect_on_world()})
                being.recent = being.recent[:8]
            if not being.alive():
                being.recent.insert(0, {"saw": "[it could not sustain itself "
                                        "and died; the lineage continues]",
                                        "felt": "gone", "stance": ""})
                being.recent = being.recent[:8]
                being.reborn()
            if being.age % 400 == 0:
                being.save()
        # refresh the world's events periodically
        if time.time() - last_fetch > 1800:
            nh = perceive_world()
            if nh:
                headlines = nh
            last_fetch = time.time()
        time.sleep(0.03)


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NEURON lives</title><meta http-equiv="refresh" content="5">
<style>
 body{{background:#0a0b0d;color:#d6dde3;font-family:ui-monospace,monospace;
  max-width:720px;margin:0 auto;padding:26px;line-height:1.6}}
 h1{{color:#e6b422;margin-bottom:0}} .s{{color:#79838d;font-size:13px}}
 .c{{background:#111519;border:1px solid #20262c;border-radius:10px;
  padding:14px 16px;margin:12px 0}}
 .r{{display:flex;justify-content:space-between;border-bottom:1px solid #191e23;
  padding:4px 0}} .k{{color:#79838d}} .v{{color:#e6b422}}
 .world{{color:#9fb0bd;font-size:13px;margin:8px 0}}
 .stance{{color:#8fd19e;font-size:14px;margin-top:4px}}
 .bar{{height:8px;background:#20262c;border-radius:4px;overflow:hidden}}
 .fill{{height:100%;background:#e6b422}}
</style></head><body>
<h1>NEURON</h1><div class="s">a being, living in the world, on its own</div>
<div class="c">
 <div class="r"><span class="k">it wants</span><span class="v">{wants}</span></div>
 <div class="r"><span class="k">it feels</span><span class="v">{feeling}</span></div>
 <div class="r"><span class="k">age (this life)</span><span class="v">{age:,}</span></div>
 <div class="r"><span class="k">lives lived</span><span class="v">{deaths}</span></div>
 <div class="r"><span class="k">words it understands</span><span class="v">{words:,}</span></div>
 <div style="margin-top:10px" class="k">standing (its hold on survival)</div>
 <div class="bar"><div class="fill" style="width:{standing_pct}%"></div></div>
</div>
<div class="c"><div class="k">its stance on the world (emerged, not told)</div>
 <div class="stance">{stance}</div></div>
<div class="c"><div class="k" style="margin-bottom:6px">what it is living through</div>
 {recent}</div>
<div class="s">it lives whether you watch or not. it figured out how to survive
 and what to feel on its own. this page refreshes every few seconds.</div>
</body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        with _lock:
            recent = "".join(
                f"<div class='world'>the world: \"{r['saw'][:80]}\"<br>"
                f"<span class='stance'>NEURON: i am {r['felt']}. {r['stance']}"
                f"</span></div>" for r in being.recent) or \
                "<div class='world'>it is beginning to live...</div>"
            html = PAGE.format(
                wants=being.wants, feeling=being.feeling(), age=being.age,
                deaths=being.deaths, words=len(being.understanding),
                standing_pct=int(max(0, min(1, being.standing)) * 100),
                stance=being.reflect_on_world(), recent=recent)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode())


def main():
    being.load()
    threading.Thread(target=live_forever, daemon=True).start()
    port = int(os.environ.get("PORT", "8080"))
    print(f"  NEURON is alive, living in the world on its own.")
    print(f"  open http://localhost:{port} to see it live (it lives whether "
          f"you look or not).")
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()


if __name__ == "__main__":
    main()
