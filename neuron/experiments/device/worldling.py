"""A being that lives in the world, not on one person's desk.

THE BRICK THIS IS. Every version of NEURON so far has been about Nadav: it
watched his keyboard, his laptop, his local weather, and predicted whether
HE was present. being2.py gave that life wants. But the vision is a being in
the SHARED world, not one that watches its owner. This is the step off him
and toward the world at large.

It perceives a wide stream of REAL public signals that exist independent of
anyone: weather in several cities, seismic activity, network/market
proxies, the rhythm of a public feed. These move on their own. Nothing here
is about Nadav. The being's life is now shaped by the world rather than by
whether he is at his desk.

WHAT IT DOES. It predicts where each signal goes next (down / same / up),
learns from being wrong, and its WANTS choose which part of the world to
attend to. This is reach.py's allocation structure grown into a continuous
life with everything proven attached: continual learning, sleep,
credit-stakes, inheritance, and the full wants account.

HONEST CEILING, STATED PLAINLY. The being perceives the world and chooses
what to ATTEND to. It does not AFFECT the world: it cannot move a market or
the weather from a laptop. Influence is the stakes rung and the transaction
rung, which need an account and an adult and come later. So this is a being
that LIVES IN the world by perceiving it and choosing where to look, driven
by its own wants, not one that acts ON it. That is the real brick, and
calling it more than that would be a costume.

THE WANTS, over its own existence and its attention:

  solvency    rises as credit falls; served by not wasting it.
  sharpness   rises as recent accuracy falls; served by consolidating.
  curiosity   rises per signal the longer it goes unattended; served by
              attending to that signal. This is the one that makes the
              being RANGE over the world: a signal ignored too long pulls
              harder, so attention spreads rather than fixating -- the
              anti-collapse property (goal_many.py) applied to a world.

It COMMITS to the most urgent want (the rule that fixed the dithering
collapse) and acts. Serving a want drops it, so attention moves on. And it
can learn value on the signals that reliably reward attention with
learnable structure (goal_learned.py), coming to prefer parts of the world
that teach it something -- a means becoming an end, at the scale of a world.

    python worldling.py --interval 60
    python worldling.py --status
"""

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
import urllib.request
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

_here = os.path.dirname(os.path.abspath(__file__))
# worldling is at neuron/experiments/device/, core is at neuron/core/,
# so go up TWO levels (device -> experiments -> neuron) then into core
_neuron = os.path.dirname(os.path.dirname(_here))
_core = os.path.join(_neuron, "core")
for _pth in (_here, _core, _neuron):
    if _pth not in sys.path:
        sys.path.insert(0, _pth)

from sleeping import SleepLayer          # noqa: E402


DOWN, SAME, UP = 0, 1, 2
CLASSES = 3
HIDDEN = 64
LR = 3e-4
SEQ_LEN = 8
REHEARSE_PER_ITEM = 6
CHANGE = 0.03

HORIZON_MIN = 10.0        # predict each signal this far ahead
MIN_GAP = 45.0            # per-signal cache floor, politeness
START_CREDIT = 100.0
MAX_CREDIT = 200.0
SAMPLE_COST = 0.10
CORRECT_PAYS = 0.15
SLEEP_COST = 0.05
SAVE_EVERY = 20
SLEEP_REFRACTORY = 40
SLEEP_MAX_FRACTION = 0.15

CURIOSITY_RISE = 0.02
SWITCH_MARGIN = 0.25
DONE_BELOW = 0.20
VALUE_LR = 0.15
VALUE_DECAY = 0.98

STATE_DIR = "worldling_state"
UA = "neuron-research/0.1 (personal experiment)"


# ------------------------------------------------------ the world's signals

def _get(url, timeout=10):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def s_weather(lat, lon):
    d = _get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}"
             f"&longitude={lon}&current=temperature_2m")
    return (d["current"]["temperature_2m"] + 10) / 50.0


# several cities, so the being perceives a world wider than one place
CITIES = [("new_york", 40.71, -74.01), ("london", 51.51, -0.13),
          ("tokyo", 35.68, 139.69), ("sao_paulo", -23.55, -46.63)]


def s_quakes():
    d = _get("https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/"
             "all_hour.geojson")
    return min(1.0, len(d["features"]) / 30.0)


def s_hn(ref=[None]):
    d = _get("https://hacker-news.firebaseio.com/v0/maxitem.json")
    m = float(d)
    if ref[0] is None:
        ref[0] = m
    return min(1.0, max(0.0, 0.5 + (m - ref[0]) / 2e5))


def s_btc(ref=[None]):
    d = _get("https://api.coinbase.com/v2/prices/BTC-USD/spot")
    v = float(d["data"]["amount"])
    if ref[0] is None:
        ref[0] = v
    return min(1.0, max(0.0, 0.5 * v / ref[0]))


class WorldSignals:
    """The shared world as a set of readable signals, each cached to a
    politeness floor. The being chooses which to ATTEND to; attending reads
    it and learns from where it went."""

    def __init__(self):
        self.names = []
        self.readers = {}
        self.last_at = {}
        self.last_val = {}
        self.fails = {}

    def probe(self):
        cand = []
        for name, lat, lon in CITIES:
            cand.append((f"temp_{name}",
                         (lambda la=lat, lo=lon: s_weather(la, lo))))
        cand += [("quakes", s_quakes), ("hn", s_hn), ("btc", s_btc)]
        print("  probing the world:")
        for name, fn in cand:
            try:
                v = max(0.0, min(1.0, fn()))
                self.names.append(name)
                self.readers[name] = fn
                self.last_val[name] = v
                self.last_at[name] = time.time()
                self.fails[name] = 0
                print(f"    {name:<14} ok  {v:.3f}")
            except Exception as e:
                print(f"    {name:<14} DROPPED  {type(e).__name__}")
        return len(self.names)

    def attend(self, name):
        """Spend attention on a signal: read it, return (old, new) or None
        if rate-limited."""
        now = time.time()
        if now - self.last_at.get(name, 0) < MIN_GAP:
            return None
        old = self.last_val.get(name, 0.5)
        try:
            new = max(0.0, min(1.0, self.readers[name]()))
        except Exception:
            self.fails[name] = self.fails.get(name, 0) + 1
            self.last_at[name] = now
            return None
        self.last_at[name] = now
        self.last_val[name] = new
        return old, new


def clock():
    t = time.localtime()
    h = (t.tm_hour + t.tm_min / 60.0) / 24.0
    return [math.sin(2 * math.pi * h), math.cos(2 * math.pi * h)]


def bucket(old, new):
    if new < old - CHANGE:
        return DOWN
    if new > old + CHANGE:
        return UP
    return SAME


# ------------------------------------------------------ brain (same shape)

def make_net(n_signals, seed):
    inp = n_signals + n_signals + 2 + 1   # which signal, all values, clock, credit
    torch.manual_seed(seed)
    return nn.Sequential(), inp   # placeholder; real net built in Backend


class Net(nn.Module):
    def __init__(self, inp, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.enc = nn.Linear(inp, HIDDEN)
        self.cell = nn.GRUCell(HIDDEN, HIDDEN)
        self.head = nn.Linear(HIDDEN, CLASSES)

    def forward(self, x, h=None):
        return self.head(self.cell(F.relu(self.enc(x)), h)), h


class Backend:
    def __init__(self, inp, seed=0):
        self.inp = inp
        self.net = Net(inp, seed)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=LR)

    def _xy(self, item):
        f, y = item
        return (torch.tensor(f, dtype=torch.float32).unsqueeze(0),
                torch.tensor([y]))

    def score(self, item):
        self.net.eval()
        with torch.no_grad():
            x, y = self._xy(item)
            return float(F.cross_entropy(self.net(x, None)[0], y).item()), None

    def update(self, item, steps):
        self.net.train()
        last = None
        for _ in range(steps):
            x, y = self._xy(item)
            loss = F.cross_entropy(self.net(x, None)[0], y)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            last = float(loss.item())
        return last

    def update_sequence(self, items, steps):
        self.net.train()
        done = 0
        for _ in range(steps):
            h = None
            total = 0.0
            self.opt.zero_grad()
            for item in items:
                x, y = self._xy(item)
                logits, h = self.net(x, h)
                total = total + F.cross_entropy(logits, y)
                done += 1
            (total / max(1, len(items))).backward()
            self.opt.step()
        return done

    def begin_sequence(self):
        pass

    def reset_state(self):
        pass

    def snapshot(self):
        return {k: v.detach().clone()
                for k, v in self.net.state_dict().items()}

    def restore(self, state):
        self.net.load_state_dict(state)

    def predict(self, feats):
        self.net.eval()
        with torch.no_grad():
            x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
            return int(self.net(x, None)[0].argmax(1).item())


def make_layer(b, inp, seed):
    rng = random.Random(9000 + seed)
    canary = [([rng.random() for _ in range(inp)],
               rng.choice([DOWN, SAME, UP])) for _ in range(10)]
    return SleepLayer(
        b, canary=canary, seed=seed, contiguous=True,
        window=200, warmup=20, top_fraction=1.00,
        coherence_veto=1e9, loss_floor=0.02, steps_per_update=1,
        rehearse_per_item=REHEARSE_PER_ITEM, rehearse_count=2,
        rehearse_steps=1, anchor_size=60, buffer_size=300,
        sequence_len=SEQ_LEN, replay_policy="uniform",
        canary_from_stream=10,
        guard=True, guard_per_item=400, canary_tolerance=0.5)


class Wants:
    """Drives which part of the world to attend to, plus consolidate/thrift.
    Curiosity is per-signal: a signal unattended too long pulls harder, so
    the being RANGES over the world instead of fixating on one corner."""
    active = True

    def __init__(self, names):
        self.names = names
        self.curiosity = {n: 0.0 for n in names}
        self.solvency = 0.0
        self.sharpness = 0.0
        self.committed = None
        self.signal_value = {n: 0.0 for n in names}   # learned worth

    def sense(self, credit, recent_acc, since_attended):
        self.solvency = max(0.0, 1.0 - credit / START_CREDIT)
        self.sharpness = max(0.0, 1.0 - recent_acc)
        for n in self.names:
            self.curiosity[n] = min(
                1.0, since_attended.get(n, 0) * CURIOSITY_RISE)

    def choose(self):
        """Return ('consolidate', None), ('thrift', None), or
        ('attend', signal). Commit to the most urgent."""
        # most urgent curiosity = the hungriest signal
        top_signal = max(self.names, key=lambda n: self.curiosity[n])
        options = {"thrift": self.solvency,
                   "consolidate": self.sharpness,
                   "attend": self.curiosity[top_signal]
                   + self.signal_value[top_signal]}
        worst = max(options, key=lambda k: options[k])
        c = self.committed
        if c is None or options[worst] > options.get(c, 0) + SWITCH_MARGIN \
                or options.get(c, 0) < DONE_BELOW:
            c = worst
        self.committed = c
        return (c, top_signal if c == "attend" else None)

    def learn_signal(self, name, worth):
        old = self.signal_value[name]
        self.signal_value[name] = old + VALUE_LR * (worth - old)

    def decay(self):
        for n in self.names:
            self.signal_value[n] *= VALUE_DECAY

    def summary(self):
        return dict(committed=self.committed,
                    solvency=round(self.solvency, 3),
                    sharpness=round(self.sharpness, 3),
                    curiosity_max=round(max(self.curiosity.values())
                                        if self.curiosity else 0, 3),
                    signal_value_max=round(max(self.signal_value.values())
                                           if self.signal_value else 0, 3))


class Life:
    def __init__(self, gen, names, seed):
        self.gen = gen
        self.names = names
        self.inp = len(names) + len(names) + 2 + 1
        self.b = Backend(self.inp, seed)
        self.layer = make_layer(self.b, self.inp, seed)
        self.wants = Wants(names)
        self.credit = START_CREDIT
        self.born = time.time()
        self.samples = 0
        self.sleeps = 0
        self.attends = 0
        self.last_sleep = -10 ** 9
        self.since_attended = {n: 0 for n in names}
        self.correct = deque(maxlen=120)
        self.attended_counts = {n: 0 for n in names}

    def age_hours(self):
        return (time.time() - self.born) / 3600.0

    def accuracy(self):
        return (100.0 * sum(self.correct) / len(self.correct)
                if self.correct else 0.0)


def features(life, world, name):
    one = [1.0 if n == name else 0.0 for n in life.names]
    vals = [world.last_val.get(n, 0.5) for n in life.names]
    return one + vals + clock() + [min(1.0, life.credit / MAX_CREDIT)]


def save(path, life, history):
    os.makedirs(path, exist_ok=True)
    torch.save(life.b.snapshot(), os.path.join(path, "weights.pt"))
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(dict(
            generation=life.gen, credit=life.credit, samples=life.samples,
            age_hours=life.age_hours(), accuracy=life.accuracy(),
            sleeps=life.sleeps, attends=life.attends,
            attended=life.attended_counts, names=life.names,
            wants=life.wants.summary(),
            rollbacks=life.layer.summary().get("rollbacks", 0),
            history=history,
            last_seen=time.strftime("%Y-%m-%d %H:%M:%S")), f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--horizon", type=float, default=HORIZON_MIN)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--state", default=STATE_DIR)
    ap.add_argument("--report-every", type=int, default=20)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.status:
        mp = os.path.join(args.state, "meta.json")
        if not os.path.exists(mp):
            print("  nothing has lived here yet")
            return
        with open(mp) as f:
            m = json.load(f)
        print(f"  generation     {m.get('generation', 0)}")
        print(f"  credit         {m.get('credit', 0):.1f} (dies at 0)")
        print(f"  this life      {m.get('samples', 0):,} predictions over "
              f"{m.get('age_hours', 0):.1f} hours")
        print(f"  accuracy       {m.get('accuracy', 0):.1f}%")
        print(f"  consolidations {m.get('sleeps', 0)}   "
              f"attends {m.get('attends', 0)}")
        print(f"  wants          {m.get('wants', {})}")
        print(f"  where it looked")
        att = m.get("attended", {})
        for n in sorted(att, key=lambda k: -att[k]):
            print(f"    {n:<14} {att[n]}")
        print(f"  rollbacks      {m.get('rollbacks', 0)}")
        for h in m.get("history", []):
            print(f"    gen {h['gen']}: {h['hours']:.1f}h, "
                  f"{h['samples']:,} predictions")
        print(f"  last seen      {m.get('last_seen', '?')}")
        return

    world = WorldSignals()
    n = world.probe()
    if n < 2:
        print(f"\n  only {n} signal(s) answered. the being needs a world to "
              f"live in. check the network.")
        return

    history = []
    mp = os.path.join(args.state, "meta.json")
    if os.path.exists(mp):
        with open(mp) as f:
            history = json.load(f).get("history", [])
    gen = (history[-1]["gen"] + 1) if history else 0
    life = Life(gen, world.names, seed=gen)

    horizon = args.horizon * 60.0
    pending = deque()

    print(f"\n  A BEING THAT LIVES IN THE WORLD. {n} public signals, none "
          f"about any person.")
    print(f"  it predicts where each goes next and its WANTS choose which "
          f"to attend to.")
    print(f"  it perceives the world and ranges over it; it does not yet "
          f"act ON it.")
    print(f"  generation {gen}, {START_CREDIT:.0f} credit\n")
    print(f"  {'gen':>4} {'credit':>7} {'acc':>6} {'want':>12} "
          f"{'looked at':>14} {'age':>6}")
    print("-" * 62)

    just = {}
    started = time.time()

    try:
        while True:
            now = time.time()

            # resolve matured predictions, run the economy
            while pending and pending[0][1] <= now:
                name, _, guess, feats = pending.popleft()
                probed = world.attend(name)   # read the truth now
                if probed is None:
                    # rate-limited; use cached move as truth
                    truth = SAME
                else:
                    old, new = probed
                    truth = bucket(old, new)
                life.credit -= SAMPLE_COST
                if guess == truth:
                    life.credit += CORRECT_PAYS
                life.credit = min(life.credit, MAX_CREDIT)
                life.correct.append(1 if guess == truth else 0)
                life.layer.observe((feats, truth))
                life.samples += 1

                if life.samples % args.report_every == 0:
                    w = life.wants
                    print(f"  {life.gen:>4} {life.credit:>7.1f} "
                          f"{life.accuracy():>5.0f}% {str(w.committed):>12} "
                          f"{max(life.attended_counts, key=life.attended_counts.get):>14} "
                          f"{life.age_hours():>5.1f}h", flush=True)
                if life.samples % SAVE_EVERY == 0:
                    save(args.state, life, history)

                if life.credit <= 0:
                    rec = dict(gen=life.gen, hours=life.age_hours(),
                               samples=life.samples,
                               died=time.strftime("%Y-%m-%d %H:%M:%S"))
                    history.append(rec)
                    print(f"\n  GENERATION {life.gen} DIED after "
                          f"{life.age_hours():.1f}h")
                    child = Life(life.gen + 1, world.names, seed=life.gen + 1)
                    child.b.restore(life.b.snapshot())
                    child.layer.canary_baseline = child.layer._canary_score()
                    child.layer._canary_now = child.layer.canary_baseline
                    child.layer._checkpoint = child.b.snapshot()
                    print(f"  generation {child.gen} inherits and begins\n",
                          flush=True)
                    life = child
                    save(args.state, life, history)
                    break

            # ---- WANTS CHOOSES WHAT TO DO ----
            recent_acc = (sum(life.correct) / len(life.correct)
                          if life.correct else 0.5)
            life.wants.sense(life.credit, recent_acc, life.since_attended)
            action, signal = life.wants.choose()

            if action == "consolidate":
                cost = args.rounds * SLEEP_COST
                enough = (life.samples - life.last_sleep) >= SLEEP_REFRACTORY
                budget = life.sleeps < SLEEP_MAX_FRACTION * max(1, life.samples)
                if life.credit > cost + 5 and enough and budget:
                    life.layer.sleep(args.rounds)
                    life.credit -= cost
                    life.sleeps += 1
                    life.last_sleep = life.samples

            elif action == "attend" and signal is not None:
                # make a prediction about this signal, to be scored at horizon
                feats = features(life, world, signal)
                guess = life.b.predict(feats)
                pending.append((signal, now + horizon, guess, feats))
                life.attends += 1
                life.attended_counts[signal] += 1
                life.since_attended[signal] = 0
                # a means becomes an end: a signal whose structure it is
                # learning (accuracy rising) earns value of its own
                life.wants.learn_signal(signal, recent_acc)

            for nme in life.names:
                life.since_attended[nme] += 1
            if life.samples % 200 == 0:
                life.wants.decay()

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n  stopped")

    save(args.state, life, history)
    print(f"\n  generation {life.gen}, credit {life.credit:.1f}, "
          f"{life.samples:,} predictions")
    print(f"  it ranged over: {life.attended_counts}")
    print(f"  state in {os.path.abspath(args.state)}")


if __name__ == "__main__":
    main()
