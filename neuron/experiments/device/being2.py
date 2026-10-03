"""The living system, now with wants driving what it does.

WHAT CHANGED FROM being.py. being.py had every mechanism that works, running
as one continuous life, with wants as an explicit STUB: a `Wants` class that
did nothing, sitting at the centre so the gap could not be hidden. Over the
weekend wants stopped being a gap. Five results, each a separate experiment,
each with a control:

  pursue      reach a preferred state through a learned model, no reward
              (goal.py)
  persist     keep the goal when knocked off course (goal_persist.py)
  self-form   generate a goal from an internal need, unprompted
              (goal_own.py)
  juggle      hold several needs at once without collapse, via COMMITMENT
              (goal_many.py)
  develop     acquire a NEW want from a life: a means becomes an end
              (goal_learned.py)

This file puts that account where the stub was. Wanting now steers the
system's real actions instead of being inert.

THE HONEST SHAPE OF THE ACTION SPACE. The vision wants the system acting in
the human world. It cannot yet: no account, no standing, that is a later
rung. So the actions it really has are over its own existence and its thin
window outside, and every one of them is REAL rather than pretend:

  consolidate   spend credit and perturb the weights, to predict better
                later. (being.py already had this; the over-sleep spiral of
                3 Oct is why it is now rate-limited.)
  reach         spend a tick reading an outside feed, to satisfy curiosity
                about a world it cannot control.
  attend        do neither; just watch and learn from the person-stream.

A small space, but wanting steering three real actions is more honest than
a large space that is mostly theatre.

THE NEEDS IT WANTS OVER. Internal states it reads and tries to keep healthy.
Each rises or falls on its own and an action resolves it, exactly the
structure goal_own.py and goal_many.py were built on:

  solvency    rises as credit falls. resolved by predicting well, which is
              passive, so the want it creates is "do not waste credit on
              consolidation you cannot afford".
  sharpness   rises as recent accuracy drops. resolved by consolidating,
              which is the bet that offline replay sharpens prediction.
  curiosity   rises over time since the last outside read. resolved by
              reaching out.

HOW IT CHOOSES, using the pieces that worked. At each decision it ranks its
needs, COMMITS to the most urgent (the commitment rule from goal_many.py
that fixed the dithering collapse), and takes the action that serves it.
Serving a need drops it, so attention moves on and no single want takes
over. And a LEARNED value can attach to the states that reliably precede
relief (goal_learned.py), so the system can come to value, say, the quiet
hours that let it consolidate cheaply.

WHAT THIS IS NOT. The action space is thin and the world is slow and
partly hidden, so this will not show the clean separations the grid did.
The grid PROVED the mechanisms; this tests whether they survive a real,
sparse, slow world. That is the actual frontier and it is genuinely
uncertain. A flat or messy result here is a finding about deployment, not a
refutation of the wants work, which stands on its own experiments.

    python being2.py --interval 30
    python being2.py --status
"""

import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import urllib.request
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))

from sleeping import SleepLayer          # noqa: E402


AWAY, PRESENT = 0, 1
CLASSES = 2
HIDDEN = 64
LR = 3e-4
SEQ_LEN = 8
REHEARSE_PER_ITEM = 6

HORIZON_MIN = 5.0
IDLE_PRESENT = 90.0
WEATHER_EVERY = 900
FEED_EVERY = 600

START_CREDIT = 100.0
MAX_CREDIT = 200.0
SAMPLE_COST = 0.10
CORRECT_PAYS = 0.15
SLEEP_COST = 0.05
SAVE_EVERY = 40
SLEEP_REFRACTORY = 60
SLEEP_MAX_FRACTION = 0.15

# wants
NEED_RISE_CURIOSITY = 0.02
SWITCH_MARGIN = 0.25           # commitment: another need must exceed by this
DONE_BELOW = 0.20              # a need this low counts as handled
VALUE_LR = 0.15                # learned value on states preceding relief
VALUE_DECAY = 0.98

STATE_DIR = "being2_state"
UA = "neuron-research/0.1 (personal experiment)"


# ------------------------------------------------------------ perception

def idle_seconds():
    """Seconds since last input. Run under caffeinate -s, never -i (the -i
    flag resets this counter and destroys the signal; 28 Sep)."""
    try:
        out = subprocess.run(["ioreg", "-c", "IOHIDSystem"],
                             capture_output=True, text=True,
                             timeout=5).stdout
        for line in out.splitlines():
            if "HIDIdleTime" in line:
                return int(line.split("=")[-1].strip()) / 1e9
    except Exception:
        pass
    return None


class Weather:
    def __init__(self, lat, lon):
        self.url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}"
                    f"&longitude={lon}&current=temperature_2m,"
                    f"precipitation,cloud_cover,wind_speed_10m")
        self.at = 0.0
        self.vals = [0.5, 0.0, 0.5, 0.0]

    def read(self):
        now = time.time()
        if now - self.at < WEATHER_EVERY:
            return self.vals
        self.at = now
        try:
            req = urllib.request.Request(self.url,
                                        headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=8) as r:
                c = json.load(r)["current"]
            self.vals = [min(1.0, max(0.0, (c["temperature_2m"] + 10) / 50)),
                         min(1.0, c["precipitation"] / 10.0),
                         c["cloud_cover"] / 100.0,
                         min(1.0, c["wind_speed_10m"] / 60.0)]
        except Exception:
            pass
        return self.vals


class Feeds:
    """Outside world the system can CHOOSE to read, by spending a tick on it
    (the 'reach' action). Cached so a choice to reach is cheap but real."""

    def __init__(self):
        self.vals = [0.5, 0.5]
        self.ref_hn = None
        self.reads = 0

    def reach(self):
        self.reads += 1
        q, hn = self.vals
        try:
            req = urllib.request.Request(
                "https://earthquake.usgs.gov/earthquakes/feed/v1.0/"
                "summary/all_hour.geojson", headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=8) as r:
                q = min(1.0, len(json.load(r)["features"]) / 30.0)
        except Exception:
            pass
        try:
            req = urllib.request.Request(
                "https://hacker-news.firebaseio.com/v0/maxitem.json",
                headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=8) as r:
                m = float(json.load(r))
            if self.ref_hn is None:
                self.ref_hn = m
            hn = min(1.0, max(0.0, 0.5 + (m - self.ref_hn) / 2e5))
        except Exception:
            pass
        self.vals = [q, hn]
        return self.vals


def machine():
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=None) / 100.0
        mem = psutil.virtual_memory().percent / 100.0
        try:
            b = psutil.sensors_battery()
            batt = (b.percent / 100.0) if b else 0.5
            plug = 1.0 if (b and b.power_plugged) else 0.0
        except Exception:
            batt, plug = 0.5, 0.0
        return [cpu, mem, batt, plug]
    except ImportError:
        return [min(1.0, os.getloadavg()[0] / 8.0), 0.5, 0.5, 0.0]


def clock():
    t = time.localtime()
    h = (t.tm_hour + t.tm_min / 60.0) / 24.0
    d = t.tm_wday / 7.0
    weekend = 1.0 if t.tm_wday >= 5 else 0.0
    return [math.sin(2 * math.pi * h), math.cos(2 * math.pi * h),
            math.sin(2 * math.pi * d), math.cos(2 * math.pi * d), weekend]


INPUT = 4 + 5 + 4 + 2 + 1 + 1 + 1 + 1


# ------------------------------------------------------------ wants

class Wants:
    """No longer a stub. Internal needs, a commitment rule to juggle them,
    and learned value so a means can become an end.

    This drives the system's real choice of action. It reads needs the Life
    computes, commits to the most urgent, and names the action that serves
    it. Serving drops the need, so focus moves on; no want can take over.
    """

    active = True

    def __init__(self):
        # needs in [0,1]; each is resolved by a different action
        self.need = dict(solvency=0.0, sharpness=0.0, curiosity=0.0)
        self.committed = None
        # which action serves which need
        self.server = dict(solvency="attend",       # stop spending: just watch
                           sharpness="consolidate",  # sharpen by replay
                           curiosity="reach")         # look outside
        # learned value over clock-hours that reliably precede cheap relief
        self.hour_value = [0.0] * 24

    def sense(self, credit, recent_acc, since_reach):
        """Update needs from the system's real state."""
        self.need["solvency"] = max(0.0, 1.0 - credit / START_CREDIT)
        self.need["sharpness"] = max(0.0, 1.0 - recent_acc)
        self.need["curiosity"] = min(1.0, since_reach * NEED_RISE_CURIOSITY)

    def choose(self):
        """Commit to the most urgent need and return the action that serves
        it. The commitment rule (goal_many.py) prevents the dithering
        collapse: stay on a need until it is handled or clearly overtaken."""
        worst = max(self.need, key=lambda k: self.need[k])
        c = self.committed
        if c is None:
            c = worst
        elif self.need[worst] > self.need[c] + SWITCH_MARGIN:
            c = worst
        elif self.need[c] < DONE_BELOW:
            c = worst
        self.committed = c
        return self.server[c], c

    def learn_hour(self, hour, relief):
        """A means becomes an end: an hour that reliably precedes cheap
        relief (low need, easy consolidation) earns value of its own
        (goal_learned.py)."""
        old = self.hour_value[hour]
        self.hour_value[hour] = old + VALUE_LR * (relief - old)

    def decay(self):
        for i in range(24):
            self.hour_value[i] *= VALUE_DECAY

    def summary(self):
        return dict(need={k: round(v, 3) for k, v in self.need.items()},
                    committed=self.committed,
                    hour_value_max=round(max(self.hour_value), 3))


# ------------------------------------------------------------ brain

class Net(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.enc = nn.Linear(INPUT, HIDDEN)
        self.cell = nn.GRUCell(HIDDEN, HIDDEN)
        self.head = nn.Linear(HIDDEN, CLASSES)

    def forward(self, x, h=None):
        return self.head(self.cell(F.relu(self.enc(x)), h)), h


class Backend:
    def __init__(self, seed=0):
        self.net = Net(seed)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.h = None

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
        self.h = None

    def reset_state(self):
        self.h = None

    def snapshot(self):
        return {k: v.detach().clone()
                for k, v in self.net.state_dict().items()}

    def restore(self, state):
        self.net.load_state_dict(state)

    def predict(self, feats):
        self.net.eval()
        with torch.no_grad():
            x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
            p = F.softmax(self.net(x, None)[0], dim=1)[0]
            return int(p.argmax().item()), float(p[PRESENT].item())


def make_layer(b, seed):
    rng = random.Random(9000 + seed)
    canary = [([rng.random() for _ in range(INPUT)],
               rng.choice([AWAY, PRESENT])) for _ in range(10)]
    return SleepLayer(
        b, canary=canary, seed=seed, contiguous=True,
        window=200, warmup=30, top_fraction=1.00,
        coherence_veto=1e9, loss_floor=0.02, steps_per_update=1,
        rehearse_per_item=REHEARSE_PER_ITEM, rehearse_count=2,
        rehearse_steps=1, anchor_size=100, buffer_size=500,
        sequence_len=SEQ_LEN, replay_policy="uniform",
        canary_from_stream=12,
        guard=True, guard_per_item=500, canary_tolerance=0.5)


class Life:
    def __init__(self, gen, seed):
        self.gen = gen
        self.b = Backend(seed)
        self.layer = make_layer(self.b, seed)
        self.wants = Wants()
        self.credit = START_CREDIT
        self.born = time.time()
        self.samples = 0
        self.sleeps = 0
        self.reaches = 0
        self.last_sleep = -10 ** 9
        self.since_reach = 0
        self.pending = deque()
        self.correct = deque(maxlen=120)
        self.labels = deque(maxlen=120)
        self.served = {"attend": 0, "consolidate": 0, "reach": 0}

    def age_hours(self):
        return (time.time() - self.born) / 3600.0

    def accuracy(self):
        return (100.0 * sum(self.correct) / len(self.correct)
                if self.correct else 0.0)

    def majority(self):
        if not self.labels:
            return 0.0
        return 100.0 * max(self.labels.count(PRESENT),
                           self.labels.count(AWAY)) / len(self.labels)


def save(path, life, history):
    os.makedirs(path, exist_ok=True)
    torch.save(life.b.snapshot(), os.path.join(path, "weights.pt"))
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(dict(
            generation=life.gen, credit=life.credit, samples=life.samples,
            age_hours=life.age_hours(), accuracy=life.accuracy(),
            majority=life.majority(), edge=life.accuracy() - life.majority(),
            sleeps=life.sleeps, reaches=life.reaches,
            served=life.served, wants=life.wants.summary(),
            rollbacks=life.layer.summary().get("rollbacks", 0),
            history=history,
            last_seen=time.strftime("%Y-%m-%d %H:%M:%S")), f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--horizon", type=float, default=HORIZON_MIN)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--state", default=STATE_DIR)
    ap.add_argument("--report-every", type=int, default=40)
    ap.add_argument("--lat", type=float, default=26.12)
    ap.add_argument("--lon", type=float, default=-80.14)
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
        print(f"  this life      {m.get('samples', 0):,} samples over "
              f"{m.get('age_hours', 0):.1f} hours")
        print(f"  accuracy       {m.get('accuracy', 0):.1f}% vs majority "
              f"{m.get('majority', 0):.1f}%  ({m.get('edge', 0):+.1f})")
        print(f"  consolidations {m.get('sleeps', 0)}   "
              f"outside reaches {m.get('reaches', 0)}")
        w = m.get("wants", {})
        print(f"  wants          ACTIVE")
        print(f"    needs        {w.get('need', {})}")
        print(f"    committed to {w.get('committed')}")
        print(f"    actions      {m.get('served', {})}")
        print(f"    learned hour-value peak {w.get('hour_value_max', 0)}")
        print(f"  rollbacks      {m.get('rollbacks', 0)}")
        for h in m.get("history", []):
            print(f"    gen {h['gen']}: {h['hours']:.1f}h, "
                  f"{h['samples']:,} samples, edge {h['edge']:+.1f}")
        print(f"  last seen      {m.get('last_seen', '?')}")
        return

    if idle_seconds() is None:
        print("  cannot read presence on this system. Not starting.")
        return

    weather = Weather(args.lat, args.lon)
    feeds = Feeds()
    horizon = args.horizon * 60.0

    history = []
    mp = os.path.join(args.state, "meta.json")
    if os.path.exists(mp):
        with open(mp) as f:
            history = json.load(f).get("history", [])
    gen = (history[-1]["gen"] + 1) if history else 0
    life = Life(gen, seed=gen)

    print(f"  ONE LIFE, with WANTS now driving what it does.")
    print(f"  needs: solvency, sharpness, curiosity. it commits to the "
          f"most urgent and acts.")
    print(f"  actions: attend (watch), consolidate (sharpen), reach (look "
          f"outside).")
    print(f"  wants is no longer a stub: five results over the weekend put "
          f"it here.")
    print(f"  generation {gen}, {START_CREDIT:.0f} credit\n")
    print(f"  {'gen':>4} {'credit':>7} {'edge':>6} {'want':>10} "
          f"{'slp':>4} {'rch':>4} {'age':>6}")
    print("-" * 60)

    just_slept = 0.0
    started = time.time()
    last_feed = feeds.vals

    try:
        while True:
            now = time.time()
            idle = idle_seconds()
            if idle is None:
                idle = 999.0
            here = 1.0 if idle < IDLE_PRESENT else 0.0

            feats = (machine() + clock() + weather.read() + last_feed +
                     [min(1.0, idle / 600.0), here, just_slept,
                      min(1.0, life.credit / MAX_CREDIT)])
            just_slept = 0.0

            guess, p_present = life.b.predict(feats)
            life.pending.append((feats, now + horizon, guess))

            # resolve matured predictions, run the credit economy
            while life.pending and life.pending[0][1] <= now:
                old_feats, _, old_guess = life.pending.popleft()
                label = PRESENT if here else AWAY
                life.credit -= SAMPLE_COST
                if old_guess == label:
                    life.credit += CORRECT_PAYS
                life.credit = min(life.credit, MAX_CREDIT)
                life.correct.append(1 if old_guess == label else 0)
                life.labels.append(label)
                life.layer.observe((old_feats, label))
                life.samples += 1

                if life.samples % args.report_every == 0:
                    w = life.wants
                    print(f"  {life.gen:>4} {life.credit:>7.1f} "
                          f"{life.accuracy() - life.majority():>+6.1f} "
                          f"{str(w.committed):>10} {life.sleeps:>4} "
                          f"{life.reaches:>4} {life.age_hours():>5.1f}h",
                          flush=True)
                if life.samples % SAVE_EVERY == 0:
                    save(args.state, life, history)

                if life.credit <= 0:
                    rec = dict(gen=life.gen, hours=life.age_hours(),
                               samples=life.samples,
                               edge=life.accuracy() - life.majority(),
                               died=time.strftime("%Y-%m-%d %H:%M:%S"))
                    history.append(rec)
                    print(f"\n  GENERATION {life.gen} DIED: "
                          f"{life.age_hours():.1f}h, edge {rec['edge']:+.1f}")
                    child = Life(life.gen + 1, seed=life.gen + 1)
                    child.b.restore(life.b.snapshot())
                    child.layer.canary_baseline = child.layer._canary_score()
                    child.layer._canary_now = child.layer.canary_baseline
                    child.layer._checkpoint = child.b.snapshot()
                    print(f"  generation {child.gen} inherits and begins\n",
                          flush=True)
                    life = child
                    save(args.state, life, history)
                    break

            # ---- WANTS DRIVES THE ACTION ----
            recent_acc = (sum(life.correct) / len(life.correct)
                          if life.correct else 0.5)
            life.wants.sense(life.credit, recent_acc, life.since_reach)
            action, need_name = life.wants.choose()
            hour = time.localtime().tm_hour

            if action == "consolidate":
                cost = args.rounds * SLEEP_COST
                enough_new = (life.samples - life.last_sleep) >= \
                    SLEEP_REFRACTORY
                under_budget = life.sleeps < SLEEP_MAX_FRACTION * max(
                    1, life.samples)
                if life.credit > cost + 5 and enough_new and under_budget:
                    life.layer.sleep(args.rounds)
                    life.credit -= cost
                    life.sleeps += 1
                    life.last_sleep = life.samples
                    life.served["consolidate"] += 1
                    just_slept = 1.0
                    # a means becomes an end: if consolidating now was cheap
                    # (plenty of credit), the current hour earns value
                    relief = max(0.0, life.credit / START_CREDIT - 0.5)
                    life.wants.learn_hour(hour, relief)
                else:
                    life.served["attend"] += 1

            elif action == "reach":
                # only actually spend the reach if curiosity is real and we
                # are not mid-presence (reaching out while away is cheap)
                if life.since_reach > 20:
                    last_feed = feeds.reach()
                    life.reaches += 1
                    life.since_reach = 0
                    life.served["reach"] += 1
                else:
                    life.served["attend"] += 1
            else:
                life.served["attend"] += 1

            life.since_reach += 1
            if life.samples % 200 == 0:
                life.wants.decay()

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n  stopped")

    save(args.state, life, history)
    print(f"\n  generation {life.gen}, credit {life.credit:.1f}, "
          f"{life.samples:,} samples, {len(history)} deaths")
    print(f"  wants drove {life.served} across its actions")
    print(f"  state in {os.path.abspath(args.state)}")


if __name__ == "__main__":
    main()
