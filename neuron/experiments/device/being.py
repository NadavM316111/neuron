"""One continuous life, with everything that works, in one program.

WHAT THIS IS. Until now the project has been experiments, each proving one
mechanism in isolation: continuous learning, memory, sleep, inheritance,
stakes, a drive. They lived in separate scripts and never ran as one thing.
This is the one thing. It perceives a world nobody designed, learns from it
continuously, consolidates offline, can lose a resource it needs and die,
passes what it learned to a successor, and survives being turned off.

It is the spine the rest of the project builds on, and it is deliberately
built ONLY from what has been shown to work. Where a mechanism failed, it
is absent and said to be absent, because a system that hides its gaps is
worse than one that names them.

THE WORLD IS FUSED. It perceives the person (keyboard presence, the
machine) and the outside (weather, and the open sky of public feeds through
the same channels reach.py used). Its one world is both at once, which is
the closest this can get to the real world without a transaction, and a
transaction needs an account and an adult.

WHAT IT PREDICTS. Whether the person will be present in HORIZON minutes.
A fact about a life, varying on the scale of minutes and hours, with real
daily and weekly structure that no pretrained model contains because it is
about one specific person. Validated as learnable: live2.py beat its
majority baseline by up to 37 points once presence became non-trivial.

THE MECHANISMS, each with the result that earned its place:

  CONTINUOUS LEARNING   the SleepLayer, learning one moment at a time
                        without catastrophic forgetting. The whole repo.
  MEMORY / PERSISTENCE  weights, buffer and counters to disk every
                        SAVE_EVERY samples; resumes by default. Survived a
                        20-hour gap on live2.
  SLEEP                 offline consolidation in blocks, which beat
                        interleaved by 8.9 points when free (20 Sep).
                        Here it costs credit, the third setting it has been
                        asked in.
  STAKES                credit that depletes; at zero the life ends and
                        nothing survives it. live3.py.
  INHERITANCE           a successor starts from the parent's weights.
                        Stage 4 (26 Aug); worked in a lineage, its value in
                        a population is still open.
  GUARD                 canary rollback with canary_from_stream ON, which
                        fired zero times over 79 hours of real deployment
                        against 232 and accelerating without it.

THE ONE THING THAT DOES NOT WORK, NAMED AT THE CENTRE RATHER THAN HIDDEN.

  WANTS. A system with a continuous life should WANT things, and that is
  the part closest to the heart of what this project is for. The drive —
  steering attention by accumulated prediction error — was the best
  attempt and it COLLAPSED at scale (drive_scale.py, 2 Oct): given 20
  options and a 1.5B model it fixated on two and ignored the rest, ending
  three times worse than reading everything evenly. So wants is present
  here ONLY as an explicit stub, `Wants`, below. It does nothing yet. It is
  written into the spine on purpose, so that anyone reading this code is
  looking straight at the open problem rather than a solved-looking
  placeholder. It is THE thing to build next, with a different mechanism
  than the one that failed.

HONEST ABOUT WHAT THIS IS NOT. A GRU predicting one person's presence is
not a mind, not a creature, and not alive in any sense beyond the
structural. What it is: the first time every working piece runs as one
continuous life instead of as separate proofs. The value is in the running,
over weeks, not in any single reading.

    python being.py --interval 30            # live it
    python being.py --status
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
FEED_EVERY = 600         # seconds between outside-feed reads

START_CREDIT = 100.0
MAX_CREDIT = 200.0
SAMPLE_COST = 0.10
CORRECT_PAYS = 0.15
SLEEP_COST = 0.05
SAVE_EVERY = 40
SLEEP_REFRACTORY = 60     # samples that must pass before it may sleep again
SLEEP_MAX_FRACTION = 0.15 # at most this share of samples may be sleeps

STATE_DIR = "being_state"
UA = "neuron-research/0.1 (personal experiment)"


# ========================================================================
# WANTS: THE OPEN PROBLEM, STUBBED ON PURPOSE.
# ========================================================================

class Wants:
    """The thing a living system should have and this one does not yet.

    A want is an internal state that persists and steers behaviour toward
    an end. The drive — steer by accumulated prediction error — was the
    best attempt and collapsed at scale (drive_scale.py, 2 Oct), fixating
    on a couple of options out of twenty and learning almost nothing.

    So this does nothing. It is here, named and at the centre, so the gap
    is impossible to miss. When a real wants mechanism exists, it lives
    here: it would read the system's state and bias what it attends to or
    what it does, and crucially it must NOT collapse onto one target the
    way the drive did. That anti-collapse property is the actual unsolved
    research question, not the wanting itself.

    Candidate directions not yet tried, recorded so the next attempt does
    not start from zero:
      - a want with a SATIATION term, so attending to a thing reduces its
        own pull and the system is pushed off it rather than fixating;
      - multiple competing wants with a budget, so none can take the whole
        of behaviour;
      - a want over OUTCOMES the system can affect, not over prediction
        error, so it cannot be hijacked by whatever is merely most
        surprising.
    """

    def __init__(self):
        self.active = False      # it is not. that is the point.

    def bias(self, state):
        """Would steer behaviour. Returns nothing, because wants is unsolved.
        A caller that wants the system to want something has to build this.
        """
        return None


# ========================================================================
# PERCEPTION: a fused world.
# ========================================================================

def idle_seconds():
    """Seconds since the last input event, or None. No permissions.

    Run under `caffeinate -s`, NOT `-i`: the -i flag resets this counter
    and silently destroys the presence signal (found 28 Sep, a 21-hour run
    lost to it)."""
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
            req = urllib.request.Request(self.url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=8) as r:
                c = json.load(r)["current"]
            self.vals = [
                min(1.0, max(0.0, (c["temperature_2m"] + 10) / 50)),
                min(1.0, c["precipitation"] / 10.0),
                c["cloud_cover"] / 100.0,
                min(1.0, c["wind_speed_10m"] / 60.0)]
        except Exception:
            pass
        return self.vals


class Feeds:
    """A thin slice of the outside world, cached. No keys, failure
    non-fatal. Two numbers the system cannot affect and cannot predict from
    its own state, so its world includes something genuinely external."""

    def __init__(self):
        self.at = 0.0
        self.vals = [0.5, 0.5]
        self.ref_hn = None

    def read(self):
        now = time.time()
        if now - self.at < FEED_EVERY:
            return self.vals
        self.at = now
        quakes, hn = self.vals
        try:
            req = urllib.request.Request(
                "https://earthquake.usgs.gov/earthquakes/feed/v1.0/"
                "summary/all_hour.geojson", headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=8) as r:
                quakes = min(1.0, len(json.load(r)["features"]) / 30.0)
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
        self.vals = [quakes, hn]
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


# machine 4, clock 5, weather 4, feeds 2, idle, present, slept, credit
INPUT = 4 + 5 + 4 + 2 + 1 + 1 + 1 + 1


# ========================================================================
# THE BRAIN.
# ========================================================================

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
            return float(F.cross_entropy(
                self.net(x, None)[0], y).item()), None

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
        self.wants = Wants()          # the open problem, present and inert
        self.credit = START_CREDIT
        self.born = time.time()
        self.samples = 0
        self.sleeps = 0
        self.last_sleep = -10**9
        self.pending = deque()
        self.correct = deque(maxlen=120)
        self.labels = deque(maxlen=120)

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
            sleeps=life.sleeps, wants_active=life.wants.active,
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
        print(f"  credit         {m.get('credit', 0):.1f} "
              f"(dies at 0)")
        print(f"  this life      {m.get('samples', 0):,} samples over "
              f"{m.get('age_hours', 0):.1f} hours")
        print(f"  accuracy       {m.get('accuracy', 0):.1f}% against "
              f"majority {m.get('majority', 0):.1f}%  "
              f"({m.get('edge', 0):+.1f})")
        print(f"  consolidations {m.get('sleeps', 0)}")
        print(f"  rollbacks      {m.get('rollbacks', 0)}")
        print(f"  wants          {'active' if m.get('wants_active') else 'UNSOLVED (stub)'}")
        hist = m.get("history", [])
        if hist:
            print(f"\n  LIVES THAT ENDED")
            for h in hist:
                print(f"    gen {h['gen']}: {h['hours']:.1f}h, "
                      f"{h['samples']:,} samples, edge {h['edge']:+.1f}")
        print(f"\n  last seen      {m.get('last_seen', '?')}")
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

    print(f"  ONE LIFE. perceives you and the outside, learns, sleeps, "
          f"can die, inherits.")
    print(f"  predicting your presence in {args.horizon:.0f} min. "
          f"break-even {100 * SAMPLE_COST / CORRECT_PAYS:.0f}% accuracy.")
    print(f"  WANTS IS A STUB. the drive collapsed at scale (2 Oct); "
          f"wants is the open problem.")
    print(f"  generation {gen}, {START_CREDIT:.0f} credit\n")
    print(f"  {'gen':>4} {'credit':>8} {'acc':>7} {'maj':>7} {'edge':>7} "
          f"{'here':>6} {'sleeps':>7} {'age':>7}")
    print("-" * 64)

    just_slept = 0.0
    started = time.time()

    try:
        while True:
            now = time.time()
            idle = idle_seconds()
            if idle is None:
                idle = 999.0
            here = 1.0 if idle < IDLE_PRESENT else 0.0

            feats = (machine() + clock() + weather.read() + feeds.read() +
                     [min(1.0, idle / 600.0), here, just_slept,
                      min(1.0, life.credit / MAX_CREDIT)])
            just_slept = 0.0

            # Wants would bias behaviour here. It returns None, on purpose.
            _ = life.wants.bias(feats)

            guess, p_present = life.b.predict(feats)
            life.pending.append((feats, now + horizon, guess))

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
                    print(f"  {life.gen:>4} {life.credit:>8.1f} "
                          f"{life.accuracy():>6.1f}% {life.majority():>6.1f}% "
                          f"{life.accuracy() - life.majority():>+6.1f} "
                          f"{100 * statistics.mean(life.labels):>5.0f}% "
                          f"{life.sleeps:>7} {life.age_hours():>6.1f}h",
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
                          f"{life.age_hours():.1f}h, {life.samples:,} "
                          f"samples, edge {rec['edge']:+.1f}")
                    # INHERITANCE. The successor starts from the parent's
                    # weights, not from nothing. Everything else is lost.
                    child = Life(life.gen + 1, seed=life.gen + 1)
                    child.b.restore(life.b.snapshot())
                    child.layer.canary_baseline = \
                        child.layer._canary_score()
                    child.layer._canary_now = child.layer.canary_baseline
                    child.layer._checkpoint = child.b.snapshot()
                    print(f"  generation {child.gen} inherits its weights "
                          f"and begins\n", flush=True)
                    life = child
                    save(args.state, life, history)
                    break

            # Sleep is now RATE-LIMITED. The over-sleep spiral (gen 0,
            # 3 Oct): it slept every tick it believed you were away, each
            # sleep costing credit and perturbing the weights, so a genuine
            # absence consolidated the system to death -- 168 sleeps in
            # 1,240 samples, accuracy below the majority baseline, credit
            # bleeding out. An animal does not sleep every idle minute; it
            # sleeps once a cycle. Two gates:
            #   refractory: enough NEW samples since the last sleep that
            #               there is something worth consolidating.
            #   budget:     sleeps may not exceed a fraction of all samples,
            #               so no stretch of absence can run the credit down
            #               through consolidation alone.
            cost = args.rounds * SLEEP_COST
            enough_new = (life.samples - life.last_sleep) >= SLEEP_REFRACTORY
            under_budget = life.sleeps < SLEEP_MAX_FRACTION * max(
                1, life.samples)
            if (p_present < 0.35 and here == 0.0 and life.credit > cost + 5
                    and enough_new and under_budget):
                life.layer.sleep(args.rounds)
                life.credit -= cost
                life.sleeps += 1
                life.last_sleep = life.samples
                just_slept = 1.0

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n  stopped")

    save(args.state, life, history)
    print(f"\n  generation {life.gen}, credit {life.credit:.1f}, "
          f"{life.samples:,} samples, {len(history)} deaths")
    print(f"  state in {os.path.abspath(args.state)}")
    print(f"  wants is still the open problem. that is the next build.")


if __name__ == "__main__":
    main()
