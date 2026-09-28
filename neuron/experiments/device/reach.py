"""It reaches outside, and the world answers.

THE RUNG THIS IS. Everything so far has acted only on itself. In live.py
the action was consolidating, which costs CPU, which it then observed: a
closed loop, but closed entirely inside the machine. Nothing this project
has built has ever done something that left the computer and had the world
respond.

Here the action is a request to the outside. The system chooses which
public source to ask, the internet answers, and the answer is data it could
not have obtained any other way. The loop closes through something that is
not its own substrate and that it cannot control.

WHAT IT IS ACTUALLY DOING. It tracks several public sources — weather,
earthquakes, a price, a front page — and predicts where each will be next.
It can only find out by ASKING, and it may ask for one source per tick.
So it has to decide where to spend attention:

  a source that barely moves is cheap to predict and wasteful to query
  a source that keeps surprising it is where the asking is worth it

That is a real allocation problem with a real cost, and it is the same
shape as everything above this rung. "Which source do I ask" becomes "which
person do I ask" and then "what do I spend" without changing the structure.

THREE POLICIES, RUNNING SIDE BY SIDE IN ONE PROCESS so they face identical
conditions. Each keeps its own model and makes its own choices, so they
genuinely differ in what they learn.

  round      cycle through the sources in order. The dumbest sensible
             allocation and the one to beat.
  random     ask at random.
  driven     ask where accumulated prediction error is highest. This is
             the drive from drive2.py (20 Sep), which beat every other
             action policy on the grid by 8 to 24 points, applied for the
             first time to an action that reaches outside.

THE AUDIT IS WHAT MAKES THEM COMPARABLE. Policies query different sources,
so their own query results cannot be compared directly. Every AUDIT_EVERY
ticks, every source is read once and every policy is scored on its current
prediction of every source, whether it chose to look or not. That is the
number that matters: how wrong is each policy about the world as a whole,
given the same budget.

HONEST ABOUT THE SIZE OF THIS. Reading a public JSON endpoint is a small
thing to call acting in the world. What makes it the right rung is the
structure rather than the magnitude: it leaves the machine, the world
answers on its own terms, the answer cannot be predicted from internal
state, and the budget makes the choice cost something.

POLITE BY DESIGN. One query per policy per tick, three policies, default
60-second ticks, so about three requests a minute across all of them. Every
source has a cache floor so a fast interval cannot hammer anything. Sources
that fail are dropped and reported rather than retried in a loop.

    python reach.py --minutes 30          # a first look
    python reach.py --interval 120        # leave it living
    python reach.py --status
"""

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))

from sleeping import SleepLayer          # noqa: E402


HIDDEN = 48
LR = 3e-4
SEQ_LEN = 8
REHEARSE_PER_ITEM = 6
DOWN, SAME, UP = 0, 1, 2
CLASSES = 3
CHANGE = 0.02            # fraction of range that counts as a move
MIN_GAP = 45.0           # seconds, per source, however fast we tick
AUDIT_EVERY = 10
DRIVE_DECAY = 0.85
DRIVE_TEMP = 0.25
EPSILON = 0.15
STATE_DIR = "reach_state"

UA = "neuron-research/0.1 (personal experiment)"


# ---------------------------------------------------------------- sources

def _get(url, timeout=10):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def src_weather(lat, lon):
    d = _get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}"
             f"&longitude={lon}&current=temperature_2m,wind_speed_10m")
    return (d["current"]["temperature_2m"] + 10) / 50.0


def src_wind(lat, lon):
    d = _get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}"
             f"&longitude={lon}&current=wind_speed_10m")
    return min(1.0, d["current"]["wind_speed_10m"] / 60.0)


def src_quakes(_lat, _lon):
    d = _get("https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/"
             "all_hour.geojson")
    return min(1.0, len(d["features"]) / 30.0)


def src_btc(_lat, _lon):
    d = _get("https://api.coinbase.com/v2/prices/BTC-USD/spot")
    # Normalised by a rolling reference set on first read, because an
    # absolute price has no bounded range and would swamp the input layer.
    return float(d["data"]["amount"])


def src_hn(_lat, _lon):
    d = _get("https://hacker-news.firebaseio.com/v0/maxitem.json")
    return float(d)


SOURCES = [("weather", src_weather, False),
           ("wind", src_wind, False),
           ("quakes", src_quakes, False),
           ("btc", src_btc, True),
           ("hn", src_hn, True)]


class World:
    """The outside. Each source is asked at most once per MIN_GAP.

    Sources that fail on the opening probe are dropped, because a source
    that is not answering is not a world, it is a timeout, and leaving it
    in would make every policy's allocation partly a measure of which
    endpoints are down.
    """

    def __init__(self, lat, lon):
        self.lat, self.lon = lat, lon
        self.names = []
        self.fns = {}
        self.unbounded = {}
        self.last_at = {}
        self.last_val = {}
        self.ref = {}
        self.fails = {}

    def probe(self):
        print("  probing sources:")
        for name, fn, unbounded in SOURCES:
            try:
                v = fn(self.lat, self.lon)
                self.names.append(name)
                self.fns[name] = fn
                self.unbounded[name] = unbounded
                self.ref[name] = v if unbounded else 1.0
                self.last_val[name] = self._norm(name, v)
                self.last_at[name] = time.time()
                self.fails[name] = 0
                print(f"    {name:<9} ok    raw {v:.4f}")
            except Exception as e:
                print(f"    {name:<9} DROPPED  {type(e).__name__}")
        return len(self.names)

    def _norm(self, name, v):
        """Unbounded sources are scaled by their first reading, so what the
        model sees is relative movement rather than an absolute magnitude
        it has no way to interpret."""
        if not self.unbounded[name]:
            return max(0.0, min(1.0, v))
        r = self.ref.get(name) or v or 1.0
        return max(0.0, min(1.0, 0.5 * v / r))

    def ask(self, name):
        """Spend a query. Returns the new normalised value, or None if the
        source is rate-limited right now or failed."""
        now = time.time()
        if now - self.last_at.get(name, 0) < MIN_GAP:
            return None
        try:
            v = self._norm(name, self.fns[name](self.lat, self.lon))
        except Exception:
            self.fails[name] = self.fails.get(name, 0) + 1
            self.last_at[name] = now
            return None
        self.last_at[name] = now
        self.last_val[name] = v
        return v


def clock():
    t = time.localtime()
    h = (t.tm_hour + t.tm_min / 60.0) / 24.0
    d = t.tm_wday / 7.0
    return [math.sin(2 * math.pi * h), math.cos(2 * math.pi * h),
            math.sin(2 * math.pi * d), math.cos(2 * math.pi * d)]


def bucket(old, new):
    if new < old - CHANGE:
        return DOWN
    if new > old + CHANGE:
        return UP
    return SAME


# ---------------------------------------------------------------- model

class Net(nn.Module):
    def __init__(self, n_src, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.inp = n_src + 4 + 2     # which source, clock, value and age
        self.enc = nn.Linear(self.inp, HIDDEN)
        self.cell = nn.GRUCell(HIDDEN, HIDDEN)
        self.head = nn.Linear(HIDDEN, CLASSES)

    def forward(self, x, h=None):
        return self.head(self.cell(F.relu(self.enc(x)), h)), h


class Backend:
    def __init__(self, n_src, seed=0):
        self.net = Net(n_src, seed)
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
            return int(self.net(x, None)[0].argmax(1).item())


class Policy:
    """One way of deciding what to ask, with its own model and its own
    memory. Three of these run side by side against the same world."""

    def __init__(self, name, world, seed):
        self.name = name
        self.n = len(world.names)
        self.b = Backend(self.n, seed)
        self.layer = SleepLayer(
            self.b, canary=[([random.random() for _ in
                              range(self.b.net.inp)],
                             random.choice([DOWN, SAME, UP]))
                            for _ in range(8)],
            seed=seed, contiguous=True,
            window=200, warmup=20, top_fraction=1.00,
            coherence_veto=1e9, loss_floor=0.02, steps_per_update=1,
            rehearse_per_item=REHEARSE_PER_ITEM, rehearse_count=2,
            rehearse_steps=1, anchor_size=60, buffer_size=300,
            sequence_len=SEQ_LEN, replay_policy="uniform",
            canary_from_stream=10,
            guard=True, guard_per_item=400, canary_tolerance=0.5)
        self.drive = {s: 1.0 for s in world.names}
        self.rng = random.Random(seed + 77)
        self.turn = 0
        self.asked = {s: 0 for s in world.names}
        self.audit_hits = deque(maxlen=400)

    def features(self, world, name, age):
        one = [1.0 if s == name else 0.0 for s in world.names]
        return one + clock() + [world.last_val.get(name, 0.5),
                                min(1.0, age / 3600.0)]

    def choose(self, world):
        if self.rng.random() < EPSILON:
            return self.rng.choice(world.names)
        if self.name == "round":
            s = world.names[self.turn % len(world.names)]
            self.turn += 1
            return s
        if self.name == "random":
            return self.rng.choice(world.names)
        scores = [self.drive[s] for s in world.names]
        hi = max(scores)
        w = [math.exp((v - hi) / DRIVE_TEMP) for v in scores]
        r = self.rng.random() * sum(w)
        acc = 0.0
        for s, weight in zip(world.names, w):
            acc += weight
            if r <= acc:
                return s
        return world.names[-1]

    def learn(self, world, name, before, after, age, feats=None):
        # feats MUST be captured before the query. world.ask() overwrites
        # world.last_val with the answer, and features() reads last_val,
        # so building them afterwards hands the model the very value it is
        # being asked to predict the direction of. That leak is why round
        # and random produced identical accuracy to a tenth of a point at
        # all 77 checkpoints of the 21-hour run: every policy was reading
        # the answer, so none of them was really predicting.
        f = feats if feats is not None else self.features(world, name, age)
        label = bucket(before, after)
        item = (f, label)
        s, _ = self.b.score(item)
        self.drive[name] = (DRIVE_DECAY * self.drive[name]
                            + (1 - DRIVE_DECAY) * s)
        self.layer.observe(item)
        self.asked[name] += 1

    def audit(self, world, name, before, after, age, feats=None):
        """Scored on a source whether or not it chose to look. This is the
        only number comparable across policies, because they spend their
        budgets differently by design.

        feats must be captured before the query, for the same reason as in
        learn()."""
        f = feats if feats is not None else self.features(world, name, age)
        self.audit_hits.append(
            1 if self.b.predict(f) == bucket(before, after) else 0)

    def accuracy(self):
        return (100.0 * sum(self.audit_hits) / len(self.audit_hits)
                if self.audit_hits else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--minutes", type=float, default=0)
    ap.add_argument("--lat", type=float, default=26.12)
    ap.add_argument("--lon", type=float, default=-80.14)
    ap.add_argument("--state", default=STATE_DIR)
    ap.add_argument("--report-every", type=int, default=10)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.status:
        mp = os.path.join(args.state, "meta.json")
        if not os.path.exists(mp):
            print("  nothing has reached out yet")
            return
        with open(mp) as f:
            m = json.load(f)
        print(f"  ticks          {m.get('ticks', 0):,}")
        print(f"  queries made   {m.get('queries', 0):,}")
        print(f"  alive for      {m.get('hours', 0):.1f} hours")
        print(f"  sources        {', '.join(m.get('sources', []))}")
        for name, acc in m.get("accuracy", {}).items():
            print(f"    {name:<8} {acc:.1f}%")
        print(f"  last seen      {m.get('last_seen', '?')}")
        return

    world = World(args.lat, args.lon)
    n = world.probe()
    if n < 2:
        print(f"\n  only {n} source(s) answered. With fewer than two there "
              f"is no allocation")
        print(f"  problem to solve and nothing here would measure "
              f"anything. Check the network.")
        return

    policies = [Policy(p, world, i)
                for i, p in enumerate(["round", "random", "driven"])]

    print(f"\n  {n} sources, {len(policies)} policies, one query each per "
          f"tick, every {args.interval:.0f}s")
    print(f"  every {AUDIT_EVERY} ticks all sources are read and every "
          f"policy is scored on all of them")
    print(f"  that audit is the comparable number; what each policy CHOSE "
          f"to ask is the action\n")
    print(f"  {'tick':>6} " +
          " ".join(f"{p.name:>9}" for p in policies) +
          f" {'queries':>8} {'elapsed':>9}")
    print("-" * 60)

    ticks = 0
    queries = 0
    started = time.time()
    stop = started + args.minutes * 60 if args.minutes else None

    try:
        while True:
            if stop and time.time() > stop:
                break
            ticks += 1

            for p in policies:
                # A REFUSAL IS NOT A TURN. The first run gave `driven` 8
                # queries against round-robin's 19, because the drive
                # concentrates on one source, which then sits inside its
                # MIN_GAP cache window and refuses. Round-robin never
                # repeats so it is never blocked. That penalised the drive
                # for doing exactly what it is designed to do and made the
                # comparison unfair before it started.
                #
                # So a policy keeps choosing until the world actually
                # answers, or until every source has refused it this tick.
                # Every policy now spends the same budget and the only
                # difference between them is WHERE.
                tried = set()
                while len(tried) < len(world.names):
                    name = p.choose(world)
                    if name in tried:
                        remaining = [s for s in world.names
                                     if s not in tried]
                        name = p.rng.choice(remaining)
                    tried.add(name)
                    before = world.last_val.get(name, 0.5)
                    age = time.time() - world.last_at.get(
                        name, time.time())
                    feats = p.features(world, name, age)
                    after = world.ask(name)
                    if after is None:
                        continue
                    queries += 1
                    p.learn(world, name, before, after, age, feats)
                    break

            if ticks % AUDIT_EVERY == 0:
                for name in world.names:
                    before = world.last_val.get(name, 0.5)
                    age = time.time() - world.last_at.get(name, time.time())
                    snap = {p.name: p.features(world, name, age)
                            for p in policies}
                    after = world.ask(name)
                    if after is None:
                        continue
                    for p in policies:
                        p.audit(world, name, before, after, age,
                                snap[p.name])

            if ticks % args.report_every == 0:
                print(f"  {ticks:>6} " +
                      " ".join(f"{p.accuracy():>8.1f}%" for p in policies) +
                      f" {queries:>8} "
                      f"{(time.time() - started) / 60:>8.1f}m", flush=True)

                os.makedirs(args.state, exist_ok=True)
                with open(os.path.join(args.state, "meta.json"), "w") as f:
                    json.dump(dict(
                        ticks=ticks, queries=queries,
                        hours=(time.time() - started) / 3600,
                        sources=world.names,
                        accuracy={p.name: p.accuracy() for p in policies},
                        asked={p.name: p.asked for p in policies},
                        last_seen=time.strftime("%Y-%m-%d %H:%M:%S")), f,
                        indent=2)

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n  stopped")

    print("\n" + "=" * 70)
    print(f"  {ticks} ticks, {queries} queries to the outside, "
          f"{(time.time() - started) / 60:.0f} minutes")
    print("=" * 70)
    print(f"\n  WHERE EACH POLICY SPENT ITS QUERIES")
    print(f"  {'policy':>8} " + " ".join(f"{s:>9}" for s in world.names))
    for p in policies:
        print(f"  {p.name:>8} " +
              " ".join(f"{p.asked[s]:>9}" for s in world.names))

    print(f"\n  ACCURACY ON THE AUDIT — every source, whether asked or not")
    for p in policies:
        print(f"  {p.name:>8} {p.accuracy():>7.1f}%  "
              f"({len(p.audit_hits)} scored)")

    if any(len(p.audit_hits) < 30 for p in policies):
        print(f"\n  TOO FEW AUDITS to compare. Each audit needs "
              f"{AUDIT_EVERY} ticks, so this needs")
        print(f"  hours rather than minutes. What it HAS shown is that "
              f"the loop runs: it")
        print(f"  chose, it reached outside, the world answered, and it "
              f"learned from the answer.")
    else:
        best = max(policies, key=lambda p: p.accuracy())
        rnd = [p for p in policies if p.name == "random"][0]
        gap = best.accuracy() - rnd.accuracy()
        if best.name == "driven" and gap > 3:
            print(f"\n  THE DRIVE ALLOCATES BETTER THAN CHANCE by "
                  f"{gap:.1f} points. Spending questions")
            print(f"  where it has been most wrong beats spreading them "
                  f"evenly, on a world it")
            print(f"  cannot control. Same mechanism as the grid, first "
                  f"time reaching outside.")
        elif gap < 3:
            print(f"\n  NO POLICY BEAT THE OTHERS ({gap:+.1f}). With this "
                  f"many sources and this much")
            print(f"  time, where you look does not matter. Needs more "
                  f"sources, more hours, or")
            print(f"  sources that differ more in how predictable they "
                  f"are.")

    dead = [s for s, c in world.fails.items() if c]
    if dead:
        print(f"\n  sources that failed during the run: " +
              ", ".join(f"{s} ({world.fails[s]})" for s in dead))

    print(f"\n  Reading a public endpoint is a small thing to call acting "
          f"in the world. What")
    print(f"  makes it the right step is the structure: it left the "
          f"machine, the world")
    print(f"  answered on its own terms, and the answer could not have "
          f"come from inside.")


if __name__ == "__main__":
    main()
