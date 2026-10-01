"""It can die now.

WHAT WAS MISSING FROM live2.py, AND FROM EVERYTHING BEFORE IT. The system
perceived a world nobody designed and took actions that left the machine.
Both real. But nothing it did mattered TO IT. It could be wrong forever at
no cost. A run ended when the process was killed, never because the system
failed at anything. That is the difference between something running and
something living, and it has been the gap since the first grid world.

THE ECONOMY, and every number here is a choice I am making rather than a
fact about the world, which is the honest limitation of self-imposed
stakes.

  every sample costs      SAMPLE_COST      merely existing is not free
  a correct prediction    CORRECT_PAYS     being right is what sustains it
  a wrong prediction      nothing
  each consolidation      SLEEP_COST       thinking costs more than living
  credit ceiling          MAX_CREDIT       it cannot hoard its way to safety

Break-even accuracy is SAMPLE_COST / CORRECT_PAYS. At 0.10 and 0.15 that is
67%: a system predicting worse than two in three dies, one predicting
better accumulates. The majority baseline on this task has run 75 to 90%,
so a system that learns nothing but "say PRESENT" survives, and one that is
actively wrong does not. That is the right bar. A world where the dumbest
policy dies is measuring the harness, and one where nothing dies is not
measuring stakes.

WHAT DEATH COSTS. Everything. The weights, the replay buffer, the drives,
whatever it worked out about your Tuesdays. A new life starts from random
weights in the same world, and the generation counter goes up. There is no
snapshot to restore, because a failure you can roll back from is not a
failure.

This is where the parked population work (pop.py, pop2.py, 27 Sep) rejoins
the project. Those failed in a grid world that could not show whether
inheritance helps. Here lives are days long and the world is real, so
generations accumulate slowly and honestly. Whether later generations last
longer than earlier ones is the question, and it will take weeks to answer
rather than minutes.

SLEEP IS NOW A BET IT PAYS FOR. In the 20 Sep grid sweep consolidation was
free and offline beat interleaved by 8.9 points. In lineage.py it cost
energy and the advantage vanished. Here it costs credit in real time, so
the system is spending survival on the hope of predicting better later.
That is the same trade an animal makes every night and the third different
setting this project has asked it in.

HONEST ABOUT THE LIMIT. I chose the costs. Stakes I define are not stakes
the world imposes, and a resource that only exists in this file is a long
way from one anybody else recognises. The next rung needs a transaction
with a system that does not know this is an experiment, and that needs an
account and an adult, which is a conversation rather than a commit.

    python live3.py --interval 30
    python live3.py --status
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
HIDDEN = 48
LR = 3e-4
SEQ_LEN = 8
REHEARSE_PER_ITEM = 6

HORIZON_MIN = 5.0
IDLE_PRESENT = 90.0
WEATHER_EVERY = 900

START_CREDIT = 100.0
MAX_CREDIT = 200.0
SAMPLE_COST = 0.10
CORRECT_PAYS = 0.15
SLEEP_COST = 0.05        # per consolidation round

STATE_DIR = "live3_state"


def idle_seconds():
    """Seconds since the last keyboard or mouse event.

    Reads HIDIdleTime from ioreg. Reports only THAT input happened, never
    what it was. No permissions.

    NOTE, found 28 Sep the expensive way: `caffeinate -i` resets this
    counter continuously, so running under it makes the sensor read
    "present" forever. A 21-hour run was lost to that, with the system
    reporting 99% presence through a night of sleep. Use `caffeinate -s`,
    which prevents system sleep without touching the idle timer. That is a
    class of bug no simulated world can produce: the infrastructure
    keeping the process alive silently destroyed its own sensor.
    """
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
    def __init__(self, lat=26.12, lon=-80.14):
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
            with urllib.request.urlopen(self.url, timeout=8) as r:
                c = json.load(r)["current"]
            self.vals = [
                min(1.0, max(0.0, (c["temperature_2m"] + 10) / 50)),
                min(1.0, c["precipitation"] / 10.0),
                c["cloud_cover"] / 100.0,
                min(1.0, c["wind_speed_10m"] / 60.0)]
        except Exception:
            pass
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


# machine 4, clock 5, weather 4, idle, present, just slept, credit
INPUT = 4 + 5 + 4 + 1 + 1 + 1 + 1


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
    """canary_from_stream is ON. The 3,000,000-step run on 20 Sep showed a
    fixed canary causing 232 rollbacks and accelerating, because a canary
    that knows nothing about the current world reads world-change as
    self-damage. live2 ran 79 hours with this on and fired zero
    rollbacks."""
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
    """One life. It ends when the credit runs out and nothing survives it."""

    def __init__(self, gen, seed):
        self.gen = gen
        self.b = Backend(seed)
        self.layer = make_layer(self.b, seed)
        self.credit = START_CREDIT
        self.born = time.time()
        self.samples = 0
        self.right = 0
        self.sleeps = 0
        self.spent_sleeping = 0.0
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
            generation=life.gen,
            credit=life.credit,
            samples=life.samples,
            age_hours=life.age_hours(),
            accuracy=life.accuracy(),
            majority=life.majority(),
            edge=life.accuracy() - life.majority(),
            sleeps=life.sleeps,
            spent_sleeping=life.spent_sleeping,
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
              f"(started at {START_CREDIT:.0f}, dies at 0)")
        print(f"  this life      {m.get('samples', 0):,} samples over "
              f"{m.get('age_hours', 0):.1f} hours")
        print(f"  accuracy       {m.get('accuracy', 0):.1f}% against "
              f"majority {m.get('majority', 0):.1f}%  "
              f"({m.get('edge', 0):+.1f})")
        print(f"  consolidations {m.get('sleeps', 0)}, costing "
              f"{m.get('spent_sleeping', 0):.1f} credit")
        print(f"  rollbacks      {m.get('rollbacks', 0)}")
        hist = m.get("history", [])
        if hist:
            print(f"\n  LIVES THAT ENDED")
            print(f"  {'gen':>4} {'hours':>8} {'samples':>9} "
                  f"{'accuracy':>9} {'edge':>7}")
            for h in hist:
                print(f"  {h['gen']:>4} {h['hours']:>8.1f} "
                      f"{h['samples']:>9,} {h['accuracy']:>8.1f}% "
                      f"{h['edge']:>+7.1f}")
        print(f"\n  last seen      {m.get('last_seen', '?')}")
        return

    if idle_seconds() is None:
        print("  cannot read idle time on this system, and the whole")
        print("  prediction is about presence. Not starting.")
        return

    weather = Weather(args.lat, args.lon)
    horizon = args.horizon * 60.0

    history = []
    mp = os.path.join(args.state, "meta.json")
    if os.path.exists(mp):
        with open(mp) as f:
            history = json.load(f).get("history", [])

    gen = (history[-1]["gen"] + 1) if history else 0
    life = Life(gen, seed=gen)

    print(f"  predicting whether you will be at the keyboard in "
          f"{args.horizon:.0f} minutes")
    print(f"  IT CAN DIE. Each sample costs {SAMPLE_COST}, each correct "
          f"prediction pays {CORRECT_PAYS},")
    print(f"  each consolidation round costs {SLEEP_COST}. Break-even "
          f"accuracy is "
          f"{100 * SAMPLE_COST / CORRECT_PAYS:.0f}%.")
    print(f"  At zero credit the life ends and nothing survives it.")
    print(f"  generation {gen}, starting with {START_CREDIT:.0f} credit\n")
    print(f"  {'gen':>4} {'credit':>8} {'acc':>7} {'majority':>9} "
          f"{'edge':>7} {'here':>6} {'sleeps':>7} {'age':>8}")
    print("-" * 68)

    just_slept = 0.0
    started = time.time()

    try:
        while True:
            now = time.time()
            idle = idle_seconds()
            if idle is None:
                idle = 999.0
            here = 1.0 if idle < IDLE_PRESENT else 0.0

            feats = (machine() + clock() + weather.read() +
                     [min(1.0, idle / 600.0), here, just_slept,
                      min(1.0, life.credit / MAX_CREDIT)])
            just_slept = 0.0

            guess, p_present = life.b.predict(feats)
            life.pending.append((feats, now + horizon, guess))

            while life.pending and life.pending[0][1] <= now:
                old_feats, _, old_guess = life.pending.popleft()
                label = PRESENT if here else AWAY

                # THE ECONOMY. Existing costs; being right pays. This is
                # the only place in the project where a prediction has a
                # consequence for the thing making it.
                life.credit -= SAMPLE_COST
                if old_guess == label:
                    life.credit += CORRECT_PAYS
                    life.right += 1
                life.credit = min(life.credit, MAX_CREDIT)

                life.correct.append(1 if old_guess == label else 0)
                life.labels.append(label)
                life.layer.observe((old_feats, label))
                life.samples += 1

                if life.samples % args.report_every == 0:
                    sm = life.layer.summary()
                    print(f"  {life.gen:>4} {life.credit:>8.1f} "
                          f"{life.accuracy():>6.1f}% "
                          f"{life.majority():>8.1f}% "
                          f"{life.accuracy() - life.majority():>+6.1f} "
                          f"{100 * statistics.mean(life.labels):>5.0f}% "
                          f"{life.sleeps:>7} "
                          f"{life.age_hours():>7.1f}h", flush=True)
                    save(args.state, life, history)

                if life.credit <= 0:
                    # DEATH. Nothing is saved, nothing is restored. A
                    # failure you can roll back from is not a failure.
                    rec = dict(gen=life.gen, hours=life.age_hours(),
                               samples=life.samples,
                               accuracy=life.accuracy(),
                               edge=life.accuracy() - life.majority(),
                               sleeps=life.sleeps,
                               died=time.strftime("%Y-%m-%d %H:%M:%S"))
                    history.append(rec)
                    print(f"\n  GENERATION {life.gen} DIED after "
                          f"{life.age_hours():.1f} hours and "
                          f"{life.samples:,} samples")
                    print(f"  accuracy {life.accuracy():.1f}% against "
                          f"{life.majority():.1f}%, "
                          f"{life.sleeps} consolidations costing "
                          f"{life.spent_sleeping:.1f} credit")
                    print(f"  everything it learned is gone. starting "
                          f"generation {life.gen + 1}\n", flush=True)
                    life = Life(life.gen + 1, seed=life.gen + 1)
                    save(args.state, life, history)
                    break

            # SLEEP IS A BET IT PAYS FOR. Consolidating when it believes
            # nobody is here, spending credit now against predicting
            # better later. It won on accuracy when free (20 Sep) and the
            # advantage vanished when charged (lineage.py). This is the
            # third setting, and the first in real time.
            cost = args.rounds * SLEEP_COST
            if p_present < 0.35 and here == 0.0 and life.credit > cost + 5:
                life.layer.sleep(args.rounds)
                life.credit -= cost
                life.spent_sleeping += cost
                life.sleeps += 1
                just_slept = 1.0

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n  stopped")

    save(args.state, life, history)
    print(f"\n  generation {life.gen}, credit {life.credit:.1f}, "
          f"{life.samples:,} samples, {len(history)} deaths so far")
    print(f"  state in {os.path.abspath(args.state)}")


if __name__ == "__main__":
    main()
