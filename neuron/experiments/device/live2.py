"""It stops modelling the machine and starts modelling the person.

WHAT live.py DID AND WHY IT RAN OUT. It watched CPU, memory, battery and
network and predicted the next change in CPU load. That closed the loop for
the first time: consolidating costs CPU, CPU is what it observes, so its
choices changed its own input. But after 1,200 samples the majority class
hit 100%. A laptop sitting still does not move three CPU points in ten
seconds, so every sample was SAME, and a model that always says SAME was
exactly right. The task had no variance left to predict.

Lowering the threshold would have rescued the task. It would not have
changed what the task was ABOUT, which is the machine's own substrate.

WHAT THIS PREDICTS INSTEAD. Whether the person will be at the keyboard in
HORIZON minutes.

That is a fact about a life rather than about a process. It varies on the
scale of minutes and hours, it has real structure — mornings differ from
evenings, weekdays from Sundays, plugged-in differs from on-battery — and
none of that structure was designed by anyone. It cannot be learned from a
pretrained model, because it is about one specific person, and the only way
to know it is to have been there.

IDLE TIME IS THE SENSOR, and it needs no permissions. macOS exposes
HIDIdleTime through ioreg: seconds since the last keyboard or mouse event.
Nothing is recorded about WHAT was typed or clicked, only that something
happened. No accessibility permission, no screen recording, no content.

WEATHER IS THE SECOND WORLD. Open-Meteo needs no key. It is included
because it is genuinely outside: the system cannot affect it, cannot
predict it from its own state, and it plausibly relates to whether someone
is at a desk. If it turns out to carry no signal that is a finding about
this life rather than a bug.

THE ACTION IS NOW MEANINGFUL. live.py consolidated when recent surprise was
low. Here it consolidates when it predicts the person is AWAY, which is a
bet: spend CPU while nobody is using the machine. That is what an animal
does with the quiet hours and it is the first action in this project that
is about something other than the agent itself. It still costs CPU, and CPU
is still observed, so the loop stays closed.

THE BASELINE IS PRINTED AND IT MATTERS. Predicting "present" all the time
will be right most of the time for most people. The majority column is what
the model has to beat, and until it does, it has learned nothing. live.py's
whole life was spent tied to that baseline, which is why it is now printed
next to every number rather than at the end.

WHAT WOULD MAKE THIS INTERESTING. Beating the majority baseline by a few
points on a held-out stretch would mean the system had learned something
about a particular person's rhythm that no model ships with. That is small
and it is real, and it is the smallest honest version of the thing this
project is for.

    python live2.py --interval 30              # leave it living
    python live2.py --status
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

HORIZON_MIN = 5.0        # predict presence this far ahead
IDLE_PRESENT = 90.0      # idle under this many seconds counts as present
WEATHER_EVERY = 900      # seconds between weather fetches

STATE_DIR = "live2_state"


# ---------------------------------------------------------------- sensors

def idle_seconds():
    """Seconds since the last keyboard or mouse event, or None.

    Reads HIDIdleTime out of ioreg. It reports only THAT input happened,
    never what the input was, and needs no permission of any kind. On
    anything that is not macOS this returns None and the system falls back
    to treating the machine's own activity as a weak proxy.
    """
    try:
        out = subprocess.run(
            ["ioreg", "-c", "IOHIDSystem"], capture_output=True,
            text=True, timeout=5).stdout
        for line in out.splitlines():
            if "HIDIdleTime" in line:
                return int(line.split("=")[-1].strip()) / 1e9
    except Exception:
        pass
    return None


class Weather:
    """Open-Meteo, cached. No key, and failure is not fatal.

    A world the system cannot touch, cannot predict from its own state,
    and did not have designed for it.
    """

    def __init__(self, lat=26.12, lon=-80.14):
        self.url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}"
                    f"&longitude={lon}&current=temperature_2m,"
                    f"precipitation,cloud_cover,wind_speed_10m")
        self.at = 0.0
        self.vals = [0.5, 0.0, 0.5, 0.0]
        self.ok = False

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
            self.ok = True
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
    """Time as circles. Hour 23 and hour 0 are adjacent and a raw number
    says the opposite, which matters here because the whole signal is a
    daily and weekly rhythm."""
    t = time.localtime()
    h = (t.tm_hour + t.tm_min / 60.0) / 24.0
    d = t.tm_wday / 7.0
    weekend = 1.0 if t.tm_wday >= 5 else 0.0
    return [math.sin(2 * math.pi * h), math.cos(2 * math.pi * h),
            math.sin(2 * math.pi * d), math.cos(2 * math.pi * d), weekend]


# machine 4, clock 5, weather 4, idle now 1, present now 1, just slept 1
INPUT = 4 + 5 + 4 + 1 + 1 + 1


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
        feats, label = item
        return (torch.tensor(feats, dtype=torch.float32).unsqueeze(0),
                torch.tensor([label]))

    def score(self, item):
        self.net.eval()
        with torch.no_grad():
            x, y = self._xy(item)
            logits, _ = self.net(x, None)
            return float(F.cross_entropy(logits, y).item()), None

    def update(self, item, steps):
        self.net.train()
        last = None
        for _ in range(steps):
            x, y = self._xy(item)
            logits, _ = self.net(x, None)
            loss = F.cross_entropy(logits, y)
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


def make_canary(seed):
    """Plausible situations the model should always handle.

    canary_from_stream is ON as well. The 3,000,000-step run on 20 Sep
    showed a fixed canary causing 232 rollbacks and accelerating, because
    a canary that knows nothing about the current world reads world-change
    as self-damage. A person's rhythm changes constantly, so this is
    exactly the setting that breaks a fixed one.
    """
    rng = random.Random(9000 + seed)
    return [([rng.random() for _ in range(INPUT)],
             rng.choice([AWAY, PRESENT])) for _ in range(10)]


def save(path, b, meta):
    os.makedirs(path, exist_ok=True)
    torch.save(b.snapshot(), os.path.join(path, "weights.pt"))
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def load(path, b):
    wp = os.path.join(path, "weights.pt")
    mp = os.path.join(path, "meta.json")
    meta = {}
    if os.path.exists(wp):
        b.restore(torch.load(wp, weights_only=False))
    if os.path.exists(mp):
        with open(mp) as f:
            meta = json.load(f)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--horizon", type=float, default=HORIZON_MIN,
                    help="minutes ahead to predict presence")
    ap.add_argument("--minutes", type=float, default=0)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--state", default=STATE_DIR)
    ap.add_argument("--save-every", type=int, default=40)
    ap.add_argument("--report-every", type=int, default=40)
    ap.add_argument("--lat", type=float, default=26.12)
    ap.add_argument("--lon", type=float, default=-80.14)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.status:
        mp = os.path.join(args.state, "meta.json")
        if not os.path.exists(mp):
            print("  nothing lived here yet")
            return
        with open(mp) as f:
            m = json.load(f)
        print(f"  samples        {m.get('samples', 0):,}")
        print(f"  alive for      {m.get('hours', 0):.1f} hours")
        print(f"  consolidations {m.get('sleeps', 0)}")
        print(f"  rollbacks      {m.get('rollbacks', 0)}")
        print(f"  accuracy       {m.get('accuracy', 0):.1f}% against "
              f"majority {m.get('majority', 0):.1f}%  "
              f"({m.get('edge', 0):+.1f})")
        print(f"  you were here  {m.get('present_share', 0):.0f}% of "
              f"samples")
        print(f"  last seen      {m.get('last_seen', '?')}")
        return

    if idle_seconds() is None:
        print("  WARNING: cannot read idle time on this system. The whole")
        print("  prediction is about presence, so without it this is not")
        print("  measuring anything. Not starting.")
        return

    weather = Weather(args.lat, args.lon)
    b = Backend(0)
    layer = SleepLayer(
        b, canary=make_canary(0), seed=0, contiguous=True,
        window=200, warmup=30, top_fraction=1.00,
        coherence_veto=1e9, loss_floor=0.02, steps_per_update=1,
        rehearse_per_item=REHEARSE_PER_ITEM, rehearse_count=2,
        rehearse_steps=1, anchor_size=100, buffer_size=500,
        sequence_len=SEQ_LEN, replay_policy="uniform",
        canary_from_stream=12,
        guard=True, guard_per_item=500, canary_tolerance=0.5)

    meta = {} if args.fresh else load(args.state, b)
    samples = meta.get("samples", 0)
    hours = meta.get("hours", 0.0)
    sleeps = meta.get("sleeps", 0)

    horizon = args.horizon * 60.0
    lag = max(1, int(round(horizon / args.interval)))

    print(f"  predicting whether you will be at the keyboard in "
          f"{args.horizon:.0f} minutes")
    print(f"  sampling every {args.interval:.0f}s, so the answer arrives "
          f"{lag} samples later")
    print(f"  sensors: idle time, machine, clock, weather "
          f"({args.lat:.2f}, {args.lon:.2f})")
    print(f"  it consolidates when it thinks you are AWAY, and that costs "
          f"CPU it then sees")
    if samples:
        print(f"  RESUMING a life of {samples:,} samples over "
              f"{hours:.1f} hours")
    else:
        print(f"  starting a new life in {args.state}")
    print(f"\n  {'sample':>8} {'acc':>7} {'majority':>9} {'edge':>7} "
          f"{'here':>6} {'sleeps':>7} {'rb':>4} {'elapsed':>9}")
    print("-" * 68)

    pending = deque()          # (features, due_time)
    correct = deque(maxlen=args.report_every * 3)
    labels = deque(maxlen=args.report_every * 3)
    presence = deque(maxlen=args.report_every * 3)
    just_slept = 0.0

    started = time.time()
    stop = started + args.minutes * 60 if args.minutes else None

    try:
        while True:
            if stop and time.time() > stop:
                break
            now = time.time()

            idle = idle_seconds()
            if idle is None:
                idle = 999.0
            here = 1.0 if idle < IDLE_PRESENT else 0.0
            presence.append(here)

            feats = (machine() + clock() + weather.read() +
                     [min(1.0, idle / 600.0), here, just_slept])
            just_slept = 0.0

            guess, p_present = b.predict(feats)
            pending.append((feats, now + horizon, guess))

            # Anything whose horizon has arrived can now be scored and
            # learned from. Nothing is learned before the answer exists,
            # which is the difference between predicting and describing.
            while pending and pending[0][1] <= now:
                old_feats, _, old_guess = pending.popleft()
                label = PRESENT if here else AWAY
                correct.append(1 if old_guess == label else 0)
                labels.append(label)

                item = (old_feats, label)
                layer.observe(item)
                samples += 1

                if samples % args.report_every == 0:
                    acc = 100.0 * sum(correct) / max(1, len(correct))
                    maj = 100.0 * max(labels.count(PRESENT),
                                      labels.count(AWAY)) / max(1, len(labels))
                    sm = layer.summary()
                    print(f"  {samples:>8,} {acc:>6.1f}% {maj:>8.1f}% "
                          f"{acc - maj:>+6.1f} "
                          f"{100 * statistics.mean(presence):>5.0f}% "
                          f"{sleeps:>7} {sm.get('rollbacks', 0):>4} "
                          f"{(time.time() - started) / 60:>8.1f}m",
                          flush=True)

                if samples % args.save_every == 0:
                    acc = 100.0 * sum(correct) / max(1, len(correct))
                    maj = 100.0 * max(labels.count(PRESENT),
                                      labels.count(AWAY)) / max(1, len(labels))
                    sm = layer.summary()
                    save(args.state, b, dict(
                        samples=samples,
                        hours=hours + (time.time() - started) / 3600,
                        sleeps=sleeps,
                        rollbacks=sm.get("rollbacks", 0),
                        accuracy=acc, majority=maj, edge=acc - maj,
                        present_share=100 * statistics.mean(presence),
                        last_seen=time.strftime("%Y-%m-%d %H:%M:%S")))

            # THE ACTION. Consolidate when it believes nobody is here.
            # A bet: spend CPU during the quiet hours. It costs, and the
            # cost lands in the machine readings it takes next.
            if p_present < 0.35 and here == 0.0:
                layer.sleep(args.rounds)
                sleeps += 1
                just_slept = 1.0

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n  stopped")

    acc = 100.0 * sum(correct) / max(1, len(correct))
    maj = 100.0 * max(labels.count(PRESENT), labels.count(AWAY)) / \
        max(1, len(labels))
    sm = layer.summary()
    save(args.state, b, dict(
        samples=samples, hours=hours + (time.time() - started) / 3600,
        sleeps=sleeps, rollbacks=sm.get("rollbacks", 0),
        accuracy=acc, majority=maj, edge=acc - maj,
        present_share=100 * statistics.mean(presence) if presence else 0,
        last_seen=time.strftime("%Y-%m-%d %H:%M:%S")))

    print("\n" + "=" * 68)
    print(f"  {samples:,} scored predictions, {sleeps} consolidations, "
          f"{sm.get('rollbacks', 0)} rollbacks")
    print(f"  accuracy {acc:.1f}% against a majority baseline of "
          f"{maj:.1f}%  ({acc - maj:+.1f})")
    if acc <= maj + 2:
        print(f"\n  NOT LEARNING YET. Guessing the commoner answer would "
              f"do as well. A rhythm")
        print(f"  is a daily and weekly thing, and a short run contains "
              f"none of one. This needs")
        print(f"  days before the number means anything.")
    else:
        print(f"\n  It knows something about YOUR rhythm that guessing "
              f"does not. Small, and not")
        print(f"  in any pretrained model, because it is about one "
              f"person.")
    print(f"\n  state in {os.path.abspath(args.state)}, resumes by "
          f"default")


if __name__ == "__main__":
    main()
