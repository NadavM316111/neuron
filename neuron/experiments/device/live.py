"""The first body. It watches this machine and its choices change what it sees.

WHAT EVERY OTHER EXPERIMENT IN THIS REPO HAS IN COMMON. The world was
designed. Grid worlds, invented vocabularies, synthetic contradictions:
every rule was written by hand, which means every result is partly a result
about the person who wrote the rule. Even the weather run (rung 4) was a
fixed file replayed after the fact.

This machine was not designed for this. Its load rises because something
else opened a tab. Its battery falls at a rate nobody chose. There are
patterns in it that nobody knows, including whoever owns it, and there is
no way to have tuned anything to them in advance.

THE LOOP IS ACTUALLY CLOSED, and that is the point rather than a detail.
The system decides WHEN to consolidate. Consolidating costs CPU.
CPU load is part of what it observes. So its own choices change its own
future input, permanently, in a way it can neither undo nor rerun. Nothing
else in this repo has that: in the grid worlds the agent acted, but the
episode reset; here there is no reset, because it is Tuesday afternoon
exactly once.

WHAT IT PREDICTS. Given the current reading of the machine, what happens to
CPU load on the next sample: down, roughly the same, or up. Three classes,
the same shape as the grid-world outcome prediction, so the backend and the
layer are unchanged and anything that differs is the world rather than the
machinery.

WHAT IT DECIDES. Whether to consolidate now.

  fixed    consolidate every N samples regardless. The control, and what
           neuron_system.py does today.
  quiet    consolidate when the machine has recently been PREDICTABLE.
           The bet is that a boring stretch is a cheap time to spend CPU,
           because little is being missed, and that a surprising stretch
           should be watched rather than slept through. That is roughly
           what an animal does and it is a real decision with a real cost.

HONEST ABOUT THE ACTION. Consolidating raises load by a small amount for a
short time. The effect on the next observation is real but weak, so this is
a closed loop rather than a strong one. A stronger action would mean doing
something to the machine that a person would notice, which is not a thing
to build before the mechanism is understood.

WHY REAL TIME MATTERS HERE AND CANNOT BE COMPRESSED. The grid ran 3,000,000
steps in 16 minutes because the world was a function call. This world
arrives at one sample per interval and no faster. --interval 0.5 gives a
usable stream in 20 minutes, which proves the loop works. Left at 5 or 10
seconds for days it accumulates something a fast run cannot fake: the
difference between morning and night, between plugged in and not, between
Tuesday and Saturday. That is the part the vision actually needs.

PERSISTENCE. Weights, buffer and counters are written to disk every
--save-every samples, so this survives being killed, closing the laptop, or
a restart. Resuming is the default; --fresh starts over.

    python live.py --interval 0.5 --minutes 20       # prove the loop
    python live.py --interval 5 &                    # leave it living
    python live.py --status                          # what has it learned
"""

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "neuron", "core"))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))

from sleeping import SleepLayer          # noqa: E402


DOWN, SAME, UP = 0, 1, 2
CLASSES = 3
HIDDEN = 48
LR = 3e-4
SEQ_LEN = 8
REHEARSE_PER_ITEM = 6

# What counts as a change rather than noise. CPU percent wanders by a point
# or two constantly, so a threshold is needed or every sample is "up" or
# "down" at random and the task is unlearnable.
CHANGE_THRESHOLD = 3.0

# The quiet policy's window and its idea of boring.
QUIET_WINDOW = 40
QUIET_FRACTION = 0.40      # consolidate when surprise is in the calmest 40%

STATE_DIR = "live_state"


# ---------------------------------------------------------------- sensors

def make_sensor():
    """Read the machine. psutil if it is there, stdlib if not.

    Returns (reader, names). The reader gives a list of floats already
    scaled to roughly 0..1, because an unnormalised feature that ranges
    over thousands dominates a small network's input layer and the others
    stop mattering.
    """
    try:
        import psutil
    except ImportError:
        psutil = None

    if psutil is not None:
        psutil.cpu_percent(interval=None)     # prime the counter

        def read():
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory().percent
            try:
                b = psutil.sensors_battery()
                batt = (b.percent / 100.0) if b else 0.5
                plugged = 1.0 if (b and b.power_plugged) else 0.0
            except Exception:
                batt, plugged = 0.5, 0.0
            try:
                io = psutil.disk_io_counters()
                disk = min(1.0, (io.read_bytes + io.write_bytes) / 1e11)
            except Exception:
                disk = 0.0
            try:
                n = psutil.net_io_counters()
                net = min(1.0, (n.bytes_sent + n.bytes_recv) / 1e11)
            except Exception:
                net = 0.0
            load = min(1.0, os.getloadavg()[0] / 8.0)
            return [cpu / 100.0, mem / 100.0, batt, plugged, disk, net,
                    load], cpu
        names = ["cpu", "mem", "battery", "plugged", "disk", "net", "load"]
        return read, names

    print("  psutil not installed, falling back to load average only.")
    print("  pip install psutil gives battery, memory, disk and network.")

    def read():
        l1, l5, l15 = os.getloadavg()
        cpu = min(100.0, l1 * 12.5)
        return [min(1.0, l1 / 8.0), min(1.0, l5 / 8.0),
                min(1.0, l15 / 8.0), 0.0, 0.0, 0.0,
                min(1.0, l1 / 8.0)], cpu
    return read, ["load1", "load5", "load15", "-", "-", "-", "load"]


def clock_features(now=None):
    """Time of day and day of week as circles, not as numbers.

    Hour 23 and hour 0 are adjacent, and a raw 0..23 feature tells the
    network the opposite. Sine and cosine make the adjacency true, which
    matters because most of what a machine does is on a daily cycle and
    that cycle is the thing this system is here to learn.
    """
    t = time.localtime(now if now is not None else time.time())
    h = (t.tm_hour + t.tm_min / 60.0) / 24.0
    d = t.tm_wday / 7.0
    return [math.sin(2 * math.pi * h), math.cos(2 * math.pi * h),
            math.sin(2 * math.pi * d), math.cos(2 * math.pi * d)]


def bucket(delta):
    if delta < -CHANGE_THRESHOLD:
        return DOWN
    if delta > CHANGE_THRESHOLD:
        return UP
    return SAME


# ---------------------------------------------------------------- model

INPUT = 7 + 4 + 1          # sensors, clock, and "did I just consolidate"


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
    """Same five methods the layer wants, so nothing else changes."""

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
            logits, _ = self.net(x, None)
            return int(logits.argmax(1).item())


def make_canary(seed):
    """Fixed readings the model should always handle.

    Built as plausible machine states rather than sampled from this
    machine's history, so the guard measures general competence at the
    task and not competence at whatever the last hour happened to look
    like.

    canary_from_stream is ON. The 3,000,000-step grid run on 20 Sep showed
    the guard firing 232 times and accelerating, because a canary that
    knows nothing about the current world reads world-change as
    self-damage. stability.py has carried the fix since 10 Sep and nothing
    has ever used it. A machine's behaviour changes constantly, so this is
    exactly the setting that breaks a fixed canary.
    """
    rng = random.Random(4000 + seed)
    out = []
    for _ in range(10):
        feats = [rng.random() for _ in range(7)] + \
            clock_features(rng.random() * 86400 * 7) + [0.0]
        out.append((feats, rng.choice([DOWN, SAME, UP])))
    return out


# ---------------------------------------------------------------- living

def save(path, b, layer, meta):
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
    ap.add_argument("--interval", type=float, default=5.0,
                    help="seconds between samples")
    ap.add_argument("--minutes", type=float, default=0,
                    help="stop after this long; 0 means run until killed")
    ap.add_argument("--policy", choices=["fixed", "quiet"],
                    default="quiet")
    ap.add_argument("--consolidate-every", type=int, default=60,
                    help="samples between consolidations for 'fixed', and "
                         "the average rate 'quiet' is held to")
    ap.add_argument("--rounds", type=int, default=10,
                    help="consolidation rounds per sleep")
    ap.add_argument("--state", default=STATE_DIR)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--report-every", type=int, default=60)
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
        print(f"  alive for      {m.get('hours', 0):.1f} hours of real "
              f"time")
        print(f"  consolidations {m.get('sleeps', 0)}")
        print(f"  rollbacks      {m.get('rollbacks', 0)}")
        print(f"  last accuracy  {m.get('accuracy', 0):.1f}% against "
              f"majority {m.get('majority', 0):.1f}%")
        print(f"  last seen      {m.get('last_seen', '?')}")
        return

    read, names = make_sensor()
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

    print(f"  sensors: {', '.join(n for n in names if n != '-')}")
    print(f"  every {args.interval}s, policy '{args.policy}', "
          f"{args.rounds} rounds per consolidation")
    if samples:
        print(f"  RESUMING a life of {samples:,} samples over "
              f"{hours:.1f} hours")
    else:
        print(f"  starting a new life in {args.state}")
    print(f"  predicting the next change in CPU load: down / same / up")
    print(f"  consolidating costs CPU, which it then observes\n")
    print(f"  {'sample':>8} {'acc':>7} {'majority':>9} {'surprise':>9} "
          f"{'sleeps':>7} {'rb':>4} {'elapsed':>9}")
    print("-" * 64)

    recent = deque(maxlen=QUIET_WINDOW)
    correct = deque(maxlen=args.report_every)
    labels = deque(maxlen=args.report_every)
    since = 0
    just_slept = 0.0
    prev_feats = None
    prev_cpu = None

    started = time.time()
    stop = started + args.minutes * 60 if args.minutes else None

    try:
        while True:
            if stop and time.time() > stop:
                break

            sensors, cpu = read()
            feats = sensors + clock_features() + [just_slept]
            just_slept = 0.0

            if prev_feats is not None:
                label = bucket(cpu - prev_cpu)
                guess = b.predict(prev_feats)
                correct.append(1 if guess == label else 0)
                labels.append(label)

                item = (prev_feats, label)
                s, _ = b.score(item)
                recent.append(s)
                layer.observe(item)
                samples += 1
                since += 1

                # THE DECISION. fixed ignores the world; quiet spends CPU
                # when the world has been predictable, on the bet that a
                # calm stretch is a cheap time to sleep.
                do_sleep = False
                if args.policy == "fixed":
                    do_sleep = since >= args.consolidate_every
                else:
                    if since >= args.consolidate_every // 3 and \
                            len(recent) >= QUIET_WINDOW:
                        cut = sorted(recent)[
                            int(len(recent) * QUIET_FRACTION)]
                        do_sleep = s <= cut
                    if since >= args.consolidate_every * 3:
                        do_sleep = True     # never go forever without it

                if do_sleep:
                    layer.sleep(args.rounds)
                    sleeps += 1
                    since = 0
                    just_slept = 1.0

                if samples % args.report_every == 0:
                    acc = 100.0 * sum(correct) / max(1, len(correct))
                    maj = 100.0 * max(
                        labels.count(c) for c in (DOWN, SAME, UP)
                    ) / max(1, len(labels))
                    el = time.time() - started
                    sm = layer.summary()
                    print(f"  {samples:>8,} {acc:>6.1f}% {maj:>8.1f}% "
                          f"{statistics.mean(recent):>9.3f} "
                          f"{sleeps:>7} {sm.get('rollbacks', 0):>4} "
                          f"{el / 60:>8.1f}m", flush=True)

                if samples % args.save_every == 0:
                    sm = layer.summary()
                    save(args.state, b, layer, dict(
                        samples=samples,
                        hours=hours + (time.time() - started) / 3600,
                        sleeps=sleeps,
                        rollbacks=sm.get("rollbacks", 0),
                        accuracy=100.0 * sum(correct) / max(1, len(correct)),
                        majority=100.0 * max(
                            labels.count(c) for c in (DOWN, SAME, UP)
                        ) / max(1, len(labels)),
                        last_seen=time.strftime("%Y-%m-%d %H:%M:%S")))

            prev_feats, prev_cpu = feats, cpu
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n  stopped")

    sm = layer.summary()
    acc = 100.0 * sum(correct) / max(1, len(correct))
    maj = 100.0 * max(labels.count(c) for c in (DOWN, SAME, UP)) / \
        max(1, len(labels))
    save(args.state, b, layer, dict(
        samples=samples, hours=hours + (time.time() - started) / 3600,
        sleeps=sleeps, rollbacks=sm.get("rollbacks", 0),
        accuracy=acc, majority=maj,
        last_seen=time.strftime("%Y-%m-%d %H:%M:%S")))

    print("\n" + "=" * 64)
    print(f"  {samples:,} samples, {sleeps} consolidations, "
          f"{sm.get('rollbacks', 0)} rollbacks")
    print(f"  accuracy {acc:.1f}% against a majority baseline of "
          f"{maj:.1f}%")
    if acc <= maj + 2:
        print(f"\n  NOT LEARNING YET. Always guessing the commonest class "
              f"would do as well.")
        print(f"  On a short run that is expected: most of what a machine "
              f"does is on a daily")
        print(f"  cycle and twenty minutes contains none of it. Leave it "
              f"running for a day")
        print(f"  before reading anything into this.")
    else:
        print(f"\n  It is beating the majority baseline by "
              f"{acc - maj:.1f} points on a world")
        print(f"  nobody designed.")
    print(f"\n  state in {os.path.abspath(args.state)}, resumes by "
          f"default")


if __name__ == "__main__":
    main()
