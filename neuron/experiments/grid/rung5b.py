"""Rung 5: a record of a person, at proper scale.

The shell-history attempt failed on data volume: 1,731 commands, most of
them a single day, against weather's 35,000. Nothing beat a bigram lookup,
which is what happens when there is not enough of a life to learn one from.

File modification times are the same kind of record and there are 218,880 of
them, going back years. What was worked on, when, in what order. Real
timestamps, so time of day and day of week come free — which the shell
history did not have.

WHAT IS KEPT PER EVENT: the top-level folder under home, and the file
extension. Nothing else. No filenames, no paths, no contents. `Documents`
and `.pdf`, not which document.

THE TASK: given the recent activity and the time, predict what KIND of thing
gets touched next. Same shape as the weather task — predict the near future
from the recent past.

Baselines, same discipline as every rung:
  majority   always the most common activity type
  repeat     the same as the last event
  bigram     what usually follows the last event. The strong one, and the
             thing that beat everything on the shell history.

Arms: frozen, memoryless, online, guarded. Chronological split, tested on
the most recent slice, which the model never trains on.

The retention question transfers: old projects are old, recent work is
recent, and an unprotected learner should forget what it has moved past.
"""

import json
import os
import random
import subprocess
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from stability import StabilityLayer


CACHE = "activity.json"
SINCE = "2024-01-01"

LAG = 6                   # recent events carried in the input
HIDDEN = 128
LR = 3e-4
SEEDS = [0, 1, 2]
EPISODE = 500
SEQ_LEN = 10
SEQ_COUNT = 2
TEST_FRACTION = 0.2
MIN_COUNT = 200           # activity types rarer than this become "other"
MAX_EVENTS = 60000        # cap, so this runs in minutes not hours

SKIP = ("/Library/", "/.Trash/", "/node_modules/", "/.git/", "/.cache/",
        "/.npm/", "/.venv/", "/site-packages/", "/__pycache__/",
        "/.vscode/", "/.local/")


def collect(force=False):
    """Walk the home directory once, keeping only folder, extension, time.

    Nothing identifying is retained. The cache file contains only those
    three fields per event.
    """
    if os.path.exists(CACHE) and not force:
        with open(CACHE) as f:
            return json.load(f)

    home = os.path.expanduser("~")
    print(f"walking {home} ... (a minute or two)")
    out = subprocess.run(
        ["find", home, "-type", "f", "-newermt", SINCE,
         "-not", "-path", "*/Library/*",
         "-not", "-path", "*/.Trash/*",
         "-not", "-path", "*/node_modules/*",
         "-not", "-path", "*/.git/*",
         "-printf", "%T@ %p\n"],
        capture_output=True, text=True)

    lines = out.stdout.splitlines()
    if not lines:
        # BSD find on macOS has no -printf, so fall back to stat
        print("  using stat fallback ...")
        paths = subprocess.run(
            ["find", home, "-type", "f", "-newermt", SINCE,
             "-not", "-path", "*/Library/*",
             "-not", "-path", "*/.Trash/*",
             "-not", "-path", "*/node_modules/*",
             "-not", "-path", "*/.git/*"],
            capture_output=True, text=True).stdout.splitlines()
        events = []
        for p in paths:
            if any(s in p for s in SKIP):
                continue
            try:
                mt = os.path.getmtime(p)
            except OSError:
                continue
            rel = os.path.relpath(p, home)
            folder = rel.split(os.sep)[0]
            ext = os.path.splitext(p)[1].lower() or "none"
            events.append([mt, folder, ext])
    else:
        events = []
        for line in lines:
            try:
                ts, p = line.split(" ", 1)
                mt = float(ts)
            except ValueError:
                continue
            if any(s in p for s in SKIP):
                continue
            rel = os.path.relpath(p, home)
            folder = rel.split(os.sep)[0]
            ext = os.path.splitext(p)[1].lower() or "none"
            events.append([mt, folder, ext])

    events.sort(key=lambda e: e[0])
    with open(CACHE, "w") as f:
        json.dump(events, f)
    print(f"  cached {len(events)} events to {CACHE}")
    return events


def build(events):
    """Turn events into (context, label) moments.

    An activity type is folder plus extension. The label is the NEXT
    activity type. The input also carries hour of day and day of week,
    which is genuine structure a person has and weather does not.
    """
    if len(events) > MAX_EVENTS:
        events = events[-MAX_EVENTS:]

    types = [f"{f}/{e}" for _, f, e in events]
    counts = Counter(types)
    keep = {t for t, n in counts.items() if n >= MIN_COUNT}
    types = [t if t in keep else "other" for t in types]

    vocab = sorted(set(types))
    idx = {t: i for i, t in enumerate(vocab)}

    rows = []
    for i in range(LAG, len(types)):
        ctx = [idx[types[i - b]] for b in range(LAG, 0, -1)]
        lt = time.localtime(events[i - 1][0])
        rows.append((ctx, lt.tm_hour, lt.tm_wday, idx[types[i]]))
    return rows, vocab, counts


def baselines(train, test, n_vocab):
    counts = [0] * n_vocab
    for r in train:
        counts[r[3]] += 1
    top = counts.index(max(counts))
    majority = 100.0 * sum(1 for r in test if r[3] == top) / len(test)
    repeat = 100.0 * sum(1 for r in test if r[0][-1] == r[3]) / len(test)

    follow = {}
    for r in train:
        d = follow.setdefault(r[0][-1], {})
        d[r[3]] = d.get(r[3], 0) + 1
    best = {p: max(d, key=d.get) for p, d in follow.items()}
    bigram = 100.0 * sum(1 for r in test
                         if best.get(r[0][-1], top) == r[3]) / len(test)
    return dict(majority=majority, repeat=repeat, bigram=bigram)


def encode(row, n_vocab):
    """Recent activity one-hot, plus hour and weekday one-hot."""
    v = torch.zeros(LAG * n_vocab + 24 + 7)
    for j, c in enumerate(row[0]):
        v[j * n_vocab + c] = 1.0
    v[LAG * n_vocab + row[1]] = 1.0
    v[LAG * n_vocab + 24 + row[2]] = 1.0
    return v.unsqueeze(0)


def in_dim(n_vocab):
    return LAG * n_vocab + 24 + 7


class MLP(nn.Module):
    def __init__(self, n_vocab, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(in_dim(n_vocab), HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, n_vocab))

    def forward(self, x, h=None):
        return self.net(x), None


class GRU(nn.Module):
    def __init__(self, n_vocab, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.enc = nn.Linear(in_dim(n_vocab), HIDDEN)
        self.cell = nn.GRUCell(HIDDEN, HIDDEN)
        self.head = nn.Linear(HIDDEN, n_vocab)

    def forward(self, x, h=None):
        e = F.relu(self.enc(x))
        h = self.cell(e, h)
        return self.head(h), h


class Backend:
    def __init__(self, recurrent, n_vocab, seed=0):
        self.n = n_vocab
        self.net = GRU(n_vocab, seed) if recurrent else MLP(n_vocab, seed)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.h = None
        self._saved = None

    def begin_sequence(self):
        self._saved = self.h
        self.h = None

    def _xy(self, row):
        return encode(row, self.n), torch.tensor([row[3]])

    def score(self, row):
        self.net.eval()
        with torch.no_grad():
            x, y = self._xy(row)
            logits, _ = self.net(x, self.h)
            return float(F.cross_entropy(logits, y).item()), None

    def update(self, row, steps):
        self.net.train()
        last = None
        for _ in range(steps):
            x, y = self._xy(row)
            logits, h = self.net(x, self.h)
            loss = F.cross_entropy(logits, y)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.h = h.detach() if h is not None else None
            last = float(loss.item())
        return last

    def reset_state(self):
        self.h = None
        self._saved = None

    def snapshot(self):
        return {k: v.detach().clone()
                for k, v in self.net.state_dict().items()}

    def restore(self, state):
        self.net.load_state_dict(state)

    def on_rollback(self):
        for g in self.opt.param_groups:
            g["lr"] = max(1e-6, g["lr"] * 0.5)


def evaluate(net, test, n_vocab):
    net.eval()
    h = None
    hit = seen = 0
    per = [[0, 0], [0, 0]]
    mid = len(test) // 2
    with torch.no_grad():
        for i, row in enumerate(test):
            if i % EPISODE == 0:
                h = None
            logits, h = net(encode(row, n_vocab), h)
            right = int(logits.argmax(1).item()) == row[3]
            seen += 1
            hit += right
            k = 0 if i < mid else 1
            per[k][0] += right
            per[k][1] += 1
    return (100.0 * hit / seen,
            [100.0 * a / b if b else 0.0 for a, b in per])


def run(arm, train, test, n_vocab, seed):
    b = Backend(arm != "memoryless", n_vocab, seed)
    layer = None
    if arm == "guarded":
        layer = StabilityLayer(
            b, canary=train[:40], seed=seed,
            window=200, warmup=30, top_fraction=0.50,
            coherence_veto=1e9, loss_floor=0.02, steps_per_update=1,
            rehearse_per_item=5, rehearse_count=SEQ_COUNT, rehearse_steps=1,
            anchor_size=100, buffer_size=500, sequence_len=SEQ_LEN,
            replay_policy="uniform",
            guard=True, guard_per_item=1000, canary_tolerance=0.5)

    if arm != "frozen":
        for i, row in enumerate(train):
            if i % EPISODE == 0:
                b.reset_state()
            if layer is not None:
                layer.observe(row)
            else:
                b.update(row, 1)

    acc, halves = evaluate(b.net, test, n_vocab)
    del b, layer
    return acc, halves


ARMS = ["frozen", "memoryless", "online", "guarded"]


if __name__ == "__main__":
    events = collect()
    rows, vocab, counts = build(events)

    split = int(len(rows) * (1 - TEST_FRACTION))
    train, test = rows[:split], rows[split:]

    span_lo = time.strftime("%Y-%m-%d", time.localtime(events[0][0]))
    span_hi = time.strftime("%Y-%m-%d", time.localtime(events[-1][0]))
    print(f"\nRung 5: {len(events)} events, {span_lo} to {span_hi}")
    print(f"{len(vocab)} activity types, "
          f"{len(train)} training moments, {len(test)} test moments")
    print(f"input carries the last {LAG} activities plus hour and weekday\n")

    print("most common activity types:")
    for t, n in counts.most_common(10):
        print(f"  {t[:34]:>34}: {n:>6}")

    b = baselines(train, test, len(vocab))
    best = max(b.values())
    print(f"\nbaselines on the held-out tail:")
    print(f"  majority        {b['majority']:5.2f}%")
    print(f"  repeat last     {b['repeat']:5.2f}%")
    print(f"  bigram          {b['bigram']:5.2f}%")
    print(f"  strongest       {best:5.2f}%\n")

    def mean(xs):
        return sum(xs) / len(xs)

    results = {}
    for arm in ARMS:
        accs, f1, f2 = [], [], []
        for seed in SEEDS:
            a, h = run(arm, train, test, len(vocab), seed)
            accs.append(a)
            f1.append(h[0])
            f2.append(h[1])
        results[arm] = dict(acc=mean(accs), first=mean(f1), second=mean(f2))
        d = mean(accs) - best
        mark = "  BEATS BASELINE" if d > 0 else ""
        print(f"  {arm:>11}: {mean(accs):5.2f}%  ({d:+5.2f}){mark}",
              flush=True)

    print("\n" + "=" * 62)
    print(f"{'arm':>11} {'overall':>9} {'first half':>12} "
          f"{'second half':>12}")
    print("-" * 62)
    print(f"{'bigram':>11} {b['bigram']:>8.2f}%")
    print("-" * 62)
    for arm in ARMS:
        r = results[arm]
        print(f"{arm:>11} {r['acc']:>8.2f}% {r['first']:>11.1f}% "
              f"{r['second']:>11.1f}%")
    print("=" * 62)

    winners = [a for a in ARMS
               if a != "frozen" and results[a]["acc"] > best]
    print()
    if winners:
        print(f"  RUNG 5 CLEARED by {', '.join(winners)}.")
        print("  A network grown from random weights, learning from a record")
        print("  of one person's actual work one event at a time, beat the")
        print("  best simple rule. That is the last rung.")
    else:
        print("  NOT CLEARED. The bigram still captures more structure than")
        print("  the learned model does. On highly repetitive personal data")
        print("  a lookup table is a genuinely strong opponent.")

    print("""
This is a record of a person rather than a world or a planet, at a scale
that can actually support learning.

The two halves show drift within the held-out span. And the bigram is the
real opponent: personal activity is repetitive, so "what usually follows
what" is hard to beat. Beating it means the model found structure in the
timing and the longer context that a lookup table cannot represent.
""")
    with open("rung5b.json", "w") as f:
        json.dump(dict(results=results, baselines=b, vocab=len(vocab),
                       events=len(events)), f, indent=2)
    print("wrote rung5b.json")