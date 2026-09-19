"""Rung 5, properly: can it learn a person from data collected on purpose?

Three earlier attempts failed on data. Shell history was 1,731 commands,
mostly one day. File modification times turned out to be a record of package
managers, not of a person — rustup unpacking twenty thousand HTML files in a
burst. Browsing history was closest but came six points short of a bigram.

So a logger was started and left running: the frontmost application name and
a timestamp, sampled every ten seconds. Nothing else — no window titles, no
URLs, no keystrokes, no screen contents. Seven thousand eight hundred
samples across twelve apps.

THE TASK: given recent activity and the time, predict which application will
be in focus FIVE MINUTES from now. Five minutes rather than ten seconds,
because at ten seconds the answer is almost always "the same app" and
predicting that is worthless.

THE HONEST PROBLEM WITH THIS DATA, stated up front: ten-second samples are
heavily correlated. Sitting in one app for an hour produces 360 near-identical
rows. So the effective sample size is much smaller than 7,843, and the real
information is in the SWITCHES. The script reports how many there are before
training, because if switches are rare the task is unlearnable regardless of
method.

Baselines, same discipline as every rung:
  majority     always the most common app
  persistence  whatever app is in focus now, will still be in focus. On this
               data this is strong and it is the number to beat.
  bigram       what usually follows the current app

Arms: frozen, memoryless, online, guarded. Chronological split — trained on
the earlier portion, tested on the most recent, which the model never sees.
"""

import json
import math
import os
import time
from collections import Counter
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from stability import StabilityLayer


LOG = os.path.expanduser("~/neuron/activity_log.tsv")
HORIZON = 30            # samples ahead to predict: 30 x 10s = 5 minutes
LAG = 12                # recent samples in the input = 2 minutes of context
HIDDEN = 96
LR = 3e-4
SEEDS = [0, 1, 2]
EPISODE = 360           # an hour of samples, for hidden-state resets
TEST_FRACTION = 0.25
MIN_COUNT = 20          # apps rarer than this become "other"
SEQ_LEN = 10
SEQ_COUNT = 2


def load():
    """Read the log. Only the app name and the timestamp are in it."""
    rows = []
    with open(LOG) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            try:
                t = datetime.fromisoformat(parts[0])
            except ValueError:
                continue
            rows.append((t, parts[1]))
    rows.sort(key=lambda r: r[0])
    return rows


def build(rows):
    """Context of recent apps plus hour and weekday, predicting the app
    HORIZON samples ahead."""
    counts = Counter(a for _, a in rows)
    keep = {a for a, n in counts.items() if n >= MIN_COUNT}
    apps = [a if a in keep else "other" for _, a in rows]
    times = [t for t, _ in rows]

    vocab = sorted(set(apps))
    idx = {a: i for i, a in enumerate(vocab)}

    out = []
    for i in range(LAG, len(apps) - HORIZON):
        ctx = [idx[apps[i - b]] for b in range(LAG, 0, -1)]
        t = times[i]
        out.append((ctx, t.hour, t.weekday(), idx[apps[i + HORIZON]]))
    return out, vocab, counts


def switch_rate(rows):
    """How often the app actually changes. This is where the information
    is; long identical stretches carry almost none."""
    n = sum(1 for i in range(1, len(rows)) if rows[i][1] != rows[i - 1][1])
    return n, 100.0 * n / max(1, len(rows) - 1)


def baselines(train, test, n_vocab):
    counts = [0] * n_vocab
    for r in train:
        counts[r[3]] += 1
    top = counts.index(max(counts))
    majority = 100.0 * sum(1 for r in test if r[3] == top) / len(test)

    # persistence: the app in focus now is the app in focus later
    persist = 100.0 * sum(1 for r in test if r[0][-1] == r[3]) / len(test)

    follow = {}
    for r in train:
        d = follow.setdefault(r[0][-1], {})
        d[r[3]] = d.get(r[3], 0) + 1
    best = {p: max(d, key=d.get) for p, d in follow.items()}
    bigram = 100.0 * sum(1 for r in test
                         if best.get(r[0][-1], top) == r[3]) / len(test)
    return dict(majority=majority, persistence=persist, bigram=bigram)


def in_dim(n):
    return LAG * n + 24 + 7


def encode(row, n):
    v = torch.zeros(in_dim(n))
    for j, c in enumerate(row[0]):
        v[j * n + c] = 1.0
    v[LAG * n + row[1]] = 1.0
    v[LAG * n + 24 + row[2]] = 1.0
    return v.unsqueeze(0)


class MLP(nn.Module):
    def __init__(self, n, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(in_dim(n), HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, n))

    def forward(self, x, h=None):
        return self.net(x), None


class GRU(nn.Module):
    def __init__(self, n, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.enc = nn.Linear(in_dim(n), HIDDEN)
        self.cell = nn.GRUCell(HIDDEN, HIDDEN)
        self.head = nn.Linear(HIDDEN, n)

    def forward(self, x, h=None):
        e = F.relu(self.enc(x))
        h = self.cell(e, h)
        return self.head(h), h


class Backend:
    def __init__(self, recurrent, n, seed=0):
        self.n = n
        self.net = GRU(n, seed) if recurrent else MLP(n, seed)
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


def evaluate(net, test, n):
    """Overall accuracy, plus accuracy on the subset where the app actually
    CHANGES. The second number is the real measure: predicting no change is
    what persistence already does for free."""
    net.eval()
    h = None
    hit = seen = 0
    ch_hit = ch_seen = 0
    with torch.no_grad():
        for i, row in enumerate(test):
            if i % EPISODE == 0:
                h = None
            logits, h = net(encode(row, n), h)
            right = int(logits.argmax(1).item()) == row[3]
            seen += 1
            hit += right
            if row[0][-1] != row[3]:
                ch_seen += 1
                ch_hit += right
    return (100.0 * hit / seen,
            100.0 * ch_hit / ch_seen if ch_seen else None,
            ch_seen)


def run(arm, train, test, n, seed):
    b = Backend(arm != "memoryless", n, seed)
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

    acc, ch_acc, _ = evaluate(b.net, test, n)
    del b, layer
    return acc, ch_acc


ARMS = ["frozen", "memoryless", "online", "guarded"]


if __name__ == "__main__":
    rows = load()
    n_sw, pct_sw = switch_rate(rows)
    span = rows[-1][0] - rows[0][0]

    print(f"Rung 5: {len(rows)} samples over {span}")
    print(f"app switches: {n_sw} ({pct_sw:.1f}% of samples)")
    print(f"predicting {HORIZON} samples ahead = "
          f"{HORIZON * 10 // 60} minutes\n")

    if n_sw < 200:
        print("  WARNING: very few switches. The signal is thin and any\n"
              "  result should be treated as indicative.\n")

    data, vocab, counts = build(rows)
    split = int(len(data) * (1 - TEST_FRACTION))
    train, test = data[:split], data[split:]

    print(f"{len(vocab)} app classes, {len(train)} training moments, "
          f"{len(test)} test moments, chronological split")
    print("\napps:")
    for a, n in counts.most_common(10):
        print(f"  {a:>22}: {n:>5} ({100.0 * n / len(rows):5.1f}%)")

    b = baselines(train, test, len(vocab))
    best = max(b.values())
    print(f"\nbaselines on the held-out tail:")
    print(f"  majority        {b['majority']:5.2f}%")
    print(f"  persistence     {b['persistence']:5.2f}%  "
          f"(same app in {HORIZON * 10 // 60} minutes)")
    print(f"  bigram          {b['bigram']:5.2f}%")
    print(f"  strongest       {best:5.2f}%\n")

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    results = {}
    for arm in ARMS:
        accs, chs = [], []
        for seed in SEEDS:
            a, c = run(arm, train, test, len(vocab), seed)
            accs.append(a)
            chs.append(c)
        results[arm] = dict(acc=mean(accs), change=mean(chs))
        d = mean(accs) - best
        mark = "  BEATS BASELINE" if d > 0 else ""
        print(f"  {arm:>11}: {mean(accs):5.2f}%  ({d:+5.2f})  "
              f"on switches {mean(chs):5.2f}%{mark}", flush=True)

    print("\n" + "=" * 68)
    print(f"{'arm':>11} {'overall':>9} {'vs base':>9} "
          f"{'when it changes':>17}")
    print("-" * 68)
    print(f"{'persistence':>11} {b['persistence']:>8.2f}% "
          f"{b['persistence'] - best:>+8.2f} {0.0:>16.2f}%")
    print(f"{'bigram':>11} {b['bigram']:>8.2f}% "
          f"{b['bigram'] - best:>+8.2f}")
    print("-" * 68)
    for arm in ARMS:
        r = results[arm]
        print(f"{arm:>11} {r['acc']:>8.2f}% {r['acc'] - best:>+8.2f} "
              f"{r['change']:>16.2f}%")
    print("=" * 68)

    winners = [a for a in ARMS
               if a != "frozen" and results[a]["acc"] > best]
    print()
    if winners:
        print(f"  RUNG 5 CLEARED by {', '.join(winners)}.")
        print("  A network grown from random weights, learning from a record")
        print("  of one person's actual attention one moment at a time, beat")
        print("  the best simple rule. That is the top of the ladder.")
    else:
        print("  NOT CLEARED on overall accuracy.")

    ch = max(results[a]["change"] for a in ARMS if a != "frozen")
    print(f"\n  best accuracy WHEN THE APP ACTUALLY CHANGES: {ch:.2f}%")
    print("  persistence scores 0% there by definition, so anything above")
    print("  zero is something a simple rule cannot do at all.")

    print("""
Two numbers matter and they say different things.

  OVERALL vs the baseline decides the rung. Persistence is strong on this
      data because people stay in apps, so beating it means finding
      structure a "nothing changes" rule cannot represent.

  WHEN IT CHANGES is the more interesting one. Persistence scores exactly
      zero there — it can never predict a switch, by construction. Any
      score above zero is the model doing something no simple rule can.
      That is the part that would actually be useful in a product: knowing
      when someone is about to move, not that they are still where they
      were.
""")
    with open("rung5d.json", "w") as f:
        json.dump(dict(results=results, baselines=b, vocab=len(vocab),
                       samples=len(rows), switches=n_sw), f, indent=2)
    print("wrote rung5d.json")