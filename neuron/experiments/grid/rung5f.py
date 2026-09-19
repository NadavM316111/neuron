"""Rung 5, sixth attempt: predict a person, with the causes included.

Five attempts failed. The diagnosis was always the same, and it was never
the method:

  shell history        1,731 commands, mostly one day's work
  file timestamps      a record of package managers, not of a person
  browser history      8,244 events, six points short of a bigram
  app focus            collapsed to the majority class
  app focus, reframed  no collapse, but nothing beat "you will still be
                       where you are"

THE DIAGNOSIS: the logger recorded WHAT happened and never WHY. A message
arriving, a build finishing, someone calling, a train of thought ending —
all invisible. The model was asked to predict effects from a stream that did
not contain the causes.

THIS DATA HAS THE CAUSES, or the observable part of them:

  dwell         how long the current app has been in focus. People leave
                things after a while, and this is the strongest
                non-content predictor there is.
  idle          seconds since the last keyboard or mouse event. This
                separates reading a long page from having walked away —
                identical in every previous logger.
  since_switch  time since the last change of any kind.
  battery       charge and whether plugged in. A proxy for location.
  hour, weekday from the timestamp.

515 switches over nine days, which is thin — the browsing attempt had 446
and lost. So the volume is not what is new here. THE FEATURES ARE THE
EXPERIMENT. If dwell and idle carry real signal, thin data should still show
it; if they do not, more of the same would not rescue them.

THE TASK is deliberately the one that matters rather than the one that is
easy: WILL THE APP CHANGE in the next five minutes, and if so to what.
Predicting "no change" is what the previous attempt's baseline did for free.

An ablation runs alongside: the same model WITHOUT dwell and idle. That is
the comparison the whole attempt exists for — everything else is held
constant, so any difference is the new features.
"""

import json
import os
import random
from collections import Counter
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from stability import StabilityLayer


LOG = os.path.expanduser("~/neuron/neuron/results/activity2.tsv")
HORIZON = 20            # samples ahead: 20 x 15s = 5 minutes
LAG = 8                 # recent apps in the context
HIDDEN = 96
LR = 3e-4
SEEDS = [0, 1, 2]
EPISODE = 240
TEST_FRACTION = 0.25
MIN_COUNT = 20
GAP_SECONDS = 300       # a jump larger than this means the Mac slept
SEQ_LEN = 10

STAY = 0


def load():
    """Read the log, and mark where the machine slept.

    Gaps matter: the Mac sleeps when the lid closes, so nine calendar days
    produced 36 hours of samples. A gap is not a five-minute transition and
    must not be treated as one.
    """
    rows = []
    with open(LOG) as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 10:
                continue
            try:
                t = datetime.fromisoformat(p[0])
            except ValueError:
                continue
            rows.append(dict(
                t=t, app=p[1], dwell=int(p[2]), since=int(p[3]),
                idle=int(p[4]), battery=int(p[6]), plugged=int(p[7])))
    rows.sort(key=lambda r: r["t"])

    # split into continuous segments
    segments, current = [], [rows[0]]
    for prev, row in zip(rows, rows[1:]):
        if (row["t"] - prev["t"]).total_seconds() > GAP_SECONDS:
            if len(current) > HORIZON + LAG:
                segments.append(current)
            current = []
        current.append(row)
    if len(current) > HORIZON + LAG:
        segments.append(current)
    return rows, segments


def build(segments, use_timing=True):
    """Predict whether the app changes, and to what.

    Only moments with a full HORIZON of continuous samples ahead are used,
    so a sleep gap never masquerades as a transition.
    """
    counts = Counter(r["app"] for seg in segments for r in seg)
    keep = {a for a, n in counts.items() if n >= MIN_COUNT}
    vocab = sorted(keep | {"other"})
    idx = {a: i for i, a in enumerate(vocab)}
    n_labels = 1 + len(vocab)

    def app_of(r):
        return r["app"] if r["app"] in keep else "other"

    rows = []
    for seg in segments:
        for i in range(LAG, len(seg) - HORIZON):
            ctx = [idx[app_of(seg[i - b])] for b in range(LAG, 0, -1)]
            now = app_of(seg[i])
            later = app_of(seg[i + HORIZON])
            label = STAY if later == now else 1 + idx[later]
            r = seg[i]
            timing = []
            if use_timing:
                # the new features, scaled to a comparable range
                timing = [
                    min(r["dwell"], 3600) / 3600.0,
                    min(r["since"], 3600) / 3600.0,
                    min(r["idle"], 600) / 600.0,
                    1.0 if r["idle"] > 60 else 0.0,   # away from keyboard
                    r["battery"] / 100.0,
                    float(r["plugged"]),
                ]
            rows.append((ctx, r["t"].hour, r["t"].weekday(),
                         timing, label))
    return rows, vocab, n_labels


def in_dim(n_apps, n_timing):
    return LAG * n_apps + 24 + 7 + n_timing


def encode(row, n_apps, n_timing):
    v = torch.zeros(in_dim(n_apps, n_timing))
    for j, c in enumerate(row[0]):
        v[j * n_apps + c] = 1.0
    base = LAG * n_apps
    v[base + row[1]] = 1.0
    v[base + 24 + row[2]] = 1.0
    for j, t in enumerate(row[3]):
        v[base + 24 + 7 + j] = t
    return v.unsqueeze(0)


class GRU(nn.Module):
    def __init__(self, n_apps, n_timing, n_labels, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.enc = nn.Linear(in_dim(n_apps, n_timing), HIDDEN)
        self.cell = nn.GRUCell(HIDDEN, HIDDEN)
        self.head = nn.Linear(HIDDEN, n_labels)

    def forward(self, x, h=None):
        e = F.relu(self.enc(x))
        h = self.cell(e, h)
        return self.head(h), h


class Backend:
    def __init__(self, n_apps, n_timing, n_labels, weights, seed=0):
        self.n_apps = n_apps
        self.n_timing = n_timing
        self.w = weights
        self.net = GRU(n_apps, n_timing, n_labels, seed)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.h = None

    def begin_sequence(self):
        self.h = None

    def _xy(self, row):
        return (encode(row, self.n_apps, self.n_timing),
                torch.tensor([row[4]]))

    def score(self, row):
        self.net.eval()
        with torch.no_grad():
            x, y = self._xy(row)
            logits, _ = self.net(x, self.h)
            return float(F.cross_entropy(logits, y,
                                         weight=self.w).item()), None

    def update(self, row, steps):
        self.net.train()
        last = None
        for _ in range(steps):
            x, y = self._xy(row)
            logits, h = self.net(x, self.h)
            loss = F.cross_entropy(logits, y, weight=self.w)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.h = h.detach()
            last = float(loss.item())
        return last

    def reset_state(self):
        self.h = None

    def snapshot(self):
        return {k: v.detach().clone()
                for k, v in self.net.state_dict().items()}

    def restore(self, state):
        self.net.load_state_dict(state)

    def on_rollback(self):
        for g in self.opt.param_groups:
            g["lr"] = max(1e-6, g["lr"] * 0.5)


def class_weights(train, n_labels, power=0.5):
    counts = torch.zeros(n_labels)
    for r in train:
        counts[r[4]] += 1
    counts = counts.clamp_min(1.0)
    w = (counts.sum() / counts) ** power
    return w / w.mean()


def evaluate(net, test, n_apps, n_timing):
    """Overall, on real switches, and how often a switch is predicted.

    The last one catches the collapse that ruined the fourth attempt: a
    model that never predicts a switch has silently become the always-stay
    baseline whatever its accuracy says.
    """
    net.eval()
    h = None
    hit = seen = 0
    sw_hit = sw_seen = 0
    predicted = 0
    with torch.no_grad():
        for i, row in enumerate(test):
            if i % EPISODE == 0:
                h = None
            logits, h = net(encode(row, n_apps, n_timing), h)
            pred = int(logits.argmax(1).item())
            hit += pred == row[4]
            seen += 1
            if pred != STAY:
                predicted += 1
            if row[4] != STAY:
                sw_seen += 1
                sw_hit += pred == row[4]
    return (100.0 * hit / seen,
            100.0 * sw_hit / sw_seen if sw_seen else 0.0,
            100.0 * predicted / seen)


def run(arm, train, test, n_apps, n_timing, n_labels, weights, seed):
    b = Backend(n_apps, n_timing, n_labels, weights, seed)
    layer = None
    if arm == "guarded":
        layer = StabilityLayer(
            b, canary=train[:40], seed=seed,
            window=200, warmup=30, top_fraction=0.50,
            coherence_veto=1e9, loss_floor=0.02, steps_per_update=1,
            rehearse_per_item=5, rehearse_count=2, rehearse_steps=1,
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

    acc, sw, pred = evaluate(b.net, test, n_apps, n_timing)
    del b, layer
    return acc, sw, pred


def baselines(train, test, n_labels):
    stay = 100.0 * sum(1 for r in test if r[4] == STAY) / len(test)
    counts = [0] * n_labels
    for r in train:
        counts[r[4]] += 1
    top = counts.index(max(counts))
    majority = 100.0 * sum(1 for r in test if r[4] == top) / len(test)
    follow = {}
    for r in train:
        d = follow.setdefault(r[0][-1], {})
        d[r[4]] = d.get(r[4], 0) + 1
    best = {p: max(d, key=d.get) for p, d in follow.items()}
    bigram = 100.0 * sum(1 for r in test
                         if best.get(r[0][-1], top) == r[4]) / len(test)
    return dict(always_stay=stay, majority=majority, bigram=bigram)


ARMS = ["frozen", "memoryless", "online", "guarded"]


if __name__ == "__main__":
    rows, segments = load()
    span = rows[-1]["t"] - rows[0]["t"]
    total_switches = sum(
        1 for seg in segments
        for a, b in zip(seg, seg[1:]) if a["app"] != b["app"])

    print("Rung 5, sixth attempt: with the causes included\n")
    print(f"{len(rows)} samples over {span}")
    print(f"{len(segments)} continuous segments "
          f"(the Mac sleeps, so the stream has gaps)")
    print(f"{total_switches} app switches\n")
    print("NEW in this attempt: dwell, idle time, time since last switch,")
    print("battery. The previous five loggers recorded WHAT happened and")
    print("never WHY.\n")

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    results = {}
    for use_timing, label in [(True, "with timing"),
                              (False, "without timing")]:
        data, vocab, n_labels = build(segments, use_timing)
        n_timing = len(data[0][3])
        split = int(len(data) * (1 - TEST_FRACTION))
        train, test = data[:split], data[split:]
        weights = class_weights(train, n_labels)

        if label == "with timing":
            b = baselines(train, test, n_labels)
            best = max(b.values())
            print(f"{len(vocab)} apps, {len(train)} training moments, "
                  f"{len(test)} test moments")
            print(f"switch labels: "
                  f"{sum(1 for r in data if r[4] != STAY)} of {len(data)}\n")
            print("baselines:")
            print(f"  always stay     {b['always_stay']:5.2f}%")
            print(f"  majority        {b['majority']:5.2f}%")
            print(f"  bigram          {b['bigram']:5.2f}%")
            print(f"  strongest       {best:5.2f}%\n")

        print(f"--- {label} ---")
        for arm in ARMS:
            accs, sws, preds = [], [], []
            for seed in SEEDS:
                a, s, p = run(arm, train, test, len(vocab), n_timing,
                              n_labels, weights, seed)
                accs.append(a)
                sws.append(s)
                preds.append(p)
            results[(label, arm)] = dict(acc=mean(accs), sw=mean(sws),
                                         pred=mean(preds))
            print(f"  {arm:>11}: {mean(accs):5.2f}%  "
                  f"on switches {mean(sws):5.2f}%  "
                  f"predicts a switch {mean(preds):4.1f}%", flush=True)
        print(flush=True)

    print("=" * 78)
    print(f"{'arm':>12} {'with timing':>13} {'without':>10} "
          f"{'difference':>12} {'on switches':>13}")
    print("-" * 78)
    for arm in ARMS:
        w = results[("with timing", arm)]
        o = results[("without timing", arm)]
        print(f"{arm:>12} {w['acc']:>12.2f}% {o['acc']:>9.2f}% "
              f"{w['acc'] - o['acc']:>+11.2f} {w['sw']:>12.2f}%")
    print("=" * 78)

    print(f"\nDO THE NEW FEATURES HELP? "
          f"(with timing minus without, on the guarded arm)")
    g_w = results[("with timing", "guarded")]
    g_o = results[("without timing", "guarded")]
    print(f"  overall    {g_w['acc'] - g_o['acc']:+.2f}")
    print(f"  on switches {g_w['sw'] - g_o['sw']:+.2f}")

    print(f"\nVS THE STRONGEST BASELINE ({best:.2f}%):")
    for arm in ARMS:
        w = results[("with timing", arm)]
        mark = "  BEATS IT" if w["acc"] > best else ""
        print(f"  {arm:>11}: {w['acc'] - best:+6.2f}{mark}")

    collapsed = [a for a in ARMS if a != "frozen"
                 and results[("with timing", a)]["pred"] < 1.0]
    print()
    if collapsed:
        print(f"  {', '.join(collapsed)} never predict a switch, so they")
        print("  have silently become the always-stay baseline.")
    elif any(results[("with timing", a)]["acc"] > best
             for a in ARMS if a != "frozen"):
        print("  RUNG 5 CLEARED. A network grown from random weights,")
        print("  learning from a record of one person's attention one")
        print("  moment at a time, beat the best simple rule. That is the")
        print("  top of the ladder and it took six attempts.")
    else:
        print("  NOT CLEARED. The timing features are the sixth attempt's")
        print("  hypothesis; the difference column above says whether they")
        print("  carried any signal at all, which is worth knowing even")
        print("  when the rung does not clear.")

    print("""
The difference column is the experiment. Five attempts failed because the
logger recorded effects without causes, so this one added dwell and idle
time and holds everything else constant.

  timing helps and the rung clears -> the diagnosis was right and the data
      was the problem all along.
  timing helps but the rung does not clear -> the diagnosis was right and
      515 switches is simply too few. More waiting would then be worth it.
  timing does not help -> the diagnosis was wrong. What drives a person's
      attention is not visible in dwell and idle, and predicting it needs
      content the logger deliberately does not record.
""")
    with open("rung5f.json", "w") as f:
        json.dump({f"{k[0]}|{k[1]}": v for k, v in results.items()},
                  f, indent=2)
    print("wrote rung5f.json")