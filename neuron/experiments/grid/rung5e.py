"""Rung 5, reformulated: predict the CHANGE, not the state.

The previous attempt collapsed. All three learning arms scored exactly
38.15%, identical to each other and to the majority baseline, meaning they
gave up and predicted Safari every time. Worse, persistence scores 72.96% by
copying the current app, which sits right there in the model's input — so
the model settled on the constant when the copy was available and better.

Cause: 60% of the answers are one class, the task is hard, and predicting
the majority minimises the loss. The same class-imbalance failure that
appeared at rung 1 and again in the big world.

TWO CHANGES.

  PREDICT THE CHANGE, NOT THE STATE. The question becomes "will the app be
      different in five minutes, and if so which one", rather than "which
      app". That removes the imbalance at its source, because "same" is now
      one option rather than 60% of the label mass hiding inside every
      class. It is also the question a product would ask — nobody needs
      telling they will still be in the app they are already in.

  WEIGHT THE CLASSES. Rare apps get proportionally more weight in the loss,
      so ignoring them is no longer free.

The label set is therefore: STAY, or SWITCH-TO-<app>. Persistence becomes
the trivial "always STAY" baseline, and the model has to earn anything above
it by identifying WHEN someone is about to move.

A CHANGE-ONLY accuracy is also reported: of the moments where a switch
really happened, how many were called. Persistence scores zero there by
construction, so any score above zero is something no simple rule can do.
"""

import json
import os
import time
from collections import Counter
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from stability import StabilityLayer


LOG = os.path.expanduser("~/neuron/activity_log.tsv")
HORIZON = 30            # 30 samples x 10s = 5 minutes ahead
LAG = 12                # 2 minutes of context
HIDDEN = 96
LR = 3e-4
SEEDS = [0, 1, 2]
EPISODE = 360
TEST_FRACTION = 0.25
MIN_COUNT = 20
SEQ_LEN = 10
SEQ_COUNT = 2
WEIGHT_POWER = 0.5      # 0 = no weighting, 1 = fully inverse-frequency

STAY = 0                # label 0 is always "no change"


def load():
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
    """Label is STAY, or SWITCH-TO-<app>.

    The context still carries the recent apps, so the model knows where it
    is. What changes is that the ANSWER is about movement.
    """
    counts = Counter(a for _, a in rows)
    keep = {a for a, n in counts.items() if n >= MIN_COUNT}
    apps = [a if a in keep else "other" for _, a in rows]
    times = [t for t, _ in rows]

    app_vocab = sorted(set(apps))
    app_idx = {a: i for i, a in enumerate(app_vocab)}
    # label 0 = STAY, label 1+i = switch to app i
    n_labels = 1 + len(app_vocab)

    out = []
    for i in range(LAG, len(apps) - HORIZON):
        ctx = [app_idx[apps[i - b]] for b in range(LAG, 0, -1)]
        now = apps[i]
        later = apps[i + HORIZON]
        label = STAY if later == now else 1 + app_idx[later]
        t = times[i]
        out.append((ctx, t.hour, t.weekday(), label))
    return out, app_vocab, n_labels, counts


def class_weights(train, n_labels):
    """Inverse-frequency weighting, softened by WEIGHT_POWER.

    Full inverse frequency makes the rarest class dominate and destabilises
    training on a stream this small; the square root is a common compromise.
    """
    counts = torch.zeros(n_labels)
    for r in train:
        counts[r[3]] += 1
    counts = counts.clamp_min(1.0)
    w = (counts.sum() / counts) ** WEIGHT_POWER
    return w / w.mean()


def baselines(train, test, n_labels):
    stay = 100.0 * sum(1 for r in test if r[3] == STAY) / len(test)

    counts = [0] * n_labels
    for r in train:
        counts[r[3]] += 1
    top = counts.index(max(counts))
    majority = 100.0 * sum(1 for r in test if r[3] == top) / len(test)

    follow = {}
    for r in train:
        d = follow.setdefault(r[0][-1], {})
        d[r[3]] = d.get(r[3], 0) + 1
    best = {p: max(d, key=d.get) for p, d in follow.items()}
    bigram = 100.0 * sum(1 for r in test
                         if best.get(r[0][-1], top) == r[3]) / len(test)
    return dict(always_stay=stay, majority=majority, bigram=bigram)


def in_dim(n_apps):
    return LAG * n_apps + 24 + 7


def encode(row, n_apps):
    v = torch.zeros(in_dim(n_apps))
    for j, c in enumerate(row[0]):
        v[j * n_apps + c] = 1.0
    v[LAG * n_apps + row[1]] = 1.0
    v[LAG * n_apps + 24 + row[2]] = 1.0
    return v.unsqueeze(0)


class MLP(nn.Module):
    def __init__(self, n_apps, n_labels, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(in_dim(n_apps), HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, n_labels))

    def forward(self, x, h=None):
        return self.net(x), None


class GRU(nn.Module):
    def __init__(self, n_apps, n_labels, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.enc = nn.Linear(in_dim(n_apps), HIDDEN)
        self.cell = nn.GRUCell(HIDDEN, HIDDEN)
        self.head = nn.Linear(HIDDEN, n_labels)

    def forward(self, x, h=None):
        e = F.relu(self.enc(x))
        h = self.cell(e, h)
        return self.head(h), h


class Backend:
    def __init__(self, recurrent, n_apps, n_labels, weights, seed=0):
        self.n_apps = n_apps
        self.w = weights
        self.net = (GRU(n_apps, n_labels, seed) if recurrent
                    else MLP(n_apps, n_labels, seed))
        self.opt = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.h = None
        self._saved = None

    def begin_sequence(self):
        self._saved = self.h
        self.h = None

    def _xy(self, row):
        return encode(row, self.n_apps), torch.tensor([row[3]])

    def score(self, row):
        self.net.eval()
        with torch.no_grad():
            x, y = self._xy(row)
            logits, _ = self.net(x, self.h)
            return float(F.cross_entropy(logits, y, weight=self.w).item()), None

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


def evaluate(net, test, n_apps):
    """Overall accuracy, accuracy on real switches, and how often the model
    predicts a switch at all — the last one catches a collapse to STAY."""
    net.eval()
    h = None
    hit = seen = 0
    sw_hit = sw_seen = 0
    predicted_switch = 0
    with torch.no_grad():
        for i, row in enumerate(test):
            if i % EPISODE == 0:
                h = None
            logits, h = net(encode(row, n_apps), h)
            pred = int(logits.argmax(1).item())
            right = pred == row[3]
            seen += 1
            hit += right
            if pred != STAY:
                predicted_switch += 1
            if row[3] != STAY:
                sw_seen += 1
                sw_hit += right
    return (100.0 * hit / seen,
            100.0 * sw_hit / sw_seen if sw_seen else None,
            100.0 * predicted_switch / seen)


def run(arm, train, test, n_apps, n_labels, weights, seed):
    b = Backend(arm != "memoryless", n_apps, n_labels, weights, seed)
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

    acc, sw, pred_sw = evaluate(b.net, test, n_apps)
    del b, layer
    return acc, sw, pred_sw


ARMS = ["frozen", "memoryless", "online", "guarded"]


if __name__ == "__main__":
    rows = load()
    data, app_vocab, n_labels, counts = build(rows)
    split = int(len(data) * (1 - TEST_FRACTION))
    train, test = data[:split], data[split:]
    weights = class_weights(train, n_labels)

    span = rows[-1][0] - rows[0][0]
    n_switch_labels = sum(1 for r in data if r[3] != STAY)

    print(f"Rung 5 reformulated: predict CHANGE, not state")
    print(f"{len(rows)} samples over {span}")
    print(f"{len(app_vocab)} apps, {n_labels} labels "
          f"(STAY + one per app)")
    print(f"{len(train)} training, {len(test)} test, "
          f"chronological split")
    print(f"switch labels: {n_switch_labels} of {len(data)} "
          f"({100.0 * n_switch_labels / len(data):.1f}%)")
    print(f"class weighting: inverse frequency ^ {WEIGHT_POWER}\n")

    b = baselines(train, test, n_labels)
    best = max(b.values())
    print("baselines on the held-out tail:")
    print(f"  always STAY     {b['always_stay']:5.2f}%")
    print(f"  majority label  {b['majority']:5.2f}%")
    print(f"  bigram          {b['bigram']:5.2f}%")
    print(f"  strongest       {best:5.2f}%\n")

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    results = {}
    for arm in ARMS:
        accs, sws, preds = [], [], []
        for seed in SEEDS:
            a, s, p = run(arm, train, test, len(app_vocab), n_labels,
                          weights, seed)
            accs.append(a)
            sws.append(s)
            preds.append(p)
        results[arm] = dict(acc=mean(accs), switch=mean(sws),
                            predicts=mean(preds),
                            spread=max(accs) - min(accs))
        d = mean(accs) - best
        mark = "  BEATS BASELINE" if d > 0 else ""
        print(f"  {arm:>11}: {mean(accs):5.2f}%  ({d:+5.2f})  "
              f"on switches {mean(sws):5.2f}%  "
              f"predicts a switch {mean(preds):4.1f}% of the time{mark}",
              flush=True)

    print("\n" + "=" * 82)
    print(f"{'arm':>11} {'overall':>9} {'vs base':>9} {'on switches':>13} "
          f"{'predicts switch':>17} {'spread':>8}")
    print("-" * 82)
    print(f"{'always STAY':>11} {b['always_stay']:>8.2f}% "
          f"{b['always_stay'] - best:>+8.2f} {0.0:>12.2f}% "
          f"{0.0:>16.1f}%")
    print("-" * 82)
    for arm in ARMS:
        r = results[arm]
        print(f"{arm:>11} {r['acc']:>8.2f}% {r['acc'] - best:>+8.2f} "
              f"{r['switch']:>12.2f}% {r['predicts']:>16.1f}% "
              f"{r['spread']:>7.2f}")
    print("=" * 82)

    collapsed = [a for a in ARMS if a != "frozen"
                 and results[a]["predicts"] < 1.0]
    identical = (len({round(results[a]["acc"], 2)
                      for a in ARMS if a != "frozen"}) == 1)

    print()
    if identical:
        print("  COLLAPSED AGAIN. All learning arms scored identically,")
        print("  which means they converged on the same trivial answer.")
        print("  The reformulation did not fix it.")
    elif collapsed:
        print(f"  {', '.join(collapsed)} never predict a switch, so they")
        print("  have collapsed to always-STAY. Class weighting was not")
        print("  strong enough — try WEIGHT_POWER closer to 1.")
    else:
        winners = [a for a in ARMS
                   if a != "frozen" and results[a]["acc"] > best]
        if winners:
            print(f"  RUNG 5 CLEARED by {', '.join(winners)}.")
        else:
            print("  Not cleared on overall accuracy, but the arms are")
            print("  learning rather than collapsing. Read the switch")
            print("  column — that is the part a simple rule cannot do.")

    print("""
Three columns, and each catches a different failure.

  OVERALL vs the baseline decides the rung.

  ON SWITCHES is the useful number. Always-STAY scores zero there by
      construction. Anything above zero is the model calling a moment that
      no simple rule can call.

  PREDICTS SWITCH is the collapse detector. If a model never predicts a
      switch, it has quietly become the always-STAY baseline no matter what
      its accuracy says. That is exactly what went wrong last time and was
      only visible because all three arms produced identical numbers.
""")
    with open("rung5e.json", "w") as f:
        json.dump(dict(results=results, baselines=b,
                       labels=n_labels, samples=len(rows)), f, indent=2)
    print("wrote rung5e.json")