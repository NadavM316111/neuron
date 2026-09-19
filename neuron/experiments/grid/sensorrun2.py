"""Rung 4, fair fight: can it beat extrapolation given the same information?

The first attempt lost to extrapolation (53.6%), best learning arm 50.7%.
But the comparison was unfair in a way that made it the wrong test.

Extrapolation continues the last hour's direction, so it is HANDED the
previous reading. The model got only the current one and had to reconstruct
the previous from memory. So the test was really asking whether a GRU can
perfectly reproduce one number it saw one step ago, which is a duller
question than the one worth asking.

Fixed here: the input is the current reading PLUS the previous LAG readings.
Now the model has everything extrapolation has, and the question becomes
the real one — can a system learning online from real data find something
about weather that a simple rule does not capture?

Also fixed: online-K20 collapsed to 37.6% last time, barely above the
random floor, after helping in the grid world. A lower-learning-rate arm is
included to test whether that was instability rather than a real failure of
the method on continuous data.

Everything else is as before. Chronological split, trained on 2019-2022,
tested on 2023. Seasonal breakdown reported, because the retention finding
lived there: training ends in December, so summer is the material furthest
back, and only the guarded arm held onto it.
"""

import json
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from sensor import fetch, build_stream, CLASSES, VARIABLES
from stability import StabilityLayer


LAG = 3                  # previous readings included alongside the current
HIDDEN = 128
LR = 3e-4
LOW_LR = 3e-5
SEEDS = [0, 1]
EPISODE = 720
K = 20
SEQ_LEN = 10
SEQ_COUNT = 2
OFFLINE_EPOCHS = 5
OFFLINE_BATCH = 32
TEST_YEAR = "2023"

N_FEAT = len(VARIABLES) * (LAG + 1)
SEASONS = ["winter", "spring", "summer", "autumn"]


def add_lags(stream):
    """Each moment carries the current reading and the previous LAG.

    The stream is already in time order, so the previous readings are just
    the preceding entries. The first LAG moments are dropped rather than
    padded, so nothing is invented.
    """
    out = []
    for i in range(LAG, len(stream)):
        feats = []
        for back in range(LAG, -1, -1):
            feats.extend(stream[i - back][0])
        out.append((feats, stream[i][1], stream[i][2]))
    return out


def extrapolation_baseline(stream):
    """Continue the last hour's direction. Now computed from the SAME
    information the model has, so the comparison is like for like."""
    n_raw = len(VARIABLES)
    right = 0
    for feats, label, _ in stream:
        # temperature now, minus temperature one step back
        now = feats[LAG * n_raw]
        prev = feats[(LAG - 1) * n_raw]
        delta = now - prev
        guess = ("rising" if delta > 0.02 else
                 "falling" if delta < -0.02 else "holding")
        if guess == label:
            right += 1
    return 100.0 * right / len(stream)


def majority_baseline(stream):
    counts = {}
    for _, lab, _ in stream:
        counts[lab] = counts.get(lab, 0) + 1
    return 100.0 * max(counts.values()) / len(stream)


def encode(feats):
    return torch.tensor(feats, dtype=torch.float32).unsqueeze(0)


class MLP(nn.Module):
    """No memory. Sees the current reading and the last LAG, nothing more."""

    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(N_FEAT, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, len(CLASSES)),
        )

    def forward(self, x, h=None):
        return self.net(x), None


class GRU(nn.Module):
    """Carries state, so it can infer longer patterns — time of day, the
    passage of a front, the season — beyond the lag window."""

    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.enc = nn.Linear(N_FEAT, HIDDEN)
        self.cell = nn.GRUCell(HIDDEN, HIDDEN)
        self.head = nn.Linear(HIDDEN, len(CLASSES))

    def forward(self, x, h=None):
        e = F.relu(self.enc(x))
        h = self.cell(e, h)
        return self.head(h), h


class Backend:
    def __init__(self, recurrent, seed=0, lr=LR):
        self.net = GRU(seed) if recurrent else MLP(seed)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.h = None
        self._saved = None

    def begin_sequence(self):
        self._saved = self.h
        self.h = None

    def _xy(self, item):
        return encode(item[0]), torch.tensor([CLASSES.index(item[1])])

    def score(self, item):
        self.net.eval()
        with torch.no_grad():
            x, y = self._xy(item)
            logits, _ = self.net(x, self.h)
            return float(F.cross_entropy(logits, y).item()), None

    def update(self, item, steps):
        self.net.train()
        last = None
        for _ in range(steps):
            x, y = self._xy(item)
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


def evaluate(net, test):
    net.eval()
    h = None
    hit = seen = 0
    per = {}
    with torch.no_grad():
        for i, (feats, label, stamp) in enumerate(test):
            if i % EPISODE == 0:
                h = None
            logits, h = net(encode(feats), h)
            right = int(logits.argmax(1).item()) == CLASSES.index(label)
            seen += 1
            hit += right
            m = int(stamp[5:7])
            season = ("winter" if m in (12, 1, 2) else
                      "spring" if m in (3, 4, 5) else
                      "summer" if m in (6, 7, 8) else "autumn")
            a, b = per.get(season, (0, 0))
            per[season] = (a + right, b + 1)
    return (100.0 * hit / seen,
            {k: 100.0 * a / b for k, (a, b) in per.items()})


def run_online(mode, train, test, seed):
    recurrent = mode != "memoryless"
    lr = LOW_LR if "lowlr" in mode else LR
    b = Backend(recurrent, seed, lr)
    t0 = time.time()

    layer = None
    if mode == "guarded":
        layer = StabilityLayer(
            b, canary=train[:40], seed=seed,
            window=200, warmup=30, top_fraction=0.50,
            coherence_veto=1e9, loss_floor=0.02, steps_per_update=1,
            rehearse_per_item=5, rehearse_count=SEQ_COUNT, rehearse_steps=1,
            anchor_size=100, buffer_size=500, sequence_len=SEQ_LEN,
            replay_policy="uniform",
            guard=True, guard_per_item=2000, canary_tolerance=0.5)

    if mode.startswith("K20"):
        h_anchor = None
        buf = []
        b.net.train()
        for i, item in enumerate(train):
            if i % EPISODE == 0:
                h_anchor = None
                buf = []
            buf.append(item)
            if len(buf) > K:
                with torch.no_grad():
                    first = buf.pop(0)
                    _, h_anchor = b.net(encode(first[0]), h_anchor)
                    h_anchor = h_anchor.detach()
            h = h_anchor
            losses = []
            for it in buf:
                logits, h = b.net(encode(it[0]), h)
                losses.append(F.cross_entropy(
                    logits, torch.tensor([CLASSES.index(it[1])])))
            b.opt.zero_grad()
            torch.stack(losses).mean().backward()
            b.opt.step()
    else:
        for i, item in enumerate(train):
            if i % EPISODE == 0:
                b.reset_state()
            if layer is not None:
                layer.observe(item)
            else:
                b.update(item, 1)

    acc, per = evaluate(b.net, test)
    stats = layer.summary() if layer else {}
    out = dict(acc=acc, per=per, stats=stats,
               minutes=(time.time() - t0) / 60)
    del b, layer
    return out


def run_offline(train, test, seed):
    torch.manual_seed(seed)
    net = GRU(seed)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    rng = random.Random(seed)
    t0 = time.time()
    net.train()
    for _ in range(OFFLINE_EPOCHS):
        order = list(train)
        rng.shuffle(order)
        for i in range(0, len(order) - OFFLINE_BATCH, OFFLINE_BATCH):
            batch = order[i:i + OFFLINE_BATCH]
            x = torch.tensor([b[0] for b in batch], dtype=torch.float32)
            y = torch.tensor([CLASSES.index(b[1]) for b in batch])
            logits, _ = net(x, None)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    acc, per = evaluate(net, test)
    out = dict(acc=acc, per=per, stats={},
               minutes=(time.time() - t0) / 60)
    del net, opt
    return out


def run_frozen(test, seed):
    torch.manual_seed(seed)
    net = GRU(seed)
    acc, per = evaluate(net, test)
    return dict(acc=acc, per=per, stats={}, minutes=0.0)


ARMS = ["frozen", "memoryless", "online", "K20", "K20-lowlr", "guarded",
        "offline"]


if __name__ == "__main__":
    raw = build_stream(fetch())
    stream = add_lags(raw)
    train = [s for s in stream if s[2][:4] != TEST_YEAR]
    test = [s for s in stream if s[2][:4] == TEST_YEAR]

    print(f"Rung 4, fair fight. Input is the current reading plus the "
          f"previous {LAG}.")
    print(f"{N_FEAT} features, {len(train)} training hours, "
          f"{len(test)} test hours\n")

    maj = majority_baseline(test)
    ext = extrapolation_baseline(test)
    print(f"baselines on the test year:")
    print(f"  majority class      {maj:5.2f}%")
    print(f"  extrapolation       {ext:5.2f}%  "
          f"(now using the SAME information the model has)\n")

    def mean(xs):
        return sum(xs) / len(xs)

    results = {}
    for arm in ARMS:
        runs = []
        for seed in SEEDS:
            if arm == "frozen":
                r = run_frozen(test, seed)
            elif arm == "offline":
                r = run_offline(train, test, seed)
            else:
                r = run_online(arm, train, test, seed)
            runs.append(r)
        results[arm] = runs
        extra = ""
        if runs[0]["stats"]:
            st = runs[0]["stats"]
            extra = f"  gate {st['updates']}  buffer {st['buffer']}"
        print(f"  {arm:>12}: {mean([r['acc'] for r in runs]):5.2f}%"
              f"{extra}  {sum(r['minutes'] for r in runs):.1f}m",
              flush=True)

    print("\n" + "=" * 78)
    print(f"{'arm':>12} {'overall':>9} " +
          "  ".join(f"{s:>8}" for s in SEASONS))
    print("-" * 78)
    print(f"{'majority':>12} {maj:>8.2f}%")
    print(f"{'extrapolation':>12} {ext:>8.2f}%")
    print("-" * 78)
    for arm in ARMS:
        runs = results[arm]
        cells = "  ".join(
            f"{mean([r['per'].get(s, 0.0) for r in runs]):>7.1f}%"
            for s in SEASONS)
        print(f"{arm:>12} {mean([r['acc'] for r in runs]):>8.2f}% {cells}")
    print("=" * 78)

    best = max(maj, ext)
    print(f"\nagainst the strongest baseline ({best:.2f}%):")
    cleared = []
    for arm in ARMS:
        d = mean([r["acc"] for r in results[arm]]) - best
        mark = ""
        if d > 0 and arm not in ("frozen",):
            mark = "  BEATS IT"
            cleared.append(arm)
        print(f"  {arm:>12}: {d:+6.2f}{mark}")

    print()
    online_arms = [a for a in cleared if a not in ("offline",)]
    if online_arms:
        print(f"  RUNG 4 CLEARED by {', '.join(online_arms)}.")
        print("  A network grown from random weights, learning from real "
              "data one hour\n  at a time and never looking back, beat the "
              "best simple rule on a\n  stream nobody designed. That is out "
              "of the toy.")
    elif cleared:
        print("  Only the offline arm beat the baseline. Conventional "
              "training clears\n  it; single-pass online learning does not "
              "yet.")
    else:
        print("  NOT CLEARED. Nothing beat the baseline even with matched "
              "information,\n  so extrapolation is capturing something the "
              "learned model is not.")

    print("""
Watch the seasons as well. Training ends in December, so winter is the most
recent material and summer the furthest back. Last time the unprotected arms
followed that recency exactly and only the guarded arm held onto summer. If
that repeats with the fairer input, it is a solid retention result on real
data rather than a one-off.
""")
    with open("sensorrun2.json", "w") as f:
        json.dump({k: [dict(acc=r["acc"], per=r["per"]) for r in v]
                   for k, v in results.items()}, f, indent=2)
    print("wrote sensorrun2.json")