"""Rung 4: the system against a stream nobody designed.

Four years of hourly weather from a real station. No hidden variable anyone
planted, no balanced classes, no rules we wrote. The model sees temperature,
humidity, pressure and wind, with NO clock and NO calendar, and predicts
what temperature does three hours ahead.

To do well it has to infer time of day and season from the pattern of what
it has seen. That is the same problem as the grid world's key, except nobody
put it there.

There is no ceiling arm here, because nobody knows what the hidden state is.
So the comparison is against baselines:

  majority        always the most common class            42.3%
  extrapolation   continue the last hour's direction      54.1%

Extrapolation is the number that matters, and beating it is not automatic:
it uses the PREVIOUS reading, which the model never sees. To match it the
model must remember the last hour. To beat it, more than that.

Arms:
  frozen        random weights, never trained. The floor.
  memoryless    an MLP, online, single pass. Has no way to remember.
  online        a GRU, online, single pass, one-step credit (K=1).
  online-K20    a GRU with overlapping-window credit assignment.
  guarded       a GRU through stability.py with sequence replay.
  offline       shuffled, five passes. What a training run would achieve.

CHRONOLOGICAL SPLIT. Trained on 2019 to 2022, tested on 2023, which the
model never sees. Random splitting would leak: neighbouring hours are nearly
identical, so a shuffled test set would sit inside the training data.

Accuracy is reported per season as well as overall, because the seasonal
swing is the non-stationarity and an online learner should handle the season
it is currently in better than one it left months ago.
"""

import json
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from sensor import fetch, build_stream, baselines, CLASSES, VARIABLES
from stability import StabilityLayer


HIDDEN = 128
LR = 3e-4
SEEDS = [0, 1]
EPISODE = 720            # a month of hours, for hidden-state resets
K = 20
SEQ_LEN = 10
SEQ_COUNT = 2
OFFLINE_EPOCHS = 5
OFFLINE_BATCH = 32
TEST_YEAR = "2023"

N_FEAT = len(VARIABLES)


def encode(feats):
    return torch.tensor(feats, dtype=torch.float32).unsqueeze(0)


class MLP(nn.Module):
    """No memory. Sees only the current reading."""

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
    """Carries state, so it can infer time of day and season from the
    pattern of readings rather than being told."""

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
    """Wraps a network for the stability layer, and for plain online use."""

    def __init__(self, recurrent, seed=0):
        self.net = GRU(seed) if recurrent else MLP(seed)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=LR)
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
            g["lr"] = max(1e-5, g["lr"] * 0.5)


def evaluate(net, test):
    """Overall and per-season accuracy on the held-out year, replayed in
    order so a recurrent model has the history it needs."""
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
    """One pass, in order, never revisited."""
    recurrent = mode != "memoryless"
    b = Backend(recurrent, seed)
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

    if mode == "online-K20":
        # Overlapping windows: update every step, backpropagate through the
        # last K from a detached anchor.
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
    """The normal way: shuffled, several passes. No hidden state carried,
    because a shuffled batch has no coherent history."""
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


ARMS = ["frozen", "memoryless", "online", "online-K20", "guarded",
        "offline"]
SEASONS = ["winter", "spring", "summer", "autumn"]


if __name__ == "__main__":
    stream = build_stream(fetch())
    train = [s for s in stream if s[2][:4] != TEST_YEAR]
    test = [s for s in stream if s[2][:4] == TEST_YEAR]

    print(f"Rung 4: real weather, {len(train)} training hours, "
          f"{len(test)} test hours")
    print(f"trained on everything before {TEST_YEAR}, tested on "
          f"{TEST_YEAR}, chronological split\n")

    b = baselines(test)
    print("baselines on the TEST year:")
    print(f"  majority class      {b['majority']:5.1f}%")
    print(f"  extrapolation       {b['extrapolation']:5.1f}%  "
          f"(uses the previous reading, which the model never sees)\n")

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

    print("\n" + "=" * 76)
    print(f"{'arm':>12} {'overall':>9} " +
          "  ".join(f"{s:>8}" for s in SEASONS))
    print("-" * 76)
    print(f"{'majority':>12} {b['majority']:>8.2f}%")
    print(f"{'extrapolation':>12} {b['extrapolation']:>8.2f}%")
    print("-" * 76)
    for arm in ARMS:
        runs = results[arm]
        cells = "  ".join(
            f"{mean([r['per'].get(s, 0.0) for r in runs]):>7.1f}%"
            for s in SEASONS)
        print(f"{arm:>12} {mean([r['acc'] for r in runs]):>8.2f}% {cells}")
    print("=" * 76)

    best_base = max(b["majority"], b["extrapolation"])
    print(f"\nstrongest baseline: {best_base:.2f}%")
    for arm in ARMS:
        d = mean([r["acc"] for r in results[arm]]) - best_base
        mark = "  BEATS IT" if d > 0 else ""
        print(f"  {arm:>12}: {d:+6.2f}{mark}")

    print("""
This is the first stream in the project that nobody designed.

  online arms beat extrapolation -> a network grown from random weights,
      learning from real data one hour at a time and never looking back,
      has learned something real about weather that a simple rule does not
      capture. That is rung 4 and it is out of the toy.

  they beat the memoryless arm but not extrapolation -> memory helps, but
      not enough to beat continuing the last change. Honest partial result.

  they do not beat memoryless -> the recurrent state buys nothing on real
      data, and everything demonstrated so far depended on worlds where the
      thing to remember was planted deliberately.

Watch the seasons too. The test year runs January to December in order, so
an online learner meets winter first and summer last. If summer is much
better than winter, it is still adapting as it goes, which is what a
continual learner is supposed to do.
""")
    with open("sensorrun.json", "w") as f:
        json.dump({k: [dict(acc=r["acc"], per=r["per"]) for r in v]
                   for k, v in results.items()}, f, indent=2)
    print("wrote sensorrun.json")