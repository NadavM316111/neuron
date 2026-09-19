"""Does rung 4 hold anywhere other than Chicago?

The result: guarded 59.24% against extrapolation's 53.63%, plus a clean
retention finding where unprotected arms lost 26 points on the season
furthest from the end of training and guarded lost none.

That was two seeds, one weather station, one task. Thin for something to
build on. This runs the same code against five climates that behave in
genuinely different ways:

  Chicago      strong four-season continental. The original.
  Singapore    equatorial. Almost no seasonal swing at all, so the
               retention finding has nothing to hold onto and should
               vanish. That is the control.
  Phoenix      arid, huge daily swing, mild winter
  Reykjavik    maritime, small swing, weather that changes hour to hour
  Darwin       monsoon, two seasons rather than four, wet and dry

Singapore is the important one. If guarded still shows a summer advantage
somewhere with no seasons, the effect was never about retention and the
Chicago reading was a coincidence dressed up as a mechanism.

Only three arms, to keep this to twenty minutes: the memoryless baseline,
plain online, and guarded. K20-lowlr is dropped since it is slow and
already understood, and offline is dropped since it is not the claim.
"""

import json
import math
import os
import time
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

from sensor import VARIABLES, HORIZON, RISE, CLASSES
from stability import StabilityLayer


PLACES = {
    "chicago":   (41.88, -87.63, "continental, strong four seasons"),
    "singapore": (1.35, 103.82, "equatorial, almost no seasonal swing"),
    "phoenix":   (33.45, -112.07, "arid, huge daily swing"),
    "reykjavik": (64.15, -21.94, "maritime, hour-to-hour changeability"),
    "darwin":    (-12.46, 130.84, "monsoon, wet and dry rather than four"),
}

START = "2019-01-01"
END = "2023-12-31"
TEST_YEAR = "2023"

LAG = 3
HIDDEN = 128
LR = 3e-4
SEEDS = [0, 1]
EPISODE = 720
SEQ_LEN = 10
SEQ_COUNT = 2

N_FEAT = len(VARIABLES) * (LAG + 1)
SEASONS = ["winter", "spring", "summer", "autumn"]


def fetch_place(name, lat, lon):
    cache = f"weather_{name}.json"
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    url = ("https://archive-api.open-meteo.com/v1/archive"
           f"?latitude={lat}&longitude={lon}"
           f"&start_date={START}&end_date={END}"
           f"&hourly={','.join(VARIABLES)}")
    req = urllib.request.Request(url, headers={"User-Agent": "neuron"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read().decode())
    with open(cache, "w") as f:
        json.dump(data, f)
    return data


def build(data):
    """Standardised readings plus the previous LAG, labelled by what
    temperature does HORIZON hours ahead."""
    h = data["hourly"]
    cols = [h[v] for v in VARIABLES]
    n = len(h["time"])
    keep = [i for i in range(n)
            if all(c[i] is not None for c in cols)
            and i + HORIZON < n and cols[0][i + HORIZON] is not None]

    stats = []
    for c in cols:
        vals = [c[i] for i in keep]
        mu = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals)) or 1.0
        stats.append((mu, sd))

    base = []
    temps = cols[0]
    for i in keep:
        feats = [(cols[j][i] - stats[j][0]) / stats[j][1]
                 for j in range(len(cols))]
        change = temps[i + HORIZON] - temps[i]
        label = ("rising" if change > RISE else
                 "falling" if change < -RISE else "holding")
        base.append((feats, label, h["time"][i]))

    out = []
    for i in range(LAG, len(base)):
        feats = []
        for back in range(LAG, -1, -1):
            feats.extend(base[i - back][0])
        out.append((feats, base[i][1], base[i][2]))
    return out


def baselines(stream):
    counts = {}
    for _, lab, _ in stream:
        counts[lab] = counts.get(lab, 0) + 1
    maj = 100.0 * max(counts.values()) / len(stream)

    nraw = len(VARIABLES)
    right = 0
    for feats, label, _ in stream:
        delta = feats[LAG * nraw] - feats[(LAG - 1) * nraw]
        guess = ("rising" if delta > 0.02 else
                 "falling" if delta < -0.02 else "holding")
        right += guess == label
    return maj, 100.0 * right / len(stream), counts


def swing(stream):
    """Seasonal swing: the spread of monthly mean temperature. Near zero
    means there are effectively no seasons, which is what makes Singapore
    and Darwin the controls."""
    by_month = {}
    for feats, _, t in stream:
        by_month.setdefault(t[5:7], []).append(feats[LAG * len(VARIABLES)])
    means = [sum(v) / len(v) for v in by_month.values()]
    return max(means) - min(means)


def encode(feats):
    return torch.tensor(feats, dtype=torch.float32).unsqueeze(0)


class MLP(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(N_FEAT, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, len(CLASSES)))

    def forward(self, x, h=None):
        return self.net(x), None


class GRU(nn.Module):
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
            g["lr"] = max(1e-6, g["lr"] * 0.5)


def season_of(stamp, southern):
    m = int(stamp[5:7])
    if southern:
        m = (m + 6 - 1) % 12 + 1        # flip hemispheres
    return ("winter" if m in (12, 1, 2) else
            "spring" if m in (3, 4, 5) else
            "summer" if m in (6, 7, 8) else "autumn")


def evaluate(net, test, southern):
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
            s = season_of(stamp, southern)
            a, b = per.get(s, (0, 0))
            per[s] = (a + right, b + 1)
    return (100.0 * hit / seen,
            {k: 100.0 * a / b for k, (a, b) in per.items()})


def run(arm, train, test, seed, southern):
    b = Backend(arm != "memoryless", seed)
    layer = None
    if arm == "guarded":
        layer = StabilityLayer(
            b, canary=train[:40], seed=seed,
            window=200, warmup=30, top_fraction=0.50,
            coherence_veto=1e9, loss_floor=0.02, steps_per_update=1,
            rehearse_per_item=5, rehearse_count=SEQ_COUNT, rehearse_steps=1,
            anchor_size=100, buffer_size=500, sequence_len=SEQ_LEN,
            replay_policy="uniform",
            guard=True, guard_per_item=2000, canary_tolerance=0.5)

    for i, item in enumerate(train):
        if i % EPISODE == 0:
            b.reset_state()
        if layer is not None:
            layer.observe(item)
        else:
            b.update(item, 1)

    acc, per = evaluate(b.net, test, southern)
    del b, layer
    return acc, per


ARMS = ["memoryless", "online", "guarded"]


if __name__ == "__main__":
    print("Does rung 4 hold outside Chicago?\n")
    print("Singapore is the control: no seasons means nothing for "
          "rehearsal to\nprotect, so the retention advantage should "
          "VANISH there.\n")

    def mean(xs):
        return sum(xs) / len(xs)

    summary = {}
    for name, (lat, lon, desc) in PLACES.items():
        southern = lat < 0
        stream = build(fetch_place(name, lat, lon))
        train = [s for s in stream if s[2][:4] != TEST_YEAR]
        test = [s for s in stream if s[2][:4] == TEST_YEAR]
        maj, ext, counts = baselines(test)
        sw = swing(stream)

        print(f"--- {name} ({desc}) ---", flush=True)
        print(f"  seasonal swing {sw:.2f}   majority {maj:.1f}%   "
              f"extrapolation {ext:.1f}%")

        place = dict(swing=sw, majority=maj, extrapolation=ext, arms={})
        for arm in ARMS:
            accs, pers = [], []
            for seed in SEEDS:
                a, p = run(arm, train, test, seed, southern)
                accs.append(a)
                pers.append(p)
            avg = mean(accs)
            per_avg = {s: mean([p.get(s, 0.0) for p in pers])
                       for s in SEASONS}
            place["arms"][arm] = dict(acc=avg, per=per_avg)
            d = avg - max(maj, ext)
            mark = "  BEATS BASELINE" if d > 0 else ""
            print(f"  {arm:>11}: {avg:5.2f}%  ({d:+5.2f}){mark}",
                  flush=True)

        # The retention signal: how much accuracy is lost on the season
        # furthest from the end of training, relative to the best season.
        for arm in ARMS:
            p = place["arms"][arm]["per"]
            place["arms"][arm]["spread"] = max(p.values()) - min(p.values())
        print(f"  seasonal spread: " + "  ".join(
            f"{a} {place['arms'][a]['spread']:.1f}" for a in ARMS))
        summary[name] = place
        print()

    print("=" * 80)
    print("DOES GUARDED BEAT THE BASELINE EVERYWHERE?")
    print(f"{'place':>11} {'swing':>7} {'baseline':>9} " +
          "  ".join(f"{a:>11}" for a in ARMS))
    print("-" * 80)
    for name, p in summary.items():
        base = max(p["majority"], p["extrapolation"])
        cells = "  ".join(
            f"{p['arms'][a]['acc']:>10.2f}%" for a in ARMS)
        print(f"{name:>11} {p['swing']:>7.2f} {base:>8.2f}% {cells}")
    print("=" * 80)

    print("\nSEASONAL SPREAD (max minus min across seasons)")
    print("large spread means the model handles some seasons far worse,")
    print("which is what forgetting looks like on this data")
    print(f"{'place':>11} {'swing':>7} " + "  ".join(f"{a:>11}" for a in ARMS))
    print("-" * 62)
    for name, p in summary.items():
        cells = "  ".join(f"{p['arms'][a]['spread']:>10.1f}" for a in ARMS)
        print(f"{name:>11} {p['swing']:>7.2f} {cells}")

    wins = sum(1 for p in summary.values()
               if p["arms"]["guarded"]["acc"]
               > max(p["majority"], p["extrapolation"]))
    better = sum(1 for p in summary.values()
                 if p["arms"]["guarded"]["spread"]
                 < p["arms"]["online"]["spread"])

    print(f"\n  guarded beats the baseline in {wins} of {len(summary)} "
          f"climates")
    print(f"  guarded has a smaller seasonal spread than online in "
          f"{better} of {len(summary)}")

    print("""
Two questions.

  BEATS THE BASELINE EVERYWHERE -> rung 4 generalises and the Chicago
      result was not a fluke of one climate.

  RETENTION TRACKS THE SEASONAL SWING -> the mechanism is what we think it
      is. Guarded's advantage in spread should be LARGE where the swing is
      large (Chicago, Phoenix) and SMALL OR ABSENT where there are no
      seasons (Singapore). If guarded shows the same advantage in Singapore,
      the effect is not retention and the Chicago reading was a coincidence
      given a mechanism.
""")
    with open("replicate.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote replicate.json")