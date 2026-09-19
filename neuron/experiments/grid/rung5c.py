"""Rung 5: a record of a person, at last.

Three attempts failed on data. Shell history was 1,731 commands, mostly one
day. File modification times turned out to be a record of package managers
and compilers rather than of a person — twenty thousand HTML files unpacked
by rustup in one burst. Chrome had 69 visits.

Safari has 28,592 visits over three and a half months. That is comparable in
size to the weather stream and it is genuinely a record of what one person
did, in order, with real timestamps.

PRIVACY. Only the DOMAIN and the TIMESTAMP are read. No URLs, no paths, no
page titles, no queries. Every domain is then replaced with an opaque label
(site_00, site_01, ...) before anything is printed or saved. The mapping
from label to domain is written to a separate local file that stays on this
machine. Nothing identifying appears in the output.

NOISE FILTERING, learned from the file-modification failure: browser history
is full of CDNs, analytics, ad domains and redirect chains, which are the
browser working rather than the person browsing. Domains below a visit
threshold are collapsed into "other", and rapid bursts to the same domain
are collapsed into single events.

THE TASK: given the last few sites and the time, predict the next site.

Baselines, same discipline as every rung:
  majority   always the most-visited site
  repeat     the same site as last time
  bigram     whatever usually follows the last site. The strong one, and
             the thing that beat everything on the shell history.
"""

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import time
from collections import Counter
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F

from stability import StabilityLayer


SAFARI = os.path.expanduser("~/Library/Safari/History.db")
CACHE = "browsing.json"
MAPPING = "browsing_mapping_local.json"   # stays on this machine

LAG = 5
HIDDEN = 128
LR = 3e-4
SEEDS = [0, 1, 2]
EPISODE = 400
SEQ_LEN = 10
SEQ_COUNT = 2
TEST_FRACTION = 0.2
MIN_VISITS = 40            # domains below this become "other"
BURST_SECONDS = 20         # repeat visits within this are one event

APPLE_EPOCH = datetime(2001, 1, 1)


def domain_of(url):
    """Host only, with www stripped. Nothing after the host is read."""
    s = url.split("//", 1)[-1]
    host = s.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    host = host.split(":", 1)[0].lower()
    return host[4:] if host.startswith("www.") else host


def collect(force=False):
    """Read domain and timestamp only, then label every domain opaquely.

    The cache written to disk contains labels and times. The mapping file
    contains the real domains and is never printed.
    """
    if os.path.exists(CACHE) and not force:
        with open(CACHE) as f:
            return json.load(f)

    tmp = tempfile.mktemp()
    shutil.copy2(SAFARI, tmp)
    con = sqlite3.connect(tmp)
    rows = con.execute(
        "select v.visit_time, i.url "
        "from history_visits v join history_items i "
        "on v.history_item = i.id order by v.visit_time").fetchall()
    con.close()
    os.remove(tmp)

    events = []
    for t, url in rows:
        if not url:
            continue
        d = domain_of(url)
        if d:
            events.append([float(t), d])

    counts = Counter(d for _, d in events)
    keep = {d for d, n in counts.items() if n >= MIN_VISITS}

    # Opaque labels, assigned by visit count so the ordering is stable.
    ordered = [d for d, _ in counts.most_common() if d in keep]
    label = {d: f"site_{i:02d}" for i, d in enumerate(ordered)}
    with open(MAPPING, "w") as f:
        json.dump({v: k for k, v in label.items()}, f, indent=2)

    # Collapse bursts: the same domain hit repeatedly within a few seconds
    # is one visit, not many.
    out = []
    last_d, last_t = None, -1e18
    for t, d in events:
        lab = label.get(d, "other")
        if lab == last_d and t - last_t < BURST_SECONDS:
            last_t = t
            continue
        out.append([t, lab])
        last_d, last_t = lab, t

    with open(CACHE, "w") as f:
        json.dump(out, f)
    print(f"  {len(rows)} raw visits -> {len(out)} events after burst "
          f"collapsing")
    print(f"  {len(keep)} domains kept, mapping in {MAPPING} (local only)")
    return out


def build(events):
    """Context of recent sites, plus hour and weekday, predicting the next
    site."""
    labels = [e[1] for e in events]
    vocab = sorted(set(labels))
    idx = {t: i for i, t in enumerate(vocab)}
    rows = []
    for i in range(LAG, len(labels)):
        ctx = [idx[labels[i - b]] for b in range(LAG, 0, -1)]
        dt = APPLE_EPOCH + timedelta(seconds=events[i - 1][0])
        rows.append((ctx, dt.hour, dt.weekday(), idx[labels[i]]))
    return rows, vocab


def baselines(train, test, n):
    counts = [0] * n
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
    net.eval()
    h = None
    hit = seen = 0
    per = [[0, 0], [0, 0]]
    mid = len(test) // 2
    with torch.no_grad():
        for i, row in enumerate(test):
            if i % EPISODE == 0:
                h = None
            logits, h = net(encode(row, n), h)
            right = int(logits.argmax(1).item()) == row[3]
            seen += 1
            hit += right
            k = 0 if i < mid else 1
            per[k][0] += right
            per[k][1] += 1
    return (100.0 * hit / seen,
            [100.0 * a / b if b else 0.0 for a, b in per])


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

    acc, halves = evaluate(b.net, test, n)
    stats = layer.summary() if layer else {}
    del b, layer
    return acc, halves, stats


ARMS = ["frozen", "memoryless", "online", "guarded"]


if __name__ == "__main__":
    print("Rung 5: browsing, domain and time only, labels anonymised\n")
    events = collect()
    rows, vocab = build(events)

    split = int(len(rows) * (1 - TEST_FRACTION))
    train, test = rows[:split], rows[split:]

    lo = APPLE_EPOCH + timedelta(seconds=events[0][0])
    hi = APPLE_EPOCH + timedelta(seconds=events[-1][0])
    print(f"\n{len(events)} events, {lo:%Y-%m-%d} to {hi:%Y-%m-%d}")
    print(f"{len(vocab)} site labels, {len(train)} training moments, "
          f"{len(test)} test moments, chronological split")
    print(f"input: last {LAG} sites, plus hour of day and weekday\n")

    b = baselines(train, test, len(vocab))
    best = max(b.values())
    print("baselines on the held-out tail:")
    print(f"  majority        {b['majority']:5.2f}%")
    print(f"  repeat last     {b['repeat']:5.2f}%")
    print(f"  bigram          {b['bigram']:5.2f}%")
    print(f"  strongest       {best:5.2f}%\n")

    def mean(xs):
        return sum(xs) / len(xs)

    results = {}
    for arm in ARMS:
        accs, f1, f2 = [], [], []
        extra = ""
        for seed in SEEDS:
            a, h, st = run(arm, train, test, len(vocab), seed)
            accs.append(a)
            f1.append(h[0])
            f2.append(h[1])
            if st and not extra:
                extra = f"  gate {st['updates']}  buffer {st['buffer']}"
        results[arm] = dict(acc=mean(accs), first=mean(f1), second=mean(f2))
        d = mean(accs) - best
        mark = "  BEATS BASELINE" if d > 0 else ""
        print(f"  {arm:>11}: {mean(accs):5.2f}%  ({d:+5.2f}){mark}{extra}",
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
        print("  of one person's actual behaviour one event at a time, beat")
        print("  the best simple rule. That is the top of the ladder.")
    else:
        print("  NOT CLEARED. The simple rules still capture more than the")
        print("  learned model does. Personal behaviour is highly repetitive,")
        print("  so a lookup table is a genuinely strong opponent.")

    print(f"""
The mapping from labels to real domains is in {MAPPING}, on this machine
only. Nothing in this output identifies anything.

The two halves show drift within the held-out span. Beating the bigram means
the model found structure in the timing and the longer context that "what
usually follows what" cannot represent.
""")
    with open("rung5c.json", "w") as f:
        json.dump(dict(results=results, baselines=b, vocab=len(vocab),
                       events=len(events)), f, indent=2)
    print("wrote rung5c.json")