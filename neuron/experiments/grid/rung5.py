"""Rung 5: something with a person in it.

Every stream so far was either a world with rules we wrote, or real data
about the physical world. This is a record of what one person actually did,
in the order they did it.

The data is shell history: 1,733 commands, in order, no timestamps. Order is
the part that matters.

The task: given the recent commands, predict what the next one will be.
Commands are reduced to their FIRST WORD plus a coarse subtype, so `python
grow.py` and `python bigrun.py` are the same class. Otherwise almost every
command is unique and the task is impossible rather than hard.

This is much thinner than the weather stream — 1,733 items against 35,000 —
so the honest expectation is a noisier result. But command sequences have
strong local structure (cd then ls, git add then git commit), so there is
real signal per item.

Baselines, same discipline as rung 4:
  majority     always predict the most common command
  repeat       predict the same command as last time
  bigram       predict whatever most often FOLLOWED the last command. This
               is the strong one, and it is a real memory-free baseline.

Arms:
  frozen       random weights
  memoryless   sees a fixed window of recent commands, no state
  online       recurrent, single pass, unprotected
  guarded      the same through stability.py with sequence replay

The retention question transfers directly. Early history is old work; recent
history is today's grid-world experiments. An unprotected online learner
should be good at the recent material and bad at the old, and the guarded
arm should be even across both. That is the same shape as the seasonal
result, on a person instead of a planet.
"""

import json
import os
import random
import re
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from stability import StabilityLayer


HISTORY = os.path.expanduser("~/.zsh_history")
LAG = 4                  # how many recent commands the input carries
HIDDEN = 128
LR = 3e-4
SEEDS = [0, 1, 2]
EPISODE = 200
SEQ_LEN = 10
SEQ_COUNT = 2
TEST_FRACTION = 0.2      # the LAST fifth, chronologically
MIN_COUNT = 5            # commands rarer than this become "other"


def classify(line):
    """Reduce a command to a class.

    First word, plus a subtype for the few commands where the subcommand
    matters more than the program (git, pip, brew). Everything else is just
    the program name, so `python a.py` and `python b.py` are one class.

    Arguments are DISCARDED entirely. Nothing here retains a filename, a
    path, a URL or a flag.
    """
    line = line.strip()
    if line.startswith(":"):
        # zsh extended format: ": <epoch>:<elapsed>;<command>"
        parts = line.split(";", 1)
        line = parts[1] if len(parts) > 1 else ""
    line = line.strip()
    if not line:
        return None

    words = re.split(r"\s+", line)
    head = os.path.basename(words[0])

    if head in ("git", "pip", "pip3", "brew", "npm") and len(words) > 1:
        sub = words[1]
        if re.match(r"^[a-z][a-z-]*$", sub):
            return f"{head} {sub}"
    return head


def load():
    """Read the history file, reduce every line to a class, drop the rest.

    The returned stream contains ONLY class labels. The original commands,
    their arguments, and anything identifying are not carried forward.
    """
    with open(HISTORY, "rb") as f:
        raw = f.read().decode("utf-8", errors="replace")

    classes = []
    for line in raw.splitlines():
        c = classify(line)
        if c:
            classes.append(c)

    counts = {}
    for c in classes:
        counts[c] = counts.get(c, 0) + 1
    keep = {c for c, n in counts.items() if n >= MIN_COUNT}
    stream = [c if c in keep else "other" for c in classes]
    return stream, counts


def build(stream, vocab):
    """Each moment: the previous LAG commands, and the next one as label."""
    idx = {c: i for i, c in enumerate(vocab)}
    out = []
    for i in range(LAG, len(stream)):
        context = [idx[stream[i - back]] for back in range(LAG, 0, -1)]
        out.append((context, idx[stream[i]]))
    return out


def baselines(train, test, vocab):
    """majority, repeat, and bigram. Bigram is the one to beat."""
    n = len(vocab)

    counts = [0] * n
    for _, y in train:
        counts[y] += 1
    top = counts.index(max(counts))
    majority = 100.0 * sum(1 for _, y in test if y == top) / len(test)

    repeat = 100.0 * sum(1 for ctx, y in test if ctx[-1] == y) / len(test)

    follow = {}
    for ctx, y in train:
        prev = ctx[-1]
        d = follow.setdefault(prev, {})
        d[y] = d.get(y, 0) + 1
    best = {p: max(d, key=d.get) for p, d in follow.items()}
    bigram = 100.0 * sum(1 for ctx, y in test
                         if best.get(ctx[-1], top) == y) / len(test)

    return dict(majority=majority, repeat=repeat, bigram=bigram)


def encode(ctx, n_vocab):
    v = torch.zeros(LAG * n_vocab)
    for j, c in enumerate(ctx):
        v[j * n_vocab + c] = 1.0
    return v.unsqueeze(0)


class MLP(nn.Module):
    def __init__(self, n_vocab, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(LAG * n_vocab, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, n_vocab))

    def forward(self, x, h=None):
        return self.net(x), None


class GRU(nn.Module):
    def __init__(self, n_vocab, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.enc = nn.Linear(LAG * n_vocab, HIDDEN)
        self.cell = nn.GRUCell(HIDDEN, HIDDEN)
        self.head = nn.Linear(HIDDEN, n_vocab)

    def forward(self, x, h=None):
        e = F.relu(self.enc(x))
        h = self.cell(e, h)
        return self.head(h), h


class Backend:
    def __init__(self, recurrent, n_vocab, seed=0):
        self.n = n_vocab
        self.net = (GRU(n_vocab, seed) if recurrent
                    else MLP(n_vocab, seed))
        self.opt = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.h = None
        self._saved = None

    def begin_sequence(self):
        self._saved = self.h
        self.h = None

    def _xy(self, item):
        return encode(item[0], self.n), torch.tensor([item[1]])

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


def evaluate(net, test, n_vocab, halves=True):
    """Accuracy overall, and split into the first and second half of the
    test span, so drift within the held-out period is visible."""
    net.eval()
    h = None
    hit = seen = 0
    per = [[0, 0], [0, 0]]
    mid = len(test) // 2
    with torch.no_grad():
        for i, item in enumerate(test):
            if i % EPISODE == 0:
                h = None
            logits, h = net(encode(item[0], n_vocab), h)
            right = int(logits.argmax(1).item()) == item[1]
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
            guard=True, guard_per_item=500, canary_tolerance=0.5)

    if arm != "frozen":
        for i, item in enumerate(train):
            if i % EPISODE == 0:
                b.reset_state()
            if layer is not None:
                layer.observe(item)
            else:
                b.update(item, 1)

    acc, halves = evaluate(b.net, test, n_vocab)
    del b, layer
    return acc, halves


ARMS = ["frozen", "memoryless", "online", "guarded"]


if __name__ == "__main__":
    stream, counts = load()
    vocab = sorted(set(stream))
    data = build(stream, vocab)

    split = int(len(data) * (1 - TEST_FRACTION))
    train, test = data[:split], data[split:]

    print(f"Rung 5: {len(stream)} commands, {len(vocab)} classes")
    print(f"{len(train)} training moments, {len(test)} test moments, "
          f"chronological split\n")

    print("most common classes:")
    for c, n in sorted(counts.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {c:>16}: {n:>4} ({100.0 * n / len(stream):5.1f}%)")

    b = baselines(train, test, vocab)
    print(f"\nbaselines on the held-out tail:")
    print(f"  majority        {b['majority']:5.2f}%")
    print(f"  repeat last     {b['repeat']:5.2f}%")
    print(f"  bigram          {b['bigram']:5.2f}%  "
          f"(what usually follows the last command)")
    best = max(b.values())
    print(f"  strongest       {best:5.2f}%\n")

    if len(train) < 500:
        print("  WARNING: very little training data. Treat any result as "
              "indicative\n  rather than measured.\n")

    def mean(xs):
        return sum(xs) / len(xs)

    results = {}
    for arm in ARMS:
        accs, firsts, seconds = [], [], []
        for seed in SEEDS:
            a, h = run(arm, train, test, len(vocab), seed)
            accs.append(a)
            firsts.append(h[0])
            seconds.append(h[1])
        results[arm] = dict(acc=mean(accs), first=mean(firsts),
                            second=mean(seconds))
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

    print("""
This is the first stream in the project that is a record of a person rather
than a world or a planet.

  anything beats the bigram baseline -> the system learned something about
      how this person works that a simple "what usually follows what" rule
      does not capture.
  nothing does -> at this data size the bigram captures most of the
      structure, which is an honest result. Command sequences are highly
      repetitive and a lookup table is hard to beat on 1,700 samples.

The two halves show drift within the held-out span. If the second half is
much better, the recent past resembles the test period more closely, which
is the same recency effect the weather seasons showed.

CAVEAT WORTH STATING PLAINLY: 1,733 commands is about 5% of the weather
stream, and much of it is one day's work. Any result here is indicative.
The proper version of this rung needs months of data.
""")
    with open("rung5.json", "w") as f:
        json.dump(dict(results=results, baselines=b,
                       n=len(stream), vocab=len(vocab)), f, indent=2)
    print("wrote rung5.json")