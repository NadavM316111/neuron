"""Confirming the fix at full length.

overlap.py showed that updating every step while backpropagating through
the last 20 closes the memory gap from -11.2 to -2.6, in a single pass over
a stream that is never revisited.

But it ran at 30,000 steps with one seed, so everything was undertrained:
the ceiling reached only 67.8% simple against 86.9% in the 60,000-step run.
A gap measured against an undertrained ceiling is easier to close, so that
result is promising rather than solid.

This is the same experiment at full length with three seeds. The stride=5
arm is dropped, since it already showed that fewer updates hurt regardless
of window length, and the time is better spent on seeds.

Three arms:
  ceiling         reads the hidden state. The upper bound.
  K=1             one-step credit. Every experiment in this project until
                  today, and the thing being corrected.
  K=20 stride=1   overlapping windows, single pass. The claim.

If the gap stays near zero at full length across three seeds, the fix is
established and a lot of earlier measurements need redoing.
"""

import json
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from bigworld import BigWorld, ACTIONS, EVENTS, FORKS


STEPS = 60000
EPISODE = 400
HIDDEN = 128
LR = 3e-4
SEEDS = [0, 1, 2]
N_TERRAIN = 14
PATCH = 25
N_STATE = 7
TEST_STEPS = 8000
REPORT_EVERY = 10000

SIMPLE = ["locked door", "dark cell", "ice", "heavy door"]
CONJ = ["chasm", "cold room"]

FORK_OF = {}
for name, (yes, no) in FORKS.items():
    for e in yes + no:
        FORK_OF[e] = name


def obs_dim(see_state):
    return PATCH * N_TERRAIN + len(ACTIONS) + (N_STATE if see_state else 0)


def encode(obs, action, see_state):
    v = torch.zeros(obs_dim(see_state))
    for i, cell in enumerate(obs["patch"]):
        v[i * N_TERRAIN + cell] = 1.0
    i = PATCH * N_TERRAIN
    v[i + ACTIONS.index(action)] = 1.0
    if see_state:
        i += len(ACTIONS)
        for j, s in enumerate(obs["state"]):
            v[i + j] = float(s)
    return v


class GRU(nn.Module):
    def __init__(self, see_state, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.enc = nn.Linear(obs_dim(see_state), HIDDEN)
        self.cell = nn.GRUCell(HIDDEN, HIDDEN)
        self.head = nn.Linear(HIDDEN, len(EVENTS))

    def forward(self, x, h=None):
        e = F.relu(self.enc(x))
        h = self.cell(e, h)
        return self.head(h), h


def walk(layout_seed, walk_seed, n, see_state, episode=EPISODE):
    rng = random.Random(walk_seed)
    world = BigWorld(layout_seed)
    out = []
    for i in range(n):
        if i % episode == 0:
            world.reset()
        obs = world.observe(hide_state=not see_state)
        action = rng.choice(ACTIONS)
        _, event = world.step(action)
        out.append((obs, action, event))
    return out


def evaluate(net, test, see_state):
    net.eval()
    h = None
    fh = {k: 0 for k in FORKS}
    fs = {k: 0 for k in FORKS}
    hit = seen = 0
    with torch.no_grad():
        for i, item in enumerate(test):
            if i % EPISODE == 0:
                h = None
            x = encode(item[0], item[1], see_state).unsqueeze(0)
            y = EVENTS.index(item[2])
            logits, h = net(x, h)
            right = int(logits.argmax(1).item()) == y
            seen += 1
            hit += right
            fork = FORK_OF.get(item[2])
            if fork:
                fs[fork] += 1
                fh[fork] += right
    return (100.0 * hit / seen,
            {k: (100.0 * fh[k] / fs[k] if fs[k] else None) for k in FORKS})


def run(name, window, see_state, seed):
    """One pass. Update every step, backpropagate through the last
    `window` steps from a detached anchor."""
    train = walk(seed, seed * 100 + 1, STEPS, see_state)
    test = walk(seed, 90000 + seed, TEST_STEPS, see_state)

    net = GRU(see_state, seed)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    t0 = time.time()
    updates = 0

    h_anchor = None
    buf = []

    net.train()
    for i, item in enumerate(train):
        if i % EPISODE == 0:
            h_anchor = None
            buf = []

        buf.append(item)
        if len(buf) > window:
            with torch.no_grad():
                first = buf.pop(0)
                x = encode(first[0], first[1], see_state).unsqueeze(0)
                _, h_anchor = net(x, h_anchor)
                h_anchor = h_anchor.detach()

        h = h_anchor
        losses = []
        for it in buf:
            x = encode(it[0], it[1], see_state).unsqueeze(0)
            y = torch.tensor([EVENTS.index(it[2])])
            logits, h = net(x, h)
            losses.append(F.cross_entropy(logits, y))

        opt.zero_grad()
        torch.stack(losses).mean().backward()
        opt.step()
        updates += 1

        if (i + 1) % REPORT_EVERY == 0:
            print(f"    {name} seed {seed}: {i + 1}/{STEPS}, "
                  f"{(time.time() - t0) / 60:.1f}m", flush=True)

    acc, forks = evaluate(net, test, see_state)
    out = dict(acc=acc, forks=forks, updates=updates,
               minutes=(time.time() - t0) / 60)
    del net, opt
    return out


ARMS = [
    ("ceiling",       1,  True),
    ("K=1",           1,  False),
    ("K=20 stride=1", 20, False),
]


if __name__ == "__main__":
    print(f"{STEPS} steps, ONE PASS, {HIDDEN} units, seeds {SEEDS}")
    print("confirming the overlapping-window fix at full length\n")

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    results = {}
    for name, window, see_state in ARMS:
        print(f"--- {name} ---", flush=True)
        runs = []
        for seed in SEEDS:
            r = run(name, window, see_state, seed)
            runs.append(r)
            s = mean([r["forks"][k] for k in SIMPLE])
            c = mean([r["forks"][k] for k in CONJ])
            print(f"  seed {seed}: simple {s:5.1f}%  conj {c:5.1f}%  "
                  f"{r['minutes']:.1f}m", flush=True)
        results[name] = runs
        print()

    print("=" * 88)
    print(f"{'arm':>15} {'overall':>8} " +
          "  ".join(f"{k[:9]:>9}" for k in FORKS))
    print("-" * 88)
    for name, _, _ in ARMS:
        runs = results[name]
        cells = "  ".join(
            f"{mean([r['forks'][k] for r in runs]):>8.1f}%" for k in FORKS)
        print(f"{name:>15} {mean([r['acc'] for r in runs]):>7.1f}% {cells}")
    print("=" * 88)

    ce = results["ceiling"]
    ce_s = mean([mean([r["forks"][k] for k in SIMPLE]) for r in ce])
    ce_c = mean([mean([r["forks"][k] for k in CONJ]) for r in ce])

    print(f"\n{'arm':>15} {'simple':>8} {'conj':>8} {'gap':>8} "
          f"{'per-seed gaps':>28}")
    print("-" * 74)
    for name, _, _ in ARMS:
        runs = results[name]
        s = mean([mean([r["forks"][k] for k in SIMPLE]) for r in runs])
        c = mean([mean([r["forks"][k] for k in CONJ]) for r in runs])
        per = []
        for r, cr in zip(runs, ce):
            rs = mean([r["forks"][k] for k in SIMPLE])
            rc = mean([r["forks"][k] for k in CONJ])
            cs = mean([cr["forks"][k] for k in SIMPLE])
            cc = mean([cr["forks"][k] for k in CONJ])
            per.append((rs - cs + rc - cc) / 2)
        print(f"{name:>15} {s:>7.1f}% {c:>7.1f}% "
              f"{(s - ce_s + c - ce_c) / 2:>7.1f} "
              f"{'  '.join(f'{p:+6.1f}' for p in per):>28}")

    print("""
The last row is the claim, at full length, across three seeds.

  gap near zero on all three seeds -> ESTABLISHED. Detaching the hidden
      state every step was the constraint all along, and it has been the
      constraint on every measurement in this project since Stage 2. The
      earlier readings about conjunctions, distance, frequency and
      multiplicity were all downstream of it.

  gap back near K=1's -> the 30,000-step result was an artifact of an
      undertrained ceiling, and the fix does not hold at full length.

  gap in between, or varying wildly across seeds -> partial, and the honest
      report is a range rather than a number.

If this holds, the next job is re-measuring what the bug distorted, starting
with sequence replay in stability.py, which replays runs with per-step
detached updates and therefore inherits the same defect.
""")
    with open("confirm.json", "w") as f:
        json.dump({k: [dict(acc=r["acc"], forks=r["forks"]) for r in v]
                   for k, v in results.items()}, f, indent=2)
    print("wrote confirm.json")