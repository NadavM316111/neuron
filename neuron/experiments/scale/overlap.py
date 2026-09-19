"""The real fix: long credit assignment WITHOUT giving up single-pass.

bptt.py found the cause. Every experiment in this project detached the
hidden state every step, so gradients reached back exactly one step and the
network was never trained to remember anything. Fixing that closed 65% of
the memory gap.

But it cheated. K=20 with twenty passes is twenty epochs, which is the thing
this whole project exists to avoid. Single-pass K=20 failed, not because the
window was wrong but because non-overlapping windows make 20x fewer
optimiser updates.

OVERLAPPING WINDOWS fix that. Update every step, but backpropagate through
the last K steps instead of one. Full update count, long credit assignment,
ONE PASS over a stream that is never revisited.

The cost is real: every update re-runs K steps of forward and backward, so
K=20 is roughly twenty times the work per step. Hence a shorter stream, one
seed, and a stride parameter to trade cost against update count.

  stride=1   update every step. Full overlap. The real thing.
  stride=5   update every 5th step, still backpropagating 20. Cheaper, and
             shows whether partial overlap captures most of the benefit.

Arms:
  ceiling         sees the hidden state. The upper bound.
  K=1             the old setup, one-step credit. Every prior experiment.
  K=20 stride=5   partial overlap
  K=20 stride=1   full overlap, single pass. The claim.

If the last one lands near the twenty-pass result, the fix is real and the
single-pass premise survives.
"""

import json
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from bigworld import BigWorld, ACTIONS, EVENTS, FORKS


STEPS = 30000
EPISODE = 400
HIDDEN = 128
LR = 3e-4
SEED = 0
N_TERRAIN = 14
PATCH = 25
N_STATE = 7
TEST_STEPS = 8000
REPORT_EVERY = 5000

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


def run(name, window, stride, see_state, seed=SEED):
    """One pass over the stream. Never revisited.

    At each update, the last `window` items are re-run through the network
    to build a fresh graph, so the gradient reaches back that far. The
    hidden state entering the window is detached, which bounds the cost
    while still assigning credit across the window.
    """
    train = walk(seed, seed * 100 + 1, STEPS, see_state)
    test = walk(seed, 90000 + seed, TEST_STEPS, see_state)

    net = GRU(see_state, seed)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    t0 = time.time()
    updates = 0

    # Hidden state at the START of the current window, detached.
    h_anchor = None
    buf = []              # items since the anchor

    net.train()
    for i, item in enumerate(train):
        if i % EPISODE == 0:
            h_anchor = None
            buf = []

        buf.append(item)
        if len(buf) > window:
            # Slide the anchor forward one step, so the window stays
            # `window` long and the cost per update stays bounded.
            with torch.no_grad():
                first = buf.pop(0)
                x = encode(first[0], first[1], see_state).unsqueeze(0)
                _, h_anchor = net(x, h_anchor)
                h_anchor = h_anchor.detach()

        if i % stride != 0:
            continue

        # Re-run the window to build a fresh graph from the detached anchor.
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

        if updates % REPORT_EVERY == 0:
            print(f"    {name}: {i + 1}/{STEPS} steps, {updates} updates, "
                  f"{(time.time() - t0) / 60:.1f}m", flush=True)

    acc, forks = evaluate(net, test, see_state)
    out = dict(acc=acc, forks=forks, updates=updates,
               minutes=(time.time() - t0) / 60)
    del net, opt
    return out


ARMS = [
    ("ceiling",       1,  1, True),
    ("K=1",           1,  1, False),
    ("K=20 stride=5", 20, 5, False),
    ("K=20 stride=1", 20, 1, False),
]


if __name__ == "__main__":
    print(f"{STEPS} steps, ONE PASS, {HIDDEN} units, seed {SEED}")
    print("overlapping windows: update every `stride` steps, backpropagate "
          "through the last K\n")

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    results = {}
    for name, window, stride, see_state in ARMS:
        print(f"--- {name} ---", flush=True)
        r = run(name, window, stride, see_state)
        results[name] = r
        s = mean([r["forks"][k] for k in SIMPLE])
        c = mean([r["forks"][k] for k in CONJ])
        print(f"  simple {s:5.1f}%  conj {c:5.1f}%  "
              f"{r['updates']} updates  {r['minutes']:.1f}m\n", flush=True)

    print("=" * 88)
    print(f"{'arm':>15} {'overall':>8} " +
          "  ".join(f"{k[:9]:>9}" for k in FORKS) + f" {'updates':>8}")
    print("-" * 88)
    for name, _, _, _ in ARMS:
        r = results[name]
        cells = "  ".join(f"{r['forks'][k]:>8.1f}%" for k in FORKS)
        print(f"{name:>15} {r['acc']:>7.1f}% {cells} {r['updates']:>8}")
    print("=" * 88)

    ce_s = mean([results["ceiling"]["forks"][k] for k in SIMPLE])
    ce_c = mean([results["ceiling"]["forks"][k] for k in CONJ])
    print(f"\n{'arm':>15} {'simple':>8} {'conj':>8} {'gap to ceiling':>16}")
    print("-" * 52)
    for name, _, _, _ in ARMS:
        r = results[name]
        s = mean([r["forks"][k] for k in SIMPLE])
        c = mean([r["forks"][k] for k in CONJ])
        print(f"{name:>15} {s:>7.1f}% {c:>7.1f}% "
              f"{(s - ce_s + c - ce_c) / 2:>15.1f}")

    print("""
The claim is the last row. One pass over a stream that is never revisited,
full update count, and gradients that reach twenty steps back.

  it lands near the twenty-pass result (gap around -9) -> the fix is real
      and the single-pass premise survives. Every measurement in this
      project since Stage 2 was limited by a defect, not by the world.
  it lands near K=1 (gap around -25) -> overlap is not enough, and what the
      twenty passes bought was simply more training rather than better
      credit assignment. That would be worth knowing and would point back
      at update count as the constraint.

Note this is 30,000 steps against bptt.py's 60,000, so the absolute numbers
run lower. Compare the GAP column across arms, not the raw accuracies
against the earlier run.
""")
    with open("overlap.json", "w") as f:
        json.dump({k: dict(acc=v["acc"], forks=v["forks"],
                           updates=v["updates"], minutes=v["minutes"])
                   for k, v in results.items()}, f, indent=2)
    print("wrote overlap.json")