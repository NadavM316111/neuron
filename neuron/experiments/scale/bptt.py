"""Does letting gradients flow further back fix the big world?

Four diagnostic experiments in, the picture is that no single property
explains the memory failure. Conjunction, distance and frequency are
refuted; multiplicity is worth about a third. Hunting further causes has hit
diminishing returns.

So this changes the question from "what causes it" to "what fixes it", and
tests the most suspicious thing in the setup.

Every experiment so far detaches the hidden state after EVERY step:

    h = h.detach()

That was chosen for stability and for matching update counts across arms.
But it means the gradient can never reach back past one step. The network is
told "you predicted the door wrong" and can only adjust the weights that
processed THAT moment. Nothing pushes it to have stored the switch sixty
steps earlier, because no gradient ever reaches that far.

Whatever memory it currently has is incidental: the GRU happens to carry
information forward, and the one-step gradient happens to exploit it. It was
never trained to remember on purpose.

This sweeps the truncation window K. The loss is accumulated over K steps
and the update happens once per window, so gradients flow back K steps.

  K = 1     the current setup, one-step credit only
  K = 5, 20, 60    progressively longer credit assignment

Run against the REAL big world, on the same six forks, so the number that
moves is the number that matters.

Note the trade: updates happen once per window, so K=60 makes 60x fewer
optimiser steps over the same stream. If a long window wins ANYWAY, that is
a strong result. If it loses, the cause is ambiguous, so a matched-update
arm is included that runs K=20 for 20x more steps.
"""

import json
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from bigworld import BigWorld, ACTIONS, EVENTS, FORKS, fairness_check


STEPS = 60000
EPISODE = 400
HIDDEN = 128
LR = 3e-4
SEEDS = [0, 1]
N_TERRAIN = 14
PATCH = 25
N_STATE = 7
TEST_STEPS = 8000

WINDOWS = [1, 5, 20, 60]

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
    """Per-fork accuracy, replayed in order so the hidden state means
    something."""
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


def run(window, see_state, seed, passes=1):
    """Truncated BPTT with window K.

    Losses accumulate across K steps, then one backward pass and one
    optimiser step. Gradients therefore reach K steps back instead of one.
    passes > 1 repeats the stream, to separate the effect of a longer window
    from the effect of making fewer updates.
    """
    train = walk(seed, seed * 100 + 1, STEPS, see_state)
    test = walk(seed, 90000 + seed, TEST_STEPS, see_state)

    net = GRU(see_state, seed)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    t0 = time.time()
    updates = 0

    net.train()
    for _ in range(passes):
        h = None
        losses = []
        for i, item in enumerate(train):
            if i % EPISODE == 0:
                # Episode boundary: flush whatever is pending, reset state.
                if losses:
                    opt.zero_grad()
                    torch.stack(losses).mean().backward()
                    opt.step()
                    updates += 1
                    losses = []
                h = None

            x = encode(item[0], item[1], see_state).unsqueeze(0)
            y = torch.tensor([EVENTS.index(item[2])])
            logits, h = net(x, h)
            losses.append(F.cross_entropy(logits, y))

            if len(losses) >= window:
                opt.zero_grad()
                torch.stack(losses).mean().backward()
                opt.step()
                updates += 1
                losses = []
                h = h.detach()      # cut the graph, keep the state

        if losses:
            opt.zero_grad()
            torch.stack(losses).mean().backward()
            opt.step()
            updates += 1

    acc, forks = evaluate(net, test, see_state)
    out = dict(acc=acc, forks=forks, updates=updates,
               minutes=(time.time() - t0) / 60)
    del net, opt
    return out


if __name__ == "__main__":
    worst, worst_name, _, _ = fairness_check(quiet=True)
    print(f"world check: worst fork {worst_name} at {worst:.1f}%\n")

    print(f"{STEPS} steps, {HIDDEN} units, {len(SEEDS)} seeds")
    print("K is how many steps the gradient can reach back.")
    print("K=1 is every previous experiment in this project.\n")

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    results = {}
    t0 = time.time()

    # The ceiling, for reference. Reads the hidden state, so K is irrelevant.
    ce = [run(1, True, s) for s in SEEDS]
    results["ceiling"] = ce
    print(f"  ceiling      : simple "
          f"{mean([mean([r['forks'][k] for k in SIMPLE]) for r in ce]):5.1f}%"
          f"  conj "
          f"{mean([mean([r['forks'][k] for k in CONJ]) for r in ce]):5.1f}%"
          f"  ({time.time() - t0:.0f}s)")

    for w in WINDOWS:
        runs = [run(w, False, s) for s in SEEDS]
        results[f"K={w}"] = runs
        s = mean([mean([r["forks"][k] for k in SIMPLE]) for r in runs])
        c = mean([mean([r["forks"][k] for k in CONJ]) for r in runs])
        print(f"  K={w:<3}        : simple {s:5.1f}%  conj {c:5.1f}%  "
              f"({runs[0]['updates']} updates, {time.time() - t0:.0f}s)")

    # Matched-update control: K=20 with 20 passes, so the update count is
    # comparable to K=1. Separates "longer window" from "fewer updates".
    runs = [run(20, False, s, passes=20) for s in SEEDS]
    results["K=20 x20"] = runs
    s = mean([mean([r["forks"][k] for k in SIMPLE]) for r in runs])
    c = mean([mean([r["forks"][k] for k in CONJ]) for r in runs])
    print(f"  K=20 x20     : simple {s:5.1f}%  conj {c:5.1f}%  "
          f"({runs[0]['updates']} updates, {time.time() - t0:.0f}s)")

    print("\n" + "=" * 84)
    print(f"{'arm':>12} {'overall':>8} " +
          "  ".join(f"{k[:9]:>9}" for k in FORKS) + f" {'updates':>8}")
    print("-" * 84)
    for name in ["ceiling"] + [f"K={w}" for w in WINDOWS] + ["K=20 x20"]:
        runs = results[name]
        cells = "  ".join(
            f"{mean([r['forks'][k] for r in runs]):>8.1f}%" for k in FORKS)
        print(f"{name:>12} {mean([r['acc'] for r in runs]):>7.1f}% {cells} "
              f"{runs[0]['updates']:>8}")
    print("=" * 84)

    print(f"\n{'arm':>12} {'simple':>8} {'conj':>8} "
          f"{'gap to ceiling':>16}")
    print("-" * 50)
    ce_s = mean([mean([r["forks"][k] for k in SIMPLE]) for r in results["ceiling"]])
    ce_c = mean([mean([r["forks"][k] for k in CONJ]) for r in results["ceiling"]])
    for name in ["ceiling"] + [f"K={w}" for w in WINDOWS] + ["K=20 x20"]:
        runs = results[name]
        s = mean([mean([r["forks"][k] for k in SIMPLE]) for r in runs])
        c = mean([mean([r["forks"][k] for k in CONJ]) for r in runs])
        print(f"{name:>12} {s:>7.1f}% {c:>7.1f}% "
              f"{(s - ce_s + c - ce_c) / 2:>15.1f}")

    print("""
K=1 is every experiment in this project so far. If the longer windows beat
it, the memory failure was never about the world at all: it was that the
gradient could not reach back to the moment worth remembering, so the
network was never actually trained to remember anything.

  longer K clearly better -> the fix, and a real one. Whatever memory the
      system had was incidental; now it can be trained on purpose.
  K=60 loses but K=20 x20 wins -> long windows help, but the lost update
      count has to be paid back. Fixable by running longer.
  nothing helps -> credit assignment is not the constraint either, and the
      honest conclusion is that this architecture has a ceiling on the big
      world that none of the obvious levers move.
""")
    with open("bptt.json", "w") as f:
        json.dump({k: [dict(acc=r["acc"], forks=r["forks"],
                            updates=r["updates"]) for r in v]
                   for k, v in results.items()}, f, indent=2)
    print("wrote bptt.json")