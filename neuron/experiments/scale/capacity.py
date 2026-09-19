"""Is the conjunction failure capacity, or is it structural?

bigrun.py found that a 64-unit GRU learning online gets +7.9 points over
guessing on forks that depend on ONE hidden variable, and +1.0 on forks that
depend on TWO. The ceiling arm, which can see the hidden state, handles both
equally well (+26.4 and +25.8), so the task is learnable. The memory arm
simply cannot do the conjunctions.

Two explanations, and they lead opposite ways:

  CAPACITY    64 units is too small to track six time-varying variables and
              combine them. Then the fix is trivial and the finding
              evaporates.
  STRUCTURAL  learning "remember A AND B" online from a stream is genuinely
              harder than learning "remember A", regardless of size. Then it
              is a real limit, and it matters, because almost nothing in a
              person's life depends on a single condition.

Four widths, everything else identical. Two arms per width:
  online    hidden state hidden. The memory arm.
  ceiling   can see the hidden state. Shows what the width can represent
            when it does not have to remember anything, which separates
            "cannot hold it" from "cannot learn to infer it".

Scored against the MAJORITY-CLASS baseline from the fairness check, not
against the frozen network. Frozen scores near zero because it never
predicts rare classes, which flatters everything. The honest question is
whether the model beats always guessing the more common branch.

  gap closes with width          -> capacity. Scale it and move on.
  simple improves, conj flat     -> structural. The important answer.
  nothing improves               -> 60,000 steps is not enough regardless,
                                    and the whole comparison is undertrained
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
LR = 3e-4
SEEDS = [0, 1]
N_TERRAIN = 14
PATCH = 25
N_STATE = 7
TEST_STEPS = 8000

WIDTHS = [64, 128, 256, 512]

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
    def __init__(self, see_state, hidden, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.enc = nn.Linear(obs_dim(see_state), hidden)
        self.cell = nn.GRUCell(hidden, hidden)
        self.head = nn.Linear(hidden, len(EVENTS))

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


def baseline_rates():
    """What a memoryless guesser scores at each fork.

    This is the honest floor. Taken from the same simulation the fairness
    check uses, so the numbers are the ones printed by bigworld.py.
    """
    _, _, counts, _ = fairness_check(quiet=True)
    out = {}
    for name, (yes, no) in FORKS.items():
        a = sum(counts.get(e, 0) for e in yes)
        b = sum(counts.get(e, 0) for e in no)
        out[name] = 100.0 * max(a, b) / (a + b) if (a + b) else 100.0
    return out


def run(see_state, hidden, seed):
    train = walk(seed, seed * 100 + 1, STEPS, see_state)
    test = walk(seed, 90000 + seed, TEST_STEPS, see_state)

    net = GRU(see_state, hidden, seed)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    t0 = time.time()

    h = None
    net.train()
    for i, item in enumerate(train):
        if i % EPISODE == 0:
            h = None
        x = encode(item[0], item[1], see_state).unsqueeze(0)
        y = torch.tensor([EVENTS.index(item[2])])
        logits, h = net(x, h)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        h = h.detach()

    net.eval()
    h = None
    hit = seen = 0
    fh = {k: 0 for k in FORKS}
    fs = {k: 0 for k in FORKS}
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

    forks = {k: (100.0 * fh[k] / fs[k] if fs[k] else None) for k in FORKS}
    params = sum(p.numel() for p in net.parameters())
    out = dict(acc=100.0 * hit / seen, forks=forks, params=params,
               minutes=(time.time() - t0) / 60)
    del net, opt
    return out


if __name__ == "__main__":
    print("baseline: what a memoryless guesser scores at each fork\n")
    base = baseline_rates()
    for k, v in base.items():
        print(f"  {k:>14}: {v:5.1f}%")
    base_simple = sum(base[k] for k in SIMPLE) / len(SIMPLE)
    base_conj = sum(base[k] for k in CONJ) / len(CONJ)
    print(f"\n  simple mean      {base_simple:5.1f}%")
    print(f"  conjunctive mean {base_conj:5.1f}%\n")

    print(f"{STEPS} steps, {len(SEEDS)} seeds, widths {WIDTHS}")
    print("online = must remember, ceiling = can see the hidden state\n")

    results = {}
    for hidden in WIDTHS:
        for arm, see_state in [("online", False), ("ceiling", True)]:
            runs = []
            for seed in SEEDS:
                r = run(see_state, hidden, seed)
                runs.append(r)
            results[f"{arm}-{hidden}"] = runs
            s = sum(sum(r["forks"][k] for k in SIMPLE) / len(SIMPLE)
                    for r in runs) / len(runs)
            c = sum(sum(r["forks"][k] for k in CONJ) / len(CONJ)
                    for r in runs) / len(runs)
            print(f"  {arm:>7} {hidden:>4} units  "
                  f"({runs[0]['params']:>7} params)  "
                  f"simple {s:5.1f}%  conj {c:5.1f}%  "
                  f"{sum(r['minutes'] for r in runs):.1f}m")

    def mean(xs):
        return sum(xs) / len(xs)

    print("\n" + "=" * 78)
    print("ADVANTAGE OVER GUESSING (percentage points above the "
          "majority-class rate)")
    print(f"{'width':>7} " +
          f"{'online simple':>15} {'online conj':>13} {'gap':>7}   "
          f"{'ceil simple':>12} {'ceil conj':>10}")
    print("-" * 78)
    for hidden in WIDTHS:
        on = results[f"online-{hidden}"]
        ce = results[f"ceiling-{hidden}"]
        on_s = mean([mean([r["forks"][k] for k in SIMPLE]) for r in on]) - base_simple
        on_c = mean([mean([r["forks"][k] for k in CONJ]) for r in on]) - base_conj
        ce_s = mean([mean([r["forks"][k] for k in SIMPLE]) for r in ce]) - base_simple
        ce_c = mean([mean([r["forks"][k] for k in CONJ]) for r in ce]) - base_conj
        print(f"{hidden:>7} {on_s:>14.1f} {on_c:>13.1f} {on_s - on_c:>+7.1f}   "
              f"{ce_s:>11.1f} {ce_c:>10.1f}")
    print("=" * 78)

    print("\nper-fork detail, online arm:")
    print(f"{'width':>7} " + "  ".join(f"{k[:9]:>9}" for k in FORKS))
    print("-" * 78)
    for hidden in WIDTHS:
        on = results[f"online-{hidden}"]
        cells = "  ".join(
            f"{mean([r['forks'][k] for r in on]) - base[k]:>+8.1f}"
            for k in FORKS)
        print(f"{hidden:>7} {cells}")

    print("""
Read the "online conj" column down the widths.

  it climbs with width      -> CAPACITY. 64 units could not hold six
      time-varying variables and combine them. Scale it and the finding
      evaporates.
  it stays flat while "online simple" climbs -> STRUCTURAL. Learning
      "remember A AND B" online from a stream is genuinely harder than
      learning "remember A", and no amount of width fixes it. This is the
      important answer, and it matters because almost nothing in a real
      life depends on a single condition.
  neither climbs            -> 60,000 steps is not enough at any size, and
      every comparison so far has been undertrained.

The ceiling columns are the control. If they are flat too, the width is not
the constraint on anything and the problem is elsewhere entirely.
""")
    with open("capacity.json", "w") as f:
        json.dump({k: [dict(acc=r["acc"], forks=r["forks"],
                            params=r["params"]) for r in v]
                   for k, v in results.items()}, f, indent=2)
    print("wrote capacity.json")