"""Milestone 3: does anything beyond one-step association get learned?

Fourth attempt. The previous three failed for reasons now all fixed in
world.py: too sparse, a visible leak, no uncertainty, and a spatial
shortcut. The third attempt was suggestive — the recurrent arm beat the
floor by about 30 points in two of three seeds — but one memoryless seed
scored 97.2%, so the floor was unstable and n=3 could not settle it.

This run: ten seeds, and a layout that changes with the seed so no spatial
pattern can be memorised.

  mlp+key      memoryless WITH has_key. The ceiling.
  mlp-nokey    memoryless WITHOUT has_key. The floor.
  gru-nokey    recurrent WITHOUT has_key. The real test.

The comparison that matters is PER SEED, not the mean. Each seed gets its
own world and its own stream, so the paired difference is the statistic.
"""

import json
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from world import GridWorld, ACTIONS, EVENTS, balance_check


STEPS = 20000
EPISODE = 200
HIDDEN = 64
LR = 3e-4
SEEDS = list(range(10))
N_CELLS = 5
CHECK_EVERY = 2000
TEST_SEED_OFFSET = 500      # held-out stream uses the SAME layout, new walk


def obs_dim(use_key):
    return 9 * N_CELLS + (1 if use_key else 0) + len(ACTIONS)


def encode(obs, action, use_key):
    v = torch.zeros(obs_dim(use_key))
    for i, cell in enumerate(obs["patch"]):
        v[i * N_CELLS + cell] = 1.0
    i = 9 * N_CELLS
    if use_key:
        v[i] = float(obs["has_key"])
        i += 1
    v[i + ACTIONS.index(action)] = 1.0
    return v


class MLP(nn.Module):
    """No memory. Sees only the current moment."""

    def __init__(self, use_key, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(obs_dim(use_key), HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, len(EVENTS)),
        )

    def forward(self, x, h=None):
        return self.net(x), None


class GRU(nn.Module):
    """Carries state between steps, so it CAN remember a key pickup."""

    def __init__(self, use_key, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.enc = nn.Linear(obs_dim(use_key), HIDDEN)
        self.cell = nn.GRUCell(HIDDEN, HIDDEN)
        self.head = nn.Linear(HIDDEN, len(EVENTS))

    def forward(self, x, h=None):
        e = F.relu(self.enc(x))
        h = self.cell(e, h)
        return self.head(h), h


class Runner:
    def __init__(self, kind, use_key, seed=0):
        self.use_key = use_key
        self.recurrent = kind == "gru"
        self.net = (GRU(use_key, seed) if self.recurrent
                    else MLP(use_key, seed))
        self.opt = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.h = None

    def _x(self, item):
        obs, action, _ = item
        return encode(obs, action, self.use_key).unsqueeze(0)

    def _y(self, item):
        return torch.tensor([EVENTS.index(item[2])])

    def learn(self, item, new_episode):
        """One online update per step in every arm, so update counts match.

        The hidden state carries forward but is detached after each step, so
        the graph never spans an optimiser step.
        """
        if new_episode:
            self.h = None
        self.net.train()
        logits, h = self.net(self._x(item), self.h)
        loss = F.cross_entropy(logits, self._y(item))
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.h = h.detach() if h is not None else None

    def evaluate(self, test, episode=EPISODE):
        self.net.eval()
        h = None
        correct = 0
        per = {e: [0, 0] for e in EVENTS}
        with torch.no_grad():
            for i, item in enumerate(test):
                if i % episode == 0:
                    h = None
                logits, h = self.net(self._x(item), h)
                guess = EVENTS[int(logits.argmax(1).item())]
                truth = item[2]
                per[truth][1] += 1
                if guess == truth:
                    correct += 1
                    per[truth][0] += 1
        acc = 100.0 * correct / len(test)
        per_acc = {e: (100.0 * c / n if n else None)
                   for e, (c, n) in per.items()}
        return acc, per_acc


def make_stream(layout_seed, walk_seed, n, episode=EPISODE):
    """Layout comes from layout_seed, the wandering from walk_seed. Train
    and test share a layout but not a path."""
    rng = random.Random(walk_seed)
    world = GridWorld(layout_seed)
    out = []
    for i in range(n):
        if i % episode == 0:
            world.reset()
        obs = world.observe()
        action = rng.choice(ACTIONS)
        _, event = world.step(action)
        out.append((obs, action, event))
    return out


ARMS = [
    ("mlp+key",   "mlp", True),
    ("mlp-nokey", "mlp", False),
    ("gru-nokey", "gru", False),
]


def run(kind, use_key, seed, train, test):
    r = Runner(kind, use_key, seed)
    checks = []
    for i, item in enumerate(train):
        r.learn(item, new_episode=(i % EPISODE == 0))
        if (i + 1) % CHECK_EVERY == 0:
            _, per = r.evaluate(test)
            checks.append(per.get("unlocked"))
    acc, per = r.evaluate(test)
    tail = [c for c in checks[-3:] if c is not None]
    alive = all(torch.isfinite(p).all() for p in r.net.parameters())
    out = dict(acc=acc, per=per, alive=alive,
               tail=sum(tail) / len(tail) if tail else None)
    del r
    return out


if __name__ == "__main__":
    print(f"{STEPS} steps, {len(SEEDS)} seeds, layout randomised per seed, "
          f"one update per step in every arm\n")

    print("balance check across all layouts:")
    maj = balance_check(SEEDS)
    if maj > 70:
        print("\nABORT: a memoryless guesser scores too well. The task "
              "cannot measure memory.")
        raise SystemExit
    print()

    t0 = time.time()
    per_seed = {name: [] for name, _, _ in ARMS}
    detail = {name: [] for name, _, _ in ARMS}

    print(f"{'seed':>5} " + "  ".join(f"{n:>11}" for n, _, _ in ARMS) +
          f" {'gru-floor':>10}")
    print("-" * 60)
    for seed in SEEDS:
        train = make_stream(seed, seed, STEPS)
        test = make_stream(seed, seed + TEST_SEED_OFFSET, 3000)
        row = {}
        for name, kind, use_key in ARMS:
            r = run(kind, use_key, seed, train, test)
            per_seed[name].append(r["tail"])
            detail[name].append(r)
            row[name] = r["tail"]
        diff = row["gru-nokey"] - row["mlp-nokey"]
        print(f"{seed:>5} " +
              "  ".join(f"{row[n]:>10.1f}%" for n, _, _ in ARMS) +
              f" {diff:>+9.1f}")

    print("-" * 60)

    def mean(xs):
        return sum(xs) / len(xs)

    diffs = [g - m for g, m in zip(per_seed["gru-nokey"],
                                   per_seed["mlp-nokey"])]
    wins = sum(1 for d in diffs if d > 0)
    ceiling = mean(per_seed["mlp+key"])
    floor = mean(per_seed["mlp-nokey"])
    gru = mean(per_seed["gru-nokey"])

    print(f"\n{'':>13} {'mean':>8} {'min':>8} {'max':>8}")
    for name, _, _ in ARMS:
        xs = per_seed[name]
        print(f"{name:>13} {mean(xs):>7.1f}% {min(xs):>7.1f}% "
              f"{max(xs):>7.1f}%")

    print(f"\nper-seed difference (gru minus floor):")
    print(f"  mean {mean(diffs):+.1f}   "
          f"min {min(diffs):+.1f}   max {max(diffs):+.1f}")
    print(f"  gru wins on {wins} of {len(SEEDS)} seeds")

    span = ceiling - floor
    closed = 100.0 * (gru - floor) / span if span > 1e-6 else 0.0
    print(f"\nfloor {floor:.1f}%  gru {gru:.1f}%  ceiling {ceiling:.1f}%")
    print(f"the recurrent arm closes {closed:.0f}% of the gap")
    print(f"\ntotal {(time.time() - t0) / 60:.1f} min")

    print("""
The paired per-seed difference is the statistic. Each seed is a different
world and a different stream, so the two arms face the same task on each row
and the comparison is like-for-like.

  wins on 8+ of 10, mean difference clearly positive -> a network grown from
      random weights, learning online from a sequential stream with no
      training run, uses memory of an earlier event. Milestone 3.
  wins on 5-7, or a mean difference smaller than the spread -> unresolved,
      and the honest answer is that this setup cannot distinguish it.
  wins on few -> having memory in the architecture does not mean online
      single-pass learning finds it.
""")
    with open("grow_memory.json", "w") as f:
        json.dump(dict(per_seed=per_seed, diffs=diffs), f, indent=2)
    print("wrote grow_memory.json")