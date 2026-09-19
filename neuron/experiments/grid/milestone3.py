"""Milestone 3: does online learning beat ordinary training on early material?

Previous run said offline wins, 96.4% against guarded's 93.3%. But that
number is inflated. Most of the stream — walls, floor, ordinary moves — is
identical across all three phases, and only about 9% of steps differ
between the first two. A model that nails the shared cases and coin-flips
the contradicting ones lands near 96%, which is roughly what offline got.

So this version scores two things separately:

  SHARED     events that are the same regardless of which rules are in
             force. Everyone should get these, and they say nothing about
             retention.
  CONTESTED  events where THIS phase disagrees with at least one other
             phase given the same observation and action. These are the
             only steps where remembering the early rules matters.

Contested events are identified by replaying each test moment under every
rule set and keeping the ones where the outcome differs.

A life of three phases with contradictory rules on one layout:
  keyed  a locked door opens only while carrying a key
  open   a locked door always opens
  trap   a locked door never opens, and keys block you

Four arms over identical moments: offline (shuffled, many passes), online
(one pass in order), guarded (the same through stability.py), frozen.
"""

import json
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from world import (GridWorld, ACTIONS, EVENTS, DELTA, LOCKED, KEY,
                   contradiction_check)
from stability import StabilityLayer


PHASES = ["keyed", "open", "trap"]
PHASE_STEPS = 6000
EPISODE = 200
HIDDEN = 64
LR = 3e-4
SEEDS = [0, 1, 2]
N_CELLS = 5
OFFLINE_EPOCHS = 5
OFFLINE_BATCH = 32
TEST_STEPS = 2000


def obs_dim():
    return 9 * N_CELLS + 1 + len(ACTIONS)


def encode(obs, action):
    v = torch.zeros(obs_dim())
    for i, cell in enumerate(obs["patch"]):
        v[i * N_CELLS + cell] = 1.0
    v[9 * N_CELLS] = float(obs["has_key"])
    v[9 * N_CELLS + 1 + ACTIONS.index(action)] = 1.0
    return v


class MLP(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(obs_dim(), HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, len(EVENTS)),
        )

    def forward(self, x):
        return self.net(x)


class Backend:
    """The four stability.py callbacks, plus batch training for offline."""

    def __init__(self, seed=0):
        self.net = MLP(seed)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.grad_steps = 0

    def _xy(self, item):
        obs, action, event = item[0], item[1], item[2]
        return encode(obs, action).unsqueeze(0), \
            torch.tensor([EVENTS.index(event)])

    def score(self, item):
        self.net.eval()
        with torch.no_grad():
            x, y = self._xy(item)
            return float(F.cross_entropy(self.net(x), y).item()), None

    def update(self, item, steps):
        self.net.train()
        last = None
        for _ in range(steps):
            x, y = self._xy(item)
            loss = F.cross_entropy(self.net(x), y)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.grad_steps += 1
            last = float(loss.item())
        return last

    def update_batch(self, items):
        self.net.train()
        x = torch.stack([encode(i[0], i[1]) for i in items])
        y = torch.tensor([EVENTS.index(i[2]) for i in items])
        loss = F.cross_entropy(self.net(x), y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.grad_steps += len(items)
        return float(loss.item())

    def snapshot(self):
        return {k: v.detach().clone()
                for k, v in self.net.state_dict().items()}

    def restore(self, state):
        self.net.load_state_dict(state)

    def on_rollback(self):
        for g in self.opt.param_groups:
            g["lr"] = max(1e-5, g["lr"] * 0.5)

    def evaluate_split(self, test):
        """Accuracy on contested events and on shared events, separately.

        test items are (obs, action, event, contested).
        """
        self.net.eval()
        hit = {True: 0, False: 0}
        seen = {True: 0, False: 0}
        with torch.no_grad():
            for item in test:
                contested = item[3]
                x, y = self._xy(item)
                seen[contested] += 1
                if int(self.net(x).argmax(1).item()) == int(y.item()):
                    hit[contested] += 1
        return (100.0 * hit[True] / seen[True] if seen[True] else None,
                100.0 * hit[False] / seen[False] if seen[False] else None,
                seen[True], seen[False])


def outcome_under(world_state, action, rules, grid):
    """What WOULD happen from this state under a given rule set.

    Used to decide whether a moment is contested. Pure function of the
    state, so it does not disturb the world being walked.
    """
    r, c, keys = world_state
    dr, dc = DELTA[action]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < len(grid) and 0 <= nc < len(grid[0])):
        return "blocked"
    cell = grid[nr][nc]
    if cell == LOCKED:
        if rules == "open":
            return "unlocked"
        if rules == "trap":
            return "blocked"
        return "unlocked" if keys > 0 else "blocked"
    if cell == KEY:
        if rules == "trap":
            return "blocked"
        return "moved" if keys >= 1 else "got_key"
    if cell == 1:                      # WALL
        return "blocked"
    return "moved"


def walk(layout_seed, walk_seed, rules, n, episode=EPISODE, mark=False):
    """A stream of moments. With mark=True each moment also carries whether
    it is CONTESTED — whether any other rule set would have produced a
    different event from the same state and action."""
    rng = random.Random(walk_seed)
    world = GridWorld(layout_seed, rules=rules)
    out = []
    for i in range(n):
        if i % episode == 0:
            world.reset()
        obs = world.observe()
        action = rng.choice(ACTIONS)
        state = (world.r, world.c, world.keys)
        _, event = world.step(action)
        if mark:
            others = {outcome_under(state, action, other, world.grid)
                      for other in PHASES if other != rules}
            out.append((obs, action, event, event in others
                        and len(others) == 1 and False or
                        any(o != event for o in others)))
        else:
            out.append((obs, action, event))
    return out


def build_life(seed):
    out = []
    for i, rules in enumerate(PHASES):
        out += walk(seed, seed * 100 + i, rules, PHASE_STEPS)
    return out


def build_tests(seed):
    return {rules: walk(seed, 9000 + seed * 10 + i, rules, TEST_STEPS,
                        mark=True)
            for i, rules in enumerate(PHASES)}


def run(mode, seed, life, tests):
    b = Backend(seed)
    t0 = time.time()

    if mode == "frozen":
        pass
    elif mode == "offline":
        rng = random.Random(seed)
        for _ in range(OFFLINE_EPOCHS):
            order = list(life)
            rng.shuffle(order)
            for i in range(0, len(order) - OFFLINE_BATCH, OFFLINE_BATCH):
                b.update_batch(order[i:i + OFFLINE_BATCH])
    elif mode == "online":
        for item in life:
            b.update(item, 1)
    else:  # guarded
        layer = StabilityLayer(
            b, canary=life[:40], seed=seed,
            window=200, warmup=30, top_fraction=0.50,
            coherence_veto=1e9, loss_floor=0.02, steps_per_update=1,
            rehearse_per_item=5, rehearse_count=2, rehearse_steps=1,
            anchor_size=100, buffer_size=500,
            guard=True, guard_per_item=500, canary_tolerance=0.5)
        for item in life:
            layer.observe(item)
        del layer

    scores = {}
    for r in PHASES:
        con, sha, n_con, n_sha = b.evaluate_split(tests[r])
        scores[r] = dict(contested=con, shared=sha,
                         n_contested=n_con, n_shared=n_sha)
    out = dict(scores=scores, grad_steps=b.grad_steps,
               minutes=(time.time() - t0) / 60)
    del b
    return out


ARMS = ["frozen", "offline", "online", "guarded"]


if __name__ == "__main__":
    print("do the phases genuinely contradict each other?")
    worst = contradiction_check(SEEDS)
    if worst < 5:
        print(f"\nABORT: only {worst:.1f}% differ at worst.")
        raise SystemExit
    print(f"\n  good, {worst:.1f}% at worst\n")

    print(f"a life of {len(PHASES)} phases x {PHASE_STEPS} steps: "
          f"{' -> '.join(PHASES)}")
    print("scoring CONTESTED events (where phases disagree) separately "
          "from SHARED ones\n")

    results = {a: {p: dict(contested=[], shared=[]) for p in PHASES}
               for a in ARMS}
    grads = {a: [] for a in ARMS}
    counts = None

    for seed in SEEDS:
        life = build_life(seed)
        tests = build_tests(seed)
        if counts is None:
            counts = {p: (tests[p][0] and
                          sum(1 for t in tests[p] if t[3]),
                          len(tests[p])) for p in PHASES}
        for arm in ARMS:
            r = run(arm, seed, life, tests)
            for p in PHASES:
                results[arm][p]["contested"].append(
                    r["scores"][p]["contested"])
                results[arm][p]["shared"].append(r["scores"][p]["shared"])
            grads[arm].append(r["grad_steps"])
        print(f"  seed {seed} done")

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    print("\ncontested events per test stream:")
    for p in PHASES:
        n_con, n_tot = counts[p]
        print(f"  {p:>6}: {n_con} of {n_tot} "
              f"({100.0 * n_con / n_tot:.1f}%)")

    print("\n" + "=" * 78)
    print("CONTESTED events only — where remembering the phase matters")
    print(f"{'arm':>9} " + "  ".join(f"{p:>10}" for p in PHASES) +
          f" {'grad':>9}")
    print(f"{'':>9} " + f"{'(1st)':>10}  {'':>10}  {'(last)':>10}")
    print("-" * 78)
    for arm in ARMS:
        cells = "  ".join(
            f"{mean(results[arm][p]['contested']):>9.1f}%" for p in PHASES)
        print(f"{arm:>9} {cells} {mean(grads[arm]):>9.0f}")
    print("=" * 78)

    print("\nSHARED events — the same under every rule set")
    print(f"{'arm':>9} " + "  ".join(f"{p:>10}" for p in PHASES))
    print("-" * 78)
    for arm in ARMS:
        cells = "  ".join(
            f"{mean(results[arm][p]['shared']):>9.1f}%" for p in PHASES)
        print(f"{arm:>9} {cells}")
    print("=" * 78)

    first = PHASES[0]
    print(f"\nEARLY MATERIAL, contested events in phase '{first}' "
          f"(last seen {PHASE_STEPS * (len(PHASES) - 1)} steps ago):")
    for arm in ARMS:
        xs = [x for x in results[arm][first]["contested"] if x is not None]
        print(f"  {arm:>9}: {mean(xs):>6.1f}%   "
              f"(min {min(xs):.1f}, max {max(xs):.1f})")

    off = results["offline"][first]["contested"]
    on = results["online"][first]["contested"]
    gu = results["guarded"][first]["contested"]
    pair = lambda a, b: sum(1 for x, y in zip(a, b)
                            if x is not None and y is not None and x > y)

    print(f"\n  online  minus offline: {mean(on) - mean(off):+6.1f}  "
          f"(wins {pair(on, off)}/{len(SEEDS)})")
    print(f"  guarded minus offline: {mean(gu) - mean(off):+6.1f}  "
          f"(wins {pair(gu, off)}/{len(SEEDS)})")
    print(f"  guarded minus online : {mean(gu) - mean(on):+6.1f}  "
          f"(wins {pair(gu, on)}/{len(SEEDS)})")

    print("""
The contested column for the first phase is the real measurement. Those are
the moments where the early rules said one thing and the later rules said
another, and only a system that retained the early rules can get them right.

The shared column should be high for every trained arm. If it is not,
something is broken rather than forgotten.

  guarded or online above offline on contested/first -> continual learning
      retains contradicted early material better than a training run
  offline still ahead -> it wins by seeing all phases mixed in every epoch,
      which is not retention but does beat it here
  everyone near chance on contested -> the early rules are simply gone, and
      the previous 96% was entirely the shared events
""")
    with open("milestone3.json", "w") as f:
        json.dump(dict(results=results, grads=grads), f, indent=2)
    print("wrote milestone3.json")