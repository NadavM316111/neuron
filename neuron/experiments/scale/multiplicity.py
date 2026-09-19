"""Is the limit how MANY things it has to remember at once?

Three hypotheses dead, all inferred from the big world and all killed by
clean tests:
  conjunction  the ceiling handles A-AND-B as well as A. Refuted.
  distance     +30 advantage even at 240 steps back. Refuted.
  frequency    93% accuracy with the cause at 0.63% of the stream. Refuted.

Given ONE thing to remember, the system is excellent and almost nothing
about the task matters. The big world asked it to track SIX at once, and it
failed there. Multiplicity is what is left.

This tests it directly. N independent switch/door pairs, each with its own
terrain type, its own armed flag, and no interaction between them. Every
pair behaves exactly like the single pair that worked at 95%. The only
thing that changes is how many run at the same time.

  N = 1, 2, 4, 8

Doors are shared out so the TOTAL number of door tests stays roughly
constant. Otherwise N would also change how much testing happens, which is
the confound that wasted three hypotheses.

Two arms:
  online    must remember all N flags
  ceiling   sees all N flags. Shows the task stays learnable as N grows, so
            a drop in the online arm is about REMEMBERING several things,
            not about the problem getting harder in general.

  online falls as N rises while ceiling holds -> concurrent memory tasks
      interfere. That is the limit, and it is the one that matters for a
      real life, where dozens of things are true at once.
  both hold -> multiplicity is not it either, and the difference between
      the clean world and the big world is something else again:
      input width, number of outcomes, or terrain variety.
"""

import json
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


HORIZON = 60
N_PAIRS = [1, 2, 4, 8]
TOTAL_SWITCHES = 40        # shared across pairs, so supply stays constant
TOTAL_DOORS = 32           # shared across pairs, so testing stays constant
STEPS = 50000
EPISODE = 500
HIDDEN = 128
LR = 3e-4
SEEDS = [0, 1]
TEST_STEPS = 10000
SIZE = 17

EMPTY, WALL = 0, 1
FIRST_SWITCH = 2           # terrain 2..2+N-1 are switches
MAX_PAIRS = 8

ACTIONS = ["up", "down", "left", "right"]
DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

# Events are shared across pairs: which door opened matters less than
# whether it opened, and merging keeps the output space constant as N grows.
EVENTS = ["moved", "blocked", "armed", "opened", "shut"]

N_TERRAIN = 2 + 2 * MAX_PAIRS      # empty, wall, then a switch and door
PATCH = 9                          # per pair


def switch_id(i):
    return FIRST_SWITCH + i


def door_id(i):
    return FIRST_SWITCH + MAX_PAIRS + i


class MultiWorld:
    """N independent things to remember, each identical to the single one
    that worked at 95%.

    Pair i has its own switch terrain, its own door terrain, and its own
    armed flag. Nothing couples them. If tracking eight is harder than
    tracking one, it is not because the rules got harder.
    """

    def __init__(self, seed=0, n_pairs=1):
        self.n = n_pairs
        self.rng = random.Random(seed)
        self.h = self.w = SIZE
        self._build()
        self.reset()

    def _build(self):
        self.grid = [[WALL if (r in (0, self.h - 1) or c in (0, self.w - 1))
                      else EMPTY for c in range(self.w)]
                     for r in range(self.h)]
        inner = [(r, c) for r in range(1, self.h - 1)
                 for c in range(1, self.w - 1)]
        self.start = (self.h // 2, 1)
        spots = [p for p in inner if p != self.start]
        self.rng.shuffle(spots)

        per_sw = TOTAL_SWITCHES // self.n
        per_dr = TOTAL_DOORS // self.n
        i = 0
        for p in range(self.n):
            for r, c in spots[i:i + per_sw]:
                self.grid[r][c] = switch_id(p)
            i += per_sw
            for r, c in spots[i:i + per_dr]:
                self.grid[r][c] = door_id(p)
            i += per_dr
        for r, c in spots[i:i + 20]:
            self.grid[r][c] = WALL

    def reset(self):
        self.r, self.c = self.start
        self.armed_until = [-1] * self.n
        self.steps = 0
        return self.observe()

    def armed(self, p):
        return self.steps < self.armed_until[p]

    def observe(self, hide_state=True):
        """A 3x3 patch of terrain. hide_state=False adds all N armed flags,
        which is the ceiling arm reading what the other must remember."""
        patch = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r, c = self.r + dr, self.c + dc
                patch.append(self.grid[r][c]
                             if 0 <= r < self.h and 0 <= c < self.w else WALL)
        obs = dict(patch=patch)
        if not hide_state:
            obs["state"] = [1 if self.armed(p) else 0 for p in range(self.n)]
        return obs

    def step(self, action):
        self.steps += 1
        dr, dc = DELTA[action]
        r, c = self.r + dr, self.c + dc

        if not (0 <= r < self.h and 0 <= c < self.w):
            return self.observe(), "blocked"

        cell = self.grid[r][c]

        if cell == WALL:
            return self.observe(), "blocked"

        if cell == EMPTY:
            self.r, self.c = r, c
            return self.observe(), "moved"

        if FIRST_SWITCH <= cell < FIRST_SWITCH + MAX_PAIRS:
            p = cell - FIRST_SWITCH
            self.armed_until[p] = self.steps + HORIZON
            self.r, self.c = r, c
            return self.observe(), "armed"

        p = cell - FIRST_SWITCH - MAX_PAIRS
        if self.armed(p):
            self.r, self.c = r, c
            return self.observe(), "opened"
        return self.observe(), "shut"


def walk(seed, walk_seed, n_pairs, n, see_state, episode=EPISODE):
    rng = random.Random(walk_seed)
    world = MultiWorld(seed, n_pairs)
    out = []
    for i in range(n):
        if i % episode == 0:
            world.reset()
        obs = world.observe(hide_state=not see_state)
        action = rng.choice(ACTIONS)
        _, event = world.step(action)
        out.append((obs, action, event))
    return out


def stream_stats(n_pairs, seeds=range(3), n=12000):
    a = b = armed = total = 0
    for s in seeds:
        for _, _, e in walk(s, 5000 + s, n_pairs, n, False):
            total += 1
            if e == "opened":
                a += 1
            elif e == "shut":
                b += 1
            elif e == "armed":
                armed += 1
    return dict(guess=100.0 * max(a, b) / (a + b) if (a + b) else 100.0,
                tests=a + b, armed=armed,
                test_pct=100.0 * (a + b) / total)


def obs_dim(see_state, n_pairs):
    return PATCH * N_TERRAIN + len(ACTIONS) + (n_pairs if see_state else 0)


def encode(obs, action, see_state, n_pairs):
    v = torch.zeros(obs_dim(see_state, n_pairs))
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
    def __init__(self, see_state, n_pairs, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.enc = nn.Linear(obs_dim(see_state, n_pairs), HIDDEN)
        self.cell = nn.GRUCell(HIDDEN, HIDDEN)
        self.head = nn.Linear(HIDDEN, len(EVENTS))

    def forward(self, x, h=None):
        e = F.relu(self.enc(x))
        h = self.cell(e, h)
        return self.head(h), h


def run(n_pairs, see_state, seed):
    train = walk(seed, seed * 100 + 1, n_pairs, STEPS, see_state)
    test = walk(seed, 90000 + seed, n_pairs, TEST_STEPS, see_state)

    net = GRU(see_state, n_pairs, seed)
    opt = torch.optim.Adam(net.parameters(), lr=LR)

    h = None
    net.train()
    for i, item in enumerate(train):
        if i % EPISODE == 0:
            h = None
        x = encode(item[0], item[1], see_state, n_pairs).unsqueeze(0)
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
    with torch.no_grad():
        for i, item in enumerate(test):
            if i % EPISODE == 0:
                h = None
            x = encode(item[0], item[1], see_state, n_pairs).unsqueeze(0)
            logits, h = net(x, h)
            if item[2] in ("opened", "shut"):
                seen += 1
                hit += int(logits.argmax(1).item()) == EVENTS.index(item[2])
    del net, opt
    return 100.0 * hit / seen if seen else None


if __name__ == "__main__":
    print(f"N independent switch/door pairs, horizon {HORIZON}")
    print(f"{TOTAL_SWITCHES} switches and {TOTAL_DOORS} doors shared across "
          f"the pairs,\nso the amount of testing stays constant as N "
          f"grows\n")

    print(f"  {'pairs':>6} {'tests':>7} {'test%':>7} {'armed':>7} "
          f"{'guess':>7}")
    print("  " + "-" * 40)
    stats = {}
    for n in N_PAIRS:
        s = stream_stats(n)
        stats[n] = s
        print(f"  {n:>6} {s['tests']:>7} {s['test_pct']:>6.2f}% "
              f"{s['armed']:>7} {s['guess']:>6.1f}%")
    print()

    print(f"{STEPS} steps, {HIDDEN} units, {len(SEEDS)} seeds\n")

    results = {}
    t0 = time.time()
    for n in N_PAIRS:
        on = [run(n, False, s) for s in SEEDS]
        ce = [run(n, True, s) for s in SEEDS]
        m = lambda xs: sum(xs) / len(xs)
        results[n] = dict(online=on, ceiling=ce, **stats[n])
        print(f"  {n} pair{'s' if n > 1 else ' '}: "
              f"online {m(on):5.1f}%  ceiling {m(ce):5.1f}%  "
              f"guess {stats[n]['guess']:5.1f}%  "
              f"({time.time() - t0:.0f}s)")

    print("\n" + "=" * 70)
    print("ACCURACY ON THE DOOR FORK, by how many flags must be tracked")
    print(f"{'pairs':>6} {'online':>8} {'ceiling':>8} {'guess':>8} "
          f"{'online-ceiling':>16}")
    print("-" * 70)
    for n in N_PAIRS:
        r = results[n]
        m = lambda xs: sum(xs) / len(xs)
        on, ce = m(r["online"]), m(r["ceiling"])
        print(f"{n:>6} {on:>7.1f}% {ce:>7.1f}% {r['guess']:>7.1f}% "
              f"{on - ce:>15.1f}")
    print("=" * 70)

    print("""
Absolute accuracy, not advantage. The guess baseline moves with N, and the
last experiment showed that reading advantage against a moving baseline
produces a curve that is entirely an artifact.

The online column against N is the answer.

  online falls as N rises while ceiling holds -> CONCURRENT MEMORY TASKS
      INTERFERE. That is the limit. It matters more than any of the three
      already refuted, because a real life has dozens of things true at
      once, not one.

  both hold -> multiplicity is not it either, and what separates the clean
      world from the big world is something duller: input width, number of
      outcomes, or terrain variety. Worth knowing, and it would mean the
      mechanisms are in better shape than the big world suggested.

The ceiling column is the control throughout. It reads every flag directly,
so if it drops the task itself got harder and the comparison is void.
""")
    with open("multiplicity.json", "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print("wrote multiplicity.json")