"""How far back can it actually remember?

capacity.py found the limit is not conjunction and not width. Per-fork
advantage tracked the DURATION of the hidden variable almost perfectly:
wet at 30 steps gave +18.4, fed at 60 gave +9.1, lamp at 90 gave +0.4, and
the two long-lived ones gave nothing. The decisive case was the dark cell,
a SIMPLE one-variable fork that failed exactly like the conjunctions purely
because its variable lasted 90 steps.

But that was inferred across six forks that differ in several ways at once.
This holds everything else fixed and varies ONLY the lookback distance.

One rule, one variable, one fork:

  a switch cell arms a door for D steps
  a door cell opens if the switch was hit within D steps, otherwise not

D is the only thing that changes. Cell counts are adjusted per D so the
fork stays near 50/50, which the fairness check verifies before training.

Two arms per horizon:
  online    must remember. The measurement.
  ceiling   sees the armed flag. The control, showing the task is learnable
            at that D and the fork is fair.

  a clean falloff -> the horizon limit is established directly, and the
      number where it falls is the thing to attack
  flat and high across all D -> the earlier reading was wrong and something
      other than distance explains the six forks
  flat and low across all D -> even short horizons do not work here, and
      this test is broken rather than informative
"""

import json
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


HORIZONS = [15, 30, 60, 120, 240]
STEPS = 40000
EPISODE = 500
HIDDEN = 128            # capacity.py: 128 was as good as 512 here
LR = 3e-4
SEEDS = [0, 1]
TEST_STEPS = 8000
SIZE = 15

EMPTY, WALL, SWITCH, DOOR = range(4)
SYMBOL = {EMPTY: ".", WALL: "#", SWITCH: "S", DOOR: "D"}
ACTIONS = ["up", "down", "left", "right"]
DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
EVENTS = ["moved", "blocked", "armed", "opened", "shut"]

N_TERRAIN = 4
PATCH = 9

# Longer horizons mean the armed state is on more of the time, so the fork
# drifts toward "always opened". Fewer switches compensates. Tuned so every
# horizon lands near 50/50, which the check below verifies.
SWITCHES = {15: 34, 30: 20, 60: 11, 120: 6, 240: 3}
N_DOORS = 26


class SwitchWorld:
    """One thing to remember: was the switch hit within the last D steps?

    Deliberately minimal. No conjunctions, no competing variables, no
    terrain that does anything else. If memory fails here it fails on the
    simplest possible version of the problem.
    """

    def __init__(self, seed=0, horizon=30):
        self.horizon = horizon
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
        n_sw = SWITCHES[self.horizon]
        for r, c in spots[:n_sw]:
            self.grid[r][c] = SWITCH
        for r, c in spots[n_sw:n_sw + N_DOORS]:
            self.grid[r][c] = DOOR
        for r, c in spots[n_sw + N_DOORS:n_sw + N_DOORS + 18]:
            self.grid[r][c] = WALL

    def reset(self):
        self.r, self.c = self.start
        self.armed_until = -1
        self.steps = 0
        return self.observe()

    @property
    def armed(self):
        return self.steps < self.armed_until

    def observe(self, hide_state=True):
        """A 3x3 patch of terrain. hide_state=False adds the armed flag,
        which is the ceiling arm reading what the other must remember."""
        patch = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r, c = self.r + dr, self.c + dc
                patch.append(self.grid[r][c]
                             if 0 <= r < self.h and 0 <= c < self.w else WALL)
        obs = dict(patch=patch)
        if not hide_state:
            obs["state"] = [1 if self.armed else 0]
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

        if cell == SWITCH:
            self.armed_until = self.steps + self.horizon
            self.r, self.c = r, c
            return self.observe(), "armed"

        if cell == DOOR:
            # THE MEASUREMENT. Depends only on how long ago the switch was
            # hit, and nothing else in the world.
            if self.armed:
                self.r, self.c = r, c
                return self.observe(), "opened"
            return self.observe(), "shut"

        self.r, self.c = r, c
        return self.observe(), "moved"


def walk(seed, walk_seed, horizon, n, see_state, episode=EPISODE):
    rng = random.Random(walk_seed)
    world = SwitchWorld(seed, horizon)
    out = []
    for i in range(n):
        if i % episode == 0:
            world.reset()
        obs = world.observe(hide_state=not see_state)
        action = rng.choice(ACTIONS)
        _, event = world.step(action)
        out.append((obs, action, event))
    return out


def guess_rate(horizon, seeds=range(3), n=12000):
    """What a memoryless guesser scores at the door."""
    a = b = 0
    for s in seeds:
        for _, _, e in walk(s, 5000 + s, horizon, n, False):
            if e == "opened":
                a += 1
            elif e == "shut":
                b += 1
    return 100.0 * max(a, b) / (a + b) if (a + b) else 100.0, a, b


def obs_dim(see_state):
    return PATCH * N_TERRAIN + len(ACTIONS) + (1 if see_state else 0)


def encode(obs, action, see_state):
    v = torch.zeros(obs_dim(see_state))
    for i, cell in enumerate(obs["patch"]):
        v[i * N_TERRAIN + cell] = 1.0
    i = PATCH * N_TERRAIN
    v[i + ACTIONS.index(action)] = 1.0
    if see_state:
        v[i + len(ACTIONS)] = float(obs["state"][0])
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


def run(horizon, see_state, seed):
    train = walk(seed, seed * 100 + 1, horizon, STEPS, see_state)
    test = walk(seed, 90000 + seed, horizon, TEST_STEPS, see_state)

    net = GRU(see_state, seed)
    opt = torch.optim.Adam(net.parameters(), lr=LR)

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
    with torch.no_grad():
        for i, item in enumerate(test):
            if i % EPISODE == 0:
                h = None
            x = encode(item[0], item[1], see_state).unsqueeze(0)
            logits, h = net(x, h)
            if item[2] in ("opened", "shut"):
                seen += 1
                hit += int(logits.argmax(1).item()) == EVENTS.index(item[2])
    del net, opt
    return 100.0 * hit / seen if seen else None


if __name__ == "__main__":
    print("is the door fork fair at every horizon?\n")
    print(f"  {'horizon':>8} {'opened':>8} {'shut':>8} {'guesser':>9}")
    print("  " + "-" * 38)
    base = {}
    bad = []
    for d in HORIZONS:
        g, a, b = guess_rate(d)
        base[d] = g
        flag = ""
        if g > 75:
            bad.append(d)
            flag = "  <-- unfair"
        print(f"  {d:>8} {a:>8} {b:>8} {g:>8.1f}%{flag}")
    if bad:
        print(f"\n  ABORT: horizons {bad} are too predictable. Adjust "
              f"SWITCHES for those.")
        raise SystemExit
    print("\n  all fair\n")

    print(f"{STEPS} steps, {HIDDEN} units, {len(SEEDS)} seeds")
    print("one rule, one variable, one fork. Only the lookback distance "
          "changes.\n")

    results = {}
    t0 = time.time()
    for d in HORIZONS:
        on = [run(d, False, s) for s in SEEDS]
        ce = [run(d, True, s) for s in SEEDS]
        results[d] = dict(online=on, ceiling=ce, base=base[d])
        m = lambda xs: sum(xs) / len(xs)
        print(f"  horizon {d:>4}: online {m(on):5.1f}%  "
              f"ceiling {m(ce):5.1f}%  guess {base[d]:5.1f}%  "
              f"advantage {m(on) - base[d]:+5.1f}  "
              f"({time.time() - t0:.0f}s)")

    print("\n" + "=" * 66)
    print("ADVANTAGE OVER GUESSING, by how far back it must remember")
    print(f"{'horizon':>8} {'online':>9} {'ceiling':>9}   curve")
    print("-" * 66)
    for d in HORIZONS:
        r = results[d]
        m = lambda xs: sum(xs) / len(xs)
        on = m(r["online"]) - r["base"]
        ce = m(r["ceiling"]) - r["base"]
        bar = "#" * max(0, int(on))
        print(f"{d:>8} {on:>+8.1f} {ce:>+8.1f}   {bar}")
    print("=" * 66)

    print("""
The online column against horizon is the curve.

  it falls off around 60-90 -> the horizon limit is confirmed directly, and
      the number where it falls is the thing to attack. The likely cause is
      truncated backpropagation: the hidden state is detached every step, so
      gradients cannot assign credit to anything further back than one step,
      and whatever the network retains beyond that is incidental.
  flat and high everywhere  -> distance is not the constraint and the big
      world's six forks differed in some other way.
  flat and low everywhere   -> even 15 steps fails here, so this test is
      broken rather than informative. Check the ceiling column: if it is
      also low, the fork is not learnable at all.
""")
    with open("horizon.json", "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print("wrote horizon.json")