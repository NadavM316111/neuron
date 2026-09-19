"""Is the limit how OFTEN it sees the cause, not how far back it was?

Two hypotheses down. Conjunctions: refuted, the ceiling handles them fine.
Distance: refuted, horizon.py got +30 advantage even at 240 steps back.

The third comes from the big world's own event counts. Advantage tracked the
ratio of ENABLING events to TEST events almost perfectly:

  ice          soaked 629 seen / 345 tests   ratio 1.82   +18.4
  heavy door   ate    257 / 329              ratio 0.78    +9.1
  locked door  got_key 149 / 351             ratio 0.42    +7.8
  dark cell    got_lamp 95 / 450             ratio 0.21    +0.4
  chasm        got_rope 30 / 211             ratio 0.14    -4.0

Which is the rung-one finding returning: events under about 1% of a stream
get ignored, because ignoring them minimises the loss. It never went away.
It moved from the TESTED events to the ENABLING ones.

This holds everything else fixed and varies only that ratio. Same switch
world, horizon fixed at 60, and the number of switch cells swept from many
to very few. Doors stay constant, so only the supply of causes changes.

  advantage falls as switches get rare -> confirmed. And it is GOOD news:
      a frequency limit is tractable in a way a distance limit is not. You
      cannot make the past closer, but you can make rare things count more,
      which is what a surprise gate is for.
  advantage stays flat -> wrong a third time, and the explanation is
      somewhere else entirely.
"""

import json
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


HORIZON = 60              # fixed. horizon.py showed distance does not matter
SWITCH_COUNTS = [40, 20, 10, 5, 2, 1]
N_DOORS = 26
STEPS = 40000
EPISODE = 500
HIDDEN = 128
LR = 3e-4
SEEDS = [0, 1]
TEST_STEPS = 8000
SIZE = 15

EMPTY, WALL, SWITCH, DOOR = range(4)
ACTIONS = ["up", "down", "left", "right"]
DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
EVENTS = ["moved", "blocked", "armed", "opened", "shut"]

N_TERRAIN = 4
PATCH = 9


class SwitchWorld:
    """One thing to remember: was the switch hit within the last HORIZON
    steps? The only variable is how many switch cells exist."""

    def __init__(self, seed=0, n_switches=20):
        self.n_switches = n_switches
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
        i = self.n_switches
        for r, c in spots[:i]:
            self.grid[r][c] = SWITCH
        for r, c in spots[i:i + N_DOORS]:
            self.grid[r][c] = DOOR
        for r, c in spots[i + N_DOORS:i + N_DOORS + 18]:
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
            self.armed_until = self.steps + HORIZON
            self.r, self.c = r, c
            return self.observe(), "armed"

        if cell == DOOR:
            if self.armed:
                self.r, self.c = r, c
                return self.observe(), "opened"
            return self.observe(), "shut"

        self.r, self.c = r, c
        return self.observe(), "moved"


def walk(seed, walk_seed, n_switches, n, see_state, episode=EPISODE):
    rng = random.Random(walk_seed)
    world = SwitchWorld(seed, n_switches)
    out = []
    for i in range(n):
        if i % episode == 0:
            world.reset()
        obs = world.observe(hide_state=not see_state)
        action = rng.choice(ACTIONS)
        _, event = world.step(action)
        out.append((obs, action, event))
    return out


def stream_stats(n_switches, seeds=range(3), n=12000):
    """Guess rate at the door, and how often the CAUSE is actually seen."""
    a = b = armed = total = 0
    for s in seeds:
        for _, _, e in walk(s, 5000 + s, n_switches, n, False):
            total += 1
            if e == "opened":
                a += 1
            elif e == "shut":
                b += 1
            elif e == "armed":
                armed += 1
    guess = 100.0 * max(a, b) / (a + b) if (a + b) else 100.0
    return dict(guess=guess, opened=a, shut=b, armed=armed,
                armed_pct=100.0 * armed / total,
                ratio=armed / (a + b) if (a + b) else 0.0)


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


def run(n_switches, see_state, seed):
    train = walk(seed, seed * 100 + 1, n_switches, STEPS, see_state)
    test = walk(seed, 90000 + seed, n_switches, TEST_STEPS, see_state)

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
    print(f"horizon fixed at {HORIZON}, doors fixed at {N_DOORS}, "
          f"only the number of switches changes\n")

    print(f"  {'switches':>9} {'armed':>7} {'armed%':>8} "
          f"{'cause/test':>11} {'guess':>7}")
    print("  " + "-" * 48)
    stats = {}
    for n in SWITCH_COUNTS:
        s = stream_stats(n)
        stats[n] = s
        print(f"  {n:>9} {s['armed']:>7} {s['armed_pct']:>7.2f}% "
              f"{s['ratio']:>11.2f} {s['guess']:>6.1f}%")

    unfair = [n for n in SWITCH_COUNTS if stats[n]["guess"] > 80]
    if unfair:
        print(f"\n  NOTE: switch counts {unfair} give a guesser over 80%. "
              f"Their\n  advantage numbers are compressed by a high floor, "
              f"so read the\n  ceiling column there rather than the online "
              f"one alone.")
    print()

    print(f"{STEPS} steps, {HIDDEN} units, {len(SEEDS)} seeds\n")

    results = {}
    t0 = time.time()
    for n in SWITCH_COUNTS:
        on = [run(n, False, s) for s in SEEDS]
        ce = [run(n, True, s) for s in SEEDS]
        m = lambda xs: sum(xs) / len(xs)
        g = stats[n]["guess"]
        results[n] = dict(online=on, ceiling=ce, **stats[n])
        print(f"  {n:>2} switches (cause/test {stats[n]['ratio']:.2f}): "
              f"online {m(on):5.1f}%  ceiling {m(ce):5.1f}%  "
              f"guess {g:5.1f}%  advantage {m(on) - g:+5.1f}  "
              f"({time.time() - t0:.0f}s)")

    print("\n" + "=" * 72)
    print("ADVANTAGE OVER GUESSING, by how often the CAUSE is seen")
    print(f"{'switches':>9} {'cause/test':>11} {'online':>8} {'ceiling':>8}"
          f"   curve")
    print("-" * 72)
    for n in SWITCH_COUNTS:
        r = results[n]
        m = lambda xs: sum(xs) / len(xs)
        on = m(r["online"]) - r["guess"]
        ce = m(r["ceiling"]) - r["guess"]
        bar = "#" * max(0, int(on))
        print(f"{n:>9} {r['ratio']:>11.2f} {on:>+7.1f} {ce:>+7.1f}   {bar}")
    print("=" * 72)

    print("""
The online column against the cause/test ratio is the curve.

  it falls as switches get rare -> CONFIRMED. The limit is how often the
      model observes the cause, not how far back it happened. This is the
      rung-one class-imbalance finding, moved from the tested events to the
      enabling ones.

      And it is tractable. You cannot make the past closer, but you can make
      rare causes count more: weight them in the gate, or store them
      preferentially in the rehearsal buffer. That would be the existing
      mechanisms attacking a limit that has now been measured.

  it stays flat -> wrong a third time. The big world's six forks differ in
      some other way, and the honest move is to stop inferring from them and
      test one property at a time.

The ceiling column is the control. It reads the armed flag directly, so it
should stay high everywhere. Where it drops, the fork itself has become too
rare to learn at all, and neither arm's number means much.
""")
    with open("ratio.json", "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print("wrote ratio.json")