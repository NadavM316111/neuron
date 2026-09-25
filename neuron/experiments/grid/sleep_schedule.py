"""Does consolidation work better in blocks than spread out?

Same total replay, different shape in time. This is a schedule question and
it has never been asked here.

The interleaved schedule (consolidate every 5 items, forever) was chosen
early and is the only one anything in this repo has ever run. An animal
does the opposite: a long block of living with no consolidation, then a
block of consolidation with no living.

FOUR ARMS, all with the SAME replay budget. The budget is computed rather
than assumed, and reported, because a schedule comparison where one arm
quietly replays more is not a schedule comparison.

  continuous    consolidate every 5 items while the stream arrives. The
                current design and the control.
  nap           wake 500 items, then sleep. 36 short sleeps.
  sleep         wake 2,000 items, then sleep. 9 long sleeps.
  hibernate     wake 6,000 items, then sleep. 3 very long sleeps, one per
                phase boundary, which is the most animal-like shape and
                also the riskiest: 6,000 items of drift with nothing
                protecting earlier material.

WHAT EACH OUTCOME WOULD MEAN:

  sleeping wins     Replay works better when the live stream is not
                    competing for the weights. Consolidation gets to
                    finish instead of being immediately overwritten. That
                    would be a real argument for building an offline phase
                    into the system rather than interleaving.

  continuous wins   Protection has to be continuous because drift is
                    continuous. A long waking block loses material that a
                    later sleep cannot recover, and the interleaved design
                    was right for a reason.

  no difference     Only the total matters, not the shape, which is worth
                    knowing because it means the offline phase can be
                    scheduled for convenience rather than for learning.

  longer sleeps monotonically worse
                    The mechanism visible directly: cost scales with how
                    long the system goes unprotected.

Contiguous storage is on for every arm (20 Sep result). Dreaming is off.
"""

import json
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from world import (GridWorld, ACTIONS, EVENTS, DELTA, LOCKED, KEY,
                   contradiction_check)
from sleeping import SleepLayer


PHASES = ["keyed", "open", "trap"]
PHASE_STEPS = 6000
EPISODE = 200
HIDDEN = 64
LR = 3e-4
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
N_CELLS = 5
TEST_STEPS = 2000
SEQ_LEN = 10
SEQ_COUNT = 2

# The control's schedule. Everything else is matched to its total.
REHEARSE_PER_ITEM = 5
LIFE = PHASE_STEPS * len(PHASES)

# name -> items awake between sleeps
SCHEDULES = {"micro": 25, "small": 100, "nap": 500, "sleep": 2000}
ARMS = ["continuous"] + list(SCHEDULES)


def obs_dim():
    return 9 * N_CELLS + len(ACTIONS)


def encode(obs, action):
    v = torch.zeros(obs_dim())
    for i, cell in enumerate(obs["patch"]):
        v[i * N_CELLS + cell] = 1.0
    v[9 * N_CELLS + ACTIONS.index(action)] = 1.0
    return v


class GRU(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.enc = nn.Linear(obs_dim(), HIDDEN)
        self.cell = nn.GRUCell(HIDDEN, HIDDEN)
        self.head = nn.Linear(HIDDEN, len(EVENTS))

    def forward(self, x, h=None):
        e = F.relu(self.enc(x))
        h = self.cell(e, h)
        return self.head(h), h


class Backend:
    def __init__(self, seed=0):
        self.net = GRU(seed)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.grad_steps = 0
        self.h = None
        self._saved = None

    def begin_sequence(self):
        self._saved = self.h
        self.h = None

    def _xy(self, item):
        return encode(item[0], item[1]).unsqueeze(0), \
            torch.tensor([EVENTS.index(item[2])])

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
            self.h = h.detach()
            self.grad_steps += 1
            last = float(loss.item())
        return last

    def update_sequence(self, items, steps):
        """Replay a run in order with the loss accumulated across it."""
        self.net.train()
        done = 0
        for _ in range(steps):
            h = None
            total = 0.0
            self.opt.zero_grad()
            for item in items:
                x, y = self._xy(item)
                logits, h = self.net(x, h)
                total = total + F.cross_entropy(logits, y)
                done += 1
            (total / max(1, len(items))).backward()
            self.opt.step()
            self.grad_steps += len(items)
        return done

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
            g["lr"] = max(1e-5, g["lr"] * 0.5)

    def evaluate_split(self, test, episode=EPISODE):
        self.net.eval()
        h = None
        hit = {True: 0, False: 0}
        seen = {True: 0, False: 0}
        with torch.no_grad():
            for i, item in enumerate(test):
                if i % episode == 0:
                    h = None
                x, y = self._xy(item)
                logits, h = self.net(x, h)
                c = item[3]
                seen[c] += 1
                if int(logits.argmax(1).item()) == int(y.item()):
                    hit[c] += 1
        return (100.0 * hit[True] / seen[True] if seen[True] else None,
                100.0 * hit[False] / seen[False] if seen[False] else None)


def outcome_under(state, action, rules, grid):
    r, c, keys = state
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
    if cell == 1:
        return "blocked"
    return "moved"


def walk(layout_seed, walk_seed, rules, n, episode=EPISODE, mark=False):
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
            others = [outcome_under(state, action, o, world.grid)
                      for o in PHASES if o != rules]
            out.append((obs, action, event, any(o != event for o in others)))
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


COMMON = dict(
    window=200, warmup=30, top_fraction=0.50,
    coherence_veto=1e9, loss_floor=0.02, steps_per_update=1,
    rehearse_per_item=REHEARSE_PER_ITEM, rehearse_count=SEQ_COUNT,
    rehearse_steps=1,
    anchor_size=100, buffer_size=500, sequence_len=SEQ_LEN,
    replay_policy="uniform", replay_pool=8,
    guard=True, guard_per_item=500, canary_tolerance=0.5)


def run(mode, seed, life, tests):
    """Every arm is a SleepLayer. continuous simply sleeps every 5 items
    for one round, which reproduces the interleaved schedule exactly.

    Building the control from a different class would reintroduce the
    confound the dream rewrite existed to remove.
    """
    b = Backend(seed)
    t0 = time.time()

    layer = SleepLayer(b, canary=life[:40], seed=seed, contiguous=True,
                       **COMMON)

    if mode == "continuous":
        wake, rounds = REHEARSE_PER_ITEM, 1
    else:
        wake = SCHEDULES[mode]
        # Matched budget: the control does one consolidation every
        # REHEARSE_PER_ITEM items, so a block covering `wake` items owes
        # exactly that many.
        rounds = wake // REHEARSE_PER_ITEM

    for i, item in enumerate(life):
        if i % EPISODE == 0:
            b.reset_state()
        layer.observe(item)
        if (i + 1) % wake == 0:
            layer.sleep(rounds)

    stats = layer.summary()
    del layer

    b.reset_state()
    scores = {}
    for r in PHASES:
        con, sha = b.evaluate_split(tests[r])
        scores[r] = dict(contested=con, shared=sha)
    out = dict(scores=scores, grad_steps=b.grad_steps,
               rehearsals=stats.get("rehearsals", 0),
               sleeps=stats.get("sleeps", 0),
               consolidations=stats.get("consolidations", 0),
               rollbacks=stats.get("rollbacks", 0),
               minutes=(time.time() - t0) / 60)
    del b
    return out


if __name__ == "__main__":
    print("do the phases genuinely contradict each other?")
    worst = contradiction_check(SEEDS)
    if worst < 5:
        print(f"\nABORT: only {worst:.1f}% differ at worst.")
        raise SystemExit
    print(f"\n  good, {worst:.1f}% at worst\n")

    print(f"life: {' -> '.join(PHASES)}, {PHASE_STEPS} steps each")
    print("same total replay, spent in different shapes:")
    print(f"  continuous   consolidate every {REHEARSE_PER_ITEM} items "
          f"while living")
    for name, wake in SCHEDULES.items():
        print(f"  {name:11s}  live {wake} items, then "
              f"{wake // REHEARSE_PER_ITEM} consolidations with no input")
    print()

    results = {a: {p: dict(contested=[], shared=[]) for p in PHASES}
               for a in ARMS}
    reh = {a: [] for a in ARMS}
    sleeps = {a: [] for a in ARMS}
    rb = {a: [] for a in ARMS}
    mins = {a: [] for a in ARMS}

    for seed in SEEDS:
        life = build_life(seed)
        tests = build_tests(seed)
        for arm in ARMS:
            r = run(arm, seed, life, tests)
            for p in PHASES:
                results[arm][p]["contested"].append(
                    r["scores"][p]["contested"])
                results[arm][p]["shared"].append(r["scores"][p]["shared"])
            reh[arm].append(r["rehearsals"])
            sleeps[arm].append(r["sleeps"])
            rb[arm].append(r["rollbacks"])
            mins[arm].append(r["minutes"])
        print(f"  seed {seed} done")

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    base = mean(reh["continuous"])

    print("\n" + "=" * 92)
    print("THE PRECONDITION — is the replay budget actually matched?")
    print("=" * 92)
    print(f"  {'arm':>11} {'replays':>10} {'vs control':>11} "
          f"{'sleeps':>8} {'rollbacks':>10}")
    print("-" * 92)
    matched = True
    for arm in ARMS:
        r = mean(reh[arm])
        ratio = r / base if base else 0.0
        if abs(ratio - 1.0) > 0.10:
            matched = False
        print(f"  {arm:>11} {r:>10.0f} {ratio:>10.2f}x "
              f"{mean(sleeps[arm]):>8.0f} {mean(rb[arm]):>10.1f}")
    if not matched:
        print("\n  BUDGETS ARE NOT MATCHED. This is no longer a schedule "
              "comparison and the")
        print("  scores below confound shape with volume. Fix the "
              "arithmetic before reading on.")

    print("\n" + "=" * 92)
    print("CONTESTED events — where remembering the phase matters")
    print(f"{'arm':>11} " + "  ".join(f"{p:>10}" for p in PHASES) +
          f" {'min':>6}")
    print(f"{'':>11} {'(1st)':>10}  {'':>10}  {'(last)':>10}")
    print("-" * 92)
    for arm in ARMS:
        cells = "  ".join(
            f"{mean(results[arm][p]['contested']):>9.1f}%" for p in PHASES)
        print(f"{arm:>11} {cells} {mean(mins[arm]):>6.1f}")
    print("=" * 92)

    print("\nSHARED events — identical under every rule set")
    for arm in ARMS:
        cells = "  ".join(
            f"{mean(results[arm][p]['shared']):>9.1f}%" for p in PHASES)
        print(f"{arm:>11} {cells}")

    first = PHASES[0]
    con = mean(results["continuous"][first]["contested"])

    print("\n" + "=" * 92)
    print(f"DOES SLEEPING HELP? contested events, '{first}'")
    print("=" * 92)
    print(f"  {'arm':>11} {'score':>8} {'vs continuous':>15} "
          f"{'awake between':>15}")
    print(f"  {'continuous':>11} {con:>7.1f}% {'':>15} "
          f"{REHEARSE_PER_ITEM:>15}")
    best = ("continuous", con)
    for name, wake in SCHEDULES.items():
        m = mean(results[name][first]["contested"])
        if m > best[1]:
            best = (name, m)
        print(f"  {name:>11} {m:>7.1f}% {m - con:>+14.1f} {wake:>15}")

    if matched:
        print()
        gap = best[1] - con
        if best[0] == "continuous":
            print(f"  CONTINUOUS PROTECTION WINS. Every sleeping schedule "
                  f"lost to consolidating")
            print(f"  while living. Drift is continuous, so protection has "
                  f"to be: a long waking")
            print(f"  block loses material that a later block of replay "
                  f"cannot recover, however")
            print(f"  much replay it spends. The interleaved design was "
                  f"right for a reason.")
        elif gap >= 3:
            print(f"  SLEEPING WINS: {best[0]} beat continuous by "
                  f"{gap:.1f} points on the same replay")
            print(f"  budget. Consolidation works better when the live "
                  f"stream is not competing")
            print(f"  for the weights, which is an argument for building "
                  f"an offline phase into")
            print(f"  the system rather than interleaving it.")
        else:
            print(f"  NO REAL DIFFERENCE. The best schedule beat "
                  f"continuous by {gap:.1f}, inside seed")
            print(f"  noise. Only the total replay matters, not its shape, "
                  f"which means an offline")
            print(f"  phase can be scheduled for convenience rather than "
                  f"for learning.")

        scores = [mean(results[n][first]["contested"])
                  for n in SCHEDULES]
        if scores == sorted(scores, reverse=True):
            print(f"\n  AND THE COST IS ORDERED: longer waking blocks are "
                  f"monotonically worse")
            print(f"  ({'  >  '.join(f'{s:.1f}' for s in scores)}). The "
                  f"damage scales with how long the")
            print(f"  system goes unprotected, which is the mechanism "
                  f"visible directly.")

    print(f"\n  One world, {len(SEEDS)} seeds, matched replay budget. A "
          f"result about schedule,")
    print(f"  not about sleep in general.")

    with open("sleep_schedule.json", "w") as f:
        json.dump(dict(results=results, rehearsals=reh, sleeps=sleeps),
                  f, indent=2, default=str)
    print("\nwrote sleep_schedule.json")
