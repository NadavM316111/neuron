"""Can a better replay policy close the gap to offline training?

Milestone 3 from the build plan: beat an offline-trained equivalent on early
material. Best so far is 56.4% against offline's 68.2%, on contested events
in a phase whose rules were contradicted twice.

The unexploited lever: the rehearsal buffer holds 500 runs and samples
UNIFORMLY. The anchor fills in the first hundred moments and never changes,
but the buffer churns constantly, so over a long life early material gets
crowded out and replay is dominated by the recent past — which the live
stream is already teaching.

Three policies, everything else identical:
  uniform      sample at random. The current design.
  old          favour what was stored longest ago.
  surprising   favour what the model currently predicts worst. The surprise
               gate applied to replay rather than intake.

  any policy above offline's 68.2% -> milestone 3 passes
  old or surprising above uniform  -> the buffer was the bottleneck and this
      is a cheap improvement worth keeping
  no difference -> uniform sampling was fine and the gap is elsewhere
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
SEQ_LEN = 10
SEQ_COUNT = 2


def obs_dim():
    """No has_key. The model must remember it."""
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

    def update_batch(self, items):
        """Offline only. A shuffled batch has no coherent history, so no
        hidden state is carried."""
        self.net.train()
        x = torch.stack([encode(i[0], i[1]) for i in items])
        y = torch.tensor([EVENTS.index(i[2]) for i in items])
        logits, _ = self.net(x, None)
        loss = F.cross_entropy(logits, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.grad_steps += len(items)
        return float(loss.item())

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


ARMS = ["offline", "uniform", "old", "surprising"]


def run(mode, seed, life, tests):
    b = Backend(seed)
    t0 = time.time()
    stats = {}

    if mode == "offline":
        rng = random.Random(seed)
        for _ in range(OFFLINE_EPOCHS):
            order = list(life)
            rng.shuffle(order)
            for i in range(0, len(order) - OFFLINE_BATCH, OFFLINE_BATCH):
                b.update_batch(order[i:i + OFFLINE_BATCH])
    else:
        layer = StabilityLayer(
            b, canary=life[:40], seed=seed,
            window=200, warmup=30, top_fraction=0.50,
            coherence_veto=1e9, loss_floor=0.02, steps_per_update=1,
            rehearse_per_item=5, rehearse_count=SEQ_COUNT, rehearse_steps=1,
            anchor_size=100, buffer_size=500, sequence_len=SEQ_LEN,
            replay_policy=mode, replay_pool=8,
            guard=True, guard_per_item=500, canary_tolerance=0.5)
        for i, item in enumerate(life):
            if i % EPISODE == 0:
                b.reset_state()
            layer.observe(item)
        stats = layer.summary()
        del layer

    b.reset_state()
    scores = {}
    for r in PHASES:
        con, sha = b.evaluate_split(tests[r])
        scores[r] = dict(contested=con, shared=sha)
    out = dict(scores=scores, grad_steps=b.grad_steps,
               rehearsals=stats.get("rehearsals", 0),
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
    print("recurrent network, has_key HIDDEN, sequence replay, "
          f"runs of {SEQ_LEN}")
    print("comparing three ways of choosing WHAT to replay\n")

    results = {a: {p: dict(contested=[], shared=[]) for p in PHASES}
               for a in ARMS}
    grads = {a: [] for a in ARMS}
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
            grads[arm].append(r["grad_steps"])
            mins[arm].append(r["minutes"])
        print(f"  seed {seed} done")

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    print("\n" + "=" * 82)
    print("CONTESTED events — where remembering the phase matters")
    print(f"{'arm':>11} " + "  ".join(f"{p:>10}" for p in PHASES) +
          f" {'grad':>8} {'min':>6}")
    print(f"{'':>11} {'(1st)':>10}  {'':>10}  {'(last)':>10}")
    print("-" * 82)
    for arm in ARMS:
        cells = "  ".join(
            f"{mean(results[arm][p]['contested']):>9.1f}%" for p in PHASES)
        print(f"{arm:>11} {cells} {mean(grads[arm]):>8.0f} "
              f"{mean(mins[arm]):>6.1f}")
    print("=" * 82)

    print("\nSHARED events — identical under every rule set")
    for arm in ARMS:
        cells = "  ".join(
            f"{mean(results[arm][p]['shared']):>9.1f}%" for p in PHASES)
        print(f"{arm:>11} {cells}")

    first = PHASES[0]
    print(f"\nMILESTONE 3: contested events in the earliest phase "
          f"'{first}'")
    for arm in ARMS:
        xs = [x for x in results[arm][first]["contested"] if x is not None]
        print(f"  {arm:>11}: {mean(xs):>6.1f}%   "
              f"(min {min(xs):.1f}, max {max(xs):.1f})")

    off = results["offline"][first]["contested"]
    wins = lambda a, b: sum(1 for x, y in zip(a, b)
                            if x is not None and y is not None and x > y)
    print()
    for arm in ["uniform", "old", "surprising"]:
        a = results[arm][first]["contested"]
        d = mean(a) - mean(off)
        flag = "  MILESTONE 3 PASSED" if d > 0 else ""
        print(f"  {arm:>11} minus offline: {d:+6.1f}  "
              f"(wins {wins(a, off)}/{len(SEEDS)}){flag}")

    uni = results["uniform"][first]["contested"]
    print()
    for arm in ["old", "surprising"]:
        a = results[arm][first]["contested"]
        print(f"  {arm:>11} minus uniform: "
              f"{mean(a) - mean(uni):+6.1f}  (wins {wins(a, uni)}/{len(SEEDS)})")

    print("""
Two questions at once.

  Against OFFLINE: does any policy pass milestone 3? Offline sees every
      phase interleaved in every epoch, so it never has to retain anything.
      Beating it means online learning genuinely holds onto contradicted
      material better than a training run does.

  Against UNIFORM: was the buffer the bottleneck? If "old" or "surprising"
      beats it, choosing what to replay matters and this is a cheap
      improvement. If not, uniform sampling was fine and the remaining gap
      is somewhere else.

Note "surprising" costs extra forward passes to rank candidates, so watch
the minutes column as well as the score.
""")
    with open("replay_policy.json", "w") as f:
        json.dump(dict(results=results, grads=grads), f, indent=2)
    print("wrote replay_policy.json")