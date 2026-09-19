"""Memory and retention together, compute-matched.

The first run showed sequence replay beating single-moment replay by +19.5
points on 3 of 3 seeds. But sequences did 76,651 gradient steps against
moments' 13,212, because replaying a run of 10 performs 10 updates where a
moment performs 1. So part of that win may have been volume.

Fixed here: the moments arm replays 20 isolated moments per consolidation,
sequences replays 2 runs of 10. Same 20 replay updates either way. The ONLY
difference is whether replayed experience keeps its order.

A "moments-x1" arm is kept as well, at the original count of 2, so the
effect of volume alone is visible next to the effect of ordering.

Task: a recurrent network with has_key HIDDEN, so memory is required,
across three phases with contradictory rules, scored on the events where
the phases actually disagree.
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
SEQ_COUNT = 2                 # 2 runs of 10 = 20 replay updates
MOMENT_COUNT = SEQ_LEN * SEQ_COUNT   # 20 isolated moments, matched


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
    """Carries state between steps, so it can remember a key pickup."""

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
    """Recurrent backend behind the stability callbacks.

    begin_sequence() parks the live hidden state and starts clean, so a
    replayed run has coherent context instead of whatever the live stream
    happened to leave behind.
    """

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
        hidden state is carried — which is itself a fair reflection of what
        shuffled training can do with a recurrent model."""
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
        """Contested and shared accuracy, replaying the test stream in order
        so the hidden state means something."""
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
                contested = item[3]
                seen[contested] += 1
                if int(logits.argmax(1).item()) == int(y.item()):
                    hit[contested] += 1
        return (100.0 * hit[True] / seen[True] if seen[True] else None,
                100.0 * hit[False] / seen[False] if seen[False] else None)


def outcome_under(state, action, rules, grid):
    """What WOULD happen from this state under a given rule set."""
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


# arm -> (sequence_len, rehearse_count)
LAYER_ARMS = {
    "moments-x1": (1, 2),                 # original, low volume
    "moments":    (1, MOMENT_COUNT),      # matched volume, no ordering
    "sequences":  (SEQ_LEN, SEQ_COUNT),   # matched volume, with ordering
}


def run(mode, seed, life, tests):
    b = Backend(seed)
    t0 = time.time()
    stats = {}

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
        for i, item in enumerate(life):
            if i % EPISODE == 0:
                b.reset_state()
            b.update(item, 1)

    else:
        seq_len, count = LAYER_ARMS[mode]
        layer = StabilityLayer(
            b, canary=life[:40], seed=seed,
            window=200, warmup=30, top_fraction=0.50,
            coherence_veto=1e9, loss_floor=0.02, steps_per_update=1,
            rehearse_per_item=5, rehearse_count=count, rehearse_steps=1,
            anchor_size=100, buffer_size=500, sequence_len=seq_len,
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


ARMS = ["frozen", "offline", "online", "moments-x1", "moments", "sequences"]


if __name__ == "__main__":
    print("do the phases genuinely contradict each other?")
    worst = contradiction_check(SEEDS)
    if worst < 5:
        print(f"\nABORT: only {worst:.1f}% differ at worst.")
        raise SystemExit
    print(f"\n  good, {worst:.1f}% at worst\n")

    print(f"life: {' -> '.join(PHASES)}, {PHASE_STEPS} steps each")
    print("recurrent network, has_key HIDDEN, so it must remember")
    print(f"replay budget matched: moments does {MOMENT_COUNT} isolated "
          f"moments, sequences does {SEQ_COUNT} runs of {SEQ_LEN}")
    print(f"moments-x1 keeps the original count of 2, to show volume "
          f"separately\n")

    results = {a: {p: dict(contested=[], shared=[]) for p in PHASES}
               for a in ARMS}
    grads = {a: [] for a in ARMS}
    rehs = {a: [] for a in ARMS}

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
            rehs[arm].append(r["rehearsals"])
        print(f"  seed {seed} done")

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    print("\n" + "=" * 84)
    print("CONTESTED events — where remembering the phase matters")
    print(f"{'arm':>11} " + "  ".join(f"{p:>10}" for p in PHASES) +
          f" {'grad':>8} {'reh':>7}")
    print(f"{'':>11} {'(1st)':>10}  {'':>10}  {'(last)':>10}")
    print("-" * 84)
    for arm in ARMS:
        cells = "  ".join(
            f"{mean(results[arm][p]['contested']):>9.1f}%" for p in PHASES)
        print(f"{arm:>11} {cells} {mean(grads[arm]):>8.0f} "
              f"{mean(rehs[arm]):>7.0f}")
    print("=" * 84)

    print("\nSHARED events — identical under every rule set")
    for arm in ARMS:
        cells = "  ".join(
            f"{mean(results[arm][p]['shared']):>9.1f}%" for p in PHASES)
        print(f"{arm:>11} {cells}")

    first = PHASES[0]
    print(f"\nEARLY MATERIAL, contested events in '{first}':")
    for arm in ARMS:
        xs = [x for x in results[arm][first]["contested"] if x is not None]
        print(f"  {arm:>11}: {mean(xs):>6.1f}%   "
              f"(min {min(xs):.1f}, max {max(xs):.1f})")

    seq = results["sequences"][first]["contested"]
    mom = results["moments"][first]["contested"]
    m1 = results["moments-x1"][first]["contested"]
    off = results["offline"][first]["contested"]
    wins = lambda a, b: sum(1 for x, y in zip(a, b)
                            if x is not None and y is not None and x > y)

    print(f"\n  volume alone  (moments minus moments-x1): "
          f"{mean(mom) - mean(m1):+6.1f}  (wins {wins(mom, m1)}/{len(SEEDS)})")
    print(f"  ORDERING      (sequences minus moments)  : "
          f"{mean(seq) - mean(mom):+6.1f}  (wins {wins(seq, mom)}/{len(SEEDS)})")
    print(f"  vs offline    (sequences minus offline)  : "
          f"{mean(seq) - mean(off):+6.1f}  (wins {wins(seq, off)}/{len(SEEDS)})")

    print("""
Two effects are now separated.

  volume alone   what you get from simply replaying more, with no ordering
  ORDERING       what you get from replaying runs instead of snapshots, at
                 the SAME replay budget

  ordering still clearly positive -> replaying trajectories is what lets
      rehearsal protect a model with memory. The two halves work together
      and it is not just extra training.
  ordering near zero -> the first run's +19.5 was volume, and single-moment
      replay is fine as long as you do enough of it. Simpler, and worth
      knowing.
""")
    with open("both_halves.json", "w") as f:
        json.dump(dict(results=results, grads=grads, rehs=rehs), f, indent=2)
    print("wrote both_halves.json")