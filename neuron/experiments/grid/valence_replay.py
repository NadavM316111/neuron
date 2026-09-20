"""Does a PERSISTENT value beat an instantaneous one?

replay_policy.py asked which stored units to replay and tested three
answers: at random, oldest first, and whatever the model currently predicts
worst. All three read the pool fresh on every consolidation. None of them
remembers anything about a unit between one replay and the next.

This adds a fourth arm. Every stored unit carries a value that is born from
the surprise already measured at intake and updated from the model's current
loss each time the unit is replayed. A unit that stays hard keeps earning
replay; a unit the model has absorbed decays and is evicted before older
material is.

The comparison that matters is valence against SURPRISING, not against
uniform. Both point replay at difficult material. The difference is memory:
surprising rescores from scratch every time and treats a unit that is hard
now and easy later as two unrelated observations; valence accumulates across
the unit's whole life, so consistent difficulty outranks momentary
difficulty.

  valence above surprising  -> persistence is worth something and the
      architecture should carry value rather than recompute it
  valence == surprising     -> instantaneous surprise was already enough
      and there is nothing to remember
  valence below surprising  -> stale values are worse than fresh ones, and
      the decay is either too slow or the whole idea is wrong

Watch the minutes column. Valence pays one extra score() per replay, which
is cheaper than surprising's ranking pass but not free.
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
from valence import ValenceLayer


PHASES = ["keyed", "open", "trap"]
PHASE_STEPS = 6000
EPISODE = 200
HIDDEN = 64
LR = 3e-4
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
N_CELLS = 5
OFFLINE_EPOCHS = 5
OFFLINE_BATCH = 32
TEST_STEPS = 2000
SEQ_LEN = 10
SEQ_COUNT = 2

VALENCE_DECAY = 0.9
VALENCE_FLOOR = 0.0001
VALENCE_POOL = 16


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


ARMS = ["offline", "uniform", "old", "surprising", "valence"]

# Everything except the policy itself is identical across the online arms,
# so any difference is the policy and not the configuration.
COMMON = dict(
    window=200, warmup=30, top_fraction=0.50,
    coherence_veto=1e9, loss_floor=0.02, steps_per_update=1,
    rehearse_per_item=5, rehearse_count=SEQ_COUNT, rehearse_steps=1,
    anchor_size=100, buffer_size=500, sequence_len=SEQ_LEN,
    replay_pool=8,
    guard=True, guard_per_item=500, canary_tolerance=0.5)


def build_layer(mode, backend, life, seed):
    if mode == "valence":
        return ValenceLayer(
            backend, canary=life[:40], seed=seed,
            valence_decay=VALENCE_DECAY, valence_floor=VALENCE_FLOOR,
            valence_pool=VALENCE_POOL,
            replay_policy="uniform",   # unread; _pick is overridden
            **COMMON)
    return StabilityLayer(
        backend, canary=life[:40], seed=seed,
        replay_policy=mode, **COMMON)


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
        layer = build_layer(mode, b, life, seed)
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
               summary=stats,
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
    print("four ways of choosing WHAT to replay, one of them persistent\n")

    results = {a: {p: dict(contested=[], shared=[]) for p in PHASES}
               for a in ARMS}
    grads = {a: [] for a in ARMS}
    mins = {a: [] for a in ARMS}
    vsum = []

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
            if arm == "valence":
                vsum.append(r["summary"])
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
    for arm in ["uniform", "old", "surprising", "valence"]:
        a = results[arm][first]["contested"]
        d = mean(a) - mean(off)
        flag = "  MILESTONE 3 PASSED" if d > 0 else ""
        print(f"  {arm:>11} minus offline: {d:+6.1f}  "
              f"(wins {wins(a, off)}/{len(SEEDS)}){flag}")

    uni = results["uniform"][first]["contested"]
    print()
    for arm in ["old", "surprising", "valence"]:
        a = results[arm][first]["contested"]
        print(f"  {arm:>11} minus uniform: "
              f"{mean(a) - mean(uni):+6.1f}  "
              f"(wins {wins(a, uni)}/{len(SEEDS)})")

    sur = results["surprising"][first]["contested"]
    val = results["valence"][first]["contested"]
    print(f"\n  THE COMPARISON THAT MATTERS")
    print(f"  valence minus surprising: {mean(val) - mean(sur):+6.1f}  "
          f"(wins {wins(val, sur)}/{len(SEEDS)})")

    if vsum:
        print("\nvalence internals, mean over seeds")
        keys = ["stored", "evicted_low", "revalued", "faded",
                "valence_max", "valence_median", "valence_min",
                "valence_at_floor"]
        for k in keys:
            xs = [s.get(k) for s in vsum if s.get(k) is not None]
            if xs:
                print(f"  {k:>18}: {mean(xs):>10.3f}")

    print("""
READ THE INTERNALS BEFORE THE SCORES.

  faded near zero        nothing ever decayed, so value is just intake
      surprise with extra steps and the comparison is meaningless.
  valence_at_floor high  almost everything decayed to the floor, so ranking
      is near-random and the decay is too fast.
  evicted_low near zero  the buffer never overflowed, so the eviction half
      of the mechanism was never exercised.

Any of those three means the arm did not test what it claims to test, and
the score is not evidence either way. Fix the constants and rerun before
drawing a conclusion.
""")
    with open("valence_replay.json", "w") as f:
        json.dump(dict(results=results, grads=grads, valence=vsum),
                  f, indent=2, default=str)
    print("wrote valence_replay.json")
