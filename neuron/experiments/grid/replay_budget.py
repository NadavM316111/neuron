"""Is the gap to offline a BUDGET problem, or a structural one?

Four replay policies have now been tested on this world. Uniform, old,
surprising and valence all land 12 to 24 points behind offline on contested
events in the earliest phase, and the spread between the four is smaller
than the gap to offline. So WHICH units get replayed is not the bottleneck.

The untested variable is HOW MUCH. Every arm so far ran at
rehearse_per_item=5, rehearse_count=2, which was chosen early and never
swept on this world. The language rehearsal sweep (7 Sep) found the
opposite of what was assumed there: a quarter of the replay budget matched
the default exactly. Nobody has checked which direction this world sits in.

This sweeps budget far past anything practical and asks one question:

  ONLINE REACHES OFFLINE AT SOME BUDGET
      The gap is a budget problem. Milestone 3 is achievable and the lever
      is volume, not selection. Find the knee and stop there.

  ONLINE PLATEAUS BELOW OFFLINE
      The gap is structural. Offline sees every phase interleaved in every
      epoch, so it never has to retain anything at all; it is not solving
      the same problem and is a CEILING rather than a target. That ends
      milestone 3 as stated, which is worth more than chasing it further.

  ONLINE GETS WORSE WITH MORE REPLAY
      Over-rehearsal. The replay itself is now the dominant stream and the
      live one is being crowded out. Also a real finding, and it means the
      current setting is already past optimal.

Budget is swept on FREQUENCY (rehearse_per_item) rather than count, because
the language sweep found that replaying fewer units more often was worse
than the same total spread across more units. Holding count at 2 and
raising frequency follows the direction that worked there.

Policy is held at uniform throughout. It was the best of the four and it
is the cheapest, so any budget effect shows up without selection cost
confounding the wall clock.
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
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
N_CELLS = 5
OFFLINE_EPOCHS = 5
OFFLINE_BATCH = 32
TEST_STEPS = 2000
SEQ_LEN = 10

# rehearse_per_item: a consolidation fires every N items seen. Lower is
# MORE replay. 5 is the setting every previous arm used; 1 is a
# consolidation on every single item, which is roughly 5x the replay the
# system has ever done and well past anything practical.
BUDGETS = [40, 20, 10, 5, 2, 1]
REHEARSE_COUNT = 2


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

    def update_batch(self, items):
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


def run_offline(seed, life, tests):
    b = Backend(seed)
    t0 = time.time()
    rng = random.Random(seed)
    for _ in range(OFFLINE_EPOCHS):
        order = list(life)
        rng.shuffle(order)
        for i in range(0, len(order) - OFFLINE_BATCH, OFFLINE_BATCH):
            b.update_batch(order[i:i + OFFLINE_BATCH])
    b.reset_state()
    scores = {r: dict(zip(("contested", "shared"), b.evaluate_split(tests[r])))
              for r in PHASES}
    out = dict(scores=scores, grad_steps=b.grad_steps, rehearsals=0,
               minutes=(time.time() - t0) / 60)
    del b
    return out


def run_online(per_item, seed, life, tests):
    b = Backend(seed)
    t0 = time.time()
    layer = StabilityLayer(
        b, canary=life[:40], seed=seed,
        window=200, warmup=30, top_fraction=0.50,
        coherence_veto=1e9, loss_floor=0.02, steps_per_update=1,
        rehearse_per_item=per_item, rehearse_count=REHEARSE_COUNT,
        rehearse_steps=1,
        anchor_size=100, buffer_size=500, sequence_len=SEQ_LEN,
        replay_policy="uniform", replay_pool=8,
        guard=True, guard_per_item=500, canary_tolerance=0.5)
    for i, item in enumerate(life):
        if i % EPISODE == 0:
            b.reset_state()
        layer.observe(item)
    stats = layer.summary()
    del layer

    b.reset_state()
    scores = {r: dict(zip(("contested", "shared"), b.evaluate_split(tests[r])))
              for r in PHASES}
    out = dict(scores=scores, grad_steps=b.grad_steps,
               rehearsals=stats.get("rehearsals", 0),
               updates=stats.get("updates", 0),
               minutes=(time.time() - t0) / 60)
    del b
    return out


ARMS = ["offline"] + [f"every_{n}" for n in BUDGETS]


if __name__ == "__main__":
    print("do the phases genuinely contradict each other?")
    worst = contradiction_check(SEEDS)
    if worst < 5:
        print(f"\nABORT: only {worst:.1f}% differ at worst.")
        raise SystemExit
    print(f"\n  good, {worst:.1f}% at worst\n")

    print(f"life: {' -> '.join(PHASES)}, {PHASE_STEPS} steps each")
    print("uniform replay throughout, sweeping HOW MUCH rather than WHICH")
    print(f"consolidate every: {BUDGETS} items  "
          f"(5 is what every previous arm used)\n")

    results = {a: {p: dict(contested=[], shared=[]) for p in PHASES}
               for a in ARMS}
    reh = {a: [] for a in ARMS}
    grads = {a: [] for a in ARMS}
    mins = {a: [] for a in ARMS}

    for seed in SEEDS:
        life = build_life(seed)
        tests = build_tests(seed)
        for arm in ARMS:
            if arm == "offline":
                r = run_offline(seed, life, tests)
            else:
                r = run_online(int(arm.split("_")[1]), seed, life, tests)
            for p in PHASES:
                results[arm][p]["contested"].append(
                    r["scores"][p]["contested"])
                results[arm][p]["shared"].append(r["scores"][p]["shared"])
            reh[arm].append(r["rehearsals"])
            grads[arm].append(r["grad_steps"])
            mins[arm].append(r["minutes"])
        print(f"  seed {seed} done")

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    first = PHASES[0]

    print("\n" + "=" * 86)
    print("CONTESTED events — where remembering the phase matters")
    print(f"{'arm':>11} " + "  ".join(f"{p:>10}" for p in PHASES) +
          f" {'replays':>9} {'grad':>8} {'min':>6}")
    print(f"{'':>11} {'(1st)':>10}  {'':>10}  {'(last)':>10}")
    print("-" * 86)
    for arm in ARMS:
        cells = "  ".join(
            f"{mean(results[arm][p]['contested']):>9.1f}%" for p in PHASES)
        print(f"{arm:>11} {cells} {mean(reh[arm]):>9.0f} "
              f"{mean(grads[arm]):>8.0f} {mean(mins[arm]):>6.1f}")
    print("=" * 86)

    print("\nSHARED events — identical under every rule set")
    for arm in ARMS:
        cells = "  ".join(
            f"{mean(results[arm][p]['shared']):>9.1f}%" for p in PHASES)
        print(f"{arm:>11} {cells}")

    off = mean(results["offline"][first]["contested"])
    print(f"\nTHE CURVE: contested events in the earliest phase '{first}'")
    print(f"{'every':>8} {'replays':>9} {'score':>8} {'vs offline':>12}")
    print("-" * 42)
    best = None
    for n in BUDGETS:
        arm = f"every_{n}"
        m = mean(results[arm][first]["contested"])
        d = m - off
        if best is None or m > best[1]:
            best = (n, m)
        print(f"{n:>8} {mean(reh[arm]):>9.0f} {m:>7.1f}% {d:>+11.1f}")
    print(f"{'offline':>8} {'-':>9} {off:>7.1f}%")

    print()
    if best[1] >= off:
        print(f"  ONLINE REACHED OFFLINE at every_{best[0]} "
              f"({best[1]:.1f}% vs {off:.1f}%).")
        print("  The gap was a BUDGET problem. Milestone 3 is achievable "
              "and volume is the lever.")
    else:
        base = mean(results["every_5"][first]["contested"])
        gained = best[1] - base
        spent = mean(reh[f"every_{best[0]}"]) / max(
            1.0, mean(reh["every_5"]))
        print(f"  ONLINE PLATEAUED at {best[1]:.1f}% (every_{best[0]}), "
              f"still {off - best[1]:.1f} behind offline.")
        print(f"  {spent:.1f}x the replay of the standard setting bought "
              f"{gained:+.1f} points.")
        print("  The gap is STRUCTURAL, not a budget problem. Offline sees "
              "every phase")
        print("  interleaved in every epoch and never has to retain "
              "anything, so it is a")
        print("  ceiling rather than a target. Milestone 3 as stated is the "
              "wrong milestone.")

    print("""
WATCH THE LAST PHASE TOO. If contested accuracy on 'trap' FALLS as replay
rises, the replay stream is crowding out the live one and the system is
over-rehearsing. That would mean the standard setting is already past
optimal, which is the opposite of what this sweep was looking for and
worth more than the answer it was looking for.
""")
    with open("replay_budget.json", "w") as f:
        json.dump(dict(results=results, rehearsals=reh, grads=grads),
                  f, indent=2, default=str)
    print("wrote replay_budget.json")
