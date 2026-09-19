"""Should the rehearsal buffer forget?

Over 480,000 steps — twenty rotations through three contradictory rule sets
— sequence replay REVERSED. The protected arm went 43.41% to 39.64% while
the unprotected one went 44.16% to 46.76%. The divergence widened from -0.75
to -7.12. Retention helped at 100,000 steps and hurt at 480,000.

THE LIKELY CAUSE is not rehearsal but WHAT gets rehearsed. The buffer keeps
everything forever. After twenty cycles it holds material from every rule
set at once, so replaying it means training on a mixture that contradicts
itself — "the door opens" and "the door never opens" rehearsed side by side.
Meanwhile the unprotected arm simply tracks the current phase and, after
twenty exposures, settles into a workable compromise.

If that is right, the fix is a buffer that FORGETS.

Four policies, all with identical rehearsal budgets so only the CONTENTS of
the buffer differ:

  keep-all      the current behaviour. Uniform sampling from everything
                ever stored. The control.
  recent        sample only from the newest portion of the buffer, so
                material the stream has genuinely moved past stops being
                replayed.
  age-decay     sample with probability falling off with age, so old
                material fades rather than being cut off. Softer than
                recent, and closer to how forgetting actually behaves.
  conflict      evict a stored item when a newer item contradicts it — same
                observation and action, different outcome. This is the
                targeted version: it removes exactly the material that
                makes the buffer incoherent, and keeps everything else
                forever.

The last one is the interesting one. It is the only policy that
distinguishes "old" from "wrong", and the hypothesis says the problem is
wrongness rather than age.

Run at 240,000 steps rather than 480,000 — ten rotations, past the point
where the reversal was clearly underway (it was visible by cycle 8) and half
the wall clock. Two arms would take six hours; four need the shorter run.
"""

import json
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from world import GridWorld, ACTIONS, EVENTS, DELTA, LOCKED, RULES
from stability import StabilityLayer


PHASE_STEPS = 8000
CYCLE = PHASE_STEPS * len(RULES)
CYCLES = 10
TOTAL_STEPS = CYCLE * CYCLES          # 240,000
EPISODE = 200
HIDDEN = 128
LR = 3e-4
SEED = 0
N_TERRAIN = 5
PATCH = 9
TEST_STEPS = 3000
SEQ_LEN = 10
SEQ_COUNT = 2
RECENT_FRACTION = 0.25                # "recent" samples from the newest 25%


def obs_dim():
    return PATCH * N_TERRAIN + len(ACTIONS)


def encode(obs, action):
    v = torch.zeros(obs_dim())
    for i, cell in enumerate(obs["patch"]):
        v[i * N_TERRAIN + cell] = 1.0
    v[PATCH * N_TERRAIN + ACTIONS.index(action)] = 1.0
    return v.unsqueeze(0)


def signature(item):
    """What identifies a situation, ignoring its outcome.

    Two items with the same signature and different outcomes are a
    contradiction — the same thing happened twice with different results,
    which means a rule changed between them.
    """
    return (tuple(item[0]["patch"]), item[1])


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
        self.h = None

    def begin_sequence(self):
        self.h = None

    def _xy(self, item):
        return encode(item[0], item[1]), \
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
            last = float(loss.item())
        return last

    def reset_state(self):
        self.h = None

    def snapshot(self):
        return {k: v.detach().clone()
                for k, v in self.net.state_dict().items()}

    def restore(self, state):
        self.net.load_state_dict(state)

    def on_rollback(self):
        for g in self.opt.param_groups:
            g["lr"] = max(1e-6, g["lr"] * 0.5)


class ForgettingBuffer:
    """A rehearsal buffer with a policy about what to keep and replay.

    Written here rather than in stability.py because it is an experiment,
    and because the layer's own buffer is load-bearing for every other
    result in the project.
    """

    def __init__(self, policy, capacity=500, seed=0):
        self.policy = policy
        self.capacity = capacity
        self.rng = random.Random(seed)
        self.runs = []                 # each entry: (age_index, [items])
        self.by_sig = {}               # signature -> outcome, newest wins
        self.counter = 0
        self.evicted = 0

    def add(self, run):
        self.counter += 1

        if self.policy == "conflict":
            # A newer item contradicts a stored one when the same situation
            # produced a different outcome. Drop the stale runs, keep the
            # rest — this removes wrongness rather than age.
            drop = set()
            for item in run:
                sig = signature(item)
                prev = self.by_sig.get(sig)
                if prev is not None and prev[1] != item[2]:
                    drop.add(prev[0])
                self.by_sig[sig] = (self.counter, item[2])
            if drop:
                before = len(self.runs)
                self.runs = [r for r in self.runs if r[0] not in drop]
                self.evicted += before - len(self.runs)

        self.runs.append((self.counter, run))
        while len(self.runs) > self.capacity:
            self.runs.pop(0)

    def sample(self, n):
        if not self.runs:
            return []
        if self.policy == "recent":
            k = max(1, int(len(self.runs) * RECENT_FRACTION))
            pool = self.runs[-k:]
            return [self.rng.choice(pool)[1] for _ in range(n)]
        if self.policy == "age-decay":
            # weight by recency, so old material fades instead of being cut
            newest = self.runs[-1][0]
            weights = [1.0 / (1.0 + (newest - age) / 50.0)
                       for age, _ in self.runs]
            total = sum(weights)
            out = []
            for _ in range(n):
                r = self.rng.random() * total
                acc = 0.0
                for w, (_, run) in zip(weights, self.runs):
                    acc += w
                    if acc >= r:
                        out.append(run)
                        break
                else:
                    out.append(self.runs[-1][1])
            return out
        # keep-all and conflict both sample uniformly; conflict differs in
        # what is IN the buffer, not in how it is drawn
        return [self.rng.choice(self.runs)[1] for _ in range(n)]

    def stats(self):
        return dict(size=len(self.runs), evicted=self.evicted)


def walk(layout_seed, walk_seed, rules, n):
    rng = random.Random(walk_seed)
    world = GridWorld(layout_seed, rules=rules)
    out = []
    for i in range(n):
        if i % EPISODE == 0:
            world.reset()
        obs = world.observe()
        action = rng.choice(ACTIONS)
        dr, dc = DELTA[action]
        r, c = world.r + dr, world.c + dc
        at_door = (0 <= r < world.h and 0 <= c < world.w
                   and world.grid[r][c] == LOCKED)
        _, event = world.step(action)
        out.append((obs, action, event, at_door))
    return out


def evaluate(net, tests):
    net.eval()
    out = {}
    with torch.no_grad():
        for rules, test in tests.items():
            h = None
            hit = seen = 0
            for i, item in enumerate(test):
                if i % EPISODE == 0:
                    h = None
                logits, h = net(encode(item[0], item[1]), h)
                if item[3]:
                    seen += 1
                    hit += int(logits.argmax(1).item()) == \
                        EVENTS.index(item[2])
            out[rules] = 100.0 * hit / seen if seen else None
    return out


def run(policy):
    """policy is 'online', or one of the buffer policies.

    Rehearsal is done here rather than through StabilityLayer, so the
    buffer policy is the only thing that varies. The gate is kept simple —
    every item enters the buffer — because the question is about buffer
    CONTENTS, not about selection.
    """
    b = Backend(SEED)
    tests = {r: walk(7000 + j, 80000 + j, r, TEST_STEPS)
             for j, r in enumerate(RULES)}
    buf = None if policy == "online" else ForgettingBuffer(policy, seed=SEED)

    curve = []
    t0 = time.time()
    step = 0
    pending = []

    for cyc in range(CYCLES):
        for rules in RULES:
            chunk = walk(10, SEED * 100000 + step, rules, PHASE_STEPS)
            for item in chunk:
                if step % EPISODE == 0:
                    b.reset_state()
                b.update(item, 1)
                step += 1

                if buf is not None:
                    pending.append(item)
                    if len(pending) >= SEQ_LEN:
                        buf.add(list(pending))
                        pending = []
                    # rehearse at the same rate the layer uses
                    if step % 5 == 0:
                        for run_items in buf.sample(SEQ_COUNT):
                            b.begin_sequence()
                            for it in run_items:
                                b.update(it, 1)
                            b.reset_state()

        b.reset_state()
        scores = evaluate(b.net, tests)
        mean_all = sum(v for v in scores.values()
                       if v is not None) / len(scores)
        st = buf.stats() if buf else {}
        curve.append(dict(cycle=cyc + 1, step=step, scores=scores,
                          mean=mean_all, **st))
        print(f"    cycle {cyc + 1:>2}/{CYCLES}  mean {mean_all:5.2f}%  "
              f"buffer {st.get('size', 0):>3}  "
              f"evicted {st.get('evicted', 0):>4}  "
              f"{(time.time() - t0) / 60:5.1f}m", flush=True)

    del b, buf
    return curve


POLICIES = ["online", "keep-all", "recent", "age-decay", "conflict"]


if __name__ == "__main__":
    print("Should the rehearsal buffer forget?\n")
    print(f"{TOTAL_STEPS} steps, {CYCLES} rotations through {RULES}")
    print("At 480,000 steps the protected arm REVERSED: -3.76 while "
          "unprotected\ngained +2.60. The buffer had filled with "
          "contradictory material.\n")
    print("All policies rehearse at the same rate. Only the CONTENTS "
          "differ.\n")

    arms = {}
    for policy in POLICIES:
        print(f"--- {policy} ---", flush=True)
        arms[policy] = run(policy)
        print()

    print("=" * 88)
    print("MEAN ACCURACY ACROSS ALL RULE SETS, per cycle")
    print(f"{'cycle':>6} " + "  ".join(f"{p[:9]:>9}" for p in POLICIES))
    print("-" * 88)
    for i in range(CYCLES):
        print(f"{i + 1:>6} " + "  ".join(
            f"{arms[p][i]['mean']:>8.2f}%" for p in POLICIES))
    print("=" * 88)

    def half(curve, first=True):
        k = len(curve) // 2
        part = curve[:k] if first else curve[k:]
        return sum(p["mean"] for p in part) / len(part)

    print("\nFIRST HALF against SECOND HALF — the reversal test:")
    for p in POLICIES:
        f, l = half(arms[p]), half(arms[p], False)
        mark = ""
        if p != "online":
            on_d = half(arms["online"], False) - half(arms["online"])
            if l - f > on_d:
                mark = "   BEATS ONLINE'S TREND"
        print(f"  {p:>10}: {f:6.2f}% -> {l:6.2f}%  ({l - f:+6.2f}){mark}")

    on_trend = half(arms["online"], False) - half(arms["online"])
    ka_trend = half(arms["keep-all"], False) - half(arms["keep-all"])

    print(f"\n  online trend   {on_trend:+.2f}")
    print(f"  keep-all trend {ka_trend:+.2f}  "
          f"(the reversal, reproduced at half the length)"
          if ka_trend < on_trend else
          f"  keep-all trend {ka_trend:+.2f}  "
          f"(the reversal did NOT reproduce at this length)")

    best = max((p for p in POLICIES if p != "online"),
               key=lambda p: half(arms[p], False))
    print(f"\n  best forgetting policy at the end: {best} at "
          f"{half(arms[best], False):.2f}%")
    print(f"  online at the end: {half(arms['online'], False):.2f}%")

    print("""
The hypothesis is that the reversal comes from WHAT is rehearsed, not from
rehearsing. The buffer keeps everything, so after many rotations it holds
every side of every contradiction at once.

  a forgetting policy beats keep-all AND beats online -> the reversal is
      fixable, rehearsal is still the right mechanism, and the scope
      condition becomes "rehearse recent material" rather than "rehearsal
      fails over long lives".

  'conflict' wins specifically -> the problem is WRONGNESS rather than AGE.
      That is the sharper result: a buffer should forget what has been
      contradicted, not what is merely old, and everything uncontradicted
      can be kept forever.

  nothing beats online -> rehearsal genuinely does not help over long
      cycling lives, whatever the buffer contains, and the horizon is a
      real limit rather than an implementation detail.

Note this run rehearses directly rather than through StabilityLayer, so the
gate is absent and every item enters the buffer. That isolates the buffer
policy, which is the variable under test.
""")
    with open("forget.json", "w") as f:
        json.dump(arms, f, indent=2, default=str)
    print("wrote forget.json")