"""Does retention hold over a genuinely long life?

Every retention result in this project is short. The 7B test was 144
sentences. The weather runs were one pass through four years. The longest
run ever done here was 100,000 steps, and it ended with sequence replay
still improving — which says it had not yet stopped, not that it never
would.

The August language work found something worse that was never resolved: even
the protected arm showed SLOW EROSION of early material, from +1.14 down to
+0.33 over 300 items. The collapse was fixed. The erosion was not.

That is the honest gap in the strongest claim this project makes. The
mechanism works. Whether it works FOREVER is untested, and forever is what a
system that learns continuously has to survive.

FIXED FROM THE FIRST ATTEMPT, which was void. Phases were 10,000 steps with
three rule sets, so a full cycle was 30,000 — but checkpoints were every
25,000. Those do not divide, so every checkpoint landed at a different
position in the cycle. Adjacent checkpoints swung by 38 points while the
trend being measured was 4.6. The aliasing was eight times the signal.

Now the checkpoint interval IS the cycle length, so every measurement is
taken at the same point in the phase rotation and the numbers are
comparable across time.

Three outcomes, all worth knowing:

  IT HOLDS      retention flat or rising at half a million steps. The
                strongest claim available here, and what "never graduates"
                requires.
  SLOW EROSION  gradual decline. A deployment needs periodic intervention,
                and this says how often.
  COLLAPSE      failure at some point. Knowing where beats assuming it does
                not happen.
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
CYCLE = PHASE_STEPS * len(RULES)      # one full rotation through the rules
CYCLES = 20
TOTAL_STEPS = CYCLE * CYCLES          # 480,000
CHECKPOINT = CYCLE                    # measure once per rotation, always at
                                      # the same point in the cycle
EPISODE = 200
HIDDEN = 128
LR = 3e-4
SEED = 0
N_TERRAIN = 5
PATCH = 9
TEST_STEPS = 3000
SEQ_LEN = 10
SEQ_COUNT = 2


def obs_dim():
    """has_key is HIDDEN, so the keyed rule genuinely requires memory."""
    return PATCH * N_TERRAIN + len(ACTIONS)


def encode(obs, action):
    v = torch.zeros(obs_dim())
    for i, cell in enumerate(obs["patch"]):
        v[i * N_TERRAIN + cell] = 1.0
    v[PATCH * N_TERRAIN + ACTIONS.index(action)] = 1.0
    return v.unsqueeze(0)


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
        self._saved = None

    def begin_sequence(self):
        self._saved = self.h
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
        self._saved = None

    def snapshot(self):
        return {k: v.detach().clone()
                for k, v in self.net.state_dict().items()}

    def restore(self, state):
        self.net.load_state_dict(state)

    def on_rollback(self):
        for g in self.opt.param_groups:
            g["lr"] = max(1e-6, g["lr"] * 0.5)


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
    """Accuracy at locked doors under every rule set.

    Scoring only the current phase would hide forgetting entirely — the
    model is always good at what it just saw.
    """
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


def run(name):
    b = Backend(SEED)
    layer = None
    if name == "guarded":
        warm = walk(10, 1, RULES[0], 200)
        layer = StabilityLayer(
            b, canary=warm[:40], seed=SEED,
            window=200, warmup=30, top_fraction=0.50,
            coherence_veto=1e9, loss_floor=0.02, steps_per_update=1,
            rehearse_per_item=5, rehearse_count=SEQ_COUNT,
            rehearse_steps=1, anchor_size=100, buffer_size=500,
            sequence_len=SEQ_LEN, replay_policy="uniform",
            guard=True, guard_per_item=2000, canary_tolerance=0.5)

    tests = {r: walk(7000 + j, 80000 + j, r, TEST_STEPS)
             for j, r in enumerate(RULES)}

    curve = []
    t0 = time.time()
    step = 0

    for cyc in range(CYCLES):
        # one full rotation through every rule set, so each checkpoint
        # measures the same point in the cycle
        for rules in RULES:
            chunk = walk(10, SEED * 100000 + step, rules, PHASE_STEPS)
            for item in chunk:
                if step % EPISODE == 0:
                    b.reset_state()
                if layer is not None:
                    layer.observe(item)
                else:
                    b.update(item, 1)
                step += 1

        b.reset_state()
        scores = evaluate(b.net, tests)
        st = layer.summary() if layer else {}
        mean_all = sum(v for v in scores.values()
                       if v is not None) / len(scores)
        curve.append(dict(cycle=cyc + 1, step=step, scores=scores,
                          mean=mean_all,
                          rollbacks=st.get("rollbacks", 0),
                          mins=(time.time() - t0) / 60))
        per = "  ".join(f"{r[:5]} {scores[r]:5.1f}" for r in RULES)
        print(f"    cycle {cyc + 1:>2}/{CYCLES}  step {step:>7}  "
              f"mean {mean_all:5.2f}%   {per}   "
              f"rb {st.get('rollbacks', 0)}  "
              f"{(time.time() - t0) / 60:5.1f}m", flush=True)

    del b, layer
    return curve


if __name__ == "__main__":
    print("Does retention hold over a genuinely long life?\n")
    print(f"{TOTAL_STEPS} steps: {CYCLES} full rotations through {RULES}")
    print(f"phases of {PHASE_STEPS}, cycle of {CYCLE}, measured once per "
          f"cycle")
    print("so every checkpoint is taken at the SAME point in the "
          "rotation.\n")
    print("The previous attempt checkpointed every 25,000 steps against a "
          "30,000-step\ncycle. The aliasing was eight times the trend and "
          "the run was void.\n")

    arms = {}
    for name in ["online", "guarded"]:
        print(f"--- {name} ---", flush=True)
        arms[name] = run(name)
        print()

    print("=" * 88)
    print("MEAN ACCURACY ACROSS ALL THREE RULE SETS, per cycle")
    print(f"{'cycle':>7} {'step':>9} {'online':>10} {'guarded':>10} "
          f"{'difference':>12}")
    print("-" * 88)
    n = min(len(arms["online"]), len(arms["guarded"]))
    for i in range(n):
        o = arms["online"][i]["mean"]
        g = arms["guarded"][i]["mean"]
        print(f"{i + 1:>7} {arms['online'][i]['step']:>9} "
              f"{o:>9.2f}% {g:>9.2f}% {g - o:>+11.2f}")
    print("=" * 88)

    def quarter(curve, first=True):
        k = max(1, len(curve) // 4)
        part = curve[:k] if first else curve[-k:]
        return sum(p["mean"] for p in part) / len(part)

    print("\nFIRST QUARTER against LAST QUARTER:")
    for name in ["online", "guarded"]:
        f, l = quarter(arms[name]), quarter(arms[name], False)
        print(f"  {name:>8}: {f:6.2f}% -> {l:6.2f}%  ({l - f:+6.2f})")

    gf, gl = quarter(arms["guarded"]), quarter(arms["guarded"], False)
    of, ol = quarter(arms["online"]), quarter(arms["online"], False)
    gd, od = gl - gf, ol - of

    print(f"\nDIVERGENCE: guarded minus online went "
          f"{gf - of:+.2f} -> {gl - ol:+.2f}")

    print()
    if gd > 1:
        print("  IT HOLDS AND IMPROVES. Retention is higher at the end of")
        print(f"  {TOTAL_STEPS} steps than at the start. That is the "
              f"strongest claim")
        print("  this project can make, and it is what 'never graduates'")
        print("  actually requires.")
    elif gd > -2:
        print("  IT HOLDS. Retention is essentially flat across "
              f"{TOTAL_STEPS} steps.")
        print("  Not improving, but the slow erosion seen in the August")
        print("  language work does not appear here.")
    else:
        print(f"  SLOW EROSION, {gd:.2f} points over the run. The mechanism")
        print("  delays forgetting rather than preventing it. A deployment")
        print("  would need periodic intervention.")

    rb = arms["guarded"][-1]["rollbacks"]
    print(f"\n  the canary guard fired {rb} times over {TOTAL_STEPS} steps")

    print("""
The comparison against 'online' matters as much as the absolute curve. In
the 100,000-step run, unprotected learning DEGRADED — 56.3% in the first
half to 54.3% in the second — while sequence replay improved. If that
divergence widens here, the mechanism is not merely delaying the problem.

Watch the rollback count. It has been zero in every grid-world run so far.
If the guard fires at this length, something happens late that has never
been observed.
""")
    with open("longlife2.json", "w") as f:
        json.dump(arms, f, indent=2)
    print("wrote longlife2.json")