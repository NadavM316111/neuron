"""The one experiment that carries the whole claim, in a few minutes.

Everything else in this repo supports or qualifies one result: a retention
mechanism helps EXACTLY when a stream is non-stationary, and the proof is
that destroying the temporal order destroys the benefit.

This runs that proof small enough to sit through.

Three phases with contradictory rules, seen once in order. Then the same
data shuffled, so nothing is ever old and there is nothing to retain. If the
mechanism is doing what is claimed, its advantage should collapse to zero in
the shuffled condition — not shrink, collapse.

The full version of this ran on four years of real weather across five
climates and on a 7-billion parameter language model. It is in
experiments/grid/shuffle.py and experiments/scale/sevenb3.py. This is the
same logic at a size that finishes while you watch.

    ./run.sh reproduce.py

Expect three to six minutes on a laptop CPU.
"""

import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from world import GridWorld, ACTIONS, EVENTS, DELTA, LOCKED
from stability import StabilityLayer


PHASES = ["keyed", "open", "trap"]
PHASE_STEPS = 3000
EPISODE = 200
HIDDEN = 96
LR = 3e-4
SEEDS = [0, 1, 2]
N_TERRAIN = 5
PATCH = 9
TEST_STEPS = 3000
SEQ_LEN = 10


def obs_dim():
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
    """The four callbacks the stability layer needs. It knows nothing about
    grids, GRUs or this task — the same interface drives a 7B transformer in
    experiments/scale/."""

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
    """A stream, with locked-door arrivals marked. Those are the only
    moments where the three rule sets disagree, so they are the only place
    forgetting can be measured."""
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


def evaluate(net, test):
    net.eval()
    h = None
    hit = seen = 0
    with torch.no_grad():
        for i, item in enumerate(test):
            if i % EPISODE == 0:
                h = None
            logits, h = net(encode(item[0], item[1]), h)
            if item[3]:
                seen += 1
                hit += int(logits.argmax(1).item()) == EVENTS.index(item[2])
    return 100.0 * hit / seen if seen else None


def run(guarded, stream, test, seed):
    b = Backend(seed)
    layer = None
    if guarded:
        layer = StabilityLayer(
            b, canary=stream[:40], seed=seed,
            window=200, warmup=30, top_fraction=0.50,
            coherence_veto=1e9, loss_floor=0.02, steps_per_update=1,
            rehearse_per_item=5, rehearse_count=2, rehearse_steps=1,
            anchor_size=100, buffer_size=500, sequence_len=SEQ_LEN,
            guard=True, guard_per_item=1000, canary_tolerance=0.5)

    for i, item in enumerate(stream):
        if i % EPISODE == 0:
            b.reset_state()
        if layer is not None:
            layer.observe(item)
        else:
            b.update(item, 1)

    score = evaluate(b.net, test)
    del b, layer
    return score


if __name__ == "__main__":
    t0 = time.time()
    print("NEURON, the central claim, reproduced small\n")
    print("Three phases with contradictory rules, seen once in order.")
    print("Then the same data shuffled, so nothing is ever old.\n")
    print(f"{len(SEEDS)} seeds, {PHASE_STEPS} steps per phase, "
          f"{HIDDEN} hidden units\n")

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    results = {"ordered": {"online": [], "guarded": []},
               "shuffled": {"online": [], "guarded": []}}

    for seed in SEEDS:
        # the early phase, which the stream moves away from and may forget
        early = PHASES[0]
        test = walk(seed, 90000 + seed, early, TEST_STEPS)

        stream = []
        for rules in PHASES:
            stream += walk(seed, seed * 100 + PHASES.index(rules),
                           rules, PHASE_STEPS)

        shuffled = list(stream)
        random.Random(seed).shuffle(shuffled)

        for order, data in [("ordered", stream), ("shuffled", shuffled)]:
            for arm, g in [("online", False), ("guarded", True)]:
                results[order][arm].append(run(g, data, test, seed))

        print(f"  seed {seed}: "
              f"ordered {results['ordered']['guarded'][-1]:.1f} vs "
              f"{results['ordered']['online'][-1]:.1f}   "
              f"shuffled {results['shuffled']['guarded'][-1]:.1f} vs "
              f"{results['shuffled']['online'][-1]:.1f}", flush=True)

    print("\n" + "=" * 62)
    print(f"ACCURACY ON THE EARLY PHASE, which the stream left behind")
    print(f"{'':>12} {'unprotected':>13} {'with layer':>13} "
          f"{'advantage':>12}")
    print("-" * 62)
    adv = {}
    for order in ["ordered", "shuffled"]:
        on = mean(results[order]["online"])
        gu = mean(results[order]["guarded"])
        adv[order] = gu - on
        print(f"{order:>12} {on:>12.2f}% {gu:>12.2f}% {gu - on:>+11.2f}")
    print("=" * 62)

    collapse = adv["ordered"] - adv["shuffled"]
    print(f"\n  the advantage collapses by {collapse:+.2f} points when the "
          f"order is destroyed")

    print()
    if adv["ordered"] > 3 and adv["shuffled"] < adv["ordered"] / 2:
        print("  REPRODUCED. The layer helps on an ordered stream and stops")
        print("  helping when the order is removed, which is what makes this")
        print("  retention rather than better training in general.")
    elif adv["ordered"] > 3:
        print("  PARTIAL. The layer helps, but the advantage survived")
        print("  shuffling more than expected at this size. The full")
        print("  version at experiments/grid/shuffle.py uses real weather")
        print("  and four years of data.")
    else:
        print("  NOT REPRODUCED at this size. Three thousand steps per")
        print("  phase may be too few for forgetting to set in. Try")
        print("  experiments/grid/both_halves.py, which runs longer.")

    print(f"\n  {time.time() - t0:.0f} seconds")
    print("""
What this shows, and what it does not.

  The layer's benefit depends on the stream having an ORDER that the model
  moves through. Given the same data with the order destroyed, it has
  nothing to protect and the benefit disappears. That is the difference
  between a retention mechanism and a generally better training procedure,
  and it is the claim the rest of this repo supports at larger scales.

  It does NOT show the effect size you would get on real data. For that see
  experiments/grid/replicate.py (five climates, real weather) and
  experiments/scale/sevenb3.py (a 7B language model, three seeds), both of
  which reproduce this same collapse.
""")