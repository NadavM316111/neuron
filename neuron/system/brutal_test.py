"""BrutalWorld: a world hard enough that only a big brain can cope.

The earlier hard world plateaued at 256 units, it was not hard enough to need
more. This builds a genuinely BRUTAL world: many more signals, far deeper
nonlinear coupling, and longer-range interactions, so that a small brain
genuinely cannot represent it and a big brain can. Then we run the consolidated
Mind at 64, 256, 1024, 2048 and see whether capability keeps climbing with
size. This is the test that shows how DEEP the scaling goes, and whether the
2048 brain (which only a GPU can run) was worth it.

    python brutal_test.py
"""

import sys
import os
import math
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world_interface import World
from mind import Mind


class BrutalWorld(World):
    """Genuinely hard: many signals, each outcome a deep nonlinear function of
    MANY coupled signals. The structure is rich enough that representing it
    needs real capacity, a small brain cannot; a big brain can."""

    def __init__(self, n_signals=40, coupling=14, seed=0):
        rng = random.Random(seed)
        self.rng = rng
        self._names = [f"sig_{i}" for i in range(n_signals)]
        self.phase = {n: rng.uniform(0, 6.28) for n in self._names}
        self.freq = {n: rng.uniform(0.3, 1.5) for n in self._names}
        self.couples = {n: rng.sample(self._names, coupling) for n in self._names}
        self.cw = {n: [rng.uniform(-1, 1) for _ in range(coupling)]
                   for n in self._names}
        self.coupling = coupling
        self.t = 0
        self._vals = {}
        self._tick()

    def _tick(self):
        for n in self._names:
            self._vals[n] = 0.5 + 0.45 * math.sin(
                self.phase[n] + self.freq[n] * self.t)

    def signals(self):
        return self._names

    def observe(self, name):
        return self._vals.get(name, 0.5)

    def outcome(self, name, past_value):
        if past_value is None:
            return None
        c = self.couples[name]
        w = self.cw[name]
        s = 0.0
        # deep nonlinear interaction across MANY coupled signals
        for k in range(len(c)):
            a = self._vals[c[k]]
            b = self._vals[c[(k + 1) % len(c)]]
            d = self._vals[c[(k + 2) % len(c)]]
            s += w[k] * math.tanh(a * b - d * a)
        if s > 0.15:
            return 2
        if s < -0.15:
            return 0
        return 1

    def step(self):
        self.t += 1
        self._tick()

    def describe(self):
        return f"BrutalWorld({len(self._names)} signals, coupling {self.coupling})"


def run(hidden, steps, seed):
    world = BrutalWorld(n_signals=40, coupling=14, seed=seed)
    mind = Mind(world, hidden=hidden, seed=seed)
    accs = []
    for i in range(steps):
        mind.live_step()
        if i > steps * 0.7:
            accs.append(mind.recent_acc())
    return sum(accs) / len(accs) if accs else 0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    print("  BRUTAL WORLD: 40 signals, deep nonlinear coupling across 14 of")
    print("  them per outcome. Hard enough that only real capacity can cope.")
    print("  Does capability keep climbing as the brain grows, deep into the")
    print("  sizes only a GPU can run?\n")

    sizes = [64, 256, 1024, 2048]
    print("=" * 56)
    print("BRAIN SIZE vs CAPABILITY ON A BRUTAL WORLD")
    print("=" * 56)
    print(f"  {'brain size':>12} {'final accuracy':>16}")
    print("  " + "-" * 32)
    results = {}
    for h in sizes:
        accs = [run(h, args.steps, s) for s in range(args.seeds)]
        m = sum(accs) / len(accs)
        results[h] = m
        bar = "#" * int(m * 40)
        print(f"  {h:>12} {m*100:>14.0f}%   {bar}")

    small = results[sizes[0]]
    big = results[sizes[-1]]
    print("\n" + "=" * 56)
    print("WHAT IT SAYS")
    print("=" * 56)
    climbed = big > small + 0.04 and results[1024] >= results[256]
    if climbed:
        print(f"  CAPABILITY KEPT CLIMBING WITH SIZE. From {small*100:.0f}% at "
              f"64 to {big*100:.0f}% at 2048,")
        print(f"  a {100*(big-small):.0f}-point gain, with each step up helping. On a "
              f"brutal world, the")
        print(f"  big brain reaches what a small one cannot. Scaling goes DEEP: "
              f"the 2048 brain")
        print(f"  (only a GPU can run it) genuinely bought capability. This is "
              f"the real payoff.")
    else:
        print(f"  SCALING PLATEAUED ({small*100:.0f}% -> {big*100:.0f}%). Even "
              f"this world was not hard enough to")
        print(f"  need the biggest brains, or they could not exploit it. The "
              f"big brain did not")
        print(f"  clearly win here; honest read. (A harder world or longer "
              f"life might change it.)")

    print(f"\n  Honest note: accuracy ceilings depend on how learnable the "
          f"world is. What matters")
    print(f"  is the SHAPE: does bigger keep winning? That is the scaling law, "
          f"and it is what")
    print(f"  carries to richer worlds and real resources later.")


if __name__ == "__main__":
    main()
