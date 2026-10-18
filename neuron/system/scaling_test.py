"""Does a bigger brain actually win? The free checkpoint before RunPod.

THE QUESTION. Scaling only helps if the world is hard enough to need the extra
capacity. On an easy world, a big brain is wasted and can even do worse. So
before spending money on a GPU, we prove, for free, on the laptop, whether a
bigger brain genuinely beats a smaller one on a GENUINELY HARD world. If it
does, the architecture scales and RunPod is worth it. If it does not, scaling
will not help and we save the money.

THE HARD WORLD. Many signals, and a prediction target that depends on COMPLEX
nonlinear interactions, conjunctions of several signals, not a single smooth
curve. A small brain cannot represent this structure; a big brain can. This is
the kind of world where capacity actually matters.

THE TEST. Run the consolidated Mind at several brain sizes on this hard world.
Measure final accuracy.

  accuracy RISES with brain size (small plateaus low, big reaches higher)
      The architecture scales: capacity buys capability on hard problems. This
      is the signal that justifies RunPod, where an even bigger brain on an
      even harder world would go further.
  bigger does not win
      The architecture does not scale on this task; RunPod would not help here.
      Honest, money-saving negative.

    python scaling_test.py
"""

import sys
import os
import math
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world_interface import World
from mind import Mind


class HardWorld(World):
    """A genuinely hard world: many signals whose predictable direction depends
    on NONLINEAR INTERACTIONS among several of them, not a single smooth curve.
    Predicting it needs real representational capacity."""

    def __init__(self, n_signals=16, seed=0):
        rng = random.Random(seed)
        self._names = [f"sig_{i}" for i in range(n_signals)]
        self.phase = {n: rng.uniform(0, 6.28) for n in self._names}
        self.freq = {n: rng.uniform(0.3, 1.2) for n in self._names}
        # the hidden coupling: each signal's FUTURE depends on a nonlinear
        # combination of several OTHER signals - a structure a small brain
        # cannot capture.
        self.couples = {n: rng.sample(self._names, 4) for n in self._names}
        self.cw = {n: [rng.uniform(-1, 1) for _ in range(4)] for n in self._names}
        self.t = 0
        self._vals = {}
        self._tick()

    def _base(self, n):
        return 0.5 + 0.45 * math.sin(self.phase[n] + self.freq[n] * self.t)

    def _tick(self):
        # current values are base sinusoids
        for n in self._names:
            self._vals[n] = self._base(n)

    def signals(self):
        return self._names

    def observe(self, name):
        return self._vals.get(name, 0.5)

    def outcome(self, name, past_value):
        # the FUTURE direction depends on a NONLINEAR INTERACTION of coupled
        # signals - the hard structure. a small brain can't represent this.
        if past_value is None:
            return None
        coupled = self.couples[name]
        w = self.cw[name]
        s = sum(w[k] * math.tanh(self._vals[coupled[k]]
                                 * self._vals[coupled[(k + 1) % 4]])
                for k in range(4))
        # direction: up if the interaction is positive, down if negative
        if s > 0.1:
            return 2
        if s < -0.1:
            return 0
        return 1

    def step(self):
        self.t += 1
        self._tick()

    def describe(self):
        return f"HardWorld({len(self._names)} coupled signals)"


def run_size(hidden, steps, seed):
    world = HardWorld(n_signals=16, seed=seed)
    mind = Mind(world, hidden=hidden, seed=seed)
    accs = []
    for i in range(steps):
        mind.live_step()
        if i > steps * 0.7:
            accs.append(mind.recent_acc())
    return sum(accs) / len(accs) if accs else 0, mind.snapshot_summary()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    print("  Does a bigger brain win on a GENUINELY HARD world? The free")
    print("  checkpoint before RunPod: prove scaling helps locally first.\n")
    print("  (the world's future depends on nonlinear interactions among many")
    print("   signals, structure a small brain cannot represent.)\n")

    sizes = [16, 64, 256, 512]
    print("=" * 56)
    print("BRAIN SIZE vs CAPABILITY ON A HARD WORLD")
    print("=" * 56)
    print(f"  {'brain size':>12} {'final accuracy':>16}")
    print("  " + "-" * 30)
    results = {}
    for h in sizes:
        accs = []
        for s in range(args.seeds):
            a, _ = run_size(h, args.steps, s)
            accs.append(a)
        m = sum(accs) / len(accs)
        results[h] = m
        bar = "#" * int(m * 40)
        print(f"  {h:>12} {m*100:>14.0f}%   {bar}")

    small = results[sizes[0]]
    big = results[sizes[-1]]

    print("\n" + "=" * 56)
    print("WHAT IT SAYS")
    print("=" * 56)
    if big > small + 0.06 and results[256] >= results[64]:
        gain = 100 * (big - small)
        print(f"  BIGGER BRAIN WON. Accuracy rose from {small*100:.0f}% at "
              f"size {sizes[0]} to {big*100:.0f}% at")
        print(f"  size {sizes[-1]}, a {gain:.0f}-point gain. On a hard world, "
              f"capacity buys capability.")
        print(f"  The architecture SCALES, which is the signal that justifies "
              f"RunPod: an even")
        print(f"  bigger brain on an even harder world would go further. "
              f"Worth the GPU.")
    elif big > small + 0.02:
        print(f"  A MODEST SCALING GAIN ({100*(big-small):.0f} points, "
              f"{small*100:.0f}%->{big*100:.0f}%). Bigger helped somewhat;")
        print(f"  real but not dramatic. RunPod might help but the gain may be "
              f"limited.")
    else:
        print(f"  BIGGER DID NOT WIN ({small*100:.0f}% vs {big*100:.0f}%). "
              f"The architecture did not scale on")
        print(f"  this task; a GPU would not buy capability here. Honest, "
              f"money-saving negative:")
        print(f"  do not spend on RunPod until scaling shows up locally.")

    print(f"\n  This is the honest pre-RunPod checkpoint. Scaling is only "
          f"worth paying for if")
    print(f"  bigger genuinely wins on hard problems, and this tests exactly "
          f"that, for free.")


if __name__ == "__main__":
    main()
