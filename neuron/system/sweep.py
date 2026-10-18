"""The full map: brain size vs world difficulty. Where does scaling help, and
where is the ceiling?

THE GOAL. Not one task, but a MAP. We sweep the world from easy to brutally
hard, and at each difficulty we run every brain size. This shows three things
at once:
  - where bigger brains win (the scaling zone),
  - where bigger stops helping (the plateau for that difficulty),
  - where the world becomes unlearnable for everyone (task-impossibility, not
    a NEURON limit).

Read it honestly: at max hardness everything collapses toward chance, that is
the task becoming impossible, not NEURON's "max intelligence." The useful
information is the zone of HARD-BUT-LEARNABLE, where the difficulty demands a
particular brain size. If harder worlds keep rewarding bigger brains, scaling
is worth paying for (RunPod). If even the hardest learnable world is handled by
a modest size, bigger brains have no room and a GPU would not help.

    python sweep.py
"""

import sys
import os
import math
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world_interface import World
from mind import Mind


class TunableWorld(World):
    """A world whose difficulty is a dial. `coupling` sets how many signals
    interact nonlinearly to determine each outcome (more = harder), and
    `noise` injects unpredictability (more = closer to unlearnable)."""

    def __init__(self, n_signals, coupling, noise, seed=0):
        rng = random.Random(seed)
        self.rng = rng
        self._names = [f"sig_{i}" for i in range(n_signals)]
        self.phase = {n: rng.uniform(0, 6.28) for n in self._names}
        self.freq = {n: rng.uniform(0.3, 1.2) for n in self._names}
        self.coupling = coupling
        self.noise = noise
        self.couples = {n: rng.sample(self._names, min(coupling, n_signals))
                        for n in self._names}
        self.cw = {n: [rng.uniform(-1, 1) for _ in range(len(self.couples[n]))]
                   for n in self._names}
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
        for k in range(len(c)):
            s += w[k] * math.tanh(self._vals[c[k]] * self._vals[c[(k + 1) % len(c)]])
        s += self.rng.gauss(0, self.noise)      # noise pushes toward unlearnable
        if s > 0.1:
            return 2
        if s < -0.1:
            return 0
        return 1

    def step(self):
        self.t += 1
        self._tick()

    def describe(self):
        return f"TunableWorld(c={self.coupling},n={self.noise})"


def run(hidden, world_params, steps, seed):
    world = TunableWorld(*world_params, seed=seed)
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
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    # difficulty levels: (n_signals, coupling, noise) from easy to brutal
    difficulties = [
        ("easy",    (8,  2, 0.0)),
        ("medium",  (16, 4, 0.1)),
        ("hard",    (24, 6, 0.2)),
        ("harder",  (32, 8, 0.3)),
        ("brutal",  (40, 12, 0.5)),
    ]
    sizes = [16, 64, 256, 512, 1024]

    print("  THE MAP: brain size vs world difficulty. Where does scaling help,")
    print("  where does it plateau, where does the task become impossible?\n")
    print("  (chance = 33% for 3-way prediction)\n")

    # header
    print("=" * 74)
    hdr = f"  {'difficulty':>10} |" + "".join(f"{('h='+str(h)):>12}" for h in sizes)
    print(hdr)
    print("  " + "-" * 70)

    grid = {}
    for dname, dparams in difficulties:
        row = f"  {dname:>10} |"
        for h in sizes:
            accs = [run(h, dparams, args.steps, s) for s in range(args.seeds)]
            m = sum(accs) / len(accs)
            grid[(dname, h)] = m
            row += f"{m*100:>11.0f}%"
        print(row)

    print("\n" + "=" * 74)
    print("WHAT THE MAP SHOWS")
    print("=" * 74)

    # for each difficulty, find the brain size where accuracy stops improving
    for dname, _ in difficulties:
        accs = [(h, grid[(dname, h)]) for h in sizes]
        best_h, best_a = max(accs, key=lambda t: t[1])
        smallest_a = accs[0][1]
        gain = best_a - smallest_a
        if best_a < 0.40:
            verdict = "near chance - this world is ~unlearnable for all sizes"
        elif gain > 0.06:
            verdict = f"bigger wins: best at h={best_h} (+{gain*100:.0f} pts over h=16)"
        else:
            verdict = f"size barely matters here (best {best_a*100:.0f}% at h={best_h})"
        print(f"  {dname:>10}: {verdict}")

    print("\n" + "=" * 74)
    print("THE HONEST READ")
    print("=" * 74)
    # does the winning size climb as difficulty rises?
    winners = []
    for dname, _ in difficulties:
        accs = [(h, grid[(dname, h)]) for h in sizes]
        if max(a for _, a in accs) >= 0.40:
            winners.append((dname, max(accs, key=lambda t: t[1])[0]))
    if winners:
        sizes_won = [w[1] for w in winners]
        if sizes_won[-1] > sizes_won[0]:
            print("  As the world got harder, the brain size that wins CLIMBED.")
            print("  Harder worlds reward bigger brains: scaling is real and")
            print("  worth paying for. RunPod is justified, pair a big brain")
            print("  with a hard world and it keeps climbing.")
        else:
            print("  The winning brain size did NOT climb with difficulty: on")
            print("  learnable worlds, a modest brain suffices and bigger has no")
            print("  room. RunPod would not buy capability here; save the money")
            print("  until a world needs more than these sizes give.")
    print("\n  Note: the 'brutal' row collapsing toward chance is the TASK")
    print("  becoming impossible (too much noise/coupling), not a NEURON limit.")
    print("  The useful zone is hard-but-learnable, where difficulty demands")
    print("  a particular size. That zone tells you how much brain to buy.")


if __name__ == "__main__":
    main()
