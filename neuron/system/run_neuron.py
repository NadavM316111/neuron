"""Run NEURON: the launch layer for the consolidated Mind.

This is the thin runner that brings the whole thing together: it picks a world,
builds the Mind at a chosen brain size, uses a GPU if one is available, runs
the being's life, saves and resumes its state, and can populate a lineage. It
is deliberately small; all the intelligence is in the Mind. This is the layer
you point at RunPod: same code, bigger `--hidden`, a GPU underneath.

    python run_neuron.py --hidden 64 --steps 2000
    python run_neuron.py --hidden 1024 --steps 50000   # on a GPU (RunPod)
    python run_neuron.py --lineage 5 --life 3000        # a lineage
"""

import argparse
import os
import sys
import math
import random
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from world_interface import World
from mind import Mind


# ----------------------------------------------------------------------
# A default world so the runner works out of the box. Any World subclass
# can be swapped in; the Mind does not care which.
# ----------------------------------------------------------------------
class SineWorld(World):
    """A clean, scalable default world: many signals, each a learnable
    function. Stands in until a richer world is plugged in. The difficulty
    (how fast signals move) is tunable, so the same world can be easy or hard."""

    def __init__(self, n_signals=8, difficulty=0.6, seed=0):
        rng = random.Random(seed)
        self._names = [f"sig_{i}" for i in range(n_signals)]
        self.phase = {n: rng.uniform(0, 6.28) for n in self._names}
        self.freq = {n: rng.uniform(0.2, 0.2 + difficulty) for n in self._names}
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

    def step(self):
        self.t += 1
        self._tick()

    def describe(self):
        return f"SineWorld({len(self._names)} signals)"


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def save_mind(mind, path):
    # save the learned weights and the being's life-state (lightweight).
    state = {
        "weights": mind.backend.snapshot(),
        "gen": mind.gen,
        "samples": mind.samples,
        "emotion": mind.emotion.state,
        "hidden": mind.backend.net.hidden,
    }
    with open(path, "wb") as f:
        pickle.dump(state, f)


def load_into(mind, path):
    if not os.path.exists(path):
        return False
    with open(path, "rb") as f:
        state = pickle.load(f)
    try:
        mind.backend.restore(state["weights"])
        mind.gen = state.get("gen", 0)
        mind.samples = state.get("samples", 0)
        mind.emotion.state = state.get("emotion", mind.emotion.state)
        return True
    except Exception:
        return False


def run_one_life(mind, steps, report_every=0):
    for i in range(steps):
        mind.live_step()
        if report_every and (i + 1) % report_every == 0:
            s = mind.snapshot_summary()
            print(f"  step {i+1:>7} | acc {s['accuracy']:.2f} | "
                  f"brain {s['brain_size']} | emotion {s['emotion']} | "
                  f"episodes {s['episodes']}")
    return mind.snapshot_summary()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=64,
                    help="brain size (the scaling knob; raise this on a GPU)")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--signals", type=int, default=8)
    ap.add_argument("--difficulty", type=float, default=0.6)
    ap.add_argument("--state", type=str, default=None,
                    help="path to save/resume the being's state")
    ap.add_argument("--lineage", type=int, default=0,
                    help="if >0, run a lineage of this many generations")
    ap.add_argument("--life", type=int, default=2000,
                    help="steps per life in a lineage run")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = pick_device()
    print(f"  NEURON runner | device: {device} | brain size: {args.hidden}")
    print(f"  (raise --hidden on a GPU to scale toward intelligence.)\n")

    world = SineWorld(n_signals=args.signals, difficulty=args.difficulty,
                      seed=args.seed)

    if args.lineage > 0:
        # run a lineage: each generation lives, then a child inherits its
        # shaped brain (and evolved learning rate) and lives its own life.
        print(f"  running a lineage of {args.lineage} generations "
              f"({args.life} steps each)\n")
        mind = Mind(world, hidden=args.hidden, seed=args.seed)
        for g in range(args.lineage):
            summary = run_one_life(mind, args.life)
            print(f"  gen {g:>2}: acc {summary['accuracy']:.2f} | "
                  f"brain {summary['brain_size']} | "
                  f"samples {summary['samples']} | "
                  f"lr {getattr(mind, 'inherited_lr', '-')}")
            mind = mind.child(seed=args.seed * 100 + g + 1)
        print("\n  the lineage lived; each child inherited its parent's shaped "
              "brain.")
        return

    # single life
    mind = Mind(world, hidden=args.hidden, seed=args.seed)
    if args.state and load_into(mind, args.state):
        print(f"  resumed from {args.state} (gen {mind.gen}, "
              f"{mind.samples} samples)\n")

    summary = run_one_life(mind, args.steps, report_every=max(1, args.steps // 8))

    print(f"\n  final: {summary}")
    if args.state:
        save_mind(mind, args.state)
        print(f"  saved state to {args.state}")

    print(f"\n  This same runner scales: raise --hidden and run on a GPU "
          f"(RunPod) to grow")
    print(f"  the brain toward real capability. The being, its world "
          f"interface, and the")
    print(f"  scaling knob are all one system now.")


if __name__ == "__main__":
    main()
