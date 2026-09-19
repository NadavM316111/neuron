"""Rung 3: the same world, but seen through imperfect senses.

Everything built so far runs on clean, discrete, hand-designed input. Every
cell is exactly one of fourteen types, unambiguous, identical every time.
Real experience is none of those things, and this is the last rung before a
stream nobody designed.

The world's RULES are untouched. Only the seeing changes:

  continuous    each cell arrives as a real-valued vector, not a one-hot
                code. Two cells of the same type never look identical.
  noisy         gaussian noise on every reading, every step
  confusable    some types have genuinely overlapping signatures, so
                telling a key from a lamp is a judgement, not a lookup
  dropout       occasionally a cell reads as nothing at all

Why this could break the system specifically, rather than just being
harder: the gate fires on SURPRISE, and noise IS surprise. A noisy stream
could make the gate fire on everything, which makes it useless. And the gate
is coupled to rehearsal by construction — items enter the buffer only when
the gate REJECTS them — so if the gate saturates, rehearsal starves with it.

That is a plausible, specific, project-threatening failure that has never
been tested.

This file provides the sensor layer and a check on how confusable the world
has become. The experiment is in noisyrun.py.
"""

import math
import random

from bigworld import (BigWorld, ACTIONS, EVENTS, FORKS, DELTA,
                      EMPTY, WALL, KEY, LOCKED, LAMP, DARK, WATER, ICE,
                      FOOD, HEAVY, ROPE, CHASM, COLDROOM, DOOR2)


FEATURES = 8            # each cell is a point in 8-dimensional sensor space
NOISE = 0.15           # gaussian, per feature, per reading
DROPOUT = 0.05          # chance a cell reads as nothing

# Signatures are deliberately NOT orthogonal. Pairs that a real sensor might
# confuse are placed close together, so distinguishing them takes evidence
# rather than a lookup. The overlap_check below reports how hard that is.
SIGNATURES = {
    EMPTY:    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    WALL:     [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    # key and rope: both small carryable things
    KEY:      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.6, 0.0],
    ROPE:     [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.9, 0.0],
    # the three door types: all barriers, differing in degree
    LOCKED:   [0.8, 0.6, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    HEAVY:    [0.9, 0.7, 0.0, 0.8, 0.0, 0.0, 0.0, 0.2],
    DOOR2:    [0.7, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.4],
    # lamp and food: both small bright useful things
    LAMP:     [0.0, 0.0, 0.7, 0.0, 1.0, 0.0, 0.3, 0.0],
    FOOD:     [0.0, 0.0, 0.6, 0.0, 0.8, 0.0, 0.5, 0.0],
    # dark and chasm: both absences
    DARK:     [0.2, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    CHASM:    [0.3, 0.1, 0.0, 0.0, 0.0, 0.9, 0.0, 0.3],
    # water, ice, cold: a family
    WATER:    [0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ICE:      [0.2, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9],
    COLDROOM: [0.1, 0.4, 0.0, 0.0, 0.0, 0.2, 0.0, 0.8],
}


class NoisyWorld(BigWorld):
    """The same world. Imperfect senses.

    Subclasses BigWorld so the rules, the terrain counts and the fairness
    properties are all inherited unchanged. Only observe() differs, which
    keeps this a clean one-variable step up the ladder.
    """

    def __init__(self, seed=0, rules="normal", noise=NOISE,
                 dropout=DROPOUT):
        self.noise = noise
        self.dropout = dropout
        self._obs_rng = random.Random(seed * 7919 + 13)
        super().__init__(seed=seed, rules=rules)

    def observe(self, hide_state=True):
        """A 5x5 patch of NOISY, CONTINUOUS readings.

        Same patch as before, but each cell is an 8-dimensional vector with
        noise added, occasionally dropped entirely. The same cell read twice
        never gives the same numbers.
        """
        patch = []
        for dr in (-2, -1, 0, 1, 2):
            for dc in (-2, -1, 0, 1, 2):
                r, c = self.r + dr, self.c + dc
                if 0 <= r < self.h and 0 <= c < self.w:
                    cell = self.grid[r][c]
                else:
                    cell = WALL
                if self._obs_rng.random() < self.dropout:
                    sig = SIGNATURES[EMPTY]
                else:
                    sig = SIGNATURES[cell]
                patch.append([v + self._obs_rng.gauss(0.0, self.noise)
                              for v in sig])
        obs = dict(patch=patch)
        if not hide_state:
            obs["state"] = [1 if self.keys > 0 else 0,
                            1 if self.keys >= 2 else 0,
                            1 if self.lamp else 0,
                            1 if self.wet else 0,
                            1 if self.fed else 0,
                            1 if self.cold else 0,
                            1 if self.rope else 0]
        return obs


def walk(layout_seed, walk_seed, n, see_state, episode=None,
         noise=NOISE, dropout=DROPOUT):
    rng = random.Random(walk_seed)
    world = NoisyWorld(layout_seed, noise=noise, dropout=dropout)
    out = []
    for i in range(n):
        if episode and i % episode == 0:
            world.reset()
        obs = world.observe(hide_state=not see_state)
        action = rng.choice(ACTIONS)
        _, event = world.step(action)
        out.append((obs, action, event))
    return out


def overlap_check(noise=NOISE, samples=3000, quiet=False):
    """How often would a perfect nearest-signature reader get a cell wrong?

    This is the honest measure of how hard the seeing has become. If it is
    near zero, the noise is decorative and the rung has not really been
    climbed. If it is very high, nothing is learnable and the test says
    nothing.
    """
    rng = random.Random(0)
    types = list(SIGNATURES)
    wrong = 0
    confusions = {}
    for _ in range(samples):
        true = rng.choice(types)
        reading = [v + rng.gauss(0.0, noise) for v in SIGNATURES[true]]
        best, best_d = None, None
        for t, sig in SIGNATURES.items():
            d = sum((a - b) ** 2 for a, b in zip(reading, sig))
            if best_d is None or d < best_d:
                best, best_d = t, d
        if best != true:
            wrong += 1
            key = tuple(sorted((true, best)))
            confusions[key] = confusions.get(key, 0) + 1
    rate = 100.0 * wrong / samples
    if not quiet:
        names = {EMPTY: "empty", WALL: "wall", KEY: "key", LOCKED: "locked",
                 LAMP: "lamp", DARK: "dark", WATER: "water", ICE: "ice",
                 FOOD: "food", HEAVY: "heavy", ROPE: "rope",
                 CHASM: "chasm", COLDROOM: "cold", DOOR2: "door2"}
        print(f"  a perfect reader misidentifies {rate:.1f}% of cells")
        top = sorted(confusions.items(), key=lambda kv: -kv[1])[:6]
        for (a, b), n in top:
            print(f"    {names[a]:>7} <-> {names[b]:<7} "
                  f"{100.0 * n / samples:5.2f}%")
    return rate


if __name__ == "__main__":
    print("Rung 3: the same world, seen through imperfect senses.\n")
    print(f"  {FEATURES} continuous features per cell")
    print(f"  gaussian noise sd {NOISE} on every reading")
    print(f"  {DROPOUT:.0%} chance a cell reads as nothing")
    print(f"  signatures deliberately overlap for confusable pairs\n")

    print("how confusable is it?\n")
    for nz in [0.0, 0.15, 0.35, 0.6]:
        print(f"  noise {nz}:")
        overlap_check(nz)
        print()

    print("the same cell, read five times:\n")
    w = NoisyWorld(0)
    for i in range(5):
        row = w.observe()["patch"][12]      # the agent's own cell
        print("  " + "  ".join(f"{v:+5.2f}" for v in row))

    print("\nrules are untouched, so the forks stay as they were:")
    from bigworld import fairness_check
    worst, worst_name, _, _ = fairness_check(quiet=True)
    print(f"  worst fork: {worst_name} at {worst:.1f}%")

    print("""
The misidentification rate at the chosen noise level is the measure of this
rung. Near zero means the noise is decorative. Very high means nothing is
learnable and the experiment would say nothing either way.

Somewhere around 10-25% is the useful range: the world is genuinely
ambiguous, but the information is still there for a system patient enough to
accumulate it.
""")