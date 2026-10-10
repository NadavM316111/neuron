"""Ablation: does imagination (simulating futures) actually CAUSE the foresight?

SAFE. New, self-contained file. Nothing changed or deleted. imagination.py is
untouched.

THE CLAIM (from imagination.py): the being simulates possible futures through
its own world-model and avoids a trap by FORESEEING it, a tempting cell that
slides into a pit, rather than learning the pit by falling in. Foresight, not
hindsight.

ABLATION. The SAME being, SAME worlds and seeds, run two ways:
  FULL     - imagination ON: it simulates several steps ahead through the
             (true-dynamics) model and refuses the tempting cell because it
             foresees the pit beyond it.
  ABLATED  - imagination OFF: no lookahead; it acts on immediate value only
             (a reactor), grabbing the tempting +2 with no sight of the pit.
If removing imagination makes it walk into the trap far more (hits soar,
reward crashes), imagination CAUSED the foresight. If the reactor does as
well, we over-attributed.

    python ablate_imagination.py --seeds 200
"""

import argparse
import random
import statistics


GRID = 6


class World:
    """A tempting cell (+2) that SLIDES you into a pit (-5) next step. Only
    foreseeing the slide avoids it."""

    def __init__(self, seed):
        rng = random.Random(seed)
        self.reward = {}
        self.pit = set()
        self.sweet = set()
        px = GRID - 2
        for y in range(GRID):
            self.pit.add((px, y)); self.reward[(px, y)] = -5.0
            self.sweet.add((px - 1, y)); self.reward[(px - 1, y)] = 2.0
        self.goal = (0, GRID - 1); self.reward[self.goal] = 3.0

    def step(self, pos, a):
        if pos in self.sweet:
            return (pos[0] + 1, pos[1])          # the slide into the pit
        dx, dy = {"up": (0, -1), "down": (0, 1),
                  "left": (-1, 0), "right": (1, 0)}[a]
        nx, ny = max(0, min(GRID-1, pos[0]+dx)), max(0, min(GRID-1, pos[1]+dy))
        return (nx, ny)

    def rew(self, pos):
        return self.reward.get(pos, -0.1)


ACTIONS = ["up", "down", "left", "right"]


def imagine(world, pos, first, horizon):
    p = world.step(pos, first)
    total = world.rew(p)
    for _ in range(horizon - 1):
        best_a, best_r, best_p = None, -1e9, p
        for a in ACTIONS:
            np_ = world.step(p, a)
            if world.rew(np_) > best_r:
                best_r, best_a, best_p = world.rew(np_), a, np_
        p = best_p; total += best_r
    return total


def run(imagination_on, seed, steps=1500, horizon=5):
    rng = random.Random(seed)
    world = World(seed)
    pos = (0, 0)
    total = 0.0
    traps = 0
    for t in range(steps):
        if imagination_on:
            # simulate futures, pick the action whose imagined future is best
            a = max(ACTIONS, key=lambda a: imagine(world, pos, a, horizon))
        else:
            # ablated reactor: immediate value only, no lookahead
            a = max(ACTIONS, key=lambda a: world.rew(world.step(pos, a)))
        nxt = world.step(pos, a)
        total += world.rew(nxt)
        if nxt in world.pit:
            traps += 1
        pos = nxt
        if rng.random() < 0.05:
            pos = (rng.randrange(GRID), rng.randrange(GRID))
    return total, traps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=200)
    args = ap.parse_args()

    print(f"  ABLATION of imagination. the SAME being on the SAME "
          f"{args.seeds} worlds, run with")
    print(f"  lookahead ON vs OFF. a tempting cell slides into a pit; only "
          f"foreseeing it")
    print(f"  avoids it. does simulating the future actually cause the "
          f"avoidance?\n")

    full_r, abl_r, full_t, abl_t = [], [], [], []
    worse = 0
    for seed in range(args.seeds):
        fr, ft = run(True, seed)
        ar, at = run(False, seed)
        full_r.append(fr); abl_r.append(ar)
        full_t.append(ft); abl_t.append(at)
        if ar < fr - 1:
            worse += 1

    fm, am = statistics.mean(full_r), statistics.mean(abl_r)
    ftm, atm = statistics.mean(full_t), statistics.mean(abl_t)
    pct = 100.0 * worse / args.seeds

    print("=" * 66)
    print("WITH IMAGINATION vs WITHOUT (ablated)")
    print("=" * 66)
    print(f"  reward   FULL: {fm:>8.0f}      ABLATED: {am:>8.0f}")
    print(f"  traps    FULL: {ftm:>6.1f}      ABLATED: {atm:>6.1f}")

    print("\n" + "=" * 66)
    print("DID THE CLAIM SURVIVE ABLATION?")
    print("=" * 66)
    if am < fm - 0.15 * abs(fm) and atm > ftm * 1.3 and pct > 80:
        print(f"  THE CLAIM SURVIVES. Removing imagination made it walk into "
              f"the trap: {ftm:.0f} -> {atm:.0f}")
        print(f"  trap hits, reward {fm:.0f} -> {am:.0f}, worse in {pct:.0f}% "
              f"of worlds. Without simulating")
        print(f"  the future, it grabbed the tempting cell and fell into the "
              f"pit. Imagination")
        print(f"  CAUSED the foresight; it is not decorative.")
    elif am < fm - 0.05 * abs(fm) or atm > ftm * 1.1:
        print(f"  PARTIALLY. Removing it hurt ({fm:.0f} -> {am:.0f}, traps "
              f"{ftm:.0f} -> {atm:.0f}); real but modest.")
    else:
        print(f"  THE CLAIM DID NOT SURVIVE. The reactor did about as well "
              f"({fm:.0f} vs {am:.0f}). Imagination")
        print(f"  was NOT causing the foresight we claimed. Honest "
              f"correction.")

    print(f"\n  Nothing deleted; separate test. {args.seeds} worlds, same "
          f"being, lookahead toggled.")


if __name__ == "__main__":
    main()
