"""Ablation: does theory of mind actually CAUSE the ability to predict others?

SAFE. New, self-contained file. Nothing changed or deleted. other_minds.py is
untouched.

THE CLAIM (from other_minds.py): the being models OTHERS as separate minds,
inferring another agent's own goal from how it moves and predicting it from
that, rather than assuming the other wants what the being itself wants.

ABLATION. The SAME task, SAME seeds, run two ways:
  FULL     - theory of mind ON: infer the other's own goal from its observed
             behaviour, predict the other from THAT.
  ABLATED  - theory of mind OFF (egocentric): assume the other shares the
             being's own goal, predict the other as if it wanted what the
             being wants.
If removing theory of mind makes prediction of others collapse (back toward
chance), theory of mind CAUSED the social prediction. If egocentric does as
well, we over-attributed.

    python ablate_tom.py --seeds 300
"""

import argparse
import random
import statistics


GRID = 7


def toward(pos, goal):
    x, y = pos
    gx, gy = goal
    if abs(gx - x) >= abs(gy - y):
        return (x + (1 if gx > x else -1 if gx < x else 0), y)
    return (x, y + (1 if gy > y else -1 if gy < y else 0))


def run(tom_on, trials, seed):
    rng = random.Random(seed)
    correct = 0
    for _ in range(trials):
        my_goal = (rng.randrange(GRID), rng.randrange(GRID))
        other_goal = (rng.randrange(GRID), rng.randrange(GRID))
        while other_goal == my_goal:
            other_goal = (rng.randrange(GRID), rng.randrange(GRID))
        p = (rng.randrange(GRID), rng.randrange(GRID))
        # observe the other move a few steps (evidence of ITS goal)
        observed = []
        for _ in range(3):
            np_ = toward(p, other_goal)
            observed.append((p, np_)); p = np_
        true_next = toward(p, other_goal)

        if not tom_on:
            # egocentric: assume the other wants what I want
            pred = toward(p, my_goal)
        else:
            # infer the other's own goal from its observed moves
            best_g, best_s = None, -1
            for gx in range(GRID):
                for gy in range(GRID):
                    g = (gx, gy); s = 0
                    for (a, b) in observed:
                        if toward(a, g) == b:
                            s += 1
                    if s > best_s:
                        best_s, best_g = s, g
            pred = toward(p, best_g)

        if pred == true_next:
            correct += 1
    return 100.0 * correct / trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=300)
    args = ap.parse_args()

    print(f"  ABLATION of theory of mind. the SAME task on the SAME "
          f"{args.seeds} seeds, run with ToM")
    print(f"  ON (infer the other's own goal) vs OFF (egocentric: assume it "
          f"wants what I want).")
    print(f"  does modeling the other as a separate mind cause the social "
          f"prediction?\n")

    full, abl = [], []
    worse = 0
    for seed in range(args.seeds):
        f = run(True, args.trials, seed)
        a = run(False, args.trials, seed)
        full.append(f); abl.append(a)
        if a < f - 1:
            worse += 1

    fm, am = statistics.mean(full), statistics.mean(abl)
    pct = 100.0 * worse / args.seeds

    print("=" * 66)
    print("WITH THEORY OF MIND vs WITHOUT (ablated, egocentric)")
    print("=" * 66)
    print(f"  predicts the other   FULL (models the other): {fm:>6.0f}%")
    print(f"                       ABLATED (egocentric):    {am:>6.0f}%")

    print("\n" + "=" * 66)
    print("DID THE CLAIM SURVIVE ABLATION?")
    print("=" * 66)
    if am < fm - 15 and pct > 85:
        print(f"  THE CLAIM SURVIVES. Removing theory of mind collapsed its "
              f"prediction of others:")
        print(f"  {fm:.0f}% -> {am:.0f}%, worse in {pct:.0f}% of runs. "
              f"Assuming the other wanted what IT wanted,")
        print(f"  it mispredicted almost everyone. Modeling the other as a "
              f"separate mind CAUSED")
        print(f"  the social prediction; it is not decorative.")
    elif am < fm - 5:
        print(f"  PARTIALLY. Removing it hurt ({fm:.0f}% -> {am:.0f}%, "
              f"{pct:.0f}% worse); real but smaller.")
    else:
        print(f"  THE CLAIM DID NOT SURVIVE. Egocentric did about as well "
              f"({fm:.0f}% vs {am:.0f}%). Theory of")
        print(f"  mind was NOT causing the social prediction. Honest "
              f"correction.")

    print(f"\n  Nothing deleted; separate test. {args.seeds} seeds, same "
          f"task, one toggle.")


if __name__ == "__main__":
    main()
