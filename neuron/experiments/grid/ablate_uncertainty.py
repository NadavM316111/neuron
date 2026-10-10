"""Ablation: does acting on uncertainty actually CAUSE the benefit we claimed?

SAFE. New, self-contained file. Nothing is changed or deleted. uncertainty.py
is untouched.

THE CLAIM (from uncertainty.py): the being tracks how much evidence it has
for each belief and ABSTAINS when it is unsure, so it avoids confident wrong
mistakes, acting only where it is reliable. We saw it do better than a being
that always acts. But seeing the benefit while the mechanism is present does
not prove the mechanism caused it.

ABLATION. The SAME being, SAME worlds and seeds, run two ways:
  FULL     - uncertainty ON: abstains when its confidence is below a cutoff.
  ABLATED  - uncertainty OFF: always acts on its best current guess, no
             abstention, no sense of its own doubt.
If removing it makes the being worse (more costly wrong actions, lower
reward), acting-on-uncertainty CAUSED the benefit. If the ablated being does
just as well, we over-attributed. Honest either way.

    python ablate_uncertainty.py --seeds 200
"""

import argparse
import random
import statistics


RIGHT = 1.0
WRONG = -4.0


def confidence(votes):
    tot = sum(votes.values())
    if tot == 0:
        return 0.0
    top = sorted(votes.values(), reverse=True)
    second = top[1] if len(top) > 1 else 0
    return (top[0] - second) / (tot + 2.0)


def run(uncertainty_on, n_sit, steps, seed, cutoff=0.15, obs_acc=0.6):
    rng = random.Random(seed)
    answers = {i: rng.randint(0, 2) for i in range(n_sit)}
    weights = [1.0 / (i + 1) for i in range(n_sit)]
    votes = {i: {} for i in range(n_sit)}

    total = 0.0
    for t in range(steps):
        r = rng.random() * sum(weights)
        acc = 0.0
        sit = 0
        for i, w in enumerate(weights):
            acc += w
            if r <= acc:
                sit = i
                break
        truth = answers[sit]

        v = votes[sit]
        belief = max(v, key=v.get) if v else None

        # DECISION: the one thing being ablated. with uncertainty ON, abstain
        # when confidence is low. with it OFF, always act on the best guess.
        if uncertainty_on:
            do_act = confidence(v) >= cutoff and belief is not None
        else:
            do_act = True

        if do_act:
            guess = belief if belief is not None else rng.randint(0, 2)
            total += RIGHT if guess == truth else WRONG

        # noisy observation updates evidence (same for both)
        obs = truth if rng.random() < obs_acc else \
            rng.choice([a for a in range(3) if a != truth])
        v[obs] = v.get(obs, 0) + 1

    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--situations", type=int, default=80)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=200)
    args = ap.parse_args()

    print(f"  ABLATION of acting-on-uncertainty. the SAME being on the SAME "
          f"{args.seeds} worlds, run")
    print(f"  with abstention ON vs OFF. confident wrong actions cost "
          f"{WRONG:.0f}; abstaining is free.\n")

    full_r, abl_r = [], []
    worse = 0
    for seed in range(args.seeds):
        fr = run(True, args.situations, args.steps, seed)
        ar = run(False, args.situations, args.steps, seed)
        full_r.append(fr); abl_r.append(ar)
        if ar < fr - 1:
            worse += 1

    fm, am = statistics.mean(full_r), statistics.mean(abl_r)
    pct = 100.0 * worse / args.seeds

    print("=" * 66)
    print("WITH UNCERTAINTY vs WITHOUT (ablated)")
    print("=" * 66)
    print(f"  reward   FULL (abstains when unsure): {fm:>8.0f}")
    print(f"           ABLATED (always acts):       {am:>8.0f}")

    print("\n" + "=" * 66)
    print("DID THE CLAIM SURVIVE ABLATION?")
    print("=" * 66)
    if am < fm - 0.05 * abs(fm) and pct > 70:
        print(f"  THE CLAIM SURVIVES. Removing its sense of uncertainty made "
              f"it worse: reward")
        print(f"  {fm:.0f} -> {am:.0f}, worse in {pct:.0f}% of worlds. Forced "
              f"to always act, it made the")
        print(f"  confident wrong mistakes it used to avoid. Acting on "
              f"uncertainty CAUSED the")
        print(f"  benefit; it is not decorative.")
    elif am < fm - 0.01 * abs(fm):
        print(f"  PARTIALLY. Removing it hurt somewhat ({fm:.0f} -> {am:.0f}, "
              f"{pct:.0f}% of worlds worse); real")
        print(f"  but modest.")
    else:
        print(f"  THE CLAIM DID NOT SURVIVE. The always-acting being did "
              f"about as well ({fm:.0f} vs {am:.0f}).")
        print(f"  Acting on uncertainty was NOT causing the benefit we "
              f"claimed. Honest correction.")

    print(f"\n  Nothing deleted; separate test. {args.seeds} worlds, same "
          f"being, one toggle.")


if __name__ == "__main__":
    main()
