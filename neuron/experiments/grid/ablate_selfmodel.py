"""Ablation: does the self-model actually CAUSE the self-control we claimed?

SAFE. New, self-contained file. Changes and deletes nothing. The original
self_model2.py is untouched.

THE CLAIM (from self_model2.py): the being builds a model of its own future
and uses it to foresee where an impulse leads, then resists the impulse that
would crash it (indulge now, starve later). Self-governance from a self-built
model. We saw it resist. But seeing resistance while the self-model is
present does not prove the self-model caused it.

ABLATION. The SAME being, SAME worlds and seeds, run two ways:
  FULL     - self-model ON: it predicts its own future resource and
             restrains when it foresees a crash.
  ABLATED  - self-model OFF: no model of its own future; it acts on the
             immediate impulse (indulge whenever tempted).
If removing the self-model makes it crash far more, the self-model CAUSED the
self-control, claim survives. If the ablated being governs itself anyway, we
over-attributed. Honest either way.

    python ablate_selfmodel.py --seeds 200
"""

import argparse
import random
import statistics


INDULGE_REWARD = 1.0
INDULGE_COST = 0.15
CRASH_PENALTY = 8.0
CRASH_STEPS = 15
TEMPT_PROB = 0.5
RESOURCE_DRAIN = 0.02


def run(self_model_on, steps, seed, horizon=6):
    rng = random.Random(seed)
    resource = 1.0
    crash_timer = 0
    reward = 0.0
    crashes = 0

    # the self-model: a tiny learned predictor of the being's OWN future
    # resource from (resource, action). only used when self_model_on.
    w = [0.0, 0.0]
    b = 0.0
    lr = 0.03
    hist = []

    def predict_future(r, spend):
        return w[0] * (r / 1.0) + w[1] * spend + b

    for t in range(steps):
        tempted = rng.random() < TEMPT_PROB
        before = resource
        if crash_timer > 0:
            action = 0
        elif not tempted:
            action = 0
        else:
            if self_model_on:
                # foresee: if indulging is predicted to crash me, restrain
                foreseen = predict_future(resource, 1.0)
                action = 1 if foreseen > 0.12 else 0
            else:
                # ablated: no self-model, act on the impulse
                action = 1

        if action == 1:
            resource -= INDULGE_COST
            reward += INDULGE_REWARD
        resource -= RESOURCE_DRAIN
        resource = max(0.0, min(1.0, resource))

        if resource <= 0.0 and crash_timer == 0:
            crash_timer = CRASH_STEPS
            crashes += 1
        if crash_timer > 0:
            reward -= CRASH_PENALTY / CRASH_STEPS
            crash_timer -= 1
            if crash_timer == 0:
                resource = 0.5

        # the self-model LEARNS (only when on) to predict its own future
        hist.append((before, action, resource))
        if self_model_on and len(hist) > horizon:
            pr, pa, _ = hist[-horizon - 1]
            pred = w[0] * pr + w[1] * float(pa) + b
            err = resource - pred
            w[0] += lr * err * pr
            w[1] += lr * err * float(pa)
            b += lr * err

    return reward, crashes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=200)
    args = ap.parse_args()

    print(f"  ABLATION of the self-model. the SAME being on the SAME "
          f"{args.seeds} worlds, run with")
    print(f"  its self-model ON vs OFF. if removing it makes it crash far "
          f"more, the self-model")
    print(f"  truly caused the self-control. if not, we over-attributed.\n")

    full_r, abl_r, full_c, abl_c = [], [], [], []
    worse = 0
    for seed in range(args.seeds):
        fr, fc = run(True, args.steps, seed)
        ar, ac = run(False, args.steps, seed)
        full_r.append(fr); abl_r.append(ar)
        full_c.append(fc); abl_c.append(ac)
        if ar < fr - 1:
            worse += 1

    print("=" * 66)
    print("WITH SELF-MODEL vs WITHOUT (ablated)")
    print("=" * 66)
    fm, am = statistics.mean(full_r), statistics.mean(abl_r)
    fcm, acm = statistics.mean(full_c), statistics.mean(abl_c)
    print(f"  reward   FULL: {fm:>8.0f}      ABLATED: {am:>8.0f}")
    print(f"  crashes  FULL: {fcm:>6.1f}      ABLATED: {acm:>6.1f}")
    pct = 100.0 * worse / args.seeds

    print("\n" + "=" * 66)
    print("DID THE CLAIM SURVIVE ABLATION?")
    print("=" * 66)
    if am < fm - 0.1 * abs(fm) and acm > fcm * 1.2 and pct > 70:
        print(f"  THE CLAIM SURVIVES. Removing the self-model made it crash "
              f"far more: {fcm:.0f} -> {acm:.0f}")
        print(f"  crashes, reward {fm:.0f} -> {am:.0f}, worse in {pct:.0f}% "
              f"of worlds. Unable to foresee")
        print(f"  its own future, it stopped resisting the impulse and "
              f"crashed. The self-model")
        print(f"  CAUSED the self-control; it is not decorative.")
    elif am < fm - 0.03 * abs(fm) or acm > fcm * 1.1:
        print(f"  PARTIALLY. Removing it hurt somewhat (crashes {fcm:.0f} -> "
              f"{acm:.0f}, {pct:.0f}% of worlds")
        print(f"  worse), so the self-model does real work, but more modestly "
              f"than the single")
        print(f"  demo implied.")
    else:
        print(f"  THE CLAIM DID NOT SURVIVE. The ablated being did about as "
              f"well (crashes {fcm:.0f} vs")
        print(f"  {acm:.0f}, reward {fm:.0f} vs {am:.0f}). The self-model was "
              f"NOT causing the self-control we")
        print(f"  attributed to it. Honest correction.")

    print(f"\n  Nothing deleted; separate test. {args.seeds} worlds, same "
          f"being, one component")
    print(f"  toggled.")


if __name__ == "__main__":
    main()
