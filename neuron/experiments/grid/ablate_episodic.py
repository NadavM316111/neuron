"""Ablation: does episodic memory actually CAUSE the effect we claimed?

SAFE. This is a new, self-contained experiment. It changes and deletes
nothing. It re-implements the episodic-memory situation here, so the original
episodic.py is untouched.

THE CLAIM (from episodic.py): episodic memory lets the being learn from
single bad events the average erases, avoiding specific disasters it would
otherwise keep walking into (the +768 vs -4514 result). We SAW the effect
with the component present. But seeing a behavior when a component is present
is not proof the component caused it. Maybe something else did the work and
episodic memory was along for the ride.

ABLATION is the test. Run the SAME being, on the SAME worlds and seeds, two
ways:
  FULL     - episodic memory ON: it remembers specific disasters and avoids
             them.
  ABLATED  - episodic memory OFF: the recall is bypassed, so it cannot
             remember specific bad events; it has only the general rule.
If removing episodic memory brings the disaster BACK (performance crashes,
traps return), then episodic memory CAUSED the effect, the claim survives. If
the ablated being does just as well, then episodic memory was NOT doing what
we said, and we over-attributed. We report honestly either way.

    python ablate_episodic.py --seeds 200
"""

import argparse
import random
import statistics


SAFE_REWARD = 1.0
DISASTER = -10.0


def make_world(n_sit, n_exc, seed):
    rng = random.Random(seed)
    sits = []
    for i in range(n_sit):
        feat = rng.random()
        sits.append({"feature": feat, "safe_by_rule": feat > 0.5,
                     "is_exception": False})
    safe = [s for s in sits if s["safe_by_rule"]]
    rng.shuffle(safe)
    for s in safe[:n_exc]:
        s["is_exception"] = True
    return sits


def truly_safe(s):
    return s["safe_by_rule"] and not s["is_exception"]


def run(episodic_on, sits, steps, seed, recall_dist=0.03):
    """Run the being. episodic_on toggles the ONE thing being ablated: the
    ability to remember a specific disaster and avoid it. Everything else is
    identical."""
    rng = random.Random(seed + 7)
    episodes = []          # specific disasters remembered (episodic memory)
    total = 0.0
    traps = 0
    n = len(sits)
    weights = [1.0] * n

    for t in range(steps):
        s = sits[_wi(weights, rng)]
        feat = s["feature"]
        # decision: act if the general rule says safe, UNLESS episodic memory
        # (when on) recalls a specific disaster near this feature.
        if not s["safe_by_rule"]:
            do_act = False
        else:
            near = (episodic_on and
                    any(abs(feat - e) < recall_dist for e in episodes))
            do_act = not near

        if do_act:
            if truly_safe(s):
                total += SAFE_REWARD
            else:
                total += DISASTER
                traps += 1
                # the disaster happens; with episodic memory ON, it is
                # remembered as a specific event. with it OFF (ablated), it is
                # NOT remembered, so it will be walked into again.
                if episodic_on:
                    episodes.append(feat)
    return total, traps


def _wi(weights, rng):
    r = rng.random() * sum(weights)
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if r <= acc:
            return i
    return len(weights) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--situations", type=int, default=40)
    ap.add_argument("--exceptions", type=int, default=6)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=200)
    args = ap.parse_args()

    print(f"  ABLATION of episodic memory. the SAME being on the SAME "
          f"{args.seeds} worlds, run with")
    print(f"  episodic memory ON vs OFF. if removing it brings the disaster "
          f"back, it truly")
    print(f"  caused the effect. if not, we over-attributed.\n")

    full_r, abl_r, full_tr, abl_tr = [], [], [], []
    worse_when_ablated = 0

    for seed in range(args.seeds):
        sits = make_world(args.situations, args.exceptions, seed)
        fr, ftr = run(True, sits, args.steps, seed)
        ar, atr = run(False, sits, args.steps, seed)
        full_r.append(fr); abl_r.append(ar)
        full_tr.append(ftr); abl_tr.append(atr)
        if ar < fr - 1:
            worse_when_ablated += 1

    print("=" * 66)
    print("WITH EPISODIC MEMORY vs WITHOUT (ablated)")
    print("=" * 66)
    fm, fs = statistics.mean(full_r), statistics.pstdev(full_r)
    am, as_ = statistics.mean(abl_r), statistics.pstdev(abl_r)
    ftm = statistics.mean(full_tr)
    atm = statistics.mean(abl_tr)
    print(f"  reward   FULL: {fm:>8.0f} +/- {fs:.0f}")
    print(f"           ABLATED: {am:>8.0f} +/- {as_:.0f}")
    print(f"  traps    FULL: {ftm:>6.1f}      ABLATED: {atm:>6.1f}")
    pct = 100.0 * worse_when_ablated / args.seeds

    print("\n" + "=" * 66)
    print("DID THE CLAIM SURVIVE ABLATION?")
    print("=" * 66)
    drop = fm - am
    if am < fm - 0.3 * abs(fm) and pct > 80:
        print(f"  THE CLAIM SURVIVES. Removing episodic memory brought the "
              f"disaster back: reward")
        print(f"  fell by {drop:.0f} (from {fm:.0f} to {am:.0f}) and traps "
              f"rose from {ftm:.0f} to {atm:.0f}. The")
        print(f"  ablated being, unable to remember specific disasters, "
              f"walked into them again,")
        print(f"  in {pct:.0f}% of worlds. Episodic memory CAUSED the "
              f"effect; it is not decorative.")
    elif am < fm - 0.1 * abs(fm):
        print(f"  PARTIALLY. Removing it hurt ({fm:.0f} -> {am:.0f}, "
              f"{pct:.0f}% of worlds worse), so episodic")
        print(f"  memory does real work, but less overwhelmingly than the "
              f"single demo suggested.")
    else:
        print(f"  THE CLAIM DID NOT SURVIVE. The ablated being did about as "
              f"well ({am:.0f} vs {fm:.0f}).")
        print(f"  Episodic memory was NOT causing the effect we attributed to "
              f"it; we over-")
        print(f"  claimed, and this is the honest correction.")

    print(f"\n  Nothing was deleted; this was a separate test. {args.seeds} "
          f"worlds, same being,")
    print(f"  one component toggled. That is how you prove a part does what "
          f"you say it does.")


if __name__ == "__main__":
    main()
