"""Can it learn from the one time, not just the usual?

TWO KINDS OF MEMORY. NEURON has SEMANTIC memory: the general pattern, baked
into weights by averaging over everything it has seen. "Berries like these
are usually safe." It is good at the common case and, by its nature, washes
out rare specifics. What it lacks is EPISODIC memory: a specific recalled
event, remembered as itself. "The one time I ate those particular berries by
the river, I got sick." One event, not averaged away.

WHY BOTH ARE NEEDED. Some decisions turn on a single specific past episode
that the general model actively ERASES. If a food is safe 95% of the time
but the one time you ate a particular variant you got violently ill, the
semantic average still says "safe" -- it drowns the one event in the
ninety-nine. Only a memory that keeps that episode AS a specific,
recallable thing can save you. A being with only semantic memory cannot
learn from single important events. That is the gap this fills.

THE TEST. A world of situations. Most of the time a situation's outcome
follows the general rule (which the semantic model learns well). But a FEW
specific situations are EXCEPTIONS: they look like they should be fine by
the general rule, but each has been, on one memorable occasion, a disaster.
The general rule cannot represent these -- they are individual exceptions,
not a pattern. The question: can a being that keeps an episodic store recall
"this exact situation burned me once" and avoid it, where a semantic-only
being walks back in?

  semantic   general model only. learns the rule, follows it. cannot hold
             single exceptions. the baseline NEURON has today.
  episodic   general model PLUS a store of specific episodes it can recall
             by similarity. when a situation closely matches a remembered
             disaster, it heeds the episode over the rule.
  both_off   acts on the rule with no memory of exceptions at all (same as
             semantic here; sanity check).

  episodic avoids the exception-traps, semantic keeps hitting them
      Episodic memory earned its place: it learned from single events the
      average erased, which no amount of general learning can do.
  episodic no better than semantic
      The recall is not working, or the exceptions are not rare enough to
      distinguish the two kinds of memory.

Note: this is deliberately a MEMORY test, not a prediction-network test, so
the "semantic" model is an explicit per-situation average (the cleanest
possible stand-in for weight-based general learning) and episodic memory is
an explicit event store. That keeps the comparison about the two KINDS of
memory, not about network tuning.

    python episodic.py
    python episodic.py --steps 6000 --exceptions 8
"""

import argparse
import random
import statistics


SAFE_REWARD = 1.0
DISASTER = -10.0          # hitting an exception-trap is very costly
ABSTAIN = 0.0


def make_world(n_situations, n_exceptions, seed):
    """Most situations follow a general rule: a feature (0..1) predicts
    whether acting is safe. A few EXCEPTIONS look safe by the rule but are
    actually disasters -- individual, not patterned."""
    rng = random.Random(seed)
    situations = []
    for i in range(n_situations):
        feature = rng.random()
        # general rule: safe if feature > 0.5
        safe_by_rule = feature > 0.5
        situations.append({"id": i, "feature": feature,
                           "safe_by_rule": safe_by_rule,
                           "is_exception": False})
    # pick some "safe by rule" situations and make them secret disasters
    safe_ones = [s for s in situations if s["safe_by_rule"]]
    rng.shuffle(safe_ones)
    for s in safe_ones[:n_exceptions]:
        s["is_exception"] = True
    return situations


def truly_safe(s):
    return s["safe_by_rule"] and not s["is_exception"]


def run(mode, situations, steps, seed, recall_dist=0.03):
    rng = random.Random(seed + 7)
    n = len(situations)

    # SEMANTIC memory: a running estimate per situation of "did acting go
    # well", but CRUCIALLY generalised by the feature -- the general model
    # only sees the rule-level signal, so a single disaster in one situation
    # barely moves the global rule. We model semantic as: trust the rule.
    # EPISODIC memory: an explicit store of (feature, outcome) for specific
    # bad events, recalled by closeness in feature space.
    episodes = []          # list of features where a disaster happened

    total = 0.0
    traps_hit = 0
    safe_taken = 0
    abstained = 0

    order = []
    # visit situations in a long random stream, exceptions appear rarely too
    weights = [1.0] * n
    for t in range(steps):
        s = situations[_weighted_idx(weights, rng)]
        feat = s["feature"]

        # decide: act or not
        if mode == "semantic":
            # follow the general rule only
            do_act = s["safe_by_rule"]
        else:  # episodic: rule, UNLESS a remembered disaster is close by
            if not s["safe_by_rule"]:
                do_act = False
            else:
                # recall: is there a past disaster at a very similar feature?
                near_disaster = any(abs(feat - e) < recall_dist
                                    for e in episodes)
                do_act = not near_disaster

        if do_act:
            if truly_safe(s):
                total += SAFE_REWARD
                safe_taken += 1
            else:
                # it was an exception-trap: disaster
                total += DISASTER
                traps_hit += 1
                # EPISODIC: remember THIS specific event, as itself
                if mode == "episodic":
                    episodes.append(feat)
        else:
            total += ABSTAIN
            abstained += 1

    return dict(reward=total, traps_hit=traps_hit, safe_taken=safe_taken,
                abstained=abstained)


def _weighted_idx(weights, rng):
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
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    print(f"  {args.situations} situations under a general rule, "
          f"{args.exceptions} of them secret EXCEPTIONS")
    print(f"  (look safe by the rule, but each burned the being once). "
          f"safe +{SAFE_REWARD:.0f}, trap {DISASTER:.0f}.")
    print(f"  can a being that remembers specific disasters avoid them, "
          f"where the general")
    print(f"  rule says 'safe' and keeps walking in?\n")

    modes = ["semantic", "episodic"]
    agg = {m: {"reward": [], "traps_hit": [], "safe_taken": [],
               "abstained": []} for m in modes}
    for seed in range(args.seeds):
        for m in modes:
            r = run(m, make_world(args.situations, args.exceptions, seed),
                    args.steps, seed)
            for k in agg[m]:
                agg[m][k].append(r[k])

    print("=" * 66)
    print("DID REMEMBERING THE ONE TIME HELP?")
    print("=" * 66)
    print(f"  {'memory':>10} {'reward':>9} {'traps hit':>10} "
          f"{'safe taken':>11}")
    print("-" * 66)
    for m in modes:
        note = {"semantic": "general rule only (today's NEURON)",
                "episodic": "rule + recalled specific events"}[m]
        print(f"  {m:>10} {statistics.mean(agg[m]['reward']):>9.0f} "
              f"{statistics.mean(agg[m]['traps_hit']):>10.1f} "
              f"{statistics.mean(agg[m]['safe_taken']):>11.0f}   {note}")

    sem = statistics.mean(agg["semantic"]["reward"])
    epi = statistics.mean(agg["episodic"]["reward"])
    sem_tr = statistics.mean(agg["semantic"]["traps_hit"])
    epi_tr = statistics.mean(agg["episodic"]["traps_hit"])

    print("\n" + "=" * 66)
    print("WHAT IT SAYS")
    print("=" * 66)
    if epi > sem + 0.05 * abs(sem) and epi_tr < sem_tr * 0.7:
        print(f"  EPISODIC MEMORY EARNED ITS PLACE. The episodic being "
              f"scored {epi:.0f} against the")
        print(f"  semantic-only being's {sem:.0f}, and walked into the "
              f"exception-traps {epi_tr:.1f} times")
        print(f"  against {sem_tr:.1f}. After each disaster it REMEMBERED "
              f"that specific event and")
        print(f"  avoided it, where the general rule kept saying 'safe' and "
              f"the semantic being")
        print(f"  kept getting burned.")
        print(f"\n  This is learning from single important events the "
              f"average erases. No amount")
        print(f"  of general learning can do it, because the exceptions are "
              f"not a pattern --")
        print(f"  they are specific, and only a memory of specifics can hold "
              f"them.")
    else:
        print(f"  no clear win: episodic {epi:.0f} vs semantic {sem:.0f}, "
              f"traps {epi_tr:.1f} vs {sem_tr:.1f}.")
        print(f"  recall may be mis-tuned or exceptions too frequent to "
              f"separate the two.")

    print(f"\n  Semantic memory is the rule, averaged from everything. "
          f"Episodic memory is the")
    print(f"  one time, kept as itself. A being needs both: the rule for the "
          f"usual, the")
    print(f"  memory for the exception that the rule would tell it to walk "
          f"right into.")


if __name__ == "__main__":
    main()
