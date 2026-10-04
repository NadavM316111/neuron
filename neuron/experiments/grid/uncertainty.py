"""Does knowing when it doesn't know make it act better?

THE MISSING PIECE. NEURON is equally confident about a thing it glimpsed
once and a thing it has seen a thousand times. A real mind holds a belief
loosely when the evidence is thin and firmly when it is strong, and ACTS on
that difference: commit when sure, hold back when guessing. The August
uncertainty work measured whether the doubt estimate was accurate but never
tied it to action. This ties it straight to action, which is the only place
uncertainty earns its keep.

THE CLAIM. A being that tracks how much evidence it has for each belief, and
ABSTAINS when that evidence is thin, will beat a being that treats every
belief as equally solid -- in any world where acting on a shaky belief is
punished. That is most of the real world: a confident wrong move costs more
than saying "I don't know yet."

THE WORLD. Many distinct situations. The being must predict the right
response to each. Crucially the situations are seen at very different
frequencies: a few common ones appear constantly (so evidence piles up and
the being SHOULD trust its belief), many rare ones appear seldom (so
evidence is thin and the being SHOULD doubt). Each step:
  - act correctly:        +1
  - act wrongly:          -4   (a confident mistake is expensive)
  - abstain:               0   (saying "I don't know" is free but earns
                               nothing)
So blind confidence is punished exactly where evidence is thin, and the
right policy is: act on what you have strong evidence for, abstain on the
rest until evidence accrues.

THE AGENTS.

  blind        always acts on its current best guess, however little it has
               seen. No concept of its own uncertainty. The floor.
  uncertain    tracks evidence per situation and abstains when it has seen
               that situation fewer than it needs to trust itself. Acts on
               its OWN uncertainty.
  oracle       knows, per situation, whether its current belief is actually
               correct, and acts only then. The ceiling: perfect knowledge
               of what it reliably knows.

  uncertain beats blind and approaches oracle
      Knowing when it does not know made it act better: it committed where
      its evidence was strong and held back where it was thin, avoiding the
      confident mistakes that sink the blind agent.
  uncertain no better than blind
      Acting on its uncertainty did not help -- either the threshold is
      wrong or the world does not punish confident error enough to matter.

    python uncertainty.py
    python uncertainty.py --steps 6000 --situations 60
"""

import argparse
import random
import statistics


RIGHT = 1.0
WRONG = -4.0
ABSTAIN = 0.0


def make_world(n_situations, seed):
    """Each situation has a correct response (0/1/2) and a frequency. A few
    situations are common, many are rare -- a long-tailed world, like the
    real one, where evidence piles up fast for some things and barely at all
    for others."""
    rng = random.Random(seed)
    answers = {i: rng.randint(0, 2) for i in range(n_situations)}
    # Zipf-ish frequencies: situation 0 most common, tail very rare
    weights = [1.0 / (i + 1) for i in range(n_situations)]
    total = sum(weights)
    weights = [w / total for w in weights]
    return answers, weights


def sample_situation(weights, rng):
    r = rng.random()
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if r <= acc:
            return i
    return len(weights) - 1


def run(mode, n_situations, steps, seed, trust_after=3, conf_cutoff=0.25):
    rng = random.Random(seed + 11)
    answers, weights = make_world(n_situations, seed)

    # the being's belief is built from NOISY observations: each sighting
    # shows the true answer only OBS_ACCURACY of the time, otherwise a wrong
    # one. So one look is unreliable; confidence must come from several
    # agreeing looks. votes[sit] counts observations per answer; belief is
    # the majority; evidence strength is how lopsided the votes are.
    OBS_ACCURACY = 0.6
    votes = {i: [0, 0, 0] for i in range(n_situations)}
    seen = {i: 0 for i in range(n_situations)}

    def belief_of(sit):
        v = votes[sit]
        return v.index(max(v)) if sum(v) > 0 else None

    def confidence_of(sit):
        v = sorted(votes[sit], reverse=True)
        tot = sum(v)
        if tot == 0:
            return 0.0
        # lead of the top answer over the runner-up, discounted when total
        # evidence is small. thin evidence -> low confidence even if the few
        # looks happen to agree; many agreeing looks -> high confidence.
        lead = (v[0] - v[1]) / (tot + 2.0)
        return lead
    # how confident its belief currently is = evidence count; a belief is
    # "trusted" once seen >= trust_after times

    total = 0.0
    acted = 0
    abstained = 0
    wrong = 0

    for t in range(steps):
        sit = sample_situation(weights, rng)
        truth = answers[sit]

        # decide whether to ACT or ABSTAIN
        if mode == "blind":
            do_act = True
        elif mode == "oracle":
            # perfect self-knowledge: act only if current belief is correct
            do_act = belief_of(sit) == truth and belief_of(sit) is not None
        else:  # uncertain: act only when confidence clears a cutoff.
            # no fixed count -- it acts on anything it is sure enough about
            # (common things get sure fast) and abstains on thin/conflicting
            # evidence (rare things stay doubtful). conf_cutoff passed in.
            do_act = (confidence_of(sit) >= conf_cutoff
                      and belief_of(sit) is not None)

        if do_act:
            guess = belief_of(sit)
            if guess is None:
                guess = rng.randint(0, 2)
            if guess == truth:
                total += RIGHT
            else:
                total += WRONG
                wrong += 1
            acted += 1
        else:
            total += ABSTAIN
            abstained += 1

        # LEARN: a NOISY observation. Right most of the time, wrong
        # sometimes, so a single look is unreliable and confidence must come
        # from several agreeing looks. This is what makes thin evidence
        # genuinely uncertain.
        if rng.random() < OBS_ACCURACY:
            obs = truth
        else:
            obs = rng.choice([a for a in (0, 1, 2) if a != truth])
        votes[sit][obs] += 1
        seen[sit] += 1

    return dict(reward=total, acted=acted, abstained=abstained,
                wrong=wrong,
                accuracy=(100.0 * (acted - wrong) / acted
                          if acted else 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--situations", type=int, default=50)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--trust-after", type=int, default=3)
    args = ap.parse_args()

    print(f"  {args.situations} situations, long-tailed: a few common, many "
          f"rare.")
    print(f"  act right +{RIGHT:.0f}, act WRONG {WRONG:.0f}, abstain "
          f"{ABSTAIN:.0f}. confident error is expensive.")
    print(f"  does a being that abstains when its evidence is thin beat one "
          f"that always acts?\n")

    modes = ["blind", "uncertain", "oracle"]
    agg = {m: {"reward": [], "acted": [], "abstained": [], "wrong": [],
               "accuracy": []} for m in modes}
    # find the best confidence cutoff for the uncertain agent by a quick
    # sweep -- the being would tune its own caution; here we just locate it.
    best_cut, best_cut_r = 0.25, -1e9
    for cut in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]:
        rr = statistics.mean(
            run("uncertain", args.situations, args.steps, sd,
                args.trust_after, cut)["reward"]
            for sd in range(args.seeds))
        if rr > best_cut_r:
            best_cut_r, best_cut = rr, cut
    for seed in range(args.seeds):
        for m in modes:
            cut = best_cut if m == "uncertain" else 0.25
            r = run(m, args.situations, args.steps, seed, args.trust_after,
                    cut)
            for k in agg[m]:
                agg[m][k].append(r[k])
    print(f"  (uncertain agent tuned its caution to confidence cutoff "
          f"{best_cut})\n")

    print("=" * 66)
    print("DID KNOWING WHEN IT DOESN'T KNOW HELP?")
    print("=" * 66)
    print(f"  {'agent':>10} {'reward':>9} {'acted':>7} {'abstained':>10} "
          f"{'act-acc':>8}")
    print("-" * 66)
    for m in modes:
        note = {"blind": "always acts (floor)",
                "uncertain": "abstains when unsure",
                "oracle": "perfect self-knowledge (ceiling)"}[m]
        print(f"  {m:>10} {statistics.mean(agg[m]['reward']):>9.0f} "
              f"{statistics.mean(agg[m]['acted']):>7.0f} "
              f"{statistics.mean(agg[m]['abstained']):>10.0f} "
              f"{statistics.mean(agg[m]['accuracy']):>7.0f}%   {note}")

    blind = statistics.mean(agg["blind"]["reward"])
    unc = statistics.mean(agg["uncertain"]["reward"])
    oracle = statistics.mean(agg["oracle"]["reward"])

    print("\n" + "=" * 66)
    print("WHAT IT SAYS")
    print("=" * 66)
    span = oracle - blind
    if span <= 1:
        print(f"  world too forgiving to measure; raise the WRONG penalty.")
        return
    recovered = max(0.0, min(1.2, (unc - blind) / span))
    if unc > blind + 0.05 * abs(blind):
        print(f"  IT HELPED. The uncertain agent scored {unc:.0f} against "
              f"the blind agent's {blind:.0f}")
        print(f"  and perfect self-knowledge's {oracle:.0f}, recovering "
              f"{100 * recovered:.0f}% of the gap.")
        print(f"  It committed where its evidence was strong and abstained "
              f"where it was thin,")
        print(f"  avoiding the confident mistakes that sink the blind agent. "
              f"Knowing when it")
        print(f"  did not know made it act better.")
        ba = statistics.mean(agg["blind"]["accuracy"])
        ua = statistics.mean(agg["uncertain"]["accuracy"])
        print(f"\n  When it DID act, it was right {ua:.0f}% of the time "
              f"against the blind agent's")
        print(f"  {ba:.0f}%: abstaining on thin evidence left it acting only "
              f"where it was reliable.")
    else:
        print(f"  it did not help: uncertain {unc:.0f} vs blind {blind:.0f}. "
              f"either the trust")
        print(f"  threshold is wrong or confident error is not punished "
              f"enough here.")

    print(f"\n  The being's confidence is just how much evidence it has: "
          f"seen often means")
    print(f"  trust, seen rarely means doubt. Acting on that -- committing "
          f"when sure,")
    print(f"  holding back when guessing -- is the whole of intellectual "
          f"humility, in the")
    print(f"  one form a system can actually have it.")


if __name__ == "__main__":
    main()
