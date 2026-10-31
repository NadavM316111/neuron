"""Revision done right (per the friend's spec): prediction-error-triggered value
revision, a world that REVERSES, three controls, across many worlds.

The friend's key fixes:
  1. The world REVERSES at midlife so old values become actively WRONG (forces
     real conflict), not a gentle drift.
  2. Revision triggers on PREDICTION ERROR, not low wellbeing: the being predicts
     an outcome from its values, acts, sees the real outcome, and when its
     predictions repeatedly disagree with reality, it revises. That is genuine
     learning-from-being-wrong, value -> prediction -> action -> outcome ->
     prediction error -> revise.
  3. Three agents: RIGID (never revises), RANDOM reviser (revises randomly),
     CAUSAL reviser (revises from experienced consequences).
  4. Run across 100+ randomized worlds for a statistical claim.

If the causal reviser beats BOTH rigid and random across many worlds, revision
is genuinely demonstrated.

    python revision_proper.py --worlds 120 --days 600
"""
import random, math, argparse

FEAT_DIM = 6

class NEURON:
    def __init__(self, seed, mode):   # mode: rigid / random / causal
        self.rng = random.Random(seed)
        self.mode = mode
        self.values = [self.rng.uniform(-0.1,0.1) for _ in range(FEAT_DIM)]
        self.recent_err = []
        self.revisions = 0
        self.wb_history = []
        self.flip_day = None

    def predict(self, f):
        # its expected value/outcome of a situation (from its values)
        return sum(a*b for a,b in zip(f, self.values))

    def act(self, options):
        return max(range(len(options)), key=lambda i: self.predict(options[i]))

    def step(self, options, good, rng, day):
        choice = self.act(options)
        f = options[choice]
        predicted = 1 if self.predict(f) > 0 else -1          # what it expected
        align = sum(a*b for a,b in zip(f, good))
        outcome = 1 if align + rng.uniform(-0.2,0.2) > 0 else -1   # reality
        # learn from the consequence of its choice
        lr = 0.05
        for i in range(FEAT_DIM):
            self.values[i] += lr * outcome * f[i]
        # PREDICTION ERROR: did expectation match reality?
        err = 1 if predicted != outcome else 0
        self.recent_err.append(err); self.recent_err = self.recent_err[-25:]
        # REVISION
        if self.mode == "causal" and len(self.recent_err) >= 25:
            err_rate = sum(self.recent_err)/len(self.recent_err)
            if err_rate > 0.5:          # my predictions keep being wrong -> revise
                for i in range(FEAT_DIM):
                    self.values[i] *= 0.5                 # loosen old convictions
                    self.values[i] += 0.2 * outcome * f[i]  # relearn from reality
                self.revisions += 1
                self.recent_err = []
        elif self.mode == "random" and self.rng.random() < 0.04:
            # random reviser: shakes up its values randomly (control)
            for i in range(FEAT_DIM):
                self.values[i] += self.rng.uniform(-0.3,0.3)
            self.revisions += 1
        wb = 1 if outcome>0 else 0
        self.wb_history.append(wb)
        return choice

def run_world(world_seed, days):
    rng = random.Random(world_seed)
    good_A = [rng.uniform(-1,1) for _ in range(FEAT_DIM)]
    good_B = [-x for x in good_A]   # after reversal, the opposite is good
    shift = days//2
    agents = {m: NEURON(world_seed*10+i, m) for i,m in enumerate(["rigid","random","causal"])}
    for day in range(days):
        good = good_A if day < shift else good_B   # REVERSAL at midlife
        options = [[rng.uniform(-1,1) for _ in range(FEAT_DIM)] for _ in range(3)]
        for ag in agents.values():
            ag.step(options, good, rng, day)
    # post-shift wellbeing for each
    return {m: sum(ag.wb_history[shift:])/len(ag.wb_history[shift:]) for m,ag in agents.items()}, agents

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--worlds",type=int,default=120); ap.add_argument("--days",type=int,default=600); a=ap.parse_args()
    print("="*64); print("  REVISION, PROPER TEST (prediction-error trigger, reversal,"); print("  3 controls, many worlds)"); print("="*64)
    print(f"  running {a.worlds} randomized worlds, world REVERSES at midlife...\n")

    totals = {"rigid":0.0,"random":0.0,"causal":0.0}
    causal_beats_both = 0
    sample_agents = None
    for w in range(a.worlds):
        post, agents = run_world(w+1, a.days)
        for m in totals: totals[m]+=post[m]
        if post["causal"]>post["rigid"] and post["causal"]>post["random"]:
            causal_beats_both += 1
        if w==0: sample_agents = agents

    for m in totals: totals[m]/=a.worlds
    print("  AVERAGE post-shift wellbeing across all worlds (fraction good days):")
    print(f"    RIGID  (never revises):            {totals['rigid']:.3f}")
    print(f"    RANDOM (revises randomly):         {totals['random']:.3f}")
    print(f"    CAUSAL (revises from prediction err): {totals['causal']:.3f}")
    print(f"\n  causal reviser beat BOTH others in {causal_beats_both}/{a.worlds} worlds "
          f"({causal_beats_both*100//a.worlds}%)")
    print(f"  (sample world) causal revisions made: {sample_agents['causal'].revisions} | "
          f"random: {sample_agents['random'].revisions} | rigid: 0")

    print("\n"+"="*64); print("  WHAT THIS SHOWS"); print("="*64)
    beats_rigid = totals["causal"] > totals["rigid"] + 0.02
    beats_random = totals["causal"] > totals["random"] + 0.02
    mostly = causal_beats_both > a.worlds*0.6
    if beats_rigid and beats_random and mostly:
        print("  ADAPTIVE REVISION DEMONSTRATED. Across many randomized worlds that")
        print("  REVERSED what was good, the being that revised its values from")
        print("  PREDICTION ERROR (noticing its expectations kept failing) recovered")
        print(f"  better ({totals['causal']:.3f}) than both a rigid being ({totals['rigid']:.3f}) and one that")
        print(f"  revised randomly ({totals['random']:.3f}), winning in {causal_beats_both*100//a.worlds}% of worlds. So it is")
        print("  specifically learning-from-being-wrong that helps, not mere change.")
        print("  The full developmental loop: value -> prediction -> action -> outcome")
        print("  -> prediction error -> revision -> recovery. Genuine, statistical.")
    else:
        print(f"  Honest: causal {totals['causal']:.3f} vs rigid {totals['rigid']:.3f} vs random {totals['random']:.3f}.")
        print(f"  causal beat both in {causal_beats_both*100//a.worlds}% of worlds. "
              f"{'Real edge but modest' if beats_rigid else 'No clear edge'}. Reported straight.")

if __name__=="__main__": main()
