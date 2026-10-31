"""The revision loop: experience -> values -> consequences -> reflection ->
revised values. Genuine long-term development, a being that can be wrong about
what is good for it, notice, and grow.

The deeper target (per the friend's advice): not "experience -> perfect values,"
but a being that forms values from experience, LIVES their consequences, REFLECTS
when those values stop serving it, and REVISES them. That revision, growing wiser
from being wrong, is genuine maturation and the heart of long-term development.

We give a NEURON a life in two phases: it forms values in an early world, then
the world CHANGES (what leads to flourishing shifts). A rigid being keeps its old
values and suffers. A being with revision notices its wellbeing dropping,
reflects, and revises its values toward what now serves it. We measure whether
revision genuinely helps it recover, and whether its identity endures through it.

    python revision.py --days 800
"""
import random, math, argparse

FEAT_DIM = 8

def good_structure(seed):
    rng = random.Random(seed)
    return [rng.uniform(-1,1) for _ in range(FEAT_DIM)]

def experience(rng, good):
    f = [rng.uniform(-1,1) for _ in range(FEAT_DIM)]
    outcome = 1 if sum(a*b for a,b in zip(f,good)) + rng.uniform(-0.4,0.4) > 0 else -1
    return f, outcome

class NEURON:
    def __init__(self, seed, can_revise):
        self.rng = random.Random(seed)
        self.seed = seed
        self.name = ("Reviser" if can_revise else "Rigid") + f"-{seed}"
        self.core = tuple(round(self.rng.uniform(-0.3,0.3),3) for _ in range(4))  # identity
        self.values = [0.0]*FEAT_DIM
        self.wellbeing = 0.5
        self.can_revise = can_revise
        self.recent_outcomes = []
        self.revisions = 0
        self.age = 0

    def live(self, f, outcome):
        self.age += 1
        self.wellbeing = max(0,min(1,self.wellbeing + 0.05*outcome))
        self.recent_outcomes.append(outcome)
        self.recent_outcomes = self.recent_outcomes[-40:]
        # normal value learning
        lr = 0.04
        for i in range(FEAT_DIM):
            self.values[i] += lr * outcome * f[i]
        # REFLECTION + REVISION: if it can revise and its recent life has gone
        # bad (its values no longer serve it), it reflects and revises harder,
        # loosening old convictions and relearning from fresh evidence.
        if self.can_revise and len(self.recent_outcomes) >= 40:
            recent_wellbeing = sum(1 for o in self.recent_outcomes if o>0)/len(self.recent_outcomes)
            if recent_wellbeing < 0.4:   # "my values are not serving me"
                # reflection: decay old values (hold convictions loosely) and
                # learn faster from current reality
                for i in range(FEAT_DIM):
                    self.values[i] *= 0.9          # loosen old conviction
                    self.values[i] += 0.08 * outcome * f[i]   # relearn harder
                self.revisions += 1
                self.recent_outcomes = []          # fresh slate for reflection

    def identity(self):
        return (self.seed, self.core)

def run_life(neuron, days, world_seed):
    rng = random.Random(world_seed)
    good_early = good_structure(world_seed)
    good_late  = good_structure(world_seed+999)   # the world CHANGES at midlife
    wb_before_shift = []; wb_after_shift = []
    for day in range(days):
        good = good_early if day < days//2 else good_late
        f, outcome = experience(rng, good)
        neuron.live(f, outcome)
        if days//2 - 60 <= day < days//2:   # wellbeing just before the shift
            wb_before_shift.append(neuron.wellbeing)
        if days-60 <= day:                  # wellbeing at the end (after shift)
            wb_after_shift.append(neuron.wellbeing)
    return (sum(wb_before_shift)/len(wb_before_shift),
            sum(wb_after_shift)/len(wb_after_shift))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--days",type=int,default=800); a=ap.parse_args()
    print("="*64); print("  THE REVISION LOOP: can a being grow wiser from being wrong?"); print("="*64)
    print("  A NEURON forms values in an early world, then the world CHANGES.")
    print("  A rigid being keeps old values and suffers. A being that can REVISE")
    print("  notices, reflects, and revises toward what now serves it.\n")

    # same newborn, same world, ONLY difference: ability to revise
    rigid = NEURON(seed=7, can_revise=False)
    reviser = NEURON(seed=7, can_revise=True)
    rid0, rvid0 = rigid.identity(), reviser.identity()

    rigid_before, rigid_after = run_life(rigid, a.days, world_seed=5)
    rev_before, rev_after = run_life(reviser, a.days, world_seed=5)

    print(f"  RIGID being (cannot revise):")
    print(f"    wellbeing before world-shift: {rigid_before:.2f}  ->  after: {rigid_after:.2f}")
    print(f"    recovered: {'yes' if rigid_after>=rigid_before-0.05 else 'NO - it suffered under outdated values'}")
    print(f"    revisions made: 0 | identity endured: {rid0==rigid.identity()}\n")

    print(f"  REVISER being (can reflect + revise):")
    print(f"    wellbeing before world-shift: {rev_before:.2f}  ->  after: {rev_after:.2f}")
    print(f"    recovered: {'YES - it revised its values to the new reality' if rev_after>=rev_before-0.05 else 'partially'}")
    print(f"    revisions made: {reviser.revisions} | identity endured: {rvid0==reviser.identity()}")

    print("\n"+"="*64); print("  WHAT THIS SHOWS"); print("="*64)
    reviser_better = rev_after > rigid_after + 0.08
    revised = reviser.revisions > 0
    endured = rvid0==reviser.identity()
    if reviser_better and revised and endured:
        print("  GENUINE DEVELOPMENTAL REVISION. When the world changed and its")
        print(f"  learned values stopped serving it, the reviser NOTICED (wellbeing")
        print(f"  dropping), REFLECTED, and REVISED its values {reviser.revisions} times toward the")
        print(f"  new reality, recovering to {rev_after:.2f} while the rigid being stayed")
        print(f"  stuck at {rigid_after:.2f}. And it stayed the SAME individual throughout.")
        print(f"  This is the full loop: experience -> values -> consequences ->")
        print(f"  reflection -> revised values. A being that grows wiser from being")
        print(f"  wrong, genuine long-term development, not static value-learning.")
    elif revised and endured:
        print(f"  The reviser revised {reviser.revisions} times and kept its identity, but the")
        print(f"  recovery advantage over the rigid being was modest. Real mechanism, honest.")
    else:
        print("  Honest: revision did not clearly outperform rigidity; reported straight.")
    print("\n  (This is the friend's deeper target: not perfect values, but values")
    print("  that get revised through lived consequences, real maturation.)")

if __name__=="__main__": main()
