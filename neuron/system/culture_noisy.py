"""Does culture survive real-world messiness? (the friend's philosophy: add the
uncertainties a real culture faces, then let the experiment tell us where it
breaks, don't tune toward a target.)

Keeps the clean mechanism as benchmark, then layers in REAL uncertainty:
  - imperfect observation (newborn misses some interactions)
  - ambiguous approval (reactions sometimes noisy/unclear)
  - rule-breakers (some members violate the norm though they know it)
  - limited childhood exposure (newborn sees few examples)
  - conflicting subgroups (e.g. 70/30 split on the convention)
  - memory errors (observations stored imperfectly)
  - generational mutation (each transmission can drift)

Reports, across noise levels: inference accuracy, and CONFIDENCE (a newborn that
saw consistent examples is sure; one that saw a split is unsure). Then a
multi-generation test: can the culture survive being passed through generations
of imperfect learners? No tuning, we report what happens.

    python culture_noisy.py --seeds 150
"""
import random, math, argparse, statistics

N_BEHAV=4
_id=[0]
def nid(): _id[0]+=1; return _id[0]

def argmax_random(vals, rng):
    m=max(vals); best=[i for i,v in enumerate(vals) if abs(v-m)<1e-9]
    return rng.choice(best)

class Learner:
    """Observes a group's behavior and infers the norm + its confidence."""
    def __init__(self, seed):
        self.rng=random.Random(seed)
        self.approval_mem=[0.0]*N_BEHAV
        self.counts=[0]*N_BEHAV
        self.obs=0
    def observe(self, behavior, approved, memory_error):
        # memory error: sometimes misremember which behavior it was
        if self.rng.random()<memory_error:
            behavior=self.rng.randrange(N_BEHAV)
        self.obs+=1; self.counts[behavior]+=1
        w=1.0 if approved else -1.0
        self.approval_mem[behavior]=self.approval_mem[behavior]+w
    def inferred(self):
        return argmax_random(self.approval_mem, self.rng)
    def confidence(self):
        # confidence = how dominant the top approved behavior is over the rest.
        # consistent examples -> high; a split -> low. realistic uncertainty.
        pos=[max(0,v) for v in self.approval_mem]; tot=sum(pos)
        if tot<=0 or self.obs==0: return 0.0
        return max(pos)/tot

def build_group(seed, noise):
    """Make a mature group with a convention, possibly with a dissenting subgroup."""
    rng=random.Random(seed)
    convention=rng.randrange(N_BEHAV)
    # conflicting subgroup: a fraction follow a DIFFERENT convention
    alt=rng.randrange(N_BEHAV)
    members=[]
    for i in range(14):
        if noise["subgroup"]>0 and rng.random()<noise["subgroup"]:
            members.append(alt)       # dissenter
        else:
            members.append(convention)
    return convention, members, rng

def observe_group(learner, convention, members, noise, n_examples, rng):
    for _ in range(n_examples):
        # imperfect observation: sometimes miss the interaction entirely
        if rng.random()<noise["miss"]: continue
        actor_behavior=rng.choice(members)
        # rule-breakers: a known member sometimes violates anyway
        if rng.random()<noise["breakers"]:
            actor_behavior=rng.randrange(N_BEHAV)
        # approval: the group approves if actor matches the MAJORITY convention,
        # but ambiguous/noisy approval sometimes flips the signal
        true_approve = (actor_behavior==convention)
        if rng.random()<noise["ambiguous"]:
            approve = rng.random()<0.5    # unclear reaction
        else:
            approve = true_approve
        learner.observe(actor_behavior, approve, noise["memory"])

def run_level(seeds, noise, n_examples):
    correct=0; conf_when_right=[]; conf_when_wrong=[]; total=0
    for s in range(seeds):
        conv, members, rng = build_group(s*41+7, noise)
        L=Learner(s*999)
        observe_group(L, conv, members, noise, n_examples, rng)
        total+=1
        if L.inferred()==conv:
            correct+=1; conf_when_right.append(L.confidence())
        else:
            conf_when_wrong.append(L.confidence())
    return (correct/total if total else 0,
            statistics.mean(conf_when_right) if conf_when_right else 0,
            statistics.mean(conf_when_wrong) if conf_when_wrong else 0)

def run_generations(seed, noise, gens=6, n_examples=25):
    """founders have a convention; each generation learns from the PREVIOUS one
    (imperfectly), then becomes the teachers. does the culture survive?"""
    rng=random.Random(seed)
    convention=rng.randrange(N_BEHAV)
    members=[convention]*14
    history=[1.0]   # fraction holding the original convention
    orig=convention
    for g in range(gens):
        new_members=[]
        for _ in range(14):
            L=Learner(rng.randint(0,10**9))
            observe_group(L, max(set(members),key=members.count), members, noise, n_examples, rng)
            inferred=L.inferred()
            # generational mutation: small chance the learned norm drifts
            if rng.random()<noise["mutation"]:
                inferred=rng.randrange(N_BEHAV)
            new_members.append(inferred)
        members=new_members
        history.append(members.count(orig)/len(members))
    return history

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seeds",type=int,default=150); a=ap.parse_args()
    print("="*64); print("  DOES CULTURE SURVIVE REAL MESSINESS? (no tuning, let it break)"); print("="*64)

    levels={
     "clean":   dict(miss=0, ambiguous=0, breakers=0, subgroup=0, memory=0, mutation=0),
     "mild":    dict(miss=0.1, ambiguous=0.1, breakers=0.1, subgroup=0.15, memory=0.05, mutation=0.03),
     "moderate":dict(miss=0.25, ambiguous=0.25, breakers=0.2, subgroup=0.25, memory=0.15, mutation=0.08),
     "heavy":   dict(miss=0.4, ambiguous=0.4, breakers=0.3, subgroup=0.35, memory=0.3, mutation=0.15),
    }
    exposure={"clean":40,"mild":25,"moderate":15,"heavy":10}

    print("\n  SINGLE-LEARNER INFERENCE across noise levels:")
    print(f"  {'level':<10}{'inference':<12}{'confidence(right)':<18}{'confidence(wrong)'}")
    for name,noise in levels.items():
        acc,cr,cw=run_level(a.seeds, noise, exposure[name])
        print(f"  {name:<10}{acc*100:>6.0f}%     {cr:>8.2f}          {cw:>8.2f}")

    print("\n  (confidence should be HIGHER when the learner is right and LOWER")
    print("   when wrong/unsure, realistic uncertainty, not forced certainty.)")

    print("\n  MULTI-GENERATION SURVIVAL (culture passed through imperfect learners):")
    for name in ["mild","moderate","heavy"]:
        survivals=[run_generations(s*7+1, levels[name])[-1] for s in range(a.seeds)]
        print(f"    {name:<9}: original convention still held by "
              f"{statistics.mean(survivals)*100:>5.0f}% of the population after 6 generations")

    print("\n"+"="*64); print("  WHAT THIS SHOWS"); print("="*64)
    print("  This reports where culture holds and where it breaks under real")
    print("  uncertainty, imperfect observation, ambiguity, rule-breakers, limited")
    print("  exposure, dissenting subgroups, memory errors, generational drift. The")
    print("  confidence metric lets a learner be UNSURE rather than forced. And the")
    print("  multi-generation test asks the real question: can a culture survive")
    print("  being passed, imperfectly, through generations? Whatever the numbers,")
    print("  they are the honest behavior of the system under realistic stress.")

if __name__=="__main__": main()
