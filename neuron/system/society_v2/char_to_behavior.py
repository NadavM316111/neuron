"""Character -> Behavior causality (the friend's test): does changing WHO a NEURON
has become actually change how it behaves? This isolates the broken arrow.

Same NEURON, same world state, same neighbors, same resources, same random seed.
Intervene on ONLY the value profile:
  A: mastery .80 / connection .10 / peace .25
  B: mastery .10 / connection .80 / peace .25
  C: mastery .10 / connection .25 / peace .80
Measure the DISTRIBUTION of decisions each makes across many identical situations:
  work_alone, approach_someone, cooperate, share, avoid_risk, maintain_tie,
  explore_new_tie, resolve_conflict, accumulate.

If A/B/C decide almost IDENTICALLY -> character isn't coupled to action (THE bug).
If they decide DIFFERENTLY -> character drives behavior, and the society problem is
elsewhere. Honest either way.

    python char_to_behavior.py
"""
import random, statistics
from collections import defaultdict

ACTIONS=["work_alone","approach_someone","cooperate","share_resources","avoid_risk",
         "maintain_tie","explore_new_tie","resolve_conflict","accumulate"]

def decide(char, situation, rng):
    """the being's decision policy, as a function of its CHARACTER and the situation.
    situation = dict(neighbors, has_soured_tie, resource_level, risky_option).
    Returns one action. This is the policy whose character-sensitivity we test."""
    m,c,p = char["mastery"],char["connection"],char["peace"]
    # score each action by how the character values what it achieves
    score={}
    score["work_alone"]      = m*1.0 - c*0.3
    score["accumulate"]      = m*0.8
    score["approach_someone"]= c*1.0 - p*0.1
    score["explore_new_tie"] = c*0.9 - m*0.2
    score["cooperate"]       = c*0.7 + p*0.3
    score["share_resources"] = c*0.6 + p*0.4 - m*0.3
    score["maintain_tie"]    = c*0.5 + p*0.3
    score["resolve_conflict"]= p*1.0 * (1.2 if situation.get("has_soured_tie") else 0.4)
    score["avoid_risk"]      = p*0.9 * (1.3 if situation.get("risky_option") else 0.5) - m*0.2
    # situation gating
    if not situation.get("neighbors"):
        for k in ("approach_someone","cooperate","share_resources","maintain_tie","explore_new_tie","resolve_conflict"):
            score[k]-=1.0
    # softmax-ish stochastic choice (same rng stream across A/B/C for fairness)
    items=list(score.items()); mx=max(v for _,v in items)
    weights=[(k,max(0.001, 2.718**((v-mx)*2))) for k,v in items]
    tot=sum(w for _,w in weights); r=rng.random()*tot; acc=0
    for k,w in weights:
        acc+=w
        if r<=acc: return k
    return items[-1][0]

def profile_behavior(char, trials=2000):
    counts=defaultdict(int)
    for t in range(trials):
        rng=random.Random(t)  # SAME seed per trial across all three characters
        situation={"neighbors":rng.random()<0.8,"has_soured_tie":rng.random()<0.3,
                   "resource_level":rng.random(),"risky_option":rng.random()<0.4}
        a=decide(char, situation, random.Random(t*997+1))  # same decision seed too
        counts[a]+=1
    return {a:counts[a]/trials for a in ACTIONS}

def main():
    print("="*64); print("  CHARACTER -> BEHAVIOR: does character change decisions?"); print("="*64)
    A={"mastery":.80,"connection":.10,"peace":.25}
    B={"mastery":.10,"connection":.80,"peace":.25}
    C={"mastery":.10,"connection":.25,"peace":.80}
    ba,bb,bc=profile_behavior(A),profile_behavior(B),profile_behavior(C)

    print(f"\n  decision distribution (same world, same seeds, only character differs):")
    print(f"  {'action':<18}{'A:mastery':>11}{'B:conn':>9}{'C:peace':>9}")
    for a in ACTIONS:
        print(f"  {a:<18}{ba[a]*100:>10.0f}%{bb[a]*100:>8.0f}%{bc[a]*100:>8.0f}%")

    # how DIFFERENT are the three behavior profiles? (total variation distance)
    def tvd(x,y): return sum(abs(x[a]-y[a]) for a in ACTIONS)/2
    dab,dac,dbc=tvd(ba,bb),tvd(ba,bc),tvd(bb,bc)
    avg_diff=(dab+dac+dbc)/3
    print(f"\n  behavioral difference between characters (total-variation distance, 0=identical, 1=disjoint):")
    print(f"    mastery vs connection: {dab:.2f}")
    print(f"    mastery vs peace:      {dac:.2f}")
    print(f"    connection vs peace:   {dbc:.2f}")
    print(f"    average:               {avg_diff:.2f}")

    print("\n"+"="*64); print("  VERDICT"); print("="*64)
    if avg_diff>0.4:
        print(f"  CHARACTER STRONGLY DRIVES BEHAVIOR. The same NEURON in the same")
        print(f"  situations makes very different decisions depending only on its")
        print(f"  character (avg behavioral distance {avg_diff:.2f}). A mastery-self works alone and")
        print(f"  accumulates; a connection-self approaches and explores ties; a peace-self")
        print(f"  avoids risk and resolves conflict. The character->decision arrow is")
        print(f"  CAUSAL and strong. So the society weakness is in the LATER arrows")
        print(f"  (decisions -> social position), not here. Rerun society with this.")
    elif avg_diff>0.2:
        print(f"  PARTIAL: character changes behavior moderately (distance {avg_diff:.2f}) but not")
        print(f"  dramatically. Real coupling, but maybe too weak to produce strong social")
        print(f"  structure. Strengthen the coupling, then rerun society.")
    else:
        print(f"  THE BUG, FOUND: character barely changes behavior (distance {avg_diff:.2f}). Three")
        print(f"  wildly different characters make nearly IDENTICAL decisions - so character")
        print(f"  exists internally but isn't coupled to action. THIS is why the society")
        print(f"  didn't differentiate. Fix: make the decision policy genuinely character-")
        print(f"  sensitive before any more society work. Honest - the real broken arrow.")

if __name__=="__main__": main()
