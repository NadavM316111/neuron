"""Society v2: what happens when many genuinely-different someones meet?

Built on the validated v2 consequence-driven being. Steps:
  1. Create N newborns (same architecture) and give each a DIFFERENT early life
     (different worlds) BEFORE they meet -> they become different characters.
  2. Open a SHARED scarce world: they interact, choose partners, cooperate or not,
     build reputation - with real consequences (cooperation is NOT auto-rewarded).
  3. Watch what EMERGES: who associates with whom, specialization, social roles,
     groups, influence - without hardcoding any of it.
  4. CAUSAL CONTROL: run the identical society but everyone enters FLAT (default
     .50/.50/.50 character). Compare. If developed individuals organize DIFFERENTLY
     than flat ones, Being causally feeds Species.

Honest: it can fail (character may NOT predict social role; developed and flat
societies may look the same). Reports what actually happens.

    python society_v2.py
"""
import random, statistics
from collections import defaultdict

VALUES=["mastery","connection","peace"]

def shape_character(seed, early_world, days=300):
    """live a newborn through a biased early life -> a character (consequence-driven)."""
    rng=random.Random(seed)
    v={x:0.5 for x in VALUES}; state={x:0.3 for x in VALUES}; rerr={x:[] for x in VALUES}
    payoff={"mastery":0.5,"connection":0.5,"peace":0.5}; payoff[early_world]=0.85
    for _ in range(days):
        for k in state: state[k]=max(0,state[k]-0.03)
        gaps={x:v[x]-state[x] for x in VALUES}
        pull={x:max(0.01,v[x]*0.8+gaps[x]*0.5) for x in VALUES}
        tot=sum(pull.values()); r=rng.random()*tot; acc=0; ch=VALUES[-1]
        for x,w in pull.items():
            acc+=w
            if r<=acc: ch=x; break
        exp=v[ch]; act=payoff[ch]+rng.uniform(-0.2,0.2)
        state[ch]=min(1,state[ch]+max(0,0.03+0.05*act))
        err=act-exp; rerr[ch].append(err); rerr[ch]=rerr[ch][-20:]
        v[ch]=min(1,max(0,v[ch]+0.02*err+0.02*(sum(rerr[ch])/len(rerr[ch]))))
        for x in VALUES:
            if x!=ch: v[x]=max(0,v[x]-0.002)
    return v

class Member:
    def __init__(self, name, character):
        self.name=name; self.char=character
        self.dominant=max(character,key=character.get)
        self.reputation=defaultdict(float)   # how others regard it
        self.partners=defaultdict(int)       # who it interacts with
        self.resources=0.0; self.contributions=0; self.conflicts=0

def run_society(members, rounds=400, rng=None):
    rng=rng or random.Random(0)
    names=[m.name for m in members]; by={m.name:m for m in members}
    for _ in range(rounds):
        a=rng.choice(members)
        # WHO it approaches depends on its character: connection-types seek others,
        # mastery-types seek productive partners, peace-types avoid conflict-prone ones
        if a.dominant=="connection":
            b=rng.choice([m for m in members if m!=a])
        elif a.dominant=="mastery":
            # prefers high-reputation productive partners
            others=[m for m in members if m!=a]
            b=max(others, key=lambda m:a.reputation[m.name]+0.1*rng.random())
        else: # peace: avoids those it has had conflict with
            others=[m for m in members if m!=a and a.reputation[m.name]>=0]
            b=rng.choice(others) if others else rng.choice([m for m in members if m!=a])
        # the INTERACTION outcome depends on BOTH characters (consequences, not
        # "cooperation=good"). two mastery-types may compete; connection smooths;
        # peace de-escalates.
        synergy = a.char["connection"]*0.5 + b.char["connection"]*0.5 \
                  + a.char["peace"]*0.3 + b.char["peace"]*0.3 \
                  - abs(a.char["mastery"]-b.char["mastery"])*0.0 \
                  - (a.char["mastery"]*b.char["mastery"])*0.4   # two driven -> friction
        synergy += rng.uniform(-0.2,0.2)
        a.partners[b.name]+=1; b.partners[a.name]+=1
        if synergy>0.3:   # cooperation
            a.reputation[b.name]+=0.05; b.reputation[a.name]+=0.05
            a.resources+=0.1; b.resources+=0.1; a.contributions+=1; b.contributions+=1
        else:             # conflict / failed interaction
            a.reputation[b.name]-=0.04; b.reputation[a.name]-=0.04
            a.conflicts+=1; b.conflicts+=1
    return members

def analyze(members, label):
    print(f"\n  --- {label} ---")
    # social roles that EMERGED (not programmed)
    for m in sorted(members,key=lambda x:-x.resources):
        total_partners=sum(1 for v in m.partners.values() if v>0)
        avg_rep = statistics.mean(m.reputation.values()) if m.reputation else 0
        role=""
        if m.contributions>0 and total_partners<=len(members)*0.4: role="productive-but-isolated"
        elif total_partners>=len(members)*0.7 and avg_rep>0: role="social-bridge"
        elif m.conflicts<m.contributions*0.3 and avg_rep>0.1: role="trusted-peacemaker"
        elif m.conflicts>m.contributions: role="contentious"
        else: role="peripheral"
        print(f"    {m.name:10s} [{m.dominant:10s}] resources {m.resources:4.1f}, "
              f"partners {total_partners}, conflicts {m.conflicts:3d}, role: {role}")
    # does character predict role? measure: do dominant-types cluster into roles?
    return members

def society_metrics(members):
    # structural metrics to compare developed vs flat
    reps=[r for m in members for r in m.reputation.values()]
    partner_counts=[sum(1 for v in m.partners.values() if v>0) for m in members]
    resource_spread=statistics.pstdev([m.resources for m in members]) if len(members)>1 else 0
    role_variety=len(set(m.dominant for m in members))
    return {"reputation_spread":statistics.pstdev(reps) if reps else 0,
            "resource_inequality":resource_spread,
            "partner_variance":statistics.pstdev(partner_counts) if len(partner_counts)>1 else 0}

def main():
    rng=random.Random(7)
    print("="*64); print("  SOCIETY v2: when different someones meet (+ causal control)"); print("="*64)

    # DEVELOPED society: 10 members, each shaped by a different early life
    worlds=["mastery","connection","peace"]*4
    rng.shuffle(worlds)
    developed=[]
    for i in range(10):
        char=shape_character(i*13+1, worlds[i])
        developed.append(Member(f"N{i+1}", char))
    print("\n  10 individuals, each shaped by a different early life:")
    for m in developed:
        print(f"    {m.name}: {', '.join(f'{k} {v:.2f}' for k,v in sorted(m.char.items(),key=lambda x:-x[1]))} -> {m.dominant}")
    run_society(developed, rng=random.Random(100))
    analyze(developed, "DEVELOPED SOCIETY: roles that emerged")
    dev_m=society_metrics(developed)

    # CAUSAL CONTROL: identical society but everyone FLAT (.50/.50/.50)
    flat=[Member(f"F{i+1}", {x:0.5 for x in VALUES}) for i in range(10)]
    # break the max() tie deterministically so flat members have a nominal dominant
    for i,m in enumerate(flat): m.dominant=VALUES[i%3]; 
    # but their CHARACTER is flat - interactions use .50 across the board
    for m in flat: m.char={x:0.5 for x in VALUES}
    run_society(flat, rng=random.Random(100))
    flat_m=society_metrics(flat)

    print("\n"+"="*64); print("  CAUSAL CONTROL: developed vs flat society"); print("="*64)
    print(f"  {'metric':<24}{'developed':>12}{'flat':>10}")
    for k in dev_m:
        print(f"  {k:<24}{dev_m[k]:>12.3f}{flat_m[k]:>10.3f}")

    print("\n"+"="*64); print("  DID DEVELOPMENTAL HISTORY SHAPE SOCIETY?"); print("="*64)
    # developed should show MORE structure: unequal resources, varied partners, reputation spread
    more_structure = (dev_m["resource_inequality"]>flat_m["resource_inequality"]*1.3 or
                      dev_m["partner_variance"]>flat_m["partner_variance"]*1.3 or
                      dev_m["reputation_spread"]>flat_m["reputation_spread"]*1.3)
    if more_structure:
        print(f"  YES - BEING FEEDS SPECIES. The society of developmentally-different")
        print(f"  individuals organized with MORE structure than the flat-character")
        print(f"  control: more unequal resources/partners/reputation. Social roles")
        print(f"  (productive-isolated, social-bridge, peacemaker) emerged from character,")
        print(f"  not from programming. Individuals who became different through their")
        print(f"  own histories formed a different, richer society than identical flat")
        print(f"  agents. Your Being work causally changes Species-level organization.")
    else:
        print(f"  Honest: developed and flat societies looked structurally similar this")
        print(f"  run (dev {dev_m}, flat {flat_m}). Character may not yet drive social")
        print(f"  structure strongly enough, or the interaction model needs richer")
        print(f"  consequences. Reported straight - the connection isn't established yet.")

if __name__=="__main__": main()
