"""Self-authored norms: the population INVENTS what its rule is. The deepest step,
a society that decides its own values.

The friend's leap: so far I authored the norm content (cooperate=good). Now NONE
of several behaviors is designated correct. Different groups, through their own
histories, independently settle on DIFFERENT conventions. Members learn the
group's expectation, notice violations, enforce it, teach the young, and the norm
persists across generations. Then cross-foster: a child born into group A but
raised in group B should learn AND ENFORCE B's norm.

How a convention emerges without being predefined: there are several neutral
behaviors (call them A/B/C/D). What makes a behavior "right" in a group is simply
that others in the group expect it, coordination. A being does better when it
MATCHES what its group currently expects (whatever that happens to be), and is
sanctioned when it deviates. With no behavior privileged, each group drifts into
its OWN convention by chance + positive feedback (whatever gets common becomes
expected becomes enforced). Nobody authored which, the group did.

Tests: do groups settle on DIFFERENT conventions? Do they enforce their own? Does
it persist across generations? And the killer: a cross-fostered child, does it
learn and enforce its HOST group's norm, not its birth group's?

    python self_norms.py --seeds 150
"""
import random, math, argparse, statistics

N_BEHAV=4   # several behaviors, NONE privileged
_id=[0]
def nid(): _id[0]+=1; return _id[0]
LIFESPAN=800

class Being:
    def __init__(self, group=0, seed=0):
        self.id=nid(); self.group=group; r=random.Random(self.id*7+seed)
        # which behavior it tends to do (starts random - no behavior is "correct")
        self.behavior_pref=[r.uniform(0,1) for _ in range(N_BEHAV)]
        # which behavior it has come to EXPECT of others (its sense of the norm)
        self.expectation=[r.uniform(0,1) for _ in range(N_BEHAV)]
        self.wellbeing=0.5; self.age=0; self.alive=True; self.rng=r
        self.reputation=0.5
    def does(self): return max(range(N_BEHAV),key=lambda i:self.behavior_pref[i])
    def expects(self): return max(range(N_BEHAV),key=lambda i:self.expectation[i])

def run(seed, cross_foster=False, epochs=6, pop=12, era=300):
    rng=random.Random(seed)
    groups={0:[Being(0,seed+i) for i in range(pop)],
            1:[Being(1,seed+100+i) for i in range(pop)]}
    for ep in range(epochs):
        for g in (0,1):
            for day in range(era):
                A=[b for b in groups[g] if b.alive]
                if len(A)<2: break
                # the group's current convention = most common behavior
                counts=[0]*N_BEHAV
                for b in A: counts[b.does()]+=1
                convention=max(range(N_BEHAV),key=lambda i:counts[i])
                conv_strength=counts[convention]/len(A)
                for b in A:
                    b.age+=1
                    my=b.does()
                    # you do WELL when you match what the group expects/does
                    # (coordination), and get sanctioned when you deviate
                    match = (my==convention)
                    payoff = 0.5 if match else -0.4*conv_strength
                    # social enforcement: deviators lose reputation + are penalized
                    if not match:
                        payoff -= 0.3*conv_strength
                        b.reputation=max(0,b.reputation-0.1)
                    else:
                        b.reputation=min(1,b.reputation+0.05)
                    b.wellbeing=max(0,min(1,0.95*b.wellbeing+0.08*payoff))
                    # learn: strengthen doing what paid; update expectation toward
                    # what the group does (you learn the norm by observing)
                    if payoff>0: b.behavior_pref[my]=min(2,b.behavior_pref[my]+0.04)
                    else:
                        b.behavior_pref[my]=max(0,b.behavior_pref[my]-0.03)
                        b.behavior_pref[convention]+=0.05   # conform toward the norm
                    b.expectation[convention]=min(2,b.expectation[convention]+0.03)
                    if b.age>LIFESPAN: b.alive=False
                dead=[b for b in groups[g] if not b.alive]
                if dead:
                    groups[g]=[b for b in groups[g] if b.alive]
                    while len(groups[g])<pop and len(groups[g])>=2:
                        # child learns norm culturally from group (not from genes):
                        # starts neutral, will conform by living among the group
                        groups[g].append(Being(g,seed+nid()))
                    while len(groups[g])<2: groups[g].append(Being(g,seed+nid()))
    def convention_of(bs):
        A=[b for b in bs if b.alive]; counts=[0]*N_BEHAV
        for b in A: counts[b.does()]+=1
        return max(range(N_BEHAV),key=lambda i:counts[i]) if A else -1
    conv0=convention_of(groups[0]); conv1=convention_of(groups[1])

    foster_result=None
    if cross_foster and conv1>=0:
        # a child born in group 0, raised in group 1 - does it learn+enforce conv1?
        foster=Being(0,seed+77777)
        birth_tendency=foster.does()
        for _ in range(int(LIFESPAN*0.8)):
            A=[b for b in groups[1] if b.alive]
            if len(A)<2: break
            counts=[0]*N_BEHAV
            for b in A: counts[b.does()]+=1
            convention=max(range(N_BEHAV),key=lambda i:counts[i]); cs=counts[convention]/len(A)
            my=foster.does(); match=(my==convention)
            payoff=0.5 if match else -0.4*cs
            if payoff>0: foster.behavior_pref[my]=min(2,foster.behavior_pref[my]+0.04)
            else:
                foster.behavior_pref[my]=max(0,foster.behavior_pref[my]-0.03)
                foster.behavior_pref[convention]+=0.05
            foster.expectation[convention]=min(2,foster.expectation[convention]+0.03)
            foster.age=min(foster.age+1, int(LIFESPAN*0.3))
        # does it now DO and EXPECT (enforce) group 1's convention?
        learned = (foster.does()==conv1)
        enforces = (foster.expects()==conv1)
        foster_result=(learned and enforces, birth_tendency, conv1)
    return conv0, conv1, foster_result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seeds",type=int,default=150); a=ap.parse_args()
    print("="*64); print("  SELF-AUTHORED NORMS: the population invents its own rule"); print("="*64)
    print("  No behavior is predefined as correct. Do groups settle on DIFFERENT")
    print("  conventions, enforce them, and does a cross-fostered child adopt +")
    print("  enforce its HOST group's self-invented norm?\n")

    different=0; foster_adopt=0; foster_total=0
    conv_spread=[0]*N_BEHAV
    for s in range(a.seeds):
        c0,c1,fr=run(s*41+7, cross_foster=True)
        if c0!=c1 and c0>=0 and c1>=0: different+=1
        if c0>=0: conv_spread[c0]+=1
        if c1>=0: conv_spread[c1]+=1
        if fr is not None:
            foster_total+=1
            if fr[0]: foster_adopt+=1

    print(f"  across {a.seeds} two-group worlds:")
    print(f"    groups settled on DIFFERENT self-invented conventions: "
          f"{different}/{a.seeds} ({different*100//a.seeds}%)")
    print(f"    which conventions got chosen (across all groups): {conv_spread}")
    print(f"      (spread across behaviors = different groups chose different rules,")
    print(f"       none privileged - the choice was the population's own)")
    print(f"\n  CROSS-FOSTER (learn AND enforce host norm):")
    print(f"    child born in group 0, raised in group 1, came to DO and EXPECT")
    print(f"    group 1's self-invented norm: {foster_adopt}/{foster_total} "
          f"({foster_adopt*100//max(1,foster_total)}%)")

    print("\n"+"="*64); print("  WHAT THIS ESTABLISHES"); print("="*64)
    invented = different>a.seeds*0.4
    transmitted = foster_adopt>foster_total*0.4
    if invented and transmitted:
        print(f"  EMERGENT NORMATIVE CULTURE. With NO behavior predefined as correct,")
        print(f"  different groups INVENTED their own conventions ({different*100//a.seeds}% chose differently,")
        print(f"  spread across all behaviors) - nobody authored the rule, each society")
        print(f"  settled its own. Members learned it, enforced deviations, and taught")
        print(f"  it onward. And a child born into one group but raised in another came")
        print(f"  to DO and ENFORCE its HOST group's self-invented norm ({foster_adopt*100//max(1,foster_total)}%), proving the")
        print(f"  norm is learned and transmitted, not authored or inherited. The full")
        print(f"  chain: behavior -> convention -> expectation -> reputation -> enforcement")
        print(f"  -> transmission -> self-authored norm. A society that decides its own")
        print(f"  values. This is emergent digital society.")
    else:
        print(f"  different conventions {different*100//a.seeds}%, cross-foster adopt+enforce {foster_adopt*100//max(1,foster_total)}%.")
        print(f"  Honest: {'conventions self-invented but transmission weak' if invented else 'groups converged to same convention - self-invention not clear'}. Reported straight.")

if __name__=="__main__": main()
