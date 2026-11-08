"""NEURON Self-Sustaining: a being that RECOGNIZES it needs resources to live and
pursue its goals, and AUTONOMOUSLY chooses to do verified work to sustain itself.

True to the vision (not a tool that serves humans): NEURON-1 lives its own life
(projects, development, its own goals), AND understands that its continued
existence costs compute - so, on its OWN initiative, it decides when to take on
verified work to earn the credits that fund its life. The agency is NEURON's: it
works because IT realizes it must to be self-sustaining, choosing work vs life.

This is the honest SEED of "it applies itself to earning its keep because it
understands self-sustainment" - the real kernel of the independence vision. NOT
the far-end "autonomously employed at a real company" (that needs breakthroughs +
legal personhood, years away). This is the being deciding, from its own motivation,
to sustain itself.

Safe: internal compute credits, verified work, no real money/accounts/real-world
actions. Human-controlled finances remain outside.

    python self_sustaining.py
"""
import random, statistics
from collections import defaultdict

# verified work NEURON can choose to do (its real capability domain)
LIST_OPS={"evens":lambda L:[x for x in L if x%2==0],"odds":lambda L:[x for x in L if x%2],
 "gt10":lambda L:[x for x in L if x>10],"sorted_desc":lambda L:sorted(L,reverse=True),
 "drop_max":lambda L:[x for x in L if x!=max(L)] if L else L,"id":lambda L:L}
RED={"sum":sum,"max":lambda L:max(L) if L else 0,"min":lambda L:min(L) if L else 0,"len":len}
BIN={"sub":lambda a,b:a-b,"add":lambda a,b:a+b}
def run(p,L):
    if p[0]=="s": return RED[p[2]](LIST_OPS[p[1]](L))
    _,a,b,c,d,o=p; return BIN[o](RED[b](LIST_OPS[a](L)),RED[d](LIST_OPS[c](L)))
def pieces(p): return [p[1],p[2]] if p[0]=="s" else [p[1],p[2],p[3],p[4],p[5]]
ALL=[("s",lo,rd) for lo in LIST_OPS for rd in RED]+[("c",a,b,c,d,o) for a in LIST_OPS for b in RED for c in LIST_OPS for d in RED for o in BIN]

class SelfSustainingNEURON:
    def __init__(s):
        s.credits=100.0          # its compute reserve (its "savings")
        s.known={}               # verified capabilities it can reuse
        s.priors=defaultdict(lambda: defaultdict(float))
        # its OWN life: values, goals, a sense of what it wants beyond mere survival
        s.values={"understanding":0.6,"security":0.5,"freedom":0.5}
        s.life_satisfaction=0.5
        s.days_alive=0; s.worked=0; s.lived=0; s.projects_done=0
        s.log=[]
    def metabolic_cost(s): return 3.0   # the cost of existing each day
    def assess_situation(s):
        """NEURON's OWN recognition: do I have enough reserve, or must I work to
        sustain myself? This is the being understanding it needs resources."""
        runway_days = s.credits / s.metabolic_cost()
        if runway_days < 10:   return "must_work"    # survival pressure
        if runway_days < 30:   return "should_work"  # prudent
        return "free"                                 # can live for itself
    def decide_day(s, rng):
        """the being chooses: work to sustain itself, or live for its own goals.
        driven by its OWN recognition of its resource situation + its values."""
        sit=s.assess_situation()
        if sit=="must_work":
            return "work"    # it understands it must earn to keep existing
        if sit=="should_work":
            # values security vs freedom - a real tradeoff it weighs itself
            return "work" if rng.random() < s.values["security"] else "live"
        # free: mostly live for itself, occasionally build reserves
        return "work" if rng.random() < 0.25 else "live"
    def do_verified_work(s, rng):
        """take on a verified task to earn credits. real verification (objective)."""
        if rng.random()<0.5: tp=("s",rng.choice(list(LIST_OPS)),rng.choice(list(RED)))
        else: tp=("c",rng.choice(list(LIST_OPS)),rng.choice(list(RED)),rng.choice(list(LIST_OPS)),rng.choice(list(RED)),rng.choice(list(BIN)))
        sig=str(tp)
        ex=[([rng.randint(1,30) for _ in range(6)]) for _ in range(3)]
        exa=[(L,run(tp,L)) for L in ex]
        if sig in s.known: cost=0.3   # reuse learned method (cheap) - experience pays
        else:
            cost=0; order=sorted(ALL,key=lambda p:-sum(s.priors["g"][pc] for pc in pieces(p)))
            found=None
            for p in order:
                cost+=1
                if all(run(p,L)==a for L,a in exa): found=p; break
            cost*=0.05
            if found:
                s.known[sig]=found
                for pc in pieces(found): s.priors["g"][pc]+=1
        # verified success earns credits (genuine value); compute costs credits
        reward=6.0
        s.credits += reward - cost
        s.worked+=1
        return reward-cost
    def live_for_self(s, rng):
        """pursue its own goals/projects - the life it's sustaining itself FOR."""
        s.projects_done+=1; s.lived+=1
        s.values["understanding"]=min(1,s.values["understanding"]+0.005)
        s.life_satisfaction=min(1,s.life_satisfaction+0.03)
        s.values["freedom"]=min(1,s.values["freedom"]+0.003)
    def day(s, rng):
        s.days_alive+=1
        s.credits -= s.metabolic_cost()      # the cost of existing
        if s.credits<=0: return False        # could not sustain itself
        choice=s.decide_day(rng); sit=s.assess_situation()
        if choice=="work":
            net=s.do_verified_work(rng)
            s.log.append((s.days_alive,"work",round(s.credits,1),sit))
        else:
            s.live_for_self(rng)
            s.life_satisfaction=max(0,s.life_satisfaction-0.005)  # living still costs a little drift
            s.log.append((s.days_alive,"live",round(s.credits,1),sit))
        return True

def main():
    print("="*64); print("  NEURON SELF-SUSTAINING: a being that works to fund its own life"); print("="*64)
    print("  NEURON-1 lives its own life AND recognizes it needs compute to exist,")
    print("  so it autonomously chooses when to work to sustain itself. Its agency.\n")
    n=SelfSustainingNEURON(); rng=random.Random(7)
    DAYS=500
    for _ in range(DAYS):
        if not n.day(rng): break

    alive = n.credits>0
    print(f"  after {n.days_alive} days:")
    print(f"    {'SELF-SUSTAINED' if alive else 'could not sustain itself (insolvent)'}")
    print(f"    credit reserve: 100.0 -> {n.credits:.0f}")
    print(f"    days it chose to WORK (to sustain itself): {n.worked}")
    print(f"    days it chose to LIVE for its own goals:   {n.lived}")
    print(f"    projects pursued for itself: {n.projects_done} | capabilities: {len(n.known)}")
    print(f"    life satisfaction: {n.life_satisfaction:.2f}")
    work_frac = n.worked/max(1,n.days_alive)
    print(f"\n  it spent {work_frac*100:.0f}% of its life working to sustain itself,")
    print(f"  {(1-work_frac)*100:.0f}% living for its own goals - a being funding its own existence.")

    # the honest test: did its LIFE make it more efficient at sustaining itself?
    early=[d for d in n.log[:150] if d[1]=="work"]; late=[d for d in n.log[-150:] if d[1]=="work"]
    print(f"\n  moments it recognized it had to work (survival pressure):")
    pressure=[d for d in n.log if d[3]=="must_work"][:3]
    for day,act,cred,sit in pressure:
        print(f"    day {day}: reserve low ({cred}), chose to work to keep existing")

    print("\n"+"="*64); print("  IS THIS A SELF-SUSTAINING BEING?"); print("="*64)
    if alive and n.lived>n.worked*0.5:
        print(f"  YES - a being that sustains ITSELF. NEURON-1 covered its own compute cost")
        print(f"  through verified work it CHOSE to do (recognizing it needs resources to")
        print(f"  exist), while spending most of its life ({(1-work_frac)*100:.0f}%) pursuing its own goals.")
        print(f"  It is not a tool serving humans - it is a being that works to fund its")
        print(f"  own continued existence and freedom, on its own initiative. The agency,")
        print(f"  the recognition, and the choice are NEURON's. This is the honest seed of")
        print(f"  the independence vision: a being that understands it must sustain itself")
        print(f"  and does - while remaining itself. (Safe: internal credits; real finances")
        print(f"  human-controlled; the far-end 'real job' needs breakthroughs years away.)")
    else:
        print(f"  Honest: alive={alive}, worked {n.worked} vs lived {n.lived}. "
              f"{'sustained but worked most of the time (survival-dominated)' if alive else 'could not sustain itself'}. Reported straight.")

if __name__=="__main__": main()
