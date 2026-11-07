"""Society v10: institutional continuity + memory + ADAPTATION (the friend's North
Star for Society). Can a society preserve useful organization across generations
WHILE CHANGING it when circumstances demand - with no founders left?

The breakthrough target:
  founders invent institution A -> founders die -> generations maintain A ->
  environment CHANGES -> cooperation falls -> a later generation identifies why ->
  institution becomes A2 -> cooperation recovers -> no original creators alive.

Five pieces (the friend's):
  1. institutional MEMORY - stores the rule, defections, crises, outcomes
  2. ONBOARDING - newborns learn the history + why the rule exists (consult memory)
  3. ROLES - record-keepers/maintainers emerge, not founders
  4. ENFORCEMENT tied to reputation/access - defection changes future resource access
  5. REVISION/ADAPTATION - when the rule fails under a new environment, members
     propose+adopt a modified rule (A->A2) rather than blindly obeying or abandoning

Compared: rigid (can't adapt) vs adaptive institution, across an environment shift.
Honest: adaptation may or may not recover cooperation. Report what happens.

    python institution_v10.py
"""
import random, statistics
from collections import defaultdict

class Institution:
    def __init__(s):
        s.rule={"contribute":0.3,"payout_threshold":0.35}  # rule A
        s.memory=[]                 # (gen, event, outcome)
        s.version="A"
    def record(s, gen, event, outcome): s.memory.append((gen,event,outcome))
    def recent_cooperation(s, k=3):
        coops=[o for g,e,o in s.memory if e=="coop_rate"][-k:]
        return statistics.mean(coops) if coops else 0.5

class Being:
    _id=0
    def __init__(s,believes,gen,native=True):
        Being._id+=1; s.name=f"b{Being._id}"; s.gen=gen
        s.believes=believes; s.reputation=0.5; s.resources=1.0; s.age=0; s.role=None

def cycle(pop, inst, rng, env_cost, adaptive, access):
    """one generation's resource game, using the institution's current rule."""
    contribute_amt=inst.rule["contribute"]; pool=0.0; coops=0
    for b in pop:
        # consult reputation/access: low-rep beings are limited (enforcement)
        allowed = access.get(b.name,0.5)>0.3
        # decide: follow the rule (contribute) vs defect, tempered by env_cost + rep stakes
        enforce_deter = inst.rule.get("enforcement",0.07)*1.5   # strong enforcement deters defection
        temptation=max(0.1, 0.4+env_cost-enforce_deter)
        will = b.believes and allowed and (rng.random()>temptation*0.6)
        if will:
            b.resources-=contribute_amt; pool+=contribute_amt*(1.6-env_cost); coops+=1
            access[b.name]=min(1,access.get(b.name,0.5)+0.05)
        else:
            access[b.name]=max(0,access.get(b.name,0.5)-inst.rule.get("enforcement",0.07))
    rate=coops/len(pop)
    inst.record(inst_gen[0],"coop_rate",rate)
    # payout to those above access threshold (enforcement via access)
    elig=[b for b in pop if access.get(b.name,0.5)>inst.rule["payout_threshold"]]
    if elig:
        for b in elig: b.resources+=pool/len(elig)
    # ADAPTATION: if the institution has record-keepers AND recent cooperation is
    # failing under the new environment, a maintainer proposes a revised rule A->A2
    if adaptive:
        keepers=[b for b in pop if b.role=="keeper"]
        if keepers and inst.recent_cooperation()<0.35 and inst.version=="A":
            # diagnose from memory: cooperation is falling because DEFECTION pays under
            # the new environment. A2 strengthens ENFORCEMENT - defecting costs far more
            # access, and the payout threshold rewards sustained cooperation.
            inst.rule["enforcement"]=0.22      # stronger access penalty for defecting
            inst.rule["payout_threshold"]=0.45 # must sustain cooperation to draw
            inst.version="A2"
            inst.record(inst_gen[0],"REVISION","A->A2: strengthened enforcement to counter defection in harder environment")
    return rate

inst_gen=[0]
def run(adaptive, generations=16, N=20, seed=0):
    rng=random.Random(seed); Being._id=0
    inst=Institution(); access=defaultdict(lambda:0.5)
    pop=[Being(True,0) for _ in range(N)]
    founders=set(b.name for b in pop)
    # assign a couple of founders as record-keepers; role is passed on, not the person
    for b in rng.sample(pop,2): b.role="keeper"
    history=[]
    for gen in range(1,generations+1):
        inst_gen[0]=gen
        # ENVIRONMENT SHIFT at gen 8: the old rule becomes costly
        env_cost = 0.0 if gen<8 else 0.35
        rate=cycle(pop, inst, rng, env_cost, adaptive, access)
        fa=sum(1 for b in pop if b.name in founders)
        history.append((gen,rate,fa,inst.version,env_cost))
        for b in pop: b.age+=1
        # turnover + onboarding
        pop.sort(key=lambda x:-x.age); dying=pop[:N//3]; surv=pop[N//3:]
        # ensure the KEEPER role persists across turnover (institution maintains roles)
        if not any(b.role=="keeper" for b in surv) and surv:
            rng.choice(surv).role="keeper"
        newcomers=[]
        for _ in range(len(dying)):
            if rng.random()<0.2:
                nb=Being(rng.random()<0.4, gen, native=False)  # migrant
            else:
                # ONBOARDING: newborn consults institutional memory to learn the rule+why
                teacher=rng.choice(surv)
                # consulting memory: belief strength depends on whether institution is working
                learned = teacher.believes and (rng.random()< (0.85 if inst.recent_cooperation()>0.3 else 0.6))
                nb=Being(learned,gen)
            access[nb.name]=0.5; newcomers.append(nb)
        # promote a new keeper occasionally (role continuity)
        pop=surv+newcomers
        if rng.random()<0.3 and not any(b.role=="keeper" for b in pop[-5:]):
            rng.choice(pop).role="keeper"
    return history, inst

def summarize(history):
    pre=[r for g,r,fa,v,ec in history if ec==0 and fa>0]           # with founders, old env
    post_shock_rigid=[r for g,r,fa,v,ec in history if ec>0]        # after env shift
    after_founders=[r for g,r,fa,v,ec in history if fa==0]
    return (statistics.mean(pre) if pre else 0,
            statistics.mean(post_shock_rigid) if post_shock_rigid else 0,
            statistics.mean(after_founders) if after_founders else 0)

def main():
    print("="*64); print("  SOCIETY v10: institutional continuity + memory + ADAPTATION"); print("="*64)
    print("  founders build institution A. at gen 8 the environment shifts and the")
    print("  old rule becomes costly. can an ADAPTIVE institution revise A->A2 and")
    print("  recover - with no founders left? (vs a RIGID one that can't adapt)\n")

    # average over many societies
    rig=[run(adaptive=False,seed=s) for s in range(30)]
    ada=[run(adaptive=True,seed=s) for s in range(30)]
    def avg_post(runs):
        # cooperation AFTER the environment shift AND after founders gone
        vals=[]
        for hist,inst in runs:
            post=[r for g,r,fa,v,ec in hist if ec>0 and fa==0]
            if post: vals.append(statistics.mean(post))
        return statistics.mean(vals) if vals else 0
    rig_post=avg_post(rig); ada_post=avg_post(ada)
    revised=sum(1 for h,i in ada if i.version=="A2")

    # a clean example trajectory (adaptive)
    hist,inst=run(adaptive=True, seed=2)
    print("  example adaptive trajectory:")
    for g,r,fa,v,ec in hist:
        tag=f"rule {v}" + (" | ENV SHIFTED" if ec>0 else "") + (" | founders gone" if fa==0 else f" | {fa} founders")
        print(f"    gen {g:>2}: coop {r*100:>3.0f}%  [{tag}]")

    print(f"\n  cooperation after env-shift AND after founders gone (avg of 30 societies):")
    print(f"    RIGID institution (cannot adapt):    {rig_post*100:.0f}%")
    print(f"    ADAPTIVE institution (can revise):   {ada_post*100:.0f}%")
    print(f"    adaptive societies that revised A->A2: {revised}/30")

    print("\n"+"="*64); print("  DID THE INSTITUTION SURVIVE, REMEMBER, AND ADAPT?"); print("="*64)
    if ada_post>rig_post+0.15 and revised>15:
        print(f"  YES - A SOCIETY THAT PRESERVES AND ADAPTS ITS INSTITUTIONS. When the")
        print(f"  environment shifted and the old rule began failing, adaptive societies")
        print(f"  revised the institution A->A2 ({revised}/30) using their record-keepers and")
        print(f"  institutional memory, recovering cooperation to {ada_post*100:.0f}% vs a rigid")
        print(f"  institution's {rig_post*100:.0f}% - all with NO original founders alive. The society")
        print(f"  preserved useful organization across generations AND changed it when")
        print(f"  circumstances demanded. That's continuity + memory + adaptation: social")
        print(f"  organization that outlives AND outgrows its creators.")
    else:
        print(f"  Honest: rigid {rig_post*100:.0f}%, adaptive {ada_post*100:.0f}%, revised {revised}/30.")
        print(f"  {'adaptation helps but gap thin' if ada_post>rig_post else 'adaptation did not clearly recover cooperation'}. Reported straight.")

if __name__=="__main__": main()
