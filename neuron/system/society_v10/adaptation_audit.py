"""Institution adaptation audit (the friend's oracle test): before building more
adaptation machinery, find out WHICH layer is broken - institutional reasoning
(NEURONs pick the wrong revision) or institutional mechanics (even the right rule
doesn't recover cooperation).

Four conditions, same world, env shifts at gen 8:
  1. rigid A            - never changes
  2. learned adaptation - current system decides when/how to revise A->A2
  3. ORACLE A2          - at gen 8 society is HANDED the correct post-shift rule
  4. oracle A2 + enforce- oracle rule AND perfect compliance

Reading:
  oracle A2 >> rigid/learned -> adaptation CAN work, NEURONs invent the WRONG
    revision (fix institutional REASONING)
  oracle A2 ~ rigid -> even the right rule doesn't help; the post-shift world makes
    cooperation unattractive regardless (fix institutional MECHANICS/incentives)
  oracle+enforce >> oracle alone -> the rule is fine but adoption/compliance is the
    problem (fix adoption, not the rule)

Also fixes the v10 display bug and measures the REAL metric: delta-cooperation
caused by the revision.

    python adaptation_audit.py
"""
import random, statistics
from collections import defaultdict

def run(mode, generations=16, N=20, seed=0, shift_gen=8):
    rng=random.Random(seed)
    # institution parameters (a rule = contribution + enforcement strength + threshold)
    rule={"contribute":0.3,"enforcement":0.07,"threshold":0.35}
    version="A"
    access=defaultdict(lambda:0.5)
    pop=[{"name":f"b{i}","believes":True,"age":0} for i in range(N)]
    founders=set(b["name"] for b in pop); nid=N
    history=[]
    coop_pre_revision=[]; coop_post_revision=[]; revised_at=None

    for gen in range(1,generations+1):
        env_cost = 0.0 if gen<shift_gen else 0.35
        # ORACLE: hand society the correct post-shift rule exactly at the shift
        if mode in ("oracle","oracle_enforce") and gen==shift_gen and version=="A":
            rule={"contribute":0.2,"enforcement":0.25,"threshold":0.3}; version="A2"; revised_at=gen
        # LEARNED: society decides to revise if recent cooperation is low (post-shift)
        if mode=="learned" and version=="A" and gen>shift_gen:
            recent=[r for g,r in history[-3:]]
            if recent and statistics.mean(recent)<0.4:
                # NEURON's (currently naive) revision: it lowers contribution
                rule={"contribute":0.15,"enforcement":0.1,"threshold":0.25}; version="A2"; revised_at=gen

        # the resource game
        pool=0.0; coops=0
        for b in pop:
            allowed=access[b["name"]]>0.3
            enforce_deter=rule["enforcement"]*1.5
            temptation=max(0.1,0.4+env_cost-enforce_deter)
            comply = (mode=="oracle_enforce" and version=="A2")   # perfect compliance in that mode
            will = b["believes"] and allowed and (comply or rng.random()>temptation*0.6)
            if will:
                pool+=rule["contribute"]*(1.6-env_cost); coops+=1
                access[b["name"]]=min(1,access[b["name"]]+0.05)
            else:
                access[b["name"]]=max(0,access[b["name"]]-rule["enforcement"])
        rate=coops/len(pop)
        history.append((gen,rate))
        if revised_at:
            if gen<revised_at: pass
        for b in pop: b["age"]+=1
        # turnover + onboarding
        pop.sort(key=lambda x:-x["age"]); dying=pop[:N//3]; surv=pop[N//3:]
        for _ in range(len(dying)):
            if rng.random()<0.2: nb={"name":f"b{nid}","believes":rng.random()<0.4,"age":0}
            else:
                teacher=rng.choice(surv); nb={"name":f"b{nid}","believes":teacher["believes"] and rng.random()<0.8,"age":0}
            access[nb["name"]]=0.5; nid+=1; pop=surv+[nb]; surv=pop
        pop=surv

    # metrics: cooperation after shift AND after founders gone (founders gone ~gen5)
    post=[r for g,r in history if g>shift_gen]
    post_mean=statistics.mean(post) if post else 0
    # delta from revision (if any)
    d_before=d_after=None
    if revised_at:
        before=[r for g,r in history if shift_gen<=g<revised_at]
        after=[r for g,r in history if g>=revised_at]
        d_before=statistics.mean(before) if before else None
        d_after=statistics.mean(after) if after else None
    return post_mean, version, revised_at, d_before, d_after, history

def main():
    print("="*64); print("  INSTITUTION ADAPTATION AUDIT (oracle test, friend's design)"); print("="*64)
    print("  env shifts at gen 8. which layer is broken - reasoning or mechanics?\n")
    modes=["rigid","learned","oracle","oracle_enforce"]
    labels={"rigid":"1. rigid A (never changes)","learned":"2. learned adaptation (A->A2)",
            "oracle":"3. ORACLE A2 (handed correct rule)","oracle_enforce":"4. oracle A2 + enforcement"}
    res={}
    for m in modes:
        vals=[run(m,seed=s)[0] for s in range(30)]
        res[m]=statistics.mean(vals)

    print(f"  cooperation after env-shift (avg of 30 societies):")
    print(f"  {'condition':<34}{'cooperation':>12}")
    for m in modes: print(f"  {labels[m]:<34}{res[m]*100:>11.0f}%")

    # verify display + show the real delta for one learned run
    pm,ver,rev,db,da,hist=run("learned",seed=3)
    print(f"\n  [bug check] one learned society: final rule version = {ver}, revised at gen {rev}")
    if rev and db is not None and da is not None:
        print(f"  real metric - cooperation before revision {db*100:.0f}% -> after {da*100:.0f}% (delta {(da-db)*100:+.0f} pts)")

    print("\n"+"="*64); print("  WHICH LAYER IS BROKEN?"); print("="*64)
    rigid=res["rigid"]; learned=res["learned"]; oracle=res["oracle"]; oenf=res["oracle_enforce"]
    if oracle>rigid+0.2:
        print(f"  INSTITUTIONAL REASONING is the problem. The ORACLE correct rule recovers")
        print(f"  cooperation to {oracle*100:.0f}% vs rigid {rigid*100:.0f}% and learned {learned*100:.0f}%. So a good A2 EXISTS and")
        print(f"  works - the NEURONs are just INVENTING THE WRONG REVISION. Fix: give the")
        print(f"  society better reasoning about what change solves the new problem (use")
        print(f"  the institutional memory to diagnose the actual cause). The mechanics")
        print(f"  are fine; the adaptive intelligence is what's missing.")
        if oenf>oracle+0.15:
            print(f"  (And oracle+enforcement {oenf*100:.0f}% > oracle {oracle*100:.0f}%: compliance/adoption also matters.)")
    elif oenf>rigid+0.2 and oracle<rigid+0.1:
        print(f"  INSTITUTIONAL ADOPTION is the problem. The right rule alone ({oracle*100:.0f}%) doesn't")
        print(f"  help, but the right rule WITH enforced compliance ({oenf*100:.0f}%) does. So the")
        print(f"  issue isn't the rule or reasoning - it's that members don't actually")
        print(f"  follow A2. Fix: adoption/compliance, not the rule.")
    else:
        print(f"  INSTITUTIONAL MECHANICS / INCENTIVES are the problem. Even the oracle")
        print(f"  correct rule ({oracle*100:.0f}%) and oracle+enforcement ({oenf*100:.0f}%) barely beat rigid ({rigid*100:.0f}%).")
        print(f"  The post-shift world makes cooperation unattractive REGARDLESS of the")
        print(f"  institution. No revision can save it - the incentive structure itself")
        print(f"  needs fixing before institutional adaptation can matter. Honest - the")
        print(f"  deeper problem, found before building more machinery.")

if __name__=="__main__": main()
