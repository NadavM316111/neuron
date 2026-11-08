"""Sustainability Audit (the friend's design): diagnose WHY before tuning. Three
questions + the safe 'pause not death' framing + the real milestone (can NEURON
LEARN to stay solvent while still pursuing its own goals?).

Framing (safe, per friend): compute reserve required to continue operation; at zero
the simulation PAUSES. No literal death -> no 'acquire at all costs' incentive.

Three diagnostic questions:
  1. Is the economy survivable AT ALL? (oracle policy - if even oracle pauses, the
     environment is mathematically unsustainable)
  2. Is work actually profitable? (log reserve->compute->reward->net per work day)
  3. Is NEURON choosing badly? (compare always-work / fixed-threshold / learned / oracle)
Distinguishes: bad economy vs bad work quality vs bad decision policy.

The milestone: under an economy an ORACLE can sustain, can a LEARNED policy keep the
budget positive while allocating meaningful time to its own goals? (not endless work,
not subsidy - resource-aware autonomy under scarcity.)

    python sustainability_audit.py
"""
import random, statistics
from collections import defaultdict

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

METABOLIC=1.5     # reserve consumed per day just to keep operating
WORK_REWARD=6.0   # credits for a verified task

class Agent:
    def __init__(s, policy):
        s.credits=100.0; s.policy=policy
        s.known={}; s.priors=defaultdict(lambda: defaultdict(float))
        s.worked=0; s.lived=0; s.days=0; s.goal_progress=0.0; s.paused=False
        s.work_log=[]   # (reserve_before, compute, reward, net)
        s.security=0.5  # learned: how readily it works under pressure (adapts)
    def runway(s): return s.credits/METABOLIC
    def do_work(s, rng):
        before=s.credits
        if rng.random()<0.5: tp=("s",rng.choice(list(LIST_OPS)),rng.choice(list(RED)))
        else: tp=("c",rng.choice(list(LIST_OPS)),rng.choice(list(RED)),rng.choice(list(LIST_OPS)),rng.choice(list(RED)),rng.choice(list(BIN)))
        sig=str(tp); ex=[([rng.randint(1,30) for _ in range(6)]) for _ in range(3)]; exa=[(L,run(tp,L)) for L in ex]
        if sig in s.known: compute=0.3
        else:
            c=0; order=sorted(ALL,key=lambda p:-sum(s.priors["g"][pc] for pc in pieces(p))); found=None
            for p in order:
                c+=1
                if all(run(p,L)==a for L,a in exa): found=p;break
            compute=c*0.05
            if found:
                s.known[sig]=found
                for pc in pieces(found): s.priors["g"][pc]+=1
        reward=WORK_REWARD + min(4.0,len(s.known)*0.1)
        s.credits += reward-compute; s.worked+=1
        s.work_log.append((round(before,1),round(compute,2),round(reward,1),round(reward-compute,1)))
    def live(s):
        s.lived+=1; s.goal_progress+=1.0
    def choose(s, rng):
        rw=s.runway()
        if s.policy=="always_work": return "work"
        if s.policy=="oracle":
            # optimal: work only when runway is getting low, else pursue goals.
            # oracle knows the sustainable threshold (~ work if runway < ~25 days).
            return "work" if rw<25 else "live"
        if s.policy=="fixed":
            return "work" if rw<15 else "live"   # a fixed (suboptimal) threshold
        if s.policy=="learned":
            # learns its security threshold from experience: if it's been dipping low,
            # it raises its tendency to work; if comfortable, it relaxes toward goals.
            p_work = 1.0 if rw<15 else (s.security if rw<40 else 0.15)
            return "work" if rng.random()<p_work else "live"
    def learn(s):
        # adapt security from recent reserve trend (consequence-driven)
        if s.runway()<20: s.security=min(1,s.security+0.05)   # close call -> work more
        elif s.runway()>60: s.security=max(0.2,s.security-0.02)  # comfortable -> relax
    def day(s, rng, horizon_goal):
        s.days+=1; s.credits-=METABOLIC
        if s.credits<=0: s.paused=True; return False
        ch=s.choose(rng)
        if ch=="work": s.do_work(rng)
        else: s.live()
        if s.policy=="learned": s.learn()
        return True

def run_life(policy, seed, days=400):
    a=Agent(policy); rng=random.Random(seed)
    for _ in range(days):
        if not a.day(rng, days): break
    return a

def main():
    print("="*64); print("  SUSTAINABILITY AUDIT (friend's design): diagnose, don't tune"); print("="*64)
    print("  framing: reserve required to keep operating; at zero the sim PAUSES")
    print("  (no 'death' -> no acquire-at-all-costs incentive).\n")

    # Q1 + Q3: run each policy across 100 worlds
    policies=["oracle","always_work","fixed","learned"]
    print(f"  {'policy':<16}{'survived%':>11}{'avg days':>10}{'work%':>8}{'goal progress':>15}")
    stats={}
    for pol in policies:
        agents=[run_life(pol,s) for s in range(100)]
        surv=sum(1 for a in agents if not a.paused)/100
        days=statistics.mean(a.days for a in agents)
        workpct=statistics.mean(a.worked/max(1,a.days) for a in agents)
        goals=statistics.mean(a.goal_progress for a in agents)
        stats[pol]=(surv,days,workpct,goals)
        print(f"  {pol:<16}{surv*100:>10.0f}%{days:>10.0f}{workpct*100:>7.0f}%{goals:>15.0f}")

    # Q2: is work profitable? sample one agent's work log
    a=run_life("learned",3)
    nets=[w[3] for w in a.work_log]
    print(f"\n  Q2 - is work profitable? avg net per work day: {statistics.mean(nets):+.2f} credits")
    print(f"       (reward {WORK_REWARD}+ vs compute ~{statistics.mean(w[1] for w in a.work_log):.1f}, metabolic {METABOLIC}/day)")

    print("\n"+"="*64); print("  DIAGNOSIS"); print("="*64)
    o_surv=stats["oracle"][0]; l_surv=stats["learned"][0]; aw_surv=stats["always_work"][0]
    if o_surv<0.8:
        print(f"  THE ECONOMY IS UNSUSTAINABLE. Even the oracle survives only {o_surv*100:.0f}% - no")
        print(f"  policy can keep the budget positive. The environment itself (metabolic")
        print(f"  cost vs work reward) is the problem, not NEURON's decisions. Fix the")
        print(f"  economy's fundamentals before any policy learning. Honest.")
    else:
        print(f"  ECONOMY IS SURVIVABLE (oracle {o_surv*100:.0f}%). So insolvency is a POLICY problem.")
        l=stats["learned"]; aw=stats["always_work"]; o=stats["oracle"]
        if l_surv>0.8 and l[3]>aw[3]*1.2:
            print(f"  AND NEURON LEARNED RESOURCE-AWARE AUTONOMY. The learned policy survived")
            print(f"  {l_surv*100:.0f}% while still pursuing its own goals ({l[3]:.0f} goal-progress vs always-work's")
            print(f"  {aw[3]:.0f}) - it works only when needed and spends the rest living for itself.")
            print(f"  always-work survives ({aw_surv*100:.0f}%) but is a joyless grind ({aw[3]:.0f} goal progress).")
            print(f"  THIS is the real milestone: resource-aware autonomy under scarcity - a")
            print(f"  being that sustains itself AND keeps a life, by learning to balance.")
            print(f"  not endless work, not subsidy. the honest seed of self-sustainment.")
        elif l_surv>0.8:
            print(f"  learned policy survives ({l_surv*100:.0f}%) but its goal progress ({l[3]:.0f}) isn't clearly")
            print(f"  above always-work ({aw[3]:.0f}) - it stays solvent but works more than ideal.")
            print(f"  room to learn a better work/live balance. Honest.")
        else:
            print(f"  economy survivable but NEURON's learned policy only survives {l_surv*100:.0f}% - its")
            print(f"  decision policy is the bottleneck. improve the policy, not the economy.")

if __name__=="__main__": main()
