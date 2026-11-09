"""Cold-Start Advantage Over Time (the friend's deep question): with a LARGE library
of genuinely-useful capabilities, how much economic advantage does a long-lived
NEURON's accumulated experience give, and how long does it persist before a newborn
catches up?

v2 showed: 4 reusable skills is too few - a newborn relearns them in days, so no
lasting advantage. A mature NEURON should have HUNDREDS of capabilities a newborn
can't relearn immediately. This tests that at scale.

Setup: large task space (300 types). Experienced enters having learned ~200 of them
in prior life; newborn starts empty. Both get a diverse stream of related-but-new
real-verifiable tasks. REUSE genuinely avoids compute (near-zero), fresh solving
costs real search (measured). Measure the GAP at day 1/10/30/60.

Honest framing: compute = measured search steps (maps to real tokens: reuse ~ 1
unit, fresh ~ hundreds). The live-qwen run already confirmed the per-task mechanism
is real; this answers the SCALE + PERSISTENCE question (how big, how long).

    python coldstart.py
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

# a LARGE library of distinct task types (the "capabilities" a life accumulates)
def task_library(seed, size=300):
    rng=random.Random(seed); lib=[]
    while len(lib)<size:
        if rng.random()<0.5: p=("s",rng.choice(list(LIST_OPS)),rng.choice(list(RED)))
        else: p=("c",rng.choice(list(LIST_OPS)),rng.choice(list(RED)),rng.choice(list(LIST_OPS)),rng.choice(list(RED)),rng.choice(list(BIN)))
        if p not in lib: lib.append(p)
    return lib

REUSE_COST=1.0   # reusing a learned capability: ~1 unit (near-free; maps to ~15 tokens)

class Being:
    def __init__(s): s.known={}; s.priors=defaultdict(lambda: defaultdict(float)); s.compute=0.0; s.verified=0
    def solve(s, tp, rng):
        sig=str(tp)
        if sig in s.known: s.compute+=REUSE_COST; s.verified+=1; return   # genuine reuse, avoids search
        exa=[((L:=[rng.randint(1,30) for _ in range(6)]),run(tp,L)) for _ in range(3)]
        c=0; order=sorted(ALL,key=lambda p:-sum(s.priors["g"][x] for x in pieces(p))); found=None
        for p in order:
            c+=1
            if all(run(p,L)==a for L,a in exa): found=p; break
        s.compute+=c*1.0   # fresh solve: real search cost (maps to hundreds of tokens)
        if found:
            s.known[sig]=found; s.verified+=1
            for x in pieces(found): s.priors["g"][x]+=1

def main():
    print("="*64); print("  COLD-START ADVANTAGE OVER TIME (friend's deep question)"); print("="*64)
    print("  large library (300 capabilities). experienced enters with ~200 learned;")
    print("  newborn starts empty. same diverse task stream. gap over time?\n")

    checkpoints=[1,10,30,60,120]
    exp_cum=defaultdict(list); nb_cum=defaultdict(list)
    for world in range(30):
        lib=task_library(world)
        exp=Being(); nb=Being()
        # experienced learned ~200 of the 300 in prior life (seed its known + priors)
        learn_rng=random.Random(world*5+1)
        for tp in learn_rng.sample(lib,200):
            exp.solve(tp, learn_rng)
        exp.compute=0.0   # reset the METER (keep knowledge); measure only the test life
        # both live the SAME daily task stream (drawn from the full library)
        exp_daily=[]; nb_daily=[]
        rng=random.Random(world*9+3)
        for day in range(1,121):
            tasks=[rng.choice(lib) for _ in range(5)]   # 5 tasks/day
            e0=exp.compute; n0=nb.compute
            for tp in tasks:
                exp.solve(tp, random.Random(day*100+hash(str(tp))%1000))
                nb.solve(tp, random.Random(day*100+hash(str(tp))%1000))
            exp_daily.append(exp.compute-e0); nb_daily.append(nb.compute-n0)
            if day in checkpoints:
                exp_cum[day].append(sum(exp_daily)); nb_cum[day].append(sum(nb_daily))

    print(f"  cumulative COMPUTE spent (lower=better), avg of 30 worlds:")
    print(f"  {'by day':<10}{'experienced':>14}{'newborn':>10}{'newborn/exp ratio':>20}")
    for d in checkpoints:
        e=statistics.mean(exp_cum[d]); n=statistics.mean(nb_cum[d])
        print(f"  {d:<10}{e:>14.0f}{n:>10.0f}{n/e:>19.1f}x")

    print("\n"+"="*64); print("  HOW BIG, AND HOW LONG, IS THE EXPERIENCE ADVANTAGE?"); print("="*64)
    d1=statistics.mean(nb_cum[1])/statistics.mean(exp_cum[1])
    d60=statistics.mean(nb_cum[60])/statistics.mean(exp_cum[60])
    d120=statistics.mean(nb_cum[120])/statistics.mean(exp_cum[120])
    if d1>2 and d60>1.3:
        print(f"  REAL, PERSISTENT COLD-START ADVANTAGE. With a large capability library,")
        print(f"  the experienced NEURON spent far less compute early (newborn used {d1:.1f}x more")
        print(f"  on day 1) because it reused ~200 learned capabilities while the newborn")
        print(f"  solved everything fresh. And the advantage PERSISTED: even by day 60 the")
        print(f"  newborn still spent {d60:.1f}x more compute, by day 120 {d120:.1f}x. The newborn slowly")
        print(f"  catches up (learning the library over its life) but a long-lived NEURON's")
        print(f"  accumulated experience gives a large, lasting economic edge - exactly what")
        print(f"  the vision needs: a mature NEURON is cheaper to run than a fresh one for a")
        print(f"  long time, because its life built capabilities a newborn must slowly relearn.")
    else:
        print(f"  Honest: day-1 ratio {d1:.1f}x, day-60 {d60:.1f}x, day-120 {d120:.1f}x. "
              f"{'advantage real but fades' if d1>1.5 else 'weak'}. Reported straight.")

if __name__=="__main__": main()
