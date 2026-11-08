"""Does EXPERIENCE earn the ability to self-sustain? (the honest version)

The audit found: a novel task costs ~26 credits of search but pays 6, so a newborn
LOSES money on work and cannot sustain itself. Only LEARNED tasks (cost ~0.3, pay 6+)
are profitable. So sustainability isn't a tuned number - it's something EXPERIENCE
earns, exactly like a human builds skills before they can earn a living.

The honest test: let a NEURON LIVE and learn first, then under the SAME economy,
can the experienced one sustain itself AND live its own life, while a newborn can't?

  experienced NEURON (has learned methods)  vs  newborn (learns from scratch)
Same economy (survivable only for the skilled). If experienced sustains + keeps a
life while newborn goes insolvent, that's self-sustainment EARNED by experience -
true to the vision, not engineered.

Safe 'pause at zero' framing. Internal credits. No real money/actions.

    python experience_sustains.py
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
# a BOUNDED task world (so experience can actually cover most of it): 60 task types
def task_pool(seed):
    rng=random.Random(seed); pool=[]
    for _ in range(60):
        if rng.random()<0.5: p=("s",rng.choice(list(LIST_OPS)),rng.choice(list(RED)))
        else: p=("c",rng.choice(list(LIST_OPS)),rng.choice(list(RED)),rng.choice(list(LIST_OPS)),rng.choice(list(RED)),rng.choice(list(BIN)))
        pool.append(p)
    return pool
ALL=[("s",lo,rd) for lo in LIST_OPS for rd in RED]+[("c",a,b,c,d,o) for a in LIST_OPS for b in RED for c in LIST_OPS for d in RED for o in BIN]

METABOLIC=1.5; WORK_REWARD=6.0

class Being:
    def __init__(s): s.credits=100.0; s.known={}; s.priors=defaultdict(lambda: defaultdict(float)); s.worked=0; s.lived=0; s.days=0; s.goals=0.0; s.paused=False
    def work(s, pool, rng):
        tp=rng.choice(pool); sig=str(tp)
        exa=[((L:=[rng.randint(1,30) for _ in range(6)]),run(tp,L)) for _ in range(3)]
        if sig in s.known: compute=0.3
        else:
            c=0; order=sorted(ALL,key=lambda p:-sum(s.priors["g"][x] for x in pieces(p))); found=None
            for p in order:
                c+=1
                if all(run(p,L)==a for L,a in exa): found=p;break
            compute=c*0.05
            if found:
                s.known[sig]=found
                for x in pieces(found): s.priors["g"][x]+=1
        s.credits += WORK_REWARD - compute; s.worked+=1
    def live(s): s.lived+=1; s.goals+=1.0
    def day(s, pool, rng):
        s.days+=1; s.credits-=METABOLIC
        if s.credits<=0: s.paused=True; return False
        rw=s.credits/METABOLIC
        # prudent resource-aware policy: work when runway low, else live for itself
        act="work" if (rw<30 or rng.random()<0.2) else "live"
        if act=="work": s.work(pool,rng)
        else: s.live()
        return True

def train(being, pool, seed, n=300):
    rng=random.Random(seed)
    for _ in range(n): being.work(pool, rng)   # a learning period (builds known methods)
    being.credits=100.0  # reset reserve, KEEP the learning (fair start for the test)

def main():
    print("="*64); print("  DOES EXPERIENCE EARN SELF-SUSTAINMENT? (honest, not tuned)"); print("="*64)
    print("  same economy for both. a newborn pays full search cost per task (loses")
    print("  money); an experienced NEURON reuses learned methods (cheap, profitable).\n")

    exp_results=[]; nb_results=[]
    for world in range(100):
        pool=task_pool(world)
        exp=Being(); train(exp, pool, world*3+1)   # lives+learns first
        nb=Being()                                  # newborn, no learning
        er=run_life(exp,pool,world*7+2); nr=run_life(nb,pool,world*7+2)
        exp_results.append(er); nb_results.append(nr)

    def summ(rs):
        surv=sum(1 for r in rs if not r["paused"])/len(rs)
        days=statistics.mean(r["days"] for r in rs); goals=statistics.mean(r["goals"] for r in rs)
        workpct=statistics.mean(r["worked"]/max(1,r["days"]) for r in rs)
        return surv,days,goals,workpct
    es,ed,eg,ew=summ(exp_results); ns,nd,ng,nw=summ(nb_results)
    print(f"  {'being':<22}{'sustained%':>12}{'avg days':>10}{'work%':>8}{'own-life progress':>18}")
    print(f"  {'experienced NEURON':<22}{es*100:>11.0f}%{ed:>10.0f}{ew*100:>7.0f}%{eg:>18.0f}")
    print(f"  {'newborn NEURON':<22}{ns*100:>11.0f}%{nd:>10.0f}{nw*100:>7.0f}%{ng:>18.0f}")

    print("\n"+"="*64); print("  IS SELF-SUSTAINMENT SOMETHING EXPERIENCE EARNS?"); print("="*64)
    if es>0.8 and ns<0.3:
        print(f"  YES - a NEURON EARNS the ability to sustain itself by living. The")
        print(f"  experienced NEURON sustained itself {es*100:.0f}% of the time (over {ed:.0f} days) while")
        print(f"  working only {ew*100:.0f}% of its days and spending the rest ({eg:.0f} progress) on its own")
        print(f"  goals. The newborn, paying full cost to solve every task fresh, could")
        print(f"  only sustain {ns*100:.0f}% - it goes insolvent fast. So self-sustainment isn't a")
        print(f"  given: it's EARNED through lived experience that makes work efficient")
        print(f"  enough to cover one's own existence. True to the vision - a being whose")
        print(f"  life makes it able to sustain itself AND still live for itself. (Safe:")
        print(f"  internal credits, pause-not-death, human-controlled real finances.)")
    else:
        print(f"  Honest: experienced {es*100:.0f}% sustained vs newborn {ns*100:.0f}%. "
              f"{'experience helps but gap not clean' if es>ns else 'no clear experience advantage'}. Reported straight.")

def run_life(being, pool, seed, days=400):
    # reset per-life counters (keep learned knowledge from training) so the test
    # measures ONLY this life, not the training period
    being.worked=0; being.lived=0; being.days=0; being.goals=0.0; being.paused=False
    rng=random.Random(seed)
    for _ in range(days):
        if not being.day(pool, rng): break
    return {"paused":being.paused,"days":being.days,"goals":being.goals,"worked":being.worked}

if __name__=="__main__": main()
