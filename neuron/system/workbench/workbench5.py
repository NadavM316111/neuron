"""Workbench v5 (the friend's rigorous validation): 300 independent economies +
metabolic frontier sweep. Makes the self-sustainability claim solid and isolates
whether the FULL architecture beats memory-alone (synergy) or memory carries it.

Two experiments:
  1. MULTI-SEED: 300 random economies, each architecture. Report survival %, median
     cycles to insolvency, mean ending HyperFlow. Stable separation across worlds?
  2. METABOLIC FRONTIER SWEEP: raise the per-cycle burn and find the MAX sustainable
     compute cost for each architecture. If FULL sustains higher pressure than
     memory-only, that's real SYNERGY (not just memory carrying everything).

Honest: reports what actually holds. The claim stays "computational self-sustain-
ability under a bounded sandbox economy", not real-world.

    python workbench5.py
"""
import random, statistics, copy
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
DEEP=[("s",lo,rd) for lo in LIST_OPS for rd in RED]+[("c",a,b,c,d,o) for a in LIST_OPS for b in RED for c in LIST_OPS for d in RED for o in BIN]

def gen(kind, rng):
    if kind=="exact": p=("c",rng.choice(list(LIST_OPS)),rng.choice(list(RED)),rng.choice(list(LIST_OPS)),rng.choice(list(RED)),rng.choice(list(BIN)))
    elif kind=="related": p=("c",rng.choice(list(LIST_OPS)),rng.choice(list(RED)),rng.choice(list(LIST_OPS)),rng.choice(list(RED)),rng.choice(list(BIN)))
    else: p=("c",rng.choice(list(LIST_OPS)),rng.choice(list(RED)),rng.choice(list(LIST_OPS)),rng.choice(list(RED)),rng.choice(list(BIN)))
    ex=[]
    for _ in range(4):
        L=[rng.randint(1,30) for _ in range(6)]; ex.append((L,run(p,L)))
    reward=rng.randint(22,40); return str(p),ex,reward,p

class Agent:
    def __init__(self, mem, search, priors=None, known=None):
        self.hf=200.0; self.mem=mem; self.search=search
        self.known=copy.deepcopy(known) if (mem and known) else ({} if mem else None)
        self.priors=copy.deepcopy(priors) if (search and priors) else (defaultdict(lambda: defaultdict(float)) if search else None)
        self.alive=True
    def predict(self,sig):
        if self.known is not None and sig in self.known: return 0
        return 45 if (self.search and self.priors and self.priors["global"]) else 170
    def decide(self,sig,reward,metabolic):
        return (reward - self.predict(sig)*0.08 - metabolic) > 0
    def attempt(self,sig,ex,reward):
        if self.known is not None and sig in self.known:
            self.hf+=reward; return True
        if self.search and self.priors is not None and self.priors["global"]:
            order=sorted(DEEP,key=lambda p:-sum(self.priors["global"][pc] for pc in pieces(p)))
        else: order=list(DEEP)
        cost=0; found=None
        for p in order:
            cost+=1
            try:
                if all(run(p,L)==a for L,a in ex): found=p; break
            except Exception: pass
        self.hf-=cost*0.08
        if found:
            self.hf+=reward
            if self.known is not None: self.known[sig]=found
            if self.search and self.priors is not None:
                for pc in pieces(found): self.priors["global"][pc]+=1
        return found is not None

def train_knowledge(seed):
    base=Agent(True,True); rng=random.Random(seed)
    for _ in range(500):
        sig,ex,reward,p=gen(random.choice(["related","unfamiliar"]),rng)
        if base.decide(sig,reward,0): base.attempt(sig,ex,reward)
    return base.priors, base.known

def survive(mem,search,priors,known,seed,metabolic,horizon=3000):
    a=Agent(mem,search,priors,known); a.hf=200.0; rng=random.Random(seed)
    for t in range(horizon):
        a.hf-=metabolic
        if a.hf<=0: return False,t
        sig,ex,reward,p=gen(rng.choices(["exact","related","unfamiliar"],weights=[0.25,0.45,0.30])[0],rng)
        if a.decide(sig,reward,metabolic): a.attempt(sig,ex,reward)
        if a.hf<=0: return False,t
    return True,horizon

def main():
    print("="*64); print("  WORKBENCH v5: 300 economies + metabolic frontier sweep"); print("="*64)
    priors,known=train_knowledge(1)
    configs=[("full",True,True),("memory-only",True,False),("search-only",False,True),("newborn",False,False)]

    # EXPERIMENT 1: 300 independent economies at a fixed metabolic cost
    MET=18.0; SEEDS=300
    print(f"\n  EXP 1: {SEEDS} independent economies, metabolic cost {MET}/cycle:")
    print(f"  {'architecture':<16}{'survived%':>11}{'median cycles':>15}{'mean end HF':>13}")
    for label,mem,search in configs:
        outcomes=[survive(mem,search,priors,known,s*7+1,MET) for s in range(SEEDS)]
        surv=sum(1 for alive,_ in outcomes if alive)/SEEDS
        cycles=sorted(c for _,c in outcomes)
        med=cycles[len(cycles)//2]
        print(f"  {label:<16}{surv*100:>10.0f}%{med:>15}{'-':>13}")

    # EXPERIMENT 2: metabolic frontier sweep - max sustainable cost per architecture
    print(f"\n  EXP 2: metabolic frontier - MAX sustainable compute cost (survives >=80% of 40 worlds):")
    print(f"  {'architecture':<16}{'max sustainable burn':>22}")
    for label,mem,search in configs:
        max_sustain=0
        for met in [2,5,10,15,20,25,30,40,55,75,100]:
            survrate=sum(1 for s in range(40) if survive(mem,search,priors,known,s*13+1,met,horizon=1500)[0])/40
            if survrate>=0.8: max_sustain=met
            else: break
        print(f"  {label:<16}{max_sustain:>22}")

    print("\n"+"="*64); print("  WHAT DID RIGOROUS VALIDATION SHOW?"); print("="*64)
    print("  If survival separates cleanly across 300 worlds (full/memory high, newborn")
    print("  low) the self-sustainability result is ROBUST, not a lucky seed. If FULL's")
    print("  max sustainable burn exceeds memory-only's, that's real SYNERGY (search +")
    print("  memory together withstand more pressure than memory alone). If they're")
    print("  equal, memory is doing the economic work and the honest claim is narrower.")
    print("\n  (Honest scope: bounded SANDBOX economy, synthetic verified tasks, internal")
    print("  credits. This is the mechanism's seed, NOT real-world self-sustainment.)")

if __name__=="__main__": main()
