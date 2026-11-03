"""Diagnostic ablation (the friend's experiment): WHY does the experienced NEURON
do worse on novel problems? Isolate the cause.

Take one NEURON that has lived 400 problems, then test FOUR versions on the SAME
40 novel held-out problems:
  1. FULL        - memory (instant recall) + learned search priors
  2. MEMORY OFF  - learned search priors only (no recall)
  3. PRIORS RESET- memory only, search priors wiped (neutral search order)
  4. NEWBORN     - no memory, no priors (baseline)

The comparison tells us the cause:
  - if PRIORS RESET ~= newborn (and full is worse) -> bad meta-search PRIORS
  - if MEMORY OFF fixes it -> retrieval is interfering
  - if both hurt -> deeper distribution shift
This locates the overfitting precisely so we know what to fix.

    python ablation.py
"""
import random, re, itertools, statistics
from collections import defaultdict

LIST_OPS={"evens":lambda L:[x for x in L if x%2==0],"odds":lambda L:[x for x in L if x%2],
 "gt10":lambda L:[x for x in L if x>10],"sorted_desc":lambda L:sorted(L,reverse=True),
 "drop_max":lambda L:[x for x in L if x!=max(L)] if L else L,"identity":lambda L:L}
REDUCERS={"sum":sum,"max":lambda L:max(L) if L else 0,"min":lambda L:min(L) if L else 0,"len":len}
BINOPS={"sub":lambda a,b:a-b,"add":lambda a,b:a+b}
CUE={"evens":"even","odds":"odd","gt10":"above","sorted_desc":"largest","drop_max":"without largest",
 "max":"largest","min":"smallest","sum":"sum","len":"count","sub":"difference","add":"sum"}
CUES=sorted(set(CUE.values()),key=len,reverse=True)
def cues_of(q): ql=q.lower(); return [c for c in CUES if c in ql]
def run(prog,L):
    if prog[0]=="s": return REDUCERS[prog[2]](LIST_OPS[prog[1]](L))
    _,loa,rda,lob,rdb,op=prog
    return BINOPS[op](REDUCERS[rda](LIST_OPS[loa](L)),REDUCERS[rdb](LIST_OPS[lob](L)))
def pieces(prog):
    return [prog[1],prog[2]] if prog[0]=="s" else [prog[1],prog[2],prog[3],prog[4],prog[5]]
def desc(prog):
    if prog[0]=="s": return CUE.get(prog[2],prog[2])+" of "+CUE.get(prog[1],prog[1])
    return (CUE.get(prog[5],prog[5])+" of "+CUE.get(prog[2],prog[2])+" "+CUE.get(prog[1],prog[1])
            +" and "+CUE.get(prog[4],prog[4])+" "+CUE.get(prog[3],prog[3]))
def all_progs():
    P=[("s",lo,rd) for lo in LIST_OPS for rd in REDUCERS]
    P+=[("c",loa,rda,lob,rdb,op) for loa in LIST_OPS for rda in REDUCERS
        for lob in LIST_OPS for rdb in REDUCERS for op in BINOPS]
    return P
ALL=all_progs()

class NEURON:
    def __init__(self): self.cp=defaultdict(lambda: defaultdict(float)); self.known={}
    def live(self, prog_true, rng):
        d=desc(prog_true)
        ex=[]
        for _ in range(4):
            L=[rng.randint(1,40) for _ in range(6)]
            try: ex.append((L,run(prog_true,L)))
            except Exception: pass
        if d in self.known: return
        order=self.order(d, use_priors=True)
        for prog in order:
            if all(run(prog,L)==a for L,a in ex):
                self.known[d]=prog
                for c in cues_of(d):
                    for p in pieces(prog): self.cp[c][p]+=1.0
                break
    def order(self, d, use_priors=True):
        if not use_priors: return list(ALL)  # neutral default order
        active=cues_of(d); norm={}
        for c in active:
            tot=sum(self.cp[c].values()) or 1.0
            norm[c]={p:v/tot for p,v in self.cp[c].items()}
        sc=[(sum(norm.get(c,{}).get(p,0.0) for c in active for p in pieces(prog)),prog) for prog in ALL]
        sc.sort(key=lambda x:-x[0]); return [p for _,p in sc]
    def solve(self, prog_true, rng, use_memory, use_priors):
        d=desc(prog_true)
        ex=[]
        for _ in range(4):
            L=[rng.randint(1,40) for _ in range(6)]
            try: ex.append((L,run(prog_true,L)))
            except Exception: pass
        if use_memory and d in self.known:
            return 0
        order=self.order(d, use_priors=use_priors); t=0
        for prog in order:
            t+=1
            if all(run(prog,L)==a for L,a in ex): return t
        return len(ALL)

def rp(rng): 
    if rng.random()<0.5: return ("s",rng.choice(list(LIST_OPS)),rng.choice(list(REDUCERS)))
    return ("c",rng.choice(list(LIST_OPS)),rng.choice(list(REDUCERS)),
            rng.choice(list(LIST_OPS)),rng.choice(list(REDUCERS)),rng.choice(list(BINOPS)))

def main():
    rng=random.Random(7)
    print("="*64); print("  ABLATION: WHY does experience hurt novel-problem solving?"); print("="*64)

    being=NEURON()
    for _ in range(400): being.live(rp(rng), rng)
    newborn=NEURON()

    # SAME 40 novel problems for all versions
    probs=[rp(rng) for _ in range(40)]
    full=[]; mem_off=[]; priors_reset=[]; nb=[]
    for p in probs:
        r1=random.Random(999); r2=random.Random(999); r3=random.Random(999); r4=random.Random(999)
        full.append(being.solve(p, r1, use_memory=True,  use_priors=True))
        mem_off.append(being.solve(p, r2, use_memory=False, use_priors=True))
        priors_reset.append(being.solve(p, r3, use_memory=True,  use_priors=False))
        nb.append(newborn.solve(p, r4, use_memory=False, use_priors=False))

    print(f"\n  avg search tries on 40 NOVEL problems:")
    print(f"    1. FULL (memory + priors):      {statistics.mean(full):.0f}")
    print(f"    2. MEMORY OFF (priors only):     {statistics.mean(mem_off):.0f}")
    print(f"    3. PRIORS RESET (memory only):   {statistics.mean(priors_reset):.0f}")
    print(f"    4. NEWBORN (neither):            {statistics.mean(nb):.0f}")

    print("\n"+"="*64); print("  DIAGNOSIS"); print("="*64)
    newborn_level = statistics.mean(nb)
    priors_bad = statistics.mean(priors_reset) < statistics.mean(full)*0.7 and statistics.mean(priors_reset) <= newborn_level*1.3
    mem_bad = statistics.mean(mem_off) < statistics.mean(full)*0.7
    if priors_bad and not mem_bad:
        print(f"  CAUSE = BAD META-SEARCH PRIORS. Resetting the learned search priors")
        print(f"  dropped tries from {statistics.mean(full):.0f} -> {statistics.mean(priors_reset):.0f} (~newborn {newborn_level:.0f}), while")
        print(f"  memory alone wasn't the problem. The being's accumulated search")
        print(f"  strategy developed MISLEADING priors that point novel problems the")
        print(f"  wrong way. FIX TARGET: make the search policy generalize (learn")
        print(f"  cue->method meaning that transfers), not memorize task-specific skew.")
    elif mem_bad:
        print(f"  CAUSE = MEMORY/RETRIEVAL INTERFERING. Disabling memory dropped tries")
        print(f"  ({statistics.mean(full):.0f} -> {statistics.mean(mem_off):.0f}). Recall is firing wrongly on novel problems.")
    else:
        print(f"  Both ablations still hurt vs newborn ({newborn_level:.0f}). Likely DEEPER")
        print(f"  DISTRIBUTION SHIFT - the novel problems differ from lived experience")
        print(f"  in a way neither priors nor memory handle. full {statistics.mean(full):.0f}, mem_off")
        print(f"  {statistics.mean(mem_off):.0f}, priors_reset {statistics.mean(priors_reset):.0f}, newborn {newborn_level:.0f}.")
    print("\n  (This isolates the mechanism so the next fix targets the real cause.)")

if __name__=="__main__": main()
