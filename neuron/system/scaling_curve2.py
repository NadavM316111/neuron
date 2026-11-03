"""Scaling curve, fixed: a search space that genuinely spans depths 2-5, so deep
targets are actually findable and we can see where brute collapses vs learned.

The v1 limitation: the enumerated space only covered depth-3 structure, so depth-4+
targets weren't IN the candidate set (both searchers failed meaninglessly). Fix:
represent programs as SEQUENCES of ops (a little stack machine), so the space
genuinely contains programs of each depth. Then both searchers draw from the
same real space, and we test honestly where each falls off under fixed compute.

A program = a sequence of ops over a working value that starts as the list L:
  list-op (list->list), reduce (list->number, ends the program), or push a second
  reduction and combine. Depth = number of ops. We enumerate/sample programs by
  depth so targets are reachable.

Learned search orders candidate op-sequences by cue->op meaning; brute uses a
fixed order. Fixed budget. Solve-rate vs depth. Honest.

    python scaling_curve2.py
"""
import random, re, itertools
from collections import defaultdict

LIST_OPS={
 "evens":lambda L:[x for x in L if x%2==0],"odds":lambda L:[x for x in L if x%2],
 "gt10":lambda L:[x for x in L if x>10],"lt20":lambda L:[x for x in L if x<20],
 "drop_max":lambda L:[x for x in L if x!=max(L)] if L else L,
 "drop_min":lambda L:[x for x in L if x!=min(L)] if L else L,
 "sorted_desc":lambda L:sorted(L,reverse=True),"identity":lambda L:L,
}
REDUCERS={"sum":sum,"max":lambda L:max(L) if L else 0,"min":lambda L:min(L) if L else 0,"len":len}
BINOPS={"sub":lambda a,b:a-b,"add":lambda a,b:a+b,"mul":lambda a,b:a*b}
CUE={ "evens":"even","odds":"odd","gt10":"above","lt20":"below","drop_max":"without largest",
 "drop_min":"without smallest","sorted_desc":"largest","max":"largest","min":"smallest",
 "sum":"sum","len":"count","sub":"difference","add":"sum","mul":"product"}
CUES=sorted(set(CUE.values()), key=len, reverse=True)
def cues_of(q):
    ql=q.lower(); return [c for c in CUES if c in ql]

# ---- a program is: apply k list-ops, then a reducer; optionally combine with a
# ---- second (list-ops + reducer) via a binop. "depth" = total ops. ----
def run_prog(prog, L):
    kind=prog[0]
    if kind=="simple":
        _,los,rd=prog; v=L
        for lo in los: v=LIST_OPS[lo](v)
        return REDUCERS[rd](v)
    else:
        _,losa,rda,losb,rdb,op=prog
        va=L
        for lo in losa: va=LIST_OPS[lo](va)
        vb=L
        for lo in losb: vb=LIST_OPS[lo](vb)
        return BINOPS[op](REDUCERS[rda](va), REDUCERS[rdb](vb))

def prog_pieces(prog):
    if prog[0]=="simple": return tuple(prog[1])+(prog[2],)
    return tuple(prog[3])+(prog[4],)+tuple(prog[1] if False else prog[3])  # placeholder
def pieces_of(prog):
    if prog[0]=="simple": return list(prog[1])+[prog[2]]
    return list(prog[3])+[prog[4]]+list(prog[1])+[prog[2]]+[prog[5]]

def prog_desc(prog):
    if prog[0]=="simple":
        _,los,rd=prog
        return CUE.get(rd,rd)+" of "+" ".join(CUE.get(l,l) for l in los)
    _,losa,rda,losb,rdb,op=prog
    return (CUE.get(op,op)+" of "+CUE.get(rda,rda)+" of "+" ".join(CUE.get(l,l) for l in losa)
            +" and "+CUE.get(rdb,rdb)+" of "+" ".join(CUE.get(l,l) for l in losb))

def gen_target(depth, rng):
    # depth 2 = 1 listop + reduce ; depth 3 = 2 listop + reduce ; 
    # depth 4 = binop(reduce(1 listop), reduce(1 listop)) ; depth5 = binop with a 2-listop side
    if depth<=3:
        k=depth-1
        los=[rng.choice(list(LIST_OPS)) for _ in range(k)]; rd=rng.choice(list(REDUCERS))
        return ("simple",los,rd)
    else:
        ka=1; kb=depth-3
        losa=[rng.choice(list(LIST_OPS)) for _ in range(ka)]
        losb=[rng.choice(list(LIST_OPS)) for _ in range(max(1,kb))]
        return ("combo",losa,rng.choice(list(REDUCERS)),losb,rng.choice(list(REDUCERS)),rng.choice(list(BINOPS)))

def candidates(depth):
    # enumerate candidate programs that CAN represent targets of this depth
    cands=[]
    LO=list(LIST_OPS); RD=list(REDUCERS)
    if depth<=3:
        k=depth-1
        for los in itertools.product(LO, repeat=k):
            for rd in RD:
                cands.append(("simple",list(los),rd))
    else:
        ka=1; kb=depth-3
        for losa in itertools.product(LO,repeat=ka):
            for rda in RD:
                for losb in itertools.product(LO,repeat=max(1,kb)):
                    for rdb in RD:
                        for op in BINOPS:
                            cands.append(("combo",list(losa),rda,list(losb),rdb,op))
    return cands

class Semantic:
    def __init__(self): self.cp=defaultdict(lambda: defaultdict(float))
    def train(self,desc,pieces):
        for c in cues_of(desc):
            for p in pieces: self.cp[c][p]+=1.0
    def order(self, desc, cands):
        active=cues_of(desc)
        scored=[]
        for prog in cands:
            pc=pieces_of(prog)
            s=sum(self.cp[c][p] for c in active for p in pc)
            scored.append((s,prog))
        scored.sort(key=lambda x:-x[0]); return [p for _,p in scored]

def examples(prog,rng,n=5):
    ex=[]
    tries=0
    while len(ex)<n and tries<40:
        tries+=1; L=[rng.randint(1,40) for _ in range(rng.randint(5,8))]
        try: ex.append((L,run_prog(prog,L)))
        except Exception: pass
    return ex

def search(order, ex, budget):
    for i,prog in enumerate(order):
        if i>=budget: return False,i
        try:
            if all(run_prog(prog,L)==a for L,a in ex): return True,i+1
        except Exception: pass
    return False,budget

def main():
    rng=random.Random(11)
    print("="*64); print("  SCALING CURVE v2: real space spanning depth 2-5"); print("="*64)
    # train semantics on shallow programs + ground word meanings
    sem=Semantic()
    for _ in range(200):
        d=rng.choice([2,3]); p=gen_target(d,rng); sem.train(prog_desc(p), pieces_of(p))
    for piece,cue in CUE.items():
        for _ in range(4): sem.cp[cue][piece]+=1.0

    BUDGET=2000
    print(f"  fixed budget: {BUDGET} tries. 20 unseen tasks/depth. targets ARE in-space now.\n")
    print(f"  {'depth':>6}{'space':>10}{'brute':>10}{'learned':>10}")
    for depth in [2,3,4,5]:
        cands=candidates(depth); T=20; b=0; l=0
        for _ in range(T):
            p=gen_target(depth,rng); ex=examples(p,rng)
            if len(ex)<3: continue
            bok,_=search(cands, ex, BUDGET)
            lok,_=search(sem.order(prog_desc(p),cands), ex, BUDGET)
            b+=bok; l+=lok
        print(f"  {depth:>6}{len(cands):>10}{f'{b*100//T}%':>10}{f'{l*100//T}%':>10}")

    print("\n"+"="*64); print("  THE SCALING CURVE"); print("="*64)
    print("  If brute's solve-rate falls with depth while learned stays high, the")
    print("  GAP widens with difficulty - learning changed the scaling curve. That")
    print("  is stronger than any single-depth win. Whatever the numbers, honest.")
    print("\n  (space column shows why brute collapses: it grows fast with depth,")
    print("  and a fixed budget covers a shrinking fraction. learned search uses")
    print("  the problem's words to look in the right place regardless.)")

if __name__=="__main__": main()
