"""The North Star test: does learned meta-search solve DEEP programs that brute
force CANNOT reach under the same compute budget?

This is the definitive compute-wall attack. At depth 4+, the program space is so
large that brute force, under a fixed budget, simply can't reach the answer. The
question: does semantic meta-search (NEURON understanding problem words) find
deep programs within the SAME budget that brute force fails?

If yes: intelligent search solves what brute force can't under equal compute =
NEURON converts understanding into reach that raw compute can't buy. That is the
compute wall, attacked.

Protocol (strict, as before): learn cue->primitive meaning on training families,
FREEZE, test on UNSEEN deep families. Fixed budget for BOTH. Measure: does each
SOLVE within budget, and in how many tries.

    python depth4.py
"""
import random, re, itertools, statistics
from collections import defaultdict

def nums(q): return list(map(int, re.findall(r"-?\d+", q)))

LIST_OPS={
 "sorted_desc":lambda L:sorted(L,reverse=True),"sorted_asc":lambda L:sorted(L),
 "evens":lambda L:[x for x in L if x%2==0],"odds":lambda L:[x for x in L if x%2],
 "drop_max":lambda L:[x for x in L if x!=max(L)] if L else L,
 "drop_min":lambda L:[x for x in L if x!=min(L)] if L else L,
 "gt10":lambda L:[x for x in L if x>10],"lt20":lambda L:[x for x in L if x<20],
 "identity":lambda L:L,
}
REDUCERS={"sum":sum,"max":lambda L:max(L) if L else 0,"min":lambda L:min(L) if L else 0,
 "len":len,"first":lambda L:L[0] if L else 0,"last":lambda L:L[-1] if L else 0}
BINOPS={"sub":lambda a,b:a-b,"add":lambda a,b:a+b,"absdiff":lambda a,b:abs(a-b),"mul":lambda a,b:a*b}
CUES=["even","odd","largest","smallest","range","sum","two","second","highest","lowest",
      "difference","total","count","how many","above","below","product","times","minus","plus","big","small"]
ALL_PIECES=list(LIST_OPS)+list(REDUCERS)+list(BINOPS)

# DEEP families (depth-3 and depth-4 programs). depth-4 = binop(reduce(listop),reduce(listop))
# where the listops are themselves meaningful filters -> genuinely deep.
def family(name):
    F={
     # training (shallower, teach word meanings)
     "sum_evens":(lambda L:sum(x for x in L if x%2==0),"sum of evens in {L}"),
     "range_all":(lambda L:max(L)-min(L),"range of {L}"),
     "count_odd":(lambda L:sum(1 for x in L if x%2),"how many odds in {L}"),
     "sum_big":(lambda L:sum(x for x in L if x>10),"sum above 10 in {L}"),
     "prod_minmax":(lambda L:max(L)*min(L),"largest times smallest in {L}"),
     "max_evens":(lambda L:max([x for x in L if x%2==0]) if [x for x in L if x%2==0] else 0,"largest even in {L}"),
     "count_big":(lambda L:sum(1 for x in L if x>10),"how many above 10 in {L}"),
     "sum_odds":(lambda L:sum(x for x in L if x%2),"sum of odds in {L}"),
     # UNSEEN DEEP test families (depth-3/4 compositions of trained word-meanings)
     "range_evens":(lambda L:(max([x for x in L if x%2==0])-min([x for x in L if x%2==0])) if [x for x in L if x%2==0] else 0,"range of evens in {L}"),
     "range_big":(lambda L:(max([x for x in L if x>10])-min([x for x in L if x>10])) if [x for x in L if x>10] else 0,"range of numbers above 10 in {L}"),
     "sum_evens_minus_odds":(lambda L:sum(x for x in L if x%2==0)-sum(x for x in L if x%2),"sum of evens minus sum of odds in {L}"),
     "prod_bigsmall_even":(lambda L:(max([x for x in L if x%2==0])*min([x for x in L if x%2==0])) if [x for x in L if x%2==0] else 0,"largest even times smallest even in {L}"),
     "count_big_minus_small":(lambda L:sum(1 for x in L if x>10)-sum(1 for x in L if x<20),"count above 10 minus count below 20 in {L}"),
    }
    return F[name]

def mk(name,rng):
    fn,phr=family(name); L=[rng.randint(1,40) for _ in range(rng.randint(5,7))]
    try:a=fn(L)
    except Exception:a=None
    return phr.format(L=','.join(map(str,L))),a

def programs_upto_depth4():
    progs=[]
    # depth-2
    for lo_n,lo in LIST_OPS.items():
        for rd_n,rd in REDUCERS.items():
            progs.append((f"{rd_n}({lo_n})",(lambda L,lo=lo,rd=rd:rd(lo(L))),(lo_n,rd_n)))
    # depth-4: binop(reduce(listop), reduce(listop)) - the listops being filters makes this deep
    LI=list(LIST_OPS.items()); RE=list(REDUCERS.items())
    for (loa_n,loa),(rda_n,rda) in itertools.product(LI,RE):
        for (lob_n,lob),(rdb_n,rdb) in itertools.product(LI,RE):
            for op_n,op in BINOPS.items():
                progs.append((f"{op_n}({rda_n}({loa_n}),{rdb_n}({lob_n}))",
                    (lambda L,loa=loa,rda=rda,lob=lob,rdb=rdb,op=op:op(rda(loa(L)),rdb(lob(L)))),
                    (loa_n,rda_n,lob_n,rdb_n,op_n)))
    return progs
ALL=programs_upto_depth4()
def cues_of(q): ql=q.lower(); return [c for c in CUES if c in ql]

class Semantic:
    def __init__(self): self.cp=defaultdict(lambda: defaultdict(float))
    def train(self,q,pieces):
        for c in cues_of(q):
            for p in pieces: self.cp[c][p]+=1.0
    def order(self,q):
        ps=defaultdict(float)
        for c in cues_of(q):
            for p in ALL_PIECES: ps[p]+=self.cp[c][p]
        out=[(sum(ps[p] for p in pieces),n,f,pieces) for n,f,pieces in ALL]
        out.sort(key=lambda x:-x[0]); return [(n,f,p) for _,n,f,p in out]

def examples_for(name,rng,n=6):
    ex=[]
    while len(ex)<n:
        q,a=mk(name,rng)
        if a is not None: ex.append((q,a))
    return ex

def search(order, examples, budget):
    inputs=[nums(q) for q,a in examples]; answers=[a for q,a in examples]
    for i,(name,fn,pieces) in enumerate(order):
        if i>=budget: return None,None,i   # ran out of budget - FAILED to reach
        try:
            if all(fn(inp)==ans for inp,ans in zip(inputs,answers)): return name,fn,i+1
        except Exception: pass
    return None,None,budget

def main():
    rng=random.Random(5)
    print("="*64); print("  NORTH STAR: does learned search solve what brute force CANNOT?"); print("="*64)
    print(f"  program space: {len(ALL)} programs (depth up to 4).")
    train=["sum_evens","range_all","count_odd","sum_big","prod_minmax","max_evens","count_big","sum_odds"]
    test=["range_evens","range_big","sum_evens_minus_odds","prod_bigsmall_even","count_big_minus_small"]
    print(f"  train families (teach word-meanings): {len(train)}")
    print(f"  TEST families (UNSEEN, DEEP): {test}\n")

    sem=Semantic()
    for name in train:
        ex=examples_for(name,rng)
        # find the true program by full search (training time, no budget limit concern)
        n,f,t=search([(nm,fn,pc) for nm,fn,pc in ALL], ex, len(ALL))
        if f: sem.train(ex[0][0], [p for p in ALL if p[1] is f][0][2] if False else
                         next(pc for nm,fn,pc in ALL if fn is f))
    # FROZEN

    # a BUDGET that's a fraction of the space - brute force can't cover deep programs
    BUDGET=1500
    print(f"  compute BUDGET: {BUDGET} program-tries each (space is {len(ALL)}).\n")
    print(f"  {'family':24s}{'brute':>14}{'semantic':>14}")
    brute_solved=0; sem_solved=0
    for name in test:
        ex=examples_for(name,rng)
        _,bf,bt=search([(nm,fn,pc) for nm,fn,pc in ALL], ex, BUDGET)
        _,sf,st=search(sem.order(ex[0][0]), ex, BUDGET)
        b_ok = bf is not None; s_ok = sf is not None
        brute_solved+=b_ok; sem_solved+=s_ok
        bshow = f"{bt} SOLVED" if b_ok else f"{BUDGET} FAILED"
        sshow = f"{st} SOLVED" if s_ok else f"{BUDGET} FAILED"
        print(f"  {name:24s}{bshow:>14}{sshow:>14}")

    print("\n"+"="*64); print("  DID INTELLIGENCE BEAT BRUTE FORCE AT THE WALL?"); print("="*64)
    print(f"  under equal budget ({BUDGET} tries): brute solved {brute_solved}/{len(test)}, "
          f"semantic solved {sem_solved}/{len(test)}")
    if sem_solved > brute_solved:
        print(f"\n  THE COMPUTE WALL, ATTACKED. Under the SAME compute budget, learned")
        print(f"  search solved {sem_solved}/{len(test)} deep unseen problems while brute force reached")
        print(f"  only {brute_solved}/{len(test)}. NEURON solved programs that brute force literally")
        print(f"  could not reach with equal compute, by understanding what the")
        print(f"  problem's words mean and searching the right region. This is the")
        print(f"  North Star version: intelligence that converts understanding into")
        print(f"  reach raw compute can't buy. NEURON doesn't just need more compute,")
        print(f"  it needs LESS of it to go deeper, because it learned where to think.")
    elif sem_solved==brute_solved and sem_solved>0:
        print(f"\n  Both solved {sem_solved}/{len(test)}; semantic was faster but didn't reach")
        print(f"  strictly MORE. Real efficiency gain, not yet 'solves the unreachable'.")
        print(f"  Honest - a bigger budget gap or deeper programs would test it harder.")
    else:
        print(f"\n  Honest: semantic did not solve strictly more than brute under budget.")
        print(f"  Reported straight.")

if __name__=="__main__": main()
