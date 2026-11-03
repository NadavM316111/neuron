"""The scaling-curve experiment (the friend's killer test): solve-rate vs program
DEPTH (2-6) under FIXED compute, brute force vs learned search.

The decisive question: as problems get DEEPER, does brute force COLLAPSE while
learned search degrades gracefully? If the gap WIDENS with depth, that shows
something stronger than a single depth-4 win: learning changes the SCALING CURVE
of problem-solving. NEURON's intelligence scales better than brute force with
problem difficulty, which is the heart of the compute-wall attack.

Protocol: learn cue->primitive meaning on shallow training families, FREEZE, then
test on UNSEEN families at each depth under a FIXED compute budget. Plot solve
rate at each depth for brute vs learned. Honest: report wherever each curve falls.

Honest wording (per the friend): "did not reach within the fixed budget", not
"could not reach ever".

    python scaling_curve.py
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
 "gt5":lambda L:[x for x in L if x>5],"identity":lambda L:L,
}
REDUCERS={"sum":sum,"max":lambda L:max(L) if L else 0,"min":lambda L:min(L) if L else 0,
 "len":len,"first":lambda L:L[0] if L else 0,"last":lambda L:L[-1] if L else 0}
BINOPS={"sub":lambda a,b:a-b,"add":lambda a,b:a+b,"absdiff":lambda a,b:abs(a-b),"mul":lambda a,b:a*b}
CUES=["even","odd","largest","smallest","range","sum","two","second","highest","lowest",
      "difference","total","count","how many","above","below","product","times","minus","plus"]
ALL_PIECES=list(LIST_OPS)+list(REDUCERS)+list(BINOPS)
def cues_of(q): ql=q.lower(); return [c for c in CUES if c in ql]

# ---- generate RANDOM ground-truth programs at a target depth, with a worded cue ----
# depth 2: reduce(listop)   depth 3: binop(reduce(listop), number-ish)  etc.
# we build depth by composing: each extra "op" is one more step. We make worded
# problems whose cues hint the pieces (so semantics can transfer).
PIECE_CUE={ "evens":"even","odds":"odd","gt10":"above","lt20":"below","gt5":"above",
 "sorted_desc":"largest","sorted_asc":"smallest","max":"largest","min":"smallest",
 "sum":"sum","len":"count","sub":"difference","add":"sum","mul":"product","first":"first","last":"last"}

def make_program(depth, rng):
    """Build a random program of ~given depth and a worded description from cues."""
    lo=rng.choice(list(LIST_OPS)); rd=rng.choice(list(REDUCERS))
    def f2(L): return REDUCERS[rd](LIST_OPS[lo](L))
    words=[PIECE_CUE.get(rd,rd), "of", PIECE_CUE.get(lo,lo)]
    prog=(f2, (lo,rd))
    cur_depth=2
    while cur_depth<depth:
        # wrap: binop(current, reduce(listop))
        lo2=rng.choice(list(LIST_OPS)); rd2=rng.choice(list(REDUCERS)); op=rng.choice(list(BINOPS))
        prev_f, prev_pieces = prog
        def fn(L, pf=prev_f, lo2=lo2, rd2=rd2, op=op):
            return BINOPS[op](pf(L), REDUCERS[rd2](LIST_OPS[lo2](L)))
        words=[PIECE_CUE.get(op,op)]+words+[PIECE_CUE.get(rd2,rd2),"of",PIECE_CUE.get(lo2,lo2)]
        prog=(fn, prev_pieces+(lo2,rd2,op))
        cur_depth+=1
    desc=" ".join(words)
    return prog[0], prog[1], desc

def programs_at_depth(depth):
    """enumerate candidate programs up to the needed structure. depth 2 = reduce(listop);
    depth>=3 = binop(reduce(listop), reduce(listop)) [we cap enumerated structure at
    this 2-reduction form, which covers our generated targets up to the binop wrap]."""
    progs=[]
    for lo_n,lo in LIST_OPS.items():
        for rd_n,rd in REDUCERS.items():
            progs.append(((lambda L,lo=lo,rd=rd:rd(lo(L))),(lo_n,rd_n)))
    if depth>=3:
        LI=list(LIST_OPS.items()); RE=list(REDUCERS.items())
        for (loa_n,loa),(rda_n,rda) in itertools.product(LI,RE):
            for (lob_n,lob),(rdb_n,rdb) in itertools.product(LI,RE):
                for op_n,op in BINOPS.items():
                    progs.append(((lambda L,loa=loa,rda=rda,lob=lob,rdb=rdb,op=op:op(rda(loa(L)),rdb(lob(L)))),
                                  (loa_n,rda_n,lob_n,rdb_n,op_n)))
    return progs

class Semantic:
    def __init__(self): self.cp=defaultdict(lambda: defaultdict(float))
    def train(self,desc,pieces):
        for c in cues_of(desc):
            for p in pieces: self.cp[c][p]+=1.0
    def order(self, desc, progs):
        ps=defaultdict(float)
        for c in cues_of(desc):
            for p in ALL_PIECES: ps[p]+=self.cp[c][p]
        out=[(sum(ps[p] for p in pieces),fn,pieces) for fn,pieces in progs]
        out.sort(key=lambda x:-x[0]); return [(fn,pc) for _,fn,pc in out]

def examples(fn, rng, n=5):
    ex=[]
    while len(ex)<n:
        L=[rng.randint(1,40) for _ in range(rng.randint(5,7))]
        try: a=fn(L); ex.append((L,a))
        except Exception: pass
    return ex

def search(order, ex, budget):
    for i,(fn,pieces) in enumerate(order):
        if i>=budget: return False,i
        try:
            if all(fn(L)==a for L,a in ex): return True,i+1
        except Exception: pass
    return False,budget

def main():
    rng=random.Random(7)
    print("="*64); print("  SCALING CURVE: solve-rate vs DEPTH under fixed compute"); print("="*64)

    # TRAIN semantics on many shallow random programs (depth 2-3), FREEZE
    sem=Semantic()
    for _ in range(120):
        d=rng.choice([2,3]); fn,pieces,desc=make_program(d,rng)
        sem.train(desc,pieces)
    # also train word->piece directly from the cue map so meanings are grounded
    for piece,cue in PIECE_CUE.items():
        for _ in range(3): sem.cp[cue][piece]+=1.0

    BUDGET=2000
    print(f"  fixed compute budget: {BUDGET} program-tries. 20 unseen tasks per depth.\n")
    print(f"  {'depth':>6}{'space':>9}{'brute solve':>14}{'learned solve':>15}")
    for depth in [2,3,4,5,6]:
        progs=programs_at_depth(depth)
        brute_order=progs  # fixed order
        bsolved=0; lsolved=0; T=20
        for _ in range(T):
            fn,pieces,desc=make_program(depth,rng)
            ex=examples(fn,rng)
            bok,_=search(brute_order, ex, BUDGET)
            lok,_=search(sem.order(desc,progs), ex, BUDGET)
            bsolved+=bok; lsolved+=lok
        print(f"  {depth:>6}{len(progs):>9}{f'{bsolved*100//T}%':>14}{f'{lsolved*100//T}%':>15}")

    print("\n"+"="*64); print("  DOES LEARNING CHANGE THE SCALING CURVE?"); print("="*64)
    print("  If brute force collapses as depth rises while learned search holds")
    print("  meaningful solve rates, learning changed how problem-solving scales")
    print("  with difficulty, a stronger claim than any single-depth win. Whatever")
    print("  the curves show is the honest behavior under fixed compute.")
    print("\n  (Honest scope, per the friend: small symbolic program-search world.")
    print("  'did not reach within the fixed budget', not 'impossible'. The real")
    print("  further test is transfer to other domains - math, planning, logic, code.)")

if __name__=="__main__": main()
