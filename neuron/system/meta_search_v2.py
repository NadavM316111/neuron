"""Meta-search v2: learn cue -> primitive SEMANTICS that transfer to unseen tasks.

Why v1 failed the clean test: it learned cue -> SPECIFIC-program associations
(memorizing which whole programs solved training tasks). Those don't transfer to
unseen families needing different programs.

v2 learns the MEANING of cue-words at the PRIMITIVE level: "evens" -> the evens
list-op is useful; "difference/range" -> a subtraction; "two/second" -> sorted
operations; "count/how many" -> len/count reducers. These transfer because they
capture what WORDS imply about operations, independent of which task they came
from. A new unseen task saying "range of the evens" should get BOTH the evens-op
AND subtraction prioritized, even though that exact task was never trained on -
because it learned what "range" and "evens" each mean.

Same strict protocol: train on 7 families, FREEZE, test on 5 DISJOINT unseen
families, compare vs brute force AND frequency heuristic. Honest verdict.

    python meta_search_v2.py
"""
import random, re, itertools, statistics
from collections import defaultdict

def nums(q): return list(map(int, re.findall(r"-?\d+", q)))

LIST_OPS={
 "sorted_desc":lambda L:sorted(L,reverse=True),"sorted_asc":lambda L:sorted(L),
 "evens":lambda L:[x for x in L if x%2==0],"odds":lambda L:[x for x in L if x%2],
 "drop_max":lambda L:[x for x in L if x!=max(L)] if L else L,
 "drop_min":lambda L:[x for x in L if x!=min(L)] if L else L,
 "gt10":lambda L:[x for x in L if x>10],"identity":lambda L:L,
}
REDUCERS={"sum":sum,"max":lambda L:max(L) if L else 0,"min":lambda L:min(L) if L else 0,
 "len":len,"first":lambda L:L[0] if L else 0,"last":lambda L:L[-1] if L else 0}
BINOPS={"sub":lambda a,b:a-b,"add":lambda a,b:a+b,"absdiff":lambda a,b:abs(a-b),"mul":lambda a,b:a*b}
CUES=["even","odd","largest","smallest","range","sum","two","second","highest","lowest",
      "difference","total","count","how many","above","product","times","minus","plus"]
ALL_PIECES=list(LIST_OPS)+list(REDUCERS)+list(BINOPS)

def family(name):
    F={
     "sum_evens":(lambda L:sum(x for x in L if x%2==0),"sum of evens in {L}"),
     "sum_odds":(lambda L:sum(x for x in L if x%2),"sum of odds in {L}"),
     "range_all":(lambda L:max(L)-min(L),"range of {L}"),
     "second_high":(lambda L:sorted(L)[-2],"second highest in {L}"),
     "second_low":(lambda L:sorted(L)[1],"second lowest in {L}"),
     "count_even":(lambda L:sum(1 for x in L if x%2==0),"how many evens in {L}"),
     "count_odd":(lambda L:sum(1 for x in L if x%2),"how many odds in {L}"),
     "sum_top2":(lambda L:sorted(L)[-1]+sorted(L)[-2],"sum of two largest in {L}"),
     "range_evens":(lambda L:(max([x for x in L if x%2==0])-min([x for x in L if x%2==0])) if [x for x in L if x%2==0] else 0,"range of evens in {L}"),
     "max_minus_len":(lambda L:max(L)-len(L),"largest minus count in {L}"),
     "sum_big":(lambda L:sum(x for x in L if x>10),"sum above 10 in {L}"),
     "diff_hilo_odd":(lambda L:(max([x for x in L if x%2])-min([x for x in L if x%2])) if [x for x in L if x%2] else 0,"range of odds in {L}"),
     "prod_minmax":(lambda L:max(L)*min(L),"largest times smallest in {L}"),
     "sum_sorted_top":(lambda L:sorted(L)[-1]+sorted(L)[-2],"total of two highest in {L}"),
    }
    return F[name]

def make_instance(name,rng):
    fn,phr=family(name); L=[rng.randint(1,40) for _ in range(rng.randint(4,6))]
    try:a=fn(L)
    except Exception:a=None
    return phr.format(L=','.join(map(str,L))),a

def all_programs():
    progs=[]
    for lo_n,lo in LIST_OPS.items():
        for rd_n,rd in REDUCERS.items():
            progs.append((f"{rd_n}({lo_n})",(lambda L,lo=lo,rd=rd:rd(lo(L))),(lo_n,rd_n)))
    for (loa_n,loa),(rda_n,rda) in itertools.product(LIST_OPS.items(),REDUCERS.items()):
        for (lob_n,lob),(rdb_n,rdb) in itertools.product(LIST_OPS.items(),REDUCERS.items()):
            for op_n,op in BINOPS.items():
                progs.append((f"{op_n}({rda_n}({loa_n}),{rdb_n}({lob_n}))",
                    (lambda L,loa=loa,rda=rda,lob=lob,rdb=rdb,op=op:op(rda(loa(L)),rdb(lob(L)))),
                    (loa_n,rda_n,lob_n,rdb_n,op_n)))
    return progs
ALL=all_programs()
def cues_of(q): ql=q.lower(); return [c for c in CUES if c in ql]

class Semantic:
    """learns cue-word -> primitive-PIECE affinity (the MEANING of words), which
    transfers across tasks."""
    def __init__(self): self.cp=defaultdict(lambda: defaultdict(float))
    def train(self, q, pieces):
        cs=cues_of(q)
        for c in cs:
            for p in pieces: self.cp[c][p]+=1.0
    def piece_score(self, q):
        sc=defaultdict(float)
        for c in cues_of(q):
            for p in ALL_PIECES: sc[p]+=self.cp[c][p]
        return sc
    def order(self, q):
        ps=self.piece_score(q); out=[]
        for name,fn,pieces in ALL:
            s=sum(ps[p] for p in pieces); out.append((s,name,fn,pieces))
        out.sort(key=lambda x:-x[0]); return [(n,f,p) for _,n,f,p in out]

class Frequency:
    def __init__(self): self.freq=defaultdict(float)
    def train(self,q,pieces):
        for p in pieces: self.freq[p]+=1.0
    def order(self,q):
        out=[]
        for name,fn,pieces in ALL:
            s=sum(self.freq[p] for p in pieces); out.append((s,name,fn,pieces))
        out.sort(key=lambda x:-x[0]); return [(n,f,p) for _,n,f,p in out]

def examples_for(name,rng,n=6):
    ex=[]
    while len(ex)<n:
        q,a=make_instance(name,rng)
        if a is not None: ex.append((q,a))
    return ex

def search(order, examples, budget=5000):
    inputs=[nums(q) for q,a in examples]; answers=[a for q,a in examples]
    for i,(name,fn,pieces) in enumerate(order):
        if i>=budget: break
        try:
            if all(fn(inp)==ans for inp,ans in zip(inputs,answers)): return name,fn,pieces,i+1
        except Exception: pass
    return None,None,None,budget

def main():
    rng=random.Random(3)
    print("="*64); print("  META-SEARCH v2 (cue->primitive semantics, clean protocol)"); print("="*64)
    fams=["sum_evens","sum_odds","range_all","second_high","second_low","count_even",
          "count_odd","sum_top2","range_evens","max_minus_len","sum_big","diff_hilo_odd",
          "prod_minmax","sum_sorted_top"]
    rng.shuffle(fams); train_fams=fams[:9]; test_fams=fams[9:]
    print(f"  TRAIN (9): {train_fams}")
    print(f"  TEST (5, UNSEEN): {test_fams}\n")

    sem=Semantic(); freq=Frequency()
    for name in train_fams:
        ex=examples_for(name,rng)
        n,f,pieces,t=search([(nm,fn,pc) for nm,fn,pc in ALL], ex)
        if pieces:
            q=ex[0][0]; sem.train(q,pieces); freq.train(q,pieces)
    # FROZEN

    print(f"  {'family':16s}{'brute':>8}{'frequency':>12}{'semantic':>11}")
    bt=[];ft=[];lt=[]
    for name in test_fams:
        ex=examples_for(name,rng)
        _,_,_,b=search([(nm,fn,pc) for nm,fn,pc in ALL], ex)
        _,_,_,fq=search(freq.order(ex[0][0]), ex)
        _,_,_,l=search(sem.order(ex[0][0]), ex)
        bt.append(b);ft.append(fq);lt.append(l)
        print(f"  {name:16s}{b:>8}{fq:>12}{l:>11}")

    mb,mf,ml=statistics.mean(bt),statistics.mean(ft),statistics.mean(lt)
    print("\n"+"="*64); print("  VERDICT (on UNSEEN families)"); print("="*64)
    print(f"  avg tries  |  brute {mb:.0f}   frequency {mf:.0f}   semantic {ml:.0f}")
    if ml<mb*0.7 and ml<mf*0.7:
        print(f"\n  REAL GENERALIZING META-SEARCH. Learning cue->primitive MEANING")
        print(f"  (what 'evens','range','two' each imply about operations) transferred to")
        print(f"  UNSEEN task families: semantic search found programs in far fewer tries")
        print(f"  than brute ({mb:.0f}->{ml:.0f}) AND beat the frequency heuristic ({mf:.0f}->{ml:.0f}). Because")
        print(f"  it learned word-meaning, not task-memorization, it points new problems")
        print(f"  at the right search region. THIS attacks the compute wall: NEURON")
        print(f"  spends search budget by understanding the problem, not brute force.")
    elif ml<mf*0.85:
        print(f"\n  Partial: semantic edges out frequency ({mf:.0f} vs {ml:.0f}) and beats brute,")
        print(f"  but the margin over the dumb baseline is thin. Some real transfer,")
        print(f"  not a decisive meta-search win. Honest.")
    else:
        print(f"\n  Honest NEGATIVE: semantic search ({ml:.0f}) did not clearly beat the")
        print(f"  frequency heuristic ({mf:.0f}). Cue->primitive learning didn't transfer")
        print(f"  decisively. Generalizing meta-search still not established. Reported straight.")

if __name__=="__main__": main()
