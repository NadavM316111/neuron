"""Clean meta-search test (the friend's rigorous design): train a search policy on
one set of task FAMILIES, FREEZE it, test on completely UNSEEN families, and
compare against brute force AND a frequency heuristic. No leakage.

The question: does NEURON genuinely learn WHERE to search (using problem CUES to
pick different search regions for different novel tasks), or did it just get lucky
on one task? The honest test:
  - TRAIN families and TEST families are DISJOINT (no task type appears in both).
  - The policy is FROZEN after training (no learning during test).
  - Three searchers compared on the held-out families:
      brute force (fixed order)
      frequency heuristic (try historically-common pieces first - a dumb baseline)
      learned search (cue -> program-piece associations from training)
  - If learned beats brute AND beats the frequency heuristic on UNSEEN families,
    that's real meta-search (it uses cues to choose regions). If it only beats
    brute but not frequency, it's not real meta-learning - reported honestly.

No teacher/engine here: we use ground-truth answers to build clean examples (so
teacher-noise doesn't confound the search question). We're testing SEARCH, not
teaching.

    python meta_search.py
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
      "difference","total","count","how many","big","above","product","times"]

# a big library of task FAMILIES, each with a ground-truth function + a phrasing
def family(name):
    F={
     "sum_evens":(lambda L:sum(x for x in L if x%2==0), "sum of evens in {L}"),
     "sum_odds":(lambda L:sum(x for x in L if x%2), "sum of odds in {L}"),
     "range_all":(lambda L:max(L)-min(L), "range of {L}"),
     "second_high":(lambda L:sorted(L)[-2], "second highest in {L}"),
     "second_low":(lambda L:sorted(L)[1], "second lowest in {L}"),
     "count_even":(lambda L:sum(1 for x in L if x%2==0), "how many evens in {L}"),
     "count_odd":(lambda L:sum(1 for x in L if x%2), "how many odds in {L}"),
     "sum_top2":(lambda L:sorted(L)[-1]+sorted(L)[-2], "sum of two largest in {L}"),
     "range_evens":(lambda L:(max([x for x in L if x%2==0])-min([x for x in L if x%2==0])) if [x for x in L if x%2==0] else 0, "range of evens in {L}"),
     "max_minus_len":(lambda L:max(L)-len(L), "largest minus count in {L}"),
     "sum_big":(lambda L:sum(x for x in L if x>10), "sum of numbers above 10 in {L}"),
     "diff_hilo_odd":(lambda L:(max([x for x in L if x%2])-min([x for x in L if x%2])) if [x for x in L if x%2] else 0, "range of odds in {L}"),
    }
    return F[name]

def make_instance(name, rng):
    fn,phr=family(name); L=[rng.randint(1,40) for _ in range(rng.randint(4,6))]
    try: a=fn(L)
    except Exception: a=None
    return phr.format(L=','.join(map(str,L))), a

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

class Learned:
    def __init__(self): self.cp=defaultdict(lambda: defaultdict(float))
    def train(self, q, pieces):
        for c in cues_of(q):
            for p in pieces: self.cp[c][p]+=1.0
    def order(self, q):
        sc=[]
        for name,fn,pieces in ALL:
            s=sum(self.cp[c][p] for c in cues_of(q) for p in pieces)
            sc.append((s,name,fn,pieces))
        sc.sort(key=lambda x:-x[0]); return [(n,f,p) for _,n,f,p in sc]

class Frequency:
    """dumb baseline: try historically most-common pieces first, ignoring cues."""
    def __init__(self): self.freq=defaultdict(float)
    def train(self, q, pieces):
        for p in pieces: self.freq[p]+=1.0
    def order(self, q):
        sc=[]
        for name,fn,pieces in ALL:
            s=sum(self.freq[p] for p in pieces); sc.append((s,name,fn,pieces))
        sc.sort(key=lambda x:-x[0]); return [(n,f,p) for _,n,f,p in sc]

def examples_for(name, rng, n=6):
    ex=[]
    while len(ex)<n:
        q,a=make_instance(name,rng)
        if a is not None: ex.append((q,a))
    return ex

def search(order, examples, budget=4000):
    inputs=[nums(q) for q,a in examples]; answers=[a for q,a in examples]
    for i,(name,fn,pieces) in enumerate(order):
        if i>=budget: break
        try:
            if all(fn(inp)==ans for inp,ans in zip(inputs,answers)):
                return name,fn,pieces,i+1
        except Exception: pass
    return None,None,None,budget

def main():
    rng=random.Random(0)
    print("="*64); print("  META-SEARCH (frozen policy, UNSEEN families, vs frequency)"); print("="*64)

    fams=list(["sum_evens","sum_odds","range_all","second_high","second_low","count_even",
               "count_odd","sum_top2","range_evens","max_minus_len","sum_big","diff_hilo_odd"])
    rng.shuffle(fams)
    train_fams=fams[:7]; test_fams=fams[7:]   # DISJOINT
    print(f"  TRAIN families (7): {train_fams}")
    print(f"  TEST families (5, UNSEEN): {test_fams}\n")

    learned=Learned(); freq=Frequency()
    # TRAIN: learn from solved training tasks only (brute-search to find program, then reward)
    for name in train_fams:
        ex=examples_for(name,rng)
        n,f,pieces,t=search([(nm,fn,pc) for nm,fn,pc in ALL], ex)  # brute to find truth
        if pieces:
            q=ex[0][0]; learned.train(q,pieces); freq.train(q,pieces)
    # FREEZE (no more training)

    print("  held-out results (tries to find the correct program):")
    print(f"  {'family':16s}{'brute':>8}{'frequency':>12}{'learned':>10}")
    bt=[]; ft=[]; lt=[]; solved=0
    for name in test_fams:
        ex=examples_for(name,rng)
        _,_,_,b=search([(nm,fn,pc) for nm,fn,pc in ALL], ex)
        _,_,_,fq=search(freq.order(ex[0][0]), ex)
        n,f,pieces,l=search(learned.order(ex[0][0]), ex)
        if pieces: solved+=1
        bt.append(b); ft.append(fq); lt.append(l)
        print(f"  {name:16s}{b:>8}{fq:>12}{l:>10}")

    print("\n"+"="*64); print("  DID NEURON LEARN TO SEARCH (on UNSEEN families)?"); print("="*64)
    mb,mf,ml=statistics.mean(bt),statistics.mean(ft),statistics.mean(lt)
    print(f"  avg tries  |  brute {mb:.0f}   frequency {mf:.0f}   learned {ml:.0f}")
    beats_brute = ml < mb*0.7
    beats_freq = ml < mf*0.7
    if beats_brute and beats_freq:
        print(f"\n  REAL META-SEARCH. On families it NEVER trained on, learned search")
        print(f"  found programs in far fewer tries than brute force ({mb:.0f}->{ml:.0f}) AND beat")
        print(f"  the frequency heuristic ({mf:.0f}->{ml:.0f}). Beating frequency is the key: it")
        print(f"  isn't just trying common pieces - it uses each problem's CUES to search")
        print(f"  a DIFFERENT region for different novel tasks. NEURON learned how to")
        print(f"  search, and it generalizes to unseen problem types. This directly")
        print(f"  attacks the compute wall: NEURON spends its search budget intelligently")
        print(f"  rather than needing more of it.")
    elif beats_brute:
        print(f"\n  Honest: learned search beat brute force ({mb:.0f}->{ml:.0f}) but NOT the frequency")
        print(f"  heuristic ({mf:.0f} vs {ml:.0f}). So far it's mostly trying common pieces first,")
        print(f"  not genuinely using cues to pick regions. Not yet real meta-search -")
        print(f"  reported straight, per the stricter bar.")
    else:
        print(f"\n  Honest: on unseen families, learned search did not clearly beat brute")
        print(f"  force ({mb:.0f} vs {ml:.0f}). The policy didn't generalize. Reported straight.")

if __name__=="__main__": main()
