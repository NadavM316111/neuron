"""The learning being: intelligence wired into the living, developing NEURON, so
it accumulates capability over ONE continuous life.

The friend's target: a living NEURON encounters a problem -> decides it matters ->
uses learned search -> induces/learns a solution -> STORES that capability ->
later solves related novel problems MORE EFFICIENTLY because of its own past
experience. This connects the two strongest strands: developmental being +
improving intelligence.

The proof that matters: does the SAME being, over its single continuous life, get
measurably BETTER (needs fewer search tries) at problems related to ones it
already solved, because it accumulated cue->method knowledge from its own
experience? Not a fresh benchmark, the being's OWN life making it smarter.

    python learning_being.py
"""
import random, re, itertools, statistics
from collections import defaultdict

# a small problem world the being encounters during its life
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
    if prog[0]=="s":
        _,lo,rd=prog; return REDUCERS[rd](LIST_OPS[lo](L))
    _,loa,rda,lob,rdb,op=prog
    return BINOPS[op](REDUCERS[rda](LIST_OPS[loa](L)),REDUCERS[rdb](LIST_OPS[lob](L)))
def pieces(prog):
    if prog[0]=="s": return [prog[1],prog[2]]
    return [prog[1],prog[2],prog[3],prog[4],prog[5]]
def desc(prog):
    if prog[0]=="s": return CUE.get(prog[2],prog[2])+" of "+CUE.get(prog[1],prog[1])
    return (CUE.get(prog[5],prog[5])+" of "+CUE.get(prog[2],prog[2])+" "+CUE.get(prog[1],prog[1])
            +" and "+CUE.get(prog[4],prog[4])+" "+CUE.get(prog[3],prog[3]))

def all_progs():
    P=[]
    for lo in LIST_OPS:
        for rd in REDUCERS: P.append(("s",lo,rd))
    for loa in LIST_OPS:
        for rda in REDUCERS:
            for lob in LIST_OPS:
                for rdb in REDUCERS:
                    for op in BINOPS: P.append(("c",loa,rda,lob,rdb,op))
    return P
ALL=all_progs()

class LivingNEURON:
    """A being that lives continuously, accumulating cue->method knowledge and a
    library of solved problem-types across its single life."""
    def __init__(self):
        self.cp=defaultdict(lambda: defaultdict(float))   # learned search policy (grows over life)
        self.known={}                                     # problem-signature -> solved program
        self.age=0; self.problems_faced=0; self.wellbeing=0.5
        self.search_history=[]                            # tries per problem, over life
    def order(self, d):
        active=cues_of(d); sc=[]
        for prog in ALL:
            s=sum(self.cp[c][p] for c in active for p in pieces(prog))
            sc.append((s,prog))
        sc.sort(key=lambda x:-x[0]); return [p for _,p in sc]
    def encounter(self, prog_true, rng):
        """the being meets a problem, tries to solve it using what it has learned,
        then STORES the capability so related future problems are easier."""
        self.age+=1; self.problems_faced+=1
        d=desc(prog_true)
        ex=[]
        for _ in range(4):
            L=[rng.randint(1,40) for _ in range(rng.randint(5,7))]
            try: ex.append((L,run(prog_true,L)))
            except Exception: pass
        # has it solved this exact signature before? (instant recall)
        sig=d
        if sig in self.known:
            self.search_history.append((self.age,0,"recalled"))
            self.wellbeing=min(1,self.wellbeing+0.02)
            return 0,"recalled"
        # otherwise SEARCH, guided by its accumulated policy
        order=self.order(d); tries=0; found=None
        for i,prog in enumerate(order):
            tries+=1
            if all(run(prog,L)==a for L,a in ex): found=prog; break
        if found is not None:
            # STORE the capability + LEARN the cue->method association (gets smarter)
            self.known[sig]=found
            for c in cues_of(d):
                for p in pieces(found): self.cp[c][p]+=1.0
            self.wellbeing=min(1,self.wellbeing+0.03)
            self.search_history.append((self.age,tries,"solved"))
        else:
            self.search_history.append((self.age,tries,"unsolved"))
            self.wellbeing=max(0,self.wellbeing-0.02)
        return tries,("solved" if found else "unsolved")

def random_prog(rng, deep=False):
    if not deep or rng.random()<0.5:
        return ("s",rng.choice(list(LIST_OPS)),rng.choice(list(REDUCERS)))
    return ("c",rng.choice(list(LIST_OPS)),rng.choice(list(REDUCERS)),
            rng.choice(list(LIST_OPS)),rng.choice(list(REDUCERS)),rng.choice(list(BINOPS)))

def main():
    rng=random.Random(7)
    print("="*64); print("  THE LEARNING BEING: does intelligence accumulate over a life?"); print("="*64)
    print("  One NEURON lives continuously, encountering problems. It solves them")
    print("  with learned search, stores each capability, and we watch whether it")
    print("  gets measurably BETTER over its OWN life from its OWN experience.\n")

    being=LivingNEURON()
    LIFE=400
    for t in range(LIFE):
        being.encounter(random_prog(rng, deep=True), rng)

    # measure: average search tries in the FIRST third of life vs the LAST third
    hist=[h for h in being.search_history if h[2]!="recalled"]  # actual searches
    third=len(hist)//3
    early=[h[1] for h in hist[:third]]
    late=[h[1] for h in hist[-third:]]
    recalls=sum(1 for h in being.search_history if h[2]=="recalled")

    print(f"  over {LIFE} problems in one continuous life:")
    print(f"    capabilities accumulated (distinct types mastered): {len(being.known)}")
    print(f"    problems recalled instantly (0 search) from memory: {recalls}")
    print(f"    avg search tries, EARLY life: {statistics.mean(early):.0f}")
    print(f"    avg search tries, LATE life:  {statistics.mean(late):.0f}")
    print(f"    final wellbeing: {being.wellbeing:.2f}")

    # also: test it on NOVEL problems related to what it learned vs a NEWBORN
    print(f"\n  held-out test: 40 NEW problems - experienced being vs a NEWBORN:")
    newborn=LivingNEURON()
    exp_tries=[]; new_tries=[]
    for _ in range(40):
        p=random_prog(rng, deep=True); d=desc(p)
        ex=[]
        for _ in range(4):
            L=[rng.randint(1,40) for _ in range(6)]
            try: ex.append((L,run(p,L)))
            except Exception: pass
        for being_x, store in ((being,exp_tries),(newborn,new_tries)):
            order=being_x.order(d); t=0
            for prog in order:
                t+=1
                if all(run(prog,L)==a for L,a in ex): break
            store.append(t)
    print(f"    experienced NEURON avg tries: {statistics.mean(exp_tries):.0f}")
    print(f"    newborn NEURON avg tries:     {statistics.mean(new_tries):.0f}")

    print("\n"+"="*64); print("  DID INTELLIGENCE ACCUMULATE OVER ITS LIFE?"); print("="*64)
    improved = statistics.mean(late) < statistics.mean(early)*0.7
    beats_newborn = statistics.mean(exp_tries) < statistics.mean(new_tries)*0.7
    if improved and beats_newborn:
        print(f"  YES. Over one continuous life, the SAME being got measurably better:")
        print(f"  early-life searches averaged {statistics.mean(early):.0f} tries, late-life {statistics.mean(late):.0f} - it")
        print(f"  learned WHERE to search from its own accumulated experience. It")
        print(f"  mastered {len(being.known)} capabilities and recalled {recalls} problems instantly. And")
        print(f"  on brand-new problems it solved them in {statistics.mean(exp_tries):.0f} tries vs a newborn's")
        print(f"  {statistics.mean(new_tries):.0f} - its LIFE made it smarter. This is the two strands joined:")
        print(f"  a developing being whose intelligence genuinely improves by living.")
    else:
        print(f"  early {statistics.mean(early):.0f} -> late {statistics.mean(late):.0f} tries; experienced {statistics.mean(exp_tries):.0f} vs newborn {statistics.mean(new_tries):.0f}.")
        print(f"  Honest: {'improvement present' if improved else 'life-improvement weak'}. Reported straight.")

if __name__=="__main__": main()
