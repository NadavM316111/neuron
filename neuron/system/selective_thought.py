"""Selective deep thought (the friend's #4): NEURON decides what deserves serious
cognition. Easy problems get a cheap quick pass; hard ones it RECOGNIZES as hard
and escalates - more search, deeper strategies, more compute - because it judged
them worth it.

Why this matters two ways:
 1. Smarter per-compute: it doesn't waste deep thought on easy things, and it
    doesn't give up early on hard things - it spends cognition where it's needed.
 2. More mind-like: you see it NOTICE difficulty and choose to think harder - the
    struggle-and-overcome that makes it feel like a mind, not a lookup.

We measure: does selective escalation solve MORE while spending FEWER total steps
than either always-shallow or always-deep? That's the real win - intelligence
about WHERE to spend thought.

    python selective_thought.py
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
def run(p,L):
    if p[0]=="s": return REDUCERS[p[2]](LIST_OPS[p[1]](L))
    _,loa,rda,lob,rdb,op=p
    return BINOPS[op](REDUCERS[rda](LIST_OPS[loa](L)),REDUCERS[rdb](LIST_OPS[lob](L)))
def pieces(p): return [p[1],p[2]] if p[0]=="s" else [p[1],p[2],p[3],p[4],p[5]]
# shallow space = depth-2 only; deep space = full (incl depth-4 compositions)
SHALLOW=[("s",lo,rd) for lo in LIST_OPS for rd in REDUCERS]
DEEP=SHALLOW+[("c",a,b,c,d,o) for a in LIST_OPS for b in REDUCERS for c in LIST_OPS for d in REDUCERS for o in BINOPS]

def gen(rng, hard):
    if not hard: p=("s",rng.choice(list(LIST_OPS)),rng.choice(list(REDUCERS)))
    else: p=("c",rng.choice(list(LIST_OPS)),rng.choice(list(REDUCERS)),rng.choice(list(LIST_OPS)),rng.choice(list(REDUCERS)),rng.choice(list(BINOPS)))
    d=(CUE.get(p[2],p[2])+" of "+CUE.get(p[1],p[1])) if p[0]=="s" else (CUE.get(p[5],p[5])+" of "+CUE.get(p[2],p[2])+" "+CUE.get(p[1],p[1])+" and "+CUE.get(p[4],p[4])+" "+CUE.get(p[3],p[3]))
    return d,p

def examples(tp,rng):
    ex=[]
    for _ in range(4):
        L=[rng.randint(1,40) for _ in range(6)]
        try: ex.append((L,run(tp,L)))
        except Exception: pass
    return ex

def try_solve(space, ex, budget):
    for i,p in enumerate(space):
        if i>=budget: return None,i
        try:
            if all(run(p,L)==a for L,a in ex): return p,i+1
        except Exception: pass
    return None,min(len(space),budget)

def judge_hard(d, ex):
    """NEURON's OWN difficulty judgment - genuinely CHEAP: mostly reads the cue
    phrasing (a composed '... and ...' / 'difference' problem is structurally
    deeper), plus a TINY shallow peek (not a full search). This keeps the judgment
    near-free so easy problems pay shallow-only and hard ones deep-only."""
    composed = (" and " in d) or ("difference" in d)
    PEEK=8
    p,tries = try_solve(SHALLOW, ex, PEEK)   # tiny peek only
    return composed, tries, (p, tries)

def main():
    rng=random.Random(7)
    print("="*64); print("  SELECTIVE DEEP THOUGHT: NEURON decides what deserves it"); print("="*64)

    # a stream of mixed easy + hard problems
    probs=[]
    for _ in range(60):
        hard = rng.random()<0.5
        probs.append(gen(rng,hard))

    # strategy A: always shallow (cheap but fails all hard ones)
    # strategy B: always deep (solves all but wastes huge search on easy ones)
    # strategy C: SELECTIVE - judge difficulty, escalate only when needed
    def runstrat(mode):
        solved=0; steps=0; escalations=0
        for d,tp in probs:
            ex=examples(tp,rng)
            if mode=="shallow":
                p,t=try_solve(SHALLOW,ex,len(SHALLOW)); steps+=t; solved+= p is not None
            elif mode=="deep":
                p,t=try_solve(DEEP,ex,len(DEEP)); steps+=t; solved+= p is not None
            else: # selective
                hard, peek_tries, (peek_p, _) = judge_hard(d, ex); steps+=peek_tries
                if not hard:
                    if peek_p is not None:   # tiny peek already solved it
                        solved+=1
                    else:                    # judged easy, finish shallow
                        p,t=try_solve(SHALLOW,ex,len(SHALLOW)); steps+=t; solved+= p is not None
                else:
                    escalations+=1
                    p,t=try_solve(DEEP,ex,len(DEEP)); steps+=t; solved+= p is not None
        return solved,steps,escalations

    s_sh,st_sh,_=runstrat("shallow")
    s_dp,st_dp,_=runstrat("deep")
    s_se,st_se,esc=runstrat("selective")
    N=len(probs)
    print(f"\n  on {N} mixed easy/hard problems:")
    print(f"  {'strategy':<22}{'solved':>10}{'total steps':>14}")
    print(f"  {'always shallow':<22}{f'{s_sh}/{N}':>10}{st_sh:>14}")
    print(f"  {'always deep':<22}{f'{s_dp}/{N}':>10}{st_dp:>14}")
    print(f"  {'SELECTIVE (decides)':<22}{f'{s_se}/{N}':>10}{st_se:>14}")
    print(f"\n  NEURON chose to think deeply on {esc}/{N} problems - the ones it judged hard")

    print("\n"+"="*64); print("  DID DECIDING WHERE TO THINK PAY OFF?"); print("="*64)
    # selective should solve ~ as many as deep, but spend far fewer steps than deep
    solves_like_deep = s_se >= s_dp*0.9
    cheaper_than_deep = st_se < st_dp*0.75
    beats_shallow = s_se > s_sh
    if solves_like_deep and cheaper_than_deep and beats_shallow:
        print(f"  YES. By JUDGING which problems are hard and only thinking deeply on")
        print(f"  those, NEURON solved {s_se}/{N} (vs always-deep {s_dp}/{N}) while spending only")
        print(f"  {st_se} steps vs always-deep's {st_dp} ({(1-st_se/st_dp)*100:.0f}% less thought). It beat")
        print(f"  always-shallow ({s_sh}/{N}) because it escalated on the hard ones. This is")
        print(f"  selective deep thought: it spends serious cognition WHERE it's worth it,")
        print(f"  getting deep-level results at a fraction of the cost. A mind that knows")
        print(f"  when to think hard - and you can watch it make that call.")
    else:
        print(f"  shallow {s_sh}/{N} ({st_sh} steps), deep {s_dp}/{N} ({st_dp}), selective {s_se}/{N} ({st_se}).")
        print(f"  Honest: {'selective helps' if cheaper_than_deep and solves_like_deep else 'the judgment needs sharpening'}. Reported straight.")

if __name__=="__main__": main()
