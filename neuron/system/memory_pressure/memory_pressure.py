"""Memory Under Pressure (the friend's reframe): make memory realistic - thousands
of experiences with noise, conflict, obsolescence, distractors - and charge real
tokens for retrieved context. NOW intelligent memory selection has something to do.

The scalable North Star: how can a NEURON with an ENORMOUS life history use very
little expensive cognition at each moment? A mature NEURON can't dump its whole
autobiography into the model every time; it must pull the right 2-3 experiences
from thousands.

First question (oracle-first, the honest discipline): does intelligent memory
SELECTION even have theoretical value here? Compare under a token budget:
  no-memory / retrieve-everything / simple-similarity / learned-selection / oracle
If ORACLE >> always-retrieve, the environment rewards selection -> worth learning.
If oracle ~ always, selection doesn't matter even under pressure -> honest null.

Offline (no hour-long live run) for the theoretical question; live validation only
if it passes.

    python memory_pressure.py
"""
import random, statistics
from collections import defaultdict

# a task has a hidden 'key' (what kind of problem it is). a memory is relevant only
# if its key matches AND it's not obsolete/wrong. retrieving a memory costs tokens
# (context). a WRONG memory (distractor/obsolete/conflicting) that gets used leads
# to a wrong answer unless the model overrides it (costs more tokens to recover).

KINDS=list(range(40))   # 40 distinct problem types

def build_memory_store(rng, N):
    """N stored experiences, mixed quality."""
    store=[]
    for _ in range(N):
        key=rng.choice(KINDS)
        r=rng.random()
        if r<0.25:   typ="useful"      # directly correct for this key
        elif r<0.40: typ="analogical"  # correct only after adaptation (partial)
        elif r<0.55: typ="distractor"  # looks similar (shares surface) but WRONG
        elif r<0.68: typ="obsolete"    # was correct, now outdated -> WRONG
        elif r<0.78: typ="conflicting" # disagrees with a useful one
        else:        typ="irrelevant"  # noise, unrelated
        # 'surface similarity' to a query of the same key is high for useful/distractor/
        # obsolete (they look relevant); low for irrelevant
        surface = rng.uniform(0.7,1.0) if typ in ("useful","distractor","obsolete","conflicting","analogical") else rng.uniform(0,0.3)
        store.append({"key":key,"typ":typ,"surface":surface,"tokens":rng.randint(30,80)})
    return store

def answer_quality(used_memories):
    """given the memories pulled into context, how likely is a correct verified answer?
    useful memory helps; distractor/obsolete/conflicting HURT; irrelevant wastes tokens."""
    q=0.5  # base: the model alone gets ~50% on these hard tasks
    for m in used_memories:
        if m["typ"]=="useful": q+=0.4
        elif m["typ"]=="analogical": q+=0.15
        elif m["typ"]=="distractor": q-=0.25
        elif m["typ"]=="obsolete": q-=0.2
        elif m["typ"]=="conflicting": q-=0.15
        # irrelevant: no quality change, just wasted tokens
    return max(0.02,min(0.98,q))

def retrieve(policy, store, query_key, rng, k_cap=3):
    cands=[m for m in store if m["key"]==query_key or m["surface"]>0.6]
    if policy=="none": return []
    if policy=="everything": return cands[:30]      # dump everything plausible
    if policy=="similarity":
        # pure surface similarity - pulls distractors/obsolete too (they look similar)
        return sorted(cands,key=lambda m:-m["surface"])[:k_cap]
    if policy=="oracle":
        # knows which are genuinely useful/analogical for this key
        good=[m for m in cands if m["key"]==query_key and m["typ"] in ("useful","analogical")]
        return sorted(good,key=lambda m:0 if m["typ"]=="useful" else 1)[:k_cap]
    if policy=="learned":
        # learned selection: prefers same-key + has learned to down-weight types that
        # historically hurt (approximated: filters obvious distractors by a learned score)
        scored=[]
        for m in cands:
            s=m["surface"]
            if m["key"]==query_key: s+=0.3
            s-=LEARNED_TYPE_PENALTY.get(m["typ"],0)   # learned over experience
            scored.append((s,m))
        scored.sort(key=lambda x:-x[0])
        return [m for _,m in scored[:k_cap]]
    return []

# learned penalties (what NEURON would learn: distractor/obsolete/conflicting are bad)
LEARNED_TYPE_PENALTY={"distractor":0.6,"obsolete":0.5,"conflicting":0.4,"irrelevant":0.3}

def run(policy, store, seed, budget=4000):
    rng=random.Random(seed); tokens=0; verified=0
    while tokens<budget:
        qkey=rng.choice(KINDS)
        mems=retrieve(policy, store, qkey, rng)
        ctx_tokens=sum(m["tokens"] for m in mems)
        call_tokens=150   # the model call itself
        if tokens+ctx_tokens+call_tokens>budget: break
        tokens+=ctx_tokens+call_tokens
        q=answer_quality(mems)
        if rng.random()<q: verified+=1
    return verified

def main():
    print("="*64); print("  MEMORY UNDER PRESSURE: does intelligent selection matter?"); print("="*64)
    for N in [500, 5000]:
        print(f"\n  --- memory store size: {N} experiences (mixed quality) ---")
        policies=[("no memory","none"),("retrieve everything","everything"),
                  ("simple similarity","similarity"),("learned selection","learned"),
                  ("oracle selection","oracle")]
        res={}
        for label,p in policies:
            res[p]=statistics.mean(run(p, build_memory_store(random.Random(s),N), s) for s in range(30))
            print(f"    {label:<24}{res[p]:>8.1f} verified / 4000 tokens")
        oracle=res["oracle"]; always=res["everything"]; sim=res["similarity"]
        gap=oracle-always
        print(f"    -> oracle - retrieve-everything = {gap:+.1f}")

    print("\n"+"="*64); print("  DOES SELECTION HAVE THEORETICAL VALUE?"); print("="*64)
    # recompute at the large store for the verdict
    N=5000
    res={p:statistics.mean(run(p, build_memory_store(random.Random(s),N), s) for s in range(30))
         for _,p in [("","none"),("","everything"),("","similarity"),("","learned"),("","oracle")]}
    if res["oracle"]>res["everything"]*1.3:
        print(f"  YES - under memory pressure, intelligent SELECTION matters. Oracle")
        print(f"  ({res['oracle']:.1f}) far beats retrieve-everything ({res['everything']:.1f}) and simple similarity")
        print(f"  ({res['similarity']:.1f}), because dumping everything pulls distractors/obsolete/")
        print(f"  conflicting memories that HURT the answer and waste context tokens. This")
        print(f"  is the right environment: selecting the right 2-3 of thousands genuinely")
        print(f"  pays off. And learned selection ({res['learned']:.1f}) {'closes much of the gap' if res['learned']>res['everything']*1.2 else 'has room to improve toward oracle'}.")
        print(f"  THIS is the scalable intelligence problem: a NEURON with a huge life")
        print(f"  history using little cognition by pulling only what matters.")
    else:
        print(f"  Honest: oracle {res['oracle']:.1f} vs retrieve-everything {res['everything']:.1f} - selection")
        print(f"  doesn't clearly beat dumping everything even under pressure. Reported straight.")

if __name__=="__main__": main()
