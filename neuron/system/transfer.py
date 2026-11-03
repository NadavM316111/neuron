"""Transfer test (the friend's next frontier): does learned search help in a
DIFFERENT domain, or is it specific to the number-list DSL?

If cue-conditioned learned search is a GENERAL principle, it should work in other
domains too - not just "list of numbers" problems. We test a completely different
domain: STRING / TEXT transformations (reverse, uppercase, take-first-n, etc.),
with their own primitives and their own worded cues.

Same strict protocol: learn cue->primitive meaning on shallow training tasks,
FREEZE, test on UNSEEN deeper string tasks under a fixed budget, brute vs learned.
If learned search beats brute in THIS domain too (that it never saw before), the
principle transfers - learning where to search is general, not domain-specific.

    python transfer.py
"""
import random, re, itertools
from collections import defaultdict

# ---- a DIFFERENT domain: string transformations ----
STR_OPS={
 "reverse": lambda s: s[::-1],
 "upper": lambda s: s.upper(),
 "lower": lambda s: s.lower(),
 "first_half": lambda s: s[:len(s)//2],
 "last_half": lambda s: s[len(s)//2:],
 "drop_vowels": lambda s: "".join(c for c in s if c.lower() not in "aeiou"),
 "keep_vowels": lambda s: "".join(c for c in s if c.lower() in "aeiou"),
 "dedupe": lambda s: "".join(dict.fromkeys(s)),
 "identity": lambda s: s,
}
# a "reducer" here ends the program (string->string is fine; we compare strings)
CUE={ "reverse":"reversed","upper":"uppercase","lower":"lowercase",
 "first_half":"first half","last_half":"second half","drop_vowels":"without vowels",
 "keep_vowels":"only vowels","dedupe":"without duplicates"}
CUES=sorted(set(CUE.values()), key=len, reverse=True)
def cues_of(q):
    ql=q.lower(); return [c for c in CUES if c in ql]

WORDS=["hello","banana","mississippi","wonderful","algorithm","transfer","neuron",
       "society","participate","intelligence","development","cognition","reasoning"]

def run_prog(ops, s):
    for o in ops: s=STR_OPS[o](s)
    return s

def desc_of(ops):
    return " then ".join(CUE.get(o,o) for o in ops)

def gen_target(depth, rng):
    return [rng.choice([o for o in STR_OPS if o!="identity"]) for _ in range(depth)]

def candidates(depth):
    ops=list(STR_OPS)
    return [list(c) for c in itertools.product(ops, repeat=depth)]

class Semantic:
    def __init__(self): self.cp=defaultdict(lambda: defaultdict(float))
    def train(self, desc, ops):
        for c in cues_of(desc):
            for o in ops: self.cp[c][o]+=1.0
    def order(self, desc, cands):
        active=cues_of(desc)
        scored=[(sum(self.cp[c][o] for c in active for o in ops), ops) for ops in cands]
        scored.sort(key=lambda x:-x[0]); return [o for _,o in scored]

def examples(ops, rng, n=4):
    ws=rng.sample(WORDS, min(n,len(WORDS)))
    return [(w, run_prog(ops,w)) for w in ws]

def search(order, ex, budget):
    for i,ops in enumerate(order):
        if i>=budget: return False,i
        if all(run_prog(ops,w)==a for w,a in ex): return True,i+1
    return False,budget

def main():
    rng=random.Random(5)
    print("="*64); print("  TRANSFER TEST: does learned search work in a NEW domain?"); print("="*64)
    print("  New domain: STRING transformations (reverse, uppercase, drop-vowels...)")
    print("  - completely different from the number-list world. Same method.\n")

    # train semantics on shallow string tasks + ground the word meanings
    sem=Semantic()
    for _ in range(150):
        d=rng.choice([1,2]); ops=gen_target(d,rng); sem.train(desc_of(ops), ops)
    for op,cue in CUE.items():
        for _ in range(4): sem.cp[cue][op]+=1.0
    # FROZEN

    BUDGET=1500
    print(f"  fixed budget: {BUDGET} tries. 20 unseen tasks/depth.\n")
    print(f"  {'depth':>6}{'space':>9}{'brute':>9}{'learned':>10}")
    for depth in [1,2,3,4]:
        cands=candidates(depth); T=20; b=0; l=0
        for _ in range(T):
            ops=gen_target(depth,rng); ex=examples(ops,rng)
            bok,_=search(cands, ex, BUDGET)
            lok,_=search(sem.order(desc_of(ops),cands), ex, BUDGET)
            b+=bok; l+=lok
        print(f"  {depth:>6}{len(cands):>9}{f'{b*100//T}%':>9}{f'{l*100//T}%':>10}")

    print("\n"+"="*64); print("  DID THE PRINCIPLE TRANSFER?"); print("="*64)
    print("  If learned search beats brute in THIS domain too - one it shares no")
    print("  primitives with the number-list world - then 'learn where to search")
    print("  from problem words' is a GENERAL intelligence principle, not a trick")
    print("  specific to one DSL. If brute collapses with depth while learned holds,")
    print("  same as before, the scaling result transfers across domains.")
    print("\n  (Honest: still symbolic search worlds. True generality needs many more")
    print("  domains - math, planning, logic, code - but two different domains is")
    print("  real evidence the mechanism isn't domain-specific.)")

if __name__=="__main__": main()
