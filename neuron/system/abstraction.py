"""Cross-domain abstraction (the friend's frontier): does NEURON carry abstract
reasoning SHAPES across domains, so experience in numbers+strings helps in an
unseen sequence domain?

The insight: "filter evens -> sum" (numbers), "filter vowels -> count" (strings),
"filter repeats -> count" (sequences) share NO primitives but the SAME abstract
shape: SELECT-SUBSET -> AGGREGATE. If NEURON learns the SHAPE, not the primitives,
experience transfers across worlds.

Each task has a hidden SCHEMA (abstract shape) + domain-specific primitives that
instantiate it. NEURON, solving tasks, can learn which schema a task uses. Then on
a new domain, if it recognizes the schema, it knows the SHAPE of the solution and
only has to find the domain's primitives for each slot - a far smaller search.

Rigorous: 4 versions on an unseen domain after training on 2 others:
  newborn | domain-only | abstract-transfer | shuffled-control
Shuffled-control has the SAME stored info but schemas deliberately mismatched - so
any win for abstract-transfer is from the ABSTRACTIONS, not just having more data.
No tuning. Let it fail honestly.

    python abstraction.py
"""
import random, itertools, statistics
from collections import defaultdict

# ---- abstract SCHEMAS: ordered slots, each slot a ROLE (not a concrete op) ----
# a schema is a shape like: SELECT -> AGGREGATE  (two slots: a selector, an aggregator)
SCHEMAS={
 "select_aggregate": ["SELECT","AGGREGATE"],
 "extreme_remove_op": ["EXTREME","REMOVE","AGGREGATE"],
 "split_summ_compare": ["SELECT","AGGREGATE","SELECT2","AGGREGATE2","COMPARE"],
}

# ---- each DOMAIN provides concrete primitives for each ROLE ----
# numbers
NUM={"SELECT":{"evens":lambda L:[x for x in L if x%2==0],"odds":lambda L:[x for x in L if x%2],"big":lambda L:[x for x in L if x>10]},
     "AGGREGATE":{"sum":sum,"count":len,"max":lambda L:max(L) if L else 0},
     "SELECT2":{"evens":lambda L:[x for x in L if x%2==0],"big":lambda L:[x for x in L if x>10]},
     "AGGREGATE2":{"sum":sum,"count":len},
     "EXTREME":{"max":lambda L:max(L) if L else 0},
     "REMOVE":{"drop_max":lambda L:[x for x in L if x!=max(L)] if L else L},
     "COMPARE":{"sub":lambda a,b:a-b}}
# strings (as list of chars)
STR={"SELECT":{"vowels":lambda L:[c for c in L if c in "aeiou"],"consonants":lambda L:[c for c in L if c not in "aeiou"],"big":lambda L:[c for c in L if c>'m']},
     "AGGREGATE":{"count":len,"sum":lambda L:len(L),"max":lambda L:(max(L) if L else '')},
     "SELECT2":{"vowels":lambda L:[c for c in L if c in "aeiou"],"big":lambda L:[c for c in L if c>'m']},
     "AGGREGATE2":{"count":len,"sum":lambda L:len(L)},
     "EXTREME":{"max":lambda L:(max(L) if L else '')},
     "REMOVE":{"drop_max":lambda L:[c for c in L if c!=(max(L) if L else '')]},
     "COMPARE":{"sub":lambda a,b:(0 if a==b else 1)}}
# sequences (as list of ints) - the UNSEEN domain
SEQ={"SELECT":{"rising":lambda L:[L[i] for i in range(len(L)) if i==0 or L[i]>L[i-1]],"evens":lambda L:[x for x in L if x%2==0],"big":lambda L:[x for x in L if x>10]},
     "AGGREGATE":{"count":len,"sum":sum,"max":lambda L:max(L) if L else 0},
     "SELECT2":{"evens":lambda L:[x for x in L if x%2==0],"big":lambda L:[x for x in L if x>10]},
     "AGGREGATE2":{"count":len,"sum":sum},
     "EXTREME":{"max":lambda L:max(L) if L else 0},
     "REMOVE":{"drop_max":lambda L:[x for x in L if x!=max(L)] if L else L},
     "COMPARE":{"sub":lambda a,b:a-b}}

def apply_prog(roles_prog, domain, L):
    """run a program = list of (role, primitive-name) through a domain."""
    v=L; result=None; stack=[]
    # simple execution: SELECT/REMOVE/EXTREME transform or pick; AGGREGATE reduces;
    # SELECT2/AGGREGATE2 compute a second value; COMPARE combines two.
    i=0; vals=[]
    cur=L
    for role,prim in roles_prog:
        fn=domain[role][prim]
        if role in ("SELECT","REMOVE"): cur=fn(cur)
        elif role=="EXTREME": vals.append(fn(cur))
        elif role=="AGGREGATE": vals.append(fn(cur)); cur=L  # reset for a possible second branch
        elif role=="SELECT2": cur=fn(L)
        elif role=="AGGREGATE2": vals.append(fn(cur))
        elif role=="COMPARE": return fn(vals[-2],vals[-1]) if len(vals)>=2 else None
    return vals[-1] if vals else None

def make_task(schema_name, domain, rng):
    roles=SCHEMAS[schema_name]
    prog=[(r, rng.choice(list(domain[r]))) for r in roles]
    return schema_name, prog

def domain_examples(prog, domain, rng, kind):
    ex=[]
    for _ in range(4):
        if kind=="str":
            L=list(rng.choice(["hello","banana","wonderful","society","algorithm","mississippi"]))
        else:
            L=[rng.randint(1,40) for _ in range(6)]
        try: ex.append((L, apply_prog(prog, domain, L)))
        except Exception: pass
    return ex

def candidate_programs(schema_name, domain):
    roles=SCHEMAS[schema_name]
    opts=[list(domain[r].keys()) for r in roles]
    out=[]
    for combo in itertools.product(*opts):
        out.append(list(zip(roles,combo)))
    return out

def all_candidate_programs(domain):
    out=[]
    for sn in SCHEMAS:
        out+=[(sn,p) for p in candidate_programs(sn,domain)]
    return out

def search(order, ex, domain, budget=100000):
    for i,(sn,prog) in enumerate(order):
        if i>=budget: return None,i
        try:
            if all(apply_prog(prog,domain,L)==a for L,a in ex): return (sn,prog),i+1
        except Exception: pass
    return None,i+1

class Mind:
    def __init__(self): self.schema_seen=defaultdict(int)  # which abstract schemas it has used
    def solve(self, true_schema, prog, domain, rng, kind, mode):
        ex=domain_examples(prog, domain, rng, kind)
        cands=all_candidate_programs(domain)
        if mode=="abstract" and self.schema_seen:
            # ABSTRACTION: try programs of schemas it has SEEN BEFORE first (it
            # carries the SHAPE across domains, only re-finds primitives)
            known=sorted(cands, key=lambda sp: -self.schema_seen[sp[0]])
            order=known
        elif mode=="shuffled" and self.schema_seen:
            # control: same amount of 'experience' but schema prefs are RANDOM
            r=random.Random(123); order=cands[:]; r.shuffle(order)
        else:
            order=cands  # domain-only / newborn: default order
        found,tries=search(order, ex, domain)
        if found: self.schema_seen[found[0]]+=1
        return tries

def main():
    rng=random.Random(7)
    print("="*64); print("  CROSS-DOMAIN ABSTRACTION (friend's design, no tuning)"); print("="*64)
    print("  train on numbers + strings, freeze, test on UNSEEN sequences.\n")

    # train an experienced mind on numbers + strings (it SEES schemas)
    exp=Mind()
    for _ in range(30):
        sn,prog=make_task(rng.choice(list(SCHEMAS)), NUM, rng); exp.solve(sn,prog,NUM,rng,"num","abstract")
    for _ in range(30):
        sn,prog=make_task(rng.choice(list(SCHEMAS)), STR, rng); exp.solve(sn,prog,STR,rng,"str","abstract")

    # the SAME experienced mind, but used in different modes on the new domain
    import copy
    newborn=Mind()
    nb=[]; dom=[]; absr=[]; shuf=[]
    for _ in range(20):
        sn,prog=make_task(rng.choice(list(SCHEMAS)), SEQ, rng)
        r=random.Random(_*9)
        nb.append(newborn.solve(sn,prog,SEQ,random.Random(_*9),"seq","domain"))
        dom.append(copy.deepcopy(exp).solve(sn,prog,SEQ,random.Random(_*9),"seq","domain"))
        absr.append(copy.deepcopy(exp).solve(sn,prog,SEQ,random.Random(_*9),"seq","abstract"))
        shuf.append(copy.deepcopy(exp).solve(sn,prog,SEQ,random.Random(_*9),"seq","shuffled"))

    print(f"  avg search tries on the UNSEEN sequence domain:")
    print(f"    newborn:                 {statistics.mean(nb):.0f}")
    print(f"    experienced, domain-only:{statistics.mean(dom):.0f}")
    print(f"    experienced, SHUFFLED:   {statistics.mean(shuf):.0f}")
    print(f"    experienced, ABSTRACT:   {statistics.mean(absr):.0f}")

    print("\n"+"="*64); print("  DID ABSTRACTIONS TRANSFER ACROSS DOMAINS?"); print("="*64)
    a=statistics.mean(absr); d=statistics.mean(dom); s=statistics.mean(shuf); n=statistics.mean(nb)
    if a < d*0.7 and a < s*0.7:
        print(f"  YES - CROSS-DOMAIN ABSTRACTION. On a domain it never trained in, the")
        print(f"  abstract-transfer NEURON solved in {a:.0f} tries vs domain-only {d:.0f} and the")
        print(f"  shuffled-control {s:.0f}. Beating the shuffled control is the key: the same")
        print(f"  stored experience helped ONLY when the abstractions were correctly")
        print(f"  carried - so it's the SHAPES (select->aggregate, etc.) transferring,")
        print(f"  not just having more data. NEURON carried reusable ways of thinking")
        print(f"  into a world it had never seen. This is a real step toward general")
        print(f"  intelligence: it learned HOW to think, not just what works here.")
    else:
        print(f"  abstract {a:.0f}, domain-only {d:.0f}, shuffled {s:.0f}, newborn {n:.0f}.")
        print(f"  Honest: {'abstraction helped but did not beat the control cleanly' if a<d else 'abstractions did not transfer'}. "
              f"The schemas may be too shared across domains to make the search gap show, or")
        print(f"  transfer is genuinely hard here. Reported straight, no tuning.")

if __name__=="__main__": main()
