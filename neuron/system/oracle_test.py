"""Oracle-abstraction test (the friend's step 1): IF NEURON were handed the correct
abstract shape for free (solution withheld), would that reduce its search on an
unseen domain? Diagnoses the RIGHT layer before building shape inference.

Compositional: a program is roles (SELECT/TRANSFORM/AGGREGATE) in order; the roles
stay constant across domains, the implementations differ. A "shape" = which roles.

If ORACLE (given correct shape) << newborn, abstraction is useful and the missing
piece is INFERENCE -> build that next. If ORACLE ~= newborn, the search can't
exploit abstraction -> stop, wrong layer. 120 seeds, fixed budget, no tuning.

    python oracle_test.py
"""
import random, itertools, statistics

def num_select(i):
    preds=[lambda x:x%2==0, lambda x:x%2==1, lambda x:x>10, lambda x:x>5, lambda x:x<20, lambda x:True]
    p=preds[i]; return lambda L: [x for x in L if p(x)]
def str_select(i):
    preds=[lambda c:c in "aeiou", lambda c:c not in "aeiou", lambda c:c>'m', lambda c:c>'f', lambda c:c<'t', lambda c:True]
    p=preds[i]; return lambda L: [c for c in L if p(c)]

def dom_numbers():
    return {
      "TRANSFORM":{"t0":lambda L:sorted(L),"t1":lambda L:sorted(L,reverse=True),"t2":lambda L:L},
      "SELECT":{f"s{i}":num_select(i) for i in range(6)},
      "AGGREGATE":{"a0":lambda L:sum(L),"a1":lambda L:(max(L) if L else 0),"a2":lambda L:(min(L) if L else 0),"a3":lambda L:len(L)},
    }
def dom_strings():
    return {
      "TRANSFORM":{"t0":lambda L:L[::-1],"t1":lambda L:sorted(L),"t2":lambda L:L},
      "SELECT":{f"s{i}":str_select(i) for i in range(6)},
      "AGGREGATE":{"a0":lambda L:len(L),"a1":lambda L:(max(L) if L else ''),"a2":lambda L:(min(L) if L else ''),"a3":lambda L:len(set(L))},
    }
def dom_sequences():
    return {
      "TRANSFORM":{"t0":lambda L:[L[k]-L[k-1] for k in range(1,len(L))],"t1":lambda L:sorted(L),"t2":lambda L:L},
      "SELECT":{f"s{i}":num_select(i) for i in range(6)},
      "AGGREGATE":{"a0":lambda L:sum(L),"a1":lambda L:(max(L) if L else 0),"a2":lambda L:(min(L) if L else 0),"a3":lambda L:len(L)},
    }

SHAPES=[["SELECT","AGGREGATE"],["TRANSFORM","SELECT","AGGREGATE"],
        ["SELECT","TRANSFORM","AGGREGATE"],["TRANSFORM","AGGREGATE"]]

def run(prog, domain, L):
    v=L
    for role,prim in prog: v=domain[role][prim](v)
    return v
def programs_of_shape(shape, domain):
    opts=[list(domain[r]) for r in shape]
    return [list(zip(shape,combo)) for combo in itertools.product(*opts)]
def all_programs(domain):
    out=[]
    for sh in SHAPES: out+=programs_of_shape(sh,domain)
    return out
def make_task(domain, rng):
    shape=rng.choice(SHAPES); return shape,[(r,rng.choice(list(domain[r]))) for r in shape]
def examples(prog, domain, rng, kind):
    out=[]
    for _ in range(4):
        L=list(rng.choice(["hello","banana","wonderful","algorithm","mississippi","society"])) if kind=="str" else [rng.randint(1,30) for _ in range(6)]
        try: out.append((L,run(prog,domain,L)))
        except Exception: pass
    return out
def search(order, ex, domain, budget=10000):
    for i,prog in enumerate(order):
        if i>=budget: return False,budget
        try:
            if all(run(prog,domain,L)==a for L,a in ex): return True,i+1
        except Exception: pass
    return False,len(order)

def main():
    print("="*64); print("  ORACLE-ABSTRACTION TEST (friend's step 1, 120 seeds)"); print("="*64)
    SEQ=dom_sequences(); space=len(all_programs(SEQ))
    print(f"  unseen-domain (sequences) program space: {space}. budget 10000.\n")
    newborn=[]; shuffled=[]; oracle=[]; snb=0; sor=0; wins=0; SEEDS=120
    for seed in range(SEEDS):
        rng=random.Random(seed*13+1)
        true_shape,prog=make_task(SEQ,rng); ex=examples(prog,SEQ,rng,"seq")
        okn,tn=search(all_programs(SEQ),ex,SEQ); newborn.append(tn); snb+=okn
        wrong=random.Random(seed).choice([s for s in SHAPES if s!=true_shape])
        oks,ts=search(programs_of_shape(wrong,SEQ),ex,SEQ); shuffled.append(ts)
        oko,to=search(programs_of_shape(true_shape,SEQ),ex,SEQ); oracle.append(to); sor+=oko
        if to<tn: wins+=1
    mn=statistics.mean(newborn); mdn=statistics.median(newborn)
    ms=statistics.mean(shuffled); mo=statistics.mean(oracle); mdo=statistics.median(oracle)
    print(f"  {'condition':<24}{'mean':>8}{'median':>9}{'solve%':>9}")
    print(f"  {'newborn (full space)':<24}{mn:>8.0f}{mdn:>9.0f}{snb*100//SEEDS:>8}%")
    print(f"  {'shuffled (wrong shape)':<24}{ms:>8.0f}{'-':>9}{'-':>9}")
    print(f"  {'ORACLE (correct shape)':<24}{mo:>8.0f}{mdo:>9.0f}{sor*100//SEEDS:>8}%")
    print(f"\n  oracle beat newborn on {wins}/{SEEDS} tasks ({wins*100//SEEDS}%)")
    print("\n"+"="*64); print("  IS ABSTRACTION EXPLOITABLE?"); print("="*64)
    if mo<mn*0.6:
        print(f"  YES. Given the correct shape, NEURON solved in {mo:.0f} tries vs newborn {mn:.0f}")
        print(f"  ({(1-mo/mn)*100:.0f}% cut), winning {wins*100//SEEDS}% of tasks. shuffled (wrong shape) {ms:.0f}")
        print(f"  confirms the CORRECT shape is what helps. Abstraction IS exploitable - the")
        print(f"  missing piece is INFERENCE (recognizing the shape from examples). Worth")
        print(f"  building next. Target to beat: oracle {mo:.0f}.")
    else:
        print(f"  NO. Even with the correct shape, oracle {mo:.0f} ~= newborn {mn:.0f}. The search")
        print(f"  can't exploit abstraction - building inference fixes the wrong layer. Stop")
        print(f"  and fix representation/search first. (Exactly why the oracle test comes first.)")

if __name__=="__main__": main()
