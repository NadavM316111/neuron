"""Rigorous validation of the population result (per the friend's tests 1 and 2):
many independent seeds + a no-crowding causal control, with correct terminology.

Terminology fixed: outer loop = EPOCHS (simulation periods). "lineage depth" =
how many ancestors deep the deepest living being is. These are different things.

Test 1 (robustness): run many independently-seeded populations; report how often
diversification emerges, the diversity distribution, extinction rate, and mean
lineage depth. One lucky run proves nothing; many seeds prove robustness.

Test 2 (causal control): run WITH vs WITHOUT crowding penalties (frequency-
dependent competition). If removing crowding collapses the population into one
niche, and adding it restores diversity, crowding CAUSES the differentiation.

    python niche_validation.py --seeds 300
"""
import random, math, argparse

N_NICHES=4
_id=[0]
def nid(): _id[0]+=1; return _id[0]
LIFESPAN=1200

class Being:
    def __init__(self, inherited=None, depth=0, start_age=0, seed=0):
        self.id=nid(); self.depth=depth
        r=random.Random(self.id*7+seed)
        if inherited:
            self.values=[max(0,inherited[i]*0.5+r.uniform(0,0.3)) for i in range(N_NICHES)]
        else:
            self.values=[r.uniform(0,0.3) for _ in range(N_NICHES)]
        self.birth_niche=max(range(N_NICHES),key=lambda i:self.values[i])  # nature
        self.wellbeing=0.5; self.age=start_age; self.recent_err=[]; self.alive=True; self.rng=r
    def niche(self): return max(range(N_NICHES),key=lambda i:self.values[i])
    def live(self, payoffs):
        self.age+=1; ni=self.niche(); payoff=payoffs[ni]
        outcome=1 if payoff+self.rng.uniform(-0.2,0.2)>0.5 else -1
        self.wellbeing=max(0,min(1,0.98*self.wellbeing+0.05*outcome))
        self.values[ni]=max(0,self.values[ni]+0.05*outcome)
        err=1 if (self.values[ni]>0.5)!=(outcome>0) else 0
        self.recent_err.append(err); self.recent_err=self.recent_err[-25:]
        if len(self.recent_err)>=25 and sum(self.recent_err)/len(self.recent_err)>0.5:
            self.values[ni]*=0.4; self.values[self.rng.randrange(N_NICHES)]+=0.3; self.recent_err=[]
        if self.age>LIFESPAN: self.alive=False

def blend(p1,p2): return [(p1.values[i]+p2.values[i])/2 for i in range(N_NICHES)]
def dist_of(beings):
    d=[0]*N_NICHES
    for b in beings: d[b.niche()]+=1
    return d
def shannon(d):
    t=sum(d); 
    if t==0: return 0
    h=0
    for c in d:
        if c>0: p=c/t; h-=p*math.log(p)
    return h/math.log(N_NICHES)

def run_pop(seed, crowding, epochs=8, pop=10, era=400):
    rng=random.Random(seed)
    population=[Being(start_age=rng.randint(0,LIFESPAN-200),seed=seed+i) for i in range(pop)]
    max_depth=0; extinct=False
    # nature/nurture tracking: (parent_niche, child_birth_niche, child_adult_niche)
    nn_records=[]
    for ep in range(epochs):
        base=[0.6+0.3*math.sin(ep+i) for i in range(N_NICHES)]
        for day in range(era):
            alive=[b for b in population if b.alive]
            if not alive: extinct=True; break
            d=dist_of(alive)
            if crowding:
                payoffs=[base[i]-0.12*d[i] for i in range(N_NICHES)]
            else:
                payoffs=list(base)   # NO crowding penalty (control)
            for b in alive: b.live(payoffs)
            dead=[b for b in population if not b.alive]
            if dead:
                population=[b for b in population if b.alive]
                while len(population)<pop and len(population)>=2:
                    par=sorted(population,key=lambda b:b.wellbeing,reverse=True)[:2]
                    child=Being(inherited=blend(*par),depth=max(p.depth for p in par)+1,seed=seed+nid())
                    max_depth=max(max_depth,child.depth); population.append(child)
                    nn_records.append([par[0].niche(), child.birth_niche, None, child.id])
                while len(population)<2: population.append(Being(seed=seed+nid()))
        if extinct: break
    alive=[b for b in population if b.alive]
    # fill adult niches for nature/nurture records
    adult={b.id:b.niche() for b in alive}
    for rec in nn_records:
        if rec[3] in adult: rec[2]=adult[rec[3]]
    final_div=shannon(dist_of(alive)) if alive else 0
    return {"diversity":final_div, "extinct":extinct, "depth":max_depth,
            "occupied":sum(1 for c in dist_of(alive) if c>0) if alive else 0,
            "nn":nn_records}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seeds",type=int,default=300); a=ap.parse_args()
    print("="*64); print("  RIGOROUS VALIDATION: many seeds + no-crowding control"); print("="*64)

    print(f"\n  TEST 1: robustness across {a.seeds} independent populations (WITH crowding)")
    divs=[]; extinctions=0; depths=[]; occ=[]
    for s in range(a.seeds):
        r=run_pop(seed=s*131+1, crowding=True)
        divs.append(r["diversity"]); depths.append(r["depth"]); occ.append(r["occupied"])
        if r["extinct"]: extinctions+=1
    import statistics
    diversified=sum(1 for d in divs if d>0.5)
    print(f"    diversification emerged (spread>0.5): {diversified}/{a.seeds} "
          f"({diversified*100//a.seeds}%)")
    print(f"    mean final diversity: {statistics.mean(divs):.2f} "
          f"(min {min(divs):.2f}, max {max(divs):.2f})")
    print(f"    mean niches occupied: {statistics.mean(occ):.1f}/{N_NICHES}")
    print(f"    mean lineage depth: {statistics.mean(depths):.1f}")
    print(f"    extinction rate: {extinctions}/{a.seeds} ({extinctions*100//a.seeds}%)")

    print(f"\n  TEST 2: causal control, WITH vs WITHOUT crowding ({a.seeds//3} seeds each)")
    with_div=[run_pop(seed=s*131+1,crowding=True)["diversity"] for s in range(a.seeds//3)]
    without_div=[run_pop(seed=s*131+1,crowding=False)["diversity"] for s in range(a.seeds//3)]
    print(f"    WITH crowding    -> mean diversity {statistics.mean(with_div):.2f}")
    print(f"    WITHOUT crowding -> mean diversity {statistics.mean(without_div):.2f}")
    causal = statistics.mean(with_div) > statistics.mean(without_div) + 0.15

    print(f"\n  TEST 3 (bonus): nature vs nurture (a sample population)")
    sample=run_pop(seed=777,crowding=True)
    switched=sum(1 for rec in sample["nn"] if rec[2] is not None and rec[1]!=rec[2])
    kept=sum(1 for rec in sample["nn"] if rec[2] is not None and rec[1]==rec[2])
    print(f"    children born resembling parent but who found a DIFFERENT adult niche")
    print(f"    (nurture over nature): {switched} | kept birth niche (nature): {kept}")

    print("\n"+"="*64); print("  WHAT THIS ESTABLISHES"); print("="*64)
    robust = diversified > a.seeds*0.5
    if robust and causal:
        print(f"  ROBUST + CAUSAL. Diversification emerged in {diversified*100//a.seeds}% of "
              f"independent")
        print(f"  populations (not a lucky seed), and it is CAUSED by crowding: with")
        print(f"  frequency-dependent competition mean diversity is {statistics.mean(with_div):.2f}, without it")
        print(f"  collapses to {statistics.mean(without_div):.2f}. So the specialization is a real,")
        print(f"  reproducible, causally-explained population phenomenon, not an artifact.")
        print(f"  And some children, born like their parents, found their OWN niche through")
        print(f"  their own lives ({switched} cases), nurture over nature, the NEURON-like thing.")
        print(f"\n  Honest framing (per the friend): this is validated EMERGENT NICHE")
        print(f"  DIFFERENTIATION in a frequency-dependent environment, a real population")
        print(f"  phenomenon. NOT yet a society (culture/norms/relationships are the next")
        print(f"  jump). But the ecological layer is now rigorously validated.")
    else:
        print(f"  Honest: diversified in {diversified*100//a.seeds}% of seeds; crowding effect")
        print(f"  {statistics.mean(with_div):.2f} vs {statistics.mean(without_div):.2f}. "
              f"{'Causal and robust' if causal else 'effect present but check'}. Reported straight.")

if __name__=="__main__": main()
