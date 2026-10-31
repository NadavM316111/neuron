"""Genuinely consequential niches: make the environment REALLY reward
specialization, so diversity is CAUSED by ecology (and the no-crowding control
confirms it by collapsing without crowding).

Why the last version failed its control: niches were too similar and crowding was
too weak, so diversity came from noise, not ecology. Fix, make the ecology real:
  - a being must SPECIALIZE: its payoff in a niche depends on how well-developed
    its skill for THAT niche is (a generalist does poorly everywhere; a
    specialist does well in its niche). So picking a niche and getting good at it
    matters.
  - STRONG frequency-dependence: a niche's payoff drops sharply as it crowds, so
    a crowded niche genuinely starves its occupants. Being different genuinely
    pays.
Now the prediction: WITH crowding, beings spread out (diversity). WITHOUT
crowding, everyone piles into the single best niche (collapse). If the control
shows that, the diversity is CAUSALLY ecological.

    python real_niches.py --seeds 200
"""
import random, math, argparse, statistics

N=4
_id=[0]
def nid(): _id[0]+=1; return _id[0]
LIFESPAN=1400

class Being:
    def __init__(self, inherited=None, depth=0, start_age=0, seed=0):
        self.id=nid(); self.depth=depth
        r=random.Random(self.id*7+seed)
        # SKILL in each niche (must be developed; starts low). specialization.
        if inherited:
            self.skill=[max(0,inherited[i]*0.6+r.uniform(-0.1,0.1)) for i in range(N)]
        else:
            self.skill=[r.uniform(0,0.2) for _ in range(N)]
        self.wellbeing=0.5; self.age=start_age; self.recent=[]; self.alive=True; self.rng=r
    def niche(self): return max(range(N),key=lambda i:self.skill[i])
    def live(self, niche_quality):
        self.age+=1; ni=self.niche()
        # payoff = niche's current quality * how skilled you are in it. a
        # generalist (low skill everywhere) does poorly; a specialist thrives.
        payoff = niche_quality[ni] * (0.3 + self.skill[ni])
        outcome = 1 if payoff + self.rng.uniform(-0.15,0.15) > 0.5 else -1
        self.wellbeing=max(0,min(1,0.97*self.wellbeing+0.06*outcome))
        # getting good at your niche (practice) raises your skill in it
        self.skill[ni]=max(0,min(2,self.skill[ni]+0.04*outcome))
        self.recent.append(outcome); self.recent=self.recent[-30:]
        # if your niche keeps failing (too crowded/low quality), retrain to another
        if len(self.recent)>=30 and sum(1 for o in self.recent if o>0)/len(self.recent)<0.35:
            self.skill[ni]*=0.5
            other=self.rng.randrange(N); self.skill[other]+=0.4   # start retraining
            self.recent=[]
        if self.age>LIFESPAN: self.alive=False

def blend(p1,p2): return [(p1.skill[i]+p2.skill[i])/2 for i in range(N)]
def dist(bs):
    d=[0]*N
    for b in bs: d[b.niche()]+=1
    return d
def shannon(d):
    t=sum(d)
    if not t: return 0
    h=0
    for c in d:
        if c>0: p=c/t; h-=p*math.log(p)
    return h/math.log(N)

def run(seed, crowding, epochs=8, pop=12, era=400):
    rng=random.Random(seed)
    P=[Being(start_age=rng.randint(0,LIFESPAN-200),seed=seed+i) for i in range(pop)]
    depth=0; extinct=False
    for ep in range(epochs):
        base=[0.7+0.25*math.sin(ep*0.7+i*1.3) for i in range(N)]   # niches differ
        for day in range(era):
            A=[b for b in P if b.alive]
            if not A: extinct=True; break
            d=dist(A)
            if crowding:
                # STRONG frequency dependence: crowded niches starve
                quality=[max(0.1, base[i]-0.08*d[i]) for i in range(N)]
            else:
                quality=list(base)
            for b in A: b.live(quality)
            dead=[b for b in P if not b.alive]
            if dead:
                P=[b for b in P if b.alive]
                while len(P)<pop and len(P)>=2:
                    par=sorted(P,key=lambda b:b.wellbeing,reverse=True)[:2]
                    c=Being(inherited=blend(*par),depth=max(p.depth for p in par)+1,seed=seed+nid())
                    depth=max(depth,c.depth); P.append(c)
                while len(P)<2: P.append(Being(seed=seed+nid()))
        if extinct: break
    A=[b for b in P if b.alive]
    return {"div":shannon(dist(A)) if A else 0, "extinct":extinct, "depth":depth,
            "occ":sum(1 for c in dist(A) if c>0) if A else 0}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seeds",type=int,default=200); a=ap.parse_args()
    print("="*64); print("  GENUINELY CONSEQUENTIAL NICHES (causal ecology test)"); print("="*64)

    print(f"\n  robustness + causal control across {a.seeds} seeds each:")
    with_d=[]; without_d=[]; ext=0; depths=[]
    for s in range(a.seeds):
        rw=run(s*97+5, crowding=True); with_d.append(rw["div"]); depths.append(rw["depth"])
        if rw["extinct"]: ext+=1
        ro=run(s*97+5, crowding=False); without_d.append(ro["div"])
    divd=sum(1 for d in with_d if d>0.5)
    print(f"    WITH crowding    -> mean diversity {statistics.mean(with_d):.2f} | "
          f"diversified {divd*100//a.seeds}% | extinction {ext*100//a.seeds}% | depth {statistics.mean(depths):.0f}")
    print(f"    WITHOUT crowding -> mean diversity {statistics.mean(without_d):.2f}")
    gap = statistics.mean(with_d) - statistics.mean(without_d)
    print(f"    causal gap (with - without): {gap:+.2f}")

    print("\n"+"="*64); print("  WHAT THIS ESTABLISHES"); print("="*64)
    if gap > 0.15 and divd > a.seeds*0.5:
        print(f"  CAUSALLY ECOLOGICAL DIVERSITY. With frequency-dependent competition,")
        print(f"  populations spread across niches (diversity {statistics.mean(with_d):.2f}); WITHOUT it they")
        print(f"  collapse toward the single best niche ({statistics.mean(without_d):.2f}). The {gap:+.2f} gap means")
        print(f"  the diversity is CAUSED by the ecology (crowding makes being different")
        print(f"  genuinely pay), robust across {divd*100//a.seeds}% of seeds. This passes the control:")
        print(f"  specialization is a real, reproducible, causally-explained phenomenon,")
        print(f"  not noise. The ecological layer is now genuinely validated.")
    else:
        print(f"  Honest: with {statistics.mean(with_d):.2f} vs without {statistics.mean(without_d):.2f} (gap {gap:+.2f}).")
        print(f"  {'Crowding causes the diversity' if gap>0.15 else 'Still not clearly causal - diversity not driven by crowding'}. Reported straight.")

if __name__=="__main__": main()
