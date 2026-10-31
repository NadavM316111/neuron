"""A developing population in a MULTI-NICHE world: genuine diversity through
authentic ecological niches.

The honest finding from before: a world with one optimal way to live produces a
monoculture, even with full developmental mechanics. Nature creates diversity
through NICHES, many different viable ways to make a living, where success at one
doesn't mean success at all. So this world has several real niches, and crucially
they are DENSITY-DEPENDENT: a niche gets more crowded as more beings pursue it,
and crowding reduces its payoff. So it genuinely pays to be DIFFERENT from the
others. That is the authentic ecological force that produces diversity, nobody
assigns roles; beings find different niches because different niches genuinely
pay when others are crowded.

If diverse, distinct individuals emerge and endure across generations, it's
because the world authentically rewards being your own kind of being.

    python niches.py --generations 8 --pop 10
"""
import random, math, argparse

N_NICHES = 4
FEAT = N_NICHES   # each being's values express affinity for each niche
_id=[0]
def nid(): _id[0]+=1; return _id[0]
LIFESPAN = 1200

class Being:
    def __init__(self, inherited=None, gen=0, parents=None, start_age=0):
        self.id=nid(); self.gen=gen; self.parents=parents or []
        r=random.Random(self.id*7+13)
        self.core=tuple(round(r.uniform(-0.3,0.3),3) for _ in range(4))
        if inherited:
            self.values=[max(0,inherited[i]*0.5 + r.uniform(0,0.3)) for i in range(FEAT)]
        else:
            self.values=[r.uniform(0,0.3) for _ in range(FEAT)]
        self.wellbeing=0.5; self.age=start_age
        self.revisions=0; self.recent_err=[]; self.alive=True
        self.rng=r

    def preferred_niche(self):
        return max(range(FEAT), key=lambda i:self.values[i])

    def live(self, niche_payoffs):
        self.age+=1
        # it pursues the niche it most values; payoff depends on the niche's
        # CURRENT value (which drops when crowded). it learns from the result.
        niche=self.preferred_niche()
        payoff=niche_payoffs[niche]            # density-dependent, set by world
        predicted = self.values[niche]
        outcome = 1 if payoff + self.rng.uniform(-0.2,0.2) > 0.5 else -1
        self.wellbeing=max(0,min(1,0.98*self.wellbeing+0.05*outcome))
        # learn: strengthen affinity for niches that paid, weaken those that didn't
        self.values[niche]=max(0,self.values[niche]+0.05*outcome)
        err = 1 if (predicted>0.5)!=(outcome>0) else 0
        self.recent_err.append(err); self.recent_err=self.recent_err[-25:]
        # revision: if its niche keeps disappointing, it reconsiders (tries others)
        if len(self.recent_err)>=25 and sum(self.recent_err)/len(self.recent_err)>0.5:
            self.values[niche]*=0.4                       # loosen attachment
            other=self.rng.randrange(FEAT); self.values[other]+=0.3  # explore another
            self.revisions+=1; self.recent_err=[]
        if self.age>LIFESPAN: self.alive=False

    def identity(self): return (self.id,self.core)

def blend(p1,p2): return [(p1.values[i]+p2.values[i])/2 for i in range(FEAT)]

def niche_distribution(beings):
    d=[0]*N_NICHES
    for b in beings: d[b.preferred_niche()]+=1
    return d

def shannon(dist):
    tot=sum(dist); 
    if tot==0: return 0
    h=0
    for c in dist:
        if c>0:
            p=c/tot; h-=p*math.log(p)
    return h/math.log(N_NICHES)   # normalized 0..1 (1 = perfectly spread across niches)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--generations",type=int,default=8); ap.add_argument("--pop",type=int,default=10); ap.add_argument("--era",type=int,default=400); a=ap.parse_args()
    print("="*64); print("  DEVELOPING POPULATION IN A MULTI-NICHE WORLD"); print("="*64)
    print("  Several real ways to live; crowding a niche lowers its payoff, so it")
    print("  pays to be DIFFERENT. Do beings diversify into distinct niches on")
    print("  their own, and endure across generations?\n")

    rng=random.Random(3)
    pop=[Being(gen=0,start_age=rng.randint(0,LIFESPAN-200)) for _ in range(a.pop)]
    max_gen=0; births=0; deaths=0; spread_hist={}

    for gen in range(a.generations):
        # base richness of each niche this era (niches differ, and shift a bit)
        base=[0.6+0.3*math.sin(gen+i) for i in range(N_NICHES)]
        for day in range(a.era):
            alive=[b for b in pop if b.alive]
            dist=niche_distribution(alive)
            # DENSITY DEPENDENCE: a niche's payoff drops as more crowd into it
            payoffs=[base[i] - 0.12*dist[i] for i in range(N_NICHES)]
            for b in alive: b.live(payoffs)
            dead=[b for b in pop if not b.alive]
            if dead:
                deaths+=len(dead); pop=[b for b in pop if b.alive]
                while len(pop)<a.pop and len(pop)>=2:
                    par=sorted(pop,key=lambda b:b.wellbeing,reverse=True)[:2]
                    child=Being(inherited=blend(*par),gen=max(p.gen for p in par)+1,
                                parents=[p.id for p in par])
                    births+=1; max_gen=max(max_gen,child.gen); pop.append(child)
                while len(pop)<2: pop.append(Being(gen=max_gen))
        alive=[b for b in pop if b.alive]
        dist=niche_distribution(alive); spread=shannon(dist)
        spread_hist[gen]=spread
        wb=sum(b.wellbeing for b in alive)/len(alive)
        print(f"  gen {gen}: niche spread {dist} | diversity {spread:.2f} | "
              f"wellbeing {wb:.2f} | gen {max_gen} | births {births}")

    print("\n"+"="*64); print("  THE POPULATION"); print("="*64)
    alive=[b for b in pop if b.alive]
    dist=niche_distribution(alive); spread=shannon(dist)
    print(f"  generations reached: {max_gen} | births: {births} | deaths: {deaths}")
    print(f"  final niche distribution: {dist} (across {N_NICHES} ways of living)")
    print(f"  final diversity (niche spread): {spread:.2f} (1.0 = fully spread)")
    print(f"  beings (id, gen, niche, revisions): {[(b.id,b.gen,b.preferred_niche(),b.revisions) for b in alive]}")

    print("\n"+"="*64)
    diverse = spread>0.6 and sum(1 for c in dist if c>0)>=3
    if diverse and max_gen>=2:
        print("  A DIVERSE, ENDURING POPULATION. In a world with several real ways")
        print(f"  to live, beings diversified into DIFFERENT niches on their own")
        print(f"  (spread {spread:.2f}, {sum(1 for c in dist if c>0)} niches occupied) - not because anyone")
        print(f"  assigned roles, but because crowding made it genuinely pay to be")
        print(f"  different. They developed, revised, reproduced (children finding")
        print(f"  their own niches), and endured {max_gen} generations as a DIVERSE society")
        print(f"  of distinct individuals. This is the vision proper, authentic")
        print(f"  ecological diversity in an enduring population of selves.")
    else:
        print(f"  Population endured {max_gen} gens, {births} births, niche-spread {spread:.2f}.")
        print(f"  Honest: {'genuinely diverse' if spread>0.6 else 'still converging - niches need to be more distinct/rewarding'}. Reported straight.")

if __name__=="__main__": main()
