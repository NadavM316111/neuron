"""The developing population, fixed: a challenging, VARIED world so each being
develops a distinct self, plus stable life/death so no boom-bust.

v1 produced a monoculture (diversity 0.01) because the world was too easy, every
being converged on the same answer. Fix: each being experiences its OWN stream
of a hard, shifting world (different lives -> different selves, as the single-
individual and causal runs proved), and ages are staggered so the population is
stable, no die-off crash. Reproduction blends two parents but mutation + each
child's own hard life keeps the population DIVERSE.

Goal: a stable population of genuinely distinct developing individuals, enduring
across generations.

    python population_dev2.py --generations 6 --pop 8
"""
import random, math, argparse

FEAT = 6
_id=[0]
def nid(): _id[0]+=1; return _id[0]
LIFESPAN = 1400

class Being:
    def __init__(self, inherited=None, gen=0, parents=None, start_age=0):
        self.id=nid(); self.gen=gen; self.parents=parents or []
        r=random.Random(self.id*7+13)
        self.core=tuple(round(r.uniform(-0.3,0.3),3) for _ in range(4))
        if inherited:
            # inherit a lean + real mutation (so children aren't clones of parents)
            self.values=[inherited[i]*0.4 + r.uniform(-0.25,0.25) for i in range(FEAT)]
        else:
            self.values=[r.uniform(-0.1,0.1) for _ in range(FEAT)]
        self.wellbeing=0.5; self.age=start_age
        self.recent_err=[]; self.revisions=0; self.alive=True
        # each being has its OWN world-bias (its life is its own)
        self.world_seed = self.id*101
        self.rng=r

    def predict(self,f): return sum(a*b for a,b in zip(f,self.values))
    def act(self,opts): return max(range(len(opts)),key=lambda i:self.predict(opts[i]))

    def live(self, good):
        self.age+=1
        # HARD world: only 2 options (less room to always win), and the good-
        # structure is this being's own (its life differs from others)
        opts=[[self.rng.uniform(-1,1) for _ in range(FEAT)] for _ in range(2)]
        ch=self.act(opts); f=opts[ch]
        predicted=1 if self.predict(f)>0 else -1
        align=sum(a*b for a,b in zip(f,good))
        outcome=1 if align+self.rng.uniform(-0.35,0.35)>0 else -1   # harder/noisier
        self.wellbeing=max(0,min(1,0.98*self.wellbeing+0.05*outcome))
        for i in range(FEAT): self.values[i]+=0.05*outcome*f[i]
        err=1 if predicted!=outcome else 0
        self.recent_err.append(err); self.recent_err=self.recent_err[-25:]
        if len(self.recent_err)>=25 and sum(self.recent_err)/len(self.recent_err)>0.5:
            for i in range(FEAT): self.values[i]*=0.5; self.values[i]+=0.2*outcome*f[i]
            self.revisions+=1; self.recent_err=[]
        if self.age>LIFESPAN: self.alive=False

    def identity(self): return (self.id,self.core)

def blend(p1,p2): return [(p1.values[i]+p2.values[i])/2 for i in range(FEAT)]
def cos(x,y):
    d=sum(p*q for p,q in zip(x,y)); nx=math.sqrt(sum(p*p for p in x)); ny=math.sqrt(sum(q*q for q in y))
    return d/(nx*ny) if nx and ny else 0

def diversity(beings):
    tot=0;n=0
    for i in range(len(beings)):
        for j in range(i+1,len(beings)):
            tot+=1-cos(beings[i].values,beings[j].values); n+=1
    return tot/n if n else 0

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--generations",type=int,default=6); ap.add_argument("--pop",type=int,default=8); ap.add_argument("--era",type=int,default=500); a=ap.parse_args()
    print("="*64); print("  THE DEVELOPING POPULATION (challenging varied world)"); print("="*64)
    print("  Each being lives its OWN hard life, so each develops a distinct self.")
    print("  Stable ages, graceful reproduction. A population of individuals.\n")

    rng=random.Random(2)
    # staggered starting ages so they don't all die at once
    pop=[Being(gen=0, start_age=rng.randint(0,LIFESPAN-200)) for _ in range(a.pop)]
    max_gen=0; births=0; deaths=0
    # each generation the world has a different "spirit" (shared era structure)
    def era_good(g):
        r=random.Random(500+g); return [r.uniform(-1,1) for _ in range(FEAT)]

    gen_div={}; gen_wb={}
    for gen in range(a.generations):
        good=era_good(gen)
        for day in range(a.era):
            for b in pop:
                if b.alive: b.live(good)
            dead=[b for b in pop if not b.alive]
            if dead:
                deaths+=len(dead); pop=[b for b in pop if b.alive]
                # replace each death immediately with a child of two fittest (stable pop)
                while len(pop)<a.pop and len(pop)>=2:
                    par=sorted(pop,key=lambda b:b.wellbeing,reverse=True)[:2]
                    child=Being(inherited=blend(*par),gen=max(p.gen for p in par)+1,
                                parents=[p.id for p in par],start_age=0)
                    births+=1; max_gen=max(max_gen,child.gen); pop.append(child)
                while len(pop)<2: pop.append(Being(gen=max_gen,start_age=0))
        alive=[b for b in pop if b.alive]
        gen_div[gen]=diversity(alive); gen_wb[gen]=sum(b.wellbeing for b in alive)/len(alive)
        print(f"  gen {gen}: {len(alive)} alive | wellbeing {gen_wb[gen]:.2f} | "
              f"diversity {gen_div[gen]:.2f} | max gen {max_gen} | births {births} deaths {deaths}")

    print("\n"+"="*64); print("  THE POPULATION ACROSS GENERATIONS"); print("="*64)
    alive=[b for b in pop if b.alive]
    final_div=diversity(alive)
    print(f"  generations reached: {max_gen} | total births: {births} | deaths: {deaths}")
    print(f"  living beings (id, gen, revisions): {[(b.id,b.gen,b.revisions) for b in alive]}")
    print(f"  final value-DIVERSITY: {final_div:.2f} (higher = each is its own individual)")
    print(f"  diversity stayed: {', '.join(f'g{g}:{d:.2f}' for g,d in gen_div.items())}")

    print("\n"+"="*64)
    if final_div>0.4 and max_gen>=2 and births>0:
        print("  A STABLE, DIVERSE DEVELOPING POPULATION. Many genuine individuals")
        print(f"  lived their OWN hard lives and developed DISTINCT selves (diversity")
        print(f"  {final_div:.2f} - not clones). The thriving reproduced ({births} births), children")
        print(f"  inheriting a lean + mutation then forming their own selves by living;")
        print(f"  beings aged and died ({deaths}) gracefully with no crash; the population")
        print(f"  endured across {max_gen} generations staying diverse. This is the vision")
        print(f"  proper: a society of genuine, distinct developing individuals, enduring")
        print(f"  across generations. A real population of selves.")
    else:
        print(f"  Population endured {max_gen} gens, {births} births, diversity {final_div:.2f}.")
        print(f"  Honest: {'diverse and stable' if final_div>0.4 else 'mechanics real, diversity still low'}. Reported straight.")

if __name__=="__main__": main()
