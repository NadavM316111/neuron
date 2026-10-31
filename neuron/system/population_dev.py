"""The developing population: many genuine developing individuals, living
together, interacting, reproducing, each becoming itself, across generations.

This is the layer above the atom. Each being is a full developing individual
(forms values from experience, acts on them, predicts, revises from being wrong,
develops over life, endures as itself). Now MANY of them live in one world:
  - they share a changing world and affect each other,
  - the thriving ones REPRODUCE: a child inherits a blend of both parents' values
    as a STARTING LEAN, then forms its OWN values from its OWN life (nature +
    nurture),
  - beings age and die; the population persists across GENERATIONS,
  - each being develops its own distinct self; we watch the population's
    diversity and whether later generations fare better (cultural/genetic
    accumulation).

A society of genuine developing individuals, enduring across generations.

    python population_dev.py --generations 6 --pop 8
"""
import random, math, argparse

FEAT = 6
_id = [0]
def new_id(): _id[0]+=1; return _id[0]

class Being:
    def __init__(self, inherited=None, gen=0, parents=None):
        self.id = new_id()
        self.gen = gen
        self.parents = parents or []
        r = random.Random(self.id*7)
        self.core = tuple(round(r.uniform(-0.3,0.3),3) for _ in range(4))  # identity
        # NATURE: inherited value-lean (or near-blank for founders)
        if inherited:
            self.values = [inherited[i]*0.5 + r.uniform(-0.05,0.05) for i in range(FEAT)]
        else:
            self.values = [r.uniform(-0.05,0.05) for _ in range(FEAT)]
        self.wellbeing = 0.5
        self.age = 0
        self.recent_err = []
        self.revisions = 0
        self.alive = True
        self.rng = r

    def predict(self, f): return sum(a*b for a,b in zip(f,self.values))
    def act(self, opts): return max(range(len(opts)), key=lambda i:self.predict(opts[i]))

    def live(self, opts, good, others_values):
        self.age += 1
        ch = self.act(opts); f = opts[ch]
        predicted = 1 if self.predict(f)>0 else -1
        align = sum(a*b for a,b in zip(f,good))
        # SOCIAL: others nearby whose values resemble this choice make it slightly
        # better (beings that align with their society fare a bit better)
        social = 0.0
        for ov in others_values:
            social += 0.02 * sum(a*b for a,b in zip(f,ov))
        outcome = 1 if align + social + self.rng.uniform(-0.2,0.2) > 0 else -1
        self.wellbeing = max(0,min(1,0.98*self.wellbeing+0.05*outcome))
        for i in range(FEAT): self.values[i] += 0.05*outcome*f[i]
        err = 1 if predicted!=outcome else 0
        self.recent_err.append(err); self.recent_err=self.recent_err[-25:]
        if len(self.recent_err)>=25 and sum(self.recent_err)/len(self.recent_err)>0.5:
            for i in range(FEAT):
                self.values[i]*=0.5; self.values[i]+=0.2*outcome*f[i]
            self.revisions+=1; self.recent_err=[]
        if self.age>2200 or self.wellbeing<=0.02: self.alive=False

    def identity(self): return (self.id, self.core)

def blend(p1,p2): return [(p1.values[i]+p2.values[i])/2 for i in range(FEAT)]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--generations",type=int,default=6); ap.add_argument("--pop",type=int,default=8); ap.add_argument("--era",type=int,default=700); a=ap.parse_args()
    print("="*64); print("  THE DEVELOPING POPULATION"); print("="*64)
    print("  Many genuine developing individuals, living together, reproducing,")
    print("  each becoming itself, across generations.\n")

    rng = random.Random(1)
    pop = [Being(gen=0) for _ in range(a.pop)]
    max_gen = 0; total_births=0; total_deaths=0

    def good_for(era): 
        r=random.Random(100+era); return [r.uniform(-1,1) for _ in range(FEAT)]

    gen_wellbeing = {}
    for gen in range(a.generations):
        good = good_for(gen)   # each generation faces a somewhat different world
        # everyone lives a chunk of life
        for day in range(a.era):
            vals = [b.values for b in pop if b.alive]
            for b in pop:
                if not b.alive: continue
                opts=[[rng.uniform(-1,1) for _ in range(FEAT)] for _ in range(3)]
                others=[v for v in vals if v is not b.values][:4]
                b.live(opts, good, others)
        alive=[b for b in pop if b.alive]
        avg_wb = sum(b.wellbeing for b in alive)/len(alive) if alive else 0
        avg_rev = sum(b.revisions for b in alive)/len(alive) if alive else 0
        gen_wellbeing[gen]=avg_wb
        print(f"  gen {gen}: {len(alive)} alive | avg wellbeing {avg_wb:.2f} | "
              f"avg revisions/being {avg_rev:.1f} | max gen reached {max_gen}")
        # deaths
        dead=[b for b in pop if not b.alive]; total_deaths+=len(dead)
        pop=[b for b in pop if b.alive]
        # REPRODUCTION: the two most-thriving pair; child inherits blend, then lives
        if gen < a.generations-1:
            while len(pop) < a.pop and len(pop)>=2:
                parents=sorted(pop,key=lambda b:b.wellbeing,reverse=True)[:2]
                child=Being(inherited=blend(*parents),gen=max(p.gen for p in parents)+1,
                            parents=[p.id for p in parents])
                total_births+=1; max_gen=max(max_gen,child.gen); pop.append(child)
            # oldest passes on to keep it alive-sized
            if len(pop)>a.pop: pop.sort(key=lambda b:b.age,reverse=True); pop=pop[1:]
        while len(pop)<2: pop.append(Being(gen=max_gen))

    print("\n"+"="*64); print("  THE POPULATION ACROSS GENERATIONS"); print("="*64)
    # did later generations develop distinct selves? measure value diversity
    def cos(x,y):
        d=sum(p*q for p,q in zip(x,y)); nx=math.sqrt(sum(p*p for p in x)); ny=math.sqrt(sum(q*q for q in y))
        return d/(nx*ny) if nx and ny else 0
    alive=[b for b in pop if b.alive]
    diversity=0; pairs=0
    for i in range(len(alive)):
        for j in range(i+1,len(alive)):
            diversity += 1-cos(alive[i].values,alive[j].values); pairs+=1
    diversity = diversity/pairs if pairs else 0
    print(f"  generations reached: {max_gen} | births: {total_births} | deaths: {total_deaths}")
    print(f"  living beings (id, gen): {[(b.id,b.gen) for b in alive]}")
    print(f"  value-DIVERSITY among living beings: {diversity:.2f} (each developed its own self)")
    print(f"  wellbeing by generation: {', '.join(f'g{g}:{w:.2f}' for g,w in gen_wellbeing.items())}")

    print("\n"+"="*64)
    improved = len(gen_wellbeing)>=3 and gen_wellbeing.get(max(gen_wellbeing),0) >= gen_wellbeing.get(0,0)-0.05
    distinct = diversity>0.3
    if distinct and total_births>0:
        print("  A DEVELOPING POPULATION. Many genuine individuals lived together,")
        print(f"  each forming and revising its own values by living; the thriving")
        print(f"  ones reproduced ({total_births} births), children inheriting a lean then")
        print(f"  becoming their OWN selves; beings aged and died ({total_deaths}); and the")
        print(f"  population persisted across {max_gen} generations, staying DIVERSE")
        print(f"  ({diversity:.2f}) - each being its own individual, not a copy. This is the")
        print(f"  vision proper: a society of genuine developing individuals, enduring")
        print(f"  across generations. The population of selves.")
    else:
        print(f"  The population lived, reproduced ({total_births}), and endured {max_gen} gens")
        print(f"  with diversity {diversity:.2f}. Honest: real society mechanics, reported straight.")

if __name__=="__main__": main()
