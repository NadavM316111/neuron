"""THE SOCIETY: every validated mechanism, living together as one continuous system.

This consolidates the whole arc into one artifact. A population of genuine
developing individuals who:
  - form values/behavior from their own experience (development)
  - live in a multi-niche world and differentiate (ecology)
  - learn behavior from each other, not just genes (culture)
  - infer their group's convention from observation, no stored rule (carried norm)
  - enforce it (approve conformers, sanction deviators) and feel CONFIDENCE
  - reproduce; children learn the culture imperfectly (transmission)
  - age and die; the society persists across generations
One living society, not separate demos. It runs continuously and reports its own
state: who's alive, what each group's convention is, how strong it is, diversity,
lineage depth.

    python the_society.py --years 8 --groups 2
"""
import random, math, argparse, statistics

N_BEHAV=4; N_NICHE=4
_id=[0]
def nid(): _id[0]+=1; return _id[0]
LIFESPAN=1400
def argmax_random(vals,rng):
    m=max(vals); best=[i for i,v in enumerate(vals) if abs(v-m)<1e-9]; return rng.choice(best)

class Being:
    def __init__(self, group, inherited_niche=None, seed=0):
        self.id=nid(); self.group=group; self.depth=0
        r=random.Random(self.id*7+seed); self.rng=r
        self.core=tuple(round(r.uniform(-0.3,0.3),3) for _ in range(3))   # identity
        # niche skill (ecology): inherited lean + own development
        if inherited_niche is not None:
            self.skill=[max(0,(0.4 if i==inherited_niche else 0)+r.uniform(0,0.2)) for i in range(N_NICHE)]
        else:
            self.skill=[r.uniform(0,0.2) for _ in range(N_NICHE)]
        # convention behavior (culture/norm): learned from the group, starts neutral
        self.behavior=[0.5]*N_BEHAV
        self.approval_mem=[0.0]*N_BEHAV; self.obs=0
        self.wellbeing=0.5; self.age=0; self.alive=True; self.reputation=0.5
        self.recent=[]; self.revisions=0
    def niche(self): return argmax_random(self.skill,self.rng)
    def does(self): return argmax_random(self.behavior,self.rng)
    def inferred_norm(self):
        if self.obs==0: return self.does()
        return argmax_random(self.approval_mem,self.rng)
    def confidence(self):
        pos=[max(0,v) for v in self.approval_mem]; t=sum(pos)
        return max(pos)/t if t>0 else 0.0
    def react(self, beh): return beh==self.inferred_norm()
    def observe(self, beh, approved):
        self.obs+=1; self.approval_mem[beh]+= (1.0 if approved else -1.0)
    def identity(self): return (self.id, self.core)

def step_group(members, niche_base, rng):
    A=[b for b in members if b.alive]
    if len(A)<3: return
    # ECOLOGY: niche payoffs, density-dependent (crowding lowers quality)
    nd=[0]*N_NICHE
    for b in A: nd[b.niche()]+=1
    quality=[max(0.1, niche_base[i]-0.08*nd[i]) for i in range(N_NICHE)]
    # the group's current CONVENTION (most common behavior) - emergent, not stored
    bd=[0]*N_BEHAV
    for b in A: bd[b.does()]+=1
    for actor in A:
        actor.age+=1
        # --- ecology: work your niche ---
        ni=actor.niche()
        eco_out = 1 if quality[ni]*(0.3+actor.skill[ni]) + actor.rng.uniform(-0.15,0.15) > 0.5 else -1
        actor.skill[ni]=max(0,min(2,actor.skill[ni]+0.04*eco_out))
        actor.recent.append(eco_out); actor.recent=actor.recent[-30:]
        if len(actor.recent)>=30 and sum(1 for o in actor.recent if o>0)/len(actor.recent)<0.35:
            actor.skill[ni]*=0.5; actor.skill[actor.rng.randrange(N_NICHE)]+=0.4
            actor.revisions+=1; actor.recent=[]
        # --- culture/norm: act, be judged by a panel inferring their own norm ---
        beh=actor.does()
        observers=[o for o in A if o is not actor]; actor.rng.shuffle(observers)
        panel=observers[:min(5,len(observers))]
        approvals=sum(1 for o in panel if o.react(beh))
        approved = approvals>=len(panel)/2 if panel else True
        for o in panel: o.observe(beh, approved)
        # learn: conform toward what's approved
        if approved:
            actor.behavior[beh]=min(2,actor.behavior[beh]+0.05); actor.reputation=min(1,actor.reputation+0.04)
        else:
            actor.behavior[beh]=max(0,actor.behavior[beh]-0.04); actor.reputation=max(0,actor.reputation-0.08)
            actor.behavior[actor.inferred_norm()]+=0.05
        soc = 0.5 if approved else -0.4
        actor.wellbeing=max(0,min(1,0.96*actor.wellbeing+0.05*eco_out+0.04*soc))
        if actor.age>LIFESPAN: actor.alive=False

def convention(members):
    A=[b for b in members if b.alive]; bd=[0]*N_BEHAV
    for b in A: bd[b.does()]+=1
    if not sum(bd): return -1,0
    c=argmax_random(bd, random.Random(1)); return c, bd[c]/sum(bd)

def diversity(members):
    A=[b for b in members if b.alive]; nd=[0]*N_NICHE
    for b in A: nd[b.niche()]+=1
    t=sum(nd); 
    if not t: return 0
    h=0
    for c in nd:
        if c>0: p=c/t; h-=p*math.log(p)
    return h/math.log(N_NICHE)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--years",type=int,default=8); ap.add_argument("--groups",type=int,default=2); ap.add_argument("--pop",type=int,default=12); a=ap.parse_args()
    print("="*64); print("  THE SOCIETY: every mechanism, living together, continuously"); print("="*64)
    print(f"  {a.groups} groups, {a.pop} beings each. developing individuals in a")
    print(f"  multi-niche world, forming culture + norms, reproducing across years.\n")

    rng=random.Random(7)
    groups={g:[Being(g,seed=g*1000+i) for i in range(a.pop)] for g in range(a.groups)}
    for g in groups:
        for b in groups[g]: b.age=rng.randint(0,LIFESPAN-200)
    births={g:0 for g in groups}; deaths={g:0 for g in groups}; maxdepth={g:0 for g in groups}

    for year in range(a.years):
        niche_base=[0.7+0.2*math.sin(year*0.6+i) for i in range(N_NICHE)]
        for day in range(350):
            for g in groups:
                step_group(groups[g], niche_base, rng)
                dead=[b for b in groups[g] if not b.alive]
                if dead:
                    deaths[g]+=len(dead); groups[g]=[b for b in groups[g] if b.alive]
                    while len(groups[g])<a.pop and len(groups[g])>=2:
                        par=sorted(groups[g],key=lambda b:b.wellbeing,reverse=True)[:2]
                        child=Being(g, inherited_niche=par[0].niche(), seed=g*9000+nid())
                        child.depth=max(p.depth for p in par)+1; maxdepth[g]=max(maxdepth[g],child.depth)
                        births[g]+=1; groups[g].append(child)
                    while len(groups[g])<2: groups[g].append(Being(g,seed=g*9000+nid()))
        # yearly report of the living society
        print(f"  --- year {year} ---")
        for g in groups:
            A=[b for b in groups[g] if b.alive]
            conv,strength=convention(groups[g])
            avg_conf=statistics.mean([b.confidence() for b in A]) if A else 0
            print(f"    group {g}: {len(A)} alive | convention #{conv} ({strength*100:.0f}% follow, "
                  f"avg confidence {avg_conf:.2f}) | niche-diversity {diversity(groups[g]):.2f} | "
                  f"lineage depth {maxdepth[g]} | births {births[g]}")

    print("\n"+"="*64); print("  THE SOCIETY NOW"); print("="*64)
    convs=[]
    for g in groups:
        A=[b for b in groups[g] if b.alive]; conv,strength=convention(groups[g]); convs.append(conv)
        print(f"  group {g}: {len(A)} beings, convention #{conv} held by {strength*100:.0f}%, "
              f"lineage depth {maxdepth[g]}, {births[g]} born / {deaths[g]} died over {a.years} years")
    distinct = len(set(c for c in convs if c>=0))>1 if a.groups>1 else True
    print(f"\n  A continuously-living society: developing individuals formed their own")
    print(f"  niches (ecology), their groups' own conventions (culture/norms, "
          f"{'DIFFERENT across groups' if distinct else 'converged'}),")
    print(f"  enforced them, reproduced, and endured across generations, every")
    print(f"  validated mechanism running together as one living system. Not separate")
    print(f"  demos, a society.")

if __name__=="__main__": main()
