"""Society audit (the friend's test): run the integrated society across many seeds
+ reintroduce validated noisy social learning, then measure everything to
characterize the integration behavior (and expose the convention-#3 bias).

Fixes vs the clean run:
  - random tie-breaking everywhere (no structural bias toward a convention)
  - reintroduced noise: imperfect observation, ambiguous approval, rule-breakers,
    memory error (the dynamics already validated in culture_noisy)
Then across 100-200 seeds, measure: convention distribution, how often two groups
differ, adherence distribution, confidence distribution, niche diversity, lineage
depth, extinction, revisions. The data tells us what's real.

    python society_audit.py --seeds 150
"""
import random, math, argparse, statistics

N_BEHAV=4; N_NICHE=4
_id=[0]
def nid(): _id[0]+=1; return _id[0]
LIFESPAN=1400
# NOISE (validated levels from culture_noisy, "moderate")
MISS=0.2; AMBIG=0.2; BREAK=0.15; MEM=0.1

def amr(vals,rng):
    m=max(vals); best=[i for i,v in enumerate(vals) if abs(v-m)<1e-9]; return rng.choice(best)

class Being:
    def __init__(self, group, inherited_niche=None, seed=0):
        self.id=nid(); self.group=group; self.depth=0
        r=random.Random(self.id*7+seed); self.rng=r
        self.core=tuple(round(r.uniform(-0.3,0.3),3) for _ in range(3))
        if inherited_niche is not None:
            self.skill=[max(0,(0.4 if i==inherited_niche else 0)+r.uniform(0,0.2)) for i in range(N_NICHE)]
        else:
            self.skill=[r.uniform(0,0.2) for _ in range(N_NICHE)]
        self.behavior=[0.5]*N_BEHAV
        self.approval_mem=[0.0]*N_BEHAV; self.obs=0
        self.wellbeing=0.5; self.age=0; self.alive=True; self.reputation=0.5
        self.recent=[]; self.revisions=0
    def niche(self): return amr(self.skill,self.rng)
    def does(self): return amr(self.behavior,self.rng)
    def inferred_norm(self):
        if self.obs==0: return self.does()
        return amr(self.approval_mem,self.rng)
    def confidence(self):
        pos=[max(0,v) for v in self.approval_mem]; t=sum(pos); return max(pos)/t if t>0 else 0.0
    def react(self, beh): return beh==self.inferred_norm()
    def observe(self, beh, approved):
        if self.rng.random()<MEM: beh=self.rng.randrange(N_BEHAV)   # memory error
        self.obs+=1; self.approval_mem[beh]+=(1.0 if approved else -1.0)

def step(members, niche_base, rng):
    A=[b for b in members if b.alive]
    if len(A)<3: return
    nd=[0]*N_NICHE
    for b in A: nd[b.niche()]+=1
    quality=[max(0.1, niche_base[i]-0.08*nd[i]) for i in range(N_NICHE)]
    for actor in A:
        actor.age+=1
        ni=actor.niche()
        eco=1 if quality[ni]*(0.3+actor.skill[ni])+actor.rng.uniform(-0.15,0.15)>0.5 else -1
        actor.skill[ni]=max(0,min(2,actor.skill[ni]+0.04*eco))
        actor.recent.append(eco); actor.recent=actor.recent[-30:]
        if len(actor.recent)>=30 and sum(1 for o in actor.recent if o>0)/len(actor.recent)<0.35:
            actor.skill[ni]*=0.5; actor.skill[actor.rng.randrange(N_NICHE)]+=0.4
            actor.revisions+=1; actor.recent=[]
        beh=actor.does()
        # RULE-BREAKER noise: sometimes act off-norm even if you know it
        if actor.rng.random()<BREAK: beh=actor.rng.randrange(N_BEHAV)
        obs=[o for o in A if o is not actor]; actor.rng.shuffle(obs); panel=obs[:min(5,len(obs))]
        approvals=sum(1 for o in panel if o.react(beh))
        true_app = approvals>=len(panel)/2 if panel else True
        for o in panel:
            if o.rng.random()<MISS: continue                         # missed observation
            app = (o.rng.random()<0.5) if o.rng.random()<AMBIG else true_app  # ambiguous
            o.observe(beh, app)
        if true_app:
            actor.behavior[beh]=min(2,actor.behavior[beh]+0.05); actor.reputation=min(1,actor.reputation+0.04)
        else:
            actor.behavior[beh]=max(0,actor.behavior[beh]-0.04); actor.reputation=max(0,actor.reputation-0.08)
            actor.behavior[actor.inferred_norm()]+=0.05
        actor.wellbeing=max(0,min(1,0.96*actor.wellbeing+0.05*eco+0.04*(0.5 if true_app else -0.4)))
        if actor.age>LIFESPAN: actor.alive=False

def convention(members, rng):
    A=[b for b in members if b.alive]; bd=[0]*N_BEHAV
    for b in A: bd[b.does()]+=1
    if not sum(bd): return -1,0
    c=amr(bd,rng); return c, bd[c]/sum(bd)   # RANDOM tie-break now

def ndiv(members):
    A=[b for b in members if b.alive]; d=[0]*N_NICHE
    for b in A: d[b.niche()]+=1
    t=sum(d)
    if not t: return 0
    h=0
    for c in d:
        if c>0: p=c/t; h-=p*math.log(p)
    return h/math.log(N_NICHE)

def run_one(seed, years=8, pop=12):
    rng=random.Random(seed)
    groups={g:[Being(g,seed=seed+g*1000+i) for i in range(pop)] for g in (0,1)}
    for g in groups:
        for b in groups[g]: b.age=rng.randint(0,LIFESPAN-200)
    maxdepth={0:0,1:0}; extinct=False
    for year in range(years):
        nb=[0.7+0.2*math.sin(year*0.6+i) for i in range(N_NICHE)]
        for day in range(350):
            for g in groups:
                step(groups[g], nb, rng)
                dead=[b for b in groups[g] if not b.alive]
                if dead:
                    groups[g]=[b for b in groups[g] if b.alive]
                    while len(groups[g])<pop and len(groups[g])>=2:
                        par=sorted(groups[g],key=lambda b:b.wellbeing,reverse=True)[:2]
                        c=Being(g,inherited_niche=par[0].niche(),seed=seed+g*9000+nid())
                        c.depth=max(p.depth for p in par)+1; maxdepth[g]=max(maxdepth[g],c.depth)
                        groups[g].append(c)
                    while len(groups[g])<2: groups[g].append(Being(g,seed=seed+g*9000+nid()))
    res={}
    for g in groups:
        A=[b for b in groups[g] if b.alive]
        conv,adh=convention(groups[g],rng)
        res[g]=dict(conv=conv, adh=adh,
                    conf=statistics.mean([b.confidence() for b in A]) if A else 0,
                    ndiv=ndiv(groups[g]), depth=maxdepth[g],
                    rev=statistics.mean([b.revisions for b in A]) if A else 0)
    return res

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seeds",type=int,default=150); a=ap.parse_args()
    print("="*64); print("  SOCIETY AUDIT: many seeds + realistic noise (friend's test)"); print("="*64)

    conv_counts=[0]*N_BEHAV; differ=0; adh=[]; conf=[]; divs=[]; depths=[]; revs=[]
    for s in range(a.seeds):
        r=run_one(s*53+11)
        for g in (0,1):
            if r[g]['conv']>=0: conv_counts[r[g]['conv']]+=1
            adh.append(r[g]['adh']); conf.append(r[g]['conf'])
            divs.append(r[g]['ndiv']); depths.append(r[g]['depth']); revs.append(r[g]['rev'])
        if r[0]['conv']!=r[1]['conv'] and r[0]['conv']>=0 and r[1]['conv']>=0: differ+=1

    print(f"\n  CONVENTION DISTRIBUTION across all groups: {conv_counts}")
    bias=max(conv_counts)/sum(conv_counts) if sum(conv_counts) else 0
    print(f"    most-common share: {bias:.2f} "
          f"({'~unbiased, groups pick varied conventions' if bias<0.4 else 'BIAS - one convention dominates'})")
    print(f"  two groups ended with DIFFERENT conventions: {differ}/{a.seeds} ({differ*100//a.seeds}%)")
    print(f"\n  ADHERENCE distribution:  mean {statistics.mean(adh):.2f} | "
          f"min {min(adh):.2f} | max {max(adh):.2f}  (100% forever would be unrealistic)")
    print(f"  CONFIDENCE distribution: mean {statistics.mean(conf):.2f} | "
          f"min {min(conf):.2f} | max {max(conf):.2f}")
    print(f"  niche diversity: mean {statistics.mean(divs):.2f}")
    print(f"  lineage depth:   mean {statistics.mean(depths):.1f}")
    print(f"  revisions/being: mean {statistics.mean(revs):.1f}")

    print("\n"+"="*64); print("  WHAT THE DATA SAYS"); print("="*64)
    unbiased = bias<0.4; diverges = differ>a.seeds*0.3; realistic = statistics.mean(adh)<0.98
    if unbiased and diverges and realistic:
        print(f"  HEALTHY INTEGRATED SOCIETY. Conventions spread across behaviors")
        print(f"  ({conv_counts}, no #3 bias), groups diverged culturally in {differ*100//a.seeds}% of runs,")
        print(f"  and adherence is realistic (mean {statistics.mean(adh):.2f}, not 100% forever). Ecology,")
        print(f"  culture, norms, lineage all co-running realistically across many seeds.")
        print(f"  The integration bug is fixed, the combined system behaves like the")
        print(f"  validated parts.")
    else:
        issues=[]
        if not unbiased: issues.append(f"convention bias ({conv_counts})")
        if not diverges: issues.append(f"groups rarely diverge ({differ*100//a.seeds}%)")
        if not realistic: issues.append(f"adherence too perfect ({statistics.mean(adh):.2f})")
        print(f"  Honest: {'; '.join(issues)}. The data characterizes the integration")
        print(f"  behavior straight - reported as-is, not tuned.")

if __name__=="__main__": main()
