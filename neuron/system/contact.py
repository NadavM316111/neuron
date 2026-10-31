"""History happens: two groups with their OWN cultures come into CONTACT. What
happens is NOT scripted, it emerges from the same validated mechanisms.

Authentic design: nothing codes "they blend" or "one wins." Two groups each
develop their own convention in isolation first. Then the boundary opens: members
can now observe and be judged by members of the OTHER group too. From there, the
same mechanisms already validated (observe behavior, infer the norm, conform
toward what's approved, reputation, enforcement) simply run across the combined
population. Whatever results, blending, one culture spreading, persistence of
both, churn, is genuine emergence, not authored.

We just watch and report what history did: did the cultures stay distinct, merge
into one, did one displace the other, or did something new form?

    python contact.py --seeds 150
"""
import random, math, argparse, statistics

N_BEHAV=4
_id=[0]
def nid(): _id[0]+=1; return _id[0]
LIFESPAN=1200
MISS=0.2; AMBIG=0.2; BREAK=0.15; MEM=0.1
def amr(vals,rng):
    m=max(vals); best=[i for i,v in enumerate(vals) if abs(v-m)<1e-9]; return rng.choice(best)

class Being:
    def __init__(self, group, seed=0):
        self.id=nid(); self.group=group; r=random.Random(self.id*7+seed); self.rng=r
        self.behavior=[0.5]*N_BEHAV
        self.approval_mem=[0.0]*N_BEHAV; self.obs=0
        self.wellbeing=0.5; self.age=0; self.alive=True; self.reputation=0.5
    def does(self): return amr(self.behavior,self.rng)
    def inferred_norm(self):
        if self.obs==0: return self.does()
        return amr(self.approval_mem,self.rng)
    def react(self, beh): return beh==self.inferred_norm()
    def observe(self, beh, approved):
        if self.rng.random()<MEM: beh=self.rng.randrange(N_BEHAV)
        self.obs+=1; self.approval_mem[beh]+=(1.0 if approved else -1.0)

def step(interacting_pool, members, rng):
    # interacting_pool = who this member can be judged by (own group, or both
    # groups after contact). members = the acting population.
    A=[b for b in members if b.alive]
    if len(A)<3: return
    for actor in A:
        actor.age+=1
        beh=actor.does()
        if actor.rng.random()<BREAK: beh=actor.rng.randrange(N_BEHAV)
        # the panel of observers is drawn from the interacting pool (post-contact,
        # this includes the OTHER group - that's the only change contact makes)
        pool=[o for o in interacting_pool if o is not actor and o.alive]
        actor.rng.shuffle(pool); panel=pool[:min(5,len(pool))]
        if not panel: continue
        approvals=sum(1 for o in panel if o.react(beh))
        true_app=approvals>=len(panel)/2
        for o in panel:
            if o.rng.random()<MISS: continue
            app=(o.rng.random()<0.5) if o.rng.random()<AMBIG else true_app
            o.observe(beh, app)
        if true_app:
            actor.behavior[beh]=min(2,actor.behavior[beh]+0.05)
        else:
            actor.behavior[beh]=max(0,actor.behavior[beh]-0.04)
            actor.behavior[actor.inferred_norm()]+=0.05
        if actor.age>LIFESPAN: actor.alive=False

def convention(members, rng):
    A=[b for b in members if b.alive]; bd=[0]*N_BEHAV
    for b in A: bd[b.does()]+=1
    if not sum(bd): return -1,0
    c=amr(bd,rng); return c, bd[c]/sum(bd)

def run_one(seed, pop=14):
    rng=random.Random(seed)
    g0=[Being(0,seed+i) for i in range(pop)]
    g1=[Being(1,seed+500+i) for i in range(pop)]
    for b in g0+g1: b.age=rng.randint(0,LIFESPAN-200)

    # PHASE 1: isolation - each group forms its OWN culture (judged only by own group)
    for day in range(1500):
        step(g0, g0, rng); step(g1, g1, rng)
        for grp,sd in ((g0,seed),(g1,seed+500)):
            dead=[b for b in grp if not b.alive]
            if dead:
                grp[:]=[b for b in grp if b.alive]
                while len(grp)<pop and len(grp)>=2: grp.append(Being(grp[0].group,sd+nid()))
                while len(grp)<2: grp.append(Being(grp[0].group,sd+nid()))
    c0_before,_=convention(g0,rng); c1_before,_=convention(g1,rng)
    cultures_differed = (c0_before!=c1_before)

    # PHASE 2: CONTACT - the boundary opens. now everyone can be judged by BOTH
    # groups. nothing else changes. we just let the mechanisms run on the
    # combined pool and see what history does.
    combined=g0+g1
    for day in range(1500):
        step(combined, combined, rng)
        dead=[b for b in combined if not b.alive]
        if dead:
            combined[:]=[b for b in combined if b.alive]
            while len(combined)<2*pop and len(combined)>=2:
                combined.append(Being(random.Random(day).randrange(2), seed+nid()))
            while len(combined)<2: combined.append(Being(0,seed+nid()))

    # what did history do? look at the combined population's conventions
    c_after,_=convention(combined,rng)
    bd=[0]*N_BEHAV
    for b in combined:
        if b.alive: bd[b.does()]+=1
    tot=sum(bd) or 1
    dominant_share=max(bd)/tot
    # outcome classification (descriptive, not prescriptive)
    n_conventions = sum(1 for c in bd if c/tot>0.2)
    return dict(differed=cultures_differed, c0=c0_before, c1=c1_before,
                after=c_after, dominant_share=dominant_share, n_conv=n_conventions,
                merged=(c_after in (c0_before,c1_before) and dominant_share>0.6),
                blended_new=(c_after not in (c0_before,c1_before) and dominant_share>0.5),
                stayed_split=(n_conventions>=2))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seeds",type=int,default=150); a=ap.parse_args()
    print("="*64); print("  HISTORY: two cultures meet. what happens is not scripted."); print("="*64)
    print("  Each group forms its own culture in isolation, then the boundary")
    print("  opens. The same mechanisms run across both. We watch what emerges.\n")

    differed=0; one_spread=0; blended=0; split=0; shares=[]
    for s in range(a.seeds):
        r=run_one(s*53+11)
        if r['differed']: differed+=1
        else: continue   # only count runs where the two cultures actually differed pre-contact
        shares.append(r['dominant_share'])
        if r['merged']: one_spread+=1
        elif r['blended_new']: blended+=1
        elif r['stayed_split']: split+=1

    print(f"  runs where the two groups had DIFFERENT cultures before contact: "
          f"{differed}/{a.seeds}")
    if differed:
        print(f"\n  WHAT HISTORY DID when different cultures met (of those {differed} runs):")
        print(f"    one culture spread to dominate:   {one_spread} ({one_spread*100//differed}%)")
        print(f"    a new blended convention formed:  {blended} ({blended*100//differed}%)")
        print(f"    both cultures persisted (split):  {split} ({split*100//differed}%)")
        print(f"    avg dominant-convention share after contact: {statistics.mean(shares):.2f}")

    print("\n"+"="*64); print("  WHAT THIS SHOWS"); print("="*64)
    print("  Two cultures met and history took its course, with NO scripted outcome.")
    print("  Whether one spread, a new culture blended, or both persisted emerged")
    print("  purely from beings observing, inferring, conforming, and enforcing")
    print("  across the opened boundary. The society doesn't just exist, things")
    print("  HAPPEN to it and it changes. This is a society with history, and the")
    print("  outcomes above are what genuinely emerged, not what anyone coded.")

if __name__=="__main__": main()
