"""Culture that survives optimization: add a social-conformity force so traditions
persist even when raw payoff would abandon them.

The honest insight from v1: pure optimizers steamroll culture, they abandon any
tradition the moment another niche pays more, so distinct group-traditions never
survive. Real culture persists because beings value BELONGING, aligning with
your group gives a social benefit (cooperation, identity, acceptance), so a
tradition can hold even against a slightly better individual option.

Fix: a being's payoff now includes a SOCIAL term, doing what most of your group
does gives a belonging bonus. So there's a real tension: raw ecology pulls toward
the best niche; belonging pulls toward your group's tradition. When belonging
matters enough, distinct traditions PERSIST and transmit culturally.

Then the cross-fostering control: a being raised among a different group should
adopt THAT group's tradition (culture), not its genetic kin's (genes).

    python culture2.py --seeds 100
"""
import random, math, argparse, statistics

N=4
_id=[0]
def nid(): _id[0]+=1; return _id[0]
LIFESPAN=1000

class Being:
    def __init__(self, genetic_lean=None, group=0, start_age=0, seed=0):
        self.id=nid(); self.group=group
        r=random.Random(self.id*7+seed)
        if genetic_lean is not None:
            self.gene=[max(0,genetic_lean[i]+r.uniform(-0.05,0.05)) for i in range(N)]
        else:
            self.gene=[r.uniform(0,0.1) for _ in range(N)]
        self.behavior=list(self.gene)
        self.birth_pref=max(range(N),key=lambda i:self.gene[i])
        self.wellbeing=0.5; self.age=start_age; self.alive=True; self.rng=r
    def niche(self): return max(range(N),key=lambda i:self.behavior[i])
    def live(self, base_quality, group_dist, elders):
        self.age+=1; ni=self.niche()
        group_total=sum(group_dist) or 1
        conformity = group_dist[ni]/group_total   # fraction of my group doing this
        # payoff = ecological quality + BELONGING bonus for matching the group
        payoff = base_quality[ni] + 0.6*conformity
        outcome=1 if payoff+self.rng.uniform(-0.12,0.12)>0.75 else -1
        self.wellbeing=max(0,min(1,0.97*self.wellbeing+0.06*outcome))
        self.behavior[ni]=max(0,self.behavior[ni]+0.04*outcome)
        # cultural learning: imitate successful elders of the group
        if self.age < LIFESPAN*0.35 and elders and self.rng.random()<0.12:
            teacher=max(elders,key=lambda e:e.wellbeing)
            if teacher.wellbeing>0.5:
                self.behavior[teacher.niche()]+=0.3
        if self.age>LIFESPAN: self.alive=False

def dominant(bs):
    d=[0]*N
    for b in bs:
        if b.alive: d[b.niche()]+=1
    return (max(range(N),key=lambda i:d[i]) if sum(d) else -1), d

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seeds",type=int,default=100); a=ap.parse_args()
    print("="*64); print("  CULTURE WITH BELONGING (traditions survive optimization)"); print("="*64)

    distinct=0; total=0; cross_adopt=0; cross_total=0
    for s in range(a.seeds):
        rng=random.Random(s*61+3)
        groups={0:[],1:[]}
        for g in (0,1):
            groups[g]=[Being(group=g,start_age=rng.randint(0,LIFESPAN-200),seed=s*1000+g*100+i) for i in range(8)]
        for b in groups[0]: b.behavior[0]+=0.5   # group 0 starts a niche-0 tradition
        for b in groups[1]: b.behavior[2]+=0.5   # group 1 starts a niche-2 tradition

        for ep in range(6):
            base=[0.7+0.15*math.sin(ep*0.6+i) for i in range(N)]
            for day in range(350):
                for g in (0,1):
                    A=[b for b in groups[g] if b.alive]
                    if not A: continue
                    _, gd = dominant(A)
                    elders=[b for b in A if b.age>LIFESPAN*0.5 and b.wellbeing>0.5]
                    for b in A: b.live(base, gd, elders)
                    dead=[b for b in groups[g] if not b.alive]
                    if dead:
                        groups[g]=[b for b in groups[g] if b.alive]
                        while len(groups[g])<8 and len(groups[g])>=2:
                            par=sorted(groups[g],key=lambda b:b.wellbeing,reverse=True)[:2]
                            child=Being(genetic_lean=par[0].gene,group=g,seed=s*5000+nid())
                            groups[g].append(child)
                        while len(groups[g])<2: groups[g].append(Being(group=g,seed=s*5000+nid()))
        trad0,_=dominant(groups[0]); trad1,_=dominant(groups[1])
        total+=1
        if trad0!=trad1 and trad0!=-1 and trad1!=-1: distinct+=1

        # cross-foster: genetic child of group 0, raised among group 1
        g0=[b for b in groups[0] if b.alive]
        if g0 and trad1!=-1:
            foster=Being(genetic_lean=g0[0].gene, group=1, seed=s*9999)
            bp=foster.birth_pref
            for _ in range(int(LIFESPAN*0.34)):
                A=[b for b in groups[1] if b.alive]
                _,gd=dominant(A)
                elders=[b for b in A if b.age>LIFESPAN*0.5 and b.wellbeing>0.5]
                foster.live([0.7]*N, gd, elders)
                foster.age=min(foster.age, int(LIFESPAN*0.34)-1)
            cross_total+=1
            if foster.niche()==trad1 and trad1!=bp: cross_adopt+=1

    print(f"\n  across {total} two-group worlds:")
    print(f"    groups developed DIFFERENT traditions: {distinct}/{total} ({distinct*100//total}%)")
    print(f"  cross-fostering (culture vs genes):")
    print(f"    genetic child of group 0, raised in group 1, adopted group 1's")
    print(f"    tradition: {cross_adopt}/{cross_total} ({cross_adopt*100//max(1,cross_total)}%)")

    print("\n"+"="*64); print("  WHAT THIS ESTABLISHES"); print("="*64)
    d_ok = distinct>total*0.5; c_ok = cross_adopt>cross_total*0.4
    if d_ok and c_ok:
        print(f"  DIGITAL CULTURE. With a belonging force, distinct traditions PERSISTED")
        print(f"  ({distinct*100//total}% of worlds the two groups kept different ways of life),")
        print(f"  surviving ecological optimization. And a being raised among non-kin")
        print(f"  adopted its GROUP's tradition, not its genetic lean ({cross_adopt*100//max(1,cross_total)}%), proving the")
        print(f"  behavior is LEARNED/cultural, not inherited. Traditions form, differ")
        print(f"  between groups, persist against optimization, and transmit culturally.")
        print(f"  This is the jump from digital ecology to digital SOCIETY.")
    else:
        print(f"  distinct traditions {distinct*100//total}%, cross-foster adoption {cross_adopt*100//max(1,cross_total)}%.")
        print(f"  Honest: {'culture forms but transmission not cleanly cultural' if d_ok else 'belonging not yet strong enough to hold distinct traditions'}. Reported straight.")

if __name__=="__main__": main()
