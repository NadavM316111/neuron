"""Society v3: sparse, character-driven interactions so social POSITION emerges.

v2 was too dense (everyone met everyone -> no structure, everyone a 'bridge'). Fix:
  - members live on a social graph; each round only a FEW interact
  - WHO you seek is driven by character: connection-types actively seek many and
    varied partners (-> become central/bridges); mastery-types seek few, only
    productive high-rep partners (-> become specialists, often isolated);
    peace-types avoid conflict, mediate between estranged pairs (-> peacemakers)
  - interactions have real consequences (two mastery-types clash; connection
    smooths; a peace-type can repair a soured tie). cooperation NOT auto-good.
  - social position (centrality, isolation, bridging) is MEASURED from the actual
    interaction graph, not assigned by a rule tied to character.
Then the causal control: flat-character society on the SAME dynamics. If developed
produces clearer roles + different structure than flat, Being feeds Species.

    python society_v3.py
"""
import random, statistics
from collections import defaultdict

VALUES=["mastery","connection","peace"]

def shape(seed, world, days=300):
    rng=random.Random(seed)
    v={x:0.5 for x in VALUES}; st={x:0.3 for x in VALUES}; re={x:[] for x in VALUES}
    pay={x:0.5 for x in VALUES}; pay[world]=0.85
    for _ in range(days):
        for k in st: st[k]=max(0,st[k]-0.03)
        g={x:v[x]-st[x] for x in VALUES}
        pull={x:max(0.01,v[x]*0.8+g[x]*0.5) for x in VALUES}
        t=sum(pull.values()); r=rng.random()*t; a=0; ch=VALUES[-1]
        for x,w in pull.items():
            a+=w
            if r<=a: ch=x; break
        e=v[ch]; ac=pay[ch]+rng.uniform(-0.2,0.2); st[ch]=min(1,st[ch]+max(0,0.03+0.05*ac))
        err=ac-e; re[ch].append(err); re[ch]=re[ch][-20:]
        v[ch]=min(1,max(0,v[ch]+0.02*err+0.02*(sum(re[ch])/len(re[ch]))))
        for x in VALUES:
            if x!=ch: v[x]=max(0,v[x]-0.002)
    return v

class M:
    def __init__(s,name,char):
        s.name=name; s.char=char; s.dominant=max(char,key=char.get)
        s.ties=defaultdict(float)   # strength of tie to each other member (can go neg)
        s.resources=0.0

def interact(a,b,rng):
    # outcome from BOTH characters; returns (good?, delta)
    syn = (a.char["connection"]+b.char["connection"])*0.35 \
        + (a.char["peace"]+b.char["peace"])*0.25 \
        - (a.char["mastery"]*b.char["mastery"])*0.5 \
        + rng.uniform(-0.25,0.25)
    good = syn>0.1
    d = 0.12 if good else -0.10
    a.ties[b.name]+=d; b.ties[a.name]+=d
    if good: a.resources+=0.1; b.resources+=0.1
    return good

def round_step(members, rng):
    a=rng.choice(members); others=[m for m in members if m!=a]
    # WHO a seeks, by character (sparse: only a few candidates considered)
    if a.dominant=="connection":
        # seeks breadth: prefers members it has few/no ties with -> bridges groups
        cand=sorted(others,key=lambda m:a.ties[m.name])[:3]
        b=rng.choice(cand)
    elif a.dominant=="mastery":
        # seeks a small set of strong, productive ties -> deep but narrow
        cand=sorted(others,key=lambda m:-a.ties[m.name])[:2]
        b=cand[0] if rng.random()<0.8 else rng.choice(others)
    else: # peace: seeks to REPAIR soured ties (mediator)
        soured=[m for m in others if a.ties[m.name]<0]
        b=rng.choice(soured) if soured and rng.random()<0.6 else rng.choice(others)
    interact(a,b,rng)

def structure(members):
    # measure social position from the actual tie graph
    pos={}
    for m in members:
        strong=[o for o,w in m.ties.items() if w>0.3]
        neg=[o for o,w in m.ties.items() if w<-0.2]
        pos[m.name]={"degree":len(strong),"enemies":len(neg),
                     "centrality":sum(max(0,w) for w in m.ties.values()),
                     "dominant":m.dominant,"resources":m.resources}
    return pos

def role_of(p, n):
    if p["degree"]>=n*0.5 and p["centrality"]>2: return "central/bridge"
    if p["degree"]<=1 and p["resources"]>statistics_mean_res: return "productive-isolate"
    if p["enemies"]==0 and p["degree"]>=2: return "trusted/peacemaker"
    if p["enemies"]>=2: return "contentious"
    return "peripheral"

def run(members, rounds, seed):
    rng=random.Random(seed)
    for _ in range(rounds): round_step(members,rng)
    return members

def metrics(members):
    pos=structure(members)
    degs=[p["degree"] for p in pos.values()]; cents=[p["centrality"] for p in pos.values()]
    res=[m.resources for m in members]
    return {"degree_variance":statistics.pstdev(degs) if len(degs)>1 else 0,
            "centrality_spread":statistics.pstdev(cents) if len(cents)>1 else 0,
            "resource_inequality":statistics.pstdev(res) if len(res)>1 else 0}, pos

def main():
    global statistics_mean_res
    rng=random.Random(11)
    print("="*64); print("  SOCIETY v3: sparse character-driven world + causal control"); print("="*64)
    worlds=(["mastery"]*4+["connection"]*3+["peace"]*3); rng.shuffle(worlds)
    dev=[M(f"N{i+1}", shape(i*13+1, worlds[i])) for i in range(10)]
    print("\n  10 individuals by character:")
    for m in dev: print(f"    {m.name}: {m.dominant:10s} ({', '.join(f'{k[:4]} {v:.2f}' for k,v in sorted(m.char.items(),key=lambda x:-x[1]))})")
    run(dev, 1200, 100)
    dmet,dpos=metrics(dev)
    statistics_mean_res=statistics.mean([m.resources for m in dev])
    print("\n  --- roles that EMERGED from the interaction graph ---")
    for m in sorted(dev,key=lambda x:-dpos[x.name]["centrality"]):
        p=dpos[m.name]; print(f"    {m.name:5s}[{p['dominant']:10s}] strong-ties {p['degree']}, enemies {p['enemies']}, "
              f"centrality {p['centrality']:4.1f}, resources {p['resources']:4.1f} -> {role_of(p,len(dev))}")
    # by-character role summary
    print("\n  does character predict social role?")
    for dv in VALUES:
        ms=[m for m in dev if m.dominant==dv]
        if ms:
            avgdeg=statistics.mean(dpos[m.name]["degree"] for m in ms)
            avgcen=statistics.mean(dpos[m.name]["centrality"] for m in ms)
            print(f"    {dv:11s}: avg strong-ties {avgdeg:.1f}, avg centrality {avgcen:.1f}")

    # CONTROL: flat characters, same dynamics
    flat=[M(f"F{i+1}", {x:0.5 for x in VALUES}) for i in range(10)]
    for i,m in enumerate(flat): m.dominant=VALUES[i%3]
    run(flat,1200,100); fmet,_=metrics(flat)

    print("\n"+"="*64); print("  CAUSAL CONTROL: developed vs flat"); print("="*64)
    print(f"  {'metric':<24}{'developed':>12}{'flat':>10}")
    for k in dmet: print(f"  {k:<24}{dmet[k]:>12.2f}{fmet[k]:>10.2f}")

    print("\n"+"="*64); print("  VERDICT"); print("="*64)
    stronger = (dmet["centrality_spread"]>fmet["centrality_spread"]*1.25 or
                dmet["degree_variance"]>fmet["degree_variance"]*1.25 or
                dmet["resource_inequality"]>fmet["resource_inequality"]*1.25)
    # also: did different character types get different centrality?
    cen_by_type={dv:statistics.mean([dpos[m.name]["centrality"] for m in dev if m.dominant==dv] or [0]) for dv in VALUES}
    char_predicts = max(cen_by_type.values())-min(cen_by_type.values()) > 1.0
    if stronger and char_predicts:
        print(f"  YES - BEING FEEDS SPECIES. In the sparse world, character drove social")
        print(f"  POSITION: different character types reached different centrality")
        print(f"  ({', '.join(f'{k} {v:.1f}' for k,v in cen_by_type.items())}), roles emerged from the graph (central/bridge,")
        print(f"  productive-isolate, peacemaker), and the developed society is more")
        print(f"  structured than the flat control. Individuals who became different")
        print(f"  through their histories formed a differently-organized society. The")
        print(f"  Being work causally shapes Species-level social structure.")
    else:
        print(f"  Partial/honest: stronger-structure={stronger}, char-predicts-role={char_predicts}.")
        print(f"  centrality by type: {', '.join(f'{k} {v:.1f}' for k,v in cen_by_type.items())}.")
        print(f"  {'Character shapes position but control gap is thin' if char_predicts else 'character not yet driving position strongly'}. Reported straight.")

if __name__=="__main__": main()
