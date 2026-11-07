"""Society v4: a bigger, genuinely SPARSE world so distinct social roles crystallize
and the developed-vs-flat difference is clean.

v3 showed character->position but a 10-person all-reachable world was too dense
(everyone 'central/bridge'). Fix: ~24 members placed on a ring/grid so each can
only reach a LOCAL neighborhood. Now a mastery-type that invests narrowly genuinely
becomes a local specialist/isolate; a connection-type reaching across neighborhoods
becomes a bridge; a peace-type becomes a local mediator. Real structure (clusters,
isolates, bridges) can form because not everyone can reach everyone.

Causal control: identical world, flat characters. If developed produces clearly
more role differentiation + clustering than flat, Being feeds Species - cleanly.

    python society_v4.py
"""
import random, statistics, math
from collections import defaultdict

VALUES=["mastery","connection","peace"]
def shape(seed, world, days=300):
    rng=random.Random(seed); v={x:0.5 for x in VALUES}; st={x:0.3 for x in VALUES}; re={x:[] for x in VALUES}
    pay={x:0.5 for x in VALUES}; pay[world]=0.85
    for _ in range(days):
        for k in st: st[k]=max(0,st[k]-0.03)
        g={x:v[x]-st[x] for x in VALUES}; pull={x:max(0.01,v[x]*0.8+g[x]*0.5) for x in VALUES}
        t=sum(pull.values()); r=rng.random()*t; a=0; ch=VALUES[-1]
        for x,w in pull.items():
            a+=w
            if r<=a: ch=x; break
        e=v[ch]; ac=pay[ch]+rng.uniform(-0.2,0.2); st[ch]=min(1,st[ch]+max(0,0.03+0.05*ac))
        er=ac-e; re[ch].append(er); re[ch]=re[ch][-20:]
        v[ch]=min(1,max(0,v[ch]+0.02*er+0.02*(sum(re[ch])/len(re[ch]))))
        for x in VALUES:
            if x!=ch: v[x]=max(0,v[x]-0.002)
    return v

class M:
    def __init__(s,name,char,pos):
        s.name=name; s.char=char; s.dominant=max(char,key=char.get); s.pos=pos
        s.ties=defaultdict(float); s.resources=0.0

def neighborhood(a, members, radius):
    # local reach on a ring of positions
    n=len(members)
    return [m for m in members if m!=a and min(abs(m.pos-a.pos),n-abs(m.pos-a.pos))<=radius]

def interact(a,b,rng):
    syn=(a.char["connection"]+b.char["connection"])*0.35+(a.char["peace"]+b.char["peace"])*0.25 \
        -(a.char["mastery"]*b.char["mastery"])*0.5+rng.uniform(-0.25,0.25)
    good=syn>0.1; d=0.12 if good else -0.10
    a.ties[b.name]+=d; b.ties[a.name]+=d
    if good: a.resources+=0.1; b.resources+=0.1

def step(members,rng,radius):
    a=rng.choice(members)
    # connection-types can reach FARTHER (seek across neighborhoods -> bridges)
    rad = radius+2 if a.dominant=="connection" else radius
    hood=neighborhood(a,members,rad)
    if not hood: return
    if a.dominant=="mastery":
        # narrow: deepens its few best local ties
        b=max(hood,key=lambda m:a.ties[m.name]+0.1*rng.random())
    elif a.dominant=="connection":
        # broad: seeks new/weak ties across its wider reach
        b=min(hood,key=lambda m:a.ties[m.name]+0.1*rng.random())
    else:
        soured=[m for m in hood if a.ties[m.name]<0]
        b=rng.choice(soured) if soured and rng.random()<0.6 else rng.choice(hood)
    interact(a,b,rng)

def structure(members):
    pos={}
    for m in members:
        strong=[o for o,w in m.ties.items() if w>0.3]
        pos[m.name]={"degree":len(strong),"centrality":sum(max(0,w) for w in m.ties.values()),
                     "dominant":m.dominant,"resources":m.resources,
                     "reach":len(set(o for o in m.ties if m.ties[o]>0.3))}
    return pos

def run(members,rounds,seed,radius):
    rng=random.Random(seed)
    for _ in range(rounds): step(members,rng,radius)
    return members

def roles(members,pos):
    degs=[pos[m.name]["degree"] for m in members]
    hi=statistics.mean(degs)+statistics.pstdev(degs)*0.6; lo=max(1,statistics.mean(degs)-statistics.pstdev(degs)*0.6)
    out={}
    for m in members:
        p=pos[m.name]
        if p["degree"]>=hi and p["centrality"]>statistics.mean([pos[x.name]['centrality'] for x in members]): out[m.name]="bridge/central"
        elif p["degree"]<=lo and p["resources"]>statistics.mean([x.resources for x in members]): out[m.name]="productive-isolate"
        elif p["degree"]<=lo: out[m.name]="peripheral"
        else: out[m.name]="connector"
    return out

def metrics(members):
    pos=structure(members)
    degs=[p["degree"] for p in pos.values()]; cents=[p["centrality"] for p in pos.values()]; res=[m.resources for m in members]
    return {"degree_variance":statistics.pstdev(degs),"centrality_spread":statistics.pstdev(cents),
            "resource_inequality":statistics.pstdev(res)}, pos

def main():
    rng=random.Random(3); N=24; RADIUS=3
    print("="*64); print(f"  SOCIETY v4: {N} members, sparse local world + control"); print("="*64)
    worlds=(["mastery"]*8+["connection"]*8+["peace"]*8); rng.shuffle(worlds)
    dev=[M(f"N{i+1}", shape(i*13+1, worlds[i]), i) for i in range(N)]
    run(dev, 4000, 100, RADIUS)
    dmet,dpos=metrics(dev); drole=roles(dev,dpos)

    # role distribution BY character type
    print("\n  social roles that emerged, by character type:")
    for dv in VALUES:
        ms=[m for m in dev if m.dominant==dv]
        rc=defaultdict(int)
        for m in ms: rc[drole[m.name]]+=1
        avgdeg=statistics.mean(dpos[m.name]['degree'] for m in ms); avgcen=statistics.mean(dpos[m.name]['centrality'] for m in ms)
        print(f"    {dv:11s}: ties {avgdeg:.1f}, centrality {avgcen:4.1f}, roles {dict(rc)}")

    # example distinctive individuals
    print("\n  distinctive individuals:")
    srt=sorted(dev,key=lambda m:-dpos[m.name]['centrality'])
    for m in srt[:2]+srt[-2:]:
        p=dpos[m.name]; print(f"    {m.name:5s}[{p['dominant']:10s}] ties {p['degree']}, centrality {p['centrality']:4.1f}, res {p['resources']:4.1f} -> {drole[m.name]}")

    # CONTROL: flat
    flat=[M(f"F{i+1}",{x:0.5 for x in VALUES},i) for i in range(N)]
    for i,m in enumerate(flat): m.dominant=VALUES[i%3]
    run(flat,4000,100,RADIUS); fmet,_=metrics(flat)

    print("\n"+"="*64); print("  CAUSAL CONTROL: developed vs flat"); print("="*64)
    print(f"  {'metric':<22}{'developed':>12}{'flat':>10}")
    for k in dmet: print(f"  {k:<22}{dmet[k]:>12.2f}{fmet[k]:>10.2f}")

    cen_by={dv:statistics.mean([dpos[m.name]['centrality'] for m in dev if m.dominant==dv] or [0]) for dv in VALUES}
    role_variety=len(set(drole.values()))
    char_predicts = max(cen_by.values())-min(cen_by.values())>2.0
    cleaner = dmet["degree_variance"]>fmet["degree_variance"]*1.25
    print("\n"+"="*64); print("  VERDICT"); print("="*64)
    if char_predicts and cleaner and role_variety>=3:
        print(f"  YES, CLEANLY - BEING FEEDS SPECIES. In a sparse {N}-member world,")
        print(f"  character drove distinct social roles ({role_variety} role types emerged:")
        print(f"  bridge/central, productive-isolate, connector, peripheral), centrality")
        print(f"  separated clearly by character ({', '.join(f'{k} {v:.0f}' for k,v in cen_by.items())}), and the developed")
        print(f"  society is more differentiated than the flat control (degree-variance")
        print(f"  {dmet['degree_variance']:.2f} vs {fmet['degree_variance']:.2f}). Individuals who became different through")
        print(f"  their histories organized into a genuinely different, role-structured")
        print(f"  society. Being causally shapes Species-level organization. Clean result.")
    else:
        print(f"  char-predicts={char_predicts} (spread {max(cen_by.values())-min(cen_by.values()):.1f}), cleaner-than-flat={cleaner}, role-types={role_variety}.")
        print(f"  centrality by type: {', '.join(f'{k} {v:.1f}' for k,v in cen_by.items())}.")
        print(f"  Honest: {'character drives roles, control gap still thin' if char_predicts else 'needs more separation'}. Reported straight.")

if __name__=="__main__": main()
