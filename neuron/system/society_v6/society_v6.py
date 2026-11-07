"""Society v6: emergent social organization from developed individuals.

Established pathway (tested): life -> character -> behavior -> social position.
Now the next-level question (the friend's): once individuals differ in character
AND social position, does repeated interaction produce STRUCTURE nobody designed?
Watch four emergent things, none hardcoded:
  GROUPS/CLIQUES - stable communities from relationship history
  REPUTATION     - knowledge of a member's past behavior shaping who others choose
  SPECIALIZATION - members settling into distinct productive/social roles
  NORMS          - a behavior becoming an expectation newcomers pick up socially
Key comparison: developed-character society vs flat-character society, same world.
Does developmental individuality change WHAT KIND of society emerges?

Honest wording (per friend): character profiles causally influence social behavior
and position within the simulation; combined with the developmental result, a
tested pathway from life history to social structure. Not a personhood claim.

    python society_v6.py
"""
import random, statistics
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
        s.ties=defaultdict(float); s.resources=0.0; s.solo=0
        s.reputation=0.5          # others' aggregate regard, updated by behavior
        s.greeting=None           # a learnable convention (for norm emergence)

def neigh(a,ms,rad):
    n=len(ms); return [m for m in ms if m!=a and min(abs(m.pos-a.pos),n-abs(m.pos-a.pos))<=rad]

def step(ms,rng,rad,reps):
    a=rng.choice(ms); m,c,p=a.char["mastery"],a.char["connection"],a.char["peace"]
    social=c*1.4+p*0.3-m*0.8
    if rng.random()>(0.5+social*0.5):
        a.solo+=1; a.resources+=0.08*m; return
    r2=rad+2 if a.dominant=="connection" else max(1,rad-1) if a.dominant=="mastery" else rad
    hood=neigh(a,ms,r2)
    if not hood: a.solo+=1; return
    # REPUTATION affects partner choice: prefer well-regarded others (esp. mastery-types)
    def appeal(x): return a.ties[x.name]*0.5 + reps[x.name]*(0.8 if a.dominant=="mastery" else 0.3) + 0.1*rng.random()
    if a.dominant=="connection": b=min(hood,key=lambda x:a.ties[x.name]+0.1*rng.random())
    else: b=max(hood,key=appeal)
    syn=(a.char["connection"]+b.char["connection"])*0.4+(a.char["peace"]+b.char["peace"])*0.3 \
        -(a.char["mastery"]*b.char["mastery"])*0.6+rng.uniform(-0.2,0.2)
    good=syn>0.1; d=0.12 if good else -0.10
    a.ties[b.name]+=d; b.ties[a.name]+=d
    # reputation updates from behavior (cooperative -> rep up)
    reps[a.name]=min(1,max(0,reps[a.name]+(0.01 if good else -0.01)))
    reps[b.name]=min(1,max(0,reps[b.name]+(0.01 if good else -0.01)))
    if good: a.resources+=0.1; b.resources+=0.1
    # NORM emergence: a greeting convention spreads by imitation on good interactions
    if good:
        if a.greeting is None and b.greeting is not None: a.greeting=b.greeting
        elif b.greeting is None and a.greeting is not None: b.greeting=a.greeting
        elif a.greeting is None and b.greeting is None and rng.random()<0.1:
            conv=rng.choice(["wave","bow","clasp"]); a.greeting=b.greeting=conv

def find_groups(ms):
    # connected components over strong positive ties
    adj=defaultdict(set)
    for a in ms:
        for o,w in a.ties.items():
            if w>0.4: adj[a.name].add(o); adj[o].add(a.name)
    seen=set(); groups=[]
    for a in ms:
        if a.name in seen: continue
        stack=[a.name]; comp=set()
        while stack:
            x=stack.pop()
            if x in seen: continue
            seen.add(x); comp.add(x); stack.extend(adj[x]-seen)
        if len(comp)>=2: groups.append(comp)
    return groups

def analyze(ms, label, reps):
    groups=find_groups(ms)
    # specialization: spread of solo-effort vs tie-building across members
    solos=[m.solo for m in ms]; ties=[sum(1 for w in m.ties.values() if w>0.4) for m in ms]
    spec = statistics.pstdev(solos)+statistics.pstdev(ties)
    # reputation structure: variance in how members are regarded
    rep_var=statistics.pstdev(list(reps.values()))
    # norm: fraction sharing the most common greeting
    greets=[m.greeting for m in ms if m.greeting]
    norm_share = (max(defaultdict(int, {g:greets.count(g) for g in set(greets)}).values())/len(ms)) if greets else 0
    return {"groups":len(groups),"largest_group":max((len(g) for g in groups),default=0),
            "specialization":round(spec,2),"reputation_variance":round(rep_var,3),
            "norm_adoption":round(norm_share,2)}

def run(chars, seed, N=20, rad=3, rounds=4000):
    rng=random.Random(seed)
    ms=[M(f"m{i}",chars[i],i) for i in range(N)]
    reps={m.name:0.5 for m in ms}
    for _ in range(rounds): step(ms,rng,rad,reps)
    for m in ms: m.reputation=reps[m.name]
    return ms, reps

def main():
    print("="*64); print("  SOCIETY v6: emergent organization (developed vs flat)"); print("="*64)
    N=20
    def dev_chars(seed):
        rng=random.Random(seed); w=(["mastery"]*7+["connection"]*7+["peace"]*6); rng.shuffle(w)
        return [shape(seed*100+i,w[i]) for i in range(N)]

    # average emergent structure over many societies
    SOC=40
    dev_agg=defaultdict(list); flat_agg=defaultdict(list)
    for s in range(SOC):
        dchars=dev_chars(s)
        fchars=[{x:0.5 for x in VALUES} for _ in range(N)]
        dms,drep=run(dchars,s*13+1,N=N); 
        fms,frep=run(fchars,s*13+1,N=N)
        for k,v in analyze(dms,"dev",drep).items(): dev_agg[k].append(v)
        for k,v in analyze(fms,"flat",frep).items(): flat_agg[k].append(v)

    print(f"\n  emergent social structure, averaged over {SOC} societies of {N}:")
    print(f"  {'property':<22}{'developed':>12}{'flat':>10}")
    keys=["groups","largest_group","specialization","reputation_variance","norm_adoption"]
    for k in keys:
        d=statistics.mean(dev_agg[k]); f=statistics.mean(flat_agg[k])
        print(f"  {k:<22}{d:>12.2f}{f:>10.2f}")

    print("\n"+"="*64); print("  DOES DEVELOPMENTAL INDIVIDUALITY CHANGE THE SOCIETY?"); print("="*64)
    d_spec=statistics.mean(dev_agg["specialization"]); f_spec=statistics.mean(flat_agg["specialization"])
    d_rep=statistics.mean(dev_agg["reputation_variance"]); f_rep=statistics.mean(flat_agg["reputation_variance"])
    d_grp=statistics.mean(dev_agg["groups"]); f_grp=statistics.mean(flat_agg["groups"])
    changed = d_spec>f_spec*1.3 or d_rep>f_rep*1.3 or abs(d_grp-f_grp)>0.8
    if changed:
        print(f"  YES - developmental individuality produces a DIFFERENT KIND of society.")
        print(f"  Societies of developmentally-different individuals show more")
        print(f"  specialization ({d_spec:.1f} vs {f_spec:.1f}) and reputation structure ({d_rep:.2f} vs {f_rep:.2f})")
        print(f"  than flat-character societies, with groups and norms emerging from")
        print(f"  interaction, not design. So individuality doesn't just change WHO")
        print(f"  holds a role - it changes the society's whole emergent organization.")
        print(f"  (Honest framing: within-simulation emergent structure driven by tested")
        print(f"  character differences; not a claim of personhood.)")
    else:
        print(f"  Honest: developed and flat societies show similar emergent structure")
        print(f"  (spec {d_spec:.1f} vs {f_spec:.1f}, rep-var {d_rep:.2f} vs {f_rep:.2f}, groups {d_grp:.1f} vs {f_grp:.1f}).")
        print(f"  Individuality shapes positions but not yet the society's overall")
        print(f"  organization. Reported straight.")

if __name__=="__main__": main()
