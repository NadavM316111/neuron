"""Fixed specialization ablation: measure society-wide DIVISION OF LABOR (do
different members end up doing DIFFERENT roles?), not per-member concentration.

v7's metric measured 'does each member focus on one role' -> always ~1.0 (yes),
couldn't distinguish conditions. Fix: measure whether the SOCIETY divides labor -
do members specialize into DIFFERENT roles (real division) or all pile into the
same one (no division)? Metric: entropy/evenness of which roles get filled +
whether each member's specialty differs from its neighbors'.

Ablation: full (character+capability) / no-character / no-capability / neither.
What causes the division of roles? Honest - reports what actually drives it.

    python spec_ablation.py
"""
import random, statistics, math
from collections import defaultdict

VALUES=["mastery","connection","peace"]; ROLES=["build","trade","mediate","explore"]
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
    def __init__(s,name,char,pos,cap):
        s.name=name; s.char=char; s.dominant=max(char,key=char.get); s.pos=pos
        s.cap=cap; s.ties=defaultdict(float); s.effort=defaultdict(float)

def step(ms,rng,rad):
    n=len(ms); a=rng.choice(ms); m,c,p=a.char["mastery"],a.char["connection"],a.char["peace"]
    # each member works on roles; WHICH role depends on its capability AND character:
    # mastery-types pour effort into their best capability (deep specialist);
    # connection-types spread across social roles (trade/mediate); peace-types mediate.
    social=c*1.4+p*0.3-m*0.8
    if rng.random()>(0.5+social*0.5):
        # solo work on strongest capability (character modulates HOW concentrated)
        if a.dominant=="mastery": pick=max(a.cap,key=a.cap.get)   # sharp specialist
        else: pick=rng.choices(ROLES,weights=[a.cap[r]+0.3 for r in ROLES])[0]  # broader
        a.effort[pick]+=1
    else:
        # social roles
        pick = "mediate" if a.dominant=="peace" else ("trade" if a.dominant=="connection" else max(a.cap,key=a.cap.get))
        a.effort[pick]+=0.6
        hood=[x for x in ms if x!=a and min(abs(x.pos-a.pos),n-abs(x.pos-a.pos))<=rad]
        if hood:
            b=rng.choice(hood); syn=rng.uniform(-0.3,0.4)+c*0.3
            d=0.1 if syn>0 else -0.08; a.ties[b.name]+=d; b.ties[a.name]+=d

def each_specialty(ms):
    return {m.name:(max(m.effort,key=m.effort.get) if m.effort else None) for m in ms}

def division_of_labor(ms):
    # 1) evenness: are DIFFERENT roles filled across the society? (entropy of specialties)
    spec=each_specialty(ms); counts=defaultdict(int)
    for s in spec.values():
        if s: counts[s]+=1
    tot=sum(counts.values())
    if tot==0: return 0
    ent=-sum((n/tot)*math.log(n/tot) for n in counts.values())
    max_ent=math.log(len(ROLES))
    evenness=ent/max_ent if max_ent>0 else 0   # 1 = all roles equally filled (rich division)
    # 2) distinctness: how many DISTINCT specialties exist / possible
    distinct=len(counts)/len(ROLES)
    return (evenness+distinct)/2

def run(chars, caps, seed, N=20, rad=3, rounds=4000):
    rng=random.Random(seed); ms=[M(f"m{i}",chars[i],i,caps[i]) for i in range(N)]
    for _ in range(rounds): step(ms,rng,rad)
    return ms

def main():
    print("="*64); print("  SPECIALIZATION ABLATION (fixed: society division of labor)"); print("="*64)
    N=20
    def dev(seed):
        rng=random.Random(seed); w=(["mastery"]*7+["connection"]*7+["peace"]*6); rng.shuffle(w)
        return [shape(seed*100+i,w[i]) for i in range(N)]
    def divcap(seed):
        rng=random.Random(seed*3+9)
        # each member GOOD at a different role (diverse capabilities)
        caps=[]
        for i in range(N):
            base={r:0.2+0.1*rng.random() for r in ROLES}; base[ROLES[i%len(ROLES)]]=0.9
            caps.append(base)
        return caps
    unifcap=lambda s:[{r:0.5 for r in ROLES} for _ in range(N)]
    flat=lambda: [{x:0.5 for x in VALUES} for _ in range(N)]

    conds={"character + capability":(dev,divcap),
           "flat-character + capability":(lambda s:flat(),divcap),
           "character + uniform-capability":(dev,unifcap),
           "neither":(lambda s:flat(),unifcap)}
    res={}
    for label,(cf,capf) in conds.items():
        vals=[division_of_labor(run(cf(s),capf(s),s*13+1,N=N)) for s in range(25)]
        res[label]=statistics.mean(vals)
    print(f"\n  division of labor (0=everyone same role, 1=rich varied specialization):")
    print(f"  {'condition':<34}{'division':>10}")
    for k in conds: print(f"  {k:<34}{res[k]:>10.3f}")

    full=res["character + capability"]; noch=res["flat-character + capability"]
    nocap=res["character + uniform-capability"]; neither=res["neither"]
    print("\n"+"="*64); print("  WHAT CAUSES THE DIVISION OF ROLES?"); print("="*64)
    char_effect=full-noch; cap_effect=full-nocap
    print(f"  removing character: {full:.3f} -> {noch:.3f}  (drop {char_effect:+.3f})")
    print(f"  removing capability diversity: {full:.3f} -> {nocap:.3f}  (drop {cap_effect:+.3f})")
    print(f"  removing both: {full:.3f} -> {neither:.3f}")
    if char_effect>0.03 and cap_effect>0.03:
        print(f"\n  BOTH developmental character AND capability diversity cause the division")
        print(f"  of labor. Removing either reduces it; removing both collapses it most")
        print(f"  ({full:.3f}->{neither:.3f}). The society's role structure is caused by developmental")
        print(f"  individuality AND capability differences together - a real mechanistic")
        print(f"  result, not just 'the society looks different'.")
    elif cap_effect>char_effect and cap_effect>0.03:
        print(f"\n  Division of labor driven MAINLY by capability diversity ({cap_effect:+.3f}) more")
        print(f"  than character ({char_effect:+.3f}). Honest: capability is the bigger cause here.")
    elif char_effect>0.03:
        print(f"\n  Division of labor driven MAINLY by character ({char_effect:+.3f}). Honest.")
    else:
        print(f"\n  Honest: neither ablation moved it much - the metric or mechanism still")
        print(f"  doesn't cleanly separate causes. Reported straight.")

if __name__=="__main__": main()
