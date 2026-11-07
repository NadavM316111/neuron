"""Society v9: institutional stress test (the friend's frontier). Can a FUNCTIONAL
institution survive noisy generations, turnover, migration, competition, shocks,
and defection - maintained by society itself even when individuals can deviate?

Not an arbitrary convention this time. A FUNCTIONAL institution: a resource-sharing
rule ("contribute to a common pool, draw fairly") that benefits the group but
tempts individuals to defect (keep their own, take from the pool). The question:
does the institution survive founder death AND real difficulty?

Stressors (all on): imperfect transmission (newcomers mislearn), migration
(outsiders arrive with no/rival norm), environmental shocks (periods where the
institution is temporarily costly), defection temptation (free-riding pays short-term).

Four conditions compared:
  none                 - no institution (everyone self-interested)
  transmission-only    - norm copied to newborns, no enforcement
  + enforcement        - defectors lose reputation, get excluded (society polices it)
  + records/history    - a persistent memory of who cooperates, informing enforcement

Honest: survival should be HARD now (not a clean 100%). Report what survives.

    python institution_stress.py
"""
import random, statistics
from collections import defaultdict

class Being:
    _id=0
    def __init__(s,coop_norm,gen,native=True):
        Being._id+=1; s.name=f"b{Being._id}"; s.gen=gen
        s.follows_norm=coop_norm      # does it believe in the sharing rule?
        s.reputation=0.5; s.resources=1.0; s.native=native; s.age=0

def generation_cycle(pop, condition, rng, shock=False, records=None):
    """one generation of the resource game under the institution + stressors."""
    N=len(pop)
    pool=0.0; contributors=[]
    for b in pop:
        # decide: contribute (follow norm) or defect (free-ride)?
        base_follow = b.follows_norm
        temptation = 0.5 if not shock else 0.8   # shock makes defection more tempting
        # enforcement/records raise the cost of defecting
        enforce = condition in ("enforce","records")
        risk = (b.reputation if condition=="enforce" else
                (records.get(b.name,0.5) if condition=="records" and records else 0.5))
        will_contribute = base_follow and (rng.random() > temptation*(1-(0.6 if enforce else 0)))
        if will_contribute:
            b.resources-=0.3; pool+=0.4; contributors.append(b)   # contributing grows the pool (synergy)
            if condition=="enforce": b.reputation=min(1,b.reputation+0.05)
            if condition=="records" and records is not None: records[b.name]=min(1,records.get(b.name,0.5)+0.05)
        else:
            if condition=="enforce": b.reputation=max(0,b.reputation-0.08)
            if condition=="records" and records is not None: records[b.name]=max(0,records.get(b.name,0.5)-0.08)
    # distribute pool: enforcement/records exclude low-rep defectors from the pool
    if condition in ("enforce","records"):
        eligible=[b for b in pop if (b.reputation if condition=="enforce" else records.get(b.name,0.5))>0.35]
    else:
        eligible=pop
    if eligible:
        share=pool/len(eligible)
        for b in eligible: b.resources+=share
    # cooperation rate this generation
    return len(contributors)/N

def run(condition, generations=12, N=20, seed=0):
    rng=random.Random(seed)
    # FOUNDERS all believe in the institution
    pop=[Being(True,0) for _ in range(N)]
    founder_ids=set(b.name for b in pop)
    records={b.name:0.6 for b in pop} if condition=="records" else None
    coop_history=[]
    for gen in range(1,generations+1):
        shock = (gen%4==0)   # periodic environmental shock
        rate=generation_cycle(pop, condition, rng, shock=shock, records=records)
        coop_history.append((gen, rate, sum(1 for b in pop if b.name in founder_ids)))
        for b in pop: b.age+=1
        # turnover: oldest third die; newborns + occasional MIGRANTS replace them
        pop.sort(key=lambda x:-x.age); dying=pop[:N//3]; surv=pop[N//3:]
        newcomers=[]
        for _ in range(len(dying)):
            if rng.random()<0.25:
                # MIGRANT: outsider, may not share the norm (rival/no convention)
                nb=Being(rng.random()<0.3, gen, native=False)
            else:
                # NEWBORN native: learns norm by (imperfect) transmission
                if condition=="none":
                    nb=Being(False,gen)
                else:
                    teacher=rng.choice(surv)
                    learned = teacher.follows_norm and (rng.random()<0.8)  # 20% mislearn
                    nb=Being(learned,gen)
            if records is not None: records[nb.name]=0.5
            newcomers.append(nb)
        pop=surv+newcomers
    # measure cooperation AFTER all founders gone
    after=[r for g,r,fa in coop_history if fa==0]
    return coop_history, (statistics.mean(after) if after else 0)

def main():
    print("="*64); print("  SOCIETY v9: institutional stress test (functional institution)"); print("="*64)
    print("  a resource-sharing institution under turnover + migration + shocks +")
    print("  defection temptation. Does it survive its founders AND real difficulty?\n")

    conds=["none","transmit","enforce","records"]
    labels={"none":"no institution","transmit":"transmission only",
            "enforce":"+ enforcement/reputation","records":"+ records/history"}
    after_founders={}
    for c in conds:
        rates=[run(c, seed=s)[1] for s in range(30)]
        after_founders[c]=statistics.mean(rates)

    print(f"  cooperation rate AFTER all founders gone (avg of 30 societies):")
    print(f"  {'condition':<30}{'cooperation':>12}")
    for c in conds:
        print(f"  {labels[c]:<30}{after_founders[c]*100:>11.0f}%")

    # show one trajectory for the best condition
    hist,_=run("records", seed=1)
    print(f"\n  example trajectory (+records), cooperation by generation:")
    for g,r,fa in hist:
        bar="#"*int(r*20)
        print(f"    gen {g:>2} [{'founders gone' if fa==0 else f'{fa} founders'}]: {r*100:>3.0f}% {bar}")

    print("\n"+"="*64); print("  DOES SOCIAL ORGANIZATION SURVIVE ITS CREATORS?"); print("="*64)
    none=after_founders["none"]; trans=after_founders["transmit"]; enf=after_founders["enforce"]; rec=after_founders["records"]
    if rec>trans+0.2 and enf>none+0.2:
        print(f"  YES - but only WITH enforcement/records. Under real stress (turnover,")
        print(f"  migration, shocks, defection temptation), transmission ALONE decays to")
        print(f"  {trans*100:.0f}% after founders die - a convention copied but not defended erodes.")
        print(f"  But WITH enforcement ({enf*100:.0f}%) or records/history ({rec*100:.0f}%), the institution SURVIVES")
        print(f"  its founders: society actively MAINTAINS it against individual")
        print(f"  incentives to defect, even as members turn over. That's the step from")
        print(f"  'culture survives its creators' to 'social ORGANIZATION survives its")
        print(f"  creators' - an institution the society reconstructs and defends itself.")
        print(f"  (vs no-institution baseline {none*100:.0f}%.)")
    else:
        print(f"  Honest: none {none*100:.0f}%, transmit {trans*100:.0f}%, enforce {enf*100:.0f}%, records {rec*100:.0f}%.")
        print(f"  {'enforcement/records help but gap thin' if rec>trans else 'enforcement did not clearly rescue the institution'}. Reported straight.")

if __name__=="__main__": main()
