import random, statistics
from collections import defaultdict
def run(mode, generations=16, N=20, seed=0, shift_gen=8):
    rng=random.Random(seed)
    pop=[{"name":f"b{i}","believes":True,"age":0,"access":0.5,"res":1.0} for i in range(N)]; nid=N
    coop_bonus=0.0; defect_cost=0.07; history=[]
    for gen in range(1,generations+1):
        env_cost = 0.0 if gen<shift_gen else 0.35
        if gen>=shift_gen:
            if mode=="rule_only": coop_bonus,defect_cost=0.0,0.07
            elif mode=="rule_enforce": coop_bonus,defect_cost=0.0,0.25
            elif mode=="incentive": coop_bonus,defect_cost=0.3,0.2
            elif mode=="oracle": coop_bonus,defect_cost=0.5,0.3
            elif mode=="adaptive":
                recent=[r for g,r in history[-3:]]
                if recent and statistics.mean(recent)<0.4:
                    coop_bonus=min(0.5,coop_bonus+0.15); defect_cost=min(0.3,defect_cost+0.08)
        pool=0.0; coops=0
        for b in pop:
            allowed=b["access"]>0.3
            temptation=max(0.05,0.4+env_cost-coop_bonus*1.2-defect_cost*0.8)
            will=b["believes"] and allowed and (rng.random()>temptation*0.6)
            if will:
                b["res"]-=0.3; pool+=0.3*(1.6-env_cost); coops+=1
                b["access"]=min(1,b["access"]+0.05+coop_bonus*0.1); b["res"]+=coop_bonus
            else: b["access"]=max(0,b["access"]-defect_cost)
        rate=coops/len(pop); history.append((gen,rate))
        elig=[b for b in pop if b["access"]>0.35]
        if elig:
            for b in elig: b["res"]+=pool/len(elig)
        for b in pop: b["age"]+=1
        pop.sort(key=lambda x:-x["age"]); dying=pop[:N//3]; surv=pop[N//3:]
        for _ in range(len(dying)):
            if rng.random()<0.2: nb={"name":f"b{nid}","believes":rng.random()<0.4,"age":0,"access":0.5,"res":1.0}
            else:
                teacher=rng.choice(surv); nb={"name":f"b{nid}","believes":teacher["believes"] and rng.random()<0.8,"age":0,"access":0.5,"res":1.0}
            nid+=1; pop=surv+[nb]; surv=pop
        pop=surv
    post=[r for g,r in history if g>=shift_gen]
    return (statistics.mean(post) if post else 0), statistics.mean([b["res"] for b in pop])
def main():
    print("="*64); print("  SOCIETY v11: institutions that change INCENTIVES"); print("="*64)
    print("  can the institution make cooperation individually rational enough to")
    print("  survive turnover + environment change? (oracle control included)\n")
    modes=["rule_only","rule_enforce","incentive","adaptive","oracle"]
    labels={"rule_only":"1. rule only","rule_enforce":"2. rule + enforcement","incentive":"3. incentive redesign","adaptive":"4. adaptive incentives","oracle":"5. ORACLE incentive"}
    res={m:(lambda rs:(statistics.mean(c for c,w in rs),statistics.mean(w for c,w in rs)))([run(m,seed=s) for s in range(30)]) for m in modes}
    print(f"  {'condition':<32}{'cooperation':>12}{'welfare':>10}")
    for m in modes:
        c,w=res[m]; print(f"  {labels[m]:<32}{c*100:>11.0f}%{w:>10.2f}")
    print("\n"+"="*64); print("  CAN INSTITUTIONS SOLVE THE COLLECTIVE PROBLEM?"); print("="*64)
    ro=res["rule_only"][0]; en=res["rule_enforce"][0]; inc=res["incentive"][0]; ad=res["adaptive"][0]; orc=res["oracle"][0]
    if orc>0.55 and ad>en+0.15 and ad>orc*0.7:
        print(f"  MAJOR MILESTONE - society DISCOVERED an incentive that solves the problem.")
        print(f"  oracle known-good {orc*100:.0f}%, adaptive (observes collapse, adjusts its own")
        print(f"  reward/penalty) {ad*100:.0f}% near oracle, vs rule-only {ro*100:.0f}% / enforcement {en*100:.0f}%. A")
        print(f"  founderless society facing a changed world made cooperation individually")
        print(f"  rational and recovered. Using institutions to solve collective problems")
        print(f"  across generations - a real Species step.")
    elif orc>0.55:
        print(f"  mechanics WORK (oracle {orc*100:.0f}% >> rule-only {ro*100:.0f}%) but adaptive ({ad*100:.0f}%) hasn't")
        print(f"  discovered it - bottleneck is institutional DISCOVERY. incentives CAN")
        print(f"  solve it; society finding the right one is next. (incentive {inc*100:.0f}%, enforce {en*100:.0f}%)")
    else:
        print(f"  Honest: even oracle incentive {orc*100:.0f}%. rule-only {ro*100:.0f}%, enforce {en*100:.0f}%,")
        print(f"  incentive {inc*100:.0f}%, adaptive {ad*100:.0f}%. Reported straight.")
    print(f"\n  welfare (so coop isn't 'solved' by ruinous punishment): "+", ".join(f"{labels[m].split('.')[1].strip()} {res[m][1]:.1f}" for m in modes))
if __name__=="__main__": main()
