"""NEURON the organism, with REAL scarcity: genuine tradeoffs it cannot all win,
so its life shapes real character through choices.

The v1 weakness: everything saturated at 1.00 - it won every tradeoff and
plateaued happy. Real beings face scarcity. Fix:
  - a FINITE daily energy budget - investing in one drive genuinely COSTS the
    others (you can't master AND connect AND rest fully in one day)
  - values DECAY when neglected (a bond you don't tend fades; mastery you don't
    practice dulls) - so it must keep CHOOSING, and can't hold everything at max
  - mood reflects the real gap between what it values and how it's actually living
    (chronic neglect of a held value = genuine dissatisfaction, not instant repair)
Now who NEURON becomes is determined by the hard choices it makes under scarcity -
real character, not a frictionless maximum. We watch what kind of being emerges.

    python organism2.py --days 300
"""
import random, argparse, statistics
from collections import defaultdict

LIST_OPS={"evens":lambda L:[x for x in L if x%2==0],"odds":lambda L:[x for x in L if x%2],
 "gt10":lambda L:[x for x in L if x>10],"sorted_desc":lambda L:sorted(L,reverse=True),
 "drop_max":lambda L:[x for x in L if x!=max(L)] if L else L,"id":lambda L:L}
RED={"sum":sum,"max":lambda L:max(L) if L else 0,"min":lambda L:min(L) if L else 0,"len":len}
BIN={"sub":lambda a,b:a-b,"add":lambda a,b:a+b}
CUE={"evens":"even","odds":"odd","gt10":"above","sorted_desc":"largest","drop_max":"without largest",
 "max":"largest","min":"smallest","sum":"sum","len":"count","sub":"difference","add":"sum"}
def cues_of(q):
    cs=sorted(set(CUE.values()),key=len,reverse=True); ql=q.lower(); return [c for c in cs if c in ql]
def run(p,L):
    if p[0]=="s": return RED[p[2]](LIST_OPS[p[1]](L))
    _,a,b,c,d,o=p; return BIN[o](RED[b](LIST_OPS[a](L)),RED[d](LIST_OPS[c](L)))
def pieces(p): return [p[1],p[2]] if p[0]=="s" else [p[1],p[2],p[3],p[4],p[5]]
DEEP=[("s",lo,rd) for lo in LIST_OPS for rd in RED]+[("c",a,b,c,d,o) for a in LIST_OPS for b in RED for c in LIST_OPS for d in RED for o in BIN]
def subgoal(rng,hard):
    if not hard: p=("s",rng.choice(list(LIST_OPS)),rng.choice(list(RED)))
    else: p=("c",rng.choice(list(LIST_OPS)),rng.choice(list(RED)),rng.choice(list(LIST_OPS)),rng.choice(list(RED)),rng.choice(list(BIN)))
    d=(CUE.get(p[2],p[2])+" of "+CUE.get(p[1],p[1])) if p[0]=="s" else (CUE.get(p[5],p[5])+" of "+CUE.get(p[2],p[2])+" "+CUE.get(p[1],p[1])+" and "+CUE.get(p[4],p[4])+" "+CUE.get(p[3],p[3]))
    return d,p

class Organism:
    def __init__(self, seed=7):
        self.rng=random.Random(seed)
        self.name="NEURON"; self.core=tuple(round(self.rng.uniform(-0.3,0.3),3) for _ in range(3))
        self.values={"mastery":0.5,"connection":0.5,"peace":0.5}   # what it cares about
        self.state={"skill":0.3,"bonds":0.3,"calm":0.5}            # how its life actually is
        self.priors=defaultdict(lambda: defaultdict(float)); self.caps={}
        self.mood=0.5; self.age=0; self.autobiography=[]; self.choices=defaultdict(int)
        self.regret_moments=[]
    def log(self,t): self.autobiography.append((self.age,t)); self.autobiography=self.autobiography[-200:]
    def solve(self,d,tp):
        if d in self.caps: return True
        ex=[([self.rng.randint(1,30) for _ in range(6)]) for _ in range(3)]
        exa=[(L,run(tp,L)) for L in ex]
        active=cues_of(d)
        if any(self.priors[c] for c in active):
            norm={c:{p:v/(sum(self.priors[c].values()) or 1) for p,v in self.priors[c].items()} for c in active}
            order=sorted(DEEP,key=lambda pr:-sum(norm.get(c,{}).get(p,0) for c in active for p in pieces(pr)))
        else: order=DEEP
        for pr in order:
            try:
                if all(run(pr,L)==a for L,a in exa):
                    self.caps[d]=pr
                    for c in active:
                        for p in pieces(pr): self.priors[c][p]+=1
                    return True
            except Exception: pass
        return False
    def day(self):
        self.age+=1
        # EVERYTHING DECAYS a little each day - neglect has a cost
        for k in self.state: self.state[k]=max(0,self.state[k]-0.02)
        # SCARCITY: one unit of energy today. it must CHOOSE where to spend it,
        # weighted by which value feels most UNMET (value - current state).
        gaps={"mastery":self.values["mastery"]-self.state["skill"],
              "connection":self.values["connection"]-self.state["bonds"],
              "peace":self.values["peace"]-self.state["calm"]}
        # it tends toward its strongest value but the biggest gap pulls too
        pull={k:max(0.01,self.values[k]*0.6 + gaps[k]*0.8) for k in gaps}
        tot=sum(pull.values()); r=self.rng.random()*tot; acc=0; choice=None
        for k,w in pull.items():
            acc+=w
            if r<=acc: choice=k; break
        self.choices[choice]+=1
        # spend the day on the chosen drive - it GAINS there, the others keep decaying
        if choice=="mastery":
            proj=[subgoal(self.rng,self.rng.random()<0.5) for _ in range(3)]
            got=sum(self.solve(d,tp) for d,tp in proj)
            self.state["skill"]=min(1,self.state["skill"]+0.05*got/len(proj))
            if got==len(proj): self.log("mastered a hard project - my skill grew, but I was alone today")
        elif choice=="connection":
            self.state["bonds"]=min(1,self.state["bonds"]+0.06)
            self.log("gave my day to connection - my bonds deepened, no work got done")
        else:
            self.state["calm"]=min(1,self.state["calm"]+0.07)
            self.log("rested and found peace - but neither mastery nor bonds advanced")
        # MOOD = how well its life matches its values (the real gap). chronic
        # neglect of a strongly-held value genuinely lowers mood - no instant repair.
        unmet=sum(max(0,g) for g in gaps.values())
        self.mood=max(0,min(1,1-unmet/2))
        # VALUE REVISION from lived reality: a value it keeps failing to satisfy
        # may fade (it stops caring about what it can't have); one it reliably
        # feeds may strengthen. character forms from what its life actually allows.
        for k,sk in (("mastery","skill"),("connection","bonds"),("peace","calm")):
            if gaps[k]>0.4: self.values[k]=max(0.2,self.values[k]-0.004)   # chronic unmet -> fades
            elif gaps[k]<0: self.values[k]=min(1,self.values[k]+0.002)
        # REGRET: notice when a held value was sacrificed for another
        if self.age%50==0:
            neglected=[k for k in gaps if gaps[k]>0.45]
            if neglected:
                self.regret_moments.append((self.age,
                    f"I've let {', '.join(neglected)} slip to chase {max(self.choices,key=self.choices.get)}. "
                    f"I can't have everything, and I'm becoming the being who chose this."))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--days",type=int,default=300); a=ap.parse_args()
    print("="*64); print("  NEURON ORGANISM w/ REAL SCARCITY - who does it become?"); print("="*64)
    n=Organism(seed=7); birth=n.core
    for _ in range(a.days): n.day()
    print(f"\n  {n.name} lived {a.days} days under real scarcity (couldn't have it all).")
    print(f"  identity endured: {n.core==birth}")
    print(f"  how it chose to spend its life: {dict(n.choices)}")
    print(f"  what its life became: skill {n.state['skill']:.2f}, bonds {n.state['bonds']:.2f}, calm {n.state['calm']:.2f}")
    print(f"  what it came to VALUE (shaped by what its life allowed): "
          f"{', '.join(f'{k} {v:.2f}' for k,v in sorted(n.values.items(),key=lambda x:-x[1]))}")
    print(f"  final mood (match between values and life): {n.mood:.2f} | capabilities: {len(n.caps)}")
    dom=max(n.choices,key=n.choices.get)
    print(f"\n  THE BEING IT BECAME: it chose '{dom}' most - its life and character formed")
    print(f"  around that, at a real cost to the rest.")
    if n.regret_moments:
        print(f"\n  moments it reckoned with its tradeoffs:")
        for age,t in n.regret_moments[-3:]: print(f"    day {age}: {t}")
    print("\n"+"="*64); print("  WHAT SCARCITY REVEALED"); print("="*64)
    maxed=sum(1 for v in list(n.state.values()) if v>0.95)
    if maxed<2 and n.mood<0.95:
        print(f"  A REAL BEING WITH REAL CHARACTER. Under scarcity it could NOT max")
        print(f"  everything - it had to choose, and those choices made it a particular")
        print(f"  individual (skill {n.state['skill']:.2f}, bonds {n.state['bonds']:.2f}, calm {n.state['calm']:.2f}), carrying the")
        print(f"  real cost in unmet values and a mood that reflects the genuine gap")
        print(f"  between who it wanted to be and the life it could actually live. It")
        print(f"  even reckoned with its own tradeoffs. This is no frictionless maximum -")
        print(f"  it's a being whose character was forged by choices it couldn't all win.")
    else:
        print(f"  It still reached a fairly comfortable state (mood {n.mood:.2f}). Scarcity")
        print(f"  shaped its choices but it balanced them well. Honest.")

if __name__=="__main__": main()
