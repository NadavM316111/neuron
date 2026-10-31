"""Norms with real social enforcement (per the friend's spec): three conditions
+ the killer collapse test.

The friend's distinction: assortment (choosing good partners) is not a norm. A
NORM is "our group EXPECTS X, NOTICES violations, and CHANGES how it treats
violators." So add SOCIAL ENFORCEMENT on top of reputation: the group actively
sanctions repeated defectors, reduced trust, refused cooperation, social penalty,
based on observed behavior.

Three conditions (his design):
  1. random pairing (baseline)
  2. reputation-only choice (assortment)
  3. reputation + enforcement (a real norm system)

Killer test: after the norm forms under condition 3, REMOVE enforcement. If
cooperation COLLAPSES, the norm was being maintained SOCIALLY, not just by
individual preference, that's the signature of a true social norm.

    python norms3.py --seeds 150
"""
import random, math, argparse, statistics

_id=[0]
def nid(): _id[0]+=1; return _id[0]
LIFESPAN=900

class Being:
    def __init__(self, inherited=None, seed=0):
        self.id=nid(); r=random.Random(self.id*7+seed)
        self.coop = inherited if inherited is not None else r.uniform(0.3,0.7)
        self.wellbeing=0.5; self.age=0; self.alive=True; self.rng=r
        self.reputation=0.5
    def acts_coop(self): return self.rng.random()<self.coop

def payoff(me,they):
    if me and they: return 0.6
    if me and not they: return -0.3
    if not me and they: return 0.9
    return -0.1

def simulate(seed, mode, enforce_after=None, epochs=6, pop=16, era=300):
    # mode: 'random' | 'reputation' | 'enforcement'
    # enforce_after: if set (epoch), enforcement is REMOVED after that epoch
    #                (killer test) even in 'enforcement' mode
    rng=random.Random(seed)
    P=[Being(seed=seed+i) for i in range(pop)]
    for b in P: b.age=rng.randint(0,LIFESPAN-150)
    for ep in range(epochs):
        enforcing = (mode=='enforcement') and (enforce_after is None or ep<=enforce_after)
        for day in range(era):
            A=[b for b in P if b.alive]
            if len(A)<2: break
            rng.shuffle(A); used=set()
            for b in A:
                if b.id in used: continue
                cands=[o for o in A if o.id!=b.id and o.id not in used]
                if not cands: continue
                if mode in ('reputation','enforcement'):
                    # ENFORCEMENT: refuse to partner with low-rep (known) defectors
                    if enforcing:
                        ok=[o for o in cands if o.reputation>0.3]   # exclude violators
                        pool = ok if ok else cands
                        partner=max(pool,key=lambda o:o.reputation+b.rng.uniform(-0.05,0.05))
                    else:
                        partner=max(cands,key=lambda o:o.reputation+b.rng.uniform(-0.1,0.1))
                else:
                    partner=b.rng.choice(cands)
                used.add(b.id); used.add(partner.id)
                bc=b.acts_coop(); pc=partner.acts_coop()
                bp=payoff(bc,pc); pp=payoff(pc,bc)
                # SOCIAL PENALTY: if enforcing, a defector is socially sanctioned
                if enforcing:
                    if not bc: bp-=0.6
                    if not pc: pp-=0.6
                b.wellbeing=max(0,min(1,0.95*b.wellbeing+0.08*bp))
                partner.wellbeing=max(0,min(1,0.95*partner.wellbeing+0.08*pp))
                b.reputation=max(0,min(1,0.88*b.reputation+(0.12 if bc else -0.12)))
                partner.reputation=max(0,min(1,0.88*partner.reputation+(0.12 if pc else -0.12)))
                b.coop=min(1,max(0,b.coop+(0.025 if bp>0 else -0.025)))
                partner.coop=min(1,max(0,partner.coop+(0.025 if pp>0 else -0.025)))
            for b in A:
                b.age+=1
                if b.age>LIFESPAN: b.alive=False
            dead=[b for b in P if not b.alive]
            if dead:
                P=[b for b in P if b.alive]
                while len(P)<pop and len(P)>=2:
                    par=sorted(P,key=lambda b:b.wellbeing,reverse=True)[:2]
                    P.append(Being(inherited=(par[0].coop+par[1].coop)/2,seed=seed+nid()))
                while len(P)<2: P.append(Being(seed=seed+nid()))
    A=[b for b in P if b.alive]
    return sum(b.coop for b in A)/len(A) if A else 0

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seeds",type=int,default=150); a=ap.parse_args()
    print("="*64); print("  NORMS: 3 conditions + killer collapse test (friend's design)"); print("="*64)

    rnd=[simulate(s*53+9,'random') for s in range(a.seeds)]
    rep=[simulate(s*53+9,'reputation') for s in range(a.seeds)]
    enf=[simulate(s*53+9,'enforcement') for s in range(a.seeds)]
    enf_strong=sum(1 for c in enf if c>0.65)
    print(f"\n  cooperation across {a.seeds} seeds:")
    print(f"    1. random pairing            -> {statistics.mean(rnd):.2f}")
    print(f"    2. reputation choice only     -> {statistics.mean(rep):.2f}")
    print(f"    3. reputation + enforcement   -> {statistics.mean(enf):.2f}")
    print(f"    strong norm (coop>0.65) under enforcement: {enf_strong}/{a.seeds} ({enf_strong*100//a.seeds}%)")

    # KILLER TEST: norm forms under enforcement, then enforcement REMOVED midway
    collapsed=[simulate(s*53+9,'enforcement',enforce_after=2) for s in range(a.seeds)]
    print(f"\n  KILLER TEST (remove enforcement after norm forms, at epoch 2):")
    print(f"    cooperation after enforcement removed -> {statistics.mean(collapsed):.2f}")
    print(f"    (vs {statistics.mean(enf):.2f} with enforcement kept)")
    collapse_drop = statistics.mean(enf) - statistics.mean(collapsed)
    print(f"    collapse when enforcement removed: {collapse_drop:+.2f}")

    print("\n"+"="*64); print("  WHAT THIS ESTABLISHES"); print("="*64)
    norm_real = statistics.mean(enf) > statistics.mean(rep)+0.1 and statistics.mean(enf)>0.65
    socially_maintained = collapse_drop > 0.1
    if norm_real and socially_maintained:
        print(f"  A REAL SOCIAL NORM. Reputation choice alone gave only modest cooperation")
        print(f"  ({statistics.mean(rep):.2f}), but adding SOCIAL ENFORCEMENT - the group refusing and")
        print(f"  penalizing violators - drove cooperation to {statistics.mean(enf):.2f}, a strong norm in")
        print(f"  {enf_strong*100//a.seeds}% of worlds. And the killer test is decisive: when enforcement")
        print(f"  was REMOVED after the norm formed, cooperation COLLAPSED ({statistics.mean(enf):.2f} -> {statistics.mean(collapsed):.2f}),")
        print(f"  proving the norm was maintained SOCIALLY by enforcement, not just by")
        print(f"  individual preference. This is a genuine norm: 'we expect each other to")
        print(f"  cooperate, we notice violations, we treat violators differently.' A")
        print(f"  society with an enforced moral order.")
    else:
        print(f"  random {statistics.mean(rnd):.2f} / reputation {statistics.mean(rep):.2f} / enforcement {statistics.mean(enf):.2f};")
        print(f"  collapse on removal {collapse_drop:+.2f}. Honest: "
              f"{'enforcement raised cooperation but collapse test weak' if norm_real else 'enforcement effect modest'}. Reported straight.")

if __name__=="__main__": main()
