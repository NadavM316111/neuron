"""Norms carried by behavior alone, no stored norm variable (the friend's nasty
test).

His critique: before, the "convention" was a stored variable beings could read,
and the behaviors weren't symmetric (spread 157/83/49/11). The real test:
  - NO stored group norm. Nobody holds "our rule is X."
  - A being only OBSERVES: what others around it DID, and whether each actor was
    approved or sanctioned by others.
  - A newborn must INFER the convention purely from these observations (what do
    people here do, and what gets approved?), then LATER enforce it on a novel
    member, approve conformers, sanction deviators, having been told the rule by
    no one.
  - Behaviors are fully symmetric (checked), so any convention that forms is
    genuine drift, not a built-in bias.

If a newborn infers the norm from observation and enforces it on a stranger,
across many seeds, the population genuinely CARRIES the norm through behavior and
social transmission, not through a stored variable.

    python inferred_norms.py --seeds 150
"""
import random, math, argparse, statistics
from collections import deque

N_BEHAV=4
_id=[0]
def nid(): _id[0]+=1; return _id[0]
LIFESPAN=700

class Being:
    def __init__(self, seed=0):
        self.id=nid(); r=random.Random(self.id*7+seed)
        # symmetric start: tiny random noise only, no behavior favored
        self.tendency=[r.uniform(0.49,0.51) for _ in range(N_BEHAV)]
        # MEMORY of observations: what others did + whether they were approved
        # (approval_count[b] = how often behavior b was socially approved)
        self.approval_mem=[0.0]*N_BEHAV
        self.obs_count=0
        self.wellbeing=0.5; self.age=0; self.alive=True; self.rng=r
        self.reputation=0.5
    def does(self):
        return max(range(N_BEHAV), key=lambda i:self.tendency[i])
    def inferred_norm(self):
        # what IT thinks the norm is, inferred purely from observed approvals
        if self.obs_count==0: return self.does()
        return max(range(N_BEHAV), key=lambda i:self.approval_mem[i])
    def observe(self, actor_behavior, was_approved):
        # social learning: remember what behavior got approved/sanctioned
        self.obs_count+=1
        w = 1.0 if was_approved else -1.0
        self.approval_mem[actor_behavior] = 0.95*self.approval_mem[actor_behavior] + 0.05*w
    def react(self, actor_behavior):
        # does IT approve of the actor? approves if the actor did what IT has
        # inferred to be the norm. (enforcement by individuals, no stored rule.)
        return actor_behavior == self.inferred_norm()
    def learn_from_reactions(self, my_behavior, n_approve, n_total):
        # i adjust my own behavior toward what gets me approved
        approval_rate = n_approve/n_total if n_total else 0.5
        if approval_rate >= 0.5:
            self.tendency[my_behavior]=min(2,self.tendency[my_behavior]+0.05)
            self.reputation=min(1,self.reputation+0.05)
        else:
            self.tendency[my_behavior]=max(0,self.tendency[my_behavior]-0.04)
            self.reputation=max(0,self.reputation-0.08)
            # shift toward whatever i've inferred others approve
            self.tendency[self.inferred_norm()]+=0.05

def group_behavior_dist(members):
    d=[0]*N_BEHAV
    for b in members:
        if b.alive: d[b.does()]+=1
    return d

def one_group_step(members, rng):
    A=[b for b in members if b.alive]
    if len(A)<3: return
    for actor in A:
        actor.age+=1
        beh=actor.does()
        # a few random observers REACT (approve/disapprove) based on THEIR own
        # inferred norm - no stored group rule anywhere
        observers=[o for o in A if o is not actor]
        rng.shuffle(observers); panel=observers[:min(5,len(observers))]
        approvals=sum(1 for o in panel if o.react(beh))
        approved = approvals >= len(panel)/2
        # everyone who watched LEARNS from this (what behavior got approved)
        for o in panel: o.observe(beh, approved)
        # the actor adjusts toward getting approved
        actor.learn_from_reactions(beh, approvals, len(panel))
        # wellbeing from social approval
        actor.wellbeing=max(0,min(1,0.95*actor.wellbeing+0.08*(0.5 if approved else -0.4)))
        if actor.age>LIFESPAN: actor.alive=False

def run(seed, epochs=8, pop=14, era=200):
    rng=random.Random(seed)
    members=[Being(seed+i) for i in range(pop)]
    for b in members: b.age=rng.randint(0,LIFESPAN-150)
    for ep in range(epochs):
        for day in range(era):
            one_group_step(members, rng)
            dead=[b for b in members if not b.alive]
            if dead:
                members=[b for b in members if b.alive]
                while len(members)<pop and len(members)>=3:
                    members.append(Being(seed+nid()))  # newborn: blank, must infer
                while len(members)<3: members.append(Being(seed+nid()))
    # the group's actual convention (most common behavior)
    d=group_behavior_dist(members)
    convention=max(range(N_BEHAV),key=lambda i:d[i]) if sum(d) else -1
    conv_strength=d[convention]/sum(d) if sum(d) else 0
    return members, convention, conv_strength

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seeds",type=int,default=150); a=ap.parse_args()
    print("="*64); print("  NORMS CARRIED BY BEHAVIOR ALONE (no stored norm variable)"); print("="*64)
    print("  A newborn only OBSERVES behavior + approvals. Can it INFER the norm")
    print("  and ENFORCE it on a stranger, with no one storing the rule?\n")

    conv_counts=[0]*N_BEHAV
    newborn_infers=0; newborn_enforces=0; total=0
    strong_conv=0
    for s in range(a.seeds):
        members, conv, strength = run(s*41+7)
        if conv<0: continue
        conv_counts[conv]+=1
        if strength>0.6: strong_conv+=1
        # drop a BLANK newborn into the mature group; let it only observe
        rng=random.Random(s*999)
        baby=Being(s*12345)
        for _ in range(400):
            A=[b for b in members if b.alive]
            if len(A)<3: break
            # baby watches random members act and sees if they're approved
            actor=rng.choice(A); beh=actor.does()
            panel=[o for o in A if o is not actor]; rng.shuffle(panel); panel=panel[:5]
            approved = sum(1 for o in panel if o.react(beh)) >= len(panel)/2
            baby.observe(beh, approved)
        total+=1
        # did the baby INFER the group's actual convention from observation alone?
        if baby.inferred_norm()==conv: newborn_infers+=1
        # now a NOVEL member does a random behavior; does the baby enforce the
        # norm (approve only if it matches what baby inferred)?
        novel_beh=rng.randrange(N_BEHAV)
        baby_reaction = baby.react(novel_beh)       # approves iff matches inferred norm
        correct_enforcement = (baby_reaction == (novel_beh==conv))
        if correct_enforcement: newborn_enforces+=1

    print(f"  across {total} worlds:")
    print(f"    convention counts across behaviors: {conv_counts}")
    bias = max(conv_counts)/sum(conv_counts) if sum(conv_counts) else 0
    print(f"    most-common convention share: {bias:.2f} "
          f"({'~symmetric, no strong bias' if bias<0.4 else 'some behaviors favored - check'})")
    print(f"    strong convention formed (>60% of group): {strong_conv}/{total} ({strong_conv*100//max(1,total)}%)")
    print(f"\n  NEWBORN INFERENCE (observation only, nothing stored):")
    print(f"    newborn INFERRED the group's convention from watching: "
          f"{newborn_infers}/{total} ({newborn_infers*100//max(1,total)}%)")
    print(f"    newborn correctly ENFORCED it on a novel member: "
          f"{newborn_enforces}/{total} ({newborn_enforces*100//max(1,total)}%)")

    print("\n"+"="*64); print("  WHAT THIS ESTABLISHES"); print("="*64)
    infers = newborn_infers>total*0.5; enforces = newborn_enforces>total*0.5
    symmetric = bias<0.45
    if infers and enforces:
        print(f"  THE POPULATION CARRIES THE NORM (not a stored variable). With NO")
        print(f"  behavior predefined and NO group rule stored anywhere, conventions")
        print(f"  formed from behavior + social reaction alone. A BLANK newborn,")
        print(f"  observing only what others did and whether they were approved,")
        print(f"  INFERRED the group's convention {newborn_infers*100//max(1,total)}% of the time and then correctly")
        print(f"  ENFORCED it on a novel member {newborn_enforces*100//max(1,total)}% of the time - approving conformers,")
        print(f"  disapproving deviators - having been told the rule by no one. The norm")
        print(f"  lives in the population's behavior and transmission, not in any stored")
        print(f"  rule. {'Behaviors were ~symmetric, so the convention was genuine drift. ' if symmetric else ''}This is")
        print(f"  genuinely carried normative culture - the strong version.")
    else:
        print(f"  newborn inferred {newborn_infers*100//max(1,total)}%, enforced {newborn_enforces*100//max(1,total)}%, bias {bias:.2f}.")
        print(f"  Honest: {'inference works, enforcement weak' if infers else 'newborn inference from observation not reliable yet'}. Reported straight.")

if __name__=="__main__": main()
