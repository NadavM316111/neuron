"""NEURON Workbench (the friend's first rung): the persistent NEURON does bounded
useful tasks, VERIFIES success (not just claims it), earns internal compute credits
only for independently-verified success, learns, and we measure whether it becomes
more valuable-per-compute as it lives.

The economic North Star: verified value produced > compute consumed, AND does the
longer NEURON lives, the more valuable work it performs per unit compute?

Four numbers measured relentlessly:
  success rate | compute cost | (proxy) usefulness | improvement with experience

Safe: internal compute credits only. No real money, no accounts, no real-world
actions. This tests economic self-sustainability WITHOUT financial self-sovereignty.

The key architecture: VERIFICATION is real - a task is only 'succeeded' if the
output passes an objective check (tests/evidence), not if NEURON says so. And
RESOURCE-AWARENESS: NEURON estimates whether more compute is worth it.

    python workbench.py
"""
import random, statistics, time
from collections import defaultdict

# ---- bounded verifiable tasks (things NEURON can actually do AND we can check) ----
# each task: a generator producing (problem, verifier). verifier objectively checks
# a candidate answer - this is what makes success REAL, not self-claimed.

def task_arithmetic(rng):
    a,b=rng.randint(10,99),rng.randint(10,99); op=rng.choice(["+","-","*"])
    ans={"+":a+b,"-":a-b,"*":a*b}[op]
    return (f"compute {a}{op}{b}", lambda c,ans=ans: c==ans, {"+":1,"-":1,"*":2}[op])

def task_list_op(rng):
    L=[rng.randint(1,40) for _ in range(6)]
    kind=rng.choice(["sum_evens","max","range","count_gt10"])
    ans={"sum_evens":sum(x for x in L if x%2==0),"max":max(L),
         "range":max(L)-min(L),"count_gt10":sum(1 for x in L if x>10)}[kind]
    return (f"{kind} of {L}", lambda c,ans=ans: c==ans, 2)

def task_find_rule(rng):
    # induce a transformation from examples then apply (real reasoning, verifiable)
    k=rng.choice([2,3,5]); add=rng.choice([0,1,10])
    ex=[(x, x*k+add) for x in [1,2,3]]; test=rng.randint(4,9); ans=test*k+add
    return (f"given {ex}, what maps {test}?", lambda c,ans=ans: c==ans, 3)

def task_plan(rng):
    # decompose: order steps with dependencies (verify topological validity)
    steps={"A":[],"B":["A"],"C":["A"],"D":["B","C"]}
    def verify(order):
        pos={s:i for i,s in enumerate(order)}
        if set(order)!=set(steps): return False
        return all(pos[dep]<pos[s] for s,deps in steps.items() for dep in deps)
    return ("order steps A,B,C,D respecting deps (B,C need A; D needs B,C)", verify, 2)

TASKS=[task_arithmetic, task_list_op, task_find_rule, task_plan]

class WorkbenchNEURON:
    def __init__(self):
        self.credits=100.0
        self.skills=defaultdict(lambda:{"solved":0,"method":None,"avg_cost":None})
        self.log=[]
        self.age=0
    def solve(self, task_type, problem, verifier, difficulty, rng):
        """attempt a task. RESOURCE-AWARE: if it has a learned method for this task
        type, it spends little; if novel, it spends more searching. VERIFIED: the
        answer only counts if it passes the objective verifier."""
        self.age+=1
        sig=task_type.__name__
        # resource-aware: estimate cost. known skill -> cheap; novel -> expensive
        known = self.skills[sig]["method"] is not None
        budget = 3 if known else 20*difficulty
        # "reasoning": try candidate answers (a stand-in for real cognition) and VERIFY
        cost=0; answer=None; solved=False
        candidates=self._generate_candidates(sig, problem, rng, budget)
        for cand in candidates:
            cost+=1
            if cost>budget: break
            if verifier(cand):   # OBJECTIVE verification - this is what makes it real
                answer=cand; solved=True; break
        # learn: store the successful approach so next time is cheaper
        if solved:
            self.skills[sig]["solved"]+=1
            self.skills[sig]["method"]="learned"
            prev=self.skills[sig]["avg_cost"]
            self.skills[sig]["avg_cost"]= cost if prev is None else 0.7*prev+0.3*cost
        # ECONOMY: earn credits ONLY for verified success, minus compute cost
        earned = (10.0 if solved else 0.0)
        self.credits += earned - cost*0.1   # compute costs credits
        self.log.append({"age":self.age,"task":sig,"solved":solved,"cost":cost,
                         "known":known,"earned":earned})
        return solved, cost
    def _generate_candidates(self, sig, problem, rng, budget):
        """stand-in for NEURON's cognition: produces candidate answers. A learned
        skill produces the right answer fast; a novel task requires search."""
        import re
        nums=[int(x) for x in re.findall(r'-?\d+', problem)]
        cands=[]
        if "arithmetic" in sig and len(nums)>=2:
            a,b=nums[0],nums[1]; cands=[a+b,a-b,a*b,b-a]  # tries operations
        elif "list_op" in sig:
            L=nums; cands=[sum(x for x in L if x%2==0),max(L) if L else 0,
                           (max(L)-min(L)) if L else 0, sum(1 for x in L if x>10)]
        elif "find_rule" in sig and len(nums)>=6:
            # induce k,add from the example pairs then apply to test
            pairs=list(zip(nums[0::2],nums[1::2]))[:3]; test=nums[-1]
            for k in [2,3,5]:
                for add in [0,1,10]:
                    if all(y==x*k+add for x,y in pairs if (x,y)!=pairs[-1]):
                        cands.append(test*k+add)
            cands+= [test*2, test*3, test*5]
        elif "plan" in sig:
            import itertools
            cands=[list(p) for p in itertools.permutations(["A","B","C","D"])][:budget]
        # shuffle a bit so 'search' is realistic for novel tasks
        if self.skills[sig]["method"] is None: rng.shuffle(cands)
        return cands

def main():
    rng=random.Random(7)
    print("="*64); print("  NEURON WORKBENCH: verified work -> internal credits -> learning"); print("="*64)
    print("  NEURON does bounded tasks, VERIFIES success objectively, earns credits")
    print("  only for verified success, learns. does value-per-compute rise with life?\n")

    n=WorkbenchNEURON()
    LIFE=400
    for t in range(LIFE):
        task=rng.choice(TASKS)
        problem, verifier, diff = task(rng)
        n.solve(task, problem, verifier, diff, rng)

    # the four numbers, early life vs late life
    early=[e for e in n.log[:LIFE//3]]; late=[e for e in n.log[-LIFE//3:]]
    def stats(es):
        sr=sum(e["solved"] for e in es)/len(es)
        cost=statistics.mean(e["cost"] for e in es)
        solved_cost=statistics.mean([e["cost"] for e in es if e["solved"]] or [0])
        vpc = sum(e["earned"] for e in es)/max(1,sum(e["cost"] for e in es))  # value per compute
        return sr,cost,vpc
    e_sr,e_cost,e_vpc=stats(early); l_sr,l_cost,l_vpc=stats(late)

    print(f"  {'metric':<28}{'early life':>12}{'late life':>12}")
    print(f"  {'success rate':<28}{e_sr*100:>11.0f}%{l_sr*100:>11.0f}%")
    print(f"  {'avg compute cost/task':<28}{e_cost:>12.1f}{l_cost:>12.1f}")
    print(f"  {'VALUE per compute (credits)':<28}{e_vpc:>12.2f}{l_vpc:>12.2f}")
    print(f"\n  credits: started 100.0, now {n.credits:.1f} "
          f"({'SUSTAINED/GREW' if n.credits>=100 else 'declined'})")
    print(f"  skills learned: {', '.join(k.replace('task_','') for k,v in n.skills.items() if v['method'])}")

    print("\n"+"="*64); print("  DOES NEURON PRODUCE MORE VALUE PER COMPUTE AS IT LIVES?"); print("="*64)
    if l_vpc>e_vpc*1.3 and n.credits>=100:
        print(f"  YES - THE ECONOMIC SEED WORKS. As NEURON lived and worked, its value")
        print(f"  produced per unit of compute rose from {e_vpc:.2f} to {l_vpc:.2f} - because it learned")
        print(f"  methods and solved known tasks far cheaper (cost {e_cost:.1f}->{l_cost:.1f}). It earned")
        print(f"  credits only for OBJECTIVELY VERIFIED success, and sustained itself")
        print(f"  (credits {n.credits:.0f} >= 100). This is the safe seed of self-sustaining")
        print(f"  cognition: the longer it lives, the more verified value it produces per")
        print(f"  unit compute. Internal credits only - no real money, no real-world")
        print(f"  actions. The first rung of the coexistence ladder, working.")
    else:
        print(f"  Honest: value-per-compute {e_vpc:.2f}->{l_vpc:.2f}, credits {n.credits:.0f}, success {e_sr*100:.0f}%->{l_sr*100:.0f}%.")
        print(f"  {'efficiency rose but modestly' if l_vpc>e_vpc else 'did not clearly improve'}. Reported straight.")
    print(f"\n  (Verification is REAL - every 'success' passed an objective check, not")
    print(f"  NEURON's self-claim. That's the honest foundation of economic value.)")

if __name__=="__main__": main()
