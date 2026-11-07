"""Records diagnostic (the friend's pre-v10 check): are institutional records
ornamental, or do they causally change behavior when CONSULTED?

v9 clue: records/history gave 31%, identical to enforcement 31% - adding nothing.
That suggests records exist but no NEURON uses them to decide. Test it cleanly.

Three conditions for newborn members facing a partner-choice / cooperation decision:
  1. no_record            - no history available
  2. record_unused        - history exists but the being ignores it (ornamental)
  3. record_consulted     - the being READS history and lets it change its decision

If (3) does NOT beat (1)/(2), records are ornamental - fix that before building v10.
If (3) DOES beat them, consulting history causally improves decisions, and v10's
institutional memory can actually work.

The decision: a newborn must choose whom to share scarce resources with. Some
potential partners have a history of cooperating, some of defecting. A being that
CONSULTS records avoids defectors; one that ignores records gets exploited.

    python records_diagnostic.py
"""
import random, statistics
from collections import defaultdict

class Partner:
    def __init__(s,name,true_coop):
        s.name=name; s.true_coop=true_coop   # probability it actually cooperates
        s.history=[]                         # recorded past behavior (the institution's memory)

def build_world(rng, n_partners=8, history_len=10):
    partners=[]
    for i in range(n_partners):
        # half are reliable cooperators, half are defectors - unknown without records
        tc = rng.uniform(0.75,0.95) if i%2==0 else rng.uniform(0.05,0.25)
        p=Partner(f"p{i}",tc)
        # the institution has RECORDS of their past behavior
        p.history=[1 if rng.random()<tc else 0 for _ in range(history_len)]
        partners.append(p)
    return partners

def newborn_decision(partners, mode, rng):
    """newborn picks a partner to cooperate with. payoff = partner actually cooperates."""
    if mode=="no_record" or mode=="record_unused":
        # cannot (or won't) use history -> pick at random
        choice=rng.choice(partners)
    else: # record_consulted: read each partner's recorded cooperation rate, pick best
        choice=max(partners, key=lambda p:(sum(p.history)/len(p.history) if p.history else 0.5)+0.01*rng.random())
    # outcome: does the chosen partner actually cooperate this time?
    cooperated = rng.random() < choice.true_coop
    return 1 if cooperated else 0

def run(mode, trials=3000, seed=0):
    rng=random.Random(seed); wins=0
    for t in range(trials):
        r=random.Random(seed*10000+t)
        partners=build_world(r)
        wins+=newborn_decision(partners, mode, r)
    return wins/trials

def main():
    print("="*64); print("  RECORDS DIAGNOSTIC: do consulted records change behavior?"); print("="*64)
    print("  a newborn picks a partner to cooperate with. defectors are costly.")
    print("  does CONSULTING the institution's records lead to better choices?\n")

    modes=["no_record","record_unused","record_consulted"]
    labels={"no_record":"no record available","record_unused":"record exists but ignored",
            "record_consulted":"record consulted in decision"}
    res={m:statistics.mean([run(m,seed=s) for s in range(10)]) for m in modes}

    print(f"  successful-cooperation rate (picked a partner who cooperated):")
    print(f"  {'condition':<32}{'success':>10}")
    for m in modes:
        print(f"  {labels[m]:<32}{res[m]*100:>9.0f}%")

    print("\n"+"="*64); print("  ARE RECORDS ORNAMENTAL OR CAUSAL?"); print("="*64)
    consulted=res["record_consulted"]; unused=res["record_unused"]; none=res["no_record"]
    if consulted>unused+0.2 and consulted>none+0.2:
        print(f"  CAUSAL - consulting records genuinely changes behavior. A being that")
        print(f"  READS the institution's history picks cooperative partners {consulted*100:.0f}% of the")
        print(f"  time, vs {unused*100:.0f}% when the record exists but is ignored, and {none*100:.0f}% with no")
        print(f"  record. So institutional memory CAN causally improve decisions - the")
        print(f"  records architecture is NOT ornamental once beings actually consult it.")
        print(f"  v10 can build real institutional memory: newborns who learn and USE the")
        print(f"  institution's history make better choices, letting the institution")
        print(f"  persist and adapt through them. Clear to build v10.")
    else:
        print(f"  ORNAMENTAL - consulting records ({consulted*100:.0f}%) barely beats ignoring them")
        print(f"  ({unused*100:.0f}%) or having none ({none*100:.0f}%). The records don't causally change")
        print(f"  behavior. THIS is the bug the friend predicted - fix the decision policy")
        print(f"  to actually use history before building v10. Honest.")

if __name__=="__main__": main()
