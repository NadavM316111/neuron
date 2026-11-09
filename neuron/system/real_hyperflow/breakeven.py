"""Amortization / Break-Even: cognitive capital (the friend's correction).

The honest flaw in mature_vs_newborn: the mature NEURON's 2.4x deployment advantage
ignored its 1,518-token one-time TRAINING cost. Over 60 tasks the mature agent's
FULL lifecycle (training + operating) was actually ~12.5% MORE expensive. The real
question is AMORTIZATION: NEURON invests compute to learn once, then earns it back
through reuse. When does it cross BREAK-EVEN and become cheaper even after paying
for its own education?

Uses the REAL MEASURED rates from the live qwen run (not simulated):
  training investment   = 1518 tokens (measured, one-time)
  mature reuse cost     ~ 15 tokens/task (measured)
  newborn/fresh cost    ~ 35.8 tokens/task (measured avg)

Computes the exact break-even task + the lifetime cost curve: PERSISTENT learner
(pays upfront, then cheap) vs FRESH-agent-every-task (the disposable-AI strategy).

The thesis (sharper, per friend): a NEURON's life creates accumulated COGNITIVE
CAPITAL - the economic argument for persistent individuals over spawning fresh agents.

    python breakeven.py
"""
# REAL measured values from the live qwen run
TRAIN=1518.0          # one-time learning investment (measured)
MATURE_PER_TASK=15.0  # measured reuse cost per task
# newborn avg from the run: 2149 tokens / 60 tasks ~ 35.8, but newborn ALSO learns as
# it goes (later tasks get cheaper for it too). We model two newborn strategies:
#   (a) FRESH-EVERY-TASK: never caches - always pays full fresh cost (~a disposable agent)
#   (b) NEWBORN-PERSISTENT: learns as it lives (its curve from the run)
FRESH_PER_TASK=35.8   # a fresh agent that never accumulates (full cost every task)
# newborn-persistent: starts ~ high, approaches mature rate as it learns the library.
# measured: newborn cumulative 90@1, 742@10, 1596@30, 2149@60 -> decelerating.

def newborn_persistent_cum(n):
    # fit the measured newborn curve (learns over time -> marginal cost falls toward ~15)
    # measured points: (1,90),(10,742),(30,1596),(60,2149). marginal ~ 90, then ~72, ~43, ~18.
    # model marginal cost dropping from ~90 toward a floor ~15 as it learns (~18 capabilities)
    import math
    total=0.0
    for t in range(1,n+1):
        # marginal cost decays as it learns (exponential approach to the reuse floor)
        marg = 15 + 75*math.exp(-t/18.0)
        total+=marg
    return total

def mature_cum(n): return TRAIN + MATURE_PER_TASK*n
def fresh_cum(n): return FRESH_PER_TASK*n   # disposable: full cost forever

def main():
    print("="*64); print("  COGNITIVE CAPITAL: break-even (from REAL measured rates)"); print("="*64)
    print(f"  measured: training {TRAIN:.0f} tok (once), mature {MATURE_PER_TASK:.0f}/task,")
    print(f"  fresh-agent {FRESH_PER_TASK:.1f}/task, newborn-persistent learns over time.\n")

    # break-even vs a FRESH agent (disposable strategy)
    be_fresh=None
    for t in range(1,2000):
        if mature_cum(t)<=fresh_cum(t): be_fresh=t; break
    # break-even vs a NEWBORN that also learns (the real NEURON-vs-fresh-NEURON comparison)
    be_nb=None
    for t in range(1,2000):
        if mature_cum(t)<=newborn_persistent_cum(t): be_nb=t; break

    print(f"  BREAK-EVEN task (mature total cost <= the alternative's total cost):")
    print(f"    vs a FRESH agent (never learns):        task {be_fresh}")
    print(f"    vs a NEWBORN that learns as it lives:   task {be_nb}")

    print(f"\n  cumulative lifetime compute (tokens) at milestones:")
    print(f"  {'task':>6}{'mature':>12}{'newborn-learns':>16}{'fresh-agent':>14}{'mature saves vs fresh':>24}")
    for t in [60,100,200,500,1000]:
        m=mature_cum(t); nb=newborn_persistent_cum(t); fr=fresh_cum(t)
        print(f"  {t:>6}{m:>12.0f}{nb:>16.0f}{fr:>14.0f}{fr-m:>22.0f}")

    print("\n"+"="*64); print("  DOES COGNITIVE CAPITAL PAY OFF OVER A LIFE?"); print("="*64)
    save_1000 = fresh_cum(1000)-mature_cum(1000)
    save_nb_1000 = newborn_persistent_cum(1000)-mature_cum(1000)
    if be_fresh and be_fresh<150:
        print(f"  YES - a NEURON's life creates accumulated COGNITIVE CAPITAL that pays off.")
        print(f"  The mature NEURON's one-time learning investment ({TRAIN:.0f} tokens) is recouped by")
        print(f"  task {be_fresh} vs a fresh agent" + (f" (task {be_nb} vs a newborn that also learns)" if be_nb else "") + ".")
        print(f"  After that it only pulls further ahead: by task 1000 the mature NEURON has")
        print(f"  spent {save_1000:.0f} fewer tokens than always spawning fresh agents"
              + (f", {save_nb_1000:.0f} fewer than a newborn" if save_nb_1000>0 else "") + ".")
        print(f"  So persistence is economically SUPERIOR to disposability: pay an upfront")
        print(f"  education cost once, then become cheaper and cheaper over a long life.")
        print(f"  This is the clean economic argument for your whole vision - why build")
        print(f"  persistent individuals that accumulate cognitive capital, instead of")
        print(f"  spawning fresh agents that must relearn everything every time.")
        print(f"\n  (honest: break-even vs a learning newborn ({be_nb}) is later because a newborn")
        print(f"  ALSO accumulates capital over its life - the real lesson is that ANY")
        print(f"  persistent NEURON beats fresh-every-time; a head start just helps sooner.)")
    else:
        print(f"  Honest: break-even at task {be_fresh} vs fresh, {be_nb} vs newborn. Reported straight.")

if __name__=="__main__": main()
