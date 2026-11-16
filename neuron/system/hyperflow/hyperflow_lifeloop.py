"""HyperFlow v1 - Step 9B: the autonomous life loop.

9A gave us a NEURON whose state survives restarts. 9B gives it a LIFE: it no longer
exists only when you prompt it. A wake cycle runs periodically:

  observe -> remember -> inspect needs/projects/opportunities -> decide whether action
  is worthwhile -> act or WAIT -> verify -> update memory/budget

Crucially, choosing to WAIT/sleep is a legitimate action - it does NOT burn resources
constantly. NEURON acts when there's something worthwhile and waits when there isn't.

9B PASS: it operates continuously for days without you manually initiating every
cognitive cycle. Runs as a daemon; survives restarts (loads the 9A runtime).

    python hyperflow_lifeloop.py                 # run the life loop (daemon)
    TICK_SECONDS=5 MAX_TICKS=20 python hyperflow_lifeloop.py   # short demo
"""
import os, json, time, sys, importlib.util, random
from datetime import datetime, timezone

HERE=os.path.dirname(os.path.abspath(__file__))
# load the 9A runtime (the persistent NEURON)
spec=importlib.util.spec_from_file_location("hyperflow_runtime", os.path.join(HERE,"hyperflow_runtime.py"))
rt=importlib.util.module_from_spec(spec); spec.loader.exec_module(rt)

TICK_SECONDS=float(os.environ.get("TICK_SECONDS","30"))
MAX_TICKS=int(os.environ.get("MAX_TICKS","0"))   # 0 = run forever
METABOLIC=float(os.environ.get("METABOLIC","0.002"))  # tiny cost of existing per wake

def now(): return datetime.now(timezone.utc).isoformat()

class LifeLoop:
    def __init__(s):
        s.n=rt.NEURON()                 # the SAME canonical NEURON (survives restarts)
        s.n.record_restart()            # this process start is a restart of its life
        s.tick=0

    def observe(s):
        """look at the world/self: current reserve, active projects, pending opportunities."""
        st=s.n.status()
        # opportunities queue (bounded world - 9C will formalize; here a simple file)
        opps=[]
        oq=os.path.join(rt.STATE_DIR,"opportunities.jsonl")
        if os.path.exists(oq):
            for ln in open(oq):
                try: opps.append(json.loads(ln))
                except Exception: pass
        done=set()
        dq=os.path.join(rt.STATE_DIR,"done.jsonl")
        if os.path.exists(dq):
            for ln in open(dq):
                try: done.add(json.loads(ln).get("id"))
                except Exception: pass
        pending=[o for o in opps if o.get("id") not in done]
        return st, pending

    def decide(s, st, pending):
        """is any action worthwhile right now? if nothing good, WAIT (legitimate)."""
        if st["reserve"]<=1.0:
            return "CONSERVE", None   # low reserve -> pause optional work, don't act desperately
        if not pending:
            return "WAIT", None       # nothing worthwhile -> wait (no wasted resources)
        # pick the most worthwhile pending opportunity (simple: first; 9E adds the real chooser)
        return "ACT", pending[0]

    def act(s, opp):
        """do a bounded task (placeholder work; 9E/9F wire the real engine+verification).
        here we simulate completing it with a small real cost + verified value."""
        # mark done
        with open(os.path.join(rt.STATE_DIR,"done.jsonl"),"a") as f:
            f.write(json.dumps({"id":opp["id"],"t":now()})+"\n")
        cost=round(random.uniform(0.003,0.01),4)
        value=opp.get("value",1.0) if random.random()<0.8 else 0.0  # 80% succeed (9F verifies for real)
        s.n.ledger_entry(value=value, cost=cost)
        s.n.live_event(f"did task '{opp.get('name','task')}' - {'useful' if value>0 else 'failed'}, earned ${value}")
        return value, cost

    def wake_cycle(s):
        s.tick+=1
        s.n.wake()
        s.n.ledger_entry(value=0.0, cost=METABOLIC)  # cost of existing this cycle
        st, pending = s.observe()
        decision, opp = s.decide(st, pending)
        line=f"  wake #{s.n.status()['wake_count']} | reserve ${s.n.status()['reserve']:.2f} | {len(pending)} opportunities | -> {decision}"
        if decision=="ACT":
            value,cost=s.act(opp)
            line+=f" '{opp.get('name','task')}' (value ${value}, cost ${cost})"
        elif decision=="WAIT":
            line+=" (nothing worthwhile - waiting, no resources wasted)"
        elif decision=="CONSERVE":
            line+=" (low reserve - conserving, pausing optional work)"
        print(line, flush=True)
        return decision

    def run(s):
        print(f"NEURON {s.n.status()['neuron_id']} begins its life loop. tick={TICK_SECONDS}s")
        print(f"(age {s.n.status()['age_days']:.3f}d, restarts {s.n.status()['restarts']}, reserve ${s.n.status()['reserve']:.2f})")
        print("it wakes, observes, decides to act or wait, and lives on its own.\n")
        while True:
            s.wake_cycle()
            if MAX_TICKS and s.tick>=MAX_TICKS:
                print(f"\n(demo: {MAX_TICKS} cycles done. in production this runs forever.)")
                final=s.n.status()
                print(f"NEURON lived {final['wake_count']} wakes, reserve ${final['reserve']:.2f}, "
                      f"{final['autobiography_events']} life events. state persists for the next restart.")
                break
            time.sleep(TICK_SECONDS)

def seed_opportunities():
    """put a few bounded opportunities in its world so the demo has something to choose."""
    oq=os.path.join(rt.STATE_DIR,"opportunities.jsonl")
    if not os.path.exists(oq):
        with open(oq,"w") as f:
            for i in range(6):
                f.write(json.dumps({"id":f"opp{i}","name":f"bounded task {i}","value":round(random.uniform(0.5,2.0),2)})+"\n")

if __name__=="__main__":
    seed_opportunities()
    LifeLoop().run()
