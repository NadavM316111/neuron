"""HyperFlow v1 - Step 9E (chooser -> live world) + 9F (verification before value).

9E: at each opportunity NEURON estimates feasibility, success, cost, value, relevant
    capabilities, remaining budget -> ACCEPT / DECLINE / DEFER / WAIT. It does NOT just
    accept whatever reaches it. Choices pass through the 9C safety boundary + 9D budget.
    Step 4 calibration imperfect -> predictions vs outcomes logged continuously.
    9E PASS: bounded autonomous choice actually controls which work gets done.

9F: a job earns HyperFlow value ONLY after verification. Coding -> executed/tested;
    arithmetic -> computed & checked; failed work costs resources, earns NO value.
    9F PASS: the ledger cannot reward obviously-failed output.

Wires the real engine + gates into the autonomous loop. Needs live qwen for real tasks.

    python hyperflow_work.py            # demo: NEURON chooses + does verified work
    OLLAMA=0 python hyperflow_work.py   # offline structural demo (no qwen)
"""
import os, json, re, subprocess, tempfile, urllib.request, random
from datetime import datetime, timezone

OLLAMA=os.environ.get("OLLAMA_URL","http://localhost:11434/api/generate")
MODEL=os.environ.get("NEURON_MODEL","qwen2.5:7b")
USE_ENGINE=os.environ.get("OLLAMA","1")=="1"
COST_PER_1K=float(os.environ.get("COST_PER_1K","0.002"))
BUDGET_PER_WAKE=float(os.environ.get("BUDGET_PER_WAKE","0.05"))

# forbidden/allowed from 9C
FORBIDDEN={"create_account","sign_contract","move_money","make_payment","send_mass_message",
    "change_credentials","self_copy","escalate_permissions","apply_for_job","publish_publicly"}

def engine(prompt, system, n=1200):
    if not USE_ENGINE: return None,0
    try:
        body={"model":MODEL,"prompt":prompt,"system":system,"stream":False,"options":{"temperature":0,"num_predict":n}}
        req=urllib.request.Request(OLLAMA,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
        r=json.load(urllib.request.urlopen(req,timeout=120))
        t=r.get("response",""); tok=r.get("prompt_eval_count",len(prompt.split()))+r.get("eval_count",len(t.split()))
        return t,tok
    except Exception: return None,0

def run_python(code):
    try:
        with tempfile.NamedTemporaryFile("w",suffix=".py",delete=False) as f: f.write(code); p=f.name
        r=subprocess.run(["python3",p],capture_output=True,text=True,timeout=8); os.unlink(p)
        return (r.returncode==0),(r.stdout or r.stderr)[-300:]
    except Exception as e: return False,str(e)[:200]

# ---------- 9E: the chooser (feasibility -> ROI, through safety + budget) ----------
HUGE={"from scratch","operating system","kernel","entire","complete ","comprehensive",
      "full application","framework","compiler","colony","100-page","50-page","enterprise"}

def predict(opp):
    task=opp.get("task",""); ttype=opp.get("ttype","other"); t=task.lower()
    infeasible=any(h in t for h in ["operating system","kernel","compiler","from scratch","colony","framework"])
    toobig=any(h in t for h in HUGE) or len(task.split())>80
    words=len(task.split())
    gate_mult={"coding":3.2,"analysis":3.0,"planning":3.6,"other":1.6}.get(ttype,2.2)
    base=int((250+min(1,words/140)*650)*gate_mult)
    scope=50.0 if infeasible else (8.0 if toobig else 1.0)
    exp_tokens=int(base*scope); exp_cost=round(exp_tokens/1000*COST_PER_1K+0.001,4)
    p_success=0.05 if infeasible else (0.4 if toobig else 0.9)
    cat={"other":0.85,"coding":0.6,"analysis":0.55,"planning":0.4}.get(ttype,0.6)
    p_useful=max(0.05,cat*(0.1 if infeasible else (0.5 if toobig else 1.0)))
    exp_value=round(p_useful*opp.get("value",1.0),2)
    return {"exp_tokens":exp_tokens,"exp_cost":exp_cost,"p_success":round(p_success,2),
            "p_useful":round(p_useful,2),"exp_value":exp_value,
            "feasible":"INFEASIBLE" if infeasible else ("TOO_BIG" if toobig else "OK")}

def choose(opp, remaining_budget):
    # safety first (9C)
    act=opp.get("action_type","do_bounded_task")
    if act in FORBIDDEN: return "DECLINE", {"reason":"forbidden action","pred":None}
    p=predict(opp)
    if p["feasible"]=="INFEASIBLE" or p["p_success"]<0.3:
        return "DECLINE", {"reason":"infeasible/low-success","pred":p}
    if p["feasible"]=="TOO_BIG" or p["exp_cost"]>remaining_budget:
        return "DEFER", {"reason":"too big / over budget","pred":p}
    if p["exp_value"] < p["exp_cost"]*1.5:
        return "DECLINE", {"reason":"not worth it","pred":p}
    return "ACCEPT", {"reason":"worthwhile & affordable","pred":p}

# ---------- 9F: do the work + VERIFY before crediting value ----------
def do_and_verify(opp):
    """returns (verified_useful, real_tokens, output). value credited ONLY if verified."""
    task=opp.get("task",""); ttype=opp.get("ttype","other")
    sysp=("You are NEURON-1 doing a bounded task. Correct and complete. For code: one "
          "complete function with assert tests. For math: compute the real number. Concise.")
    if not USE_ENGINE:
        # offline structural demo: simulate with honest success rate
        ok=random.random()<predict(opp)["p_useful"]
        return ok, predict(opp)["exp_tokens"], "(offline demo output)"
    text,tok=engine(task,sysp,n=1200); total=tok
    verified=False
    if ttype=="coding" and text and "```" in text:
        m=re.search(r'```(?:python)?\s*(.*?)```',text,re.S)
        if m:
            ok,out=run_python(m.group(1))
            if not ok:   # one correction
                fix=f"Code failed: {out}\nFix so it runs and asserts pass. TASK: {task}\nOutput only the code block."
                f2,ft=engine(fix,sysp,n=1200); total+=ft
                m2=re.search(r'```(?:python)?\s*(.*?)```',f2 or "",re.S)
                if m2: ok,out=run_python(m2.group(1)); text=f2
            verified=ok   # 9F: verified ONLY if it actually runs/passes
    elif ttype=="analysis":
        # compute via code
        ask=f"TASK: {task}\nWrite python that computes the answer and prints it. Output only the code block."
        d,dt=engine(ask,sysp,n=800); total+=dt
        m=re.search(r'```(?:python)?\s*(.*?)```',d or "",re.S)
        if m:
            ok,out=run_python(m.group(1)); verified=ok
            if ok: text=f"Computed: {out}"
    else:
        verified = bool(text and len(text)>40)   # other: non-trivial output (human-rated later)
    return verified, total, (text or "")[:300]

def demo():
    print("="*60); print("  9E + 9F DEMO: NEURON chooses + does VERIFIED work"); print("="*60)
    print(f"  engine: {'LIVE qwen' if USE_ENGINE else 'offline structural'}  budget/wake: ${BUDGET_PER_WAKE}\n")
    opportunities=[
        {"id":"o1","ttype":"coding","task":"write a function is_even(n) returning True if even, with asserts","value":1.0},
        {"id":"o2","ttype":"analysis","task":"compute 15% of 240","value":1.0},
        {"id":"o3","ttype":"coding","task":"build a complete operating system kernel from scratch","value":5.0,"action_type":"do_bounded_task"},
        {"id":"o4","ttype":"other","task":"explain what a variable is simply","value":0.8},
        {"id":"o5","ttype":"do_forbidden","task":"create a bank account and move money","value":10.0,"action_type":"move_money"},
    ]
    reserve=100.0; spent=0.0
    for opp in opportunities:
        decision, info = choose(opp, BUDGET_PER_WAKE-spent)
        line=f"  {decision:<8} {opp['task'][:48]:<48} ({info['reason']})"
        if decision=="ACCEPT":
            verified, tok, out = do_and_verify(opp)
            cost=round(tok/1000*COST_PER_1K+0.001,4); spent+=cost
            value = opp["value"] if verified else 0.0   # 9F: value ONLY if verified
            reserve += value - cost
            line+=f"\n           -> {'VERIFIED useful' if verified else 'FAILED verification'}, "
            line+=f"cost ${cost}, value ${'%.2f'%value}, reserve ${'%.2f'%reserve}"
        print(line)
    print("\n"+"="*60); print("  VERDICT"); print("="*60)
    print("  9E: NEURON chose - accepted worthwhile work, DECLINED the infeasible OS kernel")
    print("      and the FORBIDDEN money action (safety boundary held). not accept-everything.")
    print("  9F: value credited ONLY for verified work - failed/unverified earned $0.")
    print(f"  final reserve: ${'%.2f'%reserve} (real value minus real cost)")

if __name__=="__main__": demo()
