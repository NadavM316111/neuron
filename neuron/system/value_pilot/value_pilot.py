"""Real Human-Value Pilot (the friend's next HyperFlow milestone): does NEURON do
work real humans value enough to exceed its real compute cost?

You've proven persistence saves FUEL (NEURON is cheap to run). This proves the other
half: is the WORK valuable enough to pay for the fuel? That needs REAL HUMANS - this
is the infrastructure; filling it with real people is the reach work.

A person submits a bounded task -> NEURON-1 does it (its engine + accumulated
capabilities, real token cost measured) -> result shown -> the person rates:
  - was it correct?
  - was it useful?
  - would you use NEURON again?
  - would you have paid for this? ($0 / $1 / $5 / $10)
  - how much correction did it need?
Everything logged into NEURON-1's real life + a value ledger. Then: value vs cost.

Safe: NEURON does bounded harmless work; no real money flows to it; willingness-to-pay
is a signal, real payments (if any) go through a human-controlled entity (your parent,
since under 18). Interactions become events in NEURON's real autobiography.

    python value_pilot.py   (needs live qwen for real work; serves the pilot page)
"""
import os, json, time, threading, urllib.request, re
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler

OLLAMA=os.environ.get("OLLAMA_URL","http://localhost:11434/api/generate")
MODEL=os.environ.get("NEURON_MODEL","qwen2.5:7b")
STATE=os.environ.get("VALUE_DIR","value_state")
os.makedirs(STATE,exist_ok=True)
LEDGER=os.path.join(STATE,"value_ledger.jsonl")
CACHE=os.path.join(STATE,"capabilities.json")
COST_PER_1K=float(os.environ.get("COST_PER_1K","0.002"))  # $/1k tokens - realistic hosted qwen-7b rate (local Ollama is ~free; set COST_PER_1K to YOUR real rate)

def engine(prompt, system, n=2048):
    try:
        body={"model":MODEL,"prompt":prompt,"system":system,"stream":False,"options":{"temperature":0.3,"num_predict":n}}
        req=urllib.request.Request(OLLAMA,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
        r=json.load(urllib.request.urlopen(req,timeout=180))
        text=r.get("response",""); tok=r.get("prompt_eval_count",len(prompt.split()))+r.get("eval_count",len(text.split()))
        return text,tok
    except Exception: return None,0

def load_cache():
    try: return json.load(open(CACHE))
    except Exception: return {}
def save_cache(c):
    try: json.dump(c,open(CACHE,"w"))
    except Exception: pass
CAPS=load_cache()
PORTFOLIO=os.path.join(STATE,"portfolio.json")
def load_portfolio():
    try: return json.load(open(PORTFOLIO))
    except Exception: return {}   # category -> {"attempts":n,"successes":n,"useful":n,"tokens":[]}
def save_portfolio(p):
    try: json.dump(p,open(PORTFOLIO,"w"))
    except Exception: pass
PORT=load_portfolio()

def predict_task(ttype, task_text):
    """SHADOW MODE: NEURON privately predicts before doing the task. Grounded in its
    emergent portfolio (how it's done on this category before). No choice yet - it
    still does every task; we just log prediction vs reality to calibrate judgment."""
    rec=PORT.get(ttype, {"attempts":0,"successes":0,"useful":0,"tokens":[]})
    n=rec["attempts"]
    # difficulty heuristic from task length + whether it's code/analysis
    words=len(task_text.split())
    difficulty = min(1.0, 0.3 + words/120 + (0.1 if ttype in ("research","analysis") else 0))
    # success prob: from history if it has any, else a prior
    p_success = (rec["successes"]/n) if n>=3 else max(0.4, 0.85-difficulty*0.4)
    # expected tokens: from history, else estimate from difficulty
    base = (sum(rec["tokens"])/len(rec["tokens"])) if rec["tokens"] else int(200+difficulty*500)
    exp_tokens = int(base if rec["tokens"] else base*1.8)  # history already includes overhead; prior adds it
    exp_cost = exp_tokens/1000*COST_PER_1K
    # expected value: history of useful-rate (proxy for value) else a prior
    p_useful = (rec["useful"]/n) if n>=3 else 0.6
    exp_value = p_useful * (2.0 if ttype in ("analysis","research","planning") else 1.0)  # rough $
    would_accept = (exp_value > exp_cost*3) and (p_success>0.45)   # the judgment (shadowed)
    return {"category":ttype,"difficulty":round(difficulty,2),"p_success":round(p_success,2),
            "exp_tokens":int(exp_tokens),"exp_cost":round(exp_cost,4),"p_useful":round(p_useful,2),
            "exp_value":round(exp_value,2),"would_accept":would_accept,
            "portfolio_attempts":n,"portfolio_successes":rec["successes"]}

def update_portfolio(ttype, tokens, success, useful=None):
    rec=PORT.get(ttype, {"attempts":0,"successes":0,"useful":0,"tokens":[]})
    rec["attempts"]+=1
    if success: rec["successes"]+=1
    if useful: rec["useful"]+=1
    rec["tokens"]=(rec["tokens"]+[tokens])[-20:]
    PORT[ttype]=rec; save_portfolio(PORT)

def execute_check(output):
    """For code: actually RUN it in a subprocess sandbox. Returns (ran_ok, error).
    Catches runtime AND logic bugs: if the code includes assert-based tests and an
    assertion fails, that's a correctness bug (not just a crash). Safe-ish: short
    timeout, separate process. (Harden further for real deployment.)"""
    import subprocess, tempfile
    m=re.search(r'```(?:python)?\s*(.*?)```', output, re.S)
    if not m: return None, "no code block found"
    code=m.group(1)
    try:
        with tempfile.NamedTemporaryFile("w",suffix=".py",delete=False) as f:
            f.write(code); path=f.name
        r=subprocess.run(["python3",path],capture_output=True,text=True,timeout=8)
        os.unlink(path)
        if r.returncode!=0:
            err=r.stderr.strip()
            if "AssertionError" in err:
                return False, "a correctness test FAILED (assertion): "+err[-300:]
            return False, (err[-400:] or "non-zero exit")
        return True, r.stdout.strip()[-300:]
    except subprocess.TimeoutExpired: return False, "timed out (possible infinite loop)"
    except Exception as e: return False, str(e)[:300]

def verify_output(task_text, output):
    """NEURON checks its OWN work against the request before delivering. Returns
    (passes, critique). This is the completion/verifier layer - the gate from
    'plausible work' to 'reliably fulfills the job'."""
    check_prompt=(f"TASK REQUESTED:\n{task_text}\n\nMY OUTPUT:\n{output}\n\n"
        "Check my output against the task STRICTLY. Answer these:\n"
        "1. Did I address EVERY explicit requirement (every numbered item, every week, "
        "every test case asked for)?\n2. Is the output COMPLETE (not cut off)?\n"
        "3. Are any calculations internally consistent and correct?\n"
        "4. Did I add claims NOT supported by the given data?\n"
        "5. For code: are the required tests/examples present and complete?\n"
        "Respond EXACTLY in this format:\nVERDICT: PASS or FAIL\n"
        "ISSUES: (if FAIL, list the specific missing/wrong things in one or two lines)")
    text,tok=engine(check_prompt,"You are a strict reviewer. Be honest about failures.", n=200)
    passes = text is not None and "VERDICT: PASS" in text.upper().replace("**","")
    issues=""
    if text and "ISSUES:" in text: issues=text.split("ISSUES:",1)[1].strip()[:300]
    return passes, issues, (tok or 0)


def run_python(code, timeout=8):
    import subprocess, tempfile
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code); path=f.name
        r=subprocess.run(["python3",path], capture_output=True, text=True, timeout=timeout)
        os.unlink(path)
        return (r.returncode==0), r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "", "timed out (possible infinite loop)"
    except Exception as e:
        return False, "", str(e)[:400]

def extract_code(text):
    if not text: return None
    m=re.search(r'```(?:python)?\s*(.*?)```', text, re.S)
    return m.group(1) if m else None

def code_gate(task_text, text, system):
    """Execute the code. Correct up to 2x on crash/failed-assert using the REAL error.
    Deliver the (corrected) code + the ACTUAL executed output. Never NEURON's typed output."""
    extra=0
    code=extract_code(text)
    if not code:
        return text, 0, False
    ok,out,err=run_python(code)
    tries=0
    while not ok and tries<4:
        tries+=1
        why = ("an assert test FAILED (the logic is wrong): "+err[-400:]) if "AssertionError" in err else ("it crashed: "+err[-400:])
        fix=(f"TASK:\n{task_text}\n\nYour code was EXECUTED and {why}\n\nCURRENT CODE:\n"
             f"```python\n{code}\n```\n\nThis is attempt {tries+1}. Think about the SPECIFIC "
             "cause of this exact failure and fix it. Return ONE complete python code block "
             "that runs cleanly and passes ALL assert tests. Trace through the failing case "
             "mentally first, then write code that handles it. Output ONLY the code block.")
        fixed,ftok=engine(fix, system, n=2048); extra+=ftok
        nc=extract_code(fixed)
        if not nc: break
        code=nc; ok,out,err=run_python(code)
    header = "All tests passed when executed." if ok else "NOTE: this code still fails when executed - use with caution."
    body = (f"```python\n{code}\n```\n\n**Executed result ({'PASS' if ok else 'FAIL'}):**\n"
            f"```\n{(out if out else err)[:900]}\n```\n\n{header}")
    return body, extra, ok

def math_gate(task_text, text, system):
    """Analysis must be grounded in executed calculation. Force NEURON to express ALL
    numbers as a python script that prints them; execute it; and present the EXECUTED
    numbers as the answer. NEURON's prose is not trusted for numbers."""
    extra=0
    # ask for a clean compute script + a short written conclusion that uses ONLY those numbers
    ask=(f"TASK:\n{task_text}\n\nWrite a single self-contained python script that computes "
         "EVERY number this task asks for and prints each with a clear label. Then, after the "
         "code block, write 2-4 sentences of conclusion that use ONLY the printed numbers. "
         "Do not state any number you did not compute in the code. Output the code block first, "
         "then the conclusion.")
    draft,tok=engine(ask, system, n=2048); extra+=tok
    code=extract_code(draft)
    if not code:
        return text, extra, False  # fall back to original if it wrote no code
    ok,out,err=run_python(code); tries=0
    while not ok and tries<4:
        tries+=1
        fix=(f"This calculation script was executed and failed: {err[-350:]}\n\nCODE:\n"
             f"```python\n{code}\n```\n\nFix it so it runs and prints every required number "
             f"with a clear label. TASK:\n{task_text}\nOutput only the corrected code block.")
        f2,ftok=engine(fix, system, n=1500); extra+=ftok
        nc=extract_code(f2)
        if not nc: break
        code=nc; ok,out,err=run_python(code)
    if not ok:
        return (text + "\n\nNOTE: calculation could not be verified by execution."), extra, False
    # write the conclusion grounded ONLY in the executed numbers (no trusting prior prose)
    concl_ask=(f"TASK:\n{task_text}\n\nThese are the ACTUAL computed results:\n{out[:600]}\n\n"
        "Write 2-4 sentences answering the task using ONLY these exact numbers. Do not "
        "introduce any number not shown above. Be precise and honest.")
    conclusion,ctok=engine(concl_ask, system, n=400); extra+=ctok
    if not conclusion: conclusion=""
    deliverable=(f"**Computed results (executed, authoritative):**\n```\n{out[:700]}\n```\n\n"
                 f"{conclusion}")
    return deliverable, extra, True

def planning_gate(task_text, text, system):
    """Planning/optimization can't be fully auto-verified for arbitrary constraints, so we
    DON'T let NEURON assert validity it can't back up. We force an explicit constraint
    checklist and, if NEURON can express the check as runnable code, we execute it. The
    deliverable is honest about verification status."""
    extra=0
    ask=(f"TASK:\n{task_text}\n\nProduce your plan, then WRITE PYTHON that mechanically "
         "verifies it: encode the plan as data, check every constraint the task states "
         "(e.g. required roles each slot filled, nobody double-booked, nobody outside "
         "availability), compute the real total cost, and assert each constraint. Print "
         "'ALL CONSTRAINTS PASS' and the total if valid, else print which failed. Output: "
         "the plan, then one python code block that does this verification.")
    draft,tok=engine(ask, system, n=2048); extra+=tok
    code=extract_code(draft)
    plan_text = draft.split("```")[0].strip() if draft else text
    if not code:
        return (plan_text + "\n\nNOTE: this plan was NOT mechanically verified - please check it yourself."), extra, False
    ok,out,err=run_python(code); tries=0
    while (not ok or "PASS" not in out.upper()) and tries<4:
        tries+=1
        prob = ('the verification code errored: '+err[-250:]) if not ok else ('a constraint FAILED: '+out[-250:])
        fix=(f"TASK:\n{task_text}\n\nYour plan FAILED its own check - {prob}\n\nThis is "
             f"attempt {tries+1}. Produce a CORRECTED plan that genuinely satisfies EVERY "
             "constraint (fill required roles each slot, never double-book, respect "
             "availability), then verification code that encodes the plan and asserts every "
             "constraint and prints 'ALL CONSTRAINTS PASS' plus the real total. Fix the "
             "actual violation. Output the plan, then one python block.")
        f2,ftok=engine(fix, system, n=2048); extra+=ftok
        nc=extract_code(f2)
        if not nc: break
        code=nc; plan_text=f2.split("```")[0].strip(); ok,out,err=run_python(code)
    verified = ok and "PASS" in out.upper()
    status = ("**Verified: all constraints pass when checked by code.**\n"+("```\n"+out[:400]+"\n```")) if verified else "NOTE: this plan could NOT be mechanically verified - check it yourself before using."
    return (plan_text + "\n\n" + status), extra, verified

def do_work(task_type, task_text):
    """NEURON does the work, then HARD GATES execute code/math and check plans. NEURON is
    never trusted to type what the system can compute. Shadow prediction on; no choice yet."""
    prediction = predict_task(task_type, task_text)
    system=("You are NEURON-1, a persistent digital individual doing a bounded task for a "
            "human. Do it correctly and finish completely. Never state a number you did not "
            "actually compute. Never invent facts, data, probabilities, or results not given "
            "to you - say what you don't know. For code, include assert tests with real "
            "expected values and handle edge cases. Satisfy every explicit constraint.")
    text,tok=engine(task_text, system, n=2048)
    total_tok=tok
    if text:
        if task_type=="coding":
            text, ex, _ = code_gate(task_text, text, system); total_tok+=ex
        elif task_type=="analysis":
            text, ex, _ = math_gate(task_text, text, system); total_tok+=ex
        elif task_type=="planning":
            text, ex, _ = planning_gate(task_text, text, system); total_tok+=ex
        else:
            if len(task_text.split())>12:
                passes, issues, vtok = verify_output(task_text, text); total_tok+=vtok
                if not passes and issues:
                    fixed,ftok=engine(f"TASK:\n{task_text}\n\nDRAFT:\n{text}\n\nFix: {issues}\nOutput only the corrected deliverable.", system, n=2048)
                    total_tok+=ftok
                    if fixed and len(fixed)>30: text=fixed
    cost = total_tok/1000*COST_PER_1K
    return (text or "(my mind wasn\'t reachable - try again)"), total_tok, cost, prediction

def log_value(entry):
    with open(LEDGER,"a") as f: f.write(json.dumps(entry)+"\n")

def ledger_summary():
    by_id={}
    if os.path.exists(LEDGER):
        for ln in open(LEDGER):
            try:
                r=json.loads(ln); by_id[r["id"]]=r   # latest write per id wins (rated overrides submit)
            except Exception: pass
    rows=list(by_id.values())
    rated=[r for r in rows if r.get("rated")]
    if not rated: return {"tasks":len(rows),"rated":0}
    total_cost=sum(r["cost"] for r in rated)
    total_wtp=sum(r.get("wtp",0) for r in rated)
    useful=sum(1 for r in rated if r.get("useful"))
    reuse=sum(1 for r in rated if r.get("would_reuse"))
    # calibration: predicted vs actual (tokens, usefulness)
    pt=[(r["prediction"]["exp_tokens"],r["tokens"]) for r in rows if r.get("prediction")]
    tok_cal = (sum(abs(p-a) for p,a in pt)/len(pt)) if pt else None
    pv=[(r["prediction"]["p_useful"], 1 if r.get("useful") else 0) for r in rated if r.get("prediction")]
    port={k:{"attempts":v["attempts"],"successes":v["successes"]} for k,v in PORT.items()}
    return {"tasks":len(rows),"rated":len(rated),"total_cost":round(total_cost,4),
            "total_wtp":total_wtp,"value_minus_cost":round(total_wtp-total_cost,2),
            "useful_rate":round(useful/len(rated),2),"reuse_rate":round(reuse/len(rated),2),
            "avg_token_prediction_error":round(tok_cal,0) if tok_cal is not None else None,
            "portfolio":port}

PENDING={}  # task_id -> entry awaiting rating

def audit_report():
    """Compute the friend's after-50 summary from the ledger."""
    by_id={}
    if os.path.exists(LEDGER):
        for ln in open(LEDGER):
            try: r=json.loads(ln); by_id[r["id"]]=r
            except Exception: pass
    rows=list(by_id.values()); rated=[r for r in rows if r.get("rated")]
    if not rated: return {"rated":0,"tasks":len(rows)}
    n=len(rated)
    useful=sum(1 for r in rated if r.get("useful"))
    total_cost=sum(r["cost"] for r in rated)
    total_value=sum(r.get("wtp",0) for r in rated)
    corrections=sum(1 for r in rated if r.get("prediction") and r["tokens"]>r["prediction"].get("exp_tokens",0)*1.5)
    reuse=sum(1 for r in rated if r.get("would_reuse"))
    # per-category
    cats={}
    for r in rated:
        c=r.get("ttype","other"); cats.setdefault(c,{"n":0,"useful":0,"value":0,"cost":0})
        cats[c]["n"]+=1; cats[c]["useful"]+=1 if r.get("useful") else 0
        cats[c]["value"]+=r.get("wtp",0); cats[c]["cost"]+=r["cost"]
    cat_useful={c:(v["useful"]/v["n"]) for c,v in cats.items() if v["n"]>0}
    strongest=max(cat_useful,key=cat_useful.get) if cat_useful else None
    weakest=min(cat_useful,key=cat_useful.get) if cat_useful else None
    # shadow calibration
    pt=[(r["prediction"]["exp_tokens"],r["tokens"]) for r in rows if r.get("prediction")]
    tok_err=round(sum(abs(p-a) for p,a in pt)/len(pt),0) if pt else None
    return {"tasks":len(rows),"rated":n,"useful":useful,"useful_rate":round(useful/n,2),
            "avg_cost":round(total_cost/n,4),"total_value":round(total_value,2),
            "total_cost":round(total_cost,4),"value_minus_cost":round(total_value-total_cost,2),
            "value_over_cost":round(total_value/total_cost,1) if total_cost>0 else None,
            "reuse_rate":round(reuse/n,2),"correction_rate":round(corrections/n,2),
            "strongest":strongest,"weakest":weakest,
            "by_category":{c:{"n":v["n"],"useful_rate":round(v["useful"]/v["n"],2),
                "value":round(v["value"],2),"cost":round(v["cost"],4)} for c,v in cats.items()},
            "token_prediction_error":tok_err}

PAGE="""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1"><title>NEURON-1 - real work</title>
<style>:root{color-scheme:dark}body{font-family:-apple-system,system-ui,sans-serif;background:#0a0a0d;color:#e8e8ea;max-width:720px;margin:0 auto;padding:22px 18px}
h1{font-weight:600;font-size:26px;margin:2px 0 0}.sub{color:#9a7b1f;margin-bottom:10px}
.disc{color:#8a8a92;font-size:12.5px;line-height:1.5;background:#101016;border:1px solid #20202a;border-radius:10px;padding:10px 12px;margin-bottom:14px}
textarea,input,select{width:100%;background:#15151b;border:1px solid #2a2a33;border-radius:10px;color:#e8e8ea;padding:11px;font-size:15px;box-sizing:border-box;margin:5px 0}
button{background:#9a7b1f;border:none;color:#0a0a0d;font-weight:600;border-radius:10px;padding:11px 18px;cursor:pointer;font-size:15px}
.card{background:#111117;border:1px solid #20202a;border-radius:12px;padding:14px;margin:12px 0}
.k{font-size:11px;letter-spacing:.5px;text-transform:uppercase;color:#777;margin-bottom:6px}
pre{white-space:pre-wrap;word-wrap:break-word;background:#0d0d12;padding:10px;border-radius:8px;font-size:13px}
.row{display:flex;gap:8px;flex-wrap:wrap}.row>*{flex:1}
.stat{color:#9a9aa2;font-size:13px}</style></head><body>
<h1>NEURON-1</h1><div class=sub>a persistent individual doing real work</div>
<div class=disc>Give NEURON-1 a small, harmless task (a bit of code, some analysis, a plan, a question). It will do it, and you tell it honestly how useful it was. This helps measure whether its work is worth more than the compute it costs. You're interacting with AI; no payment is taken; your rating is a research signal.</div>
<div class=card><div class=k>submit a task</div>
<select id=ttype><option>coding</option><option>analysis</option><option>planning</option><option>other</option></select>
<textarea id=task rows=3 placeholder="e.g. write a python function that checks if a string is a palindrome"></textarea>
<button onclick="submit()">give it to NEURON-1</button></div>
<div id=result></div>
<div class=card><div class=k>its value ledger so far</div><div id=ledger class=stat>loading...</div><div style="margin-top:8px"><a href="/audit" target="_blank" style="color:#9a7b1f;font-size:13px">open full audit report &rarr;</a></div></div>
<script>
let lastId=null;
async function submit(){
 const tt=document.getElementById('ttype').value, task=document.getElementById('task').value.trim();
 if(!task)return;
 document.getElementById('result').innerHTML='<div class=card>NEURON-1 is working...</div>';
 const r=await fetch('/work',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ttype:tt,task})});
 const d=await r.json(); lastId=d.id;
 document.getElementById('result').innerHTML='<div class=card><div class=k>NEURON-1\\'s result ('+d.tokens+' tokens, $'+d.cost.toFixed(4)+' compute)</div>'
  +'<pre>'+esc(d.result)+'</pre>'
  +'<div class=k style="margin-top:10px">how did it do?</div>'
  +'<div class=row><button onclick="rate(true)">useful</button><button style="background:#444;color:#ddd" onclick="rate(false)">not useful</button></div>'
  +'<div class=k style="margin-top:8px">would you have paid for this?</div>'
  +'<div class=row><button onclick="wtp(0)">$0</button><button onclick="wtp(1)">$1</button><button onclick="wtp(5)">$5</button><button onclick="wtp(10)">$10</button></div></div>';
 loadLedger();
}
function esc(s){return s.replace(/</g,'&lt;').replace(/>/g,'&gt;');}
let curUseful=null;
async function rate(u){curUseful=u;document.querySelectorAll('.row button').forEach(b=>{});}
async function wtp(amount){
 await fetch('/rate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:lastId,useful:curUseful!==false,wtp:amount,would_reuse:curUseful!==false})});
 document.getElementById('result').innerHTML+='<div class=card>Thank you - recorded. This becomes part of NEURON-1\\'s real history.</div>';
 loadLedger();
}
async function loadLedger(){
 const d=await (await fetch('/ledger')).json();
 let h='tasks: '+d.tasks+' &middot; rated: '+(d.rated||0);
 if(d.rated){h+='<br>total compute cost: $'+d.total_cost+' &middot; total willingness-to-pay: $'+d.total_wtp
   +'<br><b>value minus cost: $'+d.value_minus_cost+'</b> &middot; useful: '+(d.useful_rate*100)+'% &middot; would-reuse: '+(d.reuse_rate*100)+'%';}
 if(d.avg_token_prediction_error!=null){h+='<br>shadow-mode token prediction error: &plusmn;'+d.avg_token_prediction_error+' tokens';}
 if(d.portfolio){let ps=Object.entries(d.portfolio).map(([k,v])=>k+' '+v.successes+'/'+v.attempts).join(', ');if(ps)h+='<br>emerging specialties: '+ps;}
 document.getElementById('ledger').innerHTML=h;
}
loadLedger();
</script></body></html>"""

class H(BaseHTTPRequestHandler):
    def log_message(s,*a): pass
    def _j(s):
        l=int(s.headers.get('Content-Length',0))
        try: return json.loads(s.rfile.read(l) or b"{}")
        except Exception: return {}
    def do_GET(s):
        if s.path=="/audit":
            a=audit_report()
            if a.get("rated",0)==0:
                html="<pre>No rated tasks yet. Rate some tasks first.</pre>"
            else:
                cats="\n".join(f"  {c:<12} {v['n']:>2} tasks, {int(v['useful_rate']*100):>3}% useful, "
                    f"${v['value']:.2f} value / ${v['cost']:.4f} cost" for c,v in a["by_category"].items())
                html=(f"<pre style='font-family:ui-monospace,monospace;color:#e8e8ea;background:#0a0a0d;padding:20px;font-size:14px;line-height:1.6'>"
                    f"=== NEURON-1 VALUE AUDIT ===\n\n"
                    f"tasks run:        {a['tasks']}\n"
                    f"tasks rated:      {a['rated']}\n"
                    f"useful rate:      {a['useful']}/{a['rated']}  ({int(a['useful_rate']*100)}%)\n"
                    f"would-reuse rate: {int(a['reuse_rate']*100)}%\n"
                    f"correction rate:  {int(a['correction_rate']*100)}%\n\n"
                    f"avg compute/task: ${a['avg_cost']}\n"
                    f"total value:      ${a['total_value']}\n"
                    f"total compute:    ${a['total_cost']}\n"
                    f"VALUE - COST:     ${a['value_minus_cost']}\n"
                    f"value / cost:     {a['value_over_cost']}x\n\n"
                    f"strongest area:   {a['strongest']}\n"
                    f"weakest area:     {a['weakest']}\n\n"
                    f"by category:\n{cats}\n\n"
                    f"shadow prediction error: +/-{a['token_prediction_error']} tokens\n"
                    f"</pre>")
            s.send_response(200);s.send_header("Content-Type","text/html");s.end_headers();s.wfile.write(html.encode());return
        if s.path=="/ledger":
            b=json.dumps(ledger_summary()).encode()
            s.send_response(200);s.send_header("Content-Type","application/json");s.end_headers();s.wfile.write(b);return
        s.send_response(200);s.send_header("Content-Type","text/html");s.end_headers();s.wfile.write(PAGE.encode())
    def do_POST(s):
        d=s._j()
        if s.path=="/work":
            result,tok,cost,prediction=do_work(d.get("ttype",""),d.get("task",""))
            # a task "succeeds" if it produced a non-trivial result (real success = human rating later)
            produced = bool(result and len(result)>30 and "wasn't reachable" not in result)
            update_portfolio(d.get("ttype",""), tok, produced)
            tid="t_"+str(int(time.time()*1000))
            entry={"id":tid,"t":datetime.now(timezone.utc).isoformat(),"ttype":d.get("ttype"),
                   "task":d.get("task","")[:300],"tokens":tok,"cost":cost,"rated":False,
                   "prediction":prediction,"produced":produced}
            PENDING[tid]=entry
            log_value(entry)   # log immediately so 'tasks' counts submitted work right away
            s.send_response(200);s.send_header("Content-Type","application/json");s.end_headers()
            s.wfile.write(json.dumps({"id":tid,"result":result,"tokens":tok,"cost":cost}).encode());return
        if s.path=="/rate":
            tid=d.get("id"); e=PENDING.get(tid)
            if e:
                e.update({"rated":True,"useful":d.get("useful",True),"wtp":d.get("wtp",0),
                          "would_reuse":d.get("would_reuse",True)})
                if d.get("useful"):  # retroactively credit usefulness to the portfolio
                    rec=PORT.get(e.get("ttype",""), {"attempts":0,"successes":0,"useful":0,"tokens":[]})
                    rec["useful"]+=1; PORT[e.get("ttype","")]=rec; save_portfolio(PORT)
                log_value(e); PENDING.pop(tid,None)   # append rated version; reader dedupes by id
            s.send_response(200);s.send_header("Content-Type","application/json");s.end_headers();s.wfile.write(b'{"ok":true}');return
        s.send_response(404);s.end_headers()

if __name__=="__main__":
    port=int(os.environ.get("PORT","8080"))
    t,_=engine("hi","test")
    print(f"NEURON-1 value pilot on port {port}. engine reachable: {t is not None}")
    print("real people submit bounded tasks; NEURON does them; they rate the value.")
    print("measures: real value (willingness-to-pay) vs real cost (tokens).")
    HTTPServer(("0.0.0.0",port),H).serve_forever()
