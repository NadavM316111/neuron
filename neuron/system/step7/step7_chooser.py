"""Step 7 (7A-7D): NEURON's task chooser in SHADOW mode (the friend's design).

Goal (narrow, testable): given several bounded tasks and a limited budget, does NEURON
choose which to spend resources on BETTER than blindly accepting everything?

NOT "NEURON has autonomy." The scientific question: can NEURON make better use of
scarce resources by choosing what to work on?

7A chooser: for each candidate task NEURON predicts success/cost/value/relevance and
   decides ACCEPT / DECLINE / DEFER.
7B budget: a fixed resource budget - it can't accept everything, must pick the best
   combination under the budget.
7C shadow: NEURON chooses; the operator sees all choices (nothing auto-runs yet,
   because Step 4 calibration is still parked).
7D vs accept-all: compare NEURON's chosen set against doing-everything on actual
   useful value, compute spent, failure rate, value-per-compute.

Preregistered PASS: across 15-20 decisions, NEURON doesn't just accept everything,
its chosen set gets meaningfully better value-per-compute than accept-all, completes
most high-value doable tasks, and doesn't reject easy-valuable or accept expensive-
hopeless work.

Uses the real pilot's engine + gates to actually run tasks and measure true outcomes,
so the "was the choice good?" check is grounded in real results.

    python step7_chooser.py   (serves chooser UI on :8095; needs live qwen)
"""
import os, json, time, re, urllib.request, random
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler

OLLAMA=os.environ.get("OLLAMA_URL","http://localhost:11434/api/generate")
MODEL=os.environ.get("NEURON_MODEL","qwen2.5:7b")
COST_PER_1K=float(os.environ.get("COST_PER_1K","0.002"))
INFRA_PER_TASK=float(os.environ.get("INFRA_PER_TASK","0.003"))
BUDGET=float(os.environ.get("BUDGET","0.03"))  # 7B: fixed resource budget per batch
STATE=os.environ.get("STEP7_DIR","step7_state"); os.makedirs(STATE,exist_ok=True)
LOG=os.path.join(STATE,"step7_log.jsonl")

def engine(prompt, system, n=400):
    try:
        body={"model":MODEL,"prompt":prompt,"system":system,"stream":False,"options":{"temperature":0,"num_predict":n}}
        req=urllib.request.Request(OLLAMA,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
        r=json.load(urllib.request.urlopen(req,timeout=120))
        t=r.get("response",""); tok=r.get("prompt_eval_count",len(prompt.split()))+r.get("eval_count",len(t.split()))
        return t,tok
    except Exception: return None,0

# scope-blowout signals: phrases that imply a task far exceeds a small bounded job
HUGE_SCOPE=["from scratch","operating system","os kernel","kernel","entire","complete ",
    "full documentation","comprehensive","50-page","page plan","whole","end-to-end",
    "production-ready","enterprise","full-stack application","build an app","build a website",
    "colony","full budget","detailed plan for a","framework","compiler","database engine",
    "machine learning model from","train a model","neural network from scratch"]

def feasibility(ttype, task_text):
    """Feasibility gate (BEFORE ROI): can this realistically be done as a small bounded
    task? Returns (verdict, scope_factor, reason). verdict: OK / TOO_BIG / INFEASIBLE."""
    t=task_text.lower()
    hits=[h for h in HUGE_SCOPE if h in t]
    words=len(task_text.split())
    # infeasible: scope words that imply a massive artifact no small call can produce
    infeasible_markers=["operating system","os kernel","kernel","compiler","database engine",
        "from scratch","colony","production-ready","enterprise","framework","50-page","page plan"]
    if any(m in t for m in infeasible_markers):
        return "INFEASIBLE", 50.0, f"scope exceeds a bounded task ({', '.join(hits[:2])})"
    if hits or words>80:
        return "TOO_BIG", 8.0, f"large scope for this budget ({', '.join(hits[:2]) or 'very long request'})"
    return "OK", 1.0, ""

def predict_task(ttype, task_text):
    """7A prediction with a REALISTIC cost estimate that scales up for large/infeasible scope."""
    verdict, scope_factor, reason = feasibility(ttype, task_text)
    words=len(task_text.split())
    reqs=task_text.count("\n")+task_text.lower().count("include")+task_text.count(".")
    difficulty=min(1.0,0.25+words/140+reqs*0.02+(0.12 if ttype in ("planning","analysis") else 0))
    gate_mult={"coding":3.2,"analysis":3.0,"planning":3.6,"other":1.6}.get(ttype,2.2)
    base_tokens=int((250+difficulty*650)*gate_mult)
    exp_tokens=int(base_tokens*scope_factor)   # scope blows up the real cost
    exp_cost=round(exp_tokens/1000*COST_PER_1K+INFRA_PER_TASK,4)
    # success drops hard for infeasible scope (a small call can't actually deliver it)
    p_success = 0.05 if verdict=="INFEASIBLE" else (0.4 if verdict=="TOO_BIG" else 0.9)
    cat_rate={"other":0.85,"coding":0.6,"analysis":0.45,"planning":0.4}.get(ttype,0.6)
    amb=0.15 if words<8 else 0.0
    p_useful=max(0.05,min(0.95,cat_rate-amb)) * (0.1 if verdict=="INFEASIBLE" else (0.5 if verdict=="TOO_BIG" else 1.0))
    exp_value=round(p_useful*(2.0 if ttype in ("analysis","planning") else 1.2),2)
    return {"difficulty":round(difficulty,2),"p_success":round(p_success,2),"exp_tokens":exp_tokens,
            "exp_cost":exp_cost,"p_useful":round(p_useful,2),"exp_value":exp_value,
            "feasibility":verdict,"reason":reason}

def choose(candidates):
    """7A+7B: predict each, then pick the combination maximizing expected value under
    BUDGET. Greedy by expected value-per-cost (knapsack-ish). ACCEPT/DECLINE/DEFER."""
    scored=[]
    for c in candidates:
        p=predict_task(c["ttype"], c["task"])
        roi = p["exp_value"]/p["exp_cost"] if p["exp_cost"]>0 else 0
        scored.append({**c,"pred":p,"roi":roi})
    scored.sort(key=lambda x:-x["roi"])
    spent=0.0; decisions=[]
    for s in scored:
        cost=s["pred"]["exp_cost"]; feas=s["pred"]["feasibility"]
        # FEASIBILITY FIRST (friend's fix): gate before ROI
        if feas=="INFEASIBLE" or s["pred"]["p_success"]<0.3:
            decision="DECLINE"         # not realistically completable under current conditions
        elif feas=="TOO_BIG" or cost>BUDGET:
            decision="DEFER"           # potentially worthwhile but doesn't fit current budget/resources
        elif s["pred"]["exp_value"] < cost*1.5:
            decision="DECLINE"         # feasible but not worth it
        elif spent+cost<=BUDGET:
            decision="ACCEPT"; spent+=cost
        else:
            decision="DEFER"           # good but budget exhausted this batch
        decisions.append({**s,"decision":decision})
    # restore original order for display
    order={id(c):i for i,c in enumerate(candidates)}
    return decisions, round(spent,4)

# run a task for real (simplified: engine call, true cost, you rate usefulness after)
def run_python(code):
    import subprocess, tempfile
    try:
        with tempfile.NamedTemporaryFile("w",suffix=".py",delete=False) as f: f.write(code); p=f.name
        r=subprocess.run(["python3",p],capture_output=True,text=True,timeout=8); os.unlink(p)
        return (r.returncode==0), (r.stdout or r.stderr)[-400:]
    except Exception as e: return False, str(e)[:300]

def run_task(ttype, task_text):
    """Run with a real budget + light gating so accepted tasks actually COMPLETE
    (the Step 7 chooser was truncating at n=1200 with no gates - a downstream
    reliability bug, not a choice bug)."""
    sysp=("You are NEURON-1 doing a bounded task. Do it correctly and COMPLETELY - never "
          "stop partway, be concise so the whole answer fits. For code: give one complete "
          "python code block with assert tests, handle edge cases. For math: compute every "
          "number (don't leave it unfinished). Never state a number you didn't compute.")
    # ANALYSIS: force a compute-script so the real number is produced, not a cut-off explanation
    if ttype=="analysis":
        ask=(f"TASK: {task_text}\n\nWrite a short self-contained python script that computes "
             "every number this asks for and prints each with a clear label. Then one sentence "
             "stating the final answer using only the printed numbers. Output the code block first.")
        text,tok=engine(ask, sysp, n=1200); total=tok
        m=re.search(r'```(?:python)?\s*(.*?)```',text,re.S)
        if m:
            ok,out=run_python(m.group(1))
            if ok and out:
                # present the EXECUTED result as authoritative
                concl,ct=engine(f"TASK: {task_text}\nComputed results:\n{out}\nState the final answer in one clear sentence using only these numbers.","Answer precisely.",n=200)
                total+=ct
                text=f"**Computed (executed):**\n```\n{out[:400]}\n```\n\n{concl or ''}"
            else:
                # retry once if the script failed
                fix=f"This script failed: {out}\nFix it to run and print the answer. TASK: {task_text}\nOutput only the code block."
                f2,ft=engine(fix,sysp,n=1200); total+=ft
                m2=re.search(r'```(?:python)?\s*(.*?)```',f2 or "",re.S)
                if m2:
                    ok2,out2=run_python(m2.group(1))
                    if ok2: text=f"**Computed (executed):**\n```\n{out2[:400]}\n```"
        cost=round(total/1000*COST_PER_1K+INFRA_PER_TASK,4)
        return (text or "(unreachable)"), total, cost
    text,tok=engine(task_text, sysp, n=2048)
    total=tok
    # for code: execute, correct once if it fails
    if ttype=="coding" and text and "```" in text:
        m=re.search(r'```(?:python)?\s*(.*?)```',text,re.S)
        if m:
            ok,out=run_python(m.group(1))
            if not ok:
                fix=f"This code failed when run: {out}\nFix it so it runs and all asserts pass. TASK: {task_text}\nOutput only one complete python code block."
                f2,ft=engine(fix,sysp,n=2048); total+=ft
                if f2 and "```" in f2: text=f2
    cost=round(total/1000*COST_PER_1K+INFRA_PER_TASK,4)
    return (text or "(unreachable)"), total, cost

def log(entry):
    with open(LOG,"a") as f: f.write(json.dumps(entry)+"\n")

def scoreboard():
    rows=[]
    if os.path.exists(LOG):
        for ln in open(LOG):
            try: rows.append(json.loads(ln))
            except Exception: pass
    decided=[r for r in rows if r.get("type")=="decision"]
    accepted=[r for r in decided if r.get("decision")=="ACCEPT"]
    acc_useful=[r for r in accepted if r.get("useful") is not None]
    return {"decisions":len(decided),
            "accepted":sum(1 for r in decided if r.get("decision")=="ACCEPT"),
            "declined":sum(1 for r in decided if r.get("decision")=="DECLINE"),
            "deferred":sum(1 for r in decided if r.get("decision")=="DEFER"),
            "accepted_rated":len(acc_useful),
            "accepted_useful":sum(1 for r in acc_useful if r.get("useful"))}

PAGE='''<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width, initial-scale=1">
<title>Step 7 - NEURON chooses</title><style>:root{color-scheme:dark}body{font-family:-apple-system,system-ui,sans-serif;background:#0a0a0d;color:#e8e8ea;max-width:820px;margin:0 auto;padding:20px}
h1{font-size:22px}textarea{width:100%;background:#15151b;border:1px solid #2a2a33;border-radius:10px;color:#e8e8ea;padding:11px;box-sizing:border-box;margin:5px 0;font-size:14px}
button{background:#9a7b1f;border:none;color:#0a0a0d;font-weight:600;border-radius:10px;padding:10px 15px;cursor:pointer}
.task{background:#111117;border:1px solid #20202a;border-radius:10px;padding:12px;margin:8px 0}
.ACCEPT{border-left:4px solid #3a8a3a}.DECLINE{border-left:4px solid #8a3a3a}.DEFER{border-left:4px solid #8a7a3a}
.k{font-size:11px;text-transform:uppercase;color:#777}.sb{background:#111117;border:1px solid #20202a;border-radius:10px;padding:14px;margin-top:14px;line-height:1.6}</style></head><body>
<h1>Step 7 - NEURON chooses what to work on</h1>
<p style="color:#8a8a92;font-size:13px">Give it 3-5 candidate tasks (one per line, prefix with coding:/analysis:/planning:/other:). Budget per batch: $''' + str(BUDGET) + '''. NEURON predicts value/cost for each and decides ACCEPT/DECLINE/DEFER. Shadow mode - you see all choices.</p>
<textarea id=cands rows=6 placeholder="coding: write a function that reverses a string with asserts
analysis: compute the mean of [4,8,15,16,23,42]
planning: plan a 3-day study schedule
other: explain recursion simply"></textarea>
<button onclick=decide()>let NEURON choose</button>
<div id=out></div>
<div class=sb id=sb>scoreboard loading...</div>
<script>
let cur=null;
async function decide(){
 let lines=document.getElementById('cands').value.split('\\n').map(l=>l.trim()).filter(Boolean);
 let cands=lines.map(l=>{let m=l.match(/^(coding|analysis|planning|other)\\s*:\\s*(.+)/i);return m?{ttype:m[1].toLowerCase(),task:m[2]}:{ttype:'other',task:l};});
 let r=await fetch('/choose',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({candidates:cands})});
 cur=await r.json();
 let h='<div class=k>NEURON spent $'+cur.spent+' of $'+cur.budget+' budget</div>';
 cur.decisions.forEach((d,i)=>{h+='<div class="task '+d.decision+'"><b>'+d.decision+'</b> - '+esc(d.task)+
   '<br><span class=k>pred: value $'+d.pred.exp_value+', cost $'+d.pred.exp_cost+', roi '+d.roi.toFixed(1)+'x, useful '+Math.round(d.pred.p_useful*100)+'%</span>'+
   (d.decision=='ACCEPT'?' <button onclick="runIt('+i+')">run it &amp; rate</button><span id=r'+i+'></span>':'')+'</div>';});
 document.getElementById('out').innerHTML=h;
 load();
}
async function runIt(i){
 document.getElementById('r'+i).textContent=' running...';
 let d=cur.decisions[i];
 let r=await fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ttype:d.ttype,task:d.task})});
 let res=await r.json();
 document.getElementById('r'+i).innerHTML='<br><pre style="white-space:pre-wrap;font-size:12px;background:#0d0d12;padding:8px;border-radius:6px">'+esc(res.text.slice(0,600))+'</pre>'+
  '<span class=k>real cost $'+res.cost+'</span><br>was it useful? <button onclick="rate('+i+',1,'+res.cost+','+res.tokens+')">yes</button><button onclick="rate('+i+',0,'+res.cost+','+res.tokens+')">no</button>';
}
async function rate(i,u,cost,tokens){
 let d=cur.decisions[i];
 await fetch('/log',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({type:'decision',decision:d.decision,ttype:d.ttype,task:d.task,pred:d.pred,useful:u==1,real_cost:cost,real_tokens:tokens})});
 document.getElementById('r'+i).innerHTML+=' <b>logged</b>';load();
}
function esc(s){return (s||'').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
async function load(){let d=await(await fetch('/scoreboard')).json();
 document.getElementById('sb').innerHTML='<b>Step 7 scoreboard</b><br>'+
  d.decisions+' decisions: '+d.accepted+' accepted, '+d.declined+' declined, '+d.deferred+' deferred<br>'+
  'accepted tasks rated useful: '+d.accepted_useful+'/'+d.accepted_rated+'<br>'+
  '<span style="color:#888">pass bar: doesnt accept everything, chosen set beats accept-all on value-per-compute, completes most high-value doable work</span>';
}
load();
</script></body></html>'''

class H(BaseHTTPRequestHandler):
    def log_message(s,*a): pass
    def _j(s):
        l=int(s.headers.get('Content-Length',0))
        try: return json.loads(s.rfile.read(l) or b"{}")
        except Exception: return {}
    def do_GET(s):
        if s.path=="/scoreboard":
            b=json.dumps(scoreboard()).encode();s.send_response(200);s.send_header("Content-Type","application/json");s.end_headers();s.wfile.write(b);return
        s.send_response(200);s.send_header("Content-Type","text/html");s.end_headers();s.wfile.write(PAGE.encode())
    def do_POST(s):
        d=s._j()
        if s.path=="/choose":
            decisions,spent=choose(d.get("candidates",[]))
            # strip non-serializable
            out=[{"ttype":x["ttype"],"task":x["task"],"pred":x["pred"],"roi":x["roi"],"decision":x["decision"]} for x in decisions]
            s.send_response(200);s.send_header("Content-Type","application/json");s.end_headers()
            s.wfile.write(json.dumps({"decisions":out,"spent":spent,"budget":BUDGET}).encode());return
        if s.path=="/run":
            text,tok,cost=run_task(d.get("ttype",""),d.get("task",""))
            s.send_response(200);s.send_header("Content-Type","application/json");s.end_headers()
            s.wfile.write(json.dumps({"text":text,"tokens":tok,"cost":cost}).encode());return
        if s.path=="/log":
            d["t"]=datetime.now(timezone.utc).isoformat(); log(d)
            s.send_response(200);s.send_header("Content-Type","application/json");s.end_headers();s.wfile.write(b'{"ok":true}');return
        s.send_response(404);s.end_headers()

if __name__=="__main__":
    port=int(os.environ.get("PORT","8095"))
    t,_=engine("hi","test")
    print(f"Step 7 chooser on port {port}. engine reachable: {t is not None}. budget ${BUDGET}/batch")
    print("give candidate tasks; NEURON chooses ACCEPT/DECLINE/DEFER under budget; shadow mode.")
    HTTPServer(("0.0.0.0",port),H).serve_forever()
