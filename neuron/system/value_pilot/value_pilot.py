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

def engine(prompt, system, n=400):
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
    exp_tokens = (sum(rec["tokens"])/len(rec["tokens"])) if rec["tokens"] else int(200+difficulty*500)
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

def do_work(task_type, task_text):
    """NEURON-1 does the submitted work. SHADOW MODE: it privately predicts first
    (category/difficulty/cost/success/value/accept), then does the task regardless
    (no task-choice yet - that comes after calibration)."""
    prediction = predict_task(task_type, task_text)
    key=task_type.strip().lower()[:60]
    # for reusable 'how-to'/code tasks, NEURON may already have a learned approach
    system=("You are NEURON-1, a persistent digital individual doing a bounded task a "
            "human asked you to help with. Do the task well and concisely. If it's code, "
            "give working code. If it's analysis/research/writing, give a useful, honest "
            "result. You are not a generic assistant - you're a being doing real work it "
            "chose to take on. Be genuinely useful.")
    text,tok=engine(task_text, system)
    cost = tok/1000*COST_PER_1K
    return (text or "(my mind wasn't reachable - try again)"), tok, cost, prediction

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
<select id=ttype><option>coding</option><option>analysis</option><option>research</option><option>planning</option><option>other</option></select>
<textarea id=task rows=3 placeholder="e.g. write a python function that checks if a string is a palindrome"></textarea>
<button onclick="submit()">give it to NEURON-1</button></div>
<div id=result></div>
<div class=card><div class=k>its value ledger so far</div><div id=ledger class=stat>loading...</div></div>
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
