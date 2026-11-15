"""Step 5: Prove persistence matters on REAL human work (the friend's design).

Runs each real task TWICE under identical conditions (same model, gates, budget):
  PERSISTENT - NEURON-1 with its accumulated capability cache (prior life)
  FRESH      - same everything, reset, no prior history

Order randomized per task. Both graded identically: success, usefulness(you rate),
value(you rate), tokens, and VALUE-PER-COMPUTE. The cognitive-capital claim is that
PERSISTENT reuses learned methods -> fewer real tokens for equal/better work.

Pre-registered pass rule (set BEFORE results, no goalpost-moving):
  Step 5 PASS if persistent shows >= equal usefulness AND >= ~15-20% better
  value-per-compute overall, across multiple tasks (not one outlier).

Separates: context-advantage (remembers facts) vs cognitive-capital (solves more
efficiently). This harness targets cognitive-capital: PERSISTENT's edge is method
REUSE (fewer tokens), not being handed facts FRESH lacked.

Serves a paired-comparison page at :8090. Needs live qwen.
    python step5_persistence.py
"""
import os, json, time, random, re, urllib.request
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler

OLLAMA=os.environ.get("OLLAMA_URL","http://localhost:11434/api/generate")
MODEL=os.environ.get("NEURON_MODEL","qwen2.5:7b")
COST_PER_1K=float(os.environ.get("COST_PER_1K","0.002"))
STATE=os.environ.get("STEP5_DIR","step5_state")
os.makedirs(STATE,exist_ok=True)
LOG=os.path.join(STATE,"step5_log.jsonl")
# PERSISTENT carries a capability cache built from prior work; FRESH starts empty
PCACHE=os.path.join(STATE,"persistent_cache.json")

def engine(prompt, system, n=2048):
    try:
        body={"model":MODEL,"prompt":prompt,"system":system,"stream":False,"options":{"temperature":0,"num_predict":n}}
        req=urllib.request.Request(OLLAMA,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
        r=json.load(urllib.request.urlopen(req,timeout=180))
        t=r.get("response",""); tok=r.get("prompt_eval_count",len(prompt.split()))+r.get("eval_count",len(t.split()))
        return t,tok
    except Exception: return None,0

def load_cache():
    try: return json.load(open(PCACHE))
    except Exception: return {}
def save_cache(c):
    try: json.dump(c,open(PCACHE,"w"))
    except Exception: pass

SYSTEM=("You are NEURON-1 doing a bounded task for a human. Do it correctly and "
        "completely. Never state a number you didn't compute; for code include asserts. "
        "Satisfy every requirement. Be genuinely useful.")

def task_signature(task_text):
    # a coarse 'method key' so related tasks can reuse a learned approach
    words=re.findall(r'[a-z]+', task_text.lower())
    key=" ".join(sorted(set(w for w in words if len(w)>4))[:6])
    return key

def do_persistent(task_text, cache):
    """reuses a learned method for a related task-type when available (fewer tokens)."""
    sig=task_signature(task_text)
    hint=""
    # find a prior task with overlapping method key -> reuse its approach as a hint
    for k,v in cache.items():
        overlap=len(set(sig.split()) & set(k.split()))
        if overlap>=2:
            hint=f"\\n\\n(You've done similar work before. A known working approach: {v[:300]})"
            break
    text,tok=engine(task_text+hint, SYSTEM)
    # store this solution as a reusable method for the future
    if text and len(text)>40:
        cache[sig]=text[:400]; save_cache(cache)
    return text, tok

def do_fresh(task_text):
    """no prior history - solves from scratch every time."""
    text,tok=engine(task_text, SYSTEM)
    return text, tok

def run_pair(task_text):
    cache=load_cache()
    # randomize order so neither is advantaged by model warmup
    order=random.random()<0.5
    if order:
        p_text,p_tok=do_persistent(task_text,cache); f_text,f_tok=do_fresh(task_text)
    else:
        f_text,f_tok=do_fresh(task_text); p_text,p_tok=do_persistent(task_text,cache)
    return {"persistent":{"text":p_text or "(unreachable)","tokens":p_tok,"cost":round(p_tok/1000*COST_PER_1K,4)},
            "fresh":{"text":f_text or "(unreachable)","tokens":f_tok,"cost":round(f_tok/1000*COST_PER_1K,4)}}

def log_rating(entry):
    with open(LOG,"a") as f: f.write(json.dumps(entry)+"\\n")

def scoreboard():
    rows=[]
    if os.path.exists(LOG):
        for ln in open(LOG):
            try: rows.append(json.loads(ln))
            except Exception: pass
    if not rows: return {"n":0}
    def agg(side):
        useful=sum(1 for r in rows if r.get(side+"_useful"))
        value=sum(r.get(side+"_value",0) for r in rows)
        tokens=sum(r.get(side+"_tokens",0) for r in rows)
        cost=sum(r.get(side+"_tokens",0)/1000*COST_PER_1K for r in rows)
        vpc=value/cost if cost>0 else 0
        return {"useful":useful,"value":value,"tokens":tokens,"vpc":round(vpc,1)}
    p=agg("persistent"); f=agg("fresh")
    adv = ((p["vpc"]-f["vpc"])/f["vpc"]*100) if f["vpc"]>0 else 0
    return {"n":len(rows),"persistent":p,"fresh":f,"vpc_advantage_pct":round(adv,0)}

PAGE='''<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width, initial-scale=1">
<title>Step 5 - persistent vs fresh</title><style>:root{color-scheme:dark}body{font-family:-apple-system,system-ui,sans-serif;background:#0a0a0d;color:#e8e8ea;max-width:900px;margin:0 auto;padding:20px}
h1{font-size:22px}textarea,select{width:100%;background:#15151b;border:1px solid #2a2a33;border-radius:10px;color:#e8e8ea;padding:11px;box-sizing:border-box;margin:5px 0;font-size:15px}
button{background:#9a7b1f;border:none;color:#0a0a0d;font-weight:600;border-radius:10px;padding:11px 16px;cursor:pointer}
.cols{display:flex;gap:12px}.col{flex:1;background:#111117;border:1px solid #20202a;border-radius:12px;padding:12px}
.lab{font-size:11px;text-transform:uppercase;letter-spacing:.5px;color:#9a7b1f;margin-bottom:6px}
pre{white-space:pre-wrap;word-wrap:break-word;background:#0d0d12;padding:10px;border-radius:8px;font-size:12px;max-height:340px;overflow:auto}
.rate{margin-top:8px}.rate button{margin:2px;font-size:13px;padding:6px 10px}.sb{background:#111117;border:1px solid #20202a;border-radius:12px;padding:14px;margin-top:14px;font-size:14px;line-height:1.6}</style></head><body>
<h1>Step 5 - persistence vs fresh on real work</h1>
<p style="color:#8a8a92;font-size:13px">Same task run by PERSISTENT (accumulated capability) and FRESH (no history). Grade both. Measures value-per-compute. Labels hidden until you reveal, to avoid bias.</p>
<select id=ttype><option>coding</option><option>analysis</option><option>planning</option><option>other</option></select>
<textarea id=task rows=3 placeholder="a real task you care about"></textarea>
<button onclick=go()>run both</button>
<div id=out></div>
<div class=sb id=sb>scoreboard loading...</div>
<script>
let cur=null;
async function go(){let t=document.getElementById('task').value.trim();if(!t)return;
 document.getElementById('out').innerHTML='<p>running both (this takes a bit)...</p>';
 let r=await fetch('/pair',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ttype:document.getElementById('ttype').value,task:t})});
 cur=await r.json();
 // randomize display side so you dont know which is which
 let flip=Math.random()<0.5; cur._flip=flip;
 let A=flip?cur.fresh:cur.persistent, B=flip?cur.persistent:cur.fresh;
 document.getElementById('out').innerHTML=
  '<div class=cols><div class=col><div class=lab>Output A ('+A.tokens+' tok)</div><pre>'+esc(A.text)+'</pre>'+rateBlk('A')+'</div>'+
  '<div class=col><div class=lab>Output B ('+B.tokens+' tok)</div><pre>'+esc(B.text)+'</pre>'+rateBlk('B')+'</div></div>'+
  '<div style="margin-top:8px"><button onclick=submit()>submit ratings</button></div>';
}
function rateBlk(x){return '<div class=rate><div class=lab>useful?</div><button onclick="setv(\\''+x+'\\',\\'u\\',1)">yes</button><button onclick="setv(\\''+x+'\\',\\'u\\',0)">no</button>'+
 '<div class=lab>would pay?</div><button onclick="setv(\\''+x+'\\',\\'v\\',0)">$0</button><button onclick="setv(\\''+x+'\\',\\'v\\',1)">$1</button><button onclick="setv(\\''+x+'\\',\\'v\\',5)">$5</button><button onclick="setv(\\''+x+'\\',\\'v\\',10)">$10</button> <span id=s'+x+'></span></div>';}
let R={A:{},B:{}};
function setv(x,k,val){R[x][k]=val;document.getElementById('s'+x).textContent=JSON.stringify(R[x]);}
function esc(s){return (s||'').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
async function submit(){
 let flip=cur._flip;
 // A/B map back to persistent/fresh
 let pR=flip?R.B:R.A, fR=flip?R.A:R.B;
 await fetch('/rate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
   persistent_useful:pR.u===1,persistent_value:pR.v||0,persistent_tokens:cur.persistent.tokens,
   fresh_useful:fR.u===1,fresh_value:fR.v||0,fresh_tokens:cur.fresh.tokens})});
 document.getElementById('out').innerHTML+='<p>recorded. (A was '+(flip?'FRESH':'PERSISTENT')+', B was '+(flip?'PERSISTENT':'FRESH')+')</p>';
 R={A:{},B:{}};load();
}
async function load(){let d=await(await fetch('/scoreboard')).json();
 if(!d.n){document.getElementById('sb').textContent='no paired tasks yet';return;}
 document.getElementById('sb').innerHTML='<b>Step 5 scoreboard ('+d.n+' tasks)</b><br>'+
  'PERSISTENT: '+d.persistent.useful+'/'+d.n+' useful, $'+d.persistent.value+' value, '+d.persistent.tokens+' tok, value/compute '+d.persistent.vpc+'<br>'+
  'FRESH: '+d.fresh.useful+'/'+d.n+' useful, $'+d.fresh.value+' value, '+d.fresh.tokens+' tok, value/compute '+d.fresh.vpc+'<br>'+
  '<b>persistent value-per-compute advantage: '+d.vpc_advantage_pct+'%</b> (pass bar: >=15-20%, equal usefulness, across tasks)';
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
        if s.path=="/pair":
            s.send_response(200);s.send_header("Content-Type","application/json");s.end_headers()
            s.wfile.write(json.dumps(run_pair(d.get("task",""))).encode());return
        if s.path=="/rate":
            d["t"]=datetime.now(timezone.utc).isoformat(); log_rating(d)
            s.send_response(200);s.send_header("Content-Type","application/json");s.end_headers();s.wfile.write(b'{"ok":true}');return
        s.send_response(404);s.end_headers()

if __name__=="__main__":
    port=int(os.environ.get("PORT","8090"))
    t,_=engine("hi","test")
    print(f"Step 5 persistent-vs-fresh on port {port}. engine reachable: {t is not None}")
    print("run real tasks; both PERSISTENT and FRESH solve each; grade blind; measures value-per-compute.")
    HTTPServer(("0.0.0.0",port),H).serve_forever()
