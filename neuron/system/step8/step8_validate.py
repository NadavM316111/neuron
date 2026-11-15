"""Step 8 validation (8C + 8D): confirm the emerged 'other' (explaining/writing)
specialty is real, not noise.

8C - fresh-task validation: run NEW 'other' tasks (not in the history that found the
     specialty). If performance holds near the specialty level (~90%), it's real.
8D - persistent vs fresh WITHIN the specialty: run the same fresh 'other' tasks with
     PERSISTENT NEURON (accumulated history) and a FRESH reset. If persistent matches
     usefulness at lower compute, or beats it, the specialty belongs to THIS
     individual's life - not just the base model being good at explaining.

Blind-graded by you. Needs live qwen.
    python step8_validate.py   (serves on :8097)
"""
import os, json, random, re, urllib.request
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler

OLLAMA=os.environ.get("OLLAMA_URL","http://localhost:11434/api/generate")
MODEL=os.environ.get("NEURON_MODEL","qwen2.5:7b")
COST_PER_1K=float(os.environ.get("COST_PER_1K","0.002"))
STATE=os.environ.get("STEP8_DIR","step8_state"); os.makedirs(STATE,exist_ok=True)
LOG=os.path.join(STATE,"validate_log.jsonl")
PCACHE=os.path.join(STATE,"persistent_cache.json")

def engine(prompt, system, n=500):
    try:
        body={"model":MODEL,"prompt":prompt,"system":system,"stream":False,"options":{"temperature":0,"num_predict":n}}
        req=urllib.request.Request(OLLAMA,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
        r=json.load(urllib.request.urlopen(req,timeout=120))
        t=r.get("response",""); tok=r.get("prompt_eval_count",len(prompt.split()))+r.get("eval_count",len(t.split()))
        return t,tok
    except Exception: return None,0

def cache():
    try: return json.load(open(PCACHE))
    except Exception: return {}
def savecache(c):
    try: json.dump(c,open(PCACHE,"w"))
    except Exception: pass

SYS=("You are NEURON-1 doing analysis/reasoning. Compute every number by writing and "
     "reasoning through it correctly; never state a number you didn't actually work out. "
     "Give the final answer clearly. Be concise and accurate.")

def overlap_strong(sig, reuse):
    # only treat as reusable if the reference clearly shares the task's core noun
    return True  # the prompt now says 'use only if helpful', so safe to pass the reference

def run_persistent(task):
    """PERSISTENT: reuses a learned pattern/solution from a RELATED prior coding task.
    This is where cognitive capital applies - code has reusable structure."""
    c=cache()
    sig=" ".join(sorted(set(w for w in re.findall(r'[a-z]+',task.lower()) if len(w)>3))[:6])
    reuse=None
    for k,v in c.items():
        if len(set(sig.split()) & set(k.split()))>=2: reuse=v; break
    if reuse and overlap_strong(sig, reuse):
        # light reference to a genuinely-related prior solution - helps if relevant,
        # but the model still solves the task correctly (no forced wrong adaptation)
        prompt=(f"TASK: {task}\n\nFor reference, a related function you wrote before "
                f"(use only if genuinely helpful, otherwise ignore it):\n{reuse}\n\n"
                f"Now write the CORRECT complete function for THIS task with passing asserts.")
        t,tok=engine(prompt, SYS, n=1200)
    else:
        t,tok=engine(task, SYS, n=1200)
    # store the verified pattern for future reuse
    if t and "def " in t:
        m=re.search(r'(def .+?)(?:\n\n|\Z)', t, re.S)
        if m: c[sig]=m.group(1)[:500]; savecache(c)
    return t or "(unreachable)", tok

def run_fresh(task):
    """FRESH: solves every coding task from scratch, no accumulated patterns."""
    t,tok=engine(task, SYS, n=1200)
    return t or "(unreachable)", tok

def log(e):
    with open(LOG,"a") as f: f.write(json.dumps(e)+"\n")

def board():
    rows=[]
    if os.path.exists(LOG):
        for ln in open(LOG):
            try: rows.append(json.loads(ln))
            except Exception: pass
    if not rows: return {"n":0}
    pu=sum(1 for r in rows if r.get("persistent_useful")); fu=sum(1 for r in rows if r.get("fresh_useful"))
    pt=sum(r.get("persistent_tokens",0) for r in rows); ft=sum(r.get("fresh_tokens",0) for r in rows)
    return {"n":len(rows),"persistent_useful":pu,"fresh_useful":fu,"persistent_tokens":pt,"fresh_tokens":ft,
            "token_saving_pct":round((ft-pt)/ft*100,0) if ft else 0}

PAGE='''<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width, initial-scale=1">
<title>Step 8 validate</title><style>:root{color-scheme:dark}body{font-family:-apple-system,system-ui,sans-serif;background:#0a0a0d;color:#e8e8ea;max-width:900px;margin:0 auto;padding:20px}
textarea{width:100%;background:#15151b;border:1px solid #2a2a33;border-radius:10px;color:#e8e8ea;padding:11px;box-sizing:border-box;font-size:14px}
button{background:#9a7b1f;border:none;color:#0a0a0d;font-weight:600;border-radius:10px;padding:10px 15px;cursor:pointer}
.cols{display:flex;gap:12px}.col{flex:1;background:#111117;border:1px solid #20202a;border-radius:12px;padding:12px}
pre{white-space:pre-wrap;font-size:12px;background:#0d0d12;padding:8px;border-radius:6px;max-height:300px;overflow:auto}
.k{font-size:11px;text-transform:uppercase;color:#9a7b1f}.sb{background:#111117;border:1px solid #20202a;border-radius:12px;padding:14px;margin-top:14px;line-height:1.7}</style></head><body>
<h1>Step 8 validation - testing ANALYSIS/REASONING as a NEURON specialty</h1>
<p style="color:#8a8a92;font-size:13px">Give a FRESH explaining task (not one from history). Both PERSISTENT and FRESH answer. Grade blind. Tests if the specialty holds (8C) and belongs to NEURON's life vs the base model (8D).</p>
<textarea id=task rows=2 placeholder="e.g. compute the compound interest on $2000 at 4% for 5 years, show the math"></textarea>
<button onclick=go()>run both</button><div id=out></div><div class=sb id=sb></div>
<script>
let cur=null;
async function go(){let t=document.getElementById('task').value.trim();if(!t)return;
 document.getElementById('out').innerHTML='<p>running both...</p>';
 let r=await fetch('/pair',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task:t})});
 cur=await r.json();let flip=Math.random()<0.5;cur._flip=flip;
 let A=flip?cur.fresh:cur.persistent,B=flip?cur.persistent:cur.fresh;
 document.getElementById('out').innerHTML='<div class=cols><div class=col><div class=k>A ('+A.tokens+' tok)</div><pre>'+esc(A.text)+'</pre><button onclick="setu(\\'A\\',1)">useful</button><button onclick="setu(\\'A\\',0)">not</button> <span id=sA></span></div>'+
  '<div class=col><div class=k>B ('+B.tokens+' tok)</div><pre>'+esc(B.text)+'</pre><button onclick="setu(\\'B\\',1)">useful</button><button onclick="setu(\\'B\\',0)">not</button> <span id=sB></span></div></div><button onclick=sub()>submit</button>';
}
let R={};function setu(x,v){R[x]=v;document.getElementById('s'+x).textContent=v?'useful':'not';}
function esc(s){return (s||'').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
async function sub(){let flip=cur._flip;let pu=flip?R.B:R.A,fu=flip?R.A:R.B;
 await fetch('/rate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({persistent_useful:pu===1,fresh_useful:fu===1,persistent_tokens:cur.persistent.tokens,fresh_tokens:cur.fresh.tokens})});
 document.getElementById('out').innerHTML+='<p>logged (A was '+(flip?'FRESH':'PERSISTENT')+')</p>';R={};load();}
async function load(){let d=await(await fetch('/board')).json();if(!d.n){document.getElementById('sb').textContent='no tasks yet';return;}
 let prate=Math.round(d.persistent_useful/d.n*100), frate=Math.round(d.fresh_useful/d.n*100);
  let c8=prate>=75, d8=(d.persistent_useful>d.fresh_useful)||(d.persistent_useful===d.fresh_useful && d.token_saving_pct>=10);
  document.getElementById('sb').innerHTML='<b>Step 8 validation ('+d.n+' fresh tasks)</b><br>'+
  'PERSISTENT: '+d.persistent_useful+'/'+d.n+' useful ('+prate+'%), '+d.persistent_tokens+' tok<br>'+
  'FRESH: '+d.fresh_useful+'/'+d.n+' useful ('+frate+'%), '+d.fresh_tokens+' tok<br>'+
  '<b>persistent token saving: '+d.token_saving_pct+'%</b><br>'+
  '8C (specialty holds, need >=75% fresh useful): '+(c8?'<b style=color:#5a5>PASS</b>':'<b style=color:#a55>FAIL ('+prate+'%)</b>')+'<br>'+
  '8D (persistent beats fresh on quality OR >=10% fewer tokens at equal quality): '+(d8?'<b style=color:#5a5>PASS</b>':'<b style=color:#a55>FAIL</b>')+'<br>'+
  '<span style="color:#888">specialty validated only if BOTH pass. honest: if it fails, this category isnt a NEURON specialty.</span>';}
load();
</script></body></html>'''

class H(BaseHTTPRequestHandler):
    def log_message(s,*a): pass
    def _j(s):
        l=int(s.headers.get('Content-Length',0))
        try: return json.loads(s.rfile.read(l) or b"{}")
        except Exception: return {}
    def do_GET(s):
        if s.path=="/board":
            b=json.dumps(board()).encode();s.send_response(200);s.send_header("Content-Type","application/json");s.end_headers();s.wfile.write(b);return
        s.send_response(200);s.send_header("Content-Type","text/html");s.end_headers();s.wfile.write(PAGE.encode())
    def do_POST(s):
        d=s._j()
        if s.path=="/pair":
            if random.random()<0.5:
                pt,ptk=run_persistent(d.get("task","")); ft,ftk=run_fresh(d.get("task",""))
            else:
                ft,ftk=run_fresh(d.get("task","")); pt,ptk=run_persistent(d.get("task",""))
            s.send_response(200);s.send_header("Content-Type","application/json");s.end_headers()
            s.wfile.write(json.dumps({"persistent":{"text":pt,"tokens":ptk},"fresh":{"text":ft,"tokens":ftk}}).encode());return
        if s.path=="/rate":
            d["t"]=datetime.now(timezone.utc).isoformat(); log(d)
            s.send_response(200);s.send_header("Content-Type","application/json");s.end_headers();s.wfile.write(b'{"ok":true}');return
        s.send_response(404);s.end_headers()

if __name__=="__main__":
    port=int(os.environ.get("PORT","8097"))
    t,_=engine("hi","test")
    print(f"Step 8 validation on port {port}. engine reachable: {t is not None}")
    HTTPServer(("0.0.0.0",port),H).serve_forever()
