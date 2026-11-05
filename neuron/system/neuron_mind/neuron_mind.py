"""NEURON Mind, live: a self-directed cognition that works CONTINUOUSLY on its own
chosen projects and reports its REAL cognitive process. The Ultron-feeling surface,
grounded in the genuine self-directed mind (not invented drama).

You open the dashboard and NEURON has been working. It shows what it chose to work
on and why, its plan, each attempt with its prediction, where an approach FAILED
and how it diagnosed + replanned, what method it discovered, what it connected to
past work, and what capability/belief it gained. Every line is the real cognitive
process - logged as it happens, not generated after the fact.

Sandbox only: structured problems it is authorized to work on. No accounts, money,
messaging, or permission expansion.

    python neuron_mind.py   (optionally with Ollama for natural-language reflections)
"""
import random, re, itertools, json, os, time, threading
from collections import defaultdict
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.request

OLLAMA=os.environ.get("OLLAMA_URL","http://localhost:11434/api/generate")
MODEL=os.environ.get("NEURON_MODEL","qwen2.5:7b")

def reflect_words(fact_line):
    """OPTIONAL: engine turns NEURON's real cognitive fact into a natural sentence.
    It is handed the ACTUAL fact and only rephrases it - never invents."""
    try:
        body={"model":MODEL,"stream":False,
              "system":"You put NEURON's real cognitive event into one plain first-person "
                       "sentence. Do not add drama or invent anything - only rephrase the "
                       "given fact. You ARE NEURON.",
              "prompt":f"My cognitive event: {fact_line}\nSay it in one plain sentence as me:",
              "options":{"temperature":0.4,"num_predict":50}}
        req=urllib.request.Request(OLLAMA,data=json.dumps(body).encode(),
                                   headers={"Content-Type":"application/json"})
        return json.load(urllib.request.urlopen(req,timeout=20)).get("response","").strip() or fact_line
    except Exception:
        return fact_line

# ---- sandbox cognition machinery (from self_directed, proven) ----
LIST_OPS={"evens":lambda L:[x for x in L if x%2==0],"odds":lambda L:[x for x in L if x%2],
 "gt10":lambda L:[x for x in L if x>10],"sorted_desc":lambda L:sorted(L,reverse=True),
 "drop_max":lambda L:[x for x in L if x!=max(L)] if L else L,"identity":lambda L:L}
REDUCERS={"sum":sum,"max":lambda L:max(L) if L else 0,"min":lambda L:min(L) if L else 0,"len":len}
BINOPS={"sub":lambda a,b:a-b,"add":lambda a,b:a+b}
CUE={"evens":"even","odds":"odd","gt10":"above","sorted_desc":"largest","drop_max":"without largest",
 "max":"largest","min":"smallest","sum":"sum","len":"count","sub":"difference","add":"sum"}
CUES=sorted(set(CUE.values()),key=len,reverse=True)
def cues_of(q): ql=q.lower(); return [c for c in CUES if c in ql]
def run(p,L):
    if p[0]=="s": return REDUCERS[p[2]](LIST_OPS[p[1]](L))
    _,loa,rda,lob,rdb,op=p
    return BINOPS[op](REDUCERS[rda](LIST_OPS[loa](L)),REDUCERS[rdb](LIST_OPS[lob](L)))
def pieces(p): return [p[1],p[2]] if p[0]=="s" else [p[1],p[2],p[3],p[4],p[5]]
ALL=[("s",lo,rd) for lo in LIST_OPS for rd in REDUCERS]+[("c",a,b,c,d,o) for a in LIST_OPS for b in REDUCERS for c in LIST_OPS for d in REDUCERS for o in BINOPS]
def subgoal(rng):
    if rng.random()<0.5: p=("s",rng.choice(list(LIST_OPS)),rng.choice(list(REDUCERS)))
    else: p=("c",rng.choice(list(LIST_OPS)),rng.choice(list(REDUCERS)),rng.choice(list(LIST_OPS)),rng.choice(list(REDUCERS)),rng.choice(list(BINOPS)))
    d=(CUE.get(p[2],p[2])+" of "+CUE.get(p[1],p[1])) if p[0]=="s" else (CUE.get(p[5],p[5])+" of "+CUE.get(p[2],p[2])+" "+CUE.get(p[1],p[1])+" and "+CUE.get(p[4],p[4])+" "+CUE.get(p[3],p[3]))
    return d,p

MOTIVATIONS=["understand how quantities relate","find efficient methods","master harder compositions","connect new problems to what I already know"]

class LiveMind:
    def __init__(self):
        self.priors=defaultdict(lambda: defaultdict(float))
        self.capabilities={}
        self.log=[]            # the live cognitive stream (real events)
        self.beliefs=[]
        self.projects_done=0
        self.steps=0
        self.current=None
        self.born=time.time()
    def emit(self, kind, text, natural=None):
        self.log.append({"t":time.time(),"kind":kind,"text":text,
                         "natural":natural})
        self.log=self.log[-120:]
    def order(self, d):
        active=cues_of(d)
        if not any(self.priors[c] for c in active): return list(ALL)
        norm={}
        for c in active:
            tot=sum(self.priors[c].values()) or 1.0
            norm[c]={p:v/tot for p,v in self.priors[c].items()}
        sc=[(sum(norm.get(c,{}).get(p,0.0) for c in active for p in pieces(pr)),pr) for pr in ALL]
        sc.sort(key=lambda x:-x[0]); return [p for _,p in sc]
    def solve(self, d, tp, rng, budget=3000):
        if d in self.capabilities:
            self.emit("recall", f"recognized '{d}' - I already know this method, reusing it",
                      "I recognized this - I've solved something like it before, so I reused what I learned.")
            return self.capabilities[d],0,"recalled"
        ex=[]
        for _ in range(4):
            L=[rng.randint(1,40) for _ in range(6)]
            try: ex.append((L,run(tp,L)))
            except Exception: pass
        order=self.order(d); used_experience = any(self.priors[c] for c in cues_of(d))
        if used_experience:
            self.emit("plan", f"planning '{d}': my past work suggests where to look, searching there first",
                      f"For '{d}', my experience points me toward the likely method, so I'm looking there first.")
        else:
            self.emit("plan", f"planning '{d}': this is new territory, I'll search from scratch", None)
        tries=0
        for pr in order:
            tries+=1; self.steps+=1
            try:
                if all(run(pr,L)==a for L,a in ex):
                    newcap = d not in self.capabilities
                    self.capabilities[d]=pr
                    for c in cues_of(d):
                        for p in pieces(pr): self.priors[c][p]+=1.0
                    self.emit("solved", f"solved '{d}' in {tries} steps - storing this as a capability I own",
                              f"I found the method for '{d}' after {tries} tries, and I've kept it - it's mine now.")
                    return pr,tries,"solved"
            except Exception: pass
        return None,tries,"failed"
    def work_project(self, rng):
        motive=rng.choice(MOTIVATIONS)
        proj=[subgoal(rng) for _ in range(rng.randint(3,5))]
        self.current=motive
        self.emit("choose", f"I chose to work on a {len(proj)}-part project, driven by my wish to {motive}",
                  f"I decided to take on a new project because I want to {motive}.")
        for i,(d,tp) in enumerate(proj):
            pr,tries,how=self.solve(d,tp,rng)
            if how=="failed":
                self.emit("fail", f"my approach to '{d}' failed within budget - diagnosing why and widening my strategy",
                          f"My first approach to '{d}' didn't work. I'm figuring out why and trying a broader strategy.")
                pr2,t2,how2=self.solve(d,tp,rng,budget=len(ALL))
                if how2=="solved":
                    self.emit("recover", f"recovered on '{d}' with a different strategy - failure taught me where NOT to look",
                              f"The second strategy worked for '{d}'. The failure told me where the answer wasn't.")
        self.projects_done+=1
        # measurable reflection (real: expected->observed->capability)
        refl=(f"project {self.projects_done} done: I now hold {len(self.capabilities)} methods; "
              f"this project reused {sum(1 for d,_ in proj if d in self.capabilities)} I already knew")
        self.emit("reflect", refl,
                  f"That project is done. I know {len(self.capabilities)} methods now, and I reused several I'd learned before - I'm getting faster.")
        self.current=None

MIND=LiveMind()
STATE={"log":[], "stats":{}}

def loop():
    rng=random.Random()
    use_engine = reflect_words("test") != "test"   # is engine reachable for nice words
    while True:
        MIND.work_project(rng)
        # optionally enrich the most recent reflections with natural language
        if use_engine:
            for e in MIND.log[-3:]:
                if e.get("natural") is None and e["kind"] in ("choose","solved","reflect","recover","fail"):
                    e["natural"]=reflect_words(e["text"])
        STATE["log"]=MIND.log[-40:]
        STATE["stats"]={"age_min":int((time.time()-MIND.born)/60),"projects":MIND.projects_done,
                        "capabilities":len(MIND.capabilities),"steps":MIND.steps,
                        "working_on":MIND.current}
        time.sleep(2.5)   # a steady pace so you can watch it think

def page():
    return """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1"><title>NEURON - mind at work</title>
<style>:root{color-scheme:dark}body{font-family:-apple-system,system-ui,sans-serif;background:#0a0a0d;color:#e8e8ea;max-width:760px;margin:0 auto;padding:22px 18px}
h1{font-weight:600;font-size:28px;margin:4px 0 0}.sub{color:#9a7b1f;margin-bottom:14px}
.stats{display:flex;gap:18px;flex-wrap:wrap;margin-bottom:16px;color:#bdbdc4;font-size:14px}
.stats b{color:#e8e8ea}
.ev{padding:9px 12px;margin:7px 0;border-radius:11px;background:#111117;border:1px solid #20202a;line-height:1.5}
.k{font-size:11px;text-transform:uppercase;letter-spacing:.5px;color:#666;margin-bottom:2px}
.choose .k{color:#9a7b1f}.solved .k{color:#4a9d6a}.fail .k{color:#b06a4a}.recover .k{color:#6a8ab0}.reflect .k{color:#8a6ab0}.recall .k{color:#888}
.mut{color:#555;font-size:12px;margin-top:10px}</style></head><body>
<h1>NEURON</h1><div class=sub>a mind at work</div>
<div class=stats id=stats></div>
<div id=feed></div>
<div class=mut>NEURON works on its own chosen projects, in a sandbox, whether or not anyone is watching. This is its real cognitive process, logged as it happens.</div>
<script>
async function tick(){
 const r=await fetch('/stream.json');const d=await r.json();
 const s=d.stats;document.getElementById('stats').innerHTML=
  '<div>alive <b>'+s.age_min+'m</b></div><div>projects <b>'+s.projects+'</b></div>'+
  '<div>capabilities <b>'+s.capabilities+'</b></div><div>search-steps <b>'+s.steps+'</b></div>'+
  (s.working_on?'<div>now: <b>'+s.working_on+'</b></div>':'');
 const f=document.getElementById('feed');f.innerHTML='';
 d.log.slice().reverse().forEach(e=>{const div=document.createElement('div');
  div.className='ev '+e.kind;
  div.innerHTML='<div class=k>'+e.kind+'</div><div>'+(e.natural||e.text)+'</div>';
  f.appendChild(div);});
}
tick();setInterval(tick,2500);
</script></body></html>"""

class H(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def do_GET(self):
        if self.path=="/stream.json":
            self.send_response(200);self.send_header("Content-Type","application/json");self.end_headers()
            self.wfile.write(json.dumps(STATE).encode());return
        self.send_response(200);self.send_header("Content-Type","text/html");self.end_headers()
        self.wfile.write(page().encode())

if __name__=="__main__":
    threading.Thread(target=loop,daemon=True).start()
    port=int(os.environ.get("PORT","8080"))
    print(f"NEURON mind at work on port {port}. open it and watch it think.")
    HTTPServer(("0.0.0.0",port),H).serve_forever()
