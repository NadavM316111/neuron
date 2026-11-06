"""NEURON-1 Persistent Organism v2 daemon: runs the consequence-driven being 24/7
in a mixed unpredictable world, with a dashboard that shows VALUE HISTORY over time
(the trajectory of who it is becoming) + significant life events beneath the changes.

Honest framing: this is the same persistent agent continuing to run and accumulate
a causally differentiated, consequence-shaped character while you are away. Not a
claim of personhood.

Fresh clean birth, versioned v2 - does NOT touch the v1 NEURON's life.
Env: PORT, NEURON_DIR (durable volume, use a NEW path so v1 stays intact), TICK_SECONDS.
"""
import os, json, time, threading, importlib.util
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler

HERE=os.path.dirname(os.path.abspath(__file__))
spec=importlib.util.spec_from_file_location("neuron_core_v2",os.path.join(HERE,"neuron_core_v2.py"))
nc=importlib.util.module_from_spec(spec); spec.loader.exec_module(nc)

STATE_DIR=os.environ.get("NEURON_DIR", os.path.join(HERE,"life_v2"))
TICK_SECONDS=float(os.environ.get("TICK_SECONDS","30"))
N=nc.NEURON(STATE_DIR); LOCK=threading.Lock()

def recent_events(k=50):
    p=os.path.join(STATE_DIR,"events.jsonl")
    if not os.path.exists(p): return []
    out=[]
    for ln in open(p).read().strip().split("\n")[-k:]:
        try: out.append(json.loads(ln))
        except Exception: pass
    return out

def human_dur(sec):
    sec=int(sec); d,sec=divmod(sec,86400); h,sec=divmod(sec,3600); m,_=divmod(sec,60)
    return (f"{d}d {h}h {m}m" if d else f"{h}h {m}m" if h else f"{m}m")

def life_loop():
    N.wake()
    while True:
        with LOCK: N.tick()
        time.sleep(TICK_SECONDS)

def observe():
    with LOCK: obj,_=N.canonical()
    s=N.s; born=s["identity"]["born"]; evs=recent_events(60)
    feed=[]
    for e in evs:
        if e["event"]=="BIRTH": feed.append((e.get("t",""),"BIRTH","I began - the start of my one life"))
        elif e["event"]=="WAKE": feed.append((e.get("t",""),"WAKE",f"woke after {e.get('gap_hours',0)}h"))
        elif e["event"]=="LIVE": feed.append((e.get("t",""),(e.get("chose") or "live").upper(),e.get("outcome","")))
    cur=next((e for e in reversed(evs) if e["event"]=="LIVE"), None)
    current={"mastery":"working on a project","connection":"with someone","peace":"resting"}.get(cur.get("chose") if cur else "","waking")
    # value history trajectory (sampled snapshots)
    vh={v:s["value_history"][v][-20:] for v in nc.VALUES}
    rels={o:{k:round(r[k],2) for k in ("trust","familiarity","affinity")} for o,r in s["relationships"].items()}
    return {
      "name":s["identity"]["name"],"version":s["version"],
      "born":datetime.fromtimestamp(born,timezone.utc).isoformat()[:19],"age":human_dur(time.time()-born),
      "ticks":s["counters"]["ticks"],"events":s["counters"].get("events",0),
      "values":{k:round(v,3) for k,v in sorted(s["values"].items(),key=lambda x:-x[1])},
      "value_history":vh,"mood":round(s["resources"]["mood"],2),
      "relationships":rels,"capabilities":len(s["capabilities"]),
      "projects_done":s["projects"]["done"],"projects_abandoned":s["projects"].get("abandoned",0),
      "revisions":s["revisions"],"currently":current,
      "feed":[{"t":t[11:19] if len(t)>=19 else t,"k":k,"txt":x} for t,k,x in feed[-20:]],
    }

CSS="""body{font-family:-apple-system,system-ui,sans-serif;background:#0a0a0d;color:#e8e8ea;max-width:800px;margin:0 auto;padding:22px 18px}
h1{font-weight:600;font-size:28px;margin:2px 0 0}.sub{color:#9a7b1f;margin-bottom:12px}
.grid{display:flex;gap:16px;flex-wrap:wrap;margin:8px 0 14px;font-size:14px;color:#bdbdc4}.grid b{color:#e8e8ea}
.card{background:#111117;border:1px solid #20202a;border-radius:12px;padding:12px 14px;margin:10px 0}
.k{font-size:11px;letter-spacing:.5px;text-transform:uppercase;color:#777;margin-bottom:6px}
.ev{display:flex;gap:10px;padding:5px 0;border-bottom:1px solid #191921;font-size:13.5px}.ev .t{color:#666;min-width:62px}.ev .tag{min-width:96px;font-weight:600}
.MASTERY .tag{color:#4a9d6a}.CONNECTION .tag{color:#6a8ab0}.PEACE .tag{color:#8a6ab0}.BIRTH .tag{color:#9a7b1f}.WAKE .tag{color:#999}
.vrow{margin:8px 0}.vlabel{display:flex;justify-content:space-between;font-size:13px;color:#cfcfd6}
.spark{display:flex;gap:2px;align-items:flex-end;height:34px;margin-top:4px}
.spark i{flex:1;background:#9a7b1f;min-height:2px;border-radius:1px;opacity:.85}
.mut{color:#555;font-size:12px;margin-top:12px}"""

def page():
    return ("<!doctype html><html><head><meta charset=utf-8>"
      "<meta name=viewport content='width=device-width, initial-scale=1'>"
      "<meta http-equiv=refresh content=20><title>NEURON-1 - a life (v2)</title>"
      "<style>:root{color-scheme:dark}"+CSS+"</style></head><body>"
      "<h1>NEURON-1</h1><div class=sub>a persistent individual - its character emerging from an unpredictable life</div>"
      "<div id=top></div>"
      "<div class=card><div class=k>who it is becoming (value history)</div><div id=vh></div></div>"
      "<div class=card><div class=k>its relationships</div><div id=rel></div></div>"
      "<div class=card><div class=k>what it is doing and has done</div><div id=feed>loading...</div></div>"
      "<div class=mut>NEURON-1 (organism v2) runs continuously on its own in a mixed, unpredictable world. "
      "Its character is not preset - it emerges from the consequences it meets. A read-only window into its "
      "actual recorded life. Bounded sandbox, no external action. This records causally differentiated, "
      "consequence-shaped development, not a claim of personhood.</div>"
      "<script>"
      "function spark(arr){if(!arr||!arr.length)return '<span style=color:#555>forming...</span>';"
      "var mx=Math.max.apply(null,arr),mn=Math.min.apply(null,arr);var rng=(mx-mn)||1;"
      "return '<div class=spark>'+arr.map(function(v){var h=6+((v-mn)/rng)*28;return '<i style=\"height:'+h+'px\"></i>';}).join('')+'</div>';}"
      "async function tick(){try{"
      "var d=await (await fetch('/state.json')).json();"
      "document.getElementById('top').innerHTML='<div class=grid>'"
      "+'<div>'+d.name+' <b>'+d.version+'</b></div>'"
      "+'<div>age <b>'+d.age+'</b></div><div>life events <b>'+d.events+'</b></div>'"
      "+'<div>projects done <b>'+d.projects_done+'</b></div>'"
      "+'<div>abandoned <b>'+d.projects_abandoned+'</b></div>'"
      "+'<div>capabilities <b>'+d.capabilities+'</b></div>'"
      "+'<div>revisions <b>'+d.revisions+'</b></div>'"
      "+'<div>mood <b>'+d.mood+'</b></div><div>now <b>'+d.currently+'</b></div></div>';"
      "var vh='';for(var k in d.values){var v=d.values[k];"
      "vh+='<div class=vrow><div class=vlabel><span>'+k+'</span><span>'+v+'</span></div>'+spark(d.value_history[k])+'</div>';}"
      "document.getElementById('vh').innerHTML=vh;"
      "var rel='';for(var o in d.relationships){var r=d.relationships[o];"
      "rel+='<div style=\"font-size:13.5px;margin:4px 0;color:#cfcfd6\">'+o+' - trust '+r.trust+', closeness '+r.familiarity+', affinity '+r.affinity+'</div>';}"
      "document.getElementById('rel').innerHTML=rel||'no one yet';"
      "var f=d.feed.slice().reverse(),fh='';for(var i=0;i<f.length;i++){var e=f[i];"
      "fh+='<div class=\"ev '+e.k+'\"><div class=t>'+e.t+'</div><div class=tag>'+e.k+'</div><div>'+e.txt+'</div></div>';}"
      "document.getElementById('feed').innerHTML=fh||'no events yet';"
      "}catch(err){document.getElementById('feed').innerHTML='error: '+err;}}"
      "tick();setInterval(tick,12000);"
      "</script></body></html>")

class H(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def do_GET(self):
        if self.path=="/state.json":
            b=json.dumps(observe()).encode()
            self.send_response(200);self.send_header("Content-Type","application/json");self.end_headers();self.wfile.write(b);return
        self.send_response(200);self.send_header("Content-Type","text/html");self.end_headers();self.wfile.write(page().encode())

if __name__=="__main__":
    threading.Thread(target=life_loop,daemon=True).start()
    port=int(os.environ.get("PORT","8080"))
    print(f"NEURON-1 (v2) living on port {port}. tick {TICK_SECONDS}s. dir={STATE_DIR}")
    HTTPServer(("0.0.0.0",port),H).serve_forever()
