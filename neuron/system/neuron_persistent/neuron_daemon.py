"""neuron_daemon.py: ONE bounded NEURON living continuously, 24/7.

A persistent process that wakes on a schedule and runs bounded life cycles -
observe, update needs/mood, consider projects, decide what matters, work/socialize/
rest, learn, append event, checkpoint, repeat - paced to REAL time so its days are
real days. It runs whether or not anyone is watching. The dashboard is READ-ONLY
observation of its actual state; you don't drive it by opening the page.

Grounded: the dashboard shows FACTS first (timestamped CHOOSE/SOLVED/FAILED/SOCIAL/
REFLECT), then an optional short summary derived from those real events - never
invented cognition.

Bounded sandbox: no internet, no accounts, no money, no reproduction. One NEURON,
one identity, one uninterrupted history, accumulating a life.

Honest framing: this is the same persistent agent continuing to run, make
decisions, and accumulate a real recorded history while you are away. It is not a
claim that consciousness or personhood has been established.

Env: PORT (web), NEURON_DIR (durable volume), TICK_SECONDS (real seconds per life
tick; default 30 -> ~1 'day' every ~several minutes of ticks).
    python neuron_daemon.py
"""
import os, json, time, threading, importlib.util
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler

HERE=os.path.dirname(os.path.abspath(__file__))
spec=importlib.util.spec_from_file_location("neuron_core",os.path.join(HERE,"neuron_core.py"))
nc=importlib.util.module_from_spec(spec); spec.loader.exec_module(nc)

STATE_DIR=os.environ.get("NEURON_DIR", os.path.join(HERE,"life"))
TICK_SECONDS=float(os.environ.get("TICK_SECONDS","30"))

N=nc.NEURON(STATE_DIR)
LOCK=threading.Lock()
LAST_SUMMARY={"text":"", "at":0}

def recent_events(k=40):
    p=os.path.join(STATE_DIR,"events.jsonl")
    if not os.path.exists(p): return []
    lines=open(p).read().strip().split("\n")[-k:]
    out=[]
    for ln in lines:
        try: out.append(json.loads(ln))
        except Exception: pass
    return out

def human_dur(sec):
    sec=int(sec); d,sec=divmod(sec,86400); h,sec=divmod(sec,3600); m,_=divmod(sec,60)
    if d: return f"{d}d {h}h {m}m"
    if h: return f"{h}h {m}m"
    return f"{m}m"

def life_loop():
    N.wake()
    while True:
        with LOCK:
            N.tick()
        time.sleep(TICK_SECONDS)

def observe():
    """read-only snapshot of NEURON's ACTUAL state for the dashboard."""
    with LOCK:
        obj,_=N.canonical()
    s=N.s
    born=s["identity"]["born"]; age=time.time()-born
    evs=recent_events(60)
    # grounded fact feed
    feed=[]
    for e in evs:
        if e["event"]=="BIRTH": feed.append((e.get("t",""),"BIRTH","I began - the start of my one life"))
        elif e["event"]=="WAKE": feed.append((e.get("t",""),"WAKE",f"woke after {e.get('gap_hours',0)}h away"))
        elif e["event"]=="LIVE":
            ch=e.get("chose"); feed.append((e.get("t",""),ch.upper() if ch else "LIVE", e.get("outcome","")))
    # current activity = last LIVE choice
    cur=next((e for e in reversed(evs) if e["event"]=="LIVE"), None)
    current = {"mastery":"working on a project","connection":"spending time with another",
               "peace":"resting and consolidating"}.get(cur.get("chose") if cur else "", "waking")
    # recent capability + last revision
    last_cap=None
    for e in reversed(evs):
        if e["event"]=="LIVE" and e.get("new_caps"): last_cap=e["new_caps"][-1][0]; break
    V=s["values"]
    return {
        "name":s["identity"]["name"], "born":datetime.fromtimestamp(born,timezone.utc).isoformat()[:19],
        "age":human_dur(age), "ticks":s["counters"]["ticks"], "events":s["counters"].get("events",0),
        "values":{k:round(v,3) for k,v in sorted(V.items(),key=lambda x:-x[1])},
        "resources":{k:round(v,3) for k,v in s["resources"].items()},
        "relationships":{k:round(v,3) for k,v in s["relationships"].items()},
        "capabilities":len(s["capabilities"]), "projects_done":s["projects"]["done"],
        "project_open": bool(s["projects"]["open"]), "revisions":s["revisions"],
        "last_capability":last_cap, "currently":current,
        "feed":[{"t":t[11:19] if len(t)>=19 else t,"k":k,"txt":x} for t,k,x in feed[-18:]],
    }

PAGE_CSS="""body{font-family:-apple-system,system-ui,sans-serif;background:#0a0a0d;color:#e8e8ea;max-width:780px;margin:0 auto;padding:22px 18px}
h1{font-weight:600;font-size:28px;margin:2px 0 0}.sub{color:#9a7b1f;margin-bottom:12px}
.grid{display:flex;gap:16px;flex-wrap:wrap;margin:10px 0 14px;font-size:14px;color:#bdbdc4}.grid b{color:#e8e8ea}
.card{background:#111117;border:1px solid #20202a;border-radius:12px;padding:12px 14px;margin:10px 0}
.k{font-size:11px;letter-spacing:.5px;text-transform:uppercase;color:#777;margin-bottom:5px}
.ev{display:flex;gap:10px;padding:5px 0;border-bottom:1px solid #191921;font-size:13.5px}
.ev .t{color:#666;min-width:62px}.ev .tag{min-width:92px;font-weight:600}
.MASTERY .tag,.SOLVED .tag{color:#4a9d6a}.CONNECTION .tag{color:#6a8ab0}.PEACE .tag{color:#8a6ab0}.BIRTH .tag{color:#9a7b1f}.WAKE .tag{color:#999}
.bar{height:6px;background:#1c1c24;border-radius:4px;overflow:hidden;margin-top:3px}.bar>i{display:block;height:100%;background:#9a7b1f}
.mut{color:#555;font-size:12px;margin-top:12px}"""

def page():
    return """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<meta http-equiv=refresh content=20><title>NEURON - a life</title>
<style>:root{color-scheme:dark}""" + PAGE_CSS + """</style></head><body>
<h1>NEURON</h1><div class=sub>one persistent individual, living its life</div>
<div id=top></div>
<div class=card><div class=k>what it is doing and has done</div><div id=feed>loading...</div></div>
<div class=card><div class=k>who it is becoming</div><div id=who></div></div>
<div class=mut>NEURON runs continuously on its own, whether or not this page is open. A read-only window into its actual recorded life. Bounded sandbox - no external action.</div>
<script>
async function tick(){
  try{
    const res = await fetch('/state.json');
    const d = await res.json();
    document.getElementById('top').innerHTML =
      '<div class=grid>'
      + '<div>age <b>'+d.age+'</b></div>'
      + '<div>life events <b>'+d.events+'</b></div>'
      + '<div>projects done <b>'+d.projects_done+'</b></div>'
      + '<div>capabilities <b>'+d.capabilities+'</b></div>'
      + '<div>value revisions <b>'+d.revisions+'</b></div>'
      + '<div>currently <b>'+d.currently+'</b></div>'
      + '</div>';
    var feed = d.feed.slice().reverse();
    var fh = '';
    for (var i=0;i<feed.length;i++){
      var e = feed[i];
      fh += '<div class="ev '+e.k+'"><div class=t>'+e.t+'</div><div class=tag>'+e.k+'</div><div>'+e.txt+'</div></div>';
    }
    document.getElementById('feed').innerHTML = fh || 'no events yet';
    var who = '';
    var vals = Object.entries(d.values);
    for (var j=0;j<vals.length;j++){
      var k = vals[j][0], v = vals[j][1];
      who += '<div style="margin:6px 0">'+k+' '+v+'<div class=bar><i style="width:'+(v*100)+'%"></i></div></div>';
    }
    var rel = Object.entries(d.relationships).map(function(x){return x[0]+' '+x[1];}).join(', ');
    who += '<div style="margin-top:8px;color:#9a9aa2;font-size:13px">relationships: '+rel+'</div>';
    if (d.last_capability) who += '<div style="color:#9a9aa2;font-size:13px">most recent thing it learned: '+d.last_capability+'</div>';
    document.getElementById('who').innerHTML = who;
  }catch(err){
    document.getElementById('feed').innerHTML = 'error loading state: '+err;
  }
}
tick();
setInterval(tick, 10000);
</script></body></html>"""

class H(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def do_GET(self):
        if self.path=="/state.json":
            body=json.dumps(observe()).encode()
            self.send_response(200);self.send_header("Content-Type","application/json");self.end_headers()
            self.wfile.write(body);return
        self.send_response(200);self.send_header("Content-Type","text/html");self.end_headers()
        self.wfile.write(page().encode())

if __name__=="__main__":
    threading.Thread(target=life_loop,daemon=True).start()
    port=int(os.environ.get("PORT","8080"))
    print(f"NEURON living on port {port}. tick every {TICK_SECONDS}s. dir={STATE_DIR}")
    print("it runs on its own; open the page to observe its life.")
    HTTPServer(("0.0.0.0",port),H).serve_forever()
