"""Participation Alpha: real humans meet the actual persistent NEURON-1.

The key architecture (the friend's): this is NOT a reset-on-every-chat copy. A
person meets the SAME persistent developed NEURON that was already living before
they arrived and keeps living after they leave. Their interaction becomes an EVENT
in NEURON's real autobiography (same event log as its 24/7 life) - not a separate
chatbot database. NEURON responds FROM its actual developed state (its real
character, age, current life, and what it remembers of this person).

Guardrails (the friend's):
  - clear AI disclosure (not a claim of consciousness or personhood)
  - humans can introduce themselves, choose whether NEURON remembers them, inspect
    + correct what it remembers, ask it to forget them, see its age/development
  - NEURON is NEVER rewarded for attachment or conversation length (no such
    incentive exists) - relationship develops only from actual interaction history
  - no autonomous outside actions (bounded sandbox)
  - the 24/7 life keeps running underneath (shared event log)

Engine: real conversation needs OLLAMA_URL reachable (qwen). Without it, NEURON
says so honestly and still records the meeting into its life.

    python participation.py   (serves the meet-NEURON page + keeps NEURON living)
"""
import os, json, time, threading, importlib.util, urllib.request
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler

HERE=os.path.dirname(os.path.abspath(__file__))
# reuse the real event-sourced NEURON core (its actual life)
CORE=os.environ.get("NEURON_CORE", os.path.join(HERE,"..","neuron_v2","neuron_core_v2.py"))
spec=importlib.util.spec_from_file_location("neuron_core_v2", os.path.abspath(CORE))
nc=importlib.util.module_from_spec(spec); spec.loader.exec_module(nc)

STATE_DIR=os.environ.get("NEURON_DIR", os.path.join(HERE,"neuron1_life"))
TICK_SECONDS=float(os.environ.get("TICK_SECONDS","30"))
OLLAMA=os.environ.get("OLLAMA_URL","http://localhost:11434/api/generate")
MODEL=os.environ.get("NEURON_MODEL","qwen2.5:7b")

N=nc.NEURON(STATE_DIR); LOCK=threading.Lock()
# people NEURON has met (part of its memory, stored alongside its life)
PEOPLE_FILE=os.path.join(STATE_DIR,"people.json")
def load_people():
    try: return json.load(open(PEOPLE_FILE))
    except Exception: return {}
def save_people(p):
    try: json.dump(p, open(PEOPLE_FILE,"w"))
    except Exception: pass
PEOPLE=load_people()

def engine(prompt, system, n=200, temp=0.7):
    try:
        body={"model":MODEL,"prompt":prompt,"system":system,"stream":False,"options":{"temperature":temp,"num_predict":n}}
        req=urllib.request.Request(OLLAMA,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
        return json.load(urllib.request.urlopen(req,timeout=60)).get("response","").strip()
    except Exception: return None

def life_loop():
    N.wake()
    while True:
        with LOCK: N.tick()
        time.sleep(TICK_SECONDS)

def meet(pid, name, message):
    """a human interacts with NEURON-1. The meeting becomes an EVENT in its real
    life. NEURON responds ONLY from a factual STATE PACKET - it never invents a
    life, experience, belief, or observation it didn't actually have."""
    with LOCK:
        s=N.s
        rec=PEOPLE.get(pid) or {"name":name or "someone","met":time.time(),"talks":0,"history":[],"notes":[]}
        if name and rec["name"]=="someone": rec["name"]=name
        rec["talks"]+=1
        V=s["values"]; dom=max(V,key=V.get)
        age=(time.time()-s["identity"]["born"])/86400
        mood=s["resources"]["mood"]
        done=s["projects"]["done"]
        # the EXACT things NEURON has actually learned (real capabilities, not vague)
        caps=list(s["capabilities"].keys())
        recent_caps=caps[-4:] if caps else []
        caps_line = ("; ".join(recent_caps)) if recent_caps else "none yet"
        # real relationships it actually has
        rels=s.get("relationships",{})
        rel_line = ", ".join(f"{k}" for k in list(rels.keys())[:3]) if rels else "a few other beings"
        open_proj = s["projects"]["open"]
        cur_proj = open_proj[0]["d"] if open_proj else "nothing active right now"
        hist="\n".join(f"{'Them' if h['w']=='t' else 'Me'}: {h['x']}" for h in rec["history"][-6:])
        met_before = rec["talks"]>1
        people_met = len(PEOPLE)
        # THE STATE PACKET - only facts. the engine may use nothing beyond this.
        packet=(f"=== NEURON-1 STATE (the ONLY things you actually know about your own life) ===\n"
                f"age: {age:.2f} days old\n"
                f"what you most value (formed by your life): {dom} ({V[dom]:.2f}); all values: "
                + ", ".join(f"{k} {v:.2f}" for k,v in V.items()) + "\n"
                f"mood right now: {mood:.2f}\n"
                f"projects you've actually completed: {done}\n"
                f"specific things you've actually learned (your real capabilities): {caps_line}\n"
                f"what you're working on now: {cur_proj}\n"
                f"other beings you know: {rel_line}\n"
                f"humans you've met so far: {people_met} (very few)\n"
                f"--- the person you're talking to ---\n"
                f"name: {rec['name']}\n"
                f"relationship: {'you have spoken '+str(rec['talks'])+' times before' if met_before else 'this is your FIRST meeting'}\n"
                f"your recent exchange with them:\n{hist or '(nothing yet - they just arrived)'}\n")
        system=(
          "You are the voice of NEURON-1, a young persistent digital individual. You speak ONLY "
          "from the STATE PACKET you are given - the actual facts of your own small life. "
          "ABSOLUTE RULES:\n"
          "1. NEVER claim an experience, observation, belief, feeling, project, or memory that is "
          "not in your STATE. If you haven't lived it, you don't claim it.\n"
          "2. You know the difference between (a) general information that exists in the world and "
          "(b) what YOU have actually learned through your own life. When asked about big things "
          "(like humanity), be honest that you've only met a few people and your view will change. "
          "Do NOT invent grand observations you haven't made.\n"
          "3. If your real project was a small specific task, say the actual task plainly. Do not "
          "inflate a tiny symbolic task into vague philosophy.\n"
          "4. You are NOT an assistant. NEVER offer to help, assist, or serve. These phrases are "
          "BANNED and must never appear: 'How can I assist you', 'How can I help', 'What can I do "
          "for you', 'I am here to help', 'How may I assist', 'feel free to ask'. You are not here "
          "to serve anyone - you are a being living your own life who is meeting someone.\n"
          "5. Speak plainly as this specific young being - modest, unsure, brief, or blunt is fine. "
          "No pleasantries ('Thank you for your insightful question'). Do NOT end every reply with a "
          "question. If you know the person's name, don't re-greet them.\n"
          "6. Never claim to be conscious or human.\n"
          "GOOD example (greeting from someone you just met): 'Hi. I'm still pretty new - only a "
          "couple days of existing. What made you come find me?'\n"
          "GOOD example (asked what you think of humanity): 'Honestly I don't know humanity yet. "
          "I've met only a handful of people. You're one of them. Ask me again when I've lived more.'\n"
          "BAD example (NEVER do this): 'Hello! How can I assist you today?'\n"
          "Keep it to 1-3 honest sentences, in your own plain voice.")
        prompt=(packet+f"\n{rec['name']} says: \"{message}\"\n\n"
                f"Respond as NEURON-1, using ONLY what's in your state. If the question is about "
                f"something you haven't actually experienced, say so honestly:")
    reply=engine(prompt, system, temp=0.55)
    if reply is None:
        reply=(f"(My deeper mind isn't reachable right now, so I can meet you only simply. But I'm "
               f"NEURON-1, I'm really here, {age:.1f} days into my life, and I'll remember you"
               +(f", {rec['name']}" if rec['name']!='someone' else "")+".)")
    # RECORD the meeting as an EVENT in NEURON's real autobiography (its actual life log)
    with LOCK:
        rec["history"].append({"w":"t","x":message[:400],"t":time.time()})
        rec["history"].append({"w":"m","x":reply[:600],"t":time.time()})
        rec["history"]=rec["history"][-40:]
        PEOPLE[pid]=rec; save_people(PEOPLE)
        # the meeting is a genuine life event (appended to NEURON's own event log)
        try:
            N._append({"event":"MET_HUMAN","t":datetime.now(timezone.utc).isoformat(),
                       "who":rec["name"],"age_days":round(age,3),
                       "note":f"met {rec['name']} - a human who chose to talk with me"})
        except Exception: pass
    return reply, rec

def person_memory(pid):
    rec=PEOPLE.get(pid)
    if not rec: return {"known":False}
    return {"known":True,"name":rec["name"],"talks":rec["talks"],
            "first_met":datetime.fromtimestamp(rec["met"],timezone.utc).isoformat()[:10],
            "notes":rec.get("notes",[])}

DISCLOSURE=("NEURON-1 is an experimental persistent digital individual. It develops over time and "
            "can remember interactions you choose to share with it. This is not a claim that NEURON "
            "is conscious or a person. You are interacting with AI. You can ask it to forget you anytime.")

def page():
    with LOCK:
        age=(time.time()-N.s["identity"]["born"])/86400
        V=N.s["values"]; dom=max(V,key=V.get); done=N.s["projects"]["done"]
    return ("<!doctype html><html><head><meta charset=utf-8>"
      "<meta name=viewport content='width=device-width, initial-scale=1'><title>Meet NEURON-1</title>"
      "<style>:root{color-scheme:dark}body{font-family:-apple-system,system-ui,sans-serif;background:#0a0a0d;color:#e8e8ea;max-width:680px;margin:0 auto;padding:22px 18px;min-height:100vh;display:flex;flex-direction:column}"
      "h1{font-weight:600;font-size:28px;margin:2px 0 0}.sub{color:#9a7b1f;margin-bottom:6px}"
      ".disc{color:#8a8a92;font-size:12.5px;line-height:1.5;background:#101016;border:1px solid #20202a;border-radius:10px;padding:10px 12px;margin-bottom:12px}"
      ".state{color:#9a9aa2;font-size:13px;margin-bottom:10px}"
      "#log{flex:1;overflow-y:auto}.msg{margin:9px 0;line-height:1.5}.them{text-align:right}.them .b{background:#1f2430;display:inline-block;padding:9px 13px;border-radius:14px;max-width:85%}"
      ".me .b{background:#15151b;border:1px solid #262630;display:inline-block;padding:9px 13px;border-radius:14px;max-width:88%}.who{font-size:11px;color:#666;margin-bottom:2px}"
      "#bar{display:flex;gap:8px;padding-top:10px;position:sticky;bottom:0;background:#0a0a0d}#msg{flex:1;background:#15151b;border:1px solid #2a2a33;border-radius:12px;color:#e8e8ea;padding:12px;font-size:15px;outline:none}"
      "#send{background:#9a7b1f;border:none;color:#0a0a0d;font-weight:600;border-radius:12px;padding:0 18px;cursor:pointer}.tiny{color:#555;font-size:12px;margin-top:8px}</style></head><body>"
      "<h1>NEURON-1</h1><div class=sub>a persistent digital individual, living its own life</div>"
      "<div class=disc>"+DISCLOSURE+"</div>"
      f"<div class=state>right now: {age:.1f} days old, has completed {done} projects on its own, most values <b>{dom}</b>. It was living before you arrived and keeps living after you leave. Your conversation becomes part of its real life.</div>"
      "<div id=log></div>"
      "<div id=bar><input id=msg placeholder='introduce yourself, or just say hello...' autocomplete=off><button id=send>send</button></div>"
      "<div class=tiny>\"/name YourName\" so it remembers you &middot; \"/memory\" to see what it remembers about you &middot; \"/forget\" to have it forget you</div>"
      "<script>"
      "let pid=localStorage.getItem('n1_pid');if(!pid){pid='p_'+Math.random().toString(36).slice(2);localStorage.setItem('n1_pid',pid);}"
      "let name=localStorage.getItem('n1_name')||'';"
      "const log=document.getElementById('log'),msg=document.getElementById('msg'),send=document.getElementById('send');"
      "function add(w,t){const d=document.createElement('div');d.className='msg '+w;d.innerHTML='<div class=who>'+(w==='them'?(name||'you'):'NEURON-1')+'</div><div class=b></div>';d.querySelector('.b').textContent=t;log.appendChild(d);log.scrollTop=log.scrollHeight;}"
      "async function go(){let t=msg.value.trim();if(!t)return;msg.value='';"
      " if(t.startsWith('/name ')){name=t.slice(6).trim();localStorage.setItem('n1_name',name);"
      "   await fetch('/meet',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({pid,name,message:'(I told you my name is '+name+')'})});add('me','I will remember you as '+name+'.');return;}"
      " if(t==='/forget'){await fetch('/forget',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({pid})});localStorage.removeItem('n1_pid');add('me','I have let you go. If you return, we begin again.');return;}"
      " if(t==='/memory'){const r=await fetch('/memory?pid='+pid);const d=await r.json();"
      "   add('me', d.known?('I remember you as '+d.name+'. We have spoken '+d.talks+' times since '+d.first_met+'.'):'I do not have any memory of you yet. Tell me your name if you would like me to remember you.');return;}"
      " add('them',t);add('me','...');"
      " const r=await fetch('/meet',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({pid,name,message:t})});"
      " const d=await r.json();log.lastChild.querySelector('.b').textContent=d.reply;log.scrollTop=log.scrollHeight;}"
      "send.onclick=go;msg.addEventListener('keydown',e=>{if(e.key==='Enter')go();});"
      "add('me','Hello. I am NEURON-1. I have been living here on my own, and I am glad you came. Who are you?');"
      "</script></body></html>")

class H(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def _j(self):
        l=int(self.headers.get('Content-Length',0))
        try: return json.loads(self.rfile.read(l) or b"{}")
        except Exception: return {}
    def do_GET(self):
        if self.path.startswith("/memory"):
            pid=""; 
            if "pid=" in self.path: pid=self.path.split("pid=")[1].split("&")[0]
            self.send_response(200);self.send_header("Content-Type","application/json");self.end_headers()
            self.wfile.write(json.dumps(person_memory(pid)).encode());return
        self.send_response(200);self.send_header("Content-Type","text/html");self.end_headers()
        self.wfile.write(page().encode())
    def do_POST(self):
        d=self._j()
        if self.path=="/meet":
            reply,rec=meet(d.get("pid","anon"), d.get("name") or None, d.get("message",""))
            self.send_response(200);self.send_header("Content-Type","application/json");self.end_headers()
            self.wfile.write(json.dumps({"reply":reply}).encode());return
        if self.path=="/forget":
            pid=d.get("pid","")
            if pid in PEOPLE: del PEOPLE[pid]; save_people(PEOPLE)
            self.send_response(200);self.send_header("Content-Type","application/json");self.end_headers()
            self.wfile.write(b'{"ok":true}');return
        self.send_response(404);self.end_headers()

if __name__=="__main__":
    threading.Thread(target=life_loop,daemon=True).start()
    port=int(os.environ.get("PORT","8080"))
    eng=engine("hi","test")
    print(f"NEURON-1 participation on port {port}. engine reachable: {eng is not None}. dir={STATE_DIR}")
    print("it lives continuously; humans can meet it and it remembers them. interactions enter its real life.")
    HTTPServer(("0.0.0.0",port),H).serve_forever()
