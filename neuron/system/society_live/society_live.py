"""NEURON Society, Live: the integrated society living continuously, 24/7.

Runs forever. Each "tick" advances the society; periodically history events happen
(cultural contact, resource shocks, population splits) - unscripted, emergent.
It logs its evolving state to a file and serves a live status page so you can
check in on the society anytime. Pure Python, lightweight, cheap to host.

Deploy target: Railway (or any always-on host). Reads PORT from env.
"""
import random, math, json, os, time, threading
from http.server import HTTPServer, BaseHTTPRequestHandler

N_BEHAV=4; N_NICHE=4
_id=[0]
def nid(): _id[0]+=1; return _id[0]
LIFESPAN=1400
MISS=0.2; AMBIG=0.2; BREAK=0.15; MEM=0.1
RNG=random.Random()
def amr(vals,rng):
    m=max(vals); best=[i for i,v in enumerate(vals) if abs(v-m)<1e-9]; return rng.choice(best)

class Being:
    def __init__(self, group, inherited_niche=None, seed=0):
        self.id=nid(); self.group=group; self.depth=0
        r=random.Random(self.id*7+seed+RNG.randint(0,10**6)); self.rng=r
        if inherited_niche is not None:
            self.skill=[max(0,(0.4 if i==inherited_niche else 0)+r.uniform(0,0.2)) for i in range(N_NICHE)]
        else:
            self.skill=[r.uniform(0,0.2) for _ in range(N_NICHE)]
        self.behavior=[0.5]*N_BEHAV
        self.approval_mem=[0.0]*N_BEHAV; self.obs=0
        self.wellbeing=0.5; self.age=0; self.alive=True; self.reputation=0.5
        self.recent=[]; self.revisions=0
    def niche(self): return amr(self.skill,self.rng)
    def does(self): return amr(self.behavior,self.rng)
    def inferred_norm(self):
        if self.obs==0: return self.does()
        return amr(self.approval_mem,self.rng)
    def confidence(self):
        pos=[max(0,v) for v in self.approval_mem]; t=sum(pos); return max(pos)/t if t>0 else 0.0
    def react(self,beh): return beh==self.inferred_norm()
    def observe(self,beh,app):
        if self.rng.random()<MEM: beh=self.rng.randrange(N_BEHAV)
        self.obs+=1; self.approval_mem[beh]+=(1.0 if app else -1.0)

class Society:
    def __init__(self, groups=2, pop=12):
        self.pop=pop
        self.groups={g:[Being(g,seed=g*1000+i) for i in range(pop)] for g in range(groups)}
        for g in self.groups:
            for b in self.groups[g]: b.age=RNG.randint(0,LIFESPAN-200)
        self.tick=0; self.year=0; self.history=[]; self.niche_base=[0.7,0.7,0.7,0.7]
        self.maxdepth={g:0 for g in self.groups}; self.contacted=False

    def log_event(self, text):
        self.history.append({"year":self.year,"event":text})
        self.history=self.history[-50:]

    def step_group(self, members):
        A=[b for b in members if b.alive]
        if len(A)<3: return
        nd=[0]*N_NICHE
        for b in A: nd[b.niche()]+=1
        quality=[max(0.1,self.niche_base[i]-0.08*nd[i]) for i in range(N_NICHE)]
        for actor in A:
            actor.age+=1
            ni=actor.niche()
            eco=1 if quality[ni]*(0.3+actor.skill[ni])+actor.rng.uniform(-0.15,0.15)>0.5 else -1
            actor.skill[ni]=max(0,min(2,actor.skill[ni]+0.04*eco))
            actor.recent.append(eco); actor.recent=actor.recent[-30:]
            if len(actor.recent)>=30 and sum(1 for o in actor.recent if o>0)/len(actor.recent)<0.35:
                actor.skill[ni]*=0.5; actor.skill[actor.rng.randrange(N_NICHE)]+=0.4
                actor.revisions+=1; actor.recent=[]
            beh=actor.does()
            if actor.rng.random()<BREAK: beh=actor.rng.randrange(N_BEHAV)
            obs=[o for o in A if o is not actor]; actor.rng.shuffle(obs); panel=obs[:5]
            if not panel: continue
            approvals=sum(1 for o in panel if o.react(beh)); true_app=approvals>=len(panel)/2
            for o in panel:
                if o.rng.random()<MISS: continue
                app=(o.rng.random()<0.5) if o.rng.random()<AMBIG else true_app
                o.observe(beh,app)
            if true_app: actor.behavior[beh]=min(2,actor.behavior[beh]+0.05)
            else:
                actor.behavior[beh]=max(0,actor.behavior[beh]-0.04)
                actor.behavior[actor.inferred_norm()]+=0.05
            actor.wellbeing=max(0,min(1,0.96*actor.wellbeing+0.05*eco+0.04*(0.5 if true_app else -0.4)))
            if actor.age>LIFESPAN: actor.alive=False

    def reproduce(self, members, g):
        dead=[b for b in members if not b.alive]
        if dead:
            members[:]=[b for b in members if b.alive]
            while len(members)<self.pop and len(members)>=2:
                par=sorted(members,key=lambda b:b.wellbeing,reverse=True)[:2]
                c=Being(g,inherited_niche=par[0].niche(),seed=g*9000+nid())
                c.depth=max(p.depth for p in par)+1; self.maxdepth[g]=max(self.maxdepth[g],c.depth)
                members.append(c)
            while len(members)<2: members.append(Being(g,seed=g*9000+nid()))

    def convention(self, members):
        A=[b for b in members if b.alive]; bd=[0]*N_BEHAV
        for b in A: bd[b.does()]+=1
        if not sum(bd): return -1,0
        c=amr(bd,RNG); return c, bd[c]/sum(bd)

    def one_tick(self):
        self.tick+=1
        for g in list(self.groups.keys()):
            self.step_group(self.groups[g]); self.reproduce(self.groups[g], g)
        # a "year" every 350 ticks; history events happen at year boundaries
        if self.tick % 350 == 0:
            self.year+=1
            self.niche_base=[0.7+0.2*math.sin(self.year*0.6+i) for i in range(N_NICHE)]
            self.maybe_history_event()

    def maybe_history_event(self):
        roll=RNG.random()
        if not self.contacted and len(self.groups)>=2 and roll<0.15:
            # CONTACT: two groups' boundary opens - merge their interaction
            keys=list(self.groups.keys())[:2]
            merged=self.groups[keys[0]]+self.groups[keys[1]]
            self.groups[keys[0]]=merged; del self.groups[keys[1]]
            self.contacted=True
            self.log_event(f"two cultures made CONTACT - their boundary opened")
        elif roll<0.22:
            # RESOURCE SHOCK: a niche's base quality drops hard
            i=RNG.randrange(N_NICHE); self.niche_base[i]*=0.3
            self.log_event(f"a RESOURCE SHOCK hit niche #{i} - the world changed")
        elif roll<0.30 and any(len([b for b in m if b.alive])>=8 for m in self.groups.values()):
            # POPULATION SPLIT: a big group splits into two
            for g in list(self.groups.keys()):
                A=[b for b in self.groups[g] if b.alive]
                if len(A)>=8:
                    half=A[len(A)//2:]
                    newg=max(self.groups.keys())+1
                    for b in half: b.group=newg
                    self.groups[newg]=half; self.groups[g]=A[:len(A)//2]
                    self.log_event(f"a group SPLIT - a new group {newg} formed and departed")
                    break

    def snapshot(self):
        out={"tick":self.tick,"year":self.year,"groups":{},
             "recent_history":self.history[-8:]}
        for g in self.groups:
            A=[b for b in self.groups[g] if b.alive]
            if not A: continue
            conv,adh=self.convention(self.groups[g])
            out["groups"][str(g)]={"alive":len(A),"convention":conv,
                "adherence":round(adh,2),
                "avg_confidence":round(sum(b.confidence() for b in A)/len(A),2),
                "lineage_depth":self.maxdepth.get(g,0),
                "avg_age":round(sum(b.age for b in A)/len(A))}
        return out

SOCIETY=Society()
STATE={"snap":SOCIETY.snapshot()}

def run_forever():
    while True:
        SOCIETY.one_tick()
        if SOCIETY.tick % 50 == 0:
            STATE["snap"]=SOCIETY.snapshot()
            try: json.dump(STATE["snap"], open("society_state.json","w"), indent=2)
            except Exception: pass
        time.sleep(0.01)   # ~100 ticks/sec; the society lives in real time

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def do_GET(self):
        snap=STATE["snap"]
        if self.path=="/state.json":
            self.send_response(200); self.send_header("Content-Type","application/json"); self.end_headers()
            self.wfile.write(json.dumps(snap).encode()); return
        self.send_response(200); self.send_header("Content-Type","text/html"); self.end_headers()
        rows=""
        for g,d in snap["groups"].items():
            rows+=f"<tr><td>group {g}</td><td>{d['alive']}</td><td>#{d['convention']}</td><td>{d['adherence']}</td><td>{d['avg_confidence']}</td><td>{d['lineage_depth']}</td><td>{d['avg_age']}</td></tr>"
        hist="".join(f"<li>year {h['year']}: {h['event']}</li>" for h in snap["recent_history"])
        html=f"""<!doctype html><html><head><meta charset=utf-8><meta http-equiv=refresh content=5>
<title>NEURON Society (live)</title><style>
body{{font-family:-apple-system,system-ui,sans-serif;background:#0c0c0f;color:#e8e8ea;max-width:760px;margin:40px auto;padding:0 20px}}
h1{{font-weight:600}} .sub{{color:#9a7b1f}} table{{width:100%;border-collapse:collapse;margin:16px 0}}
td,th{{padding:8px 10px;border-bottom:1px solid #26262b;text-align:left;font-size:14px}}
th{{color:#888;font-weight:500}} li{{margin:4px 0;color:#c9c9cf;font-size:14px}} .big{{font-size:28px}}</style></head>
<body><h1>NEURON Society <span class=sub>live</span></h1>
<p class=big>year {snap['year']} &middot; tick {snap['tick']}</p>
<p style="color:#888">A society of developing individuals, living continuously. Ecology, culture, norms, lineage, and history, all running on their own.</p>
<table><tr><th>group</th><th>alive</th><th>convention</th><th>adherence</th><th>confidence</th><th>lineage depth</th><th>avg age</th></tr>{rows}</table>
<h3 style="color:#9a7b1f">recent history</h3><ul>{hist or '<li>(the society is young; history is still to come)</li>'}</ul>
<p style="color:#555;font-size:12px">auto-refreshes every 5s &middot; the society runs whether or not anyone is watching</p>
</body></html>"""
        self.wfile.write(html.encode())

if __name__=="__main__":
    threading.Thread(target=run_forever, daemon=True).start()
    port=int(os.environ.get("PORT","8080"))
    print(f"NEURON Society living on port {port}")
    HTTPServer(("0.0.0.0",port), Handler).serve_forever()
