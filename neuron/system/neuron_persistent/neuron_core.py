"""NEURON core: a persistent being whose ENTIRE state is reconstructable from an
append-only event log. The event log is the single source of truth; snapshots are
only an optimization. This makes the life auditable and crash-safe.

Design (the friend's spec):
  - every state-changing fact is an EVENT appended atomically to events.jsonl
  - replaying all events from BIRTH reproduces the EXACT current state
  - snapshots are checkpoints to avoid replaying from the start, but the state is
    always == replay(events). a corrupt/half-written tail event is ignored.
  - identity written once at birth; past is immutable.

State fields fully reconstructable: identity, values(+revisions), relationships,
capabilities, projects(open+done), resources/mood, world_model, counters.
"""
import os, json, time, random, hashlib
from datetime import datetime, timezone

def iso(t): return datetime.fromtimestamp(t,timezone.utc).isoformat()
def ordinal(n): return f"{n}{'th' if 10<=n%100<=20 else {1:'st',2:'nd',3:'rd'}.get(n%10,'th')}"

# cognition substrate
LIST_OPS={"evens":lambda L:[x for x in L if x%2==0],"odds":lambda L:[x for x in L if x%2],
 "gt10":lambda L:[x for x in L if x>10],"sorted_desc":lambda L:sorted(L,reverse=True),
 "drop_max":lambda L:[x for x in L if x!=max(L)] if L else L,"id":lambda L:L}
RED={"sum":sum,"max":lambda L:max(L) if L else 0,"min":lambda L:min(L) if L else 0,"len":len}
BIN={"sub":lambda a,b:a-b,"add":lambda a,b:a+b}
CUE={"evens":"even","odds":"odd","gt10":"above","sorted_desc":"largest","drop_max":"without largest","max":"largest","min":"smallest","sum":"sum","len":"count","sub":"difference","add":"sum"}
def cues_of(q):
    cs=sorted(set(CUE.values()),key=len,reverse=True); ql=q.lower(); return [c for c in cs if c in ql]
def run(p,L):
    if p[0]=="s": return RED[p[2]](LIST_OPS[p[1]](L))
    _,a,b,c,d,o=p; return BIN[o](RED[b](LIST_OPS[a](L)),RED[d](LIST_OPS[c](L)))
def pieces(p): return [p[1],p[2]] if p[0]=="s" else [p[1],p[2],p[3],p[4],p[5]]
DEEP=[("s",lo,rd) for lo in LIST_OPS for rd in RED]+[("c",a,b,c,d,o) for a in LIST_OPS for b in RED for c in LIST_OPS for d in RED for o in BIN]
def gen_subgoal(rng,hard):
    if not hard: p=("s",rng.choice(list(LIST_OPS)),rng.choice(list(RED)))
    else: p=("c",rng.choice(list(LIST_OPS)),rng.choice(list(RED)),rng.choice(list(LIST_OPS)),rng.choice(list(RED)),rng.choice(list(BIN)))
    d=(CUE.get(p[2],p[2])+" of "+CUE.get(p[1],p[1])) if p[0]=="s" else (CUE.get(p[5],p[5])+" of "+CUE.get(p[2],p[2])+" "+CUE.get(p[1],p[1])+" and "+CUE.get(p[4],p[4])+" "+CUE.get(p[3],p[3]))
    return d,p

class NEURON:
    def __init__(self, state_dir):
        self.dir=state_dir; os.makedirs(self.dir,exist_ok=True)
        self.log_path=os.path.join(self.dir,"events.jsonl")
        self._init_state()
        self._load()               # rebuild state by replaying the event log
        self.rng=random.Random(int(self.s["identity"]["born"]*1000)%2**31 + self.s["counters"]["ticks"])

    def _init_state(self):
        self.s={"identity":None,"values":{"mastery":0.5,"connection":0.5,"peace":0.5},
                "revisions":0,"relationships":{"being-A":0.3,"being-B":0.3},
                "capabilities":{},"projects":{"open":[],"done":0},
                "resources":{"skill":0.3,"bonds":0.3,"calm":0.5,"mood":0.5},
                "counters":{"ticks":0,"events":0,"last_awake":None}}

    # ---- the ONLY way state changes: apply an event (pure function of state+event) ----
    def _apply(self, ev):
        k=ev["event"]; s=self.s
        if k=="BIRTH":
            s["identity"]={"name":ev["name"],"born":ev["born"],"core":ev["core"]}
        elif k=="WAKE":
            s["counters"]["last_awake"]=ev["at"]
        elif k=="LIVE":
            s["counters"]["ticks"]+=1
            for key,val in ev["set_resources"].items(): s["resources"][key]=val
            s["values"]=dict(ev["values"]); s["revisions"]=ev["revisions"]
            if ev.get("bond"): s["relationships"][ev["bond"][0]]=ev["bond"][1]
            if ev.get("new_caps"):
                for d,prog in ev["new_caps"]: s["capabilities"][d]=prog
            s["projects"]=ev["projects"]
        s["counters"]["events"]=s["counters"].get("events",0)+1

    def _append(self, ev):
        # atomic-ish append: write + flush + fsync so a crash can't leave a torn line
        line=json.dumps(ev)
        with open(self.log_path,"a") as fh:
            fh.write(line+"\n"); fh.flush(); os.fsync(fh.fileno())
        self._apply(ev)

    def _load(self):
        if not os.path.exists(self.log_path):
            # BIRTH
            r=random.Random(int(time.time()*1000)%2**31)
            self._append({"event":"BIRTH","name":"NEURON","born":time.time(),
                          "core":[round(r.uniform(-0.3,0.3),3) for _ in range(3)],
                          "t":iso(time.time())})
            return
        # REPLAY the log; ignore a torn/half-written final line (crash-mid-write)
        good=0
        with open(self.log_path) as fh:
            lines=fh.readlines()
        valid=[]
        for ln in lines:
            ln=ln.strip()
            if not ln: continue
            try: ev=json.loads(ln)
            except Exception: continue   # torn tail line -> ignore (crash safety)
            valid.append(ev)
        # if the very last raw line had no trailing newline or was partial, json.loads
        # above already rejected it. rewrite the log clean (drop the torn tail).
        if len(valid)!=len([l for l in lines if l.strip()]):
            with open(self.log_path,"w") as fh:
                for ev in valid: fh.write(json.dumps(ev)+"\n")
        for ev in valid: self._apply(ev)

    def age_days(self): return (time.time()-self.s["identity"]["born"])/86400

    def wake(self):
        last=self.s["counters"]["last_awake"]; gap=time.time()-last if last else 0
        self._append({"event":"WAKE","at":time.time(),"gap_hours":round(gap/3600,3),"t":iso(time.time())})
        return gap

    def _solve(self,d,tp):
        if d in self.s["capabilities"]: return None
        exa=[([self.rng.randint(1,30) for _ in range(6)]) for _ in range(3)]
        exa=[(L,run(tp,L)) for L in exa]
        for pr in DEEP:
            try:
                if all(run(pr,L)==a for L,a in exa): return "|".join(map(str,pr))
            except Exception: pass
        return None

    def tick(self):
        s=self.s; res=dict(s["resources"]); V=dict(s["values"]); rev=s["revisions"]
        for k in ("skill","bonds","calm"): res[k]=max(0,res[k]-0.03)
        gaps={"mastery":V["mastery"]-res["skill"],"connection":V["connection"]-res["bonds"],"peace":V["peace"]-res["calm"]}
        pull={k:max(0.01,V[k]*0.6+gaps[k]*0.8) for k in gaps}
        tot=sum(pull.values()); r=self.rng.random()*tot; acc=0; choice="peace"
        for k,w in pull.items():
            acc+=w
            if r<=acc: choice=k; break
        projects={"open":[dict(x) for x in s["projects"]["open"]],"done":s["projects"]["done"]}
        bond=None; new_caps=[]; outcome=""
        if choice=="mastery":
            if not projects["open"]:
                proj=[gen_subgoal(self.rng,self.rng.random()<0.5) for _ in range(3)]
                projects["open"]=[{"d":d,"p":list(p),"done":False} for d,p in proj]
            for sg in projects["open"]:
                if not sg["done"]:
                    cap=self._solve(sg["d"],tuple(sg["p"]))
                    if cap is not None or sg["d"] in s["capabilities"]:
                        sg["done"]=True
                        if cap is not None: new_caps.append((sg["d"],cap))
            if all(sg["done"] for sg in projects["open"]):
                projects["done"]+=1; projects["open"]=[]
                res["skill"]=min(1,res["skill"]+0.05); outcome=f"finished a project (my {ordinal(projects['done'])}); skill grew"
            else:
                res["skill"]=min(1,res["skill"]+0.02); outcome="worked my project; still unfinished"
        elif choice=="connection":
            o=self.rng.choice(list(s["relationships"])); nb=min(1,s["relationships"][o]+0.05)
            bond=(o,nb); res["bonds"]=min(1,res["bonds"]+0.05); outcome=f"time with {o}; bond {nb:.2f}"
        else:
            res["calm"]=min(1,res["calm"]+0.06); outcome="rested and consolidated"
        unmet=sum(max(0,g) for g in gaps.values()); res["mood"]=max(0,min(1,1-unmet/2))
        for k in V:
            if gaps[k]>0.4: V[k]=max(0.2,round(V[k]-0.004,4)); rev+=1
            elif gaps[k]<0: V[k]=min(1,round(V[k]+0.002,4))
        ev={"event":"LIVE","t":iso(time.time()),"age_days":round(self.age_days(),4),
            "chose":choice,"outcome":outcome,"set_resources":{k:round(v,4) for k,v in res.items()},
            "values":{k:round(v,4) for k,v in V.items()},"revisions":rev,
            "bond":list(bond) if bond else None,"new_caps":new_caps,"projects":projects}
        self._append(ev)
        return choice, outcome

    def canonical(self):
        """a canonical, hashable snapshot of the ENTIRE state for comparison."""
        s=self.s
        obj={"identity":s["identity"],"values":{k:round(v,4) for k,v in s["values"].items()},
             "revisions":s["revisions"],"relationships":{k:round(v,4) for k,v in s["relationships"].items()},
             "capabilities":dict(sorted(s["capabilities"].items())),
             "projects":s["projects"],"resources":{k:round(v,4) for k,v in s["resources"].items()},
             "ticks":s["counters"]["ticks"]}
        blob=json.dumps(obj,sort_keys=True)
        return obj, hashlib.sha256(blob.encode()).hexdigest()
