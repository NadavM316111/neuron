"""NEURON Persistent Organism v2: the consequence-driven character mechanism (no
choice-ratchet, no floor, plastic) in a persistent, event-sourced being that lives
a MIXED, UNPREDICTABLE life - so the character that emerges is genuinely open.

Honest claim: this produces causally differentiated, consequence-shaped character
profiles inside the architecture - behavioral/value development, not a claim of
personhood.

Versioned clean birth (ORGANISM_VERSION="v2") so its developmental laws are known
and its life isn't muddied by an earlier mechanism. Event-sourced (state ==
replay(events)), crash-safe, same integrity as v1.

Key difference from v1: value <- PREDICTION ERROR between expected payoff and the
ACTUAL consequence the unpredictable world delivers. Pursue something that keeps
disappointing and you value it LESS. Neglected values fade toward 0. No floor.
"""
import os, json, time, random, hashlib
from datetime import datetime, timezone

ORGANISM_VERSION="v2"
def iso(t): return datetime.fromtimestamp(t,timezone.utc).isoformat()
def ordinal(n): return f"{n}{'th' if 10<=n%100<=20 else {1:'st',2:'nd',3:'rd'}.get(n%10,'th')}"

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

VALUES=["mastery","connection","peace"]; STATES={"mastery":"skill","connection":"bonds","peace":"calm"}

class NEURON:
    def __init__(self, state_dir):
        self.dir=state_dir; os.makedirs(self.dir,exist_ok=True)
        self.log_path=os.path.join(self.dir,"events.jsonl")
        self._init(); self._load()
        self.rng=random.Random(int(self.s["identity"]["born"]*1000)%2**31 + self.s["counters"]["ticks"])

    def _init(self):
        self.s={"identity":None,"version":ORGANISM_VERSION,
                "values":{v:0.5 for v in VALUES},"revisions":0,
                "relationships":{},"capabilities":{},
                "projects":{"open":[],"done":0,"failed":0,"abandoned":0},
                "resources":{"skill":0.3,"bonds":0.3,"calm":0.5,"mood":0.5},
                "recent_err":{v:[] for v in VALUES},
                "value_history":{v:[] for v in VALUES},
                "counters":{"ticks":0,"events":0,"last_awake":None}}

    def _apply(self, ev):
        k=ev["event"]; s=self.s
        if k=="BIRTH":
            s["identity"]={"name":ev["name"],"born":ev["born"],"core":ev["core"]}
            s["version"]=ev.get("version",ORGANISM_VERSION)
            s["relationships"]=dict(ev.get("relationships",{}))
        elif k=="WAKE":
            s["counters"]["last_awake"]=ev["at"]
        elif k=="LIVE":
            s["counters"]["ticks"]+=1
            for key,val in ev["set_resources"].items(): s["resources"][key]=val
            s["values"]=dict(ev["values"]); s["revisions"]=ev["revisions"]
            s["recent_err"]=ev["recent_err"]
            if ev.get("bond"): s["relationships"][ev["bond"]["who"]]=ev["bond"]["state"]
            if ev.get("new_caps"):
                for d,prog in ev["new_caps"]: s["capabilities"][d]=prog
            s["projects"]=ev["projects"]
            if ev.get("snapshot_values"):
                for v in VALUES: s["value_history"][v].append(ev["snapshot_values"][v])
        s["counters"]["events"]=s["counters"].get("events",0)+1

    def _append(self, ev):
        with open(self.log_path,"a") as fh:
            fh.write(json.dumps(ev)+"\n"); fh.flush(); os.fsync(fh.fileno())
        self._apply(ev)

    def _load(self):
        if not os.path.exists(self.log_path):
            r=random.Random(int(time.time()*1000)%2**31)
            # start with two acquaintances, each a relationship with texture
            rels={o:{"trust":0.3,"familiarity":0.1,"affinity":0.3,"last_contact":0}
                  for o in ["being-A","being-B","being-C"]}
            self._append({"event":"BIRTH","name":"NEURON-1","born":time.time(),
                          "version":ORGANISM_VERSION,"relationships":rels,
                          "core":[round(r.uniform(-0.3,0.3),3) for _ in range(3)],"t":iso(time.time())})
            return
        valid=[]
        for ln in open(self.log_path):
            ln=ln.strip()
            if not ln: continue
            try: valid.append(json.loads(ln))
            except Exception: continue
        for ev in valid: self._apply(ev)

    def age_days(self): return (time.time()-self.s["identity"]["born"])/86400
    def wake(self):
        last=self.s["counters"]["last_awake"]; gap=time.time()-last if last else 0
        self._append({"event":"WAKE","at":time.time(),"gap_hours":round(gap/3600,3),"t":iso(time.time())})
        return gap

    def _solve(self,d,tp):
        if d in self.s["capabilities"]: return None, True
        exa=[([self.rng.randint(1,30) for _ in range(6)]) for _ in range(3)]
        exa=[(L,run(tp,L)) for L in exa]
        # a HARD project may not be solvable in this tick's effort (stochastic)
        budget=len(DEEP) if self.rng.random()<0.7 else int(len(DEEP)*0.15)
        for i,pr in enumerate(DEEP):
            if i>budget: break
            try:
                if all(run(pr,L)==a for L,a in exa): return "|".join(map(str,pr)), True
            except Exception: pass
        return None, False

    def _world_payoff(self, drive):
        """the MIXED UNPREDICTABLE world: each drive's real consequence drifts over
        time (seasons) + noise. Nobody set the answer; character emerges from it."""
        t=self.s["counters"]["ticks"]
        import math
        seasons={"mastery":0.55+0.25*math.sin(t/180+0),
                 "connection":0.55+0.25*math.sin(t/200+2),
                 "peace":0.55+0.2*math.sin(t/160+4)}
        return max(0,min(1, seasons[drive]+self.rng.uniform(-0.3,0.3)))

    def tick(self):
        s=self.s; res=dict(s["resources"]); V=dict(s["values"]); rev=s["revisions"]
        rerr={k:list(v) for k,v in s["recent_err"].items()}
        for k in ("skill","bonds","calm"): res[k]=max(0,res[k]-0.03)
        gaps={v:V[v]-res[STATES[v]] for v in VALUES}
        pull={v:max(0.01,V[v]*0.8+gaps[v]*0.5) for v in VALUES}
        tot=sum(pull.values()); r=self.rng.random()*tot; acc=0; choice=VALUES[-1]
        for v,w in pull.items():
            acc+=w
            if r<=acc: choice=v; break
        expected=V[choice]; actual=self._world_payoff(choice)
        bond=None; new_caps=[]; outcome=""; projects=json.loads(json.dumps(s["projects"]))
        if choice=="mastery":
            if not projects["open"]:
                hard=self.rng.random()<0.4
                proj=[gen_subgoal(self.rng,hard) for _ in range(3 if not hard else 4)]
                projects["open"]=[{"d":d,"p":list(p),"done":False,"tries":0} for d,p in proj]
                projects["open_age"]=0
            projects["open_age"]=projects.get("open_age",0)+1
            progressed=False
            for sg in projects["open"]:
                if not sg["done"]:
                    sg["tries"]+=1
                    cap,ok=self._solve(sg["d"],tuple(sg["p"]))
                    if ok:
                        sg["done"]=True; progressed=True
                        if cap: new_caps.append((sg["d"],cap))
                    break  # one subgoal per tick -> hard projects take many ticks
            if all(sg["done"] for sg in projects["open"]):
                projects["done"]+=1; n=projects["done"]; projects["open"]=[]; projects["open_age"]=0
                res["skill"]=min(1,res["skill"]+0.05); outcome=f"finished my {ordinal(n)} project after real effort; skill grew"
            elif projects["open_age"]>12 and actual<0.4:
                # a long, unrewarding project may be ABANDONED (a real life event)
                projects["abandoned"]+=1; projects["open"]=[]; projects["open_age"]=0
                res["mood"]=max(0,res["mood"]-0.05); outcome="abandoned a project that wasn't paying off - it stopped feeling worth it"
            elif progressed: outcome=f"made progress on a project ({sum(sg['done'] for sg in projects['open'])}/{len(projects['open'])} parts)"
            else: outcome="struggled on a project; no progress this time"
        elif choice=="connection":
            o=self.rng.choice(list(s["relationships"])); rel=dict(s["relationships"][o])
            # relationships with TEXTURE: familiarity grows, trust shifts with how the
            # interaction went (actual payoff), affinity drifts
            went_well=actual>0.5
            rel["familiarity"]=min(1,rel["familiarity"]+0.04)
            rel["trust"]=min(1,max(0,rel["trust"]+(0.05 if went_well else -0.04)))
            rel["affinity"]=min(1,max(0,rel["affinity"]+(0.03 if went_well else -0.02)))
            rel["last_contact"]=s["counters"]["ticks"]
            bond={"who":o,"state":rel}
            res["bonds"]=min(1,res["bonds"]+0.04*rel["affinity"]+0.02)
            outcome=(f"time with {o}; it went {'well - trust grew' if went_well else 'poorly - some trust lost'} "
                     f"(trust {rel['trust']:.2f}, closeness {rel['familiarity']:.2f})")
        else:
            res["calm"]=min(1,res["calm"]+0.06); outcome="rested and consolidated"
        # relationships DRIFT when neglected (decay familiarity/affinity for others)
        for o in s["relationships"]:
            if not bond or o!=bond["who"]:
                pass  # applied at apply-time would need logging; keep simple: drift via affinity in bond only
        unmet=sum(max(0,g) for g in gaps.values()); res["mood"]=max(0,min(1,1-unmet/2))
        # CONSEQUENCE-DRIVEN value revision (prediction error; no floor; plastic)
        err=actual-expected; rerr[choice].append(round(err,3)); rerr[choice]=rerr[choice][-20:]
        avg_err=sum(rerr[choice])/len(rerr[choice])
        before_choice=V[choice]
        V[choice]=min(1,max(0.0,V[choice]+0.02*err+0.02*avg_err))
        for v in VALUES:
            if v!=choice: V[v]=max(0.0,V[v]-0.002)
        if abs(V[choice]-before_choice)>0.005: rev+=1
        # periodic value-history snapshot (for the dashboard trajectory)
        snap = {v:round(V[v],3) for v in VALUES} if s["counters"]["ticks"]%10==0 else None
        ev={"event":"LIVE","t":iso(time.time()),"age_days":round(self.age_days(),4),
            "chose":choice,"outcome":outcome,"expected":round(expected,3),"actual":round(actual,3),
            "set_resources":{k:round(v,4) for k,v in res.items()},
            "values":{k:round(v,4) for k,v in V.items()},"revisions":rev,
            "recent_err":rerr,"bond":bond,"new_caps":new_caps,"projects":projects,
            "snapshot_values":snap}
        self._append(ev)
        return choice, outcome

    def canonical(self):
        s=self.s
        obj={"identity":s["identity"],"version":s["version"],
             "values":{k:round(v,4) for k,v in s["values"].items()},"revisions":s["revisions"],
             "relationships":s["relationships"],"capabilities":dict(sorted(s["capabilities"].items())),
             "projects":s["projects"],"resources":{k:round(v,4) for k,v in s["resources"].items()},
             "ticks":s["counters"]["ticks"]}
        return obj, hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()
