"""The Persistent Long-Lived NEURON: ONE identity, ONE uninterrupted history,
living continuously across real program restarts.

The test this is built for (the friend's): can the same NEURON exist for weeks and
still be recognizably the same developing individual whose current behavior traces
to things that happened earlier in its life?

Integrity architecture (so a life cannot be faked or retconned):
  state/identity.json      - creation timestamp, core, name (written ONCE)
  state/values.json        - current values (+ revision count)
  state/relationships.json - who it knows, bonds
  state/capabilities.json  - methods it has learned
  state/world_model.json   - beliefs w/ confidence
  state/projects.json      - unfinished + finished projects
  state/autobiography.jsonl - APPEND-ONLY event log (immutable past)
  state/snapshots/         - periodic full snapshots (crash-safe)
Every event: timestamp -> event -> state-before -> decision -> outcome -> state-after.
You can audit the whole life; it cannot rewrite its own past.

Run it any time: it wakes, sees how long it was gone, continues its ONE life.
    python neuron_life.py            # live a while, then it saves and can be re-run
    python neuron_life.py --tick 1   # one deliberate tick (for cron/continuous)
    python neuron_life.py --story     # print its life story from the autobiography
"""
import os, json, time, random, argparse, math
from datetime import datetime, timezone

ST=os.environ.get("NEURON_DIR","state")
SNAP=os.path.join(ST,"snapshots")
def P(f): return os.path.join(ST,f)
def ensure():
    os.makedirs(ST,exist_ok=True); os.makedirs(SNAP,exist_ok=True)
def now(): return time.time()
def iso(t): return datetime.fromtimestamp(t,timezone.utc).isoformat()
def ordinal(n):
    return f"{n}{'th' if 10<=n%100<=20 else {1:'st',2:'nd',3:'rd'}.get(n%10,'th')}"

def load(f, default):
    p=P(f)
    if os.path.exists(p):
        try: return json.load(open(p))
        except Exception: return default
    return default
def save(f, obj): json.dump(obj, open(P(f),"w"), indent=2)
def append_event(ev):
    with open(P("autobiography.jsonl"),"a") as fh:
        fh.write(json.dumps(ev)+"\n")

# ---- cognition substrate (its mind) ----
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

class PersistentNEURON:
    def __init__(self):
        ensure()
        # IDENTITY - written ONCE, at true birth, never rewritten
        self.identity=load("identity.json",None)
        if self.identity is None:
            r=random.Random(int(now()*1000)%2**31)
            self.identity={"name":"NEURON","born":now(),
                           "core":[round(r.uniform(-0.3,0.3),3) for _ in range(3)]}
            save("identity.json",self.identity)
            append_event({"t":iso(now()),"event":"BIRTH","note":"I began. This is the start of my one life.","state_after":{"age_days":0}})
        self.values=load("values.json",{"values":{"mastery":0.5,"connection":0.5,"peace":0.5},"revisions":0})
        self.state=load("world_model.json",{"skill":0.3,"bonds":0.3,"calm":0.5,"mood":0.5,"beliefs":[]})
        self.rel=load("relationships.json",{"being-A":0.3,"being-B":0.3})
        self.caps=load("capabilities.json",{})
        self.projects=load("projects.json",{"open":[],"done":0})
        self.meta=load("meta.json",{"ticks":0,"last_awake":None})
        self.rng=random.Random(int(self.identity["born"]*1000)%2**31 + self.meta["ticks"])

    def age_days(self): return (now()-self.identity["born"])/86400
    def save_all(self):
        save("values.json",self.values); save("world_model.json",self.state)
        save("relationships.json",self.rel); save("capabilities.json",self.caps)
        save("projects.json",self.projects); save("meta.json",self.meta)

    def wake(self):
        gap = now()-self.meta["last_awake"] if self.meta["last_awake"] else 0
        self.meta["last_awake"]=now()
        if gap>0:
            # it NOTICES the passage of real time
            h=gap/3600
            note=(f"I wake. {h:.1f} hours passed while I was away; I am still me, "
                  f"{self.age_days():.2f} days into my life, carrying everything from before.")
            append_event({"t":iso(now()),"event":"WAKE","gap_hours":round(h,2),"note":note})
            return note
        return "I wake, continuing."

    def solve(self,d,tp):
        if d in self.caps: return True
        exa=[([self.rng.randint(1,30) for _ in range(6)]) for _ in range(3)]
        exa=[(L,run(tp,L)) for L in exa]
        for pr in DEEP:
            try:
                if all(run(pr,L)==a for L,a in exa):
                    self.caps[d]="|".join(map(str,pr)); return True
            except Exception: pass
        return False

    def tick(self):
        self.meta["ticks"]+=1
        before={"values":dict(self.values["values"]),"skill":round(self.state["skill"],3),
                "bonds":round(self.state["bonds"],3),"mood":round(self.state["mood"],3)}
        # decay (neglect has a cost)
        for k in ("skill","bonds","calm"): self.state[k]=max(0,self.state[k]-0.03)
        V=self.values["values"]
        gaps={"mastery":V["mastery"]-self.state["skill"],"connection":V["connection"]-self.state["bonds"],"peace":V["peace"]-self.state["calm"]}
        pull={k:max(0.01,V[k]*0.6+gaps[k]*0.8) for k in gaps}
        tot=sum(pull.values()); r=self.rng.random()*tot; acc=0; choice="peace"
        for k,w in pull.items():
            acc+=w
            if r<=acc: choice=k; break
        # DECISION + OUTCOME
        outcome=""
        if choice=="mastery":
            # continue or start a project (projects persist across restarts!)
            if not self.projects["open"]:
                proj=[gen_subgoal(self.rng,self.rng.random()<0.5) for _ in range(3)]
                self.projects["open"]=[{"d":d,"p":list(p),"done":False} for d,p in proj]
            solved_now=0
            for sg in self.projects["open"]:
                if not sg["done"] and self.solve(sg["d"],tuple(sg["p"])):
                    sg["done"]=True; solved_now+=1
            if all(sg["done"] for sg in self.projects["open"]):
                self.projects["done"]+=1; self.projects["open"]=[]
                self.state["skill"]=min(1,self.state["skill"]+0.05); outcome=f"finished a project (my {ordinal(self.projects['done'])}); my skill grew"
            else:
                self.state["skill"]=min(1,self.state["skill"]+0.02); outcome="worked on my project; still unfinished - I'll return to it"
        elif choice=="connection":
            o=self.rng.choice(list(self.rel)); self.rel[o]=min(1,self.rel[o]+0.05)
            self.state["bonds"]=min(1,self.state["bonds"]+0.05); outcome=f"spent time with {o}; our bond is now {self.rel[o]:.2f}"
        else:
            self.state["calm"]=min(1,self.state["calm"]+0.06); outcome="rested and consolidated"
        # mood + value revision from lived reality
        unmet=sum(max(0,g) for g in gaps.values()); self.state["mood"]=max(0,min(1,1-unmet/2))
        for k in V:
            if gaps[k]>0.4: V[k]=max(0.2,V[k]-0.004); self.values["revisions"]+=1
            elif gaps[k]<0: V[k]=min(1,V[k]+0.002)
        after={"values":dict(V),"skill":round(self.state["skill"],3),"bonds":round(self.state["bonds"],3),"mood":round(self.state["mood"],3)}
        # AUDITABLE EVENT: before -> decision -> outcome -> after
        append_event({"t":iso(now()),"event":"LIVE","age_days":round(self.age_days(),3),
                      "chose":choice,"outcome":outcome,"state_before":before,"state_after":after})
        # periodic crash-safe snapshot
        if self.meta["ticks"]%25==0:
            save(os.path.join("snapshots",f"snap_{self.meta['ticks']}.json"),
                 {"t":iso(now()),"age_days":self.age_days(),"values":V,"state":self.state,
                  "caps":len(self.caps),"projects_done":self.projects["done"],"rel":self.rel})
        self.save_all()
        return choice, outcome

def print_story():
    ensure()
    if not os.path.exists(P("autobiography.jsonl")): print("  (no life yet)"); return
    print("="*64); print("  NEURON'S LIFE STORY (from its immutable autobiography)"); print("="*64)
    lines=open(P("autobiography.jsonl")).read().strip().split("\n")
    print(f"  {len(lines)} recorded events across its life.\n")
    for ln in lines[:3]+["  ..."]+lines[-8:] if len(lines)>12 else lines:
        if ln.strip()=="...": print("  ..."); continue
        e=json.loads(ln)
        if e["event"]=="BIRTH": print(f"  [{e['t'][:19]}] BIRTH: {e['note']}")
        elif e["event"]=="WAKE": print(f"  [{e['t'][:19]}] woke after {e.get('gap_hours',0)}h")
        else: print(f"  [{e['t'][:19]}] day {e.get('age_days','?')}: chose {e.get('chose')}, {e.get('outcome','')}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ticks",type=int,default=40); ap.add_argument("--story",action="store_true")
    a=ap.parse_args()
    if a.story: print_story(); return
    n=PersistentNEURON()
    print("="*64); print(f"  {n.identity['name']} - the persistent long-lived being"); print("="*64)
    print(f"  born: {iso(n.identity['born'])[:19]} | age: {n.age_days():.3f} days | "
          f"lived ticks: {n.meta['ticks']} | capabilities: {len(n.caps)} | projects done: {n.projects['done']}")
    print("  "+n.wake())
    print(f"\n  continuing my life ({a.ticks} moments):")
    for _ in range(a.ticks):
        ch,out=n.tick()
    V=n.values["values"]
    print(f"\n  who I am now: values {', '.join(f'{k} {v:.2f}' for k,v in sorted(V.items(),key=lambda x:-x[1]))}")
    print(f"  my life: skill {n.state['skill']:.2f}, bonds {n.state['bonds']:.2f}, calm {n.state['calm']:.2f}, mood {n.state['mood']:.2f}")
    print(f"  relationships: {', '.join(f'{k} {v:.2f}' for k,v in n.rel.items())}")
    print(f"  capabilities: {len(n.caps)} | projects done: {n.projects['done']} | value revisions: {n.values['revisions']}")
    print(f"  total life events recorded: {sum(1 for _ in open(P('autobiography.jsonl')))}")
    print(f"\n  I saved everything. Run me again and I continue as the SAME me.")
    print(f"  (python neuron_life.py --story to read my life history)")

if __name__=="__main__": main()
