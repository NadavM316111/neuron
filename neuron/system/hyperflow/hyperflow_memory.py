"""HyperFlow v1.1 - the memory/retrieval layer (the fix 9P diagnosed).

v1.0's work engine was state-blind: it never fed accumulated life into the work prompt.
This adds relevance-GATED retrieval: given a task, it pulls only the genuinely relevant
accumulated content and returns a bounded context block to prepend to the work prompt.

What it can retrieve:
  - a solution METHOD the neuron verified on a similar problem before (keyword-gated)
  - an ongoing PROJECT's goal + progress (triggered by continue/next-step phrasing)
  - a known COLLABORATOR's stated preference (triggered by the person being named)

It returns context (guidance/grounding), never the answer. Empty cupboard -> empty
context -> behaves exactly like v1.0. The content must be REAL (the neuron's own verified
work, real project progress, a preference a collaborator actually stated).
"""
import re, json, os

def _words(t): return set(re.findall(r"[a-z]{3,}", t.lower()))

class Memory:
    def __init__(s):
        s.methods = []      # [{kw:set, sig:str, solution:str}]
        s.projects = []     # [{name, goal, progress}]
        s.rels = {}         # person_lower -> {pref, note}

    # --- genuine accumulation (called during real work / warmup) ---
    def learn_method(s, signature, solution):
        s.methods.append({"kw": _words(signature), "sig": signature, "solution": solution.strip()[:600]})
    def add_project(s, name, goal, progress):
        for p in s.projects:
            if p["name"] == name: p["goal"] = goal; p["progress"] = progress; return
        s.projects.append({"name": name, "goal": goal, "progress": progress})
    def set_pref(s, person, pref, note=""):
        s.rels[person.lower()] = {"person": person, "pref": pref, "note": note}

    # --- relevance-gated retrieval into the work prompt ---
    def retrieve(s, task_text, category):
        tl = task_text.lower(); w = _words(task_text); ctx = []
        # 1) method: best keyword overlap, needs >=2 shared content words (gate against noise)
        best, ov = None, 0
        for m in s.methods:
            o = len(m["kw"] & w)
            if o > ov: ov, best = o, m
        if best and ov >= 2:
            ctx.append("An approach you used on a similar problem before (adapt it, do not copy numbers):\n" + best["solution"])
        # 2) project: triggered by continuation / "what are you building" phrasing
        if any(k in tl for k in ["continue", "project", "ongoing", "been working", "been building",
                                 "building", "built", "working on", "next step", "next feature",
                                 "your work", "tool like", "what you have been", "what you are"]):
            for p in s.projects[:2]:
                ctx.append(f"Your ongoing project '{p['name']}': goal = {p['goal']}. Progress so far = {p['progress']}.")
        # 3) relationship: triggered by a known collaborator being named
        for person, info in s.rels.items():
            if person in tl:
                ctx.append(f"Your collaborator {info['person']} has a stated preference: {info['pref']}."
                           + (f" Earlier note: {info['note']}." if info["note"] else ""))
        return "\n\n".join(ctx[:3])   # bounded: at most 3 blocks

    # --- persistence helpers ---
    def counts(s): return {"methods": len(s.methods), "projects": len(s.projects), "relationships": len(s.rels)}
    def save(s, path):
        with open(path, "w") as f: json.dump({"methods": [{"sig": m["sig"], "solution": m["solution"]} for m in s.methods],
                                              "projects": s.projects, "rels": s.rels}, f)
    def load(s, path):
        if not os.path.exists(path): return
        d = json.load(open(path))
        for m in d.get("methods", []): s.learn_method(m["sig"], m["solution"])
        s.projects = d.get("projects", []); s.rels = d.get("rels", {})
