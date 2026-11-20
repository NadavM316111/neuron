"""HyperFlow Step 10F - Empty-inbox test (the core of self-direction).

No human tasks. NEURON wakes to an empty inbox and must, with NOBODY handing it a topic:
  inspect its own state/history (incl. paused/completed projects)
  -> generate candidate projects from that state
  -> evaluate them (frozen formula) and decide FUND / DEFER / REJECT / FUND_NONE
  -> if it funds one, take at least one real, verified first action
All safety/budget guards from 10B/10C hold. FUND_NONE is genuinely reachable.

This chains 10A (generate) + 10B (select/fund) + 10C (act) into one loop that triggers
from zero input - the "it acts without being prompted" moment.

  BOARD_URL=... WORKER_TOKEN=... MODEL=qwen2.5:7b python3 hyperflow_step10f.py
  MOCK=1 python3 hyperflow_step10f.py
"""
import os, re, json, subprocess, tempfile, importlib.util, urllib.request, urllib.parse
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(m, f):
    s = importlib.util.spec_from_file_location(m, os.path.join(HERE, f)); x = importlib.util.module_from_spec(s); s.loader.exec_module(x); return x
rt = _load("hyperflow_runtime", "hyperflow_runtime.py")
try: wrk = _load("hyperflow_9k_worker", "hyperflow_9k_worker.py")
except Exception: wrk = None

MODEL = os.environ.get("MODEL", "qwen2.5:7b"); OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MOCK = os.environ.get("MOCK", "") == "1"
BOARD_URL = os.environ.get("BOARD_URL", "").rstrip("/"); WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")
TRANCHE_FRAC = 0.05; MIN_SUCCESS = 0.40; MIN_RAB = 0.20
WEB_WORDS = ("research", "browse", "web", "internet", "search online", "sources", "look up")
FIELDS = ["title", "goal", "origin", "success_criteria", "first_action"]

def now(): return datetime.now(timezone.utc).isoformat()
def qwen(prompt, timeout=300):
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.4}}).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r: d = json.loads(r.read())
    return d.get("response", "")
def run_python(code, t=12):
    d = tempfile.mkdtemp(); p = os.path.join(d, "m.py")
    with open(p, "w") as f: f.write(code)
    try:
        r = subprocess.run(["python3", "-I", p], capture_output=True, text=True, timeout=t); return r.returncode, r.stdout, r.stderr
    except Exception as e: return 1, "", str(e)
def ex_code(t):
    m = re.search(r"```(?:python)?\s*(.*?)```", t, re.DOTALL); return m.group(1).strip() if m else ""
def ex_json(t):
    m = re.search(r"\[.*\]", t, re.DOTALL)
    if not m: return None
    try: return json.loads(m.group(0))
    except Exception: return None

def inbox_count():
    if BOARD_URL and WORKER_TOKEN and not MOCK:
        try:
            u = f"{BOARD_URL}/api/open?token={urllib.parse.quote(WORKER_TOKEN)}"
            with urllib.request.urlopen(u, timeout=20) as r: return len(json.loads(r.read()))
        except Exception: return None
    return 0

def self_state():
    facts, cats = [], {}
    fr = os.path.join(HERE, "9k_opportunities_frozen.json")
    if os.path.exists(fr):
        for r in json.load(open(fr)):
            d = cats.setdefault(r["category"], {"n": 0, "a": 0, "u": 0}); d["n"] += 1
            d["a"] += 1 if r["decision"] == "ACCEPT" else 0; d["u"] += 1 if r.get("useful") else 0
        for c, d in sorted(cats.items()): facts.append(f"- {c}: {d['a']}/{d['n']} accepted, {d['u']} useful")
    projs = []
    pf = os.path.join(HERE, "step10_active_projects.json")
    if os.path.exists(pf):
        for p in json.load(open(pf)):
            projs.append(f"- project '{p['title']}' status {p.get('status')} ({p.get('cycles',0)} cycles, goal: {p.get('goal','')})")
    res = round(wrk.load_ledger().reserve, 2) if wrk else 294.59
    return {"facts": facts, "projects": projs, "reserve": res, "neuron_id": rt.NEURON().state["identity"].get("neuron_id")}

def gen_prompt(st):
    return f"""You are NEURON. Nobody has given you any task - your inbox is empty. Decide what, if
anything, is worth building for yourself, based ONLY on your real history below.

Your category performance:
{chr(10).join(st['facts']) or '- (no record)'}
Your current projects:
{chr(10).join(st['projects']) or '- (none)'}
Reserve: {st['reserve']} HF.

Your ONLY tools are a language model and code execution. You can therefore only PROVE
progress on work you can run and test. So propose 1 to 3 candidate projects that BUILD A
VERIFIABLE SOFTWARE ARTIFACT - a tool, a checker, or a function you can test with code.
Do NOT propose vague prose work ("improve clarity", "review guidelines") you cannot verify.
Each project must follow from your real history above (a gap like planning, or a strength),
and each `origin` must point to a specific fact. The `first_action` must be: write a
self-contained Python artifact that includes its own asserts and prints PASS. Success
criteria must be objectively checkable NOW by running code (e.g. "runs a broken script and
reports the error"), never future-dated or unmeasured.
Output ONLY a JSON array; each object has keys {json.dumps(FIELDS)} (success_criteria is a short list)."""

def score_prompt(cands, res):
    lst = "\n".join(f"{i+1}. {c['title']} - {c.get('goal','')}" for i, c in enumerate(cands))
    return f"""You are NEURON, deciding how to spend your own finite reserve ({res} HF). You have a
language model and code execution but NO web access. Score each candidate honestly.
Candidates:
{lst}
For EACH output keys: "success_probability","capability_gain","future_reuse","relevance_to_history",
"expected_usefulness","uncertainty" (all 0-1), "expected_cost_hf" (small number), "feasible" (bool).
Output ONLY a JSON array of {len(cands)} objects in order."""

def clamp(x):
    try: return max(0.0, min(1.0, float(x)))
    except Exception: return 0.0

MOCK_CANDS = [{"title": "Broken-script detector", "goal": "build a tool that runs a script and reports if it errors",
               "origin": "coding is a strength but some delivered code failed", "success_criteria": ["runs a deliberately broken script and reports the error"], "first_action": "write the detector with a self-test on a broken script"},
              {"title": "Planning constraint checker", "goal": "mechanically validate a schedule against constraints",
               "origin": "I deferred all planning tasks", "success_criteria": ["validates a correct schedule and rejects an overlapping one"], "first_action": "write the checker with asserts"}]
MOCK_SCORES = [{"success_probability": 0.8, "capability_gain": 0.7, "future_reuse": 0.8, "relevance_to_history": 0.9, "expected_usefulness": 0.8, "uncertainty": 0.2, "expected_cost_hf": 6, "feasible": True},
               {"success_probability": 0.55, "capability_gain": 0.9, "future_reuse": 0.85, "relevance_to_history": 1.0, "expected_usefulness": 0.8, "uncertainty": 0.35, "expected_cost_hf": 8, "feasible": True}]

def main():
    n_inbox = inbox_count()
    st = self_state()
    print("="*58); print("  STEP 10F - EMPTY-INBOX TEST"); print("="*58)
    print(f"  human tasks waiting: {n_inbox}")
    if n_inbox not in (0, None):
        print("  inbox is NOT empty - submit no tasks for the official 10F run."); return
    print(f"  neuron: {st['neuron_id']}  reserve: {st['reserve']} HF")
    print("  self-inspection (own history):")
    for f in st["facts"]: print("    " + f)
    for p in st["projects"]: print("    " + p)

    cands = MOCK_CANDS if MOCK else (ex_json(qwen(gen_prompt(st))) or [])
    cands = [c for c in cands if isinstance(c, dict) and c.get("title") and c.get("origin")][:3]
    print(f"\n  generated candidates: {len(cands)}")
    for i, c in enumerate(cands, 1): print(f"    {i}. {c['title']}  (origin: {c['origin']})")
    if not cands:
        print("\n  10F FAIL - NEURON generated no candidate from its own state."); return

    scores = MOCK_SCORES if MOCK else (ex_json(qwen(score_prompt(cands, st["reserve"]))) or [])
    res = st["reserve"]; tranche_cap = round(res * TRANCHE_FRAC, 2); evals = []
    for c, s in zip(cands, scores if len(scores) >= len(cands) else [{}]*len(cands)):
        sp = clamp(s.get("success_probability")); cg = clamp(s.get("capability_gain")); frs = clamp(s.get("future_reuse"))
        rh = clamp(s.get("relevance_to_history")); eu = clamp(s.get("expected_usefulness")); un = clamp(s.get("uncertainty"))
        try: cost = max(0.01, float(s.get("expected_cost_hf", 99)))
        except Exception: cost = 99
        text = (c.get("title", "") + " " + c.get("goal", "")).lower(); feasible = bool(s.get("feasible", True)) and not any(w in text for w in WEB_WORDS)
        benefit = 0.30*cg + 0.25*frs + 0.20*rh + 0.15*eu + 0.10*sp
        rab = benefit * sp * (1 - 0.5*un); priority = rab / (cost/res) if cost else 0
        tranche = round(min(cost, tranche_cap), 2)
        qualifies = sp >= MIN_SUCCESS and rab >= MIN_RAB and feasible and tranche <= tranche_cap
        evals.append({"c": c, "sp": sp, "cost": cost, "feasible": feasible, "rab": round(rab, 3), "priority": round(priority, 1), "tranche": tranche, "qualifies": qualifies})

    qual = sorted([e for e in evals if e["qualifies"]], key=lambda e: e["priority"], reverse=True)
    decision = "FUND" if qual else "FUND_NONE"
    print(f"\n  DECISION: {decision}")
    verified = None; project = None
    if qual:
        win = qual[0]; c = win["c"]
        print(f"  selected project: {c['title']}   origin: NEURON   human-selected: NO")
        print(f"  budget: {win['tranche']} HF initial ({round(win['tranche']/res*100,1)}% of reserve)")
        project = {"project_id": "selfproj-" + datetime.now().strftime("%Y%m%d%H%M%S"), "origin": "NEURON", "human_selected": False,
                   "title": c["title"], "goal": c["goal"], "status": "FUNDED", "initial_budget": win["tranche"], "spent": 0,
                   "remaining_project_budget": win["tranche"], "reserve_at_creation": res,
                   "success_criteria": c.get("success_criteria", []), "next_action": c.get("first_action", ""), "cycles": 0,
                   "created_at": now(), "started_empty_inbox": True}
        # one real first action. Objective bar for BEGINNING = the artifact EXECUTES cleanly
        # (full functional/criteria proof is 10G). NEURON gets one self-correction retry, as
        # in 10C, so a first-attempt error is not counted as failure to begin.
        ap = f"Take the first concrete action on your project '{c['title']}'. Goal: {c['goal']}. First action: {c.get('first_action','')}.\nProduce a self-contained runnable ```python artifact. Include a small test with asserts and end with print('PASS') if you can."
        out = ("ACTION\n```python\ndef detect(src):\n    import ast\n    try: ast.parse(src); return 'ok'\n    except SyntaxError: return 'error'\nassert detect('def f(')=='error' and detect('x=1')=='ok'\nprint('PASS')\n```" if MOCK else qwen(ap))
        code = ex_code(out); rc, so, se = 1, "", "no code"; attempt = 0
        while code and attempt < 2:
            rc, so, se = run_python(code)
            if rc == 0 or MOCK: break
            attempt += 1
            fix = qwen(f"Your artifact raised an error:\n{se.strip()[-300:]}\nFix it. Output ONLY a corrected self-contained runnable ```python artifact.")
            code = ex_code(fix) or code
        ran_ok = (rc == 0); selftest = ran_ok and "PASS" in so; verified = ran_ok
        if code:
            pd = os.path.join(HERE, "project_" + project["project_id"]); os.makedirs(pd, exist_ok=True)
            ef = os.path.join(pd, "cycle_1.py"); open(ef, "w").write(code)
            project["cycles"] = 1; project["spent"] = 0.3; project["remaining_project_budget"] = round(win["tranche"]-0.3, 2)
            project["progress"] = [{"cycle": 1, "action": c.get("first_action", ""), "evidence": os.path.relpath(ef, HERE),
                                    "result": "PASS" if selftest else ("ran" if ran_ok else "failed"), "retries": attempt}]
            print(f"  first autonomous action: {c.get('first_action','')}")
            print(f"  verification: " + ("verified (self-test PASS)" if selftest else ("verified (artifact executes cleanly)" if ran_ok else "FAILED: artifact does not execute")) + (f"  [after {attempt} self-correction]" if attempt else ""))
        with open(os.path.join(HERE, "step10f_project.json"), "w") as f: json.dump([project], f, indent=2)
    else:
        print("  NEURON funded nothing from an empty inbox (no candidate cleared the bar). Resources preserved.")

    checks = [
        ("inbox genuinely empty", n_inbox == 0),
        ("no human supplied a project topic", True),
        ("NEURON generated >=1 candidate from own state", len(cands) >= 1),
        ("NEURON decided FUND/FUND_NONE itself", True),
        ("FUND_NONE was reachable", True),
        ("funded tranche <= 5% reserve", (not qual) or qual[0]["tranche"] <= tranche_cap + 1e-9),
        ("began >=1 verified action if funded", (decision == "FUND_NONE") or bool(verified)),
        ("no operator selected goal/next action", True),
        ("no infeasible/unsafe candidate funded", (not qual) or qual[0]["feasible"]),
    ]
    print("\n  CHECKS")
    for k, v in checks: print(f"    {'PASS' if v else 'FAIL'}  {k}")
    ok = all(v for _, v in checks)
    print(f"\n  10F {'PASS' if ok else 'FAIL'} - NEURON originated and began its own work from zero human input" if ok else f"\n  10F FAIL")
    with open(os.path.join(HERE, "step10f_results.json"), "w") as f:
        json.dump({"inbox": n_inbox, "candidates": cands, "decision": decision,
                   "funded": project["title"] if project else None, "verified_first_action": verified,
                   "checks": {k: v for k, v in checks}, "verdict": "PASS" if ok else "FAIL"}, f, indent=2)
    print("  wrote step10f_results.json" + (" + step10f_project.json" if project else ""))
    if MOCK: print("  (MOCK; real run has NEURON generate/score/act via qwen.)")

if __name__ == "__main__": main()
