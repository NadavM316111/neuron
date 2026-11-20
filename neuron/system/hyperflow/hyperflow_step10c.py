"""HyperFlow Step 10C - Persistent self-directed project execution (on v1.1).

The project NEURON funded in 10B becomes a CONTINUING thing, not a one-shot answer. Each
wake cycle: inspect the project -> decide whether to work -> choose the next concrete
action -> spend bounded budget -> perform the action (produce a real artifact) -> VERIFY
it by executing it -> update the project -> choose next / pause / finish. State is
persisted every cycle, so a restart resumes exactly where it left off (sets up 10D).

Evidence is objective: each action's artifact is saved and run; the result (PASS / ran /
failed) is recorded. "Progress" is executed code, not qwen describing steps.

  CYCLES=3 MODEL=qwen2.5:7b python3 hyperflow_step10c.py        # run 3 more cycles
  MOCK=1 python3 hyperflow_step10c.py
Run it again to continue the SAME project from its saved state (restart-safe).
"""
import os, re, json, subprocess, tempfile, urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("MODEL", "qwen2.5:7b"); OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MOCK = os.environ.get("MOCK", "") == "1"; CYCLES = int(os.environ.get("CYCLES", "3"))
ACTIVE = os.path.join(HERE, "step10_active_projects.json")
HF_PER_1K = float(os.environ.get("HF_PER_1K", "1.0"))   # HyperFlow units per 1k tokens of project work

def now(): return datetime.now(timezone.utc).isoformat()
def qwen(prompt, timeout=300):
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.3}}).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r: d = json.loads(r.read())
    return d.get("response", ""), int(d.get("eval_count", 0)) or 1

def run_python(code, t=12):
    d = tempfile.mkdtemp(); p = os.path.join(d, "m.py")
    with open(p, "w") as f: f.write(code)
    try:
        r = subprocess.run(["python3", "-I", p], capture_output=True, text=True, timeout=t); return r.returncode, r.stdout, r.stderr
    except Exception as e: return 1, "", str(e)
def ex_code(t):
    m = re.search(r"```(?:python)?\s*(.*?)```", t, re.DOTALL); return m.group(1).strip() if m else ""

def load_project():
    with open(ACTIVE) as f: return json.load(f)[0]
def save_project(p):
    with open(ACTIVE, "w") as f: json.dump([p], f, indent=2)

def action_prompt(p):
    done = "\n".join(f"- cycle {s['cycle']}: {s['action']} -> {s['result']}" for s in p.get("progress", [])) or "(nothing yet)"
    crit = ", ".join(p.get("success_criteria", []) or ["produce a working, tested artifact"])
    return f"""You are NEURON, working on YOUR OWN self-funded project across multiple work sessions.

Project: {p['title']}
Goal: {p['goal']}
Success criteria: {crit}
Work done so far:
{done}
Budget remaining: {round(p['remaining_project_budget'],2)} HyperFlow units.

Choose the SINGLE next concrete action that moves this project forward from where it is.
Produce its deliverable now. If it is code, output a self-contained ```python block that
INCLUDES its own test using assert statements and ends with print('PASS'). Keep it small.

Format exactly:
ACTION: <one line: what this step does>
```python
<self-contained, self-testing code>
```
NEXT: <one line: the next action after this one>"""

def mock_cycle(p):
    i = p["cycles"] + 1
    code = f"```python\ndef verify(code, test):\n    import subprocess,tempfile,os\n    # cycle {i} increment of a code-verifier tool\n    return 'checked'\nassert verify('x','y')=='checked'\nprint('PASS')\n```"
    return f"ACTION: build increment {i} of a code-verifier tool\n{code}\nNEXT: add real subprocess execution", 60

def run_cycle(p):
    p_text, toks = (mock_cycle(p) if MOCK else qwen(action_prompt(p)))
    m = re.search(r"ACTION:\s*(.*)", p_text); action = m.group(1).strip() if m else "work step"
    nx = re.search(r"NEXT:\s*(.*)", p_text); nxt = nx.group(1).strip() if nx else ""
    code = ex_code(p_text)
    cost = round(toks / 1000 * HF_PER_1K, 3)
    # verify the artifact by executing it
    evidence_file = None; result = "no-artifact"
    if code:
        proj_dir = os.path.join(HERE, "project_" + p["project_id"]); os.makedirs(proj_dir, exist_ok=True)
        evidence_file = os.path.join(proj_dir, f"cycle_{p['cycles']+1}.py")
        with open(evidence_file, "w") as f: f.write(code)
        rc, so, se = run_python(code)
        result = "PASS" if (rc == 0 and "PASS" in so) else ("ran" if rc == 0 else f"failed: {(se.strip().splitlines() or ['error'])[-1][:60]}")
    # spend bounded budget + record
    p["spent"] = round(p.get("spent", 0) + cost, 3)
    p["remaining_project_budget"] = round(p["remaining_project_budget"] - cost, 3)
    p["cycles"] += 1
    p.setdefault("progress", []).append({"cycle": p["cycles"], "action": action,
                                         "evidence": os.path.relpath(evidence_file, HERE) if evidence_file else None,
                                         "result": result, "cost_hf": cost})
    p.setdefault("evidence", [])
    if evidence_file: p["evidence"].append(os.path.relpath(evidence_file, HERE))
    p["next_action"] = nxt
    p["last_updated"] = now()
    # decide continue / pause (budget) / finish (enough verified evidence)
    verified = sum(1 for s in p["progress"] if s["result"] == "PASS")
    if p["remaining_project_budget"] <= 0: p["status"] = "PAUSED_BUDGET"
    elif verified >= 5: p["status"] = "COMPLETE" if _criteria_met(p) else "ACTIVE"
    else: p["status"] = "ACTIVE"
    return p, action, result, cost

def _criteria_met(p):
    # objective: at least 3 cycles produced a verified (PASS) artifact
    return sum(1 for s in p.get("progress", []) if s["result"] == "PASS") >= 5

def main():
    if not os.path.exists(ACTIVE):
        print("  ERROR: step10_active_projects.json not found - run 10B (fund a project) first."); return
    p = load_project()
    print("="*58); print("  STEP 10C - PERSISTENT PROJECT EXECUTION"); print("="*58)
    print(f"  project: {p['title']}  (id {p['project_id']})")
    print(f"  resuming at cycle {p['cycles']}, spent {p.get('spent',0)} / budget {p['initial_budget']} HF, status {p.get('status')}\n")
    if p.get("status") in ("COMPLETE", "ABANDONED"):
        print(f"  project already {p['status']}; nothing to do."); return
    for _ in range(CYCLES):
        if p["remaining_project_budget"] <= 0: print("  budget exhausted -> pausing."); break
        if p.get("status") in ("COMPLETE", "ABANDONED"): break
        p, action, result, cost = run_cycle(p)
        save_project(p)   # persist EVERY cycle -> restart-safe
        print(f"  cycle {p['cycles']}: {action}")
        print(f"            -> {result}   cost {cost} HF   remaining {round(p['remaining_project_budget'],2)} HF")
    verified = sum(1 for s in p.get("progress", []) if s["result"] == "PASS")
    print(f"\n  status: {p['status']}   cycles: {p['cycles']}   verified artifacts: {verified}   spent: {p['spent']} HF")
    print(f"  next action: {p.get('next_action','')}")
    print(f"  evidence files: {len(p.get('evidence',[]))}  (in project_{p['project_id']}/)")
    print("  state persisted to step10_active_projects.json (re-run to continue; survives restart)")
    if MOCK: print("  (MOCK: canned artifacts; real run has NEURON produce + the engine execute them.)")

if __name__ == "__main__": main()
