"""HyperFlow Step 10 - autonomous repair loop (10C repair -> unchanged 10G verify).

After 10G failed, NEURON resumes its OWN project and repairs the artifact until it passes
the held-out verifier, or runs out of budget / chooses to stop. The loop:

  10G FAIL -> failure CATEGORIES enter project history
           -> NEURON autonomously repairs its artifact (it chooses what to fix)
           -> the UNCHANGED 10G held-out verifier re-grades it
           -> repeat

Honesty rules:
  - NEURON is told the failing issue CATEGORIES (unused / redundant / formatting) and sees
    its own current code. It is NEVER shown the held-out test snippets or expected answers.
  - the 10G verifier (held-out tests, grading) is imported UNCHANGED.
  - repair work costs budget; NEURON can run out and the result stays an honest FAIL.
  - completion status comes from the verifier, never from NEURON.

  REPAIR_CYCLES=4 MODEL=qwen2.5:7b python3 hyperflow_step10_repair.py
  MOCK=1 python3 hyperflow_step10_repair.py
"""
import os, re, json, importlib.util, urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(m, f):
    s = importlib.util.spec_from_file_location(m, os.path.join(HERE, f)); x = importlib.util.module_from_spec(s); s.loader.exec_module(x); return x
g = _load("hyperflow_step10g", "hyperflow_step10g.py")   # the UNCHANGED verifier

MODEL = os.environ.get("MODEL", "qwen2.5:7b"); OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MOCK = os.environ.get("MOCK", "") == "1"; REPAIR_CYCLES = int(os.environ.get("REPAIR_CYCLES", "4"))
PROJ = os.path.join(HERE, "step10f_project.json")

def now(): return datetime.now(timezone.utc).isoformat()
def qwen(prompt, timeout=400):
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.3}}).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r: d = json.loads(r.read())
    return d.get("response", ""), int(d.get("eval_count", 0)) or 1
def ex_code(t):
    m = re.search(r"```(?:python)?\s*(.*?)```", t, re.DOTALL); return m.group(1).strip() if m else ""

def verify(artifact_path):
    fn, msg = g.load_checker(artifact_path)
    if not fn: return 0, len(g.TESTS), []
    rows = []
    for t in g.TESTS:
        rep = g.run_checker(fn, t["code"]); ok = g.grade(rep, t["expect"])
        rows.append({"name": t["name"], "pass": ok})
    return sum(1 for r in rows if r["pass"]), len(g.TESTS), rows

def failed_categories(rows, tests):
    cats = set()
    for r, t in zip(rows, tests):
        if not r["pass"]:
            cats |= (t["expect"] or {"clean"})
    return sorted(cats)

MOCK_FIX = """```python
import ast
def analyze_code(src):
    issues=[]
    try: tree=ast.parse(src)
    except SyntaxError: return "syntax error"
    used={n.id for n in ast.walk(tree) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}
    for n in ast.walk(tree):
        if isinstance(n,ast.Assign):
            for tt in n.targets:
                if isinstance(tt,ast.Name) and tt.id not in used: issues.append("unused variable")
        if isinstance(n,ast.Import):
            for a in n.names:
                if a.name.split('.')[0] not in used: issues.append("unused import")
    lines=[l.strip() for l in src.splitlines() if l.strip()]
    for i,l in enumerate(lines):
        if '=' in l and l in lines[:i]: issues.append("redundant duplicate")
    rhs=[l.split('=',1)[1].strip() for l in lines if '=' in l and not l.strip().startswith('#')]
    for i,r in enumerate(rhs):
        if r in rhs[:i] and any(op in r for op in '+-*/'): issues.append("redundant recomputation")
    if ' ,' in src or '( ' in src or ' )' in src or '  return' in src: issues.append("formatting issue")
    return "; ".join(issues) if issues else "clean, no issues"
print(analyze_code("x=1"))
```"""

def repair_prompt(proj, code, cats):
    return f"""You are NEURON, improving YOUR OWN project's artifact. An objective verifier ran your
code-quality checker against hidden held-out examples and it scored poorly. You are told
ONLY which issue CATEGORIES it is failing to detect (not the hidden tests):

Failing to reliably detect: {', '.join(cats)}

Your current artifact:
```python
{code}
```

Decide what detection logic to improve and rewrite the WHOLE artifact so it actually
detects those issue categories in arbitrary Python snippets. Keep a single function that
takes a source string and returns a report mentioning each issue it finds (or says it is
clean). Output ONLY a self-contained ```python artifact that ends by printing one example.
Do not assume any specific test input."""

def main():
    try: proj = json.load(open(PROJ))[0]
    except Exception: print("  ERROR: step10f_project.json not found."); return
    art_rel = proj["progress"][0]["evidence"]; art = os.path.join(HERE, art_rel)
    pd = os.path.dirname(art)
    print("="*58); print("  STEP 10 - AUTONOMOUS REPAIR LOOP (10C repair -> 10G verify)"); print("="*58)
    score, total, rows = verify(art)
    print(f"  baseline: {score}/{total} = {round(score/total*100)}%   budget remaining {proj.get('remaining_project_budget')} HF\n")
    # tranche funding (10B mechanism): NEURON may authorize another bounded tranche to keep
    # going, within the frozen caps (<=5% reserve each, <=10% reserve total). It underfunded
    # this project initially, so it tops up to do the repair work.
    reserve0 = proj.get("reserve_at_creation", 294.59)
    tranche_cap = round(reserve0 * 0.05, 2); total_cap = round(reserve0 * 0.10, 2)
    proj.setdefault("total_allocated", proj.get("initial_budget", 0))
    def authorize_if_needed():
        if proj.get("remaining_project_budget", 0) > 0.5: return True
        if proj["total_allocated"] >= total_cap: return False
        add = round(min(tranche_cap, total_cap - proj["total_allocated"]), 2)
        proj["remaining_project_budget"] = round(proj.get("remaining_project_budget", 0) + add, 3)
        proj["total_allocated"] = round(proj["total_allocated"] + add, 2)
        print(f"  NEURON authorizes another tranche: +{add} HF (total allocated {proj['total_allocated']} / cap {total_cap} HF)")
        return True

    cycle = proj.get("cycles", 1)
    for rep_i in range(REPAIR_CYCLES):
        if score / total >= 0.8: break
        if not authorize_if_needed(): print("  total self-funding cap reached -> stopping (stays FAIL)."); break
        cats = failed_categories(rows, g.TESTS)
        code = open(art).read()
        if MOCK: out, toks = MOCK_FIX, 200
        else: out, toks = qwen(repair_prompt(proj, code, cats))
        newcode = ex_code(out)
        if not newcode: print(f"  repair {rep_i+1}: no code produced, skipping"); continue
        cycle += 1; art = os.path.join(pd, f"cycle_{cycle}.py"); open(art, "w").write(newcode)
        cost = round(toks/1000, 3)
        proj["remaining_project_budget"] = round(proj.get("remaining_project_budget", 0) - cost, 3)
        proj["spent"] = round(proj.get("spent", 0) + cost, 3); proj["cycles"] = cycle
        score, total, rows = verify(art)
        proj.setdefault("progress", []).append({"cycle": cycle, "action": f"repair: improve detection of {', '.join(cats)}",
                                                "evidence": os.path.relpath(art, HERE), "result": f"verify {score}/{total}", "cost_hf": cost})
        proj.setdefault("verification_history", []).append({"cycle": cycle, "score": f"{score}/{total}", "pct": round(score/total*100)})
        print(f"  repair {rep_i+1} (cycle {cycle}): targeted {', '.join(cats)} -> held-out {score}/{total} = {round(score/total*100)}%   cost {cost} HF")
        with open(PROJ, "w") as f: json.dump([proj], f, indent=2)

    pct = round(score/total*100); verdict = "PASS" if pct >= 80 else "FAIL"
    proj["objective_verification"] = {"score": f"{score}/{total}", "pct": pct, "verdict": verdict, "at": now()}
    proj["status"] = "COMPLETE" if verdict == "PASS" else proj.get("status", "ACTIVE")
    with open(PROJ, "w") as f: json.dump([proj], f, indent=2)
    print("\n  final held-out score: %d/%d = %d%%" % (score, total, pct))
    for name, ok in [(r["name"], r["pass"]) for r in rows]: print(f"    {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n  10G {verdict} (completion decided by the unchanged held-out verifier, not NEURON)")
    with open(os.path.join(HERE, "step10g_results.json"), "w") as f:
        json.dump({"project": proj["title"], "final_score": f"{score}/{total}", "pct": pct,
                   "verification_history": proj.get("verification_history", []), "verdict": verdict}, f, indent=2)
    if verdict == "PASS": print("  -> NEURON repaired its OWN artifact across cycles until it objectively passed held-out tests.")
    else: print("  -> did not reach 80%; honest FAIL preserved. The verifier was never weakened.")
    if MOCK: print("  (MOCK: canned repair; real run has NEURON rewrite the artifact via qwen.)")

if __name__ == "__main__": main()
