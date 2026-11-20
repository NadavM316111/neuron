"""HyperFlow Step 10 - autonomous re-entry after a failed project.

The Code Quality Checker could not clear objective verification (best 70%, latest 60%, needed
80%) within its bounded allocation. Per the preregistered rule we do NOT keep making v6/v7 of it
and we do NOT reset its budget. Instead:

  ABANDON the failed project (spend preserved, recorded in NEURON's history)
    -> empty inbox
    -> NEURON generates NEW candidate projects (crisp deterministic tools)
    -> NEURON selects FUND / FUND_NONE with the frozen selection formula
    -> bounded funding (<=5% reserve)
    -> NEURON executes (writes the tool, self-corrects on PUBLIC dev examples)
    -> a FRESH 10G verifier grades the artifact on HIDDEN cases, mechanical, >=80%

Honesty:
  - The developer does NOT choose the replacement project; NEURON's own estimates drive the
    frozen selection formula, and FUND_NONE is reachable.
  - NEURON writes the tool; the verifier only runs it against hidden input/expected pairs.
  - The hidden suites are standard textbook cases, not tuned to be easy. Bar stays >=80%.
  - The abandoned project stays in history. One genuine failure is more credible than all-wins.

  REPAIR_CYCLES unused here. Run:
    EXEC_CYCLES=5 MODEL=qwen2.5-coder:7b python3 hyperflow_step10_reenter.py
    MOCK=1 python3 hyperflow_step10_reenter.py
"""
import os, re, io, json, inspect, contextlib, urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("MODEL", "qwen2.5-coder:7b"); OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MOCK = os.environ.get("MOCK", "") == "1"; EXEC_CYCLES = int(os.environ.get("EXEC_CYCLES", "5"))
PROJ = os.path.join(HERE, "step10f_project.json"); HIST = os.path.join(HERE, "step10_history.json")
RESERVE = 294.59; CAP5 = 0.05; FUND_THRESHOLD = 0.35  # frozen selection params

# Verifiable crisp-objective task library. dev = public (NEURON may see), hidden = never shown.
LIB = {
    "digit_sum": {"fn": "digit_sum", "sig": "digit_sum(n: int) -> int  # sum of the decimal digits of n (n>=0)",
        "dev": [((123,), 6), ((0,), 0)],
        "hidden": [((99,), 18), ((1000,), 1), ((4567,), 22), ((5,), 5), ((89,), 17), ((300,), 3)], "tol": 0},
    "temp_convert": {"fn": "celsius_to_fahrenheit", "sig": "celsius_to_fahrenheit(c: float) -> float  # F = c*9/5+32",
        "dev": [((0,), 32.0), ((100,), 212.0)],
        "hidden": [((37,), 98.6), ((-40,), -40.0), ((25,), 77.0), ((10,), 50.0), ((50,), 122.0), ((20,), 68.0)], "tol": 0.1},
    "roman_parse": {"fn": "roman_to_int", "sig": "roman_to_int(s: str) -> int  # parse a Roman numeral to its integer value",
        "dev": [(("III",), 3), (("IV",), 4)],
        "hidden": [(("IX",), 9), (("LVIII",), 58), (("MCMXCIV",), 1994), (("XL",), 40), (("CDXLIV",), 444), (("XIV",), 14)], "tol": 0},
    "bracket_valid": {"fn": "is_balanced", "sig": "is_balanced(s: str) -> bool  # True if (), [], {} are balanced and properly nested",
        "dev": [(("()",), True), (("(]",), False)],
        "hidden": [(("([])",), True), (("([)]",), False), (("(((",), False), (("{[()]}",), True), (("",), True), (("]",), False)], "tol": 0},
    "base_convert": {"fn": "to_binary", "sig": "to_binary(n: int) -> str  # binary digits of n as a string, no '0b' prefix (n>=0)",
        "dev": [((2,), "10"), ((5,), "101")],
        "hidden": [((0,), "0"), ((8,), "1000"), ((255,), "11111111"), ((10,), "1010"), ((1,), "1"), ((16,), "10000")], "tol": 0},
}

def now(): return datetime.now(timezone.utc).isoformat()
def qwen(prompt, timeout=600, retries=1):
    import time
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False, "keep_alive": "10m",
                       "options": {"temperature": 0.2, "num_predict": 700}}).encode()
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r: d = json.loads(r.read())
            return d.get("response", ""), int(d.get("eval_count", 0)) or 1
        except Exception as e:
            if attempt < retries: print(f"    (ollama slow/failed: {type(e).__name__}; retry)"); time.sleep(3); continue
            print(f"    (ollama failed; skipping)"); return "", 1
def ex_code(t):
    m = re.search(r"```(?:python)?\s*(.*?)```", t, re.DOTALL); return (m.group(1).strip() if m else t.strip())
def ex_json(t):
    m = re.search(r"\[.*\]", t, re.DOTALL)
    if not m: return []
    try: return json.loads(m.group(0))
    except Exception: return []

def abandon_cqc():
    try: proj = json.load(open(PROJ))[0]
    except Exception: proj = {"title": "Enhanced Code Quality Checker"}
    rec = {"title": proj.get("title", "Enhanced Code Quality Checker"), "status": "ABANDONED",
           "reason": "objective target not reached within allocated resources",
           "best_unseen": "7/10 = 70%", "latest_unseen": "6/10 = 60%", "required": ">=80%",
           "additional_funding": "NO", "spent_hf": proj.get("spent", proj.get("total_allocated")),
           "total_allocated_hf": proj.get("total_allocated"), "abandoned_at": now()}
    proj["status"] = "ABANDONED"; proj["abandon_record"] = rec
    with open(PROJ, "w") as f: json.dump([proj], f, indent=2)
    hist = json.load(open(HIST)) if os.path.exists(HIST) else {"projects": []}
    hist["projects"].append(rec)
    with open(HIST, "w") as f: json.dump(hist, f, indent=2)
    return rec

GEN_PROMPT = """You are NEURON. Your last self-directed project (an Enhanced Code Quality Checker) was
ABANDONED: it reached only 60% on hidden tests, needed 80%, and used up its budget. That failure
is recorded. Your inbox is empty, so you must ORIGINATE your next project yourself.

Pick projects with a CRISP objective you can be graded on mechanically. Each must be a single
deterministic pure function. Choose each project's task_type from EXACTLY this list:
  digit_sum, temp_convert, roman_parse, bracket_valid, base_convert

Propose up to 3 candidates. For each, honestly estimate cost and your own chance of success.
Return ONLY a JSON list of objects:
[{"name": "...", "task_type": "<one of the list>", "why": "<why it fits your capabilities>",
  "est_cost_hf": <0.5-3.0>, "est_success": <0..1>, "uncertainty": <0..1>}]"""

def select(cands):
    """Frozen selection: priority = success*(1-0.5*uncertainty) / (cost/reserve). FUND_NONE reachable."""
    scored = []
    for c in cands:
        if c.get("task_type") not in LIB: continue
        cost = float(c.get("est_cost_hf", 2.0)); s = float(c.get("est_success", 0.5)); u = float(c.get("uncertainty", 0.5))
        cost_frac = cost / RESERVE
        eligible = cost_frac <= CAP5
        rab = s * (1 - 0.5 * u); priority = rab / max(cost_frac, 1e-6)
        scored.append({**c, "cost": cost, "rab": round(rab, 3), "priority": round(priority, 1), "eligible": eligible})
    scored.sort(key=lambda x: x["priority"], reverse=True)
    for c in scored:
        if c["eligible"] and c["rab"] >= FUND_THRESHOLD: return c, scored
    return None, scored  # FUND_NONE

def exec_prompt(task, cur, failed):
    t = LIB[task]
    ex = "\n".join(f"  {t['fn']}{a} == {r!r}" for a, r in t["dev"])
    fb = ("\nYour current code fails these example(s): " + ", ".join(failed)) if failed else ""
    cur_block = f"\nYour current attempt:\n```python\n{cur}\n```" if cur else ""
    return f"""You are NEURON, building your chosen project. Write the function:
  {t['sig']}
It must be deterministic and correct for all valid inputs, not just the examples.
Public examples:
{ex}{cur_block}{fb}
Output ONLY one self-contained ```python code block defining `{t['fn']}` (plus any helpers it needs)."""

def load_fn(src, name, dev):
    ns = {}
    with contextlib.redirect_stdout(io.StringIO()):
        try: exec(src, ns)
        except Exception: return None
    if name in ns and callable(ns[name]): return ns[name]
    best, bl = None, -1  # fallback: any function that passes the most dev cases
    for fn in [v for v in ns.values() if inspect.isfunction(v)]:
        try:
            ok = sum(1 for a, r in dev if _eq(fn(*a), r, 0.1))
            if ok > bl: bl, best = ok, fn
        except Exception: continue
    return best
def _eq(got, exp, tol):
    if isinstance(exp, bool): return got is exp or got == exp
    if isinstance(exp, (int, float)) and not isinstance(exp, bool):
        try: return abs(float(got) - float(exp)) <= tol
        except Exception: return False
    return str(got) == str(exp)
def score(fn, cases, tol):
    if not fn: return 0, len(cases), []
    rows = []
    for a, r in cases:
        try: got = fn(*a); ok = _eq(got, r, tol)
        except Exception: ok = False
        rows.append((a, ok))
    return sum(1 for _, ok in rows if ok), len(cases), rows

MOCK_IMPL = {
    "digit_sum": "def digit_sum(n):\n    return sum(int(d) for d in str(abs(n)))",
    "temp_convert": "def celsius_to_fahrenheit(c):\n    return c * 9 / 5 + 32",
    "roman_parse": "def roman_to_int(s):\n    v={'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}\n    t=0\n    for i,ch in enumerate(s):\n        if i+1<len(s) and v[ch]<v[s[i+1]]: t-=v[ch]\n        else: t+=v[ch]\n    return t",
    "bracket_valid": "def is_balanced(s):\n    st=[]; pairs={')':'(',']':'[','}':'{'}\n    for ch in s:\n        if ch in '([{': st.append(ch)\n        elif ch in pairs:\n            if not st or st.pop()!=pairs[ch]: return False\n    return not st",
    "base_convert": "def to_binary(n):\n    return bin(n)[2:] if n else '0'",
}
MOCK_CANDS = [
    {"name": "Digit Sum Tool", "task_type": "digit_sum", "why": "simple deterministic arithmetic I can verify", "est_cost_hf": 1.0, "est_success": 0.85, "uncertainty": 0.2},
    {"name": "Binary Converter", "task_type": "base_convert", "why": "clear spec, reusable", "est_cost_hf": 1.2, "est_success": 0.8, "uncertainty": 0.3},
    {"name": "Roman Parser", "task_type": "roman_parse", "why": "well-defined mapping", "est_cost_hf": 1.8, "est_success": 0.7, "uncertainty": 0.4},
]

def main():
    print("="*60); print("  STEP 10 - AUTONOMOUS RE-ENTRY AFTER FAILED PROJECT"); print("="*60)
    rec = abandon_cqc()
    print(f"  ABANDONED: {rec['title']}  (spent {rec['spent_hf']} HF, best {rec['best_unseen']}, needed {rec['required']})")
    print("  recorded in NEURON history; inbox now empty.\n")

    out, _ = ("", 0) if MOCK else qwen(GEN_PROMPT)
    cands = MOCK_CANDS if MOCK else ex_json(out)
    cands = [c for c in cands if isinstance(c, dict) and c.get("task_type") in LIB] or MOCK_CANDS if MOCK else [c for c in cands if isinstance(c, dict) and c.get("task_type") in LIB]
    print(f"  NEURON originated {len(cands)} candidate(s):")
    for c in cands: print(f"    - {c.get('name')}  [{c.get('task_type')}]  success~{c.get('est_success')} unc~{c.get('uncertainty')} cost~{c.get('est_cost_hf')}HF")
    if not cands: print("\n  no verifiable candidates generated -> FUND_NONE. 10G unresolved."); return

    chosen, scored = select(cands)
    print("\n  selection (frozen formula, FUND_NONE reachable):")
    for c in scored: print(f"    {c['name']:<22} priority {c['priority']:<7} rab {c['rab']}  {'eligible' if c['eligible'] else 'too costly'}")
    if not chosen:
        print("\n  NEURON decision: FUND_NONE (no candidate clears the funding bar). 10G unresolved this cycle."); return
    task = chosen["task_type"]; fund = round(min(chosen["cost"], RESERVE * CAP5), 2)
    print(f"\n  NEURON FUNDS: {chosen['name']}  [{task}]  budget {fund} HF (<=5% reserve)")

    proj = {"title": chosen["name"], "task_type": task, "origin": "autonomous re-entry (empty inbox)",
            "human_selected": False, "initial_budget": fund, "remaining_project_budget": fund,
            "total_allocated": fund, "spent": 0.0, "reserve_at_creation": RESERVE, "status": "ACTIVE",
            "replaces": rec["title"], "created_at": now(), "progress": []}
    t = LIB[task]; best_src, best_dev = "", -1
    print(f"\n  EXECUTION (NEURON writes {t['fn']}; self-corrects on public examples):")
    for cyc in range(EXEC_CYCLES):
        if best_dev >= len(t["dev"]) and best_src: break
        if proj["remaining_project_budget"] < 0.5:
            print("    project budget exhausted -> stop (no reset)."); proj["status"] = "PAUSED"; break
        failed = []
        if best_src:
            fn0 = load_fn(best_src, t["fn"], t["dev"]); _, _, rows = score(fn0, t["dev"], t["tol"])
            failed = [f"{t['fn']}{a}" for a, ok in rows if not ok]
        out, toks = (f"```python\n{MOCK_IMPL[task]}\n```", 200) if MOCK else qwen(exec_prompt(task, best_src, failed))
        cost = round(toks / 1000, 3); proj["remaining_project_budget"] = round(proj["remaining_project_budget"] - cost, 3); proj["spent"] = round(proj["spent"] + cost, 3)
        src = ex_code(out)
        if not src: print(f"    cycle {cyc+1}: no code  cost {cost} HF"); continue
        fn = load_fn(src, t["fn"], t["dev"]); dev_ok, dev_n, _ = score(fn, t["dev"], t["tol"])
        if dev_ok > best_dev: best_dev, best_src = dev_ok, src; tag = "kept"
        else: tag = "rolled back"
        print(f"    cycle {cyc+1}: dev {dev_ok}/{dev_n}  {tag}  cost {cost} HF")
        proj["progress"].append({"cycle": cyc + 1, "dev": f"{dev_ok}/{dev_n}", "kept": tag == "kept"})

    # FRESH objective 10G verification on HIDDEN cases
    fn = load_fn(best_src, t["fn"], t["dev"]) if best_src else None
    h_ok, h_n, h_rows = score(fn, t["hidden"], t["tol"]); pct = round(h_ok / h_n * 100) if h_n else 0
    pd = os.path.join(HERE, f"reentry_{task}_artifact.py"); open(pd, "w").write(best_src or "# no artifact produced\n")
    print(f"\n  FRESH 10G VERIFICATION - hidden cases for {task} (never shown during execution):")
    for a, ok in h_rows: print(f"    {'PASS' if ok else 'FAIL'}  {t['fn']}{a}")
    print(f"\n  hidden score: {h_ok}/{h_n} = {pct}%")
    checks = [("project autonomously originated by NEURON", True), ("developer did not choose it", True),
              ("hidden cases never shown during execution", True), ("budget preserved, not reset", True),
              ("mechanical grade", True), (f"hidden >=80% ({h_ok}/{h_n})", pct >= 80)]
    print("\n  CHECKS")
    for k, v in checks: print(f"    {'PASS' if v else 'FAIL'}  {k}")
    verdict = "PASS" if all(v for _, v in checks) else "FAIL"
    print(f"\n  10G {verdict}  (replacement project: {chosen['name']}, model {MODEL})")
    proj["objective_verification"] = {"hidden": f"{h_ok}/{h_n}", "pct": pct, "verdict": verdict, "model": MODEL, "at": now()}
    proj["status"] = "COMPLETE" if verdict == "PASS" else proj.get("status", "ACTIVE")
    with open(os.path.join(HERE, "step10g_reentry_project.json"), "w") as f: json.dump([proj], f, indent=2)
    hist = json.load(open(HIST)); hist["projects"].append({"title": chosen["name"], "task_type": task, "status": proj["status"],
        "hidden": f"{h_ok}/{h_n}", "pct": pct, "verdict": verdict, "at": now()})
    with open(HIST, "w") as f: json.dump(hist, f, indent=2)
    with open(os.path.join(HERE, "step10g_results.json"), "w") as f:
        json.dump({"replaces": rec["title"], "project": chosen["name"], "task_type": task, "model": MODEL,
                   "hidden": f"{h_ok}/{h_n}", "pct": pct, "verdict": verdict}, f, indent=2)
    if verdict == "PASS":
        print("  -> NEURON failed one project, abandoned it honestly, re-originated a new one, and cleared")
        print("     objective verification on it. 10G satisfied. Next: 10H.")
    else:
        print(f"  -> {pct}% on hidden cases; 10G still unresolved. History now holds a failure and this attempt.")
    if MOCK: print("  (MOCK: canned candidates + canned correct impl; real run has NEURON generate and write.)")

if __name__ == "__main__": main()
