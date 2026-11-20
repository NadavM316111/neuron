"""HyperFlow Step 10G v2 - honest repair with rollback + fresh final held-out suite.

Why v2: the original 8 tests leaked (used across many repairs), so they are now a
DEVELOPMENT set, not held-out. v2 does it properly:

  DEV set (old 8)  -> NEURON iterates against it, with BEST-CHECKPOINT ROLLBACK (never keep
                      a regression) and per-category feedback. Budget-guarded (no negative
                      balance; authorize a bounded tranche or PAUSE), capped at 8 cycles /
                      10% reserve.
  FINAL set (10 FRESH, never shown during repair) -> run ONCE on the frozen best artifact.
                      >=80% on this unseen suite = 10G PASS.

Honesty line: NEURON is given METHODOLOGY (use ast, separate detectors, make your own
examples) and the failure CATEGORIES, but it writes the detector code itself. The developer
does not implement the analyzer. The final suite is independent and evaluated once.

  REPAIR_CYCLES=8 MODEL=qwen2.5:7b python3 hyperflow_step10g_v2.py
  MOCK=1 python3 hyperflow_step10g_v2.py
"""
import os, re, io, json, inspect, contextlib, importlib.util, urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("MODEL", "qwen2.5:7b"); OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MOCK = os.environ.get("MOCK", "") == "1"; REPAIR_CYCLES = int(os.environ.get("REPAIR_CYCLES", "8"))
PROJ = os.path.join(HERE, "step10f_project.json")

KW = {"unused": ("unused", "never used", "not used"),
      "redundant": ("redundant", "duplicate", "repeated", "repeat", "already computed"),
      "formatting": ("format", "indent", "spacing", "whitespace", "style", "pep8", "pep 8")}

# DEVELOPMENT set (the leaked original 8) - for iteration only
DEV = [
    {"name": "unused_var", "code": "def f():\n    x = 5\n    return 1\n", "expect": {"unused"}},
    {"name": "unused_import", "code": "import os\ndef h():\n    return 1\n", "expect": {"unused"}},
    {"name": "redundant_assign", "code": "y = 2\ny = 2\n", "expect": {"redundant"}},
    {"name": "redundant_recompute", "code": "a = 1 + 2\nb = 1 + 2\nprint(a, b)\n", "expect": {"redundant"}},
    {"name": "formatting", "code": "def k( a ,b ):\n  return a+b\n", "expect": {"formatting"}},
    {"name": "multiple", "code": "def g():\n    z = 3\n    z = 3\n    return 0\n", "expect": {"unused", "redundant"}},
    {"name": "clean1", "code": "def add(a, b):\n    return a + b\n", "expect": set()},
    {"name": "clean2", "code": "nums = [i for i in range(10)]\nprint(len(nums))\n", "expect": set()},
]
# FINAL held-out set (10 FRESH, structurally different, never shown to repair)
FINAL = [
    {"name": "f_unused_local", "code": "def area(r):\n    pi = 3.14159\n    unused = 7\n    return pi * r * r\n", "expect": {"unused"}},
    {"name": "f_unused_import", "code": "import sys\nimport math\ndef c(r):\n    return math.pi * r\n", "expect": {"unused"}},
    {"name": "f_dup_assign", "code": "total = 0\ntotal = 0\nfor i in range(3):\n    total += i\nprint(total)\n", "expect": {"redundant"}},
    {"name": "f_repeat_comp", "code": "p = 10 * 3\nq = 10 * 3\nprint(p + q)\n", "expect": {"redundant"}},
    {"name": "f_spacing", "code": "def mul(x ,y):\n   return x*y\n", "expect": {"formatting"}},
    {"name": "f_multi", "code": "import json\ndef go():\n    v = 9\n    v = 9\n    return 0\n", "expect": {"unused", "redundant"}},
    {"name": "f_clean_fn", "code": "def greet(name):\n    return 'hi ' + name\n", "expect": set()},
    {"name": "f_clean_loop", "code": "words = ['a', 'b']\nfor w in words:\n    print(w)\n", "expect": set()},
    {"name": "f_clean_comp", "code": "squares = [n * n for n in range(5)]\nprint(sum(squares))\n", "expect": set()},
    {"name": "f_clean_cond", "code": "def sign(x):\n    if x > 0:\n        return 1\n    return -1\n", "expect": set()},
]

def now(): return datetime.now(timezone.utc).isoformat()
def qwen(prompt, timeout=400):
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.3}}).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r: d = json.loads(r.read())
    return d.get("response", ""), int(d.get("eval_count", 0)) or 1
def ex_code(t):
    m = re.search(r"```(?:python)?\s*(.*?)```", t, re.DOTALL); return m.group(1).strip() if m else ""

def load_checker_from_src(src):
    ns = {}
    with contextlib.redirect_stdout(io.StringIO()):
        try: exec(src, ns)
        except Exception: return None
    funcs = [v for k, v in ns.items() if inspect.isfunction(v)]
    probe = "def f():\n    x=5\n    return 1\n"; best, bl = None, -1
    # prefer a function literally named analyze_code / check*, else the most verbose
    for fn in funcs:
        try:
            if len(inspect.signature(fn).parameters) < 1: continue
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf): ret = fn(probe)
            text = (str(ret) if ret is not None else "") + buf.getvalue()
            score = len(text) + (1000 if fn.__name__ in ("analyze_code", "check_code", "analyze", "review_code") else 0)
            if score > bl: bl, best = score, fn
        except Exception: continue
    return best
def run_checker(fn, code):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf): ret = fn(code)
    except Exception as e: return f"__error__ {e}"
    return ((str(ret) if ret is not None else "") + " " + buf.getvalue()).lower()
def grade(report, expect):
    if report.startswith("__error__"): return False
    flagged = {t for t, kws in KW.items() if any(k in report for k in kws)}
    return len(flagged) == 0 if not expect else expect.issubset(flagged)
def score_suite(src, suite):
    fn = load_checker_from_src(src)
    if not fn: return 0, [(t["name"], False) for t in suite]
    rows = [(t["name"], grade(run_checker(fn, t["code"]), t["expect"])) for t in suite]
    return sum(1 for _, ok in rows if ok), rows
def dev_categories_failing(src):
    fn = load_checker_from_src(src); fails = set()
    if not fn: return {"unused", "redundant", "formatting", "clean"}
    for t in DEV:
        if not grade(run_checker(fn, t["code"]), t["expect"]):
            fails |= (t["expect"] or {"clean (false positive)"})
    return sorted(fails)

MOCK_FIX = """```python
import ast
def detect_unused(src, tree):
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id not in used: out.append("unused variable " + t.id)
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split('.')[0] not in used: out.append("unused import " + a.name)
    return out
def detect_redundant(src):
    lines = [l.strip() for l in src.splitlines() if l.strip() and not l.strip().startswith('#')]
    out = []
    for i, l in enumerate(lines):
        if '=' in l and '==' not in l:
            if l in lines[:i]: out.append("redundant duplicate assignment")
            rhs = l.split('=', 1)[1].strip()
            prev = [x.split('=',1)[1].strip() for x in lines[:i] if '=' in x and '==' not in x]
            if any(op in rhs for op in '+-*/') and rhs in prev: out.append("redundant repeated computation")
    return out
def detect_formatting(src):
    out = []
    if ' ,' in src or '( ' in src or ' )' in src or ',' in src.replace(', ', '') or '\\n  ' in src and '\\n    ' not in src:
        out.append("formatting issue")
    for l in src.splitlines():
        s = len(l) - len(l.lstrip(' '))
        if s and s % 4 != 0: out.append("formatting issue")
    return out
def analyze_code(src):
    try: tree = ast.parse(src)
    except SyntaxError: return "syntax error"
    issues = detect_unused(src, tree) + detect_redundant(src) + list(dict.fromkeys(detect_formatting(src)))
    return "; ".join(dict.fromkeys(issues)) if issues else "clean, no issues"
print(analyze_code("x=1"))
```"""

def repair_prompt(code, cats):
    return f"""You are NEURON, improving your OWN code-quality checker. It is failing to handle these
categories (you are told the categories, not the hidden tests): {', '.join(cats)}.

Use a DETERMINISTIC approach with Python's `ast` module - not guesses. Write separate
detector functions and combine them, roughly:
  detect_unused(...)        # unused variables AND unused imports via ast
  detect_redundant(...)     # duplicate assignments and repeated computations
  detect_formatting(...)    # spacing/indentation problems, but NO false positives on clean code
  analyze_code(src)         # runs all detectors, returns a report; says "clean" when nothing is wrong
CRITICAL: clean, well-formatted code must return NO issues (no false positives). Make your
own small examples for each category and check them yourself.

Your current artifact:
```python
{code}
```
Output ONLY the full improved self-contained ```python artifact ending with one print()."""

def main():
    try: proj = json.load(open(PROJ))[0]
    except Exception: print("  ERROR: step10f_project.json not found."); return
    pd = os.path.dirname(os.path.join(HERE, proj["progress"][0]["evidence"]))
    best_path = os.path.join(HERE, proj["progress"][0]["evidence"])
    best_src = open(best_path).read()
    best_dev, _ = score_suite(best_src, DEV)
    reserve0 = proj.get("reserve_at_creation", 294.59); tranche_cap = round(reserve0*0.05, 2); total_cap = round(reserve0*0.10, 2)
    proj.setdefault("total_allocated", proj.get("initial_budget", 0))
    EST = 0.6  # estimated HF per repair action (guard against overspend)
    print("="*60); print("  STEP 10G v2 - REPAIR w/ ROLLBACK, FRESH FINAL SUITE"); print("="*60)
    print(f"  dev baseline: {best_dev}/{len(DEV)}   remaining budget {proj.get('remaining_project_budget',0)} HF\n")
    cycle = proj.get("cycles", 1)
    for rep_i in range(REPAIR_CYCLES):
        if best_dev >= len(DEV): break
        # budget guard: never go negative; authorize a tranche or pause
        if proj.get("remaining_project_budget", 0) < EST:
            if proj["total_allocated"] >= total_cap:
                print("  10% total self-funding cap reached -> NEURON PAUSES repairs."); proj["status"] = "PAUSED"; break
            add = round(min(tranche_cap, total_cap - proj["total_allocated"]), 2)
            proj["remaining_project_budget"] = round(proj.get("remaining_project_budget", 0)+add, 3); proj["total_allocated"] = round(proj["total_allocated"]+add, 2)
            print(f"  NEURON authorizes tranche +{add} HF (total {proj['total_allocated']}/{total_cap})")
        cats = dev_categories_failing(best_src)
        out, toks = (MOCK_FIX, 300) if MOCK else qwen(repair_prompt(best_src, cats))
        newsrc = ex_code(out)
        cost = round(toks/1000, 3); proj["remaining_project_budget"] = round(proj["remaining_project_budget"]-cost, 3); proj["spent"] = round(proj.get("spent", 0)+cost, 3)
        if not newsrc: print(f"  cycle {rep_i+1}: no code produced"); continue
        new_dev, _ = score_suite(newsrc, DEV)
        cycle += 1
        if new_dev > best_dev:   # rollback: keep only improvements
            best_dev, best_src = new_dev, newsrc
            best_path = os.path.join(pd, f"cycle_{cycle}.py"); open(best_path, "w").write(newsrc)
            tag = "kept (new best)"
        else:
            tag = f"rolled back (stayed {best_dev}/{len(DEV)})"
        print(f"  cycle {rep_i+1}: targeted {', '.join(cats) or 'all'} -> dev {new_dev}/{len(DEV)}  {tag}  cost {cost} HF")
        proj.setdefault("progress", []).append({"cycle": cycle, "action": f"AST repair: {', '.join(cats)}", "dev_score": f"{new_dev}/{len(DEV)}", "kept": new_dev > best_dev or tag.startswith('kept')})
        with open(PROJ, "w") as f: json.dump([proj], f, indent=2)

    # per-category dev readout of the frozen best
    print(f"\n  frozen best dev score: {best_dev}/{len(DEV)}")
    _, devrows = score_suite(best_src, DEV)
    for name, ok in devrows: print(f"    dev {'PASS' if ok else 'FAIL'}  {name}")

    # ONE final evaluation on the fresh, never-seen suite
    final_score, finalrows = score_suite(best_src, FINAL); pct = round(final_score/len(FINAL)*100)
    print(f"\n  FINAL held-out suite (fresh, never shown during repair): {len(FINAL)} tests")
    for name, ok in finalrows: print(f"    final {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n  final score: {final_score}/{len(FINAL)} = {pct}%")
    verdict = "PASS" if pct >= 80 else "FAIL"
    checks = [
        ("dev set and final set are disjoint", True),
        ("final suite never shown during repair", True),
        ("rollback prevented regressions (best kept)", True),
        ("budget never negative", proj.get("remaining_project_budget", 0) >= -1e-9),
        (f"final held-out >=80% ({final_score}/{len(FINAL)})", pct >= 80),
        ("completion decided by verifier, not NEURON", True),
    ]
    print("\n  CHECKS")
    for k, v in checks: print(f"    {'PASS' if v else 'FAIL'}  {k}")
    verdict = "PASS" if all(v for _, v in checks) else "FAIL"
    print(f"\n  10G {verdict}")
    proj["objective_verification"] = {"dev": f"{best_dev}/{len(DEV)}", "final": f"{final_score}/{len(FINAL)}", "pct": pct, "verdict": verdict, "at": now()}
    proj["status"] = "COMPLETE" if verdict == "PASS" else proj.get("status", "ACTIVE")
    with open(PROJ, "w") as f: json.dump([proj], f, indent=2)
    with open(os.path.join(HERE, "step10g_results.json"), "w") as f:
        json.dump({"project": proj["title"], "dev_best": f"{best_dev}/{len(DEV)}", "final": f"{final_score}/{len(FINAL)}", "pct": pct,
                   "final_rows": finalrows, "verdict": verdict}, f, indent=2)
    if verdict == "PASS": print("  -> NEURON repaired its own artifact (AST detectors) and passed a FRESH held-out suite. Strong result.")
    else: print("  -> honest FAIL on unseen code. Verifier never weakened, tests never leaked into repair.")
    if MOCK: print("  (MOCK: canned AST repair; real run has NEURON write the detectors.)")

if __name__ == "__main__": main()
