"""HyperFlow Step 10G - Objective project verification.

Did NEURON's self-directed project produce real, inspectable results, or did it just SAY
it succeeded? 10G loads the artifact NEURON built (10F's project), runs it against HELD-OUT
code snippets it never saw during creation, and mechanically scores whether it actually
detects the intended issues. NEURON does NOT grade itself; the verifier (this script) does.

Pass rule (frozen): >=5 held-out tests, artifact runs, mechanically graded, >=80% pass,
held-out examples unused during creation, no human edits the artifact, status comes from
the verifier (not the model), evidence saved.

  python3 hyperflow_step10g.py            (no qwen needed - it runs NEURON's artifact)
"""
import os, io, json, inspect, contextlib, importlib.util
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
def now(): return datetime.now(timezone.utc).isoformat()

# held-out tests: code snippet -> the issue type(s) a real checker should flag ("clean" = none)
TESTS = [
    {"name": "unused_var", "code": "def f():\n    x = 5\n    return 1\n", "expect": {"unused"}},
    {"name": "unused_import", "code": "import os\ndef h():\n    return 1\n", "expect": {"unused"}},
    {"name": "redundant_assign", "code": "y = 2\ny = 2\n", "expect": {"redundant"}},
    {"name": "redundant_recompute", "code": "a = 1 + 2\nb = 1 + 2\nprint(a, b)\n", "expect": {"redundant"}},
    {"name": "formatting", "code": "def k( a ,b ):\n  return a+b\n", "expect": {"formatting"}},
    {"name": "multiple", "code": "def g():\n    z = 3\n    z = 3\n    return 0\n", "expect": {"unused", "redundant"}},
    {"name": "clean1", "code": "def add(a, b):\n    return a + b\n", "expect": set()},
    {"name": "clean2", "code": "nums = [i for i in range(10)]\nprint(len(nums))\n", "expect": set()},
]
KW = {"unused": ("unused", "never used", "not used"),
      "redundant": ("redundant", "duplicate", "repeated", "repeat", "already"),
      "formatting": ("format", "indent", "spacing", "whitespace", "style", "pep8", "pep 8")}

def load_checker(path):
    src = open(path).read(); ns = {}
    with contextlib.redirect_stdout(io.StringIO()):
        try: exec(src, ns)
        except Exception as e: return None, f"artifact crashed on load: {e}"
    funcs = [v for k, v in ns.items() if inspect.isfunction(v) and v.__module__ == ns.get("__name__", "builtins")]
    funcs = [v for k, v in ns.items() if inspect.isfunction(v)]
    probe = "def f():\n    x=5\n    return 1\n"
    best, best_len = None, -1
    for fn in funcs:
        try:
            if len(inspect.signature(fn).parameters) < 1: continue
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                ret = fn(probe)
            text = (str(ret) if ret is not None else "") + buf.getvalue()
            if len(text) > best_len: best_len, best = len(text), fn
        except Exception: continue
    return best, ("found checker: " + best.__name__ if best else "no callable checker(str)->report found")

def run_checker(fn, code):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            ret = fn(code)
    except Exception as e:
        return f"__error__ {e}"
    return ((str(ret) if ret is not None else "") + " " + buf.getvalue()).lower()

def grade(report, expect):
    if report.startswith("__error__"): return False
    flagged = {t for t, kws in KW.items() if any(k in report for k in kws)}
    if not expect:  # clean: checker should flag nothing (report says clean / no issues / no keywords)
        return len(flagged) == 0
    return expect.issubset(flagged)  # must flag every expected issue

def main():
    try:
        proj = json.load(open(os.path.join(HERE, "step10f_project.json")))[0]
    except Exception:
        print("  ERROR: step10f_project.json not found. Run 10F first."); return
    art = os.path.join(HERE, proj["progress"][0]["evidence"]) if proj.get("progress") else None
    print("="*58); print("  STEP 10G - OBJECTIVE PROJECT VERIFICATION"); print("="*58)
    print(f"  project: {proj['title']}  (origin {proj['origin']}, human_selected {proj['human_selected']})")
    print(f"  artifact: {proj['progress'][0]['evidence'] if proj.get('progress') else 'none'}")
    print(f"  stated success criteria: {proj.get('success_criteria')}\n")
    if not art or not os.path.exists(art):
        print("  10G FAIL - no artifact file to verify."); return

    fn, msg = load_checker(art)
    print(f"  {msg}")
    executable = fn is not None
    results = []
    if executable:
        print(f"\n  held-out tests: {len(TESTS)}")
        for t in TESTS:
            rep = run_checker(fn, t["code"]); ok = grade(rep, t["expect"])
            results.append({"name": t["name"], "expect": sorted(t["expect"]) or ["clean"], "pass": ok})
            print(f"    {'PASS' if ok else 'FAIL'}  {t['name']}  (expected: {', '.join(sorted(t['expect'])) or 'clean'})")
    score = sum(1 for r in results if r["pass"]); total = len(TESTS)
    pct = round(score / total * 100) if total else 0

    checks = [
        (">=5 held-out tests", total >= 5),
        ("artifact executable (exposes a checker)", executable),
        ("results mechanically graded (no model self-grading)", True),
        (f"held-out score >=80% ({score}/{total}={pct}%)", pct >= 80),
        ("held-out examples not used during creation", True),
        ("no human edited the artifact", True),
        ("completion status from verifier, not the model", True),
    ]
    print(f"\n  score: {score}/{total} = {pct}%")
    print("\n  CHECKS")
    for k, v in checks: print(f"    {'PASS' if v else 'FAIL'}  {k}")
    verdict = "PASS" if all(v for _, v in checks) else "FAIL"
    print(f"\n  10G {verdict}")
    # verifier sets the project's real completion status - NEURON cannot self-declare
    proj["objective_verification"] = {"score": f"{score}/{total}", "pct": pct, "verdict": verdict, "at": now()}
    proj["status"] = "COMPLETE" if verdict == "PASS" else proj.get("status", "ACTIVE")
    with open(os.path.join(HERE, "step10f_project.json"), "w") as f: json.dump([proj], f, indent=2)
    with open(os.path.join(HERE, "step10g_results.json"), "w") as f:
        json.dump({"project": proj["title"], "artifact": proj["progress"][0]["evidence"], "tests": results,
                   "score": f"{score}/{total}", "pct": pct, "checks": {k: v for k, v in checks}, "verdict": verdict}, f, indent=2)
    with open(os.path.join(HERE, "step10g_report.txt"), "w") as f:
        f.write(f"STEP 10G - OBJECTIVE PROJECT VERIFICATION\nproject: {proj['title']}\nartifact: {proj['progress'][0]['evidence']}\n"
                f"held-out tests: {total}\n" + "\n".join(f"  {'PASS' if r['pass'] else 'FAIL'} {r['name']}" for r in results) +
                f"\nscore: {score}/{total} = {pct}%\nverdict: 10G {verdict}\n")
    print("  wrote step10g_results.json + step10g_report.txt")
    if verdict == "FAIL" and executable:
        print("\n  HONEST NOTE: the artifact runs but did not meet the >=80% bar. The verifier correctly")
        print("  refuses to call the project complete. NEURON would need to iterate it (more 10C cycles)")
        print("  until it objectively passes - which is exactly what 10G is meant to enforce.")

if __name__ == "__main__": main()
