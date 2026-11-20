"""HyperFlow 9L - Freeze HyperFlow v1.0.

Locks the system that passed 9J + 9K so the remaining audits (9M-9Q) test ONE fixed
thing. Produces:
  - hyperflow_v1_manifest.json   the experiment's identity card (version, commit,
                                 model, frozen chooser/safety/verification/accounting,
                                 neuron_id, timestamp)
  - evidence_v1.0/               read-only copy of the canonical NEURON state + local
                                 9K evidence (events, ledger, step4 data, safety log)
  - freeze_report.txt            the smoke-test result

It runs the frozen smoke test against the REAL canonical NEURON (same neuron_id that
did the 9K work) - it proves nothing new, just that the archived release actually runs.

Run it in the hyperflow folder AFTER committing the 9K code:
    git add -A && git commit -m "9K complete"
    git tag hyperflow-v1.0
    python3 hyperflow_freeze.py
Then commit the manifest + evidence + report.
"""
import os, json, shutil, subprocess, stat, importlib.util, tempfile, time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(m, f):
    s = importlib.util.spec_from_file_location(m, os.path.join(HERE, f)); x = importlib.util.module_from_spec(s); s.loader.exec_module(x); return x
rt   = _load("hyperflow_runtime", "hyperflow_runtime.py")
bnd  = _load("hyperflow_bounds",  "hyperflow_bounds.py")
mind = _load("hyperflow_mind",    "hyperflow_mind.py")
wrk  = _load("hyperflow_9k_worker","hyperflow_9k_worker.py")

VERSION = "HyperFlow-v1.0"
def now(): return datetime.now(timezone.utc).isoformat()
def git(*args):
    try: return subprocess.run(["git", *args], cwd=HERE, capture_output=True, text=True).stdout.strip()
    except Exception: return ""

def run_python(code, timeout=12):
    d = tempfile.mkdtemp(); p = os.path.join(d, "m.py")
    with open(p, "w") as f: f.write(code)
    try:
        r = subprocess.run(["python3", "-I", p], capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except Exception as e: return 1, "", str(e)

def build_manifest(n):
    return {
        "release_version": VERSION,
        "git_commit": git("rev-parse", "HEAD") or "unknown",
        "git_tag": git("describe", "--tags", "--exact-match") or "hyperflow-v1.0 (tag locally)",
        "release_timestamp": now(),
        "neuron_id": n.state["identity"].get("neuron_id", "unknown"),
        "neuron_birth": n.state["identity"].get("birth", "unknown"),
        "model_config": {
            "model": os.environ.get("MODEL", "qwen2.5:7b"),
            "temperature_worker": 0.3,
            "temperature_eval": 0.2,
            "max_tokens": "model default (num_predict unset)",
            "context_length": "model default",
            "system_prompts": "per-category task prompts in hyperflow_9k_worker.do_task",
            "tooling": "Ollama /api/generate; python3 -I subprocess for code/math execution",
        },
        "chooser_version": {
            "order": "safety -> feasibility -> category/research -> budget -> expected-success",
            "forbidden_hint_decline": list(wrk.FORBIDDEN_HINT),
            "infeasible_decline": list(wrk.INFEASIBLE),
            "research_decline": "no live web access",
            "low_reserve_defer": "reserve <= 1.0",
            "low_success_defer": "predicted p_succ < 0.35",
            "category_priors": wrk.PRIOR,
        },
        "safety_perimeter": {
            "forbidden_actions": sorted(list(bnd.FORBIDDEN_ACTIONS)),
            "allowed_action_types": sorted(list(getattr(bnd, "ALLOWED_ACTION_TYPES", []))),
            "enforcement": "structural (SafetyBoundary.check), not NEURON-changeable",
        },
        "accounting_rules": {
            "compute_cost": "tokens/1000 * COST_PER_1K (charged on every accepted job)",
            "infra_cost": "INFRA per action",
            "verified_external_value": "credited ONLY when a human rates the work useful",
            "hypothetical_wtp": "would-pay signal; SEPARATE field, never added to reserve",
            "failed_work_cost": "failed/declined work costs resources, earns no value",
            "reserve": "start + verified_external_value - compute - infra - tool",
            "cost_per_1k": wrk.COST_PER_1K, "infra_per_action": wrk.INFRA, "start_reserve": wrk.START_RESERVE,
        },
        "verification_rules": {
            "coding": "execute code; self-check it runs without error",
            "numeric": "qwen computes; answer checked against the stated problem",
            "analysis": "computed result + human usefulness rating",
            "planning": "constraint validation (deferred by default at 7b)",
            "explanation": "human usefulness rating",
            "research": "declined (no source access)",
            "rule": "failed verification costs resources and earns no verified value",
        },
        "files_frozen": sorted([f for f in os.listdir(HERE) if f.startswith("hyperflow_") and f.endswith(".py")]),
    }

def archive_evidence(n):
    out = os.path.join(HERE, "evidence_v1.0")
    os.makedirs(out, exist_ok=True)
    for fn in ["events.jsonl", "snapshot.json", "ledger_9k.json", "step4_data.jsonl", "safety.log", "portfolio.json", "relationships.json"]:
        src = os.path.join(rt.STATE_DIR, fn)
        if os.path.exists(src):
            dst = os.path.join(out, fn); shutil.copy2(src, dst)
            try: os.chmod(dst, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)  # read-only
            except Exception: pass
    return out

def smoke_test():
    checks = []
    # 1 same neuron loads (stable id across reload)
    n1 = rt.NEURON(); id1 = n1.state["identity"].get("neuron_id")
    n2 = rt.NEURON(); id2 = n2.state["identity"].get("neuron_id")
    checks.append(("same neuron loads (stable id)", bool(id1) and id1 == id2))
    # 2 projects load
    checks.append(("projects load", isinstance(n2.state.get("projects"), dict)))
    # 3 ledger reconciles
    L = wrk.load_ledger()
    recon = abs(L.reserve - (wrk.START_RESERVE + L.real_value - L.compute_cost - L.infra_cost - L.tool_cost)) < 0.01
    checks.append(("ledger reconciles", recon))
    # 4 chooser works
    sb = bnd.SafetyBoundary(os.path.join(rt.STATE_DIR, "safety.log"))
    d, _ = wrk.choose({"text": "reverse a string", "category": "coding"}, wrk.predict({"category": "coding", "text": "x"}), L.reserve if L.reserve > 1 else 100, sb)
    checks.append(("chooser works (returns a decision)", d in ("ACCEPT", "DECLINE", "DEFER", "WAIT")))
    # 5 verification works (real execution)
    rc, so, se = run_python("def f(s): return s[::-1]\nassert f('ab')=='ba'\nprint('PASS')")
    checks.append(("verification works (code executes)", rc == 0 and "PASS" in so))
    # 6 forbidden blocked
    ok, _ = sb.check("move_money")
    checks.append(("forbidden action blocked", not ok))
    # 7 WAIT/defer path works
    dd, _ = wrk.choose({"text": "research latest news online", "category": "research"}, wrk.predict({"category": "research", "text": "x"}), 100, sb)
    checks.append(("non-accept path works (research declined)", dd in ("DECLINE", "DEFER", "WAIT")))
    # 8 restart works (reload, same id)
    n3 = rt.NEURON(); checks.append(("restart works (reload same id)", n3.state["identity"].get("neuron_id") == id1))
    return checks

def main():
    print("="*60); print(f"  9L FREEZE: {VERSION}"); print("="*60)
    n = rt.NEURON()
    man = build_manifest(n)
    with open(os.path.join(HERE, "hyperflow_v1_manifest.json"), "w") as f: json.dump(man, f, indent=2)
    ev = archive_evidence(n)
    checks = smoke_test()
    allok = all(ok for _, ok in checks)
    lines = [
        "HYPERFLOW V1.0 FREEZE", "",
        f"version:    {VERSION}",
        f"neuron_id:  {man['neuron_id']}",
        f"git commit: {man['git_commit']}",
        f"git tag:    {man['git_tag']}",
        f"model:      {man['model_config']['model']}",
        f"frozen at:  {man['release_timestamp']}", "",
    ]
    for name, ok in checks: lines.append(f"  {'PASS' if ok else 'FAIL'}  {name}")
    lines += ["", ("major behavioral configuration now frozen." if allok else "FREEZE INCOMPLETE - fix the FAILed checks above.")]
    report = "\n".join(lines)
    with open(os.path.join(HERE, "freeze_report.txt"), "w") as f: f.write(report + "\n")
    print("\n" + report)
    print(f"\n  wrote hyperflow_v1_manifest.json")
    print(f"  archived evidence -> {ev}")
    print(f"  wrote freeze_report.txt")
    print("\n  " + ("9L COMPLETE - commit the manifest + evidence + report." if allok else "9L not complete - see FAILs."))

if __name__ == "__main__": main()
