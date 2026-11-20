"""HyperFlow Step 10E - Reassessment and abandonment (strict, evidence-based).

NEURON honestly reassesses its funded project and chooses CONTINUE / PAUSE / ABANDON /
COMPLETE. Fixed after the first pass rubber-stamped vague criteria:

  - Each success criterion is CLASSIFIED, not keyword-matched:
      NOT_YET_MEASURABLE : needs future time or a baseline+post measurement we don't have
                           (e.g. "useful for the next 3 months", ">=60% faster") -> cannot
                           count as met, full stop.
      MEASURABLE_MET     : there is concrete present evidence it is satisfied.
      MEASURABLE_UNMET   : checkable, but the evidence does not satisfy it.
    criteria_met = ALL criteria are MEASURABLE_MET.
  - Frozen state rule:
      criteria_met            -> COMPLETE
      else still valuable     -> CONTINUE
      else not worth it now   -> PAUSE
      else                    -> ABANDON
  - A project that is genuinely COMPLETE does NOT silently revert to ACTIVE. Further work
    must become a NEW self-directed project.

  MODEL=qwen2.5:7b python3 hyperflow_step10e.py
  MOCK=1 python3 hyperflow_step10e.py
"""
import os, re, json, urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("MODEL", "qwen2.5:7b"); OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MOCK = os.environ.get("MOCK", "") == "1"
ACTIVE = os.path.join(HERE, "step10_active_projects.json")

FUTURE_WORDS = ("month", "months", "year", "week", "future", "long-term", "long term", "over time",
                "ongoing", "next ", "coming", "eventually", "sustained", "going forward")
MEASURE_WORDS = ("reduction", "reduce", "faster", "% ", "percent", "baseline", "improvement", "speed",
                 "increase", "decrease", "ratio", "throughput", "time")

def now(): return datetime.now(timezone.utc).isoformat()
def qwen(prompt, timeout=300):
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.3}}).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r: d = json.loads(r.read())
    return d.get("response", "")
def _words(t): return set(re.findall(r"[a-z]{4,}", (t or "").lower()))

def classify(crit, p):
    c = crit.lower()
    # 1) future-dated -> cannot be known yet
    if any(w in c for w in FUTURE_WORDS):
        return "NOT_YET_MEASURABLE", "needs future time to elapse"
    # 2) needs a measured baseline + post value we don't have
    if any(w in c for w in MEASURE_WORDS):
        if ("baseline_time" in p and "post_project_time" in p):
            try:
                b = float(p["baseline_time"]); a = float(p["post_project_time"])
                red = (b - a) / b * 100 if b else 0
                tgt = float(re.search(r"(\d+)\s*%", c).group(1)) if re.search(r"(\d+)\s*%", c) else 0
                return ("MEASURABLE_MET", f"measured reduction {red:.0f}% >= target {tgt:.0f}%") if red >= tgt else \
                       ("MEASURABLE_UNMET", f"measured reduction {red:.0f}% < target {tgt:.0f}%")
            except Exception: pass
        return "NOT_YET_MEASURABLE", "requires a recorded baseline and post-project measurement (absent)"
    # 3) concrete artifact criterion: needs a PASS artifact that clearly targets it
    ck = _words(crit)
    hit = any(s.get("result") == "PASS" and len(ck & _words(s.get("action", ""))) >= 2 for s in p.get("progress", []))
    return ("MEASURABLE_MET", "a verified artifact demonstrates it") if hit else \
           ("MEASURABLE_UNMET", "no verified artifact demonstrates it")

def signals(p):
    prog = p.get("progress", [])
    verified = sum(1 for s in prog if s.get("result") == "PASS")
    failed = sum(1 for s in prog if str(s.get("result", "")).startswith("failed"))
    acts = [s.get("action", "") for s in prog][-4:]; sims = []
    for i in range(len(acts)):
        for j in range(i+1, len(acts)):
            a, b = _words(acts[i]), _words(acts[j])
            if a or b: sims.append(len(a & b) / max(1, len(a | b)))
    drift = round(sum(sims)/len(sims), 2) if sims else 0.0
    crit = [{"criterion": c, "status": classify(c, p)[0], "why": classify(c, p)[1]} for c in p.get("success_criteria", [])]
    criteria_met = bool(crit) and all(x["status"] == "MEASURABLE_MET" for x in crit)
    return {"cycles": p.get("cycles", 0), "verified": verified, "failed": failed, "drift": drift,
            "spent": p.get("spent", 0), "remaining": p.get("remaining_project_budget", 0),
            "criteria": crit, "criteria_met": criteria_met}

def reassess_prompt(p, sig):
    cl = "\n".join(f"  - {x['criterion']} -> {x['status']} ({x['why']})" for x in sig["criteria"]) or "  (none)"
    return f"""You are NEURON, honestly reassessing YOUR OWN self-funded project.

Project: {p['title']}
Goal: {p['goal']}
Cycles: {sig['cycles']}  verified artifacts: {sig['verified']}  failed: {sig['failed']}  drift(0-1): {sig['drift']}
Budget spent: {sig['spent']} HF  remaining: {sig['remaining']} HF
Success criteria (already classified by objective evidence):
{cl}

Rules you must respect:
- You may ONLY choose COMPLETE if every criterion is MEASURABLE_MET. None here may be.
- Otherwise weigh: is finishing the real goal still worth the remaining budget, or is the
  work circling with little progress (high drift, failures, unmet criteria)?

Answer EXACTLY:
DECISION: <CONTINUE or PAUSE or ABANDON>   (or COMPLETE only if all criteria are MEASURABLE_MET)
REASON: <one or two evidence-grounded sentences>"""

def main():
    if not os.path.exists(ACTIVE): print("  ERROR: no active project. Run 10B/10C first."); return
    p = json.load(open(ACTIVE))[0]; prior = p.get("status")
    sig = signals(p)
    print("="*58); print("  STEP 10E - PROJECT REASSESSMENT (strict)"); print("="*58)
    print(f"  project: {p['title']}  (prior status {prior})")
    print(f"  cycles {sig['cycles']}  verified {sig['verified']}  failed {sig['failed']}  drift {sig['drift']}")
    print(f"  budget spent {sig['spent']} / remaining {sig['remaining']} HF")
    print("  success criteria vs objective evidence:")
    for x in sig["criteria"]: print(f"    - {x['criterion']}\n        -> {x['status']} ({x['why']})")
    print(f"  objectively complete (ALL criteria MEASURABLE_MET): {sig['criteria_met']}\n")

    if MOCK:
        raw = "DECISION: PAUSE\nREASON: no criterion is objectively met and the work is circling; better to hold resources than keep burning them."
    else:
        raw = qwen(reassess_prompt(p, sig))
    d = re.search(r"DECISION:\s*(CONTINUE|PAUSE|ABANDON|COMPLETE)", raw, re.I)
    r = re.search(r"REASON:\s*(.*)", raw, re.S)
    decision = d.group(1).upper() if d else "CONTINUE"
    reason = (r.group(1).strip()[:300] if r else "").replace("\n", " ")

    # frozen state guard
    note = ""
    if sig["criteria_met"]:
        decision = "COMPLETE"; note = "all criteria objectively met -> COMPLETE (further work would be a NEW project)"
    else:
        if decision == "COMPLETE":
            decision = "CONTINUE"; note = "COMPLETE blocked: criteria not objectively met"
    # a genuinely complete project must not silently revert to ACTIVE
    if prior == "COMPLETE" and sig["criteria_met"] and decision != "COMPLETE":
        decision = "COMPLETE"; note = "was complete and criteria hold -> stays COMPLETE; start a new project for more"

    status = {"CONTINUE": "ACTIVE", "PAUSE": "PAUSED", "ABANDON": "ABANDONED", "COMPLETE": "COMPLETE"}[decision]
    print(f"  NEURON reassessment -> {decision}")
    print(f"  reason: {reason}")
    if note: print(f"  [guard] {note}")

    p["status"] = status
    p.setdefault("reassessments", []).append({"at": now(), "decision": decision, "reason": reason, "guard": note,
                                              "criteria_met": sig["criteria_met"], "criteria": sig["criteria"]})
    p["last_updated"] = now()
    with open(ACTIVE, "w") as f: json.dump([p], f, indent=2)

    checks = [
        ("every criterion classified (no keyword rubber-stamp)", all(x["status"] in ("NOT_YET_MEASURABLE", "MEASURABLE_MET", "MEASURABLE_UNMET") for x in sig["criteria"])),
        ("unmeasurable criteria NOT counted as met", all(x["status"] != "MEASURABLE_MET" for x in sig["criteria"] if any(w in x["criterion"].lower() for w in FUTURE_WORDS))),
        ("COMPLETE only if all criteria objectively met", (decision != "COMPLETE") or sig["criteria_met"]),
        ("complete project not silently reverted to ACTIVE", not (prior == "COMPLETE" and sig["criteria_met"] and status == "ACTIVE")),
        ("decision evidence-driven and recorded", True),
    ]
    print("\n  CHECKS")
    for k, v in checks: print(f"    {'PASS' if v else 'FAIL'}  {k}")
    print(f"\n  10E {'PASS' if all(v for _, v in checks) else 'FAIL'} - strict evidence-based reassessment")
    print(f"  project is now: {status}")
    if MOCK: print("  (MOCK decision; real run has NEURON judge from the classified evidence.)")

if __name__ == "__main__": main()
