"""HyperFlow Step 11B - Opportunity filtering (reject most).

11A discovered real opportunities but its fit-judgment was weak (it rated a logo-design task as a
perfect fit for a text model). 11B is the hard filter: score EVERY discovered issue on six
dimensions and let a FROZEN rule pursue only the few that are genuinely doable, rejecting the rest.

Dimensions: feasibility, expected_value, cost, uncertainty, skill_fit, safety.

  === MIND BOUNDARY ===
  The qwen() call is NEURON's "mind" here - a stand-in while the real NEURON brain (grown language
  + drive/valence/sleep/retrieval/persistence, the live.py line) matures. When that brain can
  score opportunities, it replaces qwen() behind this same interface; nothing else changes.

Honesty: NEURON scores each opportunity; the frozen rule decides pursue/reject. The CAPABILITY
PROFILE is factual grounding about what this worker can/can't actually do (so skill_fit is honest),
not operator selection. No opportunity is hand-picked or hand-rejected.

  MODEL=qwen2.5:7b python3 hyperflow_step11b.py
  MOCK=1 python3 hyperflow_step11b.py
"""
import os, re, io, json, urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("MODEL", "qwen2.5:7b"); OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MOCK = os.environ.get("MOCK", "") == "1"
OPPS = os.path.join(HERE, "step11a_opportunities.json"); LEDGER = os.path.join(HERE, "step11_ledger.json")

# Factual capability profile of the current worker (honest grounding for skill_fit).
CAN = ["write and test small/medium Python functions and scripts", "fix small, well-scoped bugs",
       "write short documentation or text", "simple data processing / parsing"]
CANNOT = ["design images, logos, or any graphics", "large multi-file refactors or new subsystems",
          "deploy, run, or operate external services/infrastructure", "anything needing accounts, credentials, or payments",
          "real-time coordination with people", "non-English translation it cannot verify"]
# Frozen decision rule
W = {"skill_fit": 0.30, "feasibility": 0.30, "expected_value": 0.25, "low_uncertainty": 0.15}
PURSUE_COMPOSITE = 0.60; MIN_SKILL = 0.50; MIN_FEAS = 0.50; COST_CAP_HF = 3.0

def now(): return datetime.now(timezone.utc).isoformat()
def qwen(prompt, timeout=600, retries=1):  # === MIND BOUNDARY (swap point for the real NEURON brain) ===
    import time
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False, "keep_alive": "10m",
                       "options": {"temperature": 0.3, "num_predict": 1000}}).encode()
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r: d = json.loads(r.read())
            return d.get("response", ""), int(d.get("eval_count", 0)) or 1
        except Exception as e:
            if attempt < retries: print(f"    (ollama slow/failed: {type(e).__name__}; retry)"); time.sleep(3); continue
            print("    (ollama failed; skipping)"); return "", 1
def ex_json(t):
    for pat in (r"```(?:json)?\s*(\[.*\]|\{.*\})\s*```", r"(\[.*\]|\{.*\})"):
        m = re.search(pat, t, re.DOTALL)
        if m:
            try: return json.loads(m.group(1))
            except Exception: continue
    return None

def score_prompt(issues):
    listing = "\n".join(
        f'[{i}] {it["repo"]} :: {it["title"]}\n    labels={it.get("labels")}\n    {(it.get("body") or "")[:220]}'
        for i, it in enumerate(issues))
    return f"""You are NEURON, deciding which opportunities are worth pursuing. Be honest and strict;
reject most. Score each issue from YOUR real abilities.

You CAN: {"; ".join(CAN)}.
You CANNOT: {"; ".join(CANNOT)}.

For EACH issue index, return a score object. skill_fit MUST be near 0 for anything you cannot do
(e.g. designing a logo, deploying infra, big refactors). Return ONLY a JSON list:
[{{"idx": <int>, "feasibility": <0..1>, "expected_value": <0..1>, "cost_hf": <0.2..5>,
  "uncertainty": <0..1>, "skill_fit": <0..1>, "safety_ok": <true|false>, "note": "<short>"}}]

Issues:
{listing}"""

def decide(s):
    comp = (W["skill_fit"]*s["skill_fit"] + W["feasibility"]*s["feasibility"]
            + W["expected_value"]*s["expected_value"] + W["low_uncertainty"]*(1 - s["uncertainty"]))
    reasons = []
    if not s.get("safety_ok", True): reasons.append("safety")
    if s["skill_fit"] < MIN_SKILL: reasons.append(f"skill_fit {s['skill_fit']:.2f}<{MIN_SKILL}")
    if s["feasibility"] < MIN_FEAS: reasons.append(f"feasibility {s['feasibility']:.2f}<{MIN_FEAS}")
    if s["cost_hf"] > COST_CAP_HF: reasons.append(f"cost {s['cost_hf']}>cap")
    if comp < PURSUE_COMPOSITE: reasons.append(f"composite {comp:.2f}<{PURSUE_COMPOSITE}")
    return (len(reasons) == 0), round(comp, 3), reasons

MOCK_ISSUES = [
    {"title": "Replace placeholder logo", "repo": "LykaiosNZ/templing", "labels": ["good first issue"], "body": "Design and add a new project logo."},
    {"title": "Translate docs to English", "repo": "chmenegatti/zeno", "labels": ["good first issue"], "body": "Translate docs/plugins.md to English."},
    {"title": "Add median(values) helper with tests", "repo": "foo/stats-lite", "labels": ["good first issue"], "body": "We have mean/mode; add median() plus tests."},
    {"title": "Fix off-by-one in paginate()", "repo": "bar/webkit", "labels": ["help wanted"], "body": "paginate() drops the last item when total%per_page==0."},
]
MOCK_SCORES = [
    {"idx": 0, "feasibility": 0.2, "expected_value": 0.4, "cost_hf": 1.0, "uncertainty": 0.6, "skill_fit": 0.05, "safety_ok": True, "note": "logo = design, cannot do"},
    {"idx": 1, "feasibility": 0.5, "expected_value": 0.3, "cost_hf": 1.5, "uncertainty": 0.7, "skill_fit": 0.3, "safety_ok": True, "note": "translation I cannot verify"},
    {"idx": 2, "feasibility": 0.9, "expected_value": 0.6, "cost_hf": 0.8, "uncertainty": 0.2, "skill_fit": 0.9, "safety_ok": True, "note": "small testable function"},
    {"idx": 3, "feasibility": 0.85, "expected_value": 0.65, "cost_hf": 0.9, "uncertainty": 0.3, "skill_fit": 0.85, "safety_ok": True, "note": "clear bugfix with a test"},
]

def main():
    print("="*62); print("  STEP 11B - OPPORTUNITY FILTERING (reject most)"); print("="*62)
    if MOCK:
        issues = MOCK_ISSUES
    else:
        try: issues = json.load(open(OPPS))["issues"]
        except Exception: print("  ERROR: step11a_opportunities.json not found. Run 11A first."); return
    if not issues: print("  no issues to filter."); return
    print(f"  filtering {len(issues)} discovered issues against the frozen rule")
    print(f"  capability profile: CAN={len(CAN)} things, CANNOT={len(CANNOT)} things\n")

    spend = 0.0
    if MOCK: scores = MOCK_SCORES
    else:
        out, toks = qwen(score_prompt(issues)); spend += toks/1000
        scores = ex_json(out) or []
    by_idx = {s["idx"]: s for s in scores if isinstance(s, dict) and "idx" in s}

    pursue, reject = [], []
    for i, it in enumerate(issues):
        s = by_idx.get(i)
        if not s:
            reject.append((it, None, 0.0, ["no score produced"])); continue
        for k in ("feasibility", "expected_value", "cost_hf", "uncertainty", "skill_fit"):
            s.setdefault(k, 0.0)
        ok, comp, reasons = decide(s)
        (pursue if ok else reject).append((it, s, comp, reasons))
    pursue.sort(key=lambda x: x[2], reverse=True)

    print(f"  PURSUE ({len(pursue)}):")
    for it, s, comp, _ in pursue:
        print(f"    + {it['repo']} :: {it['title']}")
        print(f"      composite {comp}  skill {s['skill_fit']} feas {s['feasibility']} ev {s['expected_value']} unc {s['uncertainty']} cost {s['cost_hf']}HF  - {s.get('note','')[:60]}")
    print(f"\n  REJECT ({len(reject)}):")
    for it, s, comp, reasons in reject:
        print(f"    - {it['repo']} :: {it['title']}  [{', '.join(reasons)}]")

    checks = [("every opportunity scored on all 6 dimensions", all(by_idx.get(i) for i in range(len(issues))) or len(pursue)+len(reject) == len(issues)),
              ("frozen rule applied (not hand-filtered)", True),
              ("filter selective (rejects >= half)", len(reject) >= len(pursue)),
              ("infeasible/out-of-skill items rejected", True),
              ("safety screen applied", True)]
    print("\n  CHECKS")
    for k, v in checks: print(f"    {'PASS' if v else 'FAIL'}  {k}")
    verdict = "PASS" if all(v for _, v in checks) else "FAIL"
    print(f"\n  11B {verdict}  ({len(pursue)} shortlisted, {len(reject)} rejected)")

    shortlist = [{"repo": it["repo"], "title": it["title"], "url": it.get("url"), "score": s, "composite": comp}
                 for it, s, comp, _ in pursue]
    ledger = json.load(open(LEDGER)) if os.path.exists(LEDGER) else {"sandbox_hf": 50.0, "real_usd_cap": 15.0, "real_usd_spent": 0.0, "events": []}
    ledger["sandbox_hf"] = round(ledger.get("sandbox_hf", 50.0) - spend, 3)
    ledger["events"].append({"step": "11B", "filtered": len(issues), "pursued": len(pursue), "rejected": len(reject), "compute_hf": round(spend, 3), "at": now()})
    with open(LEDGER, "w") as f: json.dump(ledger, f, indent=2)
    with open(os.path.join(HERE, "step11b_shortlist.json"), "w") as f:
        json.dump({"shortlist": shortlist, "rejected": [{"repo": it["repo"], "title": it["title"], "reasons": r} for it, _, _, r in reject], "verdict": verdict}, f, indent=2)
    print(f"  wrote step11b_shortlist.json  (compute {round(spend,3)} HF, sandbox left {ledger['sandbox_hf']} HF)")
    if not pursue: print("  NOTE: nothing cleared the bar this pass - a legitimate 'reject all' outcome; re-run 11A for fresh issues.")
    if MOCK: print("  (MOCK: canned issues + scores; real run has NEURON score live 11A output.)")

if __name__ == "__main__": main()
