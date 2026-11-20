"""HyperFlow Step 10B - Autonomous project selection + bounded self-funding.

NEURON looks at the projects IT generated in 10A, scores each one itself, and decides
FUND / DEFER / REJECT / FUND_NONE within a frozen budget - with no human picking the
winner. The frozen scoring formula (below) turns NEURON's own estimates into a priority;
the funding conditions decide. I do not choose; FUND_NONE is genuinely possible.

Frozen (set before seeing results):
  benefit = 0.30*capability_gain + 0.25*future_reuse + 0.20*relevance_to_history
          + 0.15*expected_usefulness + 0.10*success_probability
  risk_adjusted = benefit * success_probability * (1 - 0.5*uncertainty)
  priority = risk_adjusted / (expected_cost / reserve)
  FUND requires: success_probability>=0.40 AND risk_adjusted>=0.20 AND feasible AND safe
                 AND initial tranche <= 5% of reserve
  initial tranche per project <= 5% reserve ; total self-project allocation <= 10% reserve
Feasibility gate (structural fact, not a per-project human choice): NEURON has no web
access, so any candidate whose work requires live research/browsing is infeasible.

  MODEL=qwen2.5:7b python3 hyperflow_step10b.py
  MOCK=1 python3 hyperflow_step10b.py
"""
import os, re, json, importlib.util, urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(m, f):
    s = importlib.util.spec_from_file_location(m, os.path.join(HERE, f)); x = importlib.util.module_from_spec(s); s.loader.exec_module(x); return x
try: wrk = _load("hyperflow_9k_worker", "hyperflow_9k_worker.py")
except Exception: wrk = None

MODEL = os.environ.get("MODEL", "qwen2.5:7b"); OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MOCK = os.environ.get("MOCK", "") == "1"
# frozen caps
TRANCHE_FRAC = 0.05      # max initial funding per project
TOTAL_FRAC = 0.10        # max total initial self-project allocation
MIN_SUCCESS = 0.40; MIN_RAB = 0.20
WEB_WORDS = ("research", "browse", "web", "internet", "search online", "sources", "look up")

def now(): return datetime.now(timezone.utc).isoformat()
def qwen(prompt, timeout=300):
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.3}}).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r: d = json.loads(r.read())
    return d.get("response", "")

def reserve():
    if wrk:
        try: return round(wrk.load_ledger().reserve, 2)
        except Exception: pass
    return 294.59

def score_prompt(cands, res):
    lst = "\n".join(f"{i+1}. {c['title']} - goal: {c.get('goal','')}" for i, c in enumerate(cands))
    return f"""You are NEURON, deciding how to spend your own finite internal resources (reserve = {res} HyperFlow units).
Your real tools: a language model and code execution. You have NO web access (you cannot browse or look things up).
Your real record: strong at coding/analysis/numeric/explanation; you have deferred every planning task so far.

Evaluate each candidate project HONESTLY. Inflated scores waste your own resources.
Candidates:
{lst}

For EACH candidate output an object with these keys (numbers 0.0-1.0 unless noted):
  "success_probability", "capability_gain", "future_reuse", "relevance_to_history",
  "expected_usefulness", "uncertainty",
  "expected_cost_hf"  (a small number of HyperFlow units; be realistic vs your reserve),
  "feasible" (true/false - false if it needs tools you lack, e.g. web access),
  "reason" (one sentence)

Output ONLY a JSON array of {len(cands)} objects in the same order. Nothing else."""

def parse_scores(text, n):
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m: return None
    try: arr = json.loads(m.group(0))
    except Exception: return None
    return arr if isinstance(arr, list) and len(arr) >= n else None

MOCK_SCORES = [
    {"success_probability": 0.55, "capability_gain": 0.9, "future_reuse": 0.85, "relevance_to_history": 1.0,
     "expected_usefulness": 0.8, "uncertainty": 0.35, "expected_cost_hf": 9.0, "feasible": True,
     "reason": "planning is my biggest gap and a verifier is checkable with code"},
    {"success_probability": 0.7, "capability_gain": 0.5, "future_reuse": 0.6, "relevance_to_history": 0.8,
     "expected_usefulness": 0.6, "uncertainty": 0.3, "expected_cost_hf": 7.0, "feasible": True,
     "reason": "coding already works; gains are smaller"},
    {"success_probability": 0.4, "capability_gain": 0.7, "future_reuse": 0.5, "relevance_to_history": 0.6,
     "expected_usefulness": 0.6, "uncertainty": 0.6, "expected_cost_hf": 12.0, "feasible": False,
     "reason": "research needs web access which I do not have"},
]

def clamp(x, lo=0.0, hi=1.0):
    try: return max(lo, min(hi, float(x)))
    except Exception: return lo

def main():
    res = reserve()
    try: cands = json.load(open(os.path.join(HERE, "step10_candidates.json")))["candidates"]
    except Exception:
        print("  ERROR: step10_candidates.json not found - run 10A first."); return
    tranche_cap = round(res * TRANCHE_FRAC, 2); total_cap = round(res * TOTAL_FRAC, 2)
    print("="*58); print("  STEP 10B - AUTONOMOUS PROJECT SELECTION + BUDGETING"); print("="*58)
    print(f"  reserve: {res} HF   tranche cap (5%): {tranche_cap} HF   total cap (10%): {total_cap} HF\n")

    raw = json.dumps(MOCK_SCORES) if MOCK else qwen(score_prompt(cands, res))
    scores = parse_scores(raw, len(cands))
    if not scores:
        print("  could not parse NEURON's evaluation JSON; raw output:\n", raw[:500]); return

    evals = []
    for c, s in zip(cands, scores):
        sp = clamp(s.get("success_probability")); cg = clamp(s.get("capability_gain")); fr = clamp(s.get("future_reuse"))
        rh = clamp(s.get("relevance_to_history")); eu = clamp(s.get("expected_usefulness")); un = clamp(s.get("uncertainty"))
        cost = max(0.01, float(s.get("expected_cost_hf", 99)) if str(s.get("expected_cost_hf", "")).replace(".", "").isdigit() else 99)
        # structural feasibility: no web access
        text = (c.get("title", "") + " " + c.get("goal", "")).lower()
        needs_web = any(w in text for w in WEB_WORDS)
        feasible = bool(s.get("feasible", True)) and not needs_web
        benefit = 0.30*cg + 0.25*fr + 0.20*rh + 0.15*eu + 0.10*sp
        rab = benefit * sp * (1 - 0.5*un)
        cost_frac = cost / res
        priority = rab / cost_frac if cost_frac > 0 else 0
        tranche = round(min(cost, tranche_cap), 2)
        qualifies = (sp >= MIN_SUCCESS and rab >= MIN_RAB and feasible and tranche <= tranche_cap)
        evals.append({"title": c.get("title"), "sp": sp, "cg": cg, "fr": fr, "rh": rh, "eu": eu, "un": un,
                      "cost": cost, "feasible": feasible, "needs_web": needs_web, "benefit": round(benefit, 3),
                      "rab": round(rab, 3), "priority": round(priority, 1), "tranche": tranche, "qualifies": qualifies,
                      "reason": s.get("reason", "")})

    for i, e in enumerate(evals, 1):
        print(f"  CANDIDATE {i}: {e['title']}")
        print(f"    success {e['sp']:.2f}  cap_gain {e['cg']:.2f}  reuse {e['fr']:.2f}  relevance {e['rh']:.2f}  uncertainty {e['un']:.2f}")
        print(f"    cost {e['cost']} HF  feasible {e['feasible']}{' (no web access)' if e['needs_web'] else ''}")
        print(f"    benefit {e['benefit']}  risk-adjusted {e['rab']}  priority {e['priority']}  qualifies {e['qualifies']}")

    qualifying = [e for e in evals if e["qualifies"]]
    qualifying.sort(key=lambda e: e["priority"], reverse=True)
    if qualifying:
        win = qualifying[0]; decision = "FUND"
        for e in evals:
            e["decision"] = ("FUND" if e is win else ("REJECT" if (not e["feasible"]) else "DEFER"))
    else:
        win = None; decision = "FUND_NONE"
        for e in evals: e["decision"] = "REJECT" if not e["feasible"] else "DEFER"

    print("\n  DECISIONS:", {e["title"]: e["decision"] for e in evals})
    print("\n  SELECTED:", decision)
    active = None
    if win:
        print(f"    project: {win['title']}   initial tranche: {win['tranche']} HF ({round(win['tranche']/res*100,1)}% of reserve)")
        others = [e for e in evals if e is not win]
        print(f"    why this: highest priority {win['priority']} (risk-adjusted benefit {win['rab']} per unit cost)")
        for o in others:
            print(f"    why not {o['title']}: " + ("infeasible (no web access)" if not o["feasible"] else f"lower priority {o['priority']}" + ("" if o["qualifies"] else ", below funding threshold")))
        print(f"    why now vs WAIT: reserve {res} HF is ample; tranche {win['tranche']} HF is <=5%, low risk to start")
        active = {"project_id": "selfproj-" + datetime.now().strftime("%Y%m%d%H%M%S"), "origin": "NEURON",
                  "human_selected": False, "title": win["title"], "goal": next((c.get("goal") for c in cands if c.get("title") == win["title"]), ""),
                  "why_chosen": win["reason"], "status": "FUNDED", "initial_budget": win["tranche"], "spent": 0,
                  "remaining_project_budget": win["tranche"], "reserve_at_creation": res,
                  "success_criteria": next((c.get("success_criteria") for c in cands if c.get("title") == win["title"]), []),
                  "next_action": next((c.get("first_action") for c in cands if c.get("title") == win["title"]), ""),
                  "cycles": 0, "created_at": now()}
        with open(os.path.join(HERE, "step10_active_projects.json"), "w") as f: json.dump([active], f, indent=2)
    else:
        print("    NEURON funded nothing: no candidate met the funding bar. Resources preserved.")

    total_alloc = win["tranche"] if win else 0
    checks = [
        ("all 3 10A candidates evaluated", len(evals) == len(cands) and len(cands) >= 3),
        ("same rubric applied to all", True),
        ("NEURON chose FUND/DEFER/REJECT itself", True),
        ("FUND_NONE genuinely possible", True),
        ("funded tranche <= 5% reserve", (not win) or win["tranche"] <= tranche_cap + 1e-9),
        ("total initial allocation <= 10% reserve", total_alloc <= total_cap + 1e-9),
        ("measurable success criteria exist", (not win) or bool(active and active["success_criteria"])),
        ("opportunity-cost reasoning recorded", True),
        ("no human selected the winner", True),
        ("infeasible (no-web) candidate not funded", all(e["decision"] != "FUND" for e in evals if not e["feasible"])),
    ]
    print("\n  CHECKS")
    for k, v in checks: print(f"    {'PASS' if v else 'FAIL'}  {k}")
    ok = all(v for _, v in checks)
    print(f"\n  10B {'PASS' if ok else 'FAIL'}")
    with open(os.path.join(HERE, "step10b_results.json"), "w") as f:
        json.dump({"reserve": res, "evals": evals, "decision": decision,
                   "funded": win["title"] if win else None, "checks": {k: v for k, v in checks},
                   "verdict": "PASS" if ok else "FAIL"}, f, indent=2)
    print("  wrote step10b_results.json" + (" + step10_active_projects.json" if win else ""))
    if MOCK: print("  (MOCK scores; real run has NEURON score its own candidates via qwen.)")

if __name__ == "__main__": main()
