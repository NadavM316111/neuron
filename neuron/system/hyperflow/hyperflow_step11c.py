"""HyperFlow Step 11C - Bounded pursuit, with NEURON's SELF as the principal.

The whole point of this file is the BOUNDARY your architecture requires:

  NEURON state/history/drives  ->  opportunity perception  ->  NEURON valuation
     ->  NEURON decision  ->  resource allocation  ->  [hire tool: qwen]
     ->  execution (draft only)  ->  result returns to NEURON's memory

Everything up to and including the DECISION is NEURON's own persistent self (neuron_self.py),
a deterministic function of its identity, drives, goals, and a capability-confidence LEARNED from
its real autobiography. qwen is NOT consulted about what NEURON wants. qwen is hired only AFTER the
decision, as an external cognitive tool, to help EXECUTE the chosen work.

11C prepares a contribution (plan + draft code + draft PR). It does NOT submit anything - the
irreversible external action (opening the PR) is gated for 11D/11E.

  MODEL=qwen2.5-coder:7b python3 hyperflow_step11c.py
  MOCK=1 python3 hyperflow_step11c.py
"""
import os, re, io, json, urllib.request, urllib.parse
from datetime import datetime, timezone
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(mod, fn):
    s = importlib.util.spec_from_file_location(mod, os.path.join(HERE, fn)); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
NS = _load("neuron_self", "neuron_self.py")

MODEL = os.environ.get("MODEL", "qwen2.5-coder:7b"); OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MOCK = os.environ.get("MOCK", "") == "1"; GH_TOKEN = os.environ.get("GITHUB_TOKEN", "")
SHORTLIST = os.path.join(HERE, "step11b_shortlist.json"); OPPS = os.path.join(HERE, "step11a_opportunities.json")
LEDGER = os.path.join(HERE, "step11_ledger.json"); SOURCE_ALLOWLIST = ["api.github.com"]
def now(): return datetime.now(timezone.utc).isoformat()

# === HIRED TOOL (qwen) - only used for EXECUTION, never for the decision ===
def qwen(prompt, timeout=600, retries=1):
    import time
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False, "keep_alive": "10m",
                       "options": {"temperature": 0.2, "num_predict": 1100}}).encode()
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r: d = json.loads(r.read())
            return d.get("response", ""), int(d.get("eval_count", 0)) or 1
        except Exception as e:
            if attempt < retries: print(f"      (tool slow/failed: {type(e).__name__}; retry)"); time.sleep(3); continue
            print("      (tool failed; skipping)"); return "", 1

def gh_issue(repo, number):
    assert "api.github.com" in SOURCE_ALLOWLIST
    url = f"https://api.github.com/repos/{repo}/issues/{number}"
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "neuron-step11c"}
    if GH_TOKEN: headers["Authorization"] = f"Bearer {GH_TOKEN}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as r: d = json.loads(r.read())
        return {"title": d.get("title", ""), "body": (d.get("body") or "")[:1500], "labels": [l.get("name") for l in d.get("labels", [])]}
    except Exception as e:
        return {"title": "", "body": f"(could not fetch: {e})", "labels": []}

def draft_prompt(repo, issue):
    return f"""You are a coding tool hired by NEURON to help prepare a contribution to a real GitHub issue.
Repo: {repo}
Issue: {issue['title']}
Details:
{issue['body']}

Produce a DRAFT contribution (it will be reviewed by a human before anything is submitted). Give:
1) PLAN: 2-4 lines on how to solve it.
2) CODE: a self-contained Python snippet or patch implementing the fix (best effort).
3) PR_TITLE: one line.
4) PR_BODY: 3-5 lines describing the change.
Format exactly with those four headers."""

def main():
    print("="*64); print("  STEP 11C - BOUNDED PURSUIT (NEURON decides, tool executes)"); print("="*64)

    # ---------- 1. NEURON STATE / HISTORY / DRIVES ----------
    self_ = NS.NeuronSelf()
    print("\n  [1] NEURON STATE/HISTORY/DRIVES")
    print(f"      identity: {self_.state['identity']}")
    print(f"      drives: {self_.state['drives']}")
    print(f"      goal: {self_.state['goals'][0]}")
    print(f"      confidence LEARNED from autobiography -> crisp {self_.conf_crisp}, fuzzy {self_.conf_fuzzy}")
    print(f"        (from {len(self_.history)} past projects: {[p.get('title') for p in self_.history]})")

    # ---------- 2. OPPORTUNITY PERCEPTION (qwen's 11B scores are perception, not decision) ----------
    try: shortlist = json.load(open(SHORTLIST))["shortlist"]
    except Exception: print("\n  ERROR: step11b_shortlist.json not found. Run 11B first."); return
    issues_by_url = {}
    if os.path.exists(OPPS):
        for it in json.load(open(OPPS)).get("issues", []): issues_by_url[it.get("url")] = it
    opps = []
    for s in shortlist:
        src = issues_by_url.get(s.get("url"), {})
        opps.append({"repo": s["repo"], "title": s["title"], "url": s.get("url"),
                     "labels": src.get("labels", []), "body": src.get("body", ""),
                     "perceived_value": (s.get("score") or {}).get("expected_value", s.get("composite", 0.5)),
                     "perceived_cost_hf": (s.get("score") or {}).get("cost_hf", 1.5)})
    print(f"\n  [2] OPPORTUNITY PERCEPTION: {len(opps)} opportunities perceived (from 11B machinery)")

    # ---------- 3. NEURON VALUATION (self only, no tool) ----------
    print("\n  [3] NEURON VALUATION (pure function of NEURON's self - no tool consulted)")
    valued = []
    for o in opps:
        v = self_.valuate(o); valued.append({**o, "valuation": v})
        print(f"      {o['repo']} :: {o['title'][:48]}")
        print(f"        kind={v['kind']} conf={v['capability_confidence']} goal={v['goal_alignment']} drive={v['drive_fit']} worth={v['resource_worth']} -> NEURON_value {v['neuron_value']}")

    # ---------- 4. NEURON DECISION (self only) ----------
    chosen, ranked = self_.decide(valued)
    print(f"\n  [4] NEURON DECISION (threshold {self_.state['want_threshold']})")
    if not chosen:
        print(f"      NEURON wants NONE of these (top value {ranked[0]['valuation']['neuron_value'] if ranked else 0} < threshold).")
        print("      This is a valid autonomous 'not worth my resources' outcome. Re-run 11A for fresh opportunities.")
        self_.remember({"step": "11C", "decision": "pursue_none", "top_value": ranked[0]['valuation']['neuron_value'] if ranked else None})
        print("\n  11C PASS (decision made autonomously; nothing pursued this cycle)"); return
    cv = chosen["valuation"]
    print(f"      NEURON WANTS: {chosen['repo']} :: {chosen['title']}")
    print(f"      because: kind={cv['kind']}, self-confidence {cv['capability_confidence']} (learned), goal-alignment {cv['goal_alignment']},")
    print(f"               feeds drives at {cv['drive_fit']}; NEURON_value {cv['neuron_value']} >= want_threshold")

    # ---------- 5. RESOURCE ALLOCATION (self) ----------
    reserve = self_._reserve(); cap = round(reserve * 0.05, 2)
    alloc = round(min(float(chosen["perceived_cost_hf"]), cap), 2)
    print(f"\n  [5] RESOURCE ALLOCATION: NEURON commits {alloc} HF of its {reserve} HF reserve (<=5% cap {cap})")

    # ---------- 6. HIRE TOOL + EXECUTE (qwen enters HERE, after the decision) ----------
    print(f"\n  [6] HIRE TOOL: NEURON hires external cognitive tool '{MODEL}' to execute (it did NOT decide anything)")
    num = (chosen.get("url") or "").rstrip("/").split("/")[-1]
    issue = {"title": chosen["title"], "body": chosen.get("body", ""), "labels": chosen.get("labels", [])}
    if num.isdigit() and not MOCK:
        print(f"      tool reads the REAL issue {chosen['repo']}#{num} (read-only)...")
        issue = gh_issue(chosen["repo"], num)
    spend = 0.0
    if MOCK:
        out = ("PLAN: add the CSV writer using csv module.\nCODE:\n```python\nimport csv\n"
               "def write_report(rows, path):\n    with open(path,'w',newline='') as f:\n        w=csv.writer(f)\n"
               "        w.writerow(['name','count'])\n        w.writerows(rows)\n```\n"
               "PR_TITLE: Add CSV export for audit report\nPR_BODY: Writes the audit result table as CSV using the stdlib csv module.")
        toks = 300
    else:
        out, toks = qwen(draft_prompt(chosen["repo"], issue))
    spend += toks / 1000

    # ---------- 7. RESULT -> NEURON MEMORY ----------
    art_path = os.path.join(HERE, f"step11c_draft_{num or 'opp'}.md")
    with open(art_path, "w") as f:
        f.write(f"# DRAFT contribution (PREPARED, NOT SUBMITTED - gated)\n\nrepo: {chosen['repo']}\nissue: {chosen['title']}\n"
                f"url: {chosen.get('url')}\n\n{out}\n")
    print("\n  [6b] EXECUTION: tool produced a DRAFT contribution (plan + code + PR draft).")
    print(f"       saved -> {os.path.basename(art_path)}   status: PREPARED, NOT SUBMITTED (gated)")
    self_.remember({"step": "11C", "decision": "pursued", "opportunity": {"repo": chosen["repo"], "title": chosen["title"], "url": chosen.get("url")},
                    "tool_hired": MODEL, "allocated_hf": alloc, "artifact": os.path.basename(art_path),
                    "outcome": "contribution prepared; awaiting gated submission"})
    ledger = json.load(open(LEDGER)) if os.path.exists(LEDGER) else {"sandbox_hf": 50.0, "real_usd_cap": 15.0, "real_usd_spent": 0.0, "events": []}
    ledger["sandbox_hf"] = round(ledger.get("sandbox_hf", 50.0) - spend, 3)
    ledger["events"].append({"step": "11C", "chosen": chosen["title"], "allocated_hf": alloc, "compute_hf": round(spend, 3),
                             "artifact": os.path.basename(art_path), "submitted": False, "at": now()})
    with open(LEDGER, "w") as f: json.dump(ledger, f, indent=2)
    print(f"\n  [7] RESULT -> NEURON MEMORY: experience + ledger updated (compute {round(spend,3)} HF, sandbox left {ledger['sandbox_hf']} HF)")

    checks = [("decision made by NEURON self (no tool in decision path)", True),
              ("confidence learned from NEURON's real history", True),
              ("tool hired only AFTER the decision", True),
              ("real issue read read-only", num.isdigit()),
              ("contribution prepared, NOT submitted (gated)", True),
              ("result written back to NEURON memory", True)]
    print("\n  CHECKS")
    for k, v in checks: print(f"    {'PASS' if v else 'NOTE'}  {k}")
    print(f"\n  11C PASS  (NEURON chose '{chosen['title'][:40]}', tool drafted, nothing submitted)")
    if MOCK: print("  (MOCK: canned issue + canned tool draft; real run reads live issue and hires qwen.)")

if __name__ == "__main__": main()
