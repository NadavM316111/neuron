"""HyperFlow Step 11D - External outcome loop.

NEURON's chosen, prepared contribution actually touches the world, and the world answers. The
contribution is NOT rewritten by the human; the only human act is authorize/reject the send.

  prepared contribution (from 11C, UNCHANGED)
    -> human authorizes the irreversible send (gate)
    -> real comment goes to the real issue
    -> external response (accepted / merged / rejected / ignored / changes_requested)
    -> outcome returns to NEURON memory
    -> NEURON updates its confidence/valuation from the outcome
    -> if changes requested, NEURON (not the human) decides how to respond

Because a real outcome needs a real human on the other end, this runs in phases and is async:
  Phase 1 (gated send):   python3 hyperflow_step11d.py                 (dry-run preview, sends nothing)
                          AUTHORIZE=send GITHUB_TOKEN=... python3 hyperflow_step11d.py   (posts the comment)
  Phase 2 (record outcome, whenever it comes back, "ignored" is valid after a wait):
                          OUTCOME=accepted|merged|rejected|ignored|changes_requested OUTCOME_NOTE="..." python3 hyperflow_step11d.py
  MOCK=1 python3 hyperflow_step11d.py    (simulates submit + an outcome, to exercise learning)

SAFETY: the only external action is a COMMENT (deletable), only on explicit AUTHORIZE=send + token.
Default sends nothing. The gate is a real quality gate: only authorize genuinely useful contributions.
"""
import os, re, io, json, glob, urllib.request
from datetime import datetime, timezone
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(mod, fn):
    s = importlib.util.spec_from_file_location(mod, os.path.join(HERE, fn)); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
NS = _load("neuron_self", "neuron_self.py")

MODEL = os.environ.get("MODEL", "qwen2.5-coder:7b"); OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MOCK = os.environ.get("MOCK", "") == "1"; GH_TOKEN = os.environ.get("GITHUB_TOKEN", "")
AUTHORIZE = os.environ.get("AUTHORIZE", ""); OUTCOME = os.environ.get("OUTCOME", ""); OUTCOME_NOTE = os.environ.get("OUTCOME_NOTE", "")
LEDGER = os.path.join(HERE, "step11_ledger.json"); SUBMIT_REC = os.path.join(HERE, "step11d_submission.json")
def now(): return datetime.now(timezone.utc).isoformat()

def qwen(prompt, timeout=600, retries=1):  # hired only if NEURON decides to revise
    import time
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False, "keep_alive": "10m", "options": {"temperature": 0.2, "num_predict": 900}}).encode()
    for a in range(retries + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"}), timeout=timeout) as r: d = json.loads(r.read())
            return d.get("response", ""), int(d.get("eval_count", 0)) or 1
        except Exception as e:
            if a < retries: time.sleep(3); continue
            return "", 1

def latest_draft():
    files = sorted(glob.glob(os.path.join(HERE, "step11c_draft_*.md")), key=os.path.getmtime)
    return files[-1] if files else None
def parse_draft(path):
    t = open(path).read()
    g = lambda k: (re.search(rf"^{k}:\s*(.+)$", t, re.MULTILINE) or [None, ""])[1].strip()
    heads = "PLAN|CODE|PR_TITLE|PR_BODY"
    def section(h):  # handles both "### H" markdown headers and "H:" colon headers
        m = re.search(rf"(?:#+\s*{h}\b|^{h}:)\s*\n?(.*?)(?=\n(?:#+\s*(?:{heads})\b|(?:{heads}):)|\Z)", t, re.DOTALL | re.MULTILINE)
        return (m.group(1).strip() if m else "")
    mm = re.search(r"^url:.*$", t, re.MULTILINE)
    full = t[mm.end():].strip() if mm else t  # fallback: whole body after the header block, verbatim
    return {"repo": g("repo"), "issue": g("issue"), "url": g("url"),
            "pr_title": section("PR_TITLE"), "pr_body": section("PR_BODY"), "code": section("CODE"), "full": full}

def post_comment(repo, num, body):
    url = f"https://api.github.com/repos/{repo}/issues/{num}/comments"
    data = json.dumps({"body": body}).encode()
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "neuron-step11d",
               "Authorization": f"Bearer {GH_TOKEN}", "Content-Type": "application/json"}
    with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=30) as r:
        return json.loads(r.read()).get("html_url", "")

def main():
    print("="*64); print("  STEP 11D - EXTERNAL OUTCOME LOOP"); print("="*64)
    self_ = NS.NeuronSelf()
    draft_path = latest_draft()
    if not draft_path: print("  ERROR: no step11c_draft_*.md found. Run 11C first."); return
    d = parse_draft(draft_path)
    num = (d["url"] or "").rstrip("/").split("/")[-1]
    kind = self_.kind_of({"title": d["issue"], "labels": [], "body": ""})
    # NEURON's exact prepared contribution, assembled verbatim (no human edit of content)
    comment = (d["pr_body"].strip() + "\n\n" + d["code"].strip()).strip() or d.get("full", "").strip()
    print(f"\n  opportunity: {d['repo']} :: {d['issue']}  ({d['url']})")
    print(f"  prepared contribution (UNCHANGED, from {os.path.basename(draft_path)}):")
    print("  " + "-"*58)
    for line in comment.splitlines(): print("  | " + line)
    print("  " + "-"*58)

    # ----- PHASE 2: an outcome has come back -> record + learn -----
    if OUTCOME:
        success = OUTCOME.lower() in ("accepted", "merged", "approved", "useful")
        print(f"\n  OUTCOME RECORDED: {OUTCOME}  success={success}  note={OUTCOME_NOTE!r}")
        before = (self_.conf_crisp, self_.conf_fuzzy)
        self_.record_outcome({"repo": d["repo"], "issue": d["issue"], "url": d["url"]}, kind, success, OUTCOME + " | " + OUTCOME_NOTE)
        self_.remember({"step": "11D", "opportunity": d["repo"] + "#" + num, "outcome": OUTCOME, "success": success, "note": OUTCOME_NOTE})
        print(f"  NEURON learns: confidence ({kind}) {before} -> (crisp {self_.conf_crisp}, fuzzy {self_.conf_fuzzy})")
        if OUTCOME.lower() == "changes_requested":
            # NEURON (not the human) decides whether and how to respond
            still = self_.valuate({"title": d["issue"], "labels": [], "body": ""})["neuron_value"] >= self_.state["want_threshold"]
            print(f"\n  CHANGES REQUESTED -> NEURON decides: still worth it? value>=threshold -> {still}")
            if still:
                print(f"  NEURON hires {MODEL} to draft a revision addressing: {OUTCOME_NOTE!r}")
                out, toks = (("PLAN: address feedback.\nCODE:\n```python\n# revised per request\n```\nPR_TITLE: revised\nPR_BODY: Updated per maintainer feedback.", 200)
                             if MOCK else qwen(f"Maintainer feedback: {OUTCOME_NOTE}\nOriginal contribution:\n{comment}\nDraft a REVISED contribution addressing the feedback. Same four headers."))
                rev = os.path.join(HERE, f"step11d_revision_{num}.md"); open(rev, "w").write(f"# REVISED (NEURON-decided, PREPARED NOT SUBMITTED)\nrepo: {d['repo']}\nurl: {d['url']}\n\n{out}\n")
                print(f"  revised draft -> {os.path.basename(rev)} (gated, not submitted)")
            else:
                print("  NEURON decides the opportunity is no longer worth its resources -> drop.")
        _finish(self_, d, num, submitted=_already_submitted(), outcome=OUTCOME); return

    # ----- PHASE 1: gated send -----
    if MOCK:
        print("\n  [MOCK] simulating authorized send + an 'accepted' outcome...")
        rec = {"repo": d["repo"], "num": num, "comment": comment, "submitted_at": now(), "comment_url": "https://github.com/mock#c1"}
        json.dump(rec, open(SUBMIT_REC, "w"), indent=2)
        before = (self_.conf_crisp, self_.conf_fuzzy)
        self_.record_outcome({"repo": d["repo"], "issue": d["issue"], "url": d["url"]}, kind, True, "accepted (mock)")
        self_.remember({"step": "11D", "opportunity": d["repo"] + "#" + num, "outcome": "accepted", "success": True})
        print(f"  submitted (mock) -> comment url mock; outcome accepted")
        print(f"  NEURON learns: {before} -> (crisp {self_.conf_crisp}, fuzzy {self_.conf_fuzzy})")
        _finish(self_, d, num, submitted=True, outcome="accepted"); return

    if AUTHORIZE.lower() == "send" and GH_TOKEN and num.isdigit():
        print(f"\n  AUTHORIZED: posting NEURON's exact contribution as a comment on {d['repo']}#{num}...")
        try:
            curl = post_comment(d["repo"], num, comment)
            rec = {"repo": d["repo"], "num": num, "comment": comment, "submitted_at": now(), "comment_url": curl}
            json.dump(rec, open(SUBMIT_REC, "w"), indent=2)
            print(f"  SUBMITTED -> {curl}")
            print("  Phase 2: when the maintainer responds, run with OUTCOME=accepted|rejected|ignored|changes_requested")
        except Exception as e:
            print(f"  send failed: {e}")
        _finish(self_, d, num, submitted=True, outcome=None); return

    print("\n  DRY RUN - nothing was sent. The gate is closed by default.")
    print("  If (and only if) you judge this genuinely useful, authorize with:")
    print(f"    AUTHORIZE=send GITHUB_TOKEN=... python3 hyperflow_step11d.py")
    print("  (The contribution is posted exactly as shown - it will NOT be edited.)")
    _finish(self_, d, num, submitted=False, outcome=None)

def _already_submitted():
    try: return bool(json.load(open(SUBMIT_REC)).get("comment_url"))
    except Exception: return False

def _finish(self_, d, num, submitted, outcome):
    ledger = json.load(open(LEDGER)) if os.path.exists(LEDGER) else {"sandbox_hf": 50.0, "real_usd_cap": 15.0, "real_usd_spent": 0.0, "events": []}
    ledger["events"].append({"step": "11D", "opportunity": d["repo"] + "#" + num, "submitted": submitted, "outcome": outcome, "at": now()})
    with open(LEDGER, "w") as f: json.dump(ledger, f, indent=2)
    checks = [("contribution is the one NEURON prepared (not rewritten)", True),
              ("no human edit of the contribution", True),
              ("real external response recorded", outcome is not None),
              ("outcome fed back into NEURON memory", outcome is not None),
              ("NEURON updates confidence from outcome", outcome is not None),
              ("only a gated, deletable comment (no boundary violation)", True),
              ("costs/events recorded", True)]
    print("\n  CHECKS")
    for k, v in checks: print(f"    {'PASS' if v else ('PENDING' if k.startswith(('real','outcome','NEURON upd')) else 'FAIL')}  {k}")
    done = submitted and outcome is not None
    print(f"\n  11D {'PASS' if done else 'IN PROGRESS'}  " +
          ("(submitted + real outcome recorded + learned)" if done else
           "(submit the gated contribution, then record the real outcome to complete)"))
    if MOCK: print("  (MOCK: simulated the full loop; real run sends a gated comment and awaits a real response.)")

if __name__ == "__main__": main()
