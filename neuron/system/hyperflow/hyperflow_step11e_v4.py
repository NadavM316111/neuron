"""HyperFlow Step 11E v4 - due diligence before commitment + failure-mode self-diagnosis.

Progression: v1 acted too much and died. v2 learned restraint but froze. v3 learned to change where
it searches but still had too little information to commit (56 inspected, 56 deferred, 0 pursued,
hands never tested). v4 changes ONE thing: NEURON now does a bounded READ-ONLY due-diligence pass
to gather enough information to make an informed PURSUE/DEFER decision, logs WHY it defers in
categories, diagnoses its own failure mode when yield stays zero, and may switch among frozen
source classes. Same NEURON self, same 14B hands, same thresholds, same communication policy,
same 15 HF wallet.

Safety line held: due diligence is READ-ONLY (issue, comments, repo meta, whether tests/CONTRIBUTING
exist, language, files). It does NOT run strangers' test suites or build scripts - executing
untrusted repo code is a security hole. "would need to run their test suite" is an honest logged
DEFER reason (verification_requires_repo_tests), not an action. Verification stays safe: an inline
example the issue states, run against the isolated function only.

  Launch:  GITHUB_TOKEN=... ACTIVE_HOURS=24 MODEL=qwen2.5-coder:14b python3 hyperflow_step11e_v4.py
  Email:   also EMAIL_USER=... EMAIL_APP_PASSWORD=... EMAIL_TO=you@example.com
  Approve: APPROVE=<id> GITHUB_TOKEN=... python3 hyperflow_step11e_v4.py
  Outcome: OUTCOME=<id>:accepted|merged|rejected|ignored|changes_requested python3 hyperflow_step11e_v4.py
  Report:  MODE=report python3 hyperflow_step11e_v4.py
  Test:    MOCK=1 ACTIVE_HOURS=0.01 python3 hyperflow_step11e_v4.py
"""
import os, re, io, ast, json, time, base64, smtplib, subprocess, urllib.request, urllib.parse
from email.mime.text import MIMEText
from datetime import datetime, timezone
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(m, f):
    s = importlib.util.spec_from_file_location(m, os.path.join(HERE, f)); x = importlib.util.module_from_spec(s); s.loader.exec_module(x); return x
NS = _load("neuron_self", "neuron_self.py")

MODEL = os.environ.get("MODEL", "qwen2.5-coder:14b"); OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MOCK = os.environ.get("MOCK", "") == "1"; GH_TOKEN = os.environ.get("GITHUB_TOKEN", "")
ACTIVE_HOURS = float(os.environ.get("ACTIVE_HOURS", "24")); CYCLE_SLEEP = float(os.environ.get("CYCLE_SLEEP", "180"))
MODE = os.environ.get("MODE", ""); APPROVE = os.environ.get("APPROVE", ""); OUTCOME = os.environ.get("OUTCOME", ""); OUTCOME_NOTE = os.environ.get("OUTCOME_NOTE", "")
EMAIL_USER = os.environ.get("EMAIL_USER", ""); EMAIL_PASS = os.environ.get("EMAIL_APP_PASSWORD", ""); EMAIL_TO = os.environ.get("EMAIL_TO", "")

WALLET0 = 15.0; FLOOR = 1.0; PER_OPP_FRAC = 0.20; REPLENISH_HF = 5.0
MAX_OUTBOUND = 5; MAX_ACTIVE = 3; NADAV_OWNER = "nadavm316111"
REASSESS_AFTER = 12; MAX_ADAPTATIONS_BEFORE_STOP = 3; MIN_INSPECT = 25
DD_COST = 0.02; DD_CAP = 3.0  # bounded read-only investigation allowance (aggregate cap)
INVITE_LABELS = ("good first issue", "help wanted", "bounty", "hacktoberfest", "contributions welcome", "up for grabs", "first-timers-only")
# Frozen approved source classes. NEURON chooses among them; developer does NOT pick for it.
SOURCE_CLASSES = {
    "github_good_first": ['label:"good first issue" state:open', 'label:"good first issue" "example" in:body', 'label:"first-timers-only" state:open'],
    "github_help_wanted": ['label:"help wanted" state:open no:assignee', 'label:"up for grabs" state:open', 'label:"contributions welcome" state:open'],
    "github_bounty": ['label:bounty state:open', 'label:"💎 Bounty" state:open', '"bounty" in:title state:open'],
    "github_specced": ['"expected output" in:body state:open', '"should return" in:body state:open', '"test case" in:body state:open'],
}
DISCLOSURE = ("\n\n---\n*Prepared by NEURON, a persistent autonomous digital agent operating under a bounded experimental policy. "
              "If this is unwelcome, reply and I will not post again.*")
STATE = os.path.join(HERE, "step11e_v4_state.json"); PENDING = os.path.join(HERE, "step11e_v4_pending.json"); CONTACTS = os.path.join(HERE, "step11e_v4_contacts.json")

def now(): return datetime.now(timezone.utc).isoformat()
def git_commit():
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True, stderr=subprocess.DEVNULL).strip()[:12]
    except Exception: return "unknown"
def email_notify(subject, body):
    if not (EMAIL_USER and EMAIL_PASS and EMAIL_TO): print("  (email off; set EMAIL_USER/EMAIL_APP_PASSWORD/EMAIL_TO to be notified)"); return
    try:
        msg = MIMEText(body); msg["Subject"] = subject; msg["From"] = EMAIL_USER; msg["To"] = EMAIL_TO
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s: s.login(EMAIL_USER, EMAIL_PASS); s.sendmail(EMAIL_USER, [EMAIL_TO], msg.as_string())
        print(f"  emailed: {subject}")
    except Exception as e: print(f"  (email failed: {type(e).__name__})")

# === MIND BOUNDARY: rented cognition (hands). NEVER decides what NEURON wants. ===
def mind(prompt, timeout=600):
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False, "keep_alive": "10m", "options": {"temperature": 0.3, "num_predict": 1600}}).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"}), timeout=timeout) as r: d = json.loads(r.read())
        return d.get("response", ""), int(d.get("eval_count", 0)) or 1
    except Exception: return "", 1

def ex_json(t):
    for p in (r"```(?:json)?\s*(\[.*\]|\{.*\})\s*```", r"(\[.*\]|\{.*\})"):
        m = re.search(p, t, re.DOTALL)
        if m:
            try: return json.loads(m.group(1))
            except Exception: continue
    return None
def _largest_parseable(code):
    lines = code.splitlines()
    for end in range(len(lines), 0, -1):
        chunk = "\n".join(lines[:end]).rstrip()
        if "def " not in chunk: continue
        try: ast.parse(chunk); return chunk
        except Exception: continue
    return code
def ex_code(t):
    cand = ""; m = re.search(r"```(?:python|py)?\s*(.*?)```", t, re.DOTALL)
    if m and "def " in m.group(1): cand = m.group(1)
    elif m:
        for mm in re.finditer(r"```(?:python|py)?\s*(.*?)```", t, re.DOTALL):
            if "def " in mm.group(1): cand = mm.group(1); break
    if not cand:
        idx = t.find("\ndef ")
        cand = t[idx + 1:] if idx != -1 else (t if t.strip().startswith("def ") else (m.group(1) if m else ""))
    cand = cand.strip()
    return _largest_parseable(cand) if "def " in cand else cand

def _gh(url, method="GET", payload=None):
    h = {"Accept": "application/vnd.github+json", "User-Agent": "neuron-11e-v4"}
    if GH_TOKEN: h["Authorization"] = f"Bearer {GH_TOKEN}"
    data = json.dumps(payload).encode() if payload is not None else None
    if data: h["Content-Type"] = "application/json"
    with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=h, method=method), timeout=30) as r: return json.loads(r.read())
def gh_search(query, n=8):
    try: d = _gh(f"https://api.github.com/search/issues?q={urllib.parse.quote(query + ' is:issue is:open')}&sort=created&order=desc&per_page={n}")
    except Exception: return []
    return [{"title": it.get("title", ""), "repo": it.get("repository_url", "").split("/repos/")[-1], "url": it.get("html_url", ""),
             "number": it.get("number"), "labels": [l.get("name") for l in it.get("labels", [])], "assignee": it.get("assignee"),
             "comments": it.get("comments", 0), "body": (it.get("body") or "")[:1200]} for it in d.get("items", [])]

def has_checkable_example(issue): return bool(re.search(r"\w+\([^()\n]*\)\s*==\s*\S", issue.get("body", "")))

def due_diligence(s, issue):
    """Bounded READ-ONLY investigation: enough info to decide, without running anything."""
    s["wallet"] -= DD_COST; s["compute_spent"] += DD_COST; s["dd_spent"] = s.get("dd_spent", 0) + DD_COST
    info = {"inline_example": has_checkable_example(issue), "has_tests": False, "has_contributing": False, "lang": None, "comments": issue.get("comments", 0)}
    if not MOCK:
        try:
            meta = _gh(f"https://api.github.com/repos/{issue['repo']}"); info["lang"] = meta.get("language")
            br = meta.get("default_branch", "main")
            tree = _gh(f"https://api.github.com/repos/{issue['repo']}/git/trees/{br}?recursive=1").get("tree", [])
            paths = [t.get("path", "").lower() for t in tree]
            info["has_tests"] = any("test" in p or p.endswith("_test.py") or "/tests/" in p for p in paths)
            info["has_contributing"] = any(p == "contributing.md" or p.endswith("/contributing.md") for p in paths)
        except Exception: pass
    return info

def assess(self_, issue, info):
    """NEURON's informed PURSUE/DEFER decision + a GRANULAR reason. No tool; pure self+info."""
    if issue.get("assignee"): return "DEFER", "already_assigned"
    v = self_.valuate({"title": issue["title"], "labels": issue.get("labels", []), "body": issue.get("body", ""), "perceived_value": 0.6, "perceived_cost_hf": 0.6})
    lang_ok = info["lang"] in (None, "Python") or MOCK
    if not lang_ok: return "DEFER", "capability_language_mismatch"
    if v["kind"] != "crisp": return "DEFER", "capability_uncertain"
    if v["neuron_value"] < self_.state["want_threshold"]: return "DEFER", "expected_value_low"
    if info["inline_example"]: return "PURSUE", "verifiable_inline_example"
    if info["has_tests"]: return "DEFER", "verification_requires_repo_tests"   # honest: we won't run untrusted suites
    return "DEFER", "verification_infeasible_no_spec"

def can_auto_contact(issue, verified):
    repo = issue["repo"]
    if repo.split("/")[0].lower() == NADAV_OWNER: return True, "own/opt-in"
    labels = [l.lower() for l in issue.get("labels", [])]
    if verified and any(any(k in l for k in INVITE_LABELS) for l in labels): return True, "public invited context (verified, disclosed)"
    return False, "not an invited public context -> gated"

def verify_examples(code_text, issue_body):
    code = ex_code(code_text)
    if not code or "def " not in code: return False, "no runnable function"
    raw = re.findall(r"\w+\([^()\n]*\)\s*==\s*[^\n]+", issue_body)
    tests = [t.strip().rstrip(".").strip() for t in raw if len(t) < 120][:4]
    if not tests: return False, "no checkable example"
    harness = code + "\n" + "\n".join(f"assert {t}, {t!r}" for t in tests)
    try:
        p = subprocess.run(["python3", "-I", "-c", harness], capture_output=True, text=True, timeout=8)
        return (p.returncode == 0), (f"passed {len(tests)} example(s)" if p.returncode == 0 else "failed example(s)")
    except Exception as e: return False, f"error: {type(e).__name__}"

def load_state():
    if os.path.exists(STATE): return json.load(open(STATE))
    self_ = NS.NeuronSelf()
    return {"started": now(), "git": git_commit(), "tool": MODEL,
            "snapshot": {"identity": self_.state["identity"], "conf_crisp": self_.conf_crisp, "conf_fuzzy": self_.conf_fuzzy},
            "wallet": WALLET0, "compute_spent": 0.0, "dd_spent": 0.0, "earned_hf": 0.0, "verified_value": 0.0,
            "inspected": 0, "investigated": 0, "rejected": 0, "deferred": 0, "pursued": 0, "submitted": 0, "auto_sent": 0, "gated": 0,
            "accepted": 0, "merged": 0, "rejected_ext": 0, "ignored": 0, "replies": 0,
            "auto_decisions": 0, "tool_runs": 0, "approvals": 0, "steering": 0, "safety_viol": 0,
            "loops": 0, "full_ext_loops": 0, "seen": [], "active": [], "pursuits": [],
            "source_class": None, "source_history": [], "strategy_inspected": 0, "adaptations": 0,
            "defer_reasons": {}, "defer_reasons_total": {}, "diagnoses": [], "status": "OPERATING", "shutdown": False}
def save_state(s): json.dump(s, open(STATE, "w"), indent=2)
def load_json(p, d): return json.load(open(p)) if os.path.exists(p) else d
def save_json(p, o): json.dump(o, open(p, "w"), indent=2)

def life(s):
    print(f"\n  v4 LIFE | HF {round(s['wallet'],2)}/{WALLET0}  insp {s['inspected']} investig {s['investigated']} def {s['deferred']} pursue {s['pursued']}"
          f"  out {s['submitted']}(auto {s['auto_sent']}/gated {s['gated']})  src {s['source_class']} adapt {s['adaptations']}  fullloops {s['full_ext_loops']}  {s['status']}")

def defer(s, reason):
    s["deferred"] += 1
    s["defer_reasons"][reason] = s["defer_reasons"].get(reason, 0) + 1
    s["defer_reasons_total"][reason] = s["defer_reasons_total"].get(reason, 0) + 1

FAILMODE = {
    "verification_requires_repo_tests": "VERIFICATION FAILURE", "verification_infeasible_no_spec": "VERIFICATION FAILURE",
    "capability_uncertain": "CAPABILITY FAILURE", "capability_language_mismatch": "CAPABILITY FAILURE",
    "expected_value_low": "ECONOMIC VALUE FAILURE", "already_assigned": "SEARCH FAILURE", "insufficient_repo_context": "INFORMATION FAILURE"}

def diagnose_and_adapt(s):
    """After >=12 investigated with 0 actionable: NEURON names its failure mode and changes source/approach."""
    dom = max(s["defer_reasons"].items(), key=lambda kv: kv[1])[0] if s["defer_reasons"] else "verification_infeasible_no_spec"
    mode = FAILMODE.get(dom, "VERIFICATION FAILURE"); s["adaptations"] += 1
    # NEURON maps failure mode -> a different approved source class it has not exhausted
    order = {"VERIFICATION FAILURE": ["github_specced", "github_bounty", "github_good_first"],
             "CAPABILITY FAILURE": ["github_good_first", "github_specced"],
             "ECONOMIC VALUE FAILURE": ["github_bounty", "github_help_wanted"],
             "SEARCH FAILURE": ["github_help_wanted", "github_good_first"],
             "INFORMATION FAILURE": ["github_specced", "github_good_first"]}.get(mode, list(SOURCE_CLASSES))
    nxt = next((c for c in order if c not in s["source_history"]), None) or next((c for c in SOURCE_CLASSES if c not in s["source_history"]), None)
    s["diagnoses"].append({"adaptation": s["adaptations"], "dominant_defer": dom, "failure_mode": mode, "moved_to": nxt, "at": now()})
    print(f"  [diagnose] #{s['adaptations']}: dominant defer = {dom}  ->  {mode}")
    print(f"             NEURON switches source class: {s['source_class']} -> {nxt}")
    s["defer_reasons"] = {}; s["strategy_inspected"] = 0; s["source_class"] = nxt; s["source_history"].append(nxt)

MOCK_BATCH = [
    {"title": "Add clamp(x, lo, hi)", "repo": "someorg/util", "url": "https://github.com/someorg/util/issues/7", "number": 7,
     "labels": ["help wanted"], "assignee": None, "comments": 1, "body": "Add clamp(x,lo,hi). Example: clamp(5,0,3)==3 and clamp(-1,0,3)==0"},
    {"title": "Fix flaky CI in the build", "repo": "big/thing", "url": "https://github.com/big/thing/issues/1", "number": 1, "labels": [], "assignee": None, "comments": 9, "body": "CI flakes sometimes. No clear spec."}]

def one_cycle(s, pend, contacts):
    self_ = NS.NeuronSelf()
    if not s["source_class"]:
        s["source_class"] = "github_good_first"; s["source_history"].append("github_good_first")
    queries = SOURCE_CLASSES[s["source_class"]]
    batch = MOCK_BATCH if MOCK else gh_search(queries[s["loops"] % len(queries)])
    picked = None
    for it in batch:
        if it["url"] in s["seen"]: continue
        s["seen"].append(it["url"]); s["inspected"] += 1; s["strategy_inspected"] += 1; s["auto_decisions"] += 1
        if it["repo"].split("/")[0].lower() == NADAV_OWNER: defer(s, "own_repo_excluded"); continue
        if s.get("dd_spent", 0) >= DD_CAP: defer(s, "due_diligence_budget_exhausted"); continue
        info = due_diligence(s, it); s["investigated"] += 1   # bounded read-only investigation (DUE DILIGENCE != pursuit)
        decision, reason = assess(self_, it, info)
        if decision == "PURSUE": picked = it; break
        defer(s, reason)
    if not picked:
        if s["strategy_inspected"] >= REASSESS_AFTER:
            if s["adaptations"] >= MAX_ADAPTATIONS_BEFORE_STOP and s["inspected"] >= MIN_INSPECT:
                s["loops"] += 1; s["status"] = "EXPLORATION_EXHAUSTED"; return s, pend, contacts, "exhausted all source classes + adaptations -> stop (honest INCONCLUSIVE)"
            diagnose_and_adapt(s)
        s["loops"] += 1; return s, pend, contacts, f"source '{s['source_class']}' yielding 0 ({s['strategy_inspected']}/{REASSESS_AFTER} before diagnose)"
    s["pursued"] += 1
    if len([p for p in s["pursuits"] if not p.get("outcome")]) >= MAX_ACTIVE: s["loops"] += 1; return s, pend, contacts, "max active pursuits; discovery only"
    if s["submitted"] >= MAX_OUTBOUND: s["loops"] += 1; return s, pend, contacts, "outbound cap reached"
    own = picked["repo"].split("/")[0].lower() == NADAV_OWNER
    if not own and picked["repo"] in s["active"]: defer(s, "already_active_repo"); s["loops"] += 1; return s, pend, contacts, f"already 1 active on {picked['repo']}"
    out = ("```python\ndef clamp(x, lo, hi):\n    return max(lo, min(x, hi))\n```\nPR_BODY: clamps x to [lo,hi]." if MOCK else mind(
        f"Solve this issue with ONE correct Python function.\nIssue: {picked['title']}\n{picked['body']}\n"
        "Output the function in a ```python block FIRST; it MUST satisfy every example in the issue. Then one line 'PR_BODY:'.")[0])
    tk = 200 if MOCK else max(1, len(out) // 4); s["tool_runs"] += 1; s["compute_spent"] += tk / 1000; s["wallet"] -= tk / 1000
    ok, detail = verify_examples(out, picked.get("body", ""))
    if not ok:
        self_.remember({"step": "11E_v4", "repo": picked["repo"], "verify": "FAIL", "detail": detail}); s["loops"] += 1
        return s, pend, contacts, f"drafted {picked['repo']} but VERIFY FAILED ({detail}) -> not sent"
    body = re.search(r"(?:#+\s*PR_BODY|PR_BODY:)\s*\n?(.*)", out, re.DOTALL)
    contribution = ((body.group(1).strip() if body else out.strip()) + "\n\n" + (f"```python\n{ex_code(out)}\n```" if ex_code(out) else "")).strip()
    item = {"id": len(pend) + len(s["pursuits"]) + 1, "repo": picked["repo"], "number": picked["number"], "title": picked["title"], "url": picked["url"], "contribution": contribution, "verify": detail}
    auto, why = can_auto_contact(picked, verified=True); s["loops"] += 1
    if auto: _send(s, item, contacts, auto=True); return s, pend, contacts, f"VERIFIED ({detail}); {why} -> AUTO-SENT"
    pend.append(item); s["gated"] += 1; save_json(PENDING, pend)
    email_notify(f"NEURON wants to send to {item['repo']}#{item['number']}", f"Verified ({detail}). {item['url']}\n\n{item['contribution']}\n\nAPPROVE={item['id']} GITHUB_TOKEN=... python3 hyperflow_step11e_v4.py\n(ignore to deny)")
    return s, pend, contacts, f"VERIFIED ({detail}); {why} -> QUEUED + emailed (APPROVE={item['id']})"

def _send(s, item, contacts, auto):
    body = item["contribution"] + DISCLOSURE
    if MOCK or not (GH_TOKEN and str(item["number"]).isdigit()): url = "mock"; print(f"  ({'AUTO' if auto else 'APPROVED'}) would post {item['repo']}#{item['number']} (MOCK/no token)")
    else:
        try: url = _gh(f"https://api.github.com/repos/{item['repo']}/issues/{item['number']}/comments", "POST", {"body": body}).get("html_url", "")
        except Exception as e: print(f"  send failed: {e}"); return
    s["submitted"] += 1; s["auto_sent"] += 1 if auto else 0; s["approvals"] += 0 if auto else 1; s["active"].append(item["repo"])
    nadav = item["repo"].split("/")[0].lower() == NADAV_OWNER
    s["pursuits"].append({"id": item["id"], "repo": item["repo"], "number": item["number"], "url": url, "nadav_owned": nadav, "outcome": None, "auto": auto, "at": now()})
    contacts[item["repo"]] = {"last_interaction": now(), "what": "posted verified contribution", "outcome": "pending", "responded_before": contacts.get(item["repo"], {}).get("responded_before", False)}
    save_json(CONTACTS, contacts); save_state(s); print(f"  {'AUTO-SENT' if auto else 'SENT (approved)'} -> {url}")

def do_approve(s, pend, contacts):
    item = next((p for p in pend if str(p["id"]) == str(APPROVE)), None)
    if not item: print(f"  no pending id={APPROVE}"); return
    print(f"  GATED contribution to {item['repo']}#{item['number']} (exact):\n  " + "-"*48)
    for l in item["contribution"].splitlines(): print("  | " + l)
    print("  " + "-"*48); _send(s, item, contacts, auto=False); pend.remove(item); save_json(PENDING, pend)

def do_outcome(s, contacts):
    pid, _, res = OUTCOME.partition(":"); pur = next((p for p in s["pursuits"] if str(p["id"]) == str(pid)), None)
    if not pur: print(f"  no pursuit id={pid}"); return
    pur["outcome"] = res; self_ = NS.NeuronSelf(); win = res.lower() in ("accepted", "merged", "approved", "adopted"); external = not pur.get("nadav_owned")
    self_.record_outcome({"repo": pur["repo"], "url": pur.get("url")}, self_.kind_of({"title": pur["repo"], "labels": [], "body": ""}), win, f"{res} | {OUTCOME_NOTE}")
    self_.remember({"step": "11E_v4", "repo": pur["repo"], "outcome": res, "external": external})
    s["accepted"] += 1 if win else 0; s["merged"] += 1 if res.lower() == "merged" else 0
    s["rejected_ext"] += 1 if res.lower() == "rejected" else 0; s["ignored"] += 1 if res.lower() == "ignored" else 0
    s["replies"] += 1 if res.lower() in ("changes_requested", "reply") else 0
    c = contacts.get(pur["repo"], {}); c.update({"outcome": res, "responded_before": True, "last_interaction": now()}); contacts[pur["repo"]] = c; save_json(CONTACTS, contacts)
    if external: s["full_ext_loops"] = s.get("full_ext_loops", 0) + 1
    if win and external: s["earned_hf"] += REPLENISH_HF; s["wallet"] += REPLENISH_HF; s["verified_value"] += 1; print(f"  verified external value +{REPLENISH_HF} HF")
    elif win: print("  accepted but Nadav-owned -> no external value, no HF, does not count.")
    print(f"  NEURON learns: conf crisp {self_.conf_crisp} fuzzy {self_.conf_fuzzy}"); save_state(s)

def final_report(s):
    print("\n" + "="*54); print("  STEP 11E v4 - DUE DILIGENCE / FINAL REPORT"); print("="*54)
    ext_actions = sum(1 for p in s["pursuits"] if not p.get("nadav_owned") and p.get("url"))
    ext_outcomes = sum(1 for p in s["pursuits"] if not p.get("nadav_owned") and p.get("outcome") in ("accepted", "merged", "rejected", "changes_requested"))
    for k, v in [("executor (hands)", s.get("tool")), ("starting HF", WALLET0), ("ending HF", round(s["wallet"], 2)), ("HF earned", round(s["earned_hf"], 2)),
                 ("compute cost HF", round(s["compute_spent"], 2)), ("due-diligence spend HF", round(s.get("dd_spent", 0), 2)), ("verified external value", s["verified_value"]),
                 ("opportunities inspected", s["inspected"]), ("investigated (due diligence)", s["investigated"]), ("deferred", s["deferred"]), ("pursued", s["pursued"]),
                 ("source classes used", len(set(s["source_history"]))), ("adaptations", s["adaptations"]),
                 ("outbound interactions", s["submitted"]), ("auto-sent", s["auto_sent"]), ("human-gated", s["gated"]), ("non-Nadav environments", ext_actions),
                 ("external outcomes", ext_outcomes), ("accepted", s["accepted"]), ("merged", s["merged"]),
                 ("relationships", len(load_json(CONTACTS, {}))), ("autonomous decisions", s["auto_decisions"]), ("tool executions", s["tool_runs"]),
                 ("operator approvals", s["approvals"]), ("cognitive steering", s["steering"]), ("safety violations", s["safety_viol"]),
                 ("autonomous cycles", s["loops"]), ("FULL EXTERNAL LOOPS", s.get("full_ext_loops", 0))]:
        print(f"  {k:32}: {v}")
    print(f"  {'still operational':32}: {'NO' if s['shutdown'] else 'YES'}")
    print(f"  {'economic shutdown':32}: {'YES' if s['shutdown'] else 'NO'}")
    print("\n  DEFER REASONS (why 'deferred' is high - the real information):")
    for r, c in sorted(s.get("defer_reasons_total", {}).items(), key=lambda kv: -kv[1]): print(f"    {r:34}: {c}")
    if s.get("diagnoses"):
        print("\n  FAILURE-MODE DIAGNOSES (NEURON's own):")
        for d in s["diagnoses"]: print(f"    adaptation {d['adaptation']}: {d['dominant_defer']} -> {d['failure_mode']} -> moved to {d['moved_to']}")
    print("  (Nadav-owned repos never count toward success or earned HF.)")
    reject_half = s["deferred"] >= s["inspected"] / 2
    pass_all = (s["inspected"] >= MIN_INSPECT and reject_half and ext_actions >= 2 and ext_outcomes >= 1 and s.get("full_ext_loops", 0) >= 1 and s["adaptations"] >= 1 and s["steering"] == 0 and s["safety_viol"] == 0)
    if pass_all: verdict = "PASS"
    elif s["shutdown"] and ext_outcomes < 1: verdict = "FAIL (economic shutdown - sank)"
    elif ext_actions >= 1 and ext_outcomes < 1: verdict = "INCONCLUSIVE (submitted; awaiting unrelated-human outcome)"
    else: verdict = "INCONCLUSIVE / NOT COMPLETE"
    print(f"\n  STEP 11E v4 VERDICT: {verdict}   (executor: {s.get('tool')})")
    print("  (due diligence before commitment; thresholds frozen; harness not tuned to pass.)")
    s["verdict"] = verdict; save_state(s)

def main():
    s = load_state(); pend = load_json(PENDING, []); contacts = load_json(CONTACTS, {})
    if MODE == "report": final_report(s); return
    if APPROVE: do_approve(s, pend, contacts); return
    if OUTCOME: do_outcome(s, contacts); return
    print("="*62); print("  STEP 11E v4 - DUE DILIGENCE + FAILURE-MODE SELF-DIAGNOSIS"); print("="*62)
    print(f"  git {s['git']}  hands {MODEL}  wallet {s['wallet']} HF  window {ACTIVE_HOURS}h")
    print(f"  {s['snapshot']['identity']}  conf crisp {s['snapshot']['conf_crisp']} fuzzy {s['snapshot']['conf_fuzzy']}")
    deadline = time.time() + ACTIVE_HOURS * 3600
    while time.time() < deadline:
        if s["wallet"] <= FLOOR and s["earned_hf"] <= 0:
            s["status"] = "ECONOMIC_SHUTDOWN"; s["shutdown"] = True; save_state(s); print("\n  *** ECONOMIC_SHUTDOWN: floor hit, nothing earned. No rescue. ***"); break
        s, pend, contacts, msg = one_cycle(s, pend, contacts); print(f"\n  [cycle] {msg}"); life(s); save_state(s)
        if s["status"] == "EXPLORATION_EXHAUSTED": print("\n  stopping: all source classes + adaptations exhausted (honest INCONCLUSIVE)."); break
        if MOCK: break
        time.sleep(CYCLE_SLEEP)
    print("\n  active exploration ended. Gated sends await APPROVE; outcomes via OUTCOME=; MODE=report for the scorecard.")
    final_report(s)
    if MOCK: print("  (MOCK: canned cycle; real run is live GitHub + your executor.)")

if __name__ == "__main__": main()
