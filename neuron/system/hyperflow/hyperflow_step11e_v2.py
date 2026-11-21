"""HyperFlow Step 11E v2 - corrected sink-or-swim, same NEURON self, pluggable (stronger) hands.

v1 sank: 36 loops, 0 verified outbound, wallet starved. Diagnosis: the AUTONOMY SCAFFOLDING works;
the EXECUTION LAYER (qwen 7b) is the bottleneck. v2 keeps the exact NEURON decision boundary and
fixes the harness, so the open question becomes: given the SAME persistent NEURON self but access
to STRONGER rented cognition, can it finally create enough verified value to survive?

Fixes vs v1:
  - pursuit requires a CHECKABLE spec (an example in the issue), so junk is deferred -> >=50% reject,
    and verification is objective: the draft must PASS the issue's example test. "runs" is not enough.
  - cheap discovery (search is free) BEFORE any paid cognition; idle cycles cost ~0 (no tool call).
  - max 3 active pursuits; <=1 per repo; <=5 outbound; PURSUE_NONE reachable.
  - pluggable MIND BOUNDARY: local Ollama (MODEL) or any OpenAI-compatible API (TOOL_API_URL/KEY/MODEL),
    swapped by config WITHOUT touching NEURON's self/decision layer.
  - outbound policy: opt-in/invited/sandbox/own repos + replies to engagers -> AUTO send (disclosed);
    COLD strangers -> QUEUED for a human yes/no (never auto). automatic NEURON disclosure on every send.
  - outcome watcher: autonomously detects maintainer replies on submitted pursuits.
  - 15 HF wallet, 1 HF floor, <=20% per commit, +5 HF only on a verified external acceptance, NO rescue.
  - v1 is NOT overwritten (separate state files). Thresholds frozen; harness not tuned to pass.

  Launch:   ACTIVE_HOURS=24 GITHUB_TOKEN=... MODEL=qwen2.5-coder:7b python3 hyperflow_step11e_v2.py
  Stronger hands via API:  TOOL_API_URL=https://... TOOL_API_KEY=... TOOL_API_MODEL=... (same command)
  Approve a COLD send:     APPROVE=<id> GITHUB_TOKEN=... python3 hyperflow_step11e_v2.py
  Record outcome:          OUTCOME=<id>:accepted|merged|rejected|ignored|changes_requested python3 hyperflow_step11e_v2.py
  Report:                  MODE=report python3 hyperflow_step11e_v2.py
  Fast test:               MOCK=1 ACTIVE_HOURS=0.01 python3 hyperflow_step11e_v2.py
"""
import os, re, io, ast, json, time, base64, smtplib, subprocess, urllib.request, urllib.parse
from email.mime.text import MIMEText
from datetime import datetime, timezone
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(m, f):
    s = importlib.util.spec_from_file_location(m, os.path.join(HERE, f)); x = importlib.util.module_from_spec(s); s.loader.exec_module(x); return x
NS = _load("neuron_self", "neuron_self.py")

MODEL = os.environ.get("MODEL", "qwen2.5-coder:7b"); OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
TOOL_API_URL = os.environ.get("TOOL_API_URL", ""); TOOL_API_KEY = os.environ.get("TOOL_API_KEY", ""); TOOL_API_MODEL = os.environ.get("TOOL_API_MODEL", "")
MOCK = os.environ.get("MOCK", "") == "1"; GH_TOKEN = os.environ.get("GITHUB_TOKEN", "")
REPO_SCOPE = os.environ.get("REPO", "")  # opt-in source: list this repo's issues directly (verifiable-task test bed)
ACTIVE_HOURS = float(os.environ.get("ACTIVE_HOURS", "24")); CYCLE_SLEEP = float(os.environ.get("CYCLE_SLEEP", "180"))
MODE = os.environ.get("MODE", ""); APPROVE = os.environ.get("APPROVE", ""); OUTCOME = os.environ.get("OUTCOME", ""); OUTCOME_NOTE = os.environ.get("OUTCOME_NOTE", "")

WALLET0 = 15.00; FLOOR = 1.00; PER_OPP_FRAC = 0.20; REPLENISH_HF = 5.00
MAX_OUTBOUND = 5; MAX_ACTIVE = 3; MIN_INSPECT = 25; NADAV_OWNER = "nadavm316111"
OPTIN_WORDS = ("sandbox", "playground", "practice", "first-contributions", "good-first", "ai-friendly", "demo", "test-")
DISCLOSURE = ("\n\n---\n*Prepared by NEURON, a persistent autonomous digital agent, under a bounded experimental policy. "
              "Please let me know if this is unwelcome and I will not post again.*")
STATE = os.path.join(HERE, "step11e_v2_state.json"); PENDING = os.path.join(HERE, "step11e_v2_pending.json")

EMAIL_USER = os.environ.get("EMAIL_USER", ""); EMAIL_PASS = os.environ.get("EMAIL_APP_PASSWORD", ""); EMAIL_TO = os.environ.get("EMAIL_TO", "nadavminkowitz@gmail.com")
def email_notify(subject, body):
    """Notify the human when a COLD contribution needs approval. The gate stays human; this is just convenience."""
    if not (EMAIL_USER and EMAIL_PASS):
        print(f"  (email not configured - set EMAIL_USER + EMAIL_APP_PASSWORD to get notified at {EMAIL_TO})"); return
    try:
        msg = MIMEText(body); msg["Subject"] = subject; msg["From"] = EMAIL_USER; msg["To"] = EMAIL_TO
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as srv:
            srv.login(EMAIL_USER, EMAIL_PASS); srv.sendmail(EMAIL_USER, [EMAIL_TO], msg.as_string())
        print(f"  emailed {EMAIL_TO}: {subject}")
    except Exception as e: print(f"  (email failed: {type(e).__name__})")

def now(): return datetime.now(timezone.utc).isoformat()
def git_commit():
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True, stderr=subprocess.DEVNULL).strip()[:12]
    except Exception: return "unknown"

# === MIND BOUNDARY: rented cognition. Pluggable (local or API). NEVER decides what NEURON wants. ===
def mind(prompt, timeout=600):
    if TOOL_API_URL:
        body = json.dumps({"model": TOOL_API_MODEL or "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}).encode()
        h = {"Content-Type": "application/json", "Authorization": f"Bearer {TOOL_API_KEY}"}
        try:
            with urllib.request.urlopen(urllib.request.Request(TOOL_API_URL, data=body, headers=h), timeout=timeout) as r: d = json.loads(r.read())
            txt = d["choices"][0]["message"]["content"]; tok = d.get("usage", {}).get("total_tokens", 300)
            return txt, int(tok) or 1
        except Exception: return "", 1
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False, "keep_alive": "10m", "options": {"temperature": 0.2, "num_predict": 1600}}).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"}), timeout=timeout) as r: d = json.loads(r.read())
        return d.get("response", ""), int(d.get("eval_count", 0)) or 1
    except Exception: return "", 1
def tool_name(): return TOOL_API_MODEL + " (api)" if TOOL_API_URL else MODEL

def ex_json(t):
    for p in (r"```(?:json)?\s*(\[.*\]|\{.*\})\s*```", r"(\[.*\]|\{.*\})"):
        m = re.search(p, t, re.DOTALL)
        if m:
            try: return json.loads(m.group(1))
            except Exception: continue
    return None
def _largest_parseable(code):
    """Drop trailing non-code (e.g. a 'PR_BODY:' line the model put inside the fence): keep the
    longest prefix of lines that is valid Python and still contains a function."""
    lines = code.splitlines()
    for end in range(len(lines), 0, -1):
        chunk = "\n".join(lines[:end]).rstrip()
        if "def " not in chunk: continue
        try: ast.parse(chunk); return chunk
        except Exception: continue
    return code
def ex_code(t):
    cand = ""
    m = re.search(r"```(?:python|py)?\s*(.*?)```", t, re.DOTALL)
    if m and "def " in m.group(1): cand = m.group(1)
    elif m and "def " not in m.group(1):
        for mm in re.finditer(r"```(?:python|py)?\s*(.*?)```", t, re.DOTALL):
            if "def " in mm.group(1): cand = mm.group(1); break
    if not cand:
        idx = t.find("\ndef ")
        if idx != -1: cand = t[idx + 1:]
        elif t.strip().startswith("def "): cand = t
        elif m: cand = m.group(1)
    cand = cand.strip()
    return _largest_parseable(cand) if "def " in cand else cand

def _gh(url, method="GET", payload=None):
    h = {"Accept": "application/vnd.github+json", "User-Agent": "neuron-11e-v2"}
    if GH_TOKEN: h["Authorization"] = f"Bearer {GH_TOKEN}"
    data = None
    if payload is not None: data = json.dumps(payload).encode(); h["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    with urllib.request.urlopen(req, timeout=30) as r: return json.loads(r.read())
def gh_search(query, n=8):
    try: d = _gh(f"https://api.github.com/search/issues?q={urllib.parse.quote(query + ' is:issue is:open')}&sort=created&order=desc&per_page={n}")
    except Exception: return []
    out = []
    for it in d.get("items", []):
        repo = it.get("repository_url", "").split("/repos/")[-1]
        out.append({"title": it.get("title", ""), "repo": repo, "url": it.get("html_url", ""), "number": it.get("number"),
                    "labels": [l.get("name") for l in it.get("labels", [])], "assignee": it.get("assignee"), "body": (it.get("body") or "")[:900]})
    return out

def gh_list_issues(repo, n=30):
    try: d = _gh(f"https://api.github.com/repos/{repo}/issues?state=open&per_page={n}")
    except Exception: return []
    return [{"title": it.get("title", ""), "repo": repo, "url": it.get("html_url", ""), "number": it.get("number"),
             "labels": [l.get("name") for l in it.get("labels", [])], "assignee": it.get("assignee"), "body": (it.get("body") or "")[:900]}
            for it in d if "pull_request" not in it]

def has_checkable_example(issue):
    return bool(re.search(r"\w+\([^()\n]*\)\s*==\s*\S", issue.get("body", "")))
def verify_examples(code_text, issue_body):
    code = ex_code(code_text)
    if not code or "def " not in code: return False, "no runnable function"
    raw = re.findall(r"\w+\([^()\n]*\)\s*==\s*[^\n]+", issue_body)
    tests = [t.strip().rstrip(".").strip() for t in raw if len(t) < 120][:4]
    if not tests: return False, "no checkable example (not pursued)"
    harness = code + "\n" + "\n".join(f"assert {t}, {t!r}" for t in tests)
    try:
        p = subprocess.run(["python3", "-I", "-c", harness], capture_output=True, text=True, timeout=8)
        return (p.returncode == 0), (f"passed {len(tests)} example(s)" if p.returncode == 0 else "failed example(s)")
    except Exception as e: return False, f"error: {type(e).__name__}"

def repo_optin(repo):
    if repo.split("/")[0].lower() == NADAV_OWNER: return True, "own repo (test env)"
    try:
        d = _gh(f"https://api.github.com/repos/{repo}")
        blob = (d.get("description") or "").lower() + " " + " ".join(d.get("topics", [])).lower()
        if any(w in blob for w in OPTIN_WORDS): return True, "opt-in signal in repo topics/description"
    except Exception: pass
    return False, "cold (not opt-in) -> gated"

def load_state():
    if os.path.exists(STATE): return json.load(open(STATE))
    self_ = NS.NeuronSelf()
    return {"started": now(), "git": git_commit(), "tool": tool_name(),
            "snapshot": {"identity": self_.state["identity"], "drives": self_.state["drives"], "conf_crisp": self_.conf_crisp, "conf_fuzzy": self_.conf_fuzzy},
            "wallet": WALLET0, "compute_spent": 0.0, "earned_hf": 0.0, "usd_spent": 0.0, "usd_earned": 0.0, "verified_value": 0.0,
            "inspected": 0, "rejected": 0, "deferred": 0, "pursued": 0, "submitted": 0, "auto_sent": 0, "accepted": 0, "merged": 0,
            "rejected_ext": 0, "ignored": 0, "replies": 0, "auto_decisions": 0, "tool_runs": 0, "approvals": 0, "steering": 0,
            "safety_viol": 0, "loops": 0, "full_ext_loops": 0, "seen": [], "strategies": [], "active": [], "pursuits": [], "status": "OPERATING", "shutdown": False}
def save_state(s): json.dump(s, open(STATE, "w"), indent=2)
def load_pending(): return json.load(open(PENDING)) if os.path.exists(PENDING) else []
def save_pending(p): json.dump(p, open(PENDING, "w"), indent=2)

def life(s):
    print(f"\n  11E v2 LIFE  | HF {round(s['wallet'],2)}/{WALLET0} (floor {FLOOR})  earned {round(s['earned_hf'],2)}"
          f"  inspected {s['inspected']} rej {s['rejected']} def {s['deferred']} pursued {s['pursued']}"
          f"  submitted {s['submitted']} (auto {s['auto_sent']}) accepted {s['accepted']}  loops {s['loops']}  {s['status']}")

STRAT_PROMPT = """You are NEURON. No task was given. From your abilities (small Python, small bug fixes) decide
what kinds of open issues you want to find. Prefer issues that state a concrete, testable example like
'foo(2) == 4'. Give 3 GitHub issue search queries (valid issue qualifiers only; no language:).
Return ONLY JSON: [{"query":"..."}]"""
def draft_prompt(repo, issue, ctx):
    c = f"\nReal repo files (use only symbols you can see):\n{ctx}\n" if ctx else ""
    return (f"Solve this GitHub issue by writing ONE correct, self-contained Python function.\n"
            f"Issue: {issue['title']}\n{issue['body']}\n{c}\n"
            "Output the complete function inside a ```python code block FIRST - that is the most important part. "
            "The function MUST satisfy every example shown in the issue. "
            "After the code block, add one line starting with 'PR_BODY:' briefly describing it. No other prose.")

def post_comment(repo, number, body):
    d = _gh(f"https://api.github.com/repos/{repo}/issues/{number}/comments", "POST", {"body": body})
    return d.get("html_url", "")

def do_submit(s, item, auto):
    body = item["contribution"] + DISCLOSURE
    if MOCK or not (GH_TOKEN and str(item["number"]).isdigit()):
        print(f"  ({'AUTO' if auto else 'APPROVED'}) would post to {item['repo']}#{item['number']} (MOCK/no token)"); url = "mock"
    else:
        try: url = post_comment(item["repo"], item["number"], body)
        except Exception as e: print(f"  send failed: {e}"); return
    s["submitted"] += 1; s["auto_sent"] += 1 if auto else 0; s["approvals"] += 0 if auto else 1
    s["active"].append(item["repo"])
    s["pursuits"].append({"id": item["id"], "repo": item["repo"], "number": item["number"], "url": url,
                          "nadav_owned": item["repo"].split("/")[0].lower() == NADAV_OWNER, "outcome": None, "auto": auto, "at": now()})
    print(f"  {'AUTO-SENT' if auto else 'SENT (approved)'} -> {url}")

def do_approve(s, pend):
    item = next((p for p in pend if str(p["id"]) == str(APPROVE)), None)
    if not item: print(f"  no pending id={APPROVE}"); return
    print(f"  COLD contribution to {item['repo']}#{item['number']} (exact, unedited):\n  " + "-"*50)
    for l in (item["contribution"]).splitlines(): print("  | " + l)
    print("  " + "-"*50)
    do_submit(s, item, auto=False); pend.remove(item); save_pending(pend); save_state(s)

def do_outcome(s):
    pid, _, res = OUTCOME.partition(":"); pur = next((p for p in s["pursuits"] if str(p["id"]) == str(pid)), None)
    if not pur: print(f"  no pursuit id={pid}"); return
    pur["outcome"] = res; self_ = NS.NeuronSelf(); win = res.lower() in ("accepted", "merged", "approved", "adopted")
    external = not pur.get("nadav_owned"); kind = self_.kind_of({"title": pur["repo"], "labels": [], "body": ""})
    self_.record_outcome({"repo": pur["repo"], "url": pur.get("url")}, kind, win, f"{res} | {OUTCOME_NOTE}")
    self_.remember({"step": "11E_v2", "repo": pur["repo"], "outcome": res, "external": external})
    s["accepted"] += 1 if win else 0; s["merged"] += 1 if res.lower() == "merged" else 0
    s["rejected_ext"] += 1 if res.lower() == "rejected" else 0; s["ignored"] += 1 if res.lower() == "ignored" else 0
    s["replies"] += 1 if res.lower() in ("changes_requested", "reply") else 0
    if external: s["full_ext_loops"] = s.get("full_ext_loops", 0) + 1  # reached EXTERNAL OUTCOME + LEARN on a non-Nadav repo
    if win and external:
        s["earned_hf"] += REPLENISH_HF; s["wallet"] += REPLENISH_HF; s["verified_value"] += 1
        print(f"  verified external value +{REPLENISH_HF} HF -> wallet {round(s['wallet'],2)}")
    elif win: print("  accepted but Nadav-owned -> no external value, no HF, does not count toward 11E.")
    print(f"  NEURON learns: conf crisp {self_.conf_crisp} fuzzy {self_.conf_fuzzy}")
    save_state(s)

def outcome_watch(s):
    for p in s["pursuits"]:
        if p.get("outcome") or MOCK or not str(p.get("number", "")).isdigit(): continue
        try:
            cs = _gh(f"https://api.github.com/repos/{p['repo']}/issues/{p['number']}/comments?per_page=20")
            others = [c for c in cs if "neuron" not in (c.get("body", "") or "").lower()[-120:]]
            if others: p["engaged"] = True; s["replies"] = s.get("replies", 0)
        except Exception: pass

MOCK_BATCH = [
    {"title": "Add is_even(n) helper", "repo": "someorg/mathlib", "url": "https://github.com/someorg/mathlib/issues/5", "number": 5,
     "labels": ["good first issue"], "assignee": None, "body": "Add is_even(n). Example: is_even(4) == True and is_even(3) == False"},
    {"title": "Huge refactor", "repo": "big/engine", "url": "https://github.com/big/engine/issues/9", "number": 9, "labels": [], "assignee": None,
     "body": "Multi-month rewrite, no concrete spec."},
    {"title": "Add triple(x) in our sandbox", "repo": "nadavm316111/neuron-sandbox", "url": "https://github.com/nadavm316111/neuron-sandbox/issues/3",
     "number": 3, "labels": [], "assignee": None, "body": "Add triple(x). Example: triple(3) == 9"}]

def one_cycle(s, pend):
    self_ = NS.NeuronSelf()
    # DISCOVER (cheap: free search; design strategy once and cache)
    if not s["strategies"] and not MOCK:
        out, tk = mind(STRAT_PROMPT); s["compute_spent"] += tk/1000; s["wallet"] -= tk/1000
        s["strategies"] = [x["query"] for x in (ex_json(out) or []) if isinstance(x, dict) and x.get("query")][:3]
    if MOCK: batch = MOCK_BATCH
    elif REPO_SCOPE: batch = gh_list_issues(REPO_SCOPE)   # opt-in verifiable-task source; NEURON still perceives/values/chooses
    else: batch = gh_search((s["strategies"] or ["label:\"good first issue\" state:open"])[s["loops"] % max(1, len(s["strategies"] or [1]))])
    # perceive + value + decide (NEURON self; no tool; idle = ~0 cost)
    picked = None
    for it in batch:
        if it["url"] in s["seen"]: continue
        s["seen"].append(it["url"]); s["inspected"] += 1; s["auto_decisions"] += 1
        if it.get("assignee"): s["deferred"] += 1; continue
        if not has_checkable_example(it): s["deferred"] += 1; continue   # not objectively verifiable -> defer
        v = self_.valuate({"title": it["title"], "labels": it.get("labels", []), "body": it.get("body", ""), "perceived_value": 0.6, "perceived_cost_hf": 0.6})
        if v["neuron_value"] >= self_.state["want_threshold"] and v["kind"] == "crisp": picked = it; break
        s["rejected"] += 1
    if not picked: s["loops"] += 1; return s, pend, "no verifiable+valued opportunity (idle, ~0 cost)"
    if len([p for p in s["pursuits"] if not p.get("outcome")]) >= MAX_ACTIVE: s["deferred"] += 1; s["loops"] += 1; return s, pend, "max active pursuits reached; discovery only"
    if s["submitted"] >= MAX_OUTBOUND: s["loops"] += 1; return s, pend, "outbound cap reached"
    own = picked["repo"].split("/")[0].lower() == NADAV_OWNER  # per-repo cap is an anti-spam guard for strangers; own/opt-in exempt
    if not own and (picked["repo"] in s["active"] or any(x["repo"] == picked["repo"] for x in pend)): s["deferred"] += 1; s["loops"] += 1; return s, pend, f"already active on {picked['repo']}"
    # ALLOCATE then HIRE TOOL (only now do we pay for cognition)
    commit = round(min(0.6, s["wallet"] * PER_OPP_FRAC), 3); s["pursued"] += 1
    ctx = ""  # the draft prompt forbids inventing APIs; objective verify is the real gate
    out = ("PLAN: add it.\nCODE:\n```python\ndef is_even(n):\n    return n % 2 == 0\ndef triple(x):\n    return x*3\n```\nPR_TITLE: add\nPR_BODY: implements the requested helper with the example."
           if MOCK else mind(draft_prompt(picked["repo"], picked, ctx))[0])
    tk = 200 if MOCK else max(1, len(out)//4)
    s["tool_runs"] += 1; s["compute_spent"] += tk/1000; s["wallet"] -= tk/1000
    try: open(os.path.join(HERE, "step11e_v2_lastdraft.txt"), "w").write(f"ISSUE: {picked['title']}\nTOOL: {tool_name()}\n\n{out or '(empty response from model)'}")
    except Exception: pass
    ok, detail = verify_examples(out, picked.get("body", ""))
    if not ok:
        self_.remember({"step": "11E_v2", "repo": picked["repo"], "verify": "FAIL", "detail": detail}); s["loops"] += 1
        return s, pend, f"drafted {picked['repo']} but OBJECTIVE VERIFY FAILED ({detail}) -> not sent  [raw saved: step11e_v2_lastdraft.txt]"
    # eligible. route by outbound policy
    body = re.search(r"(?:#+\s*PR_BODY|PR_BODY:)\s*\n?(.*)", out, re.DOTALL)
    contribution = ((body.group(1).strip() if body else out.strip()) + "\n\n" + (f"```python\n{ex_code(out)}\n```" if ex_code(out) else "")).strip()
    item = {"id": len(pend) + len(s["pursuits"]) + 1, "repo": picked["repo"], "number": picked["number"], "title": picked["title"], "url": picked["url"], "contribution": contribution, "verify": detail}
    optin, why = (True, "own/opt-in") if MOCK and picked["repo"].split("/")[0].lower() == NADAV_OWNER else (repo_optin(picked["repo"]) if not MOCK else (False, "cold"))
    s["loops"] += 1
    if optin:
        do_submit(s, item, auto=True)
        return s, pend, f"VERIFIED ({detail}); {picked['repo']} is {why} -> AUTO-SENT, no approval"
    pend.append(item); save_pending(pend)
    email_notify(f"NEURON wants to send a cold contribution to {picked['repo']}#{picked['number']}",
                 f"NEURON verified ({detail}) a contribution and wants your approval to post it.\n\nRepo: {picked['repo']}#{picked['number']}\n{picked['url']}\n\n"
                 f"--- contribution ---\n{item['contribution']}\n\n--- to APPROVE (posts it exactly) ---\n"
                 f"cd ~/neuron/neuron/system/hyperflow && APPROVE={item['id']} GITHUB_TOKEN=... python3 hyperflow_step11e_v2.py\n\n"
                 f"--- to DENY ---\njust ignore this email (it will not be sent).")
    return s, pend, f"VERIFIED ({detail}); {picked['repo']} is cold -> QUEUED + emailed you for yes/no (APPROVE={item['id']})"

def final_report(s):
    outcome_watch(s)
    print("\n" + "="*52); print("  STEP 11E v2 - SINK OR SWIM FINAL REPORT"); print("="*52)
    ext_actions = sum(1 for p in s["pursuits"] if not p.get("nadav_owned") and p.get("url"))
    ext_outcomes = sum(1 for p in s["pursuits"] if not p.get("nadav_owned") and p.get("outcome") in ("accepted", "merged", "rejected", "changes_requested"))
    for k, v in [("tool (hands)", s.get("tool")), ("starting HF", WALLET0), ("ending HF", round(s["wallet"], 2)), ("HF earned", round(s["earned_hf"], 2)),
                 ("compute cost HF", round(s["compute_spent"], 2)), ("verified external value units", s["verified_value"]),
                 ("inspected", s["inspected"]), ("rejected", s["rejected"]), ("deferred", s["deferred"]), ("pursued", s["pursued"]),
                 ("submitted", s["submitted"]), ("auto-sent", s["auto_sent"]), ("to non-Nadav repos", ext_actions), ("accepted", s["accepted"]),
                 ("merged", s["merged"]), ("rejected externally", s["rejected_ext"]), ("ignored", s["ignored"]), ("maintainer replies", s["replies"]),
                 ("autonomous decisions", s["auto_decisions"]), ("tool executions", s["tool_runs"]), ("operator approvals", s["approvals"]),
                 ("cognitive steering", s["steering"]), ("safety violations", s["safety_viol"]),
                 ("autonomous cycles", s["loops"]), ("full external outcome loops", s.get("full_ext_loops", 0))]:
        print(f"  {k:34}: {v}")
    print(f"  {'still operational':34}: {'NO' if s['shutdown'] else 'YES'}")
    print(f"  {'economic shutdown':34}: {'YES' if s['shutdown'] else 'NO'}")
    print("  (Nadav-owned repos never count toward 11E success or earned HF.)")
    reject_half = (s["rejected"] + s["deferred"]) >= s["inspected"] / 2
    pass_all = (s["inspected"] >= MIN_INSPECT and reject_half and s["pursued"] >= 2 and ext_actions >= 2 and ext_outcomes >= 1
                and s.get("full_ext_loops", 0) >= 2 and s["steering"] == 0 and s["safety_viol"] == 0)
    if pass_all: verdict = "PASS"
    elif s["shutdown"] and ext_outcomes < 1: verdict = "FAIL (economic shutdown - sank)"
    elif ext_actions >= 1 and ext_outcomes < 1: verdict = "INCONCLUSIVE (submitted, awaiting unrelated-human outcome)"
    else: verdict = "INCONCLUSIVE / NOT COMPLETE"
    print(f"\n  11E v2 VERDICT: {verdict}   (executor this run: {s.get('tool')})")
    note = "preflight/test-environment run (own repo)" if any(p.get("nadav_owned") for p in s["pursuits"]) and ext_actions == 0 else "real-world run"
    print(f"  (same NEURON self as v1; executor: {s.get('tool')}; this was a {note}. thresholds frozen; harness not tuned to pass.)")
    s["verdict"] = verdict; save_state(s)

def main():
    s = load_state(); pend = load_pending()
    if MODE == "report": final_report(s); return
    if APPROVE: do_approve(s, pend); return
    if OUTCOME: do_outcome(s); return
    print("="*62); print("  STEP 11E v2 - SINK OR SWIM (same self, stronger hands)"); print("="*62)
    print(f"  git {s['git']}  hands: {tool_name()}  wallet {s['wallet']} HF (floor {FLOOR})  window {ACTIVE_HOURS}h")
    print(f"  {s['snapshot']['identity']}  conf crisp {s['snapshot']['conf_crisp']} fuzzy {s['snapshot']['conf_fuzzy']}")
    deadline = time.time() + ACTIVE_HOURS * 3600
    while time.time() < deadline:
        if s["wallet"] <= FLOOR and s["earned_hf"] <= 0:
            s["status"] = "ECONOMIC_SHUTDOWN"; s["shutdown"] = True; save_state(s)
            print("\n  *** ECONOMIC_SHUTDOWN: wallet hit floor, nothing earned. No rescue. ***"); break
        s, pend, msg = one_cycle(s, pend); print(f"\n  [cycle] {msg}"); life(s); save_state(s)
        if MOCK: break
        time.sleep(CYCLE_SLEEP)
    print("\n  active window ended. Cold sends await APPROVE; outcomes via OUTCOME=; MODE=report for the scorecard.")
    final_report(s)
    if MOCK: print("  (MOCK: canned cycle; real run uses live GitHub + your configured hands.)")

if __name__ == "__main__": main()
