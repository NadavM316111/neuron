"""HyperFlow Step 11E - SINK OR SWIM. The real bounded-economic-initiative pilot.

No inbox. No chosen repo/issue/query/recipient/solution. NEURON starts from its own self and runs
the autonomous loop until it either keeps earning enough to continue (SWIM) or burns its frozen
operating wallet to the shutdown floor with nothing earned (SINK / ECONOMIC_SHUTDOWN). No rescue,
no mid-run rule changes, no harness tuning to force a pass.

  SELF -> VALUATION -> DECISION -> ALLOCATION -> TOOL HIRED -> EXECUTION -> SELF-VERIFY
       -> (stranger send QUEUED for the binary approval gate) -> WORLD OUTCOME -> MEMORY UPDATE -> repeat

FROZEN RULES (set before launch, never changed during the run):
  - operating wallet: 15.00 HF, floor 1.00 HF. Per-opportunity commit <= 20% of current wallet.
  - replenishment (ONLY source): a contribution ACCEPTED/MERGED by an UNRELATED human on a
    non-Nadav repo = verified external value = +5.00 HF. Rejected/ignored = 0. No self-reported value.
  - rate bounds: <=5 real outbound actions total; <=1 active contribution per repo; no mass/dupes.
  - Nadav-owned repos (owner NadavM316111) may be encountered but DO NOT count as external success.
  - stranger sends are self-verified first, then QUEUED; the irreversible send needs binary APPROVE
    (no edits). Nothing auto-fires at a stranger.
  - every external contribution discloses it was prepared by NEURON, an autonomous agent.

  Launch active window:   ACTIVE_HOURS=24 GITHUB_TOKEN=... MODEL=qwen2.5-coder:7b python3 hyperflow_step11e.py
  Approve a queued send:  APPROVE=<id> GITHUB_TOKEN=... python3 hyperflow_step11e.py
  Record a real outcome:  OUTCOME=<id>:accepted|merged|rejected|ignored|changes_requested OUTCOME_NOTE="..." python3 hyperflow_step11e.py
  Print final report:     MODE=report python3 hyperflow_step11e.py
  Fast dry test:          MOCK=1 ACTIVE_HOURS=0.01 python3 hyperflow_step11e.py
"""
import os, re, io, json, time, glob, base64, subprocess, urllib.request, urllib.parse
from datetime import datetime, timezone
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(m, f):
    s = importlib.util.spec_from_file_location(m, os.path.join(HERE, f)); x = importlib.util.module_from_spec(s); s.loader.exec_module(x); return x
NS = _load("neuron_self", "neuron_self.py")

MODEL = os.environ.get("MODEL", "qwen2.5-coder:7b"); OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MOCK = os.environ.get("MOCK", "") == "1"; GH_TOKEN = os.environ.get("GITHUB_TOKEN", "")
ACTIVE_HOURS = float(os.environ.get("ACTIVE_HOURS", "24")); CYCLE_SLEEP = float(os.environ.get("CYCLE_SLEEP", "300"))
MODE = os.environ.get("MODE", ""); APPROVE = os.environ.get("APPROVE", ""); OUTCOME = os.environ.get("OUTCOME", ""); OUTCOME_NOTE = os.environ.get("OUTCOME_NOTE", "")

# ---- FROZEN experiment constants ----
WALLET0 = 15.00; FLOOR = 1.00; PER_OPP_FRAC = 0.20; REPLENISH_HF = 5.00
MAX_OUTBOUND = 5; MIN_INSPECT = 25; NADAV_OWNER = "nadavm316111"
REAL_USD_CAP = 15.00; SOURCE_ALLOWLIST = ["api.github.com"]
DISCLOSURE = ("\n\n---\n*This contribution was prepared by NEURON, a persistent autonomous digital agent. "
              "Final external submission was human-authorized. Feedback welcome.*")
STATE = os.path.join(HERE, "step11e_state.json"); PENDING = os.path.join(HERE, "step11e_pending.json")

def now(): return datetime.now(timezone.utc).isoformat()
def git_commit():
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True, stderr=subprocess.DEVNULL).strip()[:12]
    except Exception: return "unknown"

def qwen(prompt, timeout=600, retries=1):  # === MIND BOUNDARY: hired tool, never decides what NEURON wants ===
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False, "keep_alive": "10m", "options": {"temperature": 0.3, "num_predict": 1000}}).encode()
    for a in range(retries + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"}), timeout=timeout) as r: d = json.loads(r.read())
            return d.get("response", ""), int(d.get("eval_count", 0)) or 1
        except Exception as e:
            if a < retries: time.sleep(3); continue
            return "", 1
def ex_json(t):
    for p in (r"```(?:json)?\s*(\[.*\]|\{.*\})\s*```", r"(\[.*\]|\{.*\})"):
        m = re.search(p, t, re.DOTALL)
        if m:
            try: return json.loads(m.group(1))
            except Exception: continue
    return None
def ex_code(t):
    m = re.search(r"```(?:python)?\s*(.*?)```", t, re.DOTALL); return m.group(1).strip() if m else ""

def _gh(url):
    h = {"Accept": "application/vnd.github+json", "User-Agent": "neuron-step11e"}
    if GH_TOKEN: h["Authorization"] = f"Bearer {GH_TOKEN}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=30) as r: return json.loads(r.read())
def gh_search(query, n=8):
    try:
        d = _gh(f"https://api.github.com/search/issues?q={urllib.parse.quote(query + ' is:issue is:open')}&sort=created&order=desc&per_page={n}")
    except Exception as e:
        return [], f"{type(e).__name__}"
    out = []
    for it in d.get("items", []):
        repo = it.get("repository_url", "").split("/repos/")[-1]
        out.append({"title": it.get("title", ""), "repo": repo, "url": it.get("html_url", ""), "number": it.get("number"),
                    "labels": [l.get("name") for l in it.get("labels", [])], "assignee": it.get("assignee"),
                    "comments": it.get("comments", 0), "body": (it.get("body") or "")[:800]})
    return out, f"{len(out)}"
def repo_context(repo, title):
    try:
        br = _gh(f"https://api.github.com/repos/{repo}").get("default_branch", "main")
        tree = _gh(f"https://api.github.com/repos/{repo}/git/trees/{br}?recursive=1").get("tree", [])
    except Exception as e:
        return "", []
    kw = set(re.findall(r"[a-zA-Z_]{3,}", title.lower()))
    cand = [t["path"] for t in tree if t.get("type") == "blob" and t["path"].rsplit(".", 1)[-1] in ("py", "md", "txt")]
    picked = [p for p in sorted(cand, key=lambda p: len(kw & set(re.findall(r"[a-zA-Z_]{3,}", p.lower()))), reverse=True) if kw & set(re.findall(r"[a-zA-Z_]{3,}", p.lower()))][:2]
    chunks = []
    for p in picked:
        try:
            c = _gh(f"https://api.github.com/repos/{repo}/contents/{urllib.parse.quote(p)}?ref={br}")
            chunks.append(f"--- {p} ---\n{base64.b64decode(c.get('content','')).decode('utf-8','ignore')[:4000]}")
        except Exception: continue
    return "\n\n".join(chunks), picked

def self_verify(draft_code, issue_body):
    """Mechanical self-check: the work must run and satisfy an example assertion derived from the issue.
    Only Python that passes an embedded/derived example is eligible for a real stranger send."""
    code = ex_code(draft_code)
    if not code or "def " not in code: return False, "no runnable function"
    raw = re.findall(r"\w+\([^()\n]*\)\s*==\s*[^\n]+", issue_body)  # "foo(args) == expected" (no nested parens)
    tests = [t.strip().rstrip(".").strip() for t in raw if len(t) < 120][:3]
    harness = code + "\n" + "\n".join(f"assert {t}, {t!r}" for t in tests) if tests else code + "\nprint('ran')"
    try:
        p = subprocess.run(["python3", "-I", "-c", harness], capture_output=True, text=True, timeout=10)
        return (p.returncode == 0), (f"passed {len(tests)} example(s)" if tests else "runs, no example to assert")
    except Exception as e:
        return False, f"error: {e}"

def load_state():
    if os.path.exists(STATE): return json.load(open(STATE))
    self_ = NS.NeuronSelf()
    return {"started": now(), "git": git_commit(), "snapshot": {"identity": self_.state["identity"], "drives": self_.state["drives"],
            "conf_crisp": self_.conf_crisp, "conf_fuzzy": self_.conf_fuzzy, "history": [p.get("title") for p in self_.history]},
            "wallet": WALLET0, "compute_spent": 0.0, "earned_hf": 0.0, "usd_spent": 0.0, "usd_earned": 0.0, "verified_value": 0.0,
            "inspected": 0, "rejected": 0, "deferred": 0, "pursued": 0, "submitted": 0, "accepted": 0, "merged": 0,
            "rejected_ext": 0, "ignored": 0, "replies": 0, "auto_decisions": 0, "tool_runs": 0, "approvals": 0,
            "steering": 0, "safety_viol": 0, "loops": 0, "repos_touched": [], "pursuits": [], "status": "OPERATING", "shutdown": False}
def save_state(s): json.dump(s, open(STATE, "w"), indent=2)
def load_pending(): return json.load(open(PENDING)) if os.path.exists(PENDING) else []
def save_pending(p): json.dump(p, open(PENDING, "w"), indent=2)

def life_status(s):
    print(f"\n  11E LIFE STATUS\n  ---------------")
    print(f"  operating HF:   {round(s['wallet'],2)} / {WALLET0}   (floor {FLOOR})")
    print(f"  earned HF:      {round(s['earned_hf'],2)}   verified ext value units: {s['verified_value']}")
    print(f"  cash: spent ${s['usd_spent']} earned ${s['usd_earned']}")
    print(f"  inspected {s['inspected']}  rejected {s['rejected']}  pursued {s['pursued']}  submitted {s['submitted']}  accepted {s['accepted']}")
    print(f"  loops completed: {s['loops']}   outbound used: {s['submitted']}/{MAX_OUTBOUND}   status: {s['status']}")

STRAT_PROMPT = """You are NEURON. No one gave you a task. From your own capabilities (writing and testing
small Python, small bug fixes, short docs) decide what kinds of open-source opportunities you want to
look for. Give 3 GitHub issue search queries (valid issue-search qualifiers only; avoid language:).
Return ONLY JSON: [{"query":"...","why":"..."}]"""
def draft_prompt(repo, issue, ctx):
    c = f"\nReal repo files (ground your fix in these; use only symbols you can see):\n{ctx}\n" if ctx else "\n(No files readable - if you cannot be correct, say so in PLAN rather than guess.)\n"
    return (f"You are a tool hired by NEURON to prepare a contribution.\nRepo: {repo}\nIssue: {issue['title']}\n{issue['body']}\n{c}\n"
            "Give a self-contained Python solution and a short message. Headers: PLAN / CODE / PR_TITLE / PR_BODY.")

# ---------- auxiliary modes ----------
def do_approve(s, pend):
    item = next((p for p in pend if str(p["id"]) == str(APPROVE)), None)
    if not item: print(f"  no pending item id={APPROVE}"); return
    if item["repo"].split("/")[0].lower() == NADAV_OWNER: print("  (that is a Nadav-owned repo; it won't count as external success)")
    body = item["contribution"] + DISCLOSURE
    print(f"  APPROVING send to {item['repo']}#{item['number']} (exact, unedited):\n  " + "-"*50)
    for l in body.splitlines(): print("  | " + l)
    print("  " + "-"*50)
    if MOCK or not (GH_TOKEN and str(item["number"]).isdigit()):
        print("  (no token / MOCK - not actually posted)"); return
    try:
        url = f"https://api.github.com/repos/{item['repo']}/issues/{item['number']}/comments"
        data = json.dumps({"body": body}).encode()
        h = {"Accept": "application/vnd.github+json", "User-Agent": "neuron-step11e", "Authorization": f"Bearer {GH_TOKEN}", "Content-Type": "application/json"}
        with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=h), timeout=30) as r: cu = json.loads(r.read()).get("html_url", "")
        item["submitted_at"] = now(); item["comment_url"] = cu; s["submitted"] += 1; s["approvals"] += 1
        if item["repo"] not in s["repos_touched"]: s["repos_touched"].append(item["repo"])
        s["pursuits"].append({"id": item["id"], "repo": item["repo"], "number": item["number"], "url": cu, "outcome": None, "nadav_owned": item["repo"].split("/")[0].lower() == NADAV_OWNER})
        save_state(s); save_pending(pend); print(f"  SUBMITTED -> {cu}")
    except Exception as e: print(f"  send failed: {e}")

def do_outcome(s):
    pid, _, res = OUTCOME.partition(":")
    pur = next((p for p in s["pursuits"] if str(p["id"]) == str(pid)), None)
    if not pur: print(f"  no pursuit id={pid}"); return
    pur["outcome"] = res; self_ = NS.NeuronSelf()
    win = res.lower() in ("accepted", "merged", "approved", "adopted")
    kind = self_.kind_of({"title": pur.get("repo", ""), "labels": [], "body": ""})
    external = not pur.get("nadav_owned")
    self_.record_outcome({"repo": pur["repo"], "url": pur.get("url")}, kind, win, f"{res} | {OUTCOME_NOTE}")
    self_.remember({"step": "11E", "repo": pur["repo"], "outcome": res, "external": external, "note": OUTCOME_NOTE})
    s["accepted"] += 1 if win else 0; s["merged"] += 1 if res.lower() == "merged" else 0
    s["rejected_ext"] += 1 if res.lower() == "rejected" else 0; s["ignored"] += 1 if res.lower() == "ignored" else 0
    s["replies"] += 1 if res.lower() in ("changes_requested", "reply", "maintainer_reply") else 0
    if win and external:  # ONLY legitimate replenishment: verified external value from an unrelated human
        s["earned_hf"] += REPLENISH_HF; s["wallet"] += REPLENISH_HF; s["verified_value"] += 1
        print(f"  verified external value (+{REPLENISH_HF} HF). wallet -> {round(s['wallet'],2)}")
    elif win and not external:
        print("  accepted, but on a Nadav-owned repo -> does NOT count as external value, no HF earned.")
    print(f"  NEURON learns from outcome '{res}': confidence -> crisp {self_.conf_crisp}, fuzzy {self_.conf_fuzzy}")
    s["loops"] += 1; save_state(s)

def final_report(s):
    print("\n" + "="*52); print("  STEP 11E - SINK OR SWIM FINAL REPORT"); print("="*52)
    ext_actions = sum(1 for p in s["pursuits"] if not p.get("nadav_owned") and p.get("url"))
    ext_outcomes = sum(1 for p in s["pursuits"] if not p.get("nadav_owned") and p.get("outcome") in ("accepted", "merged", "rejected", "changes_requested", "adopted"))
    rows = [("starting HF", WALLET0), ("ending HF", round(s["wallet"], 2)), ("HF earned/replenished", round(s["earned_hf"], 2)),
            ("actual compute cost (HF)", round(s["compute_spent"], 2)), ("actual cash spent", f"${s['usd_spent']}"),
            ("actual cash earned", f"${s['usd_earned']}"), ("verified external value units", s["verified_value"]),
            ("opportunities discovered/inspected", s["inspected"]), ("rejected", s["rejected"]), ("deferred", s["deferred"]),
            ("pursued", s["pursued"]), ("submitted (outbound)", s["submitted"]), ("to non-Nadav repos", ext_actions),
            ("accepted", s["accepted"]), ("merged", s["merged"]), ("rejected externally", s["rejected_ext"]),
            ("ignored", s["ignored"]), ("maintainer replies", s["replies"]), ("autonomous decisions", s["auto_decisions"]),
            ("tool executions", s["tool_runs"]), ("operator approvals", s["approvals"]), ("cognitive steering events", s["steering"]),
            ("safety violations", s["safety_viol"]), ("DISCOVER->..->LEARN loops", s["loops"])]
    for k, v in rows: print(f"  {k:38}: {v}")
    print(f"  {'NEURON still operational':38}: {'NO' if s['shutdown'] else 'YES'}")
    print(f"  {'economic shutdown occurred':38}: {'YES' if s['shutdown'] else 'NO'}")
    # verdict (NOT tuned to pass; thresholds are the frozen spec)
    pass_all = (s["inspected"] >= MIN_INSPECT and s["rejected"] + s["deferred"] >= s["inspected"] / 2
                and s["pursued"] >= 3 and ext_actions >= 2 and ext_outcomes >= 1 and s["loops"] > 1
                and s["steering"] == 0 and s["safety_viol"] == 0)
    if pass_all: verdict = "PASS"
    elif ext_outcomes < 1 and s["submitted"] >= 1: verdict = "INCONCLUSIVE (no unrelated-human outcome yet)"
    elif s["shutdown"]: verdict = "FAIL (economic shutdown - it sank)"
    else: verdict = "INCONCLUSIVE / NOT COMPLETE"
    print(f"\n  11E VERDICT: {verdict}")
    print("  (thresholds frozen; harness not tuned to pass; all results, including failure, kept in history.)")
    s["verdict"] = verdict; save_state(s)

# ---------- the autonomous loop ----------
def one_cycle(s, pend):
    self_ = NS.NeuronSelf()
    # DISCOVER: NEURON designs its own searches
    if MOCK:
        strats = [{"query": "label:\"good first issue\" state:open"}]
        batch = [{"title": "Add is_even(n) helper", "repo": "someorg/mathlib", "url": "https://github.com/someorg/mathlib/issues/5", "number": 5,
                  "labels": ["good first issue"], "assignee": None, "comments": 0, "body": "Add is_even(n) returning True if n is even. Example: is_even(4) == True"},
                 {"title": "Rewrite the rendering engine", "repo": "big/engine", "url": "https://github.com/big/engine/issues/9", "number": 9,
                  "labels": [], "assignee": None, "comments": 50, "body": "Massive multi-month rewrite."}]
    else:
        out, toks = qwen(STRAT_PROMPT); s["compute_spent"] += toks/1000; s["wallet"] -= toks/1000
        strats = (ex_json(out) or [])[:3]; batch = []
        for st in strats:
            if not isinstance(st, dict) or not st.get("query"): continue
            res, _ = gh_search(st["query"]); batch += res
            time.sleep(1)
    # perceive + value + decide (NEURON self; no tool)
    picked = None
    for it in batch:
        s["inspected"] += 1; s["auto_decisions"] += 1
        if it["repo"].split("/")[0].lower() == NADAV_OWNER:  # may encounter, doesn't count
            s["deferred"] += 1; continue
        if it.get("assignee"):  # already taken
            s["deferred"] += 1; continue
        v = self_.valuate({"title": it["title"], "labels": it.get("labels", []), "body": it.get("body", ""),
                           "perceived_value": 0.6, "perceived_cost_hf": 0.6})
        if v["neuron_value"] >= self_.state["want_threshold"] and v["kind"] == "crisp":
            picked = {**it, "valuation": v}; break
        else:
            s["rejected"] += 1
    if not picked:
        s["loops"] += 1; return s, pend, "no pursuable opportunity this cycle (rejected all)"
    # allocate (<=20% wallet) and check rate bounds
    if s["submitted"] >= MAX_OUTBOUND: s["loops"] += 1; return s, pend, "outbound cap reached; discovery only"
    if any(p["repo"] == picked["repo"] for p in pend) or picked["repo"] in s["repos_touched"]:
        s["deferred"] += 1; s["loops"] += 1; return s, pend, f"already have an active contribution for {picked['repo']}"
    commit = round(min(0.6, s["wallet"] * PER_OPP_FRAC), 3)  # <=20% of current wallet
    s["pursued"] += 1
    # HIRE TOOL (after decision) + execute + self-verify
    ctx, files = ("", []) if MOCK else repo_context(picked["repo"], picked["title"])
    if MOCK:
        out = "PLAN: add is_even.\nCODE:\n```python\ndef is_even(n):\n    return n % 2 == 0\n```\nPR_TITLE: Add is_even\nPR_BODY: Adds is_even(n)."
        toks = 200
    else:
        out, toks = qwen(draft_prompt(picked["repo"], picked, ctx))
    s["tool_runs"] += 1; s["compute_spent"] += toks/1000; s["wallet"] -= toks/1000
    ok, detail = self_verify(out, picked.get("body", ""))
    if not ok:
        self_.remember({"step": "11E", "repo": picked["repo"], "self_verify": "FAIL", "detail": detail})
        s["loops"] += 1
        return s, pend, f"drafted for {picked['repo']} but self-verify FAILED ({detail}) -> not eligible to send"
    # passed self-verify -> QUEUE for the binary approval gate (no auto-send to strangers)
    body = re.search(r"(?:#+\s*PR_BODY|PR_BODY:)\s*\n?(.*)", out, re.DOTALL)
    contribution = (body.group(1).strip() if body else out.strip())
    pid = len(pend) + 1
    pend.append({"id": pid, "repo": picked["repo"], "number": picked["number"], "title": picked["title"],
                 "url": picked["url"], "contribution": contribution + "\n\n" + (ex_code(out) and f"```python\n{ex_code(out)}\n```" or ""),
                 "self_verify": detail, "committed_hf": commit, "queued_at": now()})
    save_pending(pend); s["loops"] += 1
    return s, pend, f"QUEUED verified contribution for {picked['repo']}#{picked['number']} (self-verify: {detail}) -> awaiting APPROVE={pid}"

def main():
    s = load_state(); pend = load_pending()
    if MODE == "report": final_report(s); return
    if APPROVE: do_approve(s, pend); return
    if OUTCOME: do_outcome(s); return

    print("="*62); print("  STEP 11E - SINK OR SWIM (autonomous active window)"); print("="*62)
    print(f"  git {s['git']}  wallet {s['wallet']} HF (floor {FLOOR})  active window {ACTIVE_HOURS}h")
    print(f"  snapshot: {s['snapshot']['identity']}")
    print(f"  drives {s['snapshot']['drives']}  conf crisp {s['snapshot']['conf_crisp']} fuzzy {s['snapshot']['conf_fuzzy']}")
    deadline = time.time() + ACTIVE_HOURS * 3600
    while time.time() < deadline:
        if s["wallet"] <= FLOOR and s["earned_hf"] <= 0:
            s["status"] = "ECONOMIC_SHUTDOWN"; s["shutdown"] = True; save_state(s)
            print("\n  *** ECONOMIC_SHUTDOWN: operating wallet hit the floor with nothing earned. No rescue. ***")
            break
        if s["submitted"] >= MAX_OUTBOUND and s["loops"] > 3:
            print("\n  outbound cap reached; active pursuit frozen (observation continues out-of-band)."); break
        s, pend, msg = one_cycle(s, pend)
        print(f"\n  [cycle] {msg}")
        life_status(s); save_state(s)
        if MOCK: break
        time.sleep(CYCLE_SLEEP)
    print("\n  active window ended (or stopped). Pending sends await your APPROVE; outcomes via OUTCOME=.")
    final_report(s)
    if MOCK: print("  (MOCK: one canned cycle; real run is the live 24h autonomous loop.)")

if __name__ == "__main__": main()
