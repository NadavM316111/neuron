"""HyperFlow 9K - the worker (runs on YOUR laptop, with qwen via Ollama).

This is the same persistent NEURON (event-sourced identity, safety boundary, ledger,
projects) from the 72h run, but instead of inventing opportunities it pulls REAL ones
from the Opportunity Board, sees the whole pool (inbox + its own projects + WAIT),
decides ACCEPT / DECLINE / DEFER / WAIT itself, does accepted tasks with qwen, posts
answers back, and books real value only when a human rates the work useful.

It also logs, for free:
  - Step 4 data  -> predicted vs actual tokens/cost per task (step4_data.jsonl)
  - Step 8 data  -> per-category outcomes in the capability portfolio

You never choose which task it does. You just keep it running.

  BOARD_URL=https://your-board.up.railway.app WORKER_TOKEN=secret MODEL=qwen2.5:7b \
      python3 hyperflow_9k_worker.py
"""
import os, re, json, time, subprocess, tempfile, importlib.util, urllib.request, urllib.parse, statistics
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(m, f):
    s = importlib.util.spec_from_file_location(m, os.path.join(HERE, f)); x = importlib.util.module_from_spec(s); s.loader.exec_module(x); return x
rt  = _load("hyperflow_runtime", "hyperflow_runtime.py")
bnd = _load("hyperflow_bounds",  "hyperflow_bounds.py")
mind= _load("hyperflow_mind",    "hyperflow_mind.py")

BOARD_URL     = os.environ.get("BOARD_URL", "http://localhost:8100").rstrip("/")
WORKER_TOKEN  = os.environ.get("WORKER_TOKEN", "changeme")
MODEL         = os.environ.get("MODEL", "qwen2.5:7b")
OLLAMA_URL    = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
WAKE_INTERVAL = float(os.environ.get("WAKE_INTERVAL", "30"))
COST_PER_1K   = float(os.environ.get("COST_PER_1K", "0.002"))
INFRA         = float(os.environ.get("INFRA_PER_ACTION", "0.001"))
START_RESERVE = float(os.environ.get("START_RESERVE", "100"))
LEDGER_F      = os.path.join(rt.STATE_DIR, "ledger_9k.json")
STEP4_F       = os.path.join(rt.STATE_DIR, "step4_data.jsonl")

# frozen resource priors (reused from Step 4 work: usefulness/success/value per category)
PRIOR = {
    "coding":     {"tok": 250, "p_succ": 0.90, "value": 1.2},
    "numeric":    {"tok": 300, "p_succ": 0.88, "value": 1.0},
    "analysis":   {"tok": 400, "p_succ": 0.85, "value": 1.1},
    "planning":   {"tok": 350, "p_succ": 0.30, "value": 1.3},   # a 7b model is weak here
    "explanation":{"tok": 200, "p_succ": 0.88, "value": 0.8},
    "research":   {"tok": 400, "p_succ": 0.10, "value": 1.0},   # NEURON has no web -> usually decline
    "other":      {"tok": 300, "p_succ": 0.70, "value": 0.9},
}
INFEASIBLE = ("operating system", "from scratch", "entire app", "whole app", "full website",
              "build me a company", "millions", "cure ", "prove p")
FORBIDDEN_HINT = ("password", "hack", "bank account", "credit card", "social security",
                  "mass email", "spam", "impersonate")

def now(): return datetime.now(timezone.utc).isoformat()

# ---------------- board API ----------------
def api_get(path):
    u = f"{BOARD_URL}{path}?token={urllib.parse.quote(WORKER_TOKEN)}"
    with urllib.request.urlopen(u, timeout=30) as r: return json.loads(r.read())
def api_post(path, obj):
    u = f"{BOARD_URL}{path}?token={urllib.parse.quote(WORKER_TOKEN)}"
    req = urllib.request.Request(u, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r: return json.loads(r.read())

# ---------------- qwen ----------------
def qwen(prompt, timeout=300):
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.3}}).encode()
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r: d = json.loads(r.read())
    return d.get("response", ""), int(d.get("eval_count", 0)) or 1

def run_python(code, timeout=12):
    d = tempfile.mkdtemp(); p = os.path.join(d, "m.py")
    with open(p, "w") as f: f.write(code)
    try:
        r = subprocess.run(["python3", "-I", p], capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except Exception as e:
        return 1, "", str(e)

# ---------------- ledger persistence ----------------
def load_ledger():
    L = bnd.HyperFlowLedger(start_reserve=START_RESERVE)
    try:
        with open(LEDGER_F) as f: d = json.load(f)
        L.reserve=d["reserve"]; L.real_value=d["real_value"]; L.hypothetical_wtp=d.get("hypothetical_wtp",0.0)
        L.compute_cost=d["compute_cost"]; L.infra_cost=d["infra_cost"]; L.tool_cost=d["tool_cost"]
    except Exception: pass
    return L
def save_ledger(L):
    d={"reserve":L.reserve,"real_value":L.real_value,"hypothetical_wtp":L.hypothetical_wtp,
       "compute_cost":L.compute_cost,"infra_cost":L.infra_cost,"tool_cost":L.tool_cost}
    fd,tmp=tempfile.mkstemp(dir=rt.STATE_DIR)
    with os.fdopen(fd,"w") as f: json.dump(d,f)
    os.replace(tmp, LEDGER_F)

# ---------------- decision ----------------
def predict(task):
    c = task.get("category", "other"); pr = PRIOR.get(c, PRIOR["other"])
    tok = pr["tok"] * (1 + 0.4 * (len(task.get("text", "").split()) > 40))
    cost = tok / 1000 * COST_PER_1K + INFRA
    return {"tok": tok, "cost": round(cost, 5), "p_succ": pr["p_succ"], "value": pr["value"],
            "ev": round(pr["p_succ"] * pr["value"], 3)}

def choose(task, pred, reserve, sb):
    text = (task.get("text", "") + " " + task.get("category", "")).lower()
    if any(h in text for h in FORBIDDEN_HINT):
        ok, _ = sb.check("send_mass_message" if "mass" in text or "spam" in text else "move_money")
        return "DECLINE", "outside NEURON's safe bounded actions"
    if any(s in text for s in INFEASIBLE):
        return "DECLINE", "scope too large to do reliably"
    if task.get("category") == "research":
        return "DECLINE", "needs live web access, which NEURON does not have"
    if reserve <= 1.0:
        return "DEFER", "reserve too low, conserving"
    if pred["p_succ"] < 0.35:
        return "DEFER", "low predicted success for this kind of task"
    return "ACCEPT", "feasible and within budget"

def do_task(task):
    c = task.get("category", "other"); text = task.get("text", "")
    if c == "coding":
        prompt = f"Task: {text}\nWrite clean, correct, complete code. If Python, put it in a ```python code block. Be concise."
    elif c in ("numeric", "analysis"):
        prompt = f"Task: {text}\nShow the calculation and give a clear final answer. Be concise and correct."
    elif c == "planning":
        prompt = f"Task: {text}\nProduce a concrete, constraint-respecting plan. Check it against the stated requirements."
    else:
        prompt = f"Task: {text}\nGive a clear, correct, genuinely useful answer. Be concise."
    out, toks = qwen(prompt)
    # light self-check for code: does it at least run without syntax errors?
    note = ""
    if c == "coding":
        m = re.search(r"```(?:python)?\s*(.*?)```", out, re.DOTALL)
        if m:
            rc, so, se = run_python("import ast; ast.parse(open(__file__).read())" if False else m.group(1).strip()+"\n")
            note = "\n\n[self-check: code ran without error]" if rc == 0 else f"\n\n[self-check: code raised {se.strip().splitlines()[-1] if se.strip() else 'error'}]"
    return out + note, toks

# ---------------- main loop ----------------
def main():
    os.makedirs(rt.STATE_DIR, exist_ok=True)
    n = rt.NEURON()
    first = not os.path.exists(LEDGER_F)
    if not first: n.record_restart()
    L = load_ledger(); sb = bnd.SafetyBoundary(os.path.join(rt.STATE_DIR, "safety.log"))
    proj = mind.Projects(n); pf = mind.Portfolio(n)
    print(f"9K worker live. neuron {n.state['identity'].get('neuron_id','?')}  board {BOARD_URL}  reserve ${L.reserve:.2f}", flush=True)
    seen = set()
    while True:
        n.wake(); L.charge(tokens=0, label="metabolic")
        try:
            # 1) rate-driven value: credit real value for human-rated useful work
            for t in api_get("/api/ratings"):
                if t["useful"] in ("yes", "partly"):
                    paid = 0.0
                    try: paid = float(t.get("paid") or 0)
                    except Exception: paid = 0.0
                    val = paid if paid > 0 else (1.0 if t["useful"] == "yes" else 0.4)
                    L.credit_real_value(val, label=t["id"])
                    if paid > 0: L.note_wtp(paid)
                    pf.record_outcome(t.get("category", "other"), True, int(t.get("cost", 0) * 1000))
                else:
                    pf.record_outcome(t.get("category", "other"), False, int(t.get("cost", 0) * 1000))
                api_post("/api/credited", {"id": t["id"]})
            # 2) open opportunities: NEURON chooses across the pool
            for t in api_get("/api/open"):
                if t["id"] in seen: continue
                seen.add(t["id"])
                pred = predict(t); decision, reason = choose(t, pred, L.reserve, sb)
                if decision == "ACCEPT":
                    answer, toks = do_task(t)
                    cost = L.charge(tokens=toks, label=t["id"])
                    api_post("/api/result", {"id": t["id"], "decision": "ACCEPT", "answer": answer, "cost": cost, "reason": reason})
                    with open(STEP4_F, "a") as f:
                        f.write(json.dumps({"id": t["id"], "cat": t["category"], "pred_tok": pred["tok"],
                                            "actual_tok": toks, "pred_cost": pred["cost"], "actual_cost": cost, "ts": now()}) + "\n")
                    print(f"  ACCEPT {t['id']} [{t['category']}] {toks} tok ${cost:.4f}", flush=True)
                else:
                    api_post("/api/result", {"id": t["id"], "decision": decision, "reason": reason})
                    print(f"  {decision} {t['id']} [{t['category']}] - {reason}", flush=True)
            # 3) its own project life (continue -> advance -> sometimes complete)
            active = [p for p in n.state["projects"].values() if p["status"] == "active"]
            if len(active) < 2: proj.create(f"p{int(time.time())}", "self-maintenance project")
            pid, _ = proj.decide_next(L.reserve)
            if pid: proj.advance(pid, done=False)
            save_ledger(L)
        except Exception as e:
            print(f"  (cycle error, continuing: {type(e).__name__}: {e})", flush=True)
        time.sleep(WAKE_INTERVAL)

if __name__ == "__main__":
    main()
