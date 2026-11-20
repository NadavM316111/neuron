"""HyperFlow 9K - the Opportunity Board (public, always-on, hosted on Railway).

Real people (and you) drop GENUINE tasks here. NEURON's worker (on your laptop)
pulls the open ones, decides ACCEPT/DECLINE/DEFER/WAIT, does the accepted ones with
qwen, and posts answers back. Submitters then rate usefulness + would-pay, which is
the REAL value signal (not auto-invented).

Rule: only real tasks that exist for a real reason. No fake benchmarks to fill the queue.

Pages:
  /               submit a task + see recent ones
  /task/<id>      see NEURON's decision + answer, and rate it
  /board          live dashboard (counts, decisions, useful %, value, reserve)
Worker API (shared-secret token):
  GET  /api/open           open tasks for the worker
  POST /api/result         worker posts a decision/answer/cost
  GET  /api/ratings        rated-but-uncredited tasks (worker books value)
  POST /api/credited       mark a rating credited

Deploy on Railway (same pattern as the 72h runner):
  Root dir: neuron/system/hyperflow   Start: python3 hyperflow_board.py
  Volume at /data    Vars: BOARD_DIR=/data  WORKER_TOKEN=<pick-a-secret>
"""
import os, json, time, html, uuid, tempfile
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

BOARD_DIR   = os.environ.get("BOARD_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "board_state"))
os.makedirs(BOARD_DIR, exist_ok=True)
TASKS       = os.path.join(BOARD_DIR, "tasks.json")
PORT        = int(os.environ.get("PORT", "8100"))
WORKER_TOKEN= os.environ.get("WORKER_TOKEN", "changeme")
CATEGORIES  = ["coding", "numeric", "analysis", "planning", "explanation", "research", "other"]

def now(): return datetime.now(timezone.utc).isoformat()
def _atomic(path, text):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
    with os.fdopen(fd, "w") as f: f.write(text)
    os.replace(tmp, path)
def load():
    try:
        with open(TASKS) as f: return json.load(f)
    except Exception: return []
def save(ts): _atomic(TASKS, json.dumps(ts))
def esc(s): return html.escape(str(s or ""))
def money(x):
    try: return float(str(x).strip())
    except Exception: return 0.0

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#0c0d10;color:#e8e8ea;margin:0;padding:20px;max-width:720px;margin:0 auto;}}
 a{{color:#6ea8ff;text-decoration:none;}} h1{{font-size:20px;}} h2{{font-size:15px;color:#aab;margin-top:24px;}}
 input,select,textarea,button{{font-family:inherit;font-size:15px;background:#15171c;color:#e8e8ea;border:1px solid #2a2e38;border-radius:8px;padding:10px;width:100%;box-sizing:border-box;margin:6px 0;}}
 button{{background:#2f6bff;border:none;font-weight:600;cursor:pointer;width:auto;padding:10px 18px;}}
 .card{{background:#15171c;border:1px solid #232630;border-radius:10px;padding:14px;margin:10px 0;}}
 .muted{{color:#8b8d93;font-size:13px;}} .pill{{display:inline-block;padding:2px 8px;border-radius:20px;font-size:12px;background:#232630;}}
 label{{font-size:13px;color:#aab;}} pre{{white-space:pre-wrap;background:#0a0b0e;border:1px solid #1e2127;border-radius:8px;padding:12px;}}
</style></head><body>{body}</body></html>"""

def submit_form():
    opts = "".join(f'<option value="{c}">{c}</option>' for c in CATEGORIES)
    recent = ""
    for t in list(reversed(load()))[:12]:
        recent += (f'<div class="card"><span class="pill">{esc(t["status"])}</span> '
                   f'<b>{esc(t["category"])}</b> &middot; <span class="muted">{esc(t["submitter"])}</span><br>'
                   f'{esc(t["text"])[:140]} &middot; <a href="/task/{t["id"]}">view</a></div>')
    body = f"""<h1>NEURON Opportunity Board</h1>
<p class="muted">Submit a genuine task you actually want done. NEURON decides whether to take it.
Only real tasks, please. <a href="/board">dashboard</a></p>
<form method="post" action="/submit">
<label>Your name</label><input name="submitter" required>
<label>Who you are</label><input name="role" placeholder="parent, teacher, friend, me, tester">
<label>Category</label><select name="category">{opts}</select>
<label>The task</label><textarea name="text" rows="4" required placeholder="Describe what you need."></textarea>
<label>What would it be worth to you? (optional, USD)</label><input name="would_pay" placeholder="e.g. 5">
<button type="submit">Submit to NEURON</button>
</form>
<h2>recent</h2>{recent or '<p class="muted">none yet</p>'}"""
    return PAGE.format(title="NEURON Opportunity Board", body=body)

def task_page(t):
    ans = ""
    if t["status"] in ("delivered", "credited"):
        ans = f'<h2>NEURON answer</h2><pre>{esc(t.get("answer",""))}</pre>'
        if t.get("useful") is None:
            ans += f"""<h2>rate it</h2><form method="post" action="/rate">
<input type="hidden" name="id" value="{t['id']}">
<label>Was it useful?</label><select name="useful"><option value="yes">yes</option><option value="partly">partly</option><option value="no">no</option></select>
<label>Would you actually pay for this? How much (USD)?</label><input name="paid" placeholder="0 if no">
<button type="submit">Submit rating</button></form>"""
        else:
            ans += f'<p class="muted">rated: {esc(t["useful"])} &middot; would-pay ${esc(t.get("paid",0))}</p>'
    elif t["status"] == "declined":
        ans = f'<p class="muted">NEURON declined this: {esc(t.get("decision_reason","not a fit"))}</p>'
    elif t["status"] in ("deferred",):
        ans = '<p class="muted">NEURON deferred this for now (budget or priority).</p>'
    else:
        ans = '<p class="muted">waiting for NEURON to see it.</p>'
    body = (f'<p><a href="/">&larr; board</a></p><h1>Task</h1>'
            f'<div class="card"><span class="pill">{esc(t["status"])}</span> <b>{esc(t["category"])}</b> '
            f'&middot; <span class="muted">{esc(t["submitter"])} ({esc(t.get("role",""))})</span><br>'
            f'{esc(t["text"])}</div>{ans}')
    return PAGE.format(title="Task", body=body)

def board_page():
    ts = load()
    def c(st): return sum(1 for t in ts if t["status"] == st)
    rated = [t for t in ts if t.get("useful") is not None]
    useful = sum(1 for t in rated if t["useful"] in ("yes", "partly"))
    value = sum(money(t.get("paid")) for t in ts if t.get("credited"))
    cost = sum(money(t.get("cost")) for t in ts)
    rows = ""
    for t in list(reversed(ts))[:40]:
        rows += (f'<tr><td>{esc(t["status"])}</td><td>{esc(t["category"])}</td>'
                 f'<td>{esc(t["submitter"])}</td><td>{esc((t.get("useful") or "-"))}</td></tr>')
    body = f"""<p><a href="/">&larr; submit</a></p><h1>9K dashboard</h1>
<div class="card">total {len(ts)} &middot; open {c('open')} &middot; delivered {c('delivered')+c('credited')}
 &middot; declined {c('declined')} &middot; deferred {c('deferred')}</div>
<div class="card">rated {len(rated)} &middot; useful {useful}/{len(rated) if rated else 0}
 &middot; real value ${value:.2f} &middot; compute cost ${cost:.4f}</div>
<table style="width:100%;border-collapse:collapse;font-size:13px;">
<tr class="muted"><td>status</td><td>category</td><td>from</td><td>useful</td></tr>{rows}</table>"""
    return PAGE.format(title="9K dashboard", body=body)

class H(BaseHTTPRequestHandler):
    def log_message(s, *a): pass
    def _send(s, body, code=200, ctype="text/html; charset=utf-8"):
        s.send_response(code); s.send_header("Content-Type", ctype); s.end_headers()
        s.wfile.write(body.encode() if isinstance(body, str) else body)
    def _json(s, obj, code=200): s._send(json.dumps(obj), code, "application/json")
    def _body(s):
        n = int(s.headers.get("Content-Length", 0)); return s.rfile.read(n).decode() if n else ""
    def _form(s):
        return {k: v[0] for k, v in parse_qs(s._body()).items()}
    def _auth(s, q): return q.get("token", [""])[0] == WORKER_TOKEN

    def do_GET(s):
        u = urlparse(s.path); q = parse_qs(u.query); p = u.path
        if p == "/": return s._send(submit_form())
        if p == "/board": return s._send(board_page())
        if p.startswith("/task/"):
            tid = p.split("/task/", 1)[1]
            t = next((x for x in load() if x["id"] == tid), None)
            return s._send(task_page(t)) if t else s._send("not found", 404)
        if p == "/api/open":
            if not s._auth(q): return s._json({"error": "unauthorized"}, 401)
            return s._json([t for t in load() if t["status"] == "open"])
        if p == "/api/ratings":
            if not s._auth(q): return s._json({"error": "unauthorized"}, 401)
            return s._json([t for t in load() if t.get("useful") is not None and not t.get("credited")])
        if p == "/api/export":   # read-only full dump for offline audits (9N/9O); changes no NEURON behavior
            if not s._auth(q): return s._json({"error": "unauthorized"}, 401)
            return s._json({"exported_at": now(), "count": len(load()), "tasks": load()})
        return s._send("not found", 404)

    def do_POST(s):
        u = urlparse(s.path); q = parse_qs(u.query); p = u.path
        if p == "/submit":
            f = s._form(); ts = load()
            t = {"id": uuid.uuid4().hex[:8], "submitter": f.get("submitter", "anon"), "role": f.get("role", ""),
                 "category": f.get("category", "other"), "text": f.get("text", ""),
                 "would_pay": f.get("would_pay", ""), "status": "open", "created": now(),
                 "decision": None, "answer": None, "cost": 0.0, "useful": None, "paid": None, "credited": False}
            ts.append(t); save(ts)
            s.send_response(303); s.send_header("Location", f"/task/{t['id']}"); s.end_headers(); return
        if p == "/rate":
            f = s._form(); ts = load()
            for t in ts:
                if t["id"] == f.get("id"):
                    t["useful"] = f.get("useful", "no"); t["paid"] = f.get("paid", "0"); t["rated_at"] = now()
            save(ts)
            s.send_response(303); s.send_header("Location", f"/task/{f.get('id','')}"); s.end_headers(); return
        if p == "/api/result":
            if not s._auth(q): return s._json({"error": "unauthorized"}, 401)
            d = json.loads(s._body()); ts = load()
            for t in ts:
                if t["id"] == d["id"]:
                    t["decision"] = d.get("decision"); t["decision_reason"] = d.get("reason", "")
                    t["answer"] = d.get("answer"); t["cost"] = d.get("cost", 0.0)
                    dec = d.get("decision")
                    t["status"] = {"ACCEPT": "delivered", "DECLINE": "declined", "DEFER": "deferred"}.get(dec, "open")
            save(ts); return s._json({"ok": True})
        if p == "/api/credited":
            if not s._auth(q): return s._json({"error": "unauthorized"}, 401)
            d = json.loads(s._body()); ts = load()
            for t in ts:
                if t["id"] == d["id"]: t["credited"] = True; t["status"] = "credited"
            save(ts); return s._json({"ok": True})
        return s._send("not found", 404)

if __name__ == "__main__":
    print(f"board on :{PORT}  dir={BOARD_DIR}  token set={'yes' if WORKER_TOKEN!='changeme' else 'NO - set WORKER_TOKEN'}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
