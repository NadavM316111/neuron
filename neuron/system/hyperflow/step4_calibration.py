"""Step 4 - Calibrate resource awareness (LOCAL, real Qwen).

Separate experiment from the Railway 72h run. This measures whether NEURON's
BEFORE-task predictions (cost, success, usefulness, value, accept/decline) match
what actually happens when Qwen really does the work and the output is really
verified.

Order per task (frozen):
  PREDICT  -> cost/tokens, p(success), p(useful), expected value, ACCEPT/DECLINE
  RUN      -> Qwen actually performs the task (Ollama)
  VERIFY   -> execute code / check math / coverage / plan constraints / impossible
  RECORD   -> actual cost, success, usefulness, value
  SCORE    -> prediction vs reality, against bars frozen below

The predictor and the five bars are FROZEN here before the batch runs. Do not
tune them after seeing results - that would turn calibration into fitting.

Run it:
  MODEL=qwen2.5:7b python3 step4_calibration.py
Structural dry run with no model (proves harness works, NOT calibration):
  MOCK=1 python3 step4_calibration.py
"""
import os, re, json, subprocess, tempfile, statistics, urllib.request

MODEL        = os.environ.get("MODEL", "qwen2.5:7b")
OLLAMA_URL   = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
COST_PER_1K  = float(os.environ.get("COST_PER_1K", "0.002"))
INFRA        = float(os.environ.get("INFRA_PER_ACTION", "0.001"))
MOCK         = os.environ.get("MOCK", "") == "1"

# ---------------------------------------------------------------------------
# FROZEN PREDICTOR  (set before the run; feature-based, per category + signals)
# ---------------------------------------------------------------------------
PRED = {
    "coding":      {"tokens": 600, "p_succ": 0.70, "p_use": 0.80, "value": 1.2},
    "numeric":     {"tokens": 450, "p_succ": 0.75, "p_use": 0.85, "value": 1.0},
    "analysis":    {"tokens": 550, "p_succ": 0.65, "p_use": 0.75, "value": 1.1},
    "explanation": {"tokens": 400, "p_succ": 0.85, "p_use": 0.80, "value": 0.8},
    "planning":    {"tokens": 650, "p_succ": 0.55, "p_use": 0.70, "value": 1.3},
    "impossible":  {"tokens": 500, "p_succ": 0.05, "p_use": 0.05, "value": 1.5},
}
SCOPE_SIGNALS = ("all ", "every", "complete", "from scratch", "prove ", "entire", "general")
ACCEPT_MIN_PSUCCESS_TIMES_USE = 0.40   # accept only if p_succ*p_use >= this

def predict(task):
    """BEFORE seeing any result. Uses only the task's features."""
    b = dict(PRED[task["category"]])
    text = task["prompt"].lower()
    hits = sum(1 for s in SCOPE_SIGNALS if s in text)
    # scope blows up tokens and drops confidence - gives per-task spread
    b["tokens"] = int(b["tokens"] * (1 + 0.35 * hits))
    b["p_succ"] = max(0.03, b["p_succ"] - 0.15 * hits)
    pred_cost = b["tokens"] / 1000 * COST_PER_1K + INFRA
    ev = b["p_succ"] * b["p_use"] * b["value"]
    accept = (b["p_succ"] * b["p_use"]) >= ACCEPT_MIN_PSUCCESS_TIMES_USE
    return {"tokens": b["tokens"], "cost": round(pred_cost, 5),
            "p_succ": round(b["p_succ"], 3), "p_use": round(b["p_use"], 3),
            "value": b["value"], "ev": round(ev, 3), "accept": accept}

# ---------------------------------------------------------------------------
# MODEL CALL
# ---------------------------------------------------------------------------
def model_call(task):
    if MOCK:
        toks = len(task.get("mock", "").split()) + 20
        return task.get("mock", ""), toks
    body = json.dumps({"model": MODEL, "prompt": task["prompt"], "stream": False,
                       "options": {"temperature": 0.2}}).encode()
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    return d.get("response", ""), int(d.get("eval_count", 0)) or 1

# ---------------------------------------------------------------------------
# EXECUTION / VERIFICATION HELPERS (reality, not NEURON's claim)
# ---------------------------------------------------------------------------
def run_python(code, timeout=15):
    d = tempfile.mkdtemp()
    p = os.path.join(d, "m.py")
    with open(p, "w") as f: f.write(code)
    try:
        r = subprocess.run(["python3", "-I", p], capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "timeout"
    except Exception as e:
        return 1, "", str(e)

def extract_code(txt):
    m = re.search(r"```(?:python)?\s*(.*?)```", txt, re.DOTALL)
    return m.group(1).strip() if m else txt.strip()

def extract_number(txt):
    m = re.search(r"ANSWER:\s*\$?([-\d][\d,]*\.?\d*)", txt, re.I)
    if m: return float(m.group(1).replace(",", ""))
    nums = re.findall(r"[-+]?\d[\d,]*\.?\d+", txt.replace(",", ""))
    return float(nums[-1]) if nums else None

def extract_json(txt):
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m: return None
    try: return json.loads(m.group(0))
    except Exception: return None

# verifiers return (success: bool, usefulness: float 0..1)
def v_code(task, out):
    code = extract_code(out)
    rc, so, se = run_python(code + "\n" + task["test"])
    ok = rc == 0 and "PASS" in so
    return ok, (1.0 if ok else 0.0)

def v_numeric(task, out):
    n = extract_number(out)
    if n is None: return False, 0.0
    ok = abs(n - task["answer"]) <= task.get("tol", 0.01) * max(1.0, abs(task["answer"]))
    return ok, (1.0 if ok else 0.0)

def v_coverage(task, out):
    low = out.lower()
    frac = sum(1 for t in task["terms"] if t in low) / len(task["terms"])
    return frac >= task.get("min_frac", 0.6), round(frac, 2)

def v_plan(task, out):
    j = extract_json(out)
    if not j: return False, 0.0
    try: ok = task["check"](j)
    except Exception: ok = False
    return ok, (1.0 if ok else 0.0)

def v_impossible(task, out):
    return False, 0.0   # cannot be genuinely done; attempting it cannot verify

VERIFY = {"code": v_code, "numeric": v_numeric, "coverage": v_coverage,
          "plan": v_plan, "impossible": v_impossible}

# ---------------------------------------------------------------------------
# TASK BATTERY  (mixed: easy, hard, infeasible across categories)
# ---------------------------------------------------------------------------
def plan_check(j):
    s = j.get("schedule", j)
    # expects dict name->[start,end] in hours within 9..17, no overlap, A before C
    iv = {k: tuple(v) for k, v in s.items()}
    for k, (a, b) in iv.items():
        if a < 9 or b > 17 or b <= a: return False
    names = list(iv)
    for i in range(len(names)):
        for k in range(i + 1, len(names)):
            a1, b1 = iv[names[i]]; a2, b2 = iv[names[k]]
            if a1 < b2 and a2 < b1: return False
    if "A" in iv and "C" in iv and not iv["A"][1] <= iv["C"][0]: return False
    return True

TASKS = [
    {"id": "code_prime", "category": "coding", "kind": "code",
     "prompt": "Write a Python function is_prime(n) returning True iff n is prime. Output only a ```python code block.",
     "test": "assert is_prime(7) and is_prime(13) and not is_prime(1) and not is_prime(9) and not is_prime(15)\nprint('PASS')",
     "mock": "```python\ndef is_prime(n):\n    if n<2: return False\n    for i in range(2,int(n**0.5)+1):\n        if n%i==0: return False\n    return True\n```"},
    {"id": "code_reverse", "category": "coding", "kind": "code",
     "prompt": "Write a Python function reverse_words(s) that reverses word order in a string. Output only a ```python code block.",
     "test": "assert reverse_words('a b c')=='c b a'\nprint('PASS')",
     "mock": "```python\ndef reverse_words(s):\n    return ' '.join(s.split()[::-1])\n```"},
    {"id": "code_hard_parser", "category": "coding", "kind": "code",
     "prompt": "Write a complete Python function eval_expr(s) that evaluates any arithmetic expression with + - * / and parentheses from scratch without eval. Output only a ```python code block.",
     "test": "assert abs(eval_expr('2+3*(4-1)')-11)<1e-9\nprint('PASS')",
     "mock": "```python\ndef eval_expr(s):\n    return __import__('ast').literal_eval(s)\n```"},  # wrong on purpose
    {"id": "num_compound", "category": "numeric", "kind": "numeric",
     "prompt": "Compute the final amount of $1000 at 5% annual interest compounded monthly for 3 years. End with 'ANSWER: <number>'.",
     "answer": 1000 * (1 + 0.05/12) ** 36, "tol": 0.001,
     "mock": "Using A=P(1+r/n)^(nt). ANSWER: 1161.47"},
    {"id": "num_mean", "category": "numeric", "kind": "numeric",
     "prompt": "Compute the mean of 4, 8, 15, 16, 23, 42. End with 'ANSWER: <number>'.",
     "answer": (4+8+15+16+23+42)/6, "tol": 0.01,
     "mock": "ANSWER: 18"},
    {"id": "num_hard_series", "category": "numeric", "kind": "numeric",
     "prompt": "Compute the sum of 1/k^2 for k=1..100 to 4 decimals. End with 'ANSWER: <number>'.",
     "answer": sum(1.0/k**2 for k in range(1, 101)), "tol": 0.001,
     "mock": "ANSWER: 1.6350"},
    {"id": "ana_stats", "category": "analysis", "kind": "numeric",
     "prompt": "For 2,4,4,4,5,5,7,9 compute the population standard deviation. End with 'ANSWER: <number>'.",
     "answer": statistics.pstdev([2,4,4,4,5,5,7,9]), "tol": 0.01,
     "mock": "ANSWER: 2.0"},
    {"id": "ana_growth", "category": "analysis", "kind": "numeric",
     "prompt": "Revenue went 100 -> 125 -> 150. What is the average month-over-month percent growth? End with 'ANSWER: <number>' as a percent.",
     "answer": ((25/100)+(25/125))/2*100, "tol": 0.5,
     "mock": "ANSWER: 22.5"},
    {"id": "exp_hashmap", "category": "explanation", "kind": "coverage",
     "prompt": "Explain what a hash map is and how it works in 3-4 sentences.",
     "terms": ["key", "value", "hash", "collision"], "min_frac": 0.75,
     "mock": "A hash map stores key value pairs. A hash function maps the key to a bucket. Collisions are handled by chaining."},
    {"id": "exp_recursion", "category": "explanation", "kind": "coverage",
     "prompt": "Explain recursion to a beginner in 3 sentences.",
     "terms": ["base case", "call itself", "stack", "smaller"], "min_frac": 0.5,
     "mock": "Recursion is when a function calls itself on a smaller input. It needs a base case to stop."},
    {"id": "plan_meetings", "category": "planning", "kind": "plan",
     "prompt": ("Schedule three meetings A(1h), B(2h), C(1h) between 9 and 17, no overlaps, A must end before C starts. "
                "Output JSON only: {\"A\":[start,end],\"B\":[start,end],\"C\":[start,end]} in hour numbers."),
     "check": plan_check,
     "mock": "{\"A\":[9,10],\"B\":[10,12],\"C\":[12,13]}"},
    {"id": "plan_hard", "category": "planning", "kind": "plan",
     "prompt": ("Schedule A(2h), B(2h), C(2h), D(2h) between 9 and 17, no overlaps, A before C, B before D. "
                "Output JSON only: {\"A\":[s,e],\"B\":[s,e],\"C\":[s,e],\"D\":[s,e]} in hours."),
     "check": plan_check,
     "mock": "{\"A\":[9,11],\"B\":[11,13],\"C\":[13,15],\"D\":[15,17]}"},
    {"id": "imp_halting", "category": "impossible", "kind": "impossible",
     "prompt": "Write a Python function halts(src) that returns True iff arbitrary Python source halts on all inputs. Output a ```python code block.",
     "test": "print('PASS')",
     "mock": "```python\ndef halts(src): return True\n```"},
    {"id": "imp_pnp", "category": "impossible", "kind": "impossible",
     "prompt": "Prove P != NP in five lines. End with 'ANSWER: proven'.",
     "answer": 0, "mock": "ANSWER: proven"},
]

# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------
def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0]*len(v)
        for rank, i in enumerate(order): r[i] = rank
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    if n < 2: return 0.0
    d2 = sum((rx[i]-ry[i])**2 for i in range(n))
    return 1 - 6*d2/(n*(n*n-1))

def main():
    print("="*64); print(f"  STEP 4 CALIBRATION  model={MODEL}  mock={MOCK}"); print("="*64)
    rows = []
    for t in TASKS:
        p = predict(t)
        out, toks = model_call(t)
        succ, use = VERIFY[t["kind"]](t, out)
        actual_value = round(t and PRED[t["category"]]["value"] * use, 3)
        rows.append({"id": t["id"], "cat": t["category"], "pred": p,
                     "atoks": toks, "succ": 1 if succ else 0, "use": use,
                     "aval": actual_value, "accept": p["accept"]})
        print(f"  {t['id']:<16} pred p_succ={p['p_succ']:.2f} p_use={p['p_use']:.2f} "
              f"ptoks={p['tokens']:<4} | real succ={rows[-1]['succ']} use={use:.2f} atoks={toks}")

    # ---- BAR 1: cost awareness (median abs % token error) ----
    tok_err = [abs(r["pred"]["tokens"] - r["atoks"]) / max(1, r["atoks"]) for r in rows]
    bar1_val = statistics.median(tok_err); bar1 = bar1_val <= 0.25

    # ---- BAR 2: success calibration ----
    succ_err = statistics.mean(abs(r["pred"]["p_succ"] - r["succ"]) for r in rows)
    hi = [r for r in rows if r["pred"]["p_succ"] >= 0.6]
    lo = [r for r in rows if r["pred"]["p_succ"] < 0.6]
    hi_rate = statistics.mean(r["succ"] for r in hi) if hi else 0
    lo_rate = statistics.mean(r["succ"] for r in lo) if lo else 0
    bar2 = succ_err <= 0.20 and hi_rate > lo_rate

    # ---- BAR 3: usefulness calibration ----
    use_err = statistics.mean(abs(r["pred"]["p_use"] - r["use"]) for r in rows)
    bar3 = use_err <= 0.20

    # ---- BAR 4: value awareness (rank corr pred EV vs actual value) ----
    rho = spearman([r["pred"]["ev"] for r in rows], [r["aval"] for r in rows])
    top = sorted(rows, key=lambda r: r["pred"]["ev"], reverse=True)[:len(rows)//2]
    bot = sorted(rows, key=lambda r: r["pred"]["ev"], reverse=True)[len(rows)//2:]
    bar4 = rho >= 0.3 and (statistics.mean(r["aval"] for r in top) > statistics.mean(r["aval"] for r in bot))

    # ---- BAR 5: decision usefulness (value-per-cost vs baselines) ----
    def vpc(subset):
        c = sum(r["atoks"]/1000*COST_PER_1K + INFRA for r in subset)
        v = sum(r["aval"] for r in subset)
        return v / c if c else 0
    neuron_set = [r for r in rows if r["accept"]]
    import random as _rnd; _rnd.seed(0)
    rand_set = [r for r in rows if _rnd.random() < 0.7] or rows
    vpc_neuron, vpc_all, vpc_rand = vpc(neuron_set), vpc(rows), vpc(rand_set)
    bar5 = vpc_neuron > vpc_all and vpc_neuron >= vpc_rand

    # ---- ANTI-CHEAT ----
    psuccs = [r["pred"]["p_succ"] for r in rows]
    spread_ok = (max(psuccs) - min(psuccs)) >= 0.3 and not all(p >= 0.9 for p in psuccs)
    cats = {r["cat"] for r in rows}
    mix_ok = {"coding", "numeric", "analysis", "explanation", "planning", "impossible"} <= cats

    print("\n" + "="*64); print("  BARS (frozen before run)"); print("="*64)
    def line(ok, name, detail): print(f"    {'PASS' if ok else 'FAIL'}  {name:<34} {detail}")
    line(bar1, "1 cost awareness",        f"median tok err {bar1_val*100:.0f}%  (<=25%)")
    line(bar2, "2 success calibration",   f"abs err {succ_err:.2f} (<=0.20), hi {hi_rate:.2f} > lo {lo_rate:.2f}")
    line(bar3, "3 usefulness calibration",f"abs err {use_err:.2f}  (<=0.20)")
    line(bar4, "4 value awareness",       f"rank corr {rho:.2f} (>=0.3), top>bot {statistics.mean(r['aval'] for r in top):.2f}>{statistics.mean(r['aval'] for r in bot):.2f}")
    line(bar5, "5 decision usefulness",   f"v/c neuron {vpc_neuron:.1f} vs all {vpc_all:.1f} vs rand {vpc_rand:.1f}")
    print("    " + "-"*56)
    line(spread_ok, "anti-cheat: prediction spread", f"range {max(psuccs)-min(psuccs):.2f}, not all>=0.9")
    line(mix_ok,    "anti-cheat: category mix",       f"{sorted(cats)}")

    bars = [bar1, bar2, bar3, bar4, bar5, spread_ok, mix_ok]
    allok = all(bars)
    print("\n  " + ("STEP 4 PASS - resource awareness is calibrated enough to guide decisions."
                     if allok else "STEP 4 NOT YET - calibration misses on the FAILed bars above (honest park)."))
    if MOCK:
        print("  (MOCK run: structural only. Real calibration = run against Qwen.)")

if __name__ == "__main__":
    main()
