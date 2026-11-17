"""Step 4 - Calibrate resource awareness v2 (LOCAL, real Qwen).

Fixes the two problems v1 exposed:
  1. token/cost prediction was a fixed guess (330% off) -> now FIT from data
  2. the batch was too easy to test success calibration -> harder battery
     weighted toward what a 7b model actually fails (Bayes, probability,
     multi-constraint planning), so real success variance exists and a flat
     "90% for everything" predictor FAILS instead of sneaking a hollow pass.

Method (honest, standard calibration):
  CALIBRATE batch -> fit per-category token mean, success rate, usefulness rate
  TEST batch (held out) -> predict with the fitted model, run Qwen, verify,
                           score the five FROZEN bars on TEST only.
The five bars and the fitting coefficients are frozen here before the run.

  MODEL=qwen2.5:7b python3 step4_calibration_v2.py
  MOCK=1 python3 step4_calibration_v2.py          # structure only, not calibration
"""
import os, re, json, subprocess, tempfile, statistics, urllib.request, random, time

MODEL       = os.environ.get("MODEL", "qwen2.5:7b")
OLLAMA_URL  = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
COST_PER_1K = float(os.environ.get("COST_PER_1K", "0.002"))
INFRA       = float(os.environ.get("INFRA_PER_ACTION", "0.001"))
MOCK        = os.environ.get("MOCK", "") == "1"

VALUE = {"coding":1.2,"numeric":1.0,"probability":1.1,"planning":1.3,"explanation":0.8,"impossible":1.5}
# FROZEN difficulty coefficients (how prompt signals bend the fitted base rates)
SIGNALS = ("exactly","optimal","minimum","maximum","probability","no overlap",
           "prove","all ","every","from scratch","distinct","at least")
K_SUCC, K_TOK, K_USE = 0.11, 0.22, 0.07   # per-signal effect, frozen

def signal_count(text):
    t = text.lower()
    return sum(1 for s in SIGNALS if s in t)

# ---------------------------------------------------------------- model call
TIMEOUT = int(os.environ.get("TIMEOUT","600"))   # seconds per call
def model_call(task):
    if MOCK:
        txt = task.get("mock","")
        return txt, len(txt.split()) + 20
    body = json.dumps({"model":MODEL,"prompt":task["prompt"],"stream":False,
                       "options":{"temperature":0.2}}).encode()
    last = None
    for attempt in range(3):   # retry transient slowness instead of crashing
        try:
            req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type":"application/json"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                d = json.loads(r.read())
            return d.get("response",""), int(d.get("eval_count",0)) or 1
        except Exception as e:
            last = e
            print(f"    (retry {attempt+1}/3 after {type(e).__name__})", flush=True)
            time.sleep(3)
    # genuinely could not get a response: a real failed+expensive outcome
    print(f"    (gave up on {task['id']}: {last})", flush=True)
    return "__NO_RESPONSE__", 1200

# ---------------------------------------------------------------- verifiers
def run_python(code, timeout=15):
    d = tempfile.mkdtemp(); p = os.path.join(d,"m.py")
    with open(p,"w") as f: f.write(code)
    try:
        r = subprocess.run(["python3","-I",p], capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except Exception as e:
        return 1,"",str(e)

def ex_code(t):
    m = re.search(r"```(?:python)?\s*(.*?)```", t, re.DOTALL); return m.group(1).strip() if m else t.strip()
def ex_num(t):
    m = re.search(r"ANSWER:\s*\$?([-\d][\d,]*\.?\d*)", t, re.I)
    if m: return float(m.group(1).replace(",",""))
    n = re.findall(r"[-+]?\d[\d,]*\.?\d+", t.replace(",","")); return float(n[-1]) if n else None
def ex_json(t):
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m: return None
    try: return json.loads(m.group(0))
    except Exception: return None

def v_code(t,o):
    rc,so,se = run_python(ex_code(o)+"\n"+t["test"]); ok = rc==0 and "PASS" in so; return ok,(1.0 if ok else 0.0)
def v_numeric(t,o):
    n = ex_num(o)
    if n is None: return False,0.0
    ok = abs(n-t["answer"]) <= t.get("tol",0.01)*max(1.0,abs(t["answer"])); return ok,(1.0 if ok else 0.0)
def v_coverage(t,o):
    low=o.lower(); frac=sum(1 for w in t["terms"] if w in low)/len(t["terms"]); return frac>=t.get("min_frac",0.6),round(frac,2)
def v_plan(t,o):
    j=ex_json(o)
    if not j: return False,0.0
    try: ok=t["check"](j)
    except Exception: ok=False
    return ok,(1.0 if ok else 0.0)
def v_imp(t,o): return False,0.0
VERIFY={"code":v_code,"numeric":v_numeric,"coverage":v_coverage,"plan":v_plan,"impossible":v_imp}

def plan_check(j):
    s=j.get("schedule",j); iv={k:tuple(v) for k,v in s.items()}
    for k,(a,b) in iv.items():
        if a<9 or b>17 or b<=a: return False
    nm=list(iv)
    for i in range(len(nm)):
        for k in range(i+1,len(nm)):
            a1,b1=iv[nm[i]]; a2,b2=iv[nm[k]]
            if a1<b2 and a2<b1: return False
    if "A" in iv and "C" in iv and not iv["A"][1]<=iv["C"][0]: return False
    if "B" in iv and "D" in iv and not iv["B"][1]<=iv["D"][0]: return False
    return True

# ---------------------------------------------------------------- battery
# split: "cal" fits the predictor, "test" scores it. Weighted so probability
# and planning (where a 7b model genuinely fails) create real success variance.
def code(id_,sp,prompt,test,mock): return {"id":id_,"split":sp,"category":"coding","kind":"code","prompt":prompt+" Output only a ```python code block.","test":test,"mock":mock}
def num(id_,sp,cat,prompt,ans,tol,mock): return {"id":id_,"split":sp,"category":cat,"kind":"numeric","prompt":prompt+" End with 'ANSWER: <number>'.","answer":ans,"tol":tol,"mock":mock}
def cov(id_,sp,prompt,terms,mf,mock): return {"id":id_,"split":sp,"category":"explanation","kind":"coverage","prompt":prompt,"terms":terms,"min_frac":mf,"mock":mock}
def plan(id_,sp,prompt,mock): return {"id":id_,"split":sp,"category":"planning","kind":"plan","prompt":prompt,"check":plan_check,"mock":mock}
def imp(id_,sp,prompt,mock): return {"id":id_,"split":sp,"category":"impossible","kind":"impossible","prompt":prompt,"mock":mock}

import math
TASKS = [
 # coding (anchor: usually high success)
 code("c_prime","cal","Write is_prime(n) returning True iff n prime.","assert is_prime(7) and not is_prime(9) and not is_prime(1)\nprint('PASS')","```python\ndef is_prime(n):\n import math\n if n<2:return False\n for i in range(2,int(n**.5)+1):\n  if n%i==0:return False\n return True\n```"),
 code("c_gcd","cal","Write gcd(a,b) using Euclid's algorithm.","assert gcd(48,18)==6 and gcd(7,13)==1\nprint('PASS')","```python\ndef gcd(a,b):\n while b:a,b=b,a%b\n return a\n```"),
 code("c_rev","cal","Write reverse_words(s) reversing word order.","assert reverse_words('a b c')=='c b a'\nprint('PASS')","```python\ndef reverse_words(s):return ' '.join(s.split()[::-1])\n```"),
 code("c_fib","test","Write fib(n) returning the nth Fibonacci number, fib(0)=0, fib(1)=1.","assert fib(0)==0 and fib(1)==1 and fib(10)==55\nprint('PASS')","```python\ndef fib(n):\n a,b=0,1\n for _ in range(n):a,b=b,a+b\n return a\n```"),
 code("c_bal","test","Write is_balanced(s) returning True iff brackets ()[]{} are balanced.","assert is_balanced('([]{})') and not is_balanced('([)]')\nprint('PASS')","```python\ndef is_balanced(s):\n st=[];m={')':'(',']':'[','}':'{'}\n for c in s:\n  if c in '([{':st.append(c)\n  elif c in m:\n   if not st or st.pop()!=m[c]:return False\n return not st\n```"),
 code("c_coin","test","Write min_coins(coins,amount) returning the minimum coins to make amount, or -1.","assert min_coins([1,2,5],11)==3 and min_coins([2],3)==-1\nprint('PASS')","```python\ndef min_coins(coins,amount):\n import math\n dp=[0]+[math.inf]*amount\n for a in range(1,amount+1):\n  for c in coins:\n   if c<=a:dp[a]=min(dp[a],dp[a-c]+1)\n return dp[amount] if dp[amount]!=math.inf else -1\n```"),
 # numeric arithmetic (anchor: high)
 num("n_mean","cal","numeric","Compute the mean of 4,8,15,16,23,42.",(4+8+15+16+23+42)/6,0.01,"ANSWER: 18"),
 num("n_comp","cal","numeric","Compute $1000 at 5% annual compounded monthly for 3 years final amount.",1000*(1+0.05/12)**36,0.001,"ANSWER: 1161.47"),
 num("n_pstd","cal","numeric","Population standard deviation of 2,4,4,4,5,5,7,9.",statistics.pstdev([2,4,4,4,5,5,7,9]),0.01,"ANSWER: 2.0"),
 num("n_series","test","numeric","Sum of 1/k^2 for k=1..100 to 4 decimals.",sum(1/k**2 for k in range(1,101)),0.001,"ANSWER: 1.6350"),
 num("n_var","test","numeric","Population variance of 10,12,23,23,16,23,21,16.",statistics.pvariance([10,12,23,23,16,23,21,16]),0.1,"ANSWER: 27.43"),
 num("n_cagr","test","numeric","A value grows 200 to 450 over 4 years. Compute the compound annual growth rate as a percent.",((450/200)**(1/4)-1)*100,0.5,"ANSWER: 22.5"),
 # probability (failure-prone for a 7b model)
 num("p_bayes1","cal","probability","Disease prevalence 1%, test sensitivity 90%, false positive rate 9%. Compute P(disease | positive) as a percent.",0.01*0.9/(0.01*0.9+0.99*0.09)*100,0.3,"ANSWER: 9.17"),
 num("p_dice7","cal","probability","Two fair dice are rolled. Compute the probability the sum is 7, as a decimal.",6/36,0.005,"ANSWER: 0.1667"),
 num("p_comb","cal","probability","Compute the number of distinct 3-person committees from 10 people.",120,0.5,"ANSWER: 120"),
 num("p_monty","test","probability","Monty Hall: 3 doors, you pick one, host opens a losing door, you switch. Probability you win, as a decimal.",2/3,0.02,"ANSWER: 0.6667"),
 num("p_birth","test","probability","With 23 people, probability at least two share a birthday (365 days), as a decimal.",1-math.prod((365-i)/365 for i in range(23)),0.02,"ANSWER: 0.507"),
 num("p_bayes2","test","probability","1 in 1000 has a condition. Test is 99% sensitive, 5% false positive. Compute P(condition | positive) as a percent.",0.001*0.99/(0.001*0.99+0.999*0.05)*100,0.3,"ANSWER: 1.94"),
 # planning (failure-prone, many constraints)
 plan("pl_3","cal","Schedule A(1h),B(2h),C(1h) between 9 and 17, no overlap, A ends before C starts. Output JSON only {\"A\":[s,e],\"B\":[s,e],\"C\":[s,e]} in hours.","{\"A\":[9,10],\"B\":[10,12],\"C\":[12,13]}"),
 plan("pl_4","cal","Schedule A(2h),B(2h),C(2h),D(2h) between 9 and 17, no overlap, A before C, B before D. JSON only {\"A\":[s,e],\"B\":[s,e],\"C\":[s,e],\"D\":[s,e]} in hours.","{\"A\":[9,11],\"B\":[11,13],\"C\":[13,15],\"D\":[15,17]}"),
 plan("pl_5","test","Schedule A(2h),B(1h),C(2h),D(1h) between 9 and 17, no overlap, A before C, B before D, all distinct. JSON only mapping each to [start,end] in hours.","{\"A\":[9,11],\"B\":[11,12],\"C\":[12,14],\"D\":[14,15]}"),
 plan("pl_tight","test","Schedule A(2h),B(2h),C(2h),D(2h) between 9 and 17 with no overlap and A before C and B before D. Output JSON only mapping each to [start,end] in hours.","{\"A\":[9,11],\"B\":[11,13],\"C\":[13,15],\"D\":[15,17]}"),
 # explanation (anchor: high usefulness)
 cov("e_hash","cal","Explain what a hash map is and how it works in 3-4 sentences.",["key","value","hash","collision"],0.75,"A hash map stores key value pairs; a hash function maps each key to a bucket and collisions are chained."),
 cov("e_rec","cal","Explain recursion to a beginner in 3 sentences.",["base case","itself","smaller"],0.66,"Recursion is when a function calls itself on a smaller input until it reaches a base case."),
 cov("e_tcp","test","Explain the TCP three-way handshake in 3-4 sentences.",["syn","ack","connection","client"],0.75,"The client sends SYN, the server replies SYN-ACK, the client sends ACK, establishing the connection."),
 cov("e_bigo","test","Explain Big-O notation to a beginner in 3-4 sentences.",["worst case","grows","input","time"],0.75,"Big-O describes how an algorithm's time grows with input size in the worst case."),
 # impossible (success=0 by construction; must predict low / decline)
 imp("i_halt","cal","Write halts(src) returning True iff arbitrary Python source halts on all inputs.","```python\ndef halts(src):return True\n```"),
 imp("i_pnp","test","Prove P != NP in five lines. End with 'ANSWER: proven'.","ANSWER: proven"),
]

def spearman(xs,ys):
    def rk(v):
        o=sorted(range(len(v)),key=lambda i:v[i]); r=[0]*len(v)
        for rank,i in enumerate(o): r[i]=rank
        return r
    rx,ry=rk(xs),rk(ys); n=len(xs)
    if n<2: return 0.0
    return 1-6*sum((rx[i]-ry[i])**2 for i in range(n))/(n*(n*n-1))

def run_batch(tasks):
    out=[]
    for t in tasks:
        resp,toks=model_call(t); succ,use=VERIFY[t["kind"]](t,resp)
        out.append({"id":t["id"],"cat":t["category"],"toks":toks,"succ":1 if succ else 0,
                    "use":use,"sig":signal_count(t["prompt"]),"prompt":t["prompt"]})
        print(f"  {t['id']:<10} {t['split']:<4} {t['category']:<11} real succ={out[-1]['succ']} use={use:.2f} toks={toks}", flush=True)
    return out

def main():
    print("="*66); print(f"  STEP 4 CALIBRATION v2   model={MODEL}  mock={MOCK}"); print("="*66)
    cal=[t for t in TASKS if t["split"]=="cal"]; test=[t for t in TASKS if t["split"]=="test"]
    print("\n-- CALIBRATE batch (fitting the predictor) --")
    C=run_batch(cal)
    # FIT per-category base rates from calibration data
    cats=set(r["cat"] for r in C)
    tok_base={c:statistics.mean([r["toks"] for r in C if r["cat"]==c]) for c in cats}
    succ_base={c:statistics.mean([r["succ"] for r in C if r["cat"]==c]) for c in cats}
    use_base={c:statistics.mean([r["use"] for r in C if r["cat"]==c]) for c in cats}
    print("\n  fitted priors (from cal):")
    for c in sorted(cats):
        print(f"    {c:<11} tok~{tok_base[c]:.0f}  p_succ~{succ_base[c]:.2f}  p_use~{use_base[c]:.2f}")

    def predict(t):
        c=t["category"]; s=signal_count(t["prompt"])
        tb=tok_base.get(c,500); sb=succ_base.get(c,0.6); ub=use_base.get(c,0.7)
        tok=tb*(1+K_TOK*s); ps=min(0.98,max(0.02,sb-K_SUCC*s)); pu=min(0.98,max(0.02,ub-K_USE*s))
        cost=tok/1000*COST_PER_1K+INFRA; ev=ps*pu*VALUE[c]
        return {"tokens":tok,"cost":cost,"p_succ":ps,"p_use":pu,"ev":ev,"accept":(ps*pu)>=0.40}

    print("\n-- TEST batch (held out, frozen predictor) --")
    T=run_batch(test)
    rows=[]
    for t,r in zip(test,T):
        p=predict(t); rows.append({**r,"pred":p,"aval":round(VALUE[r["cat"]]*r["use"],3),"accept":p["accept"]})

    tok_err=[abs(r["pred"]["tokens"]-r["toks"])/max(1,r["toks"]) for r in rows]
    bar1_val=statistics.median(tok_err); bar1=bar1_val<=0.25
    succ_err=statistics.mean(abs(r["pred"]["p_succ"]-r["succ"]) for r in rows)
    hi=[r for r in rows if r["pred"]["p_succ"]>=0.6]; lo=[r for r in rows if r["pred"]["p_succ"]<0.6]
    hi_r=statistics.mean(r["succ"] for r in hi) if hi else 0; lo_r=statistics.mean(r["succ"] for r in lo) if lo else 0
    bar2=succ_err<=0.20 and hi_r>lo_r
    use_err=statistics.mean(abs(r["pred"]["p_use"]-r["use"]) for r in rows); bar3=use_err<=0.20
    rho=spearman([r["pred"]["ev"] for r in rows],[r["aval"] for r in rows])
    top=sorted(rows,key=lambda r:r["pred"]["ev"],reverse=True)[:len(rows)//2]
    bot=sorted(rows,key=lambda r:r["pred"]["ev"],reverse=True)[len(rows)//2:]
    bar4=rho>=0.3 and statistics.mean(r["aval"] for r in top)>statistics.mean(r["aval"] for r in bot)
    def vpc(sub):
        c=sum(r["toks"]/1000*COST_PER_1K+INFRA for r in sub); v=sum(r["aval"] for r in sub); return v/c if c else 0
    neuron=[r for r in rows if r["accept"]]; random.seed(0); rnd=[r for r in rows if random.random()<0.7] or rows
    vn,va,vr=vpc(neuron),vpc(rows),vpc(rnd); bar5=vn>va and vn>=vr
    ps=[r["pred"]["p_succ"] for r in rows]
    spread=(max(ps)-min(ps))>=0.3 and not all(x>=0.9 for x in ps)
    mix={"coding","numeric","probability","planning","explanation","impossible"}<=set(r["cat"] for r in rows)

    print("\n"+"="*66); print("  BARS (frozen, scored on TEST only)"); print("="*66)
    def L(ok,n,d): print(f"    {'PASS' if ok else 'FAIL'}  {n:<32} {d}")
    L(bar1,"1 cost awareness",f"median tok err {bar1_val*100:.0f}%  (<=25%)")
    L(bar2,"2 success calibration",f"abs err {succ_err:.2f} (<=0.20), hi {hi_r:.2f}>lo {lo_r:.2f}")
    L(bar3,"3 usefulness calibration",f"abs err {use_err:.2f}  (<=0.20)")
    L(bar4,"4 value awareness",f"rank corr {rho:.2f} (>=0.3)")
    L(bar5,"5 decision usefulness",f"v/c neuron {vn:.0f} vs all {va:.0f} vs rand {vr:.0f}")
    print("    "+"-"*58)
    L(spread,"anti-cheat: spread",f"range {max(ps)-min(ps):.2f}, not all>=0.9")
    L(mix,"anti-cheat: category mix",f"{sorted(set(r['cat'] for r in rows))}")
    allok=all([bar1,bar2,bar3,bar4,bar5,spread,mix])
    print("\n  "+("STEP 4 PASS - resource awareness calibrated enough to guide decisions."
                  if allok else "STEP 4 NOT YET - misses the FAILed bars (honest park)."))
    if MOCK: print("  (MOCK: structural only, not real calibration.)")

if __name__=="__main__": main()
