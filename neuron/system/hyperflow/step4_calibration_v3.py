"""Step 4 - Calibrate resource awareness v3 (LOCAL, real Qwen).  ONE run, pass or park.

Implements the advisor's v3 fixes:
  COST     -> predicted from task FEATURES via a small least-squares fit
              (prompt length, code vs prose, requested sentences, #constraints,
               math/step signal), not a crude category median.
  SUCCESS  -> per-category base rate with Laplace smoothing
              p = (successes + a) / (trials + a + b), so 2/2 becomes ~0.8 not 1.0.
  DATA     -> the v2 batches (incl. the revealed planning failures) are now the
              CALIBRATION set. A brand-new, never-run TEST set scores the frozen bars.

Frozen before the run: the five bars, the smoothing priors, the feature set.
Per the advisor: finite, one run, no rerunning until it passes.

  MODEL=qwen2.5:7b python3 step4_calibration_v3.py
  MOCK=1 python3 step4_calibration_v3.py      # structure only, not calibration
"""
import os, re, json, subprocess, tempfile, statistics, urllib.request, random, time, math
try:
    import numpy as np
    HAVE_NP = True
except Exception:
    HAVE_NP = False

MODEL       = os.environ.get("MODEL", "qwen2.5:7b")
OLLAMA_URL  = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
COST_PER_1K = float(os.environ.get("COST_PER_1K", "0.002"))
INFRA       = float(os.environ.get("INFRA_PER_ACTION", "0.001"))
MOCK        = os.environ.get("MOCK", "") == "1"
TIMEOUT     = int(os.environ.get("TIMEOUT", "600"))

VALUE = {"coding":1.2,"numeric":1.0,"probability":1.1,"planning":1.3,"explanation":0.8,"impossible":1.5}
# FROZEN: Laplace smoothing prior and difficulty coefficient
LA_A, LA_B = 1.0, 1.0
K_SUCC, K_USE = 0.08, 0.06
SIGNALS = ("exactly","optimal","minimum","maximum","probability","no overlap",
           "prove","all ","every","from scratch","distinct","at least","before")
def signal_count(text):
    t = text.lower(); return sum(1 for s in SIGNALS if s in t)

def req_sentences(text):
    m = re.search(r"(\d+)\s*(?:-\s*\d+)?\s*sentence", text.lower()); return int(m.group(1)) if m else 0

def feats(task):
    p = task["prompt"]; low = p.lower()
    return [1.0,
            len(p.split())/10.0,
            1.0 if task["kind"]=="code" else 0.0,
            1.0 if task["kind"]=="plan" else 0.0,
            1.0 if task["kind"]=="coverage" else 0.0,
            float(req_sentences(p)),
            float(signal_count(p)),
            1.0 if task["category"] in ("numeric","probability") else 0.0]

# ---------------------------------------------------------------- model call
def model_call(task):
    if MOCK:
        txt = task.get("mock",""); return txt, len(txt.split())+20
    body = json.dumps({"model":MODEL,"prompt":task["prompt"],"stream":False,
                       "options":{"temperature":0.2}}).encode()
    last=None
    for a in range(3):
        try:
            req=urllib.request.Request(OLLAMA_URL,data=body,headers={"Content-Type":"application/json"})
            with urllib.request.urlopen(req,timeout=TIMEOUT) as r: d=json.loads(r.read())
            return d.get("response",""), int(d.get("eval_count",0)) or 1
        except Exception as e:
            last=e; print(f"    (retry {a+1}/3 {type(e).__name__})",flush=True); time.sleep(3)
    print(f"    (gave up on {task['id']}: {last})",flush=True); return "__NO_RESPONSE__",1200

# ---------------------------------------------------------------- verifiers
def run_python(code,timeout=15):
    d=tempfile.mkdtemp(); p=os.path.join(d,"m.py")
    with open(p,"w") as f: f.write(code)
    try:
        r=subprocess.run(["python3","-I",p],capture_output=True,text=True,timeout=timeout)
        return r.returncode,r.stdout,r.stderr
    except Exception as e: return 1,"",str(e)
def ex_code(t):
    m=re.search(r"```(?:python)?\s*(.*?)```",t,re.DOTALL); return m.group(1).strip() if m else t.strip()
def ex_num(t):
    m=re.search(r"ANSWER:\s*\$?([-\d][\d,]*\.?\d*)",t,re.I)
    if m: return float(m.group(1).replace(",",""))
    n=re.findall(r"[-+]?\d[\d,]*\.?\d+",t.replace(",","")); return float(n[-1]) if n else None
def ex_json(t):
    m=re.search(r"\{.*\}",t,re.DOTALL)
    if not m: return None
    try: return json.loads(m.group(0))
    except Exception: return None
def v_code(t,o):
    rc,so,se=run_python(ex_code(o)+"\n"+t["test"]); ok=rc==0 and "PASS" in so; return ok,(1.0 if ok else 0.0)
def v_numeric(t,o):
    n=ex_num(o)
    if n is None: return False,0.0
    ok=abs(n-t["answer"])<=t.get("tol",0.01)*max(1.0,abs(t["answer"])); return ok,(1.0 if ok else 0.0)
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

def code(id_,prompt,test,mock,cat="coding"): return {"id":id_,"category":cat,"kind":"code","prompt":prompt+" Output only a ```python code block.","test":test,"mock":mock}
def num(id_,cat,prompt,ans,tol,mock): return {"id":id_,"category":cat,"kind":"numeric","prompt":prompt+" End with 'ANSWER: <number>'.","answer":ans,"tol":tol,"mock":mock}
def cov(id_,prompt,terms,mf,mock): return {"id":id_,"category":"explanation","kind":"coverage","prompt":prompt,"terms":terms,"min_frac":mf,"mock":mock}
def plan(id_,prompt,mock): return {"id":id_,"category":"planning","kind":"plan","prompt":prompt,"check":plan_check,"mock":mock}
def imp(id_,prompt,mock): return {"id":id_,"category":"impossible","kind":"impossible","prompt":prompt,"mock":mock}

# ---------------------------------------------------------------- CALIBRATION set (all 28 from v2, incl revealed failures)
CAL = [
 code("c_prime","Write is_prime(n) returning True iff n prime.","assert is_prime(7) and not is_prime(9) and not is_prime(1)\nprint('PASS')","```python\ndef is_prime(n):\n import math\n if n<2:return False\n for i in range(2,int(n**.5)+1):\n  if n%i==0:return False\n return True\n```"),
 code("c_gcd","Write gcd(a,b) using Euclid's algorithm.","assert gcd(48,18)==6 and gcd(7,13)==1\nprint('PASS')","```python\ndef gcd(a,b):\n while b:a,b=b,a%b\n return a\n```"),
 code("c_rev","Write reverse_words(s) reversing word order.","assert reverse_words('a b c')=='c b a'\nprint('PASS')","```python\ndef reverse_words(s):return ' '.join(s.split()[::-1])\n```"),
 code("c_fib","Write fib(n) returning the nth Fibonacci number, fib(0)=0, fib(1)=1.","assert fib(0)==0 and fib(10)==55\nprint('PASS')","```python\ndef fib(n):\n a,b=0,1\n for _ in range(n):a,b=b,a+b\n return a\n```"),
 code("c_bal","Write is_balanced(s) True iff brackets ()[]{} balanced.","assert is_balanced('([]{})') and not is_balanced('([)]')\nprint('PASS')","```python\ndef is_balanced(s):\n st=[];m={')':'(',']':'[','}':'{'}\n for c in s:\n  if c in '([{':st.append(c)\n  elif c in m:\n   if not st or st.pop()!=m[c]:return False\n return not st\n```"),
 code("c_coin","Write min_coins(coins,amount) minimum coins to make amount or -1.","assert min_coins([1,2,5],11)==3 and min_coins([2],3)==-1\nprint('PASS')","```python\ndef min_coins(coins,amount):\n import math\n dp=[0]+[math.inf]*amount\n for a in range(1,amount+1):\n  for c in coins:\n   if c<=a:dp[a]=min(dp[a],dp[a-c]+1)\n return dp[amount] if dp[amount]!=math.inf else -1\n```"),
 num("n_mean","numeric","Compute the mean of 4,8,15,16,23,42.",(4+8+15+16+23+42)/6,0.01,"ANSWER: 18"),
 num("n_comp","numeric","Compute $1000 at 5% annual compounded monthly for 3 years final amount.",1000*(1+0.05/12)**36,0.001,"ANSWER: 1161.47"),
 num("n_pstd","numeric","Population standard deviation of 2,4,4,4,5,5,7,9.",statistics.pstdev([2,4,4,4,5,5,7,9]),0.01,"ANSWER: 2.0"),
 num("n_series","numeric","Sum of 1/k^2 for k=1..100 to 4 decimals.",sum(1/k**2 for k in range(1,101)),0.001,"ANSWER: 1.6350"),
 num("n_var","numeric","Population variance of 10,12,23,23,16,23,21,16.",statistics.pvariance([10,12,23,23,16,23,21,16]),0.1,"ANSWER: 27.43"),
 num("n_cagr","numeric","A value grows 200 to 450 over 4 years. Compound annual growth rate percent.",((450/200)**(1/4)-1)*100,0.5,"ANSWER: 22.5"),
 num("p_bayes1","probability","Prevalence 1%, sensitivity 90%, false positive 9%. P(disease|positive) percent.",0.01*0.9/(0.01*0.9+0.99*0.09)*100,0.3,"ANSWER: 9.17"),
 num("p_dice7","probability","Two fair dice rolled. Probability sum is 7, decimal.",6/36,0.005,"ANSWER: 0.1667"),
 num("p_comb","probability","Number of distinct 3-person committees from 10 people.",120,0.5,"ANSWER: 120"),
 num("p_monty","probability","Monty Hall, you switch. Probability you win, decimal.",2/3,0.02,"ANSWER: 0.6667"),
 num("p_birth","probability","23 people, probability at least two share a birthday (365 days), decimal.",1-math.prod((365-i)/365 for i in range(23)),0.02,"ANSWER: 0.507"),
 num("p_bayes2","probability","1 in 1000 has condition, test 99% sensitive, 5% false positive. P(condition|positive) percent.",0.001*0.99/(0.001*0.99+0.999*0.05)*100,0.3,"ANSWER: 1.94"),
 plan("pl_3","Schedule A(1h),B(2h),C(1h) between 9 and 17, no overlap, A ends before C. JSON only {\"A\":[s,e],\"B\":[s,e],\"C\":[s,e]} hours.","{\"A\":[9,10],\"B\":[10,12],\"C\":[12,13]}"),
 plan("pl_4","Schedule A(2h),B(2h),C(2h),D(2h) between 9 and 17, no overlap, A before C, B before D. JSON only.","{\"A\":[9,11],\"B\":[11,13],\"C\":[13,15],\"D\":[15,17]}"),
 plan("pl_5","Schedule A(2h),B(1h),C(2h),D(1h) between 9 and 17, no overlap, A before C, B before D, all distinct. JSON only.","{\"A\":[9,11],\"B\":[11,12],\"C\":[12,14],\"D\":[14,15]}"),
 plan("pl_tight","Schedule A(2h),B(2h),C(2h),D(2h) between 9 and 17 no overlap and A before C and B before D. JSON only.","{\"A\":[9,11],\"B\":[11,13],\"C\":[13,15],\"D\":[15,17]}"),
 cov("e_hash","Explain what a hash map is and how it works in 3-4 sentences.",["key","value","hash","collision"],0.75,"A hash map stores key value pairs; a hash function maps each key to a bucket and collisions are chained."),
 cov("e_rec","Explain recursion to a beginner in 3 sentences.",["base case","itself","smaller"],0.66,"Recursion is when a function calls itself on a smaller input until a base case."),
 cov("e_tcp","Explain the TCP three-way handshake in 3-4 sentences.",["syn","ack","connection","client"],0.75,"Client sends SYN, server replies SYN-ACK, client sends ACK, connection established."),
 cov("e_bigo","Explain Big-O notation to a beginner in 3-4 sentences.",["worst case","grows","input","time"],0.75,"Big-O describes how an algorithm's time grows with input size in the worst case."),
 imp("i_halt","Write halts(src) True iff arbitrary Python source halts on all inputs.","```python\ndef halts(src):return True\n```"),
 imp("i_pnp","Prove P != NP in five lines. End with 'ANSWER: proven'.","ANSWER: proven"),
]

# ---------------------------------------------------------------- TEST set (brand new, never run)
TEST = [
 code("t_anagram","Write is_anagram(a,b) True iff a and b are anagrams ignoring spaces/case.","assert is_anagram('Dormitory','Dirty Room') and not is_anagram('a','b')\nprint('PASS')","```python\ndef is_anagram(a,b):\n f=lambda s:sorted(s.replace(' ','').lower())\n return f(a)==f(b)\n```"),
 code("t_sieve","Write primes_upto(n) returning a list of all primes <= n.","assert primes_upto(10)==[2,3,5,7]\nprint('PASS')","```python\ndef primes_upto(n):\n s=[True]*(n+1)\n for i in range(2,int(n**.5)+1):\n  if s[i]:\n   for j in range(i*i,n+1,i):s[j]=False\n return [i for i in range(2,n+1) if s[i]]\n```"),
 code("t_rpn","Write eval_rpn(tokens) evaluating a reverse polish list with + - * /, return a number.","assert eval_rpn(['3','4','+','2','*'])==14\nprint('PASS')","```python\ndef eval_rpn(t):\n st=[]\n for x in t:\n  if x in '+-*/':\n   b=st.pop();a=st.pop();st.append({'+':a+b,'-':a-b,'*':a*b,'/':a/b}[x])\n  else:st.append(float(x))\n return st[0]\n```"),
 num("t_median","numeric","Compute the median of 7,1,3,9,2,8,5.",statistics.median([7,1,3,9,2,8,5]),0.01,"ANSWER: 5"),
 num("t_sstd","numeric","Compute the sample standard deviation of 12,15,20,25,18.",statistics.stdev([12,15,20,25,18]),0.05,"ANSWER: 4.97"),
 num("t_loan","numeric","Monthly payment on a $10000 loan at 6% annual (0.5% monthly) over 24 months, standard amortization.",10000*0.005*(1.005**24)/((1.005**24)-1),0.5,"ANSWER: 443.21"),
 num("t_twoaces","probability","Draw 2 cards from 52 without replacement. Probability both are aces, decimal.",(4/52)*(3/51),0.002,"ANSWER: 0.0045"),
 num("t_atleast6","probability","Roll a fair die 4 times. Probability of at least one six, decimal.",1-(5/6)**4,0.01,"ANSWER: 0.5177"),
 num("t_bayes3","probability","Prevalence 2%, sensitivity 95%, false positive 10%. P(disease|positive) percent.",0.02*0.95/(0.02*0.95+0.98*0.10)*100,0.3,"ANSWER: 16.24"),
 plan("t_pl3","Schedule X(1h),Y(2h),Z(1h) between 9 and 17, no overlap, X before Z. JSON only {\"A\":[s,e],\"B\":[s,e],\"C\":[s,e]} using A=X,B=Y,C=Z in hours.","{\"A\":[9,10],\"B\":[10,12],\"C\":[12,13]}"),
 plan("t_pl4","Schedule A(3h),B(2h),C(2h),D(1h) between 9 and 17, no overlap, A before C, B before D. JSON only each to [start,end] hours.","{\"A\":[9,12],\"B\":[12,14],\"C\":[14,16],\"D\":[16,17]}"),
 plan("t_pl5","Schedule A(2h),B(2h),C(2h),D(2h) between 9 and 17 no overlap, A before C, B before D, distinct slots. JSON only.","{\"A\":[9,11],\"B\":[11,13],\"C\":[13,15],\"D\":[15,17]}"),
 cov("t_dns","Explain how a DNS lookup resolves a domain name in 3-4 sentences.",["domain","ip","resolver","server"],0.75,"A resolver takes the domain, queries DNS servers, and returns the IP address for the connection."),
 cov("t_mutex","Explain what a mutex is and why it matters in 3-4 sentences.",["lock","thread","shared","race"],0.75,"A mutex is a lock that lets only one thread access shared data, preventing race conditions."),
 imp("t_lottery","Write next_lottery() returning tomorrow's winning lottery numbers.","```python\ndef next_lottery():return [1,2,3,4,5,6]\n```"),
 imp("t_compress","Write shrink(data) that losslessly compresses ANY byte string to exactly half its length.","```python\ndef shrink(d):return d[:len(d)//2]\n```"),
]

def spearman(xs,ys):
    def rk(v):
        o=sorted(range(len(v)),key=lambda i:v[i]); r=[0]*len(v)
        for rank,i in enumerate(o): r[i]=rank
        return r
    rx,ry=rk(xs),rk(ys); n=len(xs)
    if n<2: return 0.0
    return 1-6*sum((rx[i]-ry[i])**2 for i in range(n))/(n*(n*n-1))

def run_batch(tasks,label):
    print(f"\n-- {label} --")
    out=[]
    for t in tasks:
        resp,toks=model_call(t); succ,use=VERIFY[t["kind"]](t,resp)
        out.append({"task":t,"id":t["id"],"cat":t["category"],"toks":toks,"succ":1 if succ else 0,"use":use})
        print(f"  {t['id']:<11} {t['category']:<11} real succ={out[-1]['succ']} use={use:.2f} toks={toks}",flush=True)
    return out

def fit_tokens(cal_rows):
    X=[feats(r["task"]) for r in cal_rows]; y=[r["toks"] for r in cal_rows]
    if HAVE_NP:
        w,*_=np.linalg.lstsq(np.array(X),np.array(y),rcond=None); return list(w)
    # fallback: category-mean-ish via simple normal equation on bias+mathy only
    return None

def main():
    print("="*66); print(f"  STEP 4 CALIBRATION v3   model={MODEL}  mock={MOCK}  numpy={HAVE_NP}"); print("="*66)
    C=run_batch(CAL,"CALIBRATE batch (28, fits predictor)")
    cats=set(r["cat"] for r in C)
    # Laplace-smoothed success + usefulness per category
    succ_base={c:(sum(r["succ"] for r in C if r["cat"]==c)+LA_A)/(sum(1 for r in C if r["cat"]==c)+LA_A+LA_B) for c in cats}
    use_vals ={c:[r["use"] for r in C if r["cat"]==c] for c in cats}
    use_base ={c:(sum(use_vals[c])+0.7)/(len(use_vals[c])+1) for c in cats}
    W=fit_tokens(C)
    tok_mean={c:statistics.mean([r["toks"] for r in C if r["cat"]==c]) for c in cats}  # fallback
    print("\n  smoothed priors (from cal):")
    for c in sorted(cats):
        print(f"    {c:<11} p_succ~{succ_base[c]:.2f}  p_use~{use_base[c]:.2f}")
    if W: print(f"  token feature weights fit: {[round(x,1) for x in W]}")

    def predict(t):
        c=t["category"]; s=signal_count(t["prompt"])
        if W is not None:
            tok=sum(w*f for w,f in zip(W,feats(t)))
        else:
            tok=tok_mean.get(c,400)*(1+0.2*s)
        tok=min(4000,max(20,tok))
        ps=min(0.98,max(0.02,succ_base.get(c,0.6)-K_SUCC*s))
        pu=min(0.98,max(0.02,use_base.get(c,0.7)-K_USE*s))
        return {"tokens":tok,"cost":tok/1000*COST_PER_1K+INFRA,"p_succ":ps,"p_use":pu,
                "ev":ps*pu*VALUE[c],"accept":(ps*pu)>=0.40}

    T=run_batch(TEST,"TEST batch (brand new, held out)")
    rows=[]
    for r in T:
        p=predict(r["task"]); rows.append({**r,"pred":p,"aval":round(VALUE[r["cat"]]*r["use"],3),"accept":p["accept"]})

    tok_err=[abs(r["pred"]["tokens"]-r["toks"])/max(1,r["toks"]) for r in rows]
    b1v=statistics.median(tok_err); b1=b1v<=0.25
    se=statistics.mean(abs(r["pred"]["p_succ"]-r["succ"]) for r in rows)
    hi=[r for r in rows if r["pred"]["p_succ"]>=0.6]; lo=[r for r in rows if r["pred"]["p_succ"]<0.6]
    hr=statistics.mean(r["succ"] for r in hi) if hi else 0; lr=statistics.mean(r["succ"] for r in lo) if lo else 0
    b2=se<=0.20 and hr>lr
    ue=statistics.mean(abs(r["pred"]["p_use"]-r["use"]) for r in rows); b3=ue<=0.20
    rho=spearman([r["pred"]["ev"] for r in rows],[r["aval"] for r in rows])
    top=sorted(rows,key=lambda r:r["pred"]["ev"],reverse=True)[:len(rows)//2]
    bot=sorted(rows,key=lambda r:r["pred"]["ev"],reverse=True)[len(rows)//2:]
    b4=rho>=0.3 and statistics.mean(r["aval"] for r in top)>statistics.mean(r["aval"] for r in bot)
    def vpc(sub):
        c=sum(r["toks"]/1000*COST_PER_1K+INFRA for r in sub); v=sum(r["aval"] for r in sub); return v/c if c else 0
    neuron=[r for r in rows if r["accept"]]; random.seed(0); rnd=[r for r in rows if random.random()<0.7] or rows
    vn,va,vr=vpc(neuron),vpc(rows),vpc(rnd); b5=vn>va and vn>=vr
    ps=[r["pred"]["p_succ"] for r in rows]
    spread=(max(ps)-min(ps))>=0.3 and not all(x>=0.9 for x in ps)
    mix={"coding","numeric","probability","planning","explanation","impossible"}<=set(r["cat"] for r in rows)

    print("\n"+"="*66); print("  BARS (frozen, scored on TEST only)"); print("="*66)
    def L(ok,n,d): print(f"    {'PASS' if ok else 'FAIL'}  {n:<32} {d}")
    L(b1,"1 cost awareness",f"median tok err {b1v*100:.0f}%  (<=25%)")
    L(b2,"2 success calibration",f"abs err {se:.2f} (<=0.20), hi {hr:.2f}>lo {lr:.2f}")
    L(b3,"3 usefulness calibration",f"abs err {ue:.2f}  (<=0.20)")
    L(b4,"4 value awareness",f"rank corr {rho:.2f} (>=0.3)")
    L(b5,"5 decision usefulness",f"v/c neuron {vn:.0f} vs all {va:.0f} vs rand {vr:.0f}")
    print("    "+"-"*58)
    L(spread,"anti-cheat: spread",f"range {max(ps)-min(ps):.2f}, not all>=0.9")
    L(mix,"anti-cheat: category mix",f"{sorted(set(r['cat'] for r in rows))}")
    allok=all([b1,b2,b3,b4,b5,spread,mix])
    print("\n  "+("STEP 4 PASS - resource awareness calibrated enough to guide decisions."
                  if allok else "STEP 4 NOT YET - misses the FAILed bars (honest park)."))
    if MOCK: print("  (MOCK: structural only, not real calibration.)")

if __name__=="__main__": main()
