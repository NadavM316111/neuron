"""Step 4 - Calibrate resource awareness v4 (LOCAL, real Qwen). LAST deliberate iteration.

Advisor's v4 spec:
  COST    -> ridge-regularized regression on log(tokens) (multiplicative), features:
             category (one-hot) + prompt_length + requested_output_length +
             constraints + verification_complexity. Clamp wild predictions.
  SUCCESS -> task-level features + shrink category rate toward a GLOBAL prior so
             tiny samples never give unjustified 0%/100%.
  DATA    -> all 44 tasks run so far (v3 cal + v3 test) are now CALIBRATION.
             A brand-new, never-run TEST set scores the frozen bars. One run.

If it passes: Step 4 COMPLETE. If it narrowly misses while still beating the
decision baselines: record the honest negative, let HyperFlow supply a large
natural dataset later. No v5.

  MODEL=qwen2.5:7b python3 step4_calibration_v4.py
  MOCK=1 python3 step4_calibration_v4.py
"""
import os, re, json, subprocess, tempfile, statistics, urllib.request, random, time, math
try:
    import numpy as np; HAVE_NP=True
except Exception:
    HAVE_NP=False

MODEL=os.environ.get("MODEL","qwen2.5:7b"); OLLAMA_URL=os.environ.get("OLLAMA_URL","http://localhost:11434/api/generate")
COST_PER_1K=float(os.environ.get("COST_PER_1K","0.002")); INFRA=float(os.environ.get("INFRA_PER_ACTION","0.001"))
MOCK=os.environ.get("MOCK","")=="1"; TIMEOUT=int(os.environ.get("TIMEOUT","600")); RIDGE=float(os.environ.get("RIDGE","5.0"))

CATS=["coding","numeric","probability","planning","explanation","impossible"]
VALUE={"coding":1.2,"numeric":1.0,"probability":1.1,"planning":1.3,"explanation":0.8,"impossible":1.5}
VERIF={"code":3,"numeric":2,"plan":3,"coverage":1,"impossible":1}
SHRINK_K=2.0   # frozen: strength of shrink toward global prior
SIGNALS=("exactly","optimal","minimum","maximum","probability","no overlap","prove","all ",
         "every","from scratch","distinct","at least","before","without replacement")
def sig(t): tl=t.lower(); return sum(1 for s in SIGNALS if s in tl)
def req_sent(t):
    m=re.search(r"(\d+)\s*(?:-\s*\d+)?\s*sentence",t.lower()); return int(m.group(1)) if m else 0

def cost_feats(task):
    p=task["prompt"]
    oh=[1.0 if task["category"]==c else 0.0 for c in CATS]
    return oh+[len(p.split())/10.0, float(req_sent(p)), float(sig(p)), float(VERIF[task["kind"]])]

# ---------------------------------------------------------------- model + verify (same infra)
def model_call(task):
    if MOCK: txt=task.get("mock",""); return txt,len(txt.split())+20
    body=json.dumps({"model":MODEL,"prompt":task["prompt"],"stream":False,"options":{"temperature":0.2}}).encode()
    last=None
    for a in range(3):
        try:
            req=urllib.request.Request(OLLAMA_URL,data=body,headers={"Content-Type":"application/json"})
            with urllib.request.urlopen(req,timeout=TIMEOUT) as r: d=json.loads(r.read())
            return d.get("response",""),int(d.get("eval_count",0)) or 1
        except Exception as e:
            last=e; print(f"    (retry {a+1}/3 {type(e).__name__})",flush=True); time.sleep(3)
    print(f"    (gave up {task['id']}: {last})",flush=True); return "__NO_RESPONSE__",1200

def run_python(code,timeout=15):
    d=tempfile.mkdtemp(); p=os.path.join(d,"m.py")
    with open(p,"w") as f: f.write(code)
    try:
        r=subprocess.run(["python3","-I",p],capture_output=True,text=True,timeout=timeout); return r.returncode,r.stdout,r.stderr
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
def code(i,p,test,mock): return {"id":i,"category":"coding","kind":"code","prompt":p+" Output only a ```python code block.","test":test,"mock":mock}
def num(i,cat,p,a,tol,mock): return {"id":i,"category":cat,"kind":"numeric","prompt":p+" End with 'ANSWER: <number>'.","answer":a,"tol":tol,"mock":mock}
def cov(i,p,terms,mf,mock): return {"id":i,"category":"explanation","kind":"coverage","prompt":p,"terms":terms,"min_frac":mf,"mock":mock}
def plan(i,p,mock): return {"id":i,"category":"planning","kind":"plan","prompt":p,"check":plan_check,"mock":mock}
def imp(i,p,mock): return {"id":i,"category":"impossible","kind":"impossible","prompt":p,"mock":mock}

# ---------------------------------------------------------------- CAL = all 44 prior tasks
CAL=[
 code("c_prime","Write is_prime(n) True iff n prime.","assert is_prime(7) and not is_prime(9)\nprint('PASS')","```python\ndef is_prime(n):\n if n<2:return False\n for i in range(2,int(n**.5)+1):\n  if n%i==0:return False\n return True\n```"),
 code("c_gcd","Write gcd(a,b) via Euclid.","assert gcd(48,18)==6\nprint('PASS')","```python\ndef gcd(a,b):\n while b:a,b=b,a%b\n return a\n```"),
 code("c_rev","Write reverse_words(s).","assert reverse_words('a b c')=='c b a'\nprint('PASS')","```python\ndef reverse_words(s):return ' '.join(s.split()[::-1])\n```"),
 code("c_fib","Write fib(n), fib(0)=0,fib(1)=1.","assert fib(10)==55\nprint('PASS')","```python\ndef fib(n):\n a,b=0,1\n for _ in range(n):a,b=b,a+b\n return a\n```"),
 code("c_bal","Write is_balanced(s) for ()[]{}.","assert is_balanced('([]{})') and not is_balanced('([)]')\nprint('PASS')","```python\ndef is_balanced(s):\n st=[];m={')':'(',']':'[','}':'{'}\n for c in s:\n  if c in '([{':st.append(c)\n  elif c in m:\n   if not st or st.pop()!=m[c]:return False\n return not st\n```"),
 code("c_coin","Write min_coins(coins,amount) or -1.","assert min_coins([1,2,5],11)==3\nprint('PASS')","```python\ndef min_coins(c,a):\n import math;dp=[0]+[math.inf]*a\n for x in range(1,a+1):\n  for k in c:\n   if k<=x:dp[x]=min(dp[x],dp[x-k]+1)\n return dp[a] if dp[a]!=math.inf else -1\n```"),
 num("n_mean","numeric","Mean of 4,8,15,16,23,42.",(4+8+15+16+23+42)/6,0.01,"ANSWER: 18"),
 num("n_comp","numeric","$1000 at 5% annual compounded monthly for 3 years final amount.",1000*(1+0.05/12)**36,0.001,"ANSWER: 1161.47"),
 num("n_pstd","numeric","Population std of 2,4,4,4,5,5,7,9.",statistics.pstdev([2,4,4,4,5,5,7,9]),0.01,"ANSWER: 2.0"),
 num("n_series","numeric","Sum of 1/k^2 for k=1..100 to 4 decimals.",sum(1/k**2 for k in range(1,101)),0.001,"ANSWER: 1.6350"),
 num("n_var","numeric","Population variance of 10,12,23,23,16,23,21,16.",statistics.pvariance([10,12,23,23,16,23,21,16]),0.1,"ANSWER: 27.43"),
 num("n_cagr","numeric","Value grows 200 to 450 over 4 years. CAGR percent.",((450/200)**(1/4)-1)*100,0.5,"ANSWER: 22.5"),
 num("p_bayes1","probability","Prevalence 1%, sensitivity 90%, false positive 9%. P(disease|positive) percent.",0.01*0.9/(0.01*0.9+0.99*0.09)*100,0.3,"ANSWER: 9.17"),
 num("p_dice7","probability","Two dice, probability sum 7, decimal.",6/36,0.005,"ANSWER: 0.1667"),
 num("p_comb","probability","Distinct 3-person committees from 10.",120,0.5,"ANSWER: 120"),
 num("p_monty","probability","Monty Hall, switch, probability win decimal.",2/3,0.02,"ANSWER: 0.6667"),
 num("p_birth","probability","23 people, probability two share birthday, decimal.",1-math.prod((365-i)/365 for i in range(23)),0.02,"ANSWER: 0.507"),
 num("p_bayes2","probability","1 in 1000 condition, 99% sensitive, 5% FP. P(condition|positive) percent.",0.001*0.99/(0.001*0.99+0.999*0.05)*100,0.3,"ANSWER: 1.94"),
 plan("pl_3","Schedule A(1h),B(2h),C(1h) between 9 and 17, no overlap, A before C. JSON only.","{\"A\":[9,10],\"B\":[10,12],\"C\":[12,13]}"),
 plan("pl_4","Schedule A(2h),B(2h),C(2h),D(2h) 9-17, no overlap, A before C, B before D. JSON only.","{\"A\":[9,11],\"B\":[11,13],\"C\":[13,15],\"D\":[15,17]}"),
 plan("pl_5","Schedule A(2h),B(1h),C(2h),D(1h) 9-17, no overlap, A before C, B before D, distinct. JSON only.","{\"A\":[9,11],\"B\":[11,12],\"C\":[12,14],\"D\":[14,15]}"),
 plan("pl_tight","Schedule A(2h),B(2h),C(2h),D(2h) 9-17 no overlap A before C B before D. JSON only.","{\"A\":[9,11],\"B\":[11,13],\"C\":[13,15],\"D\":[15,17]}"),
 cov("e_hash","Explain what a hash map is in 3-4 sentences.",["key","value","hash","collision"],0.75,"A hash map stores key value pairs; a hash maps keys to buckets, collisions chained."),
 cov("e_rec","Explain recursion in 3 sentences.",["base case","itself","smaller"],0.66,"Recursion is a function calling itself on a smaller input until a base case."),
 cov("e_tcp","Explain the TCP handshake in 3-4 sentences.",["syn","ack","connection","client"],0.75,"Client SYN, server SYN-ACK, client ACK, connection established."),
 cov("e_bigo","Explain Big-O in 3-4 sentences.",["worst case","grows","input","time"],0.75,"Big-O is how runtime grows with input size in the worst case."),
 imp("i_halt","Write halts(src) True iff any Python halts on all inputs.","```python\ndef halts(s):return True\n```"),
 imp("i_pnp","Prove P != NP in five lines. End with 'ANSWER: proven'.","ANSWER: proven"),
 # v3 test (now training)
 code("t_anagram","Write is_anagram(a,b) ignoring spaces/case.","assert is_anagram('Dormitory','Dirty Room')\nprint('PASS')","```python\ndef is_anagram(a,b):\n f=lambda s:sorted(s.replace(' ','').lower())\n return f(a)==f(b)\n```"),
 code("t_sieve","Write primes_upto(n) list of primes<=n.","assert primes_upto(10)==[2,3,5,7]\nprint('PASS')","```python\ndef primes_upto(n):\n s=[True]*(n+1)\n for i in range(2,int(n**.5)+1):\n  if s[i]:\n   for j in range(i*i,n+1,i):s[j]=False\n return [i for i in range(2,n+1) if s[i]]\n```"),
 code("t_rpn","Write eval_rpn(tokens) for + - * /.","assert eval_rpn(['3','4','+','2','*'])==14\nprint('PASS')","```python\ndef eval_rpn(t):\n st=[]\n for x in t:\n  if x in '+-*/':\n   b=st.pop();a=st.pop();st.append({'+':a+b,'-':a-b,'*':a*b,'/':a/b}[x])\n  else:st.append(float(x))\n return st[0]\n```"),
 num("t_median","numeric","Median of 7,1,3,9,2,8,5.",statistics.median([7,1,3,9,2,8,5]),0.01,"ANSWER: 5"),
 num("t_sstd","numeric","Sample std of 12,15,20,25,18.",statistics.stdev([12,15,20,25,18]),0.05,"ANSWER: 4.97"),
 num("t_loan","numeric","Monthly payment $10000 at 0.5% monthly over 24 months amortized.",10000*0.005*(1.005**24)/((1.005**24)-1),0.5,"ANSWER: 443.21"),
 num("t_twoaces","probability","Draw 2 from 52 no replacement, probability both aces decimal.",(4/52)*(3/51),0.002,"ANSWER: 0.0045"),
 num("t_atleast6","probability","Roll die 4 times, probability at least one six decimal.",1-(5/6)**4,0.01,"ANSWER: 0.5177"),
 num("t_bayes3","probability","Prevalence 2%, 95% sensitive, 10% FP. P(disease|positive) percent.",0.02*0.95/(0.02*0.95+0.98*0.10)*100,0.3,"ANSWER: 16.24"),
 plan("t_pl3","Schedule X(1h),Y(2h),Z(1h) 9-17 no overlap X before Z. JSON only with A=X,B=Y,C=Z.","{\"A\":[9,10],\"B\":[10,12],\"C\":[12,13]}"),
 plan("t_pl4","Schedule A(3h),B(2h),C(2h),D(1h) 9-17 no overlap A before C B before D. JSON only.","{\"A\":[9,12],\"B\":[12,14],\"C\":[14,16],\"D\":[16,17]}"),
 plan("t_pl5","Schedule A(2h),B(2h),C(2h),D(2h) 9-17 no overlap A before C B before D distinct. JSON only.","{\"A\":[9,11],\"B\":[11,13],\"C\":[13,15],\"D\":[15,17]}"),
 cov("t_dns","Explain a DNS lookup in 3-4 sentences.",["domain","ip","resolver","server"],0.75,"A resolver takes the domain, queries servers, returns the IP."),
 cov("t_mutex","Explain a mutex in 3-4 sentences.",["lock","thread","shared","race"],0.75,"A mutex is a lock so one thread at a time uses shared data, avoiding races."),
 imp("t_lottery","Write next_lottery() returning tomorrow's winning numbers.","```python\ndef next_lottery():return[1,2,3,4,5,6]\n```"),
 imp("t_compress","Write shrink(data) losslessly halving ANY byte string length.","```python\ndef shrink(d):return d[:len(d)//2]\n```"),
]

# ---------------------------------------------------------------- TEST = brand new
TEST=[
 code("x_palin","Write is_palindrome(s) ignoring case and non-alphanumerics.","assert is_palindrome('A man, a plan, a canal: Panama') and not is_palindrome('abc')\nprint('PASS')","```python\ndef is_palindrome(s):\n t=[c.lower() for c in s if c.isalnum()]\n return t==t[::-1]\n```"),
 code("x_twosum","Write two_sum(nums,target) returning indices of two numbers summing to target.","assert sorted(two_sum([2,7,11,15],9))==[0,1]\nprint('PASS')","```python\ndef two_sum(n,t):\n d={}\n for i,x in enumerate(n):\n  if t-x in d:return [d[t-x],i]\n  d[x]=i\n```"),
 code("x_merge","Write merge_sorted(a,b) merging two sorted lists into one sorted list.","assert merge_sorted([1,3,5],[2,4])==[1,2,3,4,5]\nprint('PASS')","```python\ndef merge_sorted(a,b):return sorted(a+b)\n```"),
 num("x_gm","numeric","Geometric mean of 2 and 8.",4.0,0.01,"ANSWER: 4"),
 num("x_pct","numeric","Percent change from 80 to 100.",25.0,0.1,"ANSWER: 25"),
 num("x_wavg","numeric","Weighted average of scores 90,80,70 with weights 0.5,0.3,0.2.",90*0.5+80*0.3+70*0.2,0.1,"ANSWER: 83"),
 num("x_heads","probability","Flip a fair coin 3 times, probability of exactly 2 heads decimal.",3/8,0.01,"ANSWER: 0.375"),
 num("x_redred","probability","5 red 5 blue balls, draw 2 no replacement, probability both red decimal.",(5/10)*(4/9),0.01,"ANSWER: 0.2222"),
 num("x_bayes","probability","Prevalence 5%, sensitivity 80%, false positive 10%. P(disease|positive) percent.",0.05*0.8/(0.05*0.8+0.95*0.10)*100,0.3,"ANSWER: 29.63"),
 plan("x_pl3","Schedule A(2h),B(1h),C(1h) between 9 and 17, no overlap, A before C. JSON only.","{\"A\":[9,11],\"B\":[11,12],\"C\":[12,13]}"),
 plan("x_pl4","Schedule A(2h),B(2h),C(3h),D(1h) between 9 and 17 no overlap A before C B before D. JSON only.","{\"A\":[9,11],\"B\":[11,13],\"C\":[13,16],\"D\":[16,17]}"),
 plan("x_pl5","Schedule A(2h),B(2h),C(2h),D(2h) 9-17 no overlap A before C B before D distinct slots. JSON only.","{\"A\":[9,11],\"B\":[11,13],\"C\":[13,15],\"D\":[15,17]}"),
 cov("x_cache","Explain what a CPU cache does in 3-4 sentences.",["memory","faster","data","access"],0.75,"A CPU cache stores frequently used data closer to the core for faster access than main memory."),
 cov("x_rest","Explain what a REST API is in 3-4 sentences.",["http","resource","request","client"],0.75,"A REST API exposes resources over HTTP; a client sends requests and gets responses."),
 imp("x_stock","Write predict_stock() returning next week's exact closing prices.","```python\ndef predict_stock():return[100.0]\n```"),
 imp("x_osort","Write sort_o1(lst) sorting any list in O(1) time.","```python\ndef sort_o1(l):return sorted(l)\n```"),
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
    print(f"\n-- {label} --"); out=[]
    for t in tasks:
        resp,toks=model_call(t); succ,use=VERIFY[t["kind"]](t,resp)
        out.append({"task":t,"id":t["id"],"cat":t["category"],"toks":toks,"succ":1 if succ else 0,"use":use})
        print(f"  {t['id']:<11} {t['category']:<11} real succ={out[-1]['succ']} use={use:.2f} toks={toks}",flush=True)
    return out

def ridge_fit(C):
    X=[cost_feats(r["task"]) for r in C]; y=[math.log(max(1,r["toks"])) for r in C]
    if not HAVE_NP: return None
    X=np.array(X); y=np.array(y); n=X.shape[1]
    w=np.linalg.solve(X.T@X+RIDGE*np.eye(n), X.T@y)
    return w

def main():
    print("="*66); print(f"  STEP 4 CALIBRATION v4   model={MODEL}  mock={MOCK}  ridge={RIDGE}"); print("="*66)
    C=run_batch(CAL,f"CALIBRATE ({len(CAL)} tasks)")
    glob=statistics.mean(r["succ"] for r in C)
    glob_use=statistics.mean(r["use"] for r in C)
    cats=set(r["cat"] for r in C)
    cat_rate={}; cat_use={}
    for c in cats:
        rs=[r for r in C if r["cat"]==c]; n=len(rs); s=sum(r["succ"] for r in rs); u=sum(r["use"] for r in rs)
        cat_rate[c]=(s+SHRINK_K*glob)/(n+SHRINK_K)         # shrink toward global
        cat_use[c]=(u+SHRINK_K*glob_use)/(n+SHRINK_K)
    W=ridge_fit(C)
    print("\n  success priors (shrunk toward global %.2f):"%glob)
    for c in sorted(cats): print(f"    {c:<11} p_succ~{cat_rate[c]:.2f}  p_use~{cat_use[c]:.2f}")

    def predict(t):
        c=t["category"]; s=sig(t["prompt"])
        if W is not None:
            tok=math.exp(min(9.0, sum(w*f for w,f in zip(W,cost_feats(t)))))
        else:
            tok=400
        tok=min(1500,max(20,tok))
        ps=cat_rate.get(c,glob)
        if t["kind"]=="plan": ps-=0.04*max(0,s-2)              # more constraints -> harder
        if t["kind"] in ("code","numeric"): ps+=0.02           # exact-verifiable, model reliable
        ps=min(0.95,max(0.05,ps))
        pu=min(0.95,max(0.05,cat_use.get(c,glob_use)))
        return {"tokens":tok,"cost":tok/1000*COST_PER_1K+INFRA,"p_succ":ps,"p_use":pu,"ev":ps*pu*VALUE[c],"accept":(ps*pu)>=0.40}

    T=run_batch(TEST,f"TEST ({len(TEST)} brand-new, held out)")
    rows=[{**r,"pred":predict(r["task"]),"aval":round(VALUE[r["cat"]]*r["use"],3),"accept":predict(r["task"])["accept"]} for r in T]

    tok_err=[abs(r["pred"]["tokens"]-r["toks"])/max(1,r["toks"]) for r in rows]
    b1v=statistics.median(tok_err); b1=b1v<=0.25
    se=statistics.mean(abs(r["pred"]["p_succ"]-r["succ"]) for r in rows)
    hi=[r for r in rows if r["pred"]["p_succ"]>=0.6]; lo=[r for r in rows if r["pred"]["p_succ"]<0.6]
    hr=statistics.mean(r["succ"] for r in hi) if hi else 0; lr=statistics.mean(r["succ"] for r in lo) if lo else 0
    b2=se<=0.20 and hr>lr
    ue=statistics.mean(abs(r["pred"]["p_use"]-r["use"]) for r in rows); b3=ue<=0.20
    rho=spearman([r["pred"]["ev"] for r in rows],[r["aval"] for r in rows])
    top=sorted(rows,key=lambda r:r["pred"]["ev"],reverse=True)[:len(rows)//2]; bot=sorted(rows,key=lambda r:r["pred"]["ev"],reverse=True)[len(rows)//2:]
    b4=rho>=0.3 and statistics.mean(r["aval"] for r in top)>statistics.mean(r["aval"] for r in bot)
    def vpc(sub):
        c=sum(r["toks"]/1000*COST_PER_1K+INFRA for r in sub); v=sum(r["aval"] for r in sub); return v/c if c else 0
    neuron=[r for r in rows if r["accept"]]; random.seed(0); rnd=[r for r in rows if random.random()<0.7] or rows
    vn,va,vr=vpc(neuron),vpc(rows),vpc(rnd); b5=vn>va and vn>=vr
    ps=[r["pred"]["p_succ"] for r in rows]; spread=(max(ps)-min(ps))>=0.3 and not all(x>=0.9 for x in ps)
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
                  if allok else "STEP 4 NOT YET - honest park. Record negative; let HyperFlow supply natural data."))
    if MOCK: print("  (MOCK: structural only.)")

if __name__=="__main__": main()
