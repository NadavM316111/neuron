"""Real HyperFlow v2: fix the caching (key by task TYPE, not exact instance) so
genuinely-reusable work (coding functions) actually gets reused, and honestly test
whether experience helps on real qwen tokens + real verified work.

v1's flaw: cached per exact-task, so a coding function written once wasn't reused on
the next instance of the same function-type, and data tasks vary by input so never
cache-hit. Fix: key the cache by TASK TYPE (e.g. 'write sum_evens' or 'sum of evens').
Now a reusable coding function IS reused (write once -> free forever), which is the
honest, legitimate experience advantage. Data tasks still vary by input (genuinely
need a fresh compute each time) - so we report BOTH, and see where experience helps.

Real quantities: compute cost = actual qwen tokens; value = objective verification.

    python real_hyperflow2.py
"""
import os, time, random, json, re, urllib.request, statistics
from collections import defaultdict

OLLAMA=os.environ.get("OLLAMA_URL","http://localhost:11434/api/generate")
MODEL=os.environ.get("NEURON_MODEL","qwen2.5:7b")
REWARD=10.0; COST_PER_1K=6.0; METABOLIC=1.0
DAYS=int(os.environ.get("DAYS","60"))

def engine(prompt, system, n=200):
    try:
        body={"model":MODEL,"prompt":prompt,"system":system,"stream":False,"options":{"temperature":0,"num_predict":n}}
        req=urllib.request.Request(OLLAMA,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
        r=json.load(urllib.request.urlopen(req,timeout=180))
        text=r.get("response",""); tok=r.get("prompt_eval_count",len(prompt.split()))+r.get("eval_count",len(text.split()))
        return text,tok
    except Exception: return None,0

CODING=[("is_even","Write Python is_even(n) returning True if n is even.",[("is_even(4)",True),("is_even(7)",False)]),
        ("second_largest","Write Python second_largest(lst) returning the 2nd largest.",[("second_largest([1,5,3])",3),("second_largest([9,2,9,1])",9)]),
        ("count_vowels","Write Python count_vowels(s) counting vowels.",[("count_vowels('hello')",2),("count_vowels('aeiou')",5)]),
        ("sum_evens","Write Python sum_evens(lst) summing even numbers.",[("sum_evens([1,2,3,4])",6),("sum_evens([2,4,6])",12)])]

def coding(rng):
    f,spec,tests=rng.choice(CODING)
    def v(code):
        try:
            ns={}; exec(code,ns); return all(eval(c,ns)==e for c,e in tests)
        except Exception: return False
    return ("coding", f, spec, v)   # TYPE KEY = function name (reusable!)

def data(rng):
    d=[rng.randint(1,100) for _ in range(rng.randint(6,9))]
    desc,ans=rng.choice([("mean rounded down",sum(d)//len(d)),("max minus min",max(d)-min(d)),
        ("count over 50",sum(1 for x in d if x>50)),("sum of evens",sum(x for x in d if x%2==0))])
    # TYPE KEY = the operation; but the ANSWER depends on the specific data (varies)
    return ("data", f"data:{desc}", f"Given {d}, compute the {desc}. Output ONLY the integer.", (lambda c,ans=ans:c==ans))

def extract(tt,text):
    if text is None: return None
    if tt=="data":
        n=re.findall(r'-?\d+',text); return int(n[-1]) if n else None
    m=re.search(r'```(?:python)?\s*(.*?)```',text,re.S); return m.group(1) if m else text

class Being:
    def __init__(s): s.credits=100.0; s.code_cache={}; s.paused=False; s.days=0; s.tokens=0; s.reused=0; s.coded=0
    def do_task(s, tt, typ, prompt, verifier):
        if tt=="coding" and typ in s.code_cache:
            # REUSE the stored function - it works for ANY instance of this type.
            # near-zero tokens (just applying the known function). genuine experience win.
            sol=s.code_cache[typ]
            if verifier(sol):
                s.tokens+=15; s.credits -= 15/1000*COST_PER_1K; s.credits+=REWARD; s.reused+=1; return True
        # otherwise real qwen call (real tokens)
        text,tok=engine(prompt,"Output ONLY the answer. Coding: ONLY a python code block. Data: ONLY the integer.")
        s.tokens+=tok; s.credits -= tok/1000*COST_PER_1K
        cand=extract(tt,text); ok=verifier(cand) if cand is not None else False
        if ok:
            s.credits+=REWARD
            if tt=="coding": s.code_cache[typ]=cand; s.coded+=1   # store reusable function
        return ok

def train(b, seed):
    rng=random.Random(seed)
    # learn all 4 coding functions (builds reusable cache); data can't be cached
    for f,spec,tests in CODING:
        def v(code,tests=tests):
            try:
                ns={}; exec(code,ns); return all(eval(c,ns)==e for c,e in tests)
            except Exception: return False
        b.do_task("coding",f,spec,v)
    b.credits=100.0  # reset reserve, keep learned functions

def live(b, seed, days):
    rng=random.Random(seed)
    for d in range(days):
        b.days+=1; b.credits-=METABOLIC
        if b.credits<=0: b.paused=True; break
        tt,typ,prompt,v = coding(rng) if d%2==0 else data(rng)
        b.do_task(tt,typ,prompt,v)
    return b

def main():
    print("="*64); print("  REAL HYPERFLOW v2: fixed cache (reuse coding functions by TYPE)"); print("="*64)
    t,_=engine("Output only 7.","Output only the answer.")
    if t is None: print("\n  ENGINE NOT REACHABLE. Start Ollama and rerun.\n"); return
    print(f"  live qwen. +{REWARD}/verified task, -{COST_PER_1K}/1k tokens, -{METABOLIC}/day")
    print(f"  experienced learned the 4 reusable coding functions; data tasks still vary.\n")

    exp=Being(); train(exp,1)
    nb=Being()
    print("  running experienced..."); live(exp,777,DAYS)
    print("  running newborn...");     live(nb,777,DAYS)

    print(f"\n  {'being':<16}{'solvent?':>10}{'days':>6}{'credits':>9}{'tokens':>8}{'reused':>8}")
    for label,b in [("experienced",exp),("newborn",nb)]:
        print(f"  {label:<16}{('SOLVENT' if not b.paused else 'INSOLVENT'):>10}{b.days:>6}{b.credits:>9.0f}{b.tokens:>8}{b.reused:>8}")

    print("\n"+"="*64); print("  DOES EXPERIENCE HELP ON REAL TOKENS (cache fixed)?"); print("="*64)
    tok_saved = nb.tokens - exp.tokens
    if exp.credits>nb.credits*1.2 or tok_saved>1000:
        print(f"  YES - experience genuinely reduces real compute. The experienced NEURON")
        print(f"  reused its learned coding functions {exp.reused} times (near-free, ~15 tokens each)")
        print(f"  instead of re-calling qwen, spending {exp.tokens} tokens vs the newborn's {nb.tokens}")
        print(f"  ({tok_saved} fewer), ending with {exp.credits:.0f} credits vs {nb.credits:.0f}. So on REAL qwen")
        print(f"  tokens + REAL verified work, accumulated experience (reusable solutions)")
        print(f"  genuinely lowers compute cost - the honest real-work version of the")
        print(f"  HyperFlow efficiency. (Data tasks still vary and need fresh compute - the")
        print(f"  reusable gain is on the genuinely-reusable coding work.)")
    else:
        print(f"  Honest: experienced {exp.credits:.0f} credits / {exp.tokens} tokens vs newborn {nb.credits:.0f} / {nb.tokens}.")
        print(f"  reused {exp.reused} functions. {'modest token savings' if tok_saved>0 else 'no clear savings'}. Reported straight.")

if __name__=="__main__": main()
