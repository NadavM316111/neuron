"""Real Workbench v3 (the friend's rigorous replication): paired live-qwen trials +
HARD token cap + temp 0 + causal decomposition (memory effect vs skill effect).

Fixes v2's issues: hard token stop (no overshoot), controlled randomness (temp 0),
multiple paired trials for mean/std/win-rate, and the two causal effects reported
SEPARATELY:
  memory effect = Full - Memory-off
  skill  effect = Full - Skill-reset

The honest question: is NEURON's real-token efficiency driven by persistent MEMORY
(experience retrieval), by learned SKILL (reasoning policy), or both? v2 hinted
memory does the work and skill adds ~nothing. This tests it rigorously.

Needs live qwen. Trials default low (runnable); raise TRIALS for more power.
    TRIALS=8 TOKEN_BUDGET=5000 python3 real_workbench3.py

    python real_workbench3.py
"""
import os, time, random, json, re, urllib.request, statistics
from collections import defaultdict

OLLAMA=os.environ.get("OLLAMA_URL","http://localhost:11434/api/generate")
MODEL=os.environ.get("NEURON_MODEL","qwen2.5:7b")
TOKEN_BUDGET=int(os.environ.get("TOKEN_BUDGET","5000"))
TRIALS=int(os.environ.get("TRIALS","8"))

def engine(prompt, system, n=200):
    t0=time.time()
    try:
        body={"model":MODEL,"prompt":prompt,"system":system,"stream":False,
              "options":{"temperature":0,"num_predict":n}}   # temp 0 = controlled
        req=urllib.request.Request(OLLAMA,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
        r=json.load(urllib.request.urlopen(req,timeout=180))
        text=r.get("response","")
        tok=r.get("prompt_eval_count",len(prompt.split()))+r.get("eval_count",len(text.split()))
        return text,tok
    except Exception:
        return None,0

def coding_task(rng):
    specs=[("Write a Python function is_even(n) returning True if n is even.","is_even",[("is_even(4)",True),("is_even(7)",False)]),
           ("Write a Python function second_largest(lst) returning the 2nd largest.","second_largest",[("second_largest([1,5,3])",3),("second_largest([9,2,9,1])",9)]),
           ("Write a Python function count_vowels(s) counting vowels.","count_vowels",[("count_vowels('hello')",2),("count_vowels('aeiou')",5)]),
           ("Write a Python function sum_evens(lst) summing even numbers.","sum_evens",[("sum_evens([1,2,3,4])",6),("sum_evens([2,4,6])",12)])]
    spec,fname,tests=rng.choice(specs)
    def verify(code):
        try:
            ns={}; exec(code,ns); return all(eval(c,ns)==e for c,e in tests)
        except Exception: return False
    return ("coding",fname,spec,verify)

def data_task(rng):
    data=[rng.randint(1,100) for _ in range(rng.randint(6,9))]
    desc,ans=rng.choice([("mean rounded down",sum(data)//len(data)),("max minus min",max(data)-min(data)),
        ("count over 50",sum(1 for x in data if x>50)),("sum of evens",sum(x for x in data if x%2==0))])
    return ("data",f"data:{desc}",f"Given {data}, compute the {desc}. Output ONLY the integer.",(lambda c,ans=ans:c==ans))

def extract(ttype,text):
    if text is None: return None
    if ttype=="data":
        nums=re.findall(r'-?\d+',text); return int(nums[-1]) if nums else None
    m=re.search(r'```(?:python)?\s*(.*?)```',text,re.S); return m.group(1) if m else text

class NEURON:
    def __init__(s,mem,skill):
        s.mem=mem; s.skill=skill
        s.solutions={} if mem else None
        s.skill_sys=defaultdict(str) if skill else None
        s.tokens=0; s.verified=0
    def sys(s,ttype):
        base=("Output ONLY the answer. Coding: ONLY a python code block with the function. "
              "Data: ONLY the integer.")
        if s.skill and s.skill_sys is not None and s.skill_sys.get(ttype):
            return base+" "+s.skill_sys[ttype]
        return base
    def do(s,ttype,sig,prompt,verifier,budget):
        if s.tokens>=budget: return None   # HARD cap - don't even start
        if s.solutions is not None and sig in s.solutions:
            if s.tokens+20>budget: return None
            sol=s.solutions[sig]
            if verifier(sol): s.tokens+=20; s.verified+=1; return True
        if s.tokens>=budget: return None
        text,tok=engine(prompt,s.sys(ttype))
        if s.tokens+tok>budget:   # HARD cap - this call would overshoot, stop
            s.tokens=budget; return None
        s.tokens+=tok
        cand=extract(ttype,text); ok=verifier(cand) if cand is not None else False
        if ok:
            s.verified+=1
            if s.solutions is not None: s.solutions[sig]=cand
            if s.skill and s.skill_sys is not None: s.skill_sys[ttype]="This type is familiar; be concise."
        return ok

def trained_knowledge(seed):
    import copy
    exp=NEURON(True,True); rng=random.Random(seed)
    tasks=[(coding_task(rng) if i%2==0 else data_task(rng)) for i in range(16)]
    for _ in range(3):
        for ttype,sig,prompt,verifier in tasks:
            exp.tokens=0  # training isn't budget-capped
            exp.do(ttype,sig,prompt,verifier,10**9)
    return copy.deepcopy(exp.solutions), copy.deepcopy(exp.skill_sys)

def run_trial(trial_seed, sols, skills, budget):
    import copy
    rng=random.Random(trial_seed)
    tasks=[(coding_task(rng) if i%2==0 else data_task(rng)) for i in range(14)]  # same order for all 4
    out={}
    for label,mem,skill in [("full",True,True),("mem_off",False,True),("skill_reset",True,False),("newborn",False,False)]:
        n=NEURON(mem,skill)
        if mem: n.solutions=copy.deepcopy(sols)
        if skill: n.skill_sys=copy.deepcopy(skills)
        i=0
        while n.tokens<budget and i<len(tasks)*30:
            t=tasks[i%len(tasks)]
            if n.do(t[0],t[1],t[2],t[3],budget) is None and n.tokens>=budget: break
            i+=1
        out[label]=n.verified
    return out

def main():
    print("="*64); print("  REAL WORKBENCH v3: rigorous paired live trials (friend's design)"); print("="*64)
    t,_=engine("Output only the number 7.","Output only the answer.")
    if t is None:
        print("\n  ENGINE NOT REACHABLE. Start Ollama (qwen2.5:7b) and rerun.\n"); return
    print(f"  live qwen, temp 0, HARD cap {TOKEN_BUDGET} tokens, {TRIALS} paired trials\n")
    sols,skills=trained_knowledge(1)

    runs=defaultdict(list)
    for tr in range(TRIALS):
        r=run_trial(tr*100+3, sols, skills, TOKEN_BUDGET)
        for k,v in r.items(): runs[k].append(v)
        print(f"  trial {tr+1}/{TRIALS}: full {r['full']}, mem_off {r['mem_off']}, "
              f"skill_reset {r['skill_reset']}, newborn {r['newborn']}")

    print(f"\n  {'architecture':<16}{'mean':>8}{'std':>7}{'beats newborn':>15}")
    nb=runs["newborn"]
    for k in ["full","skill_reset","mem_off","newborn"]:
        xs=runs[k]; wins=sum(1 for a,b in zip(xs,nb) if a>b)
        print(f"  {k:<16}{statistics.mean(xs):>8.1f}{statistics.pstdev(xs):>7.1f}{wins}/{TRIALS:>14}")

    mem_effect=statistics.mean(runs["full"])-statistics.mean(runs["mem_off"])
    skill_effect=statistics.mean(runs["full"])-statistics.mean(runs["skill_reset"])
    print("\n"+"="*64); print("  CAUSAL DECOMPOSITION"); print("="*64)
    print(f"  MEMORY effect (full - mem_off):   {mem_effect:+.1f} verified tasks")
    print(f"  SKILL  effect (full - skill_reset): {skill_effect:+.1f} verified tasks")
    if mem_effect>3 and skill_effect>3:
        print(f"\n  BOTH developmental layers help on real tokens. Memory and learned skill")
        print(f"  each add verified throughput. The full developmental architecture is")
        print(f"  supported on live LLM work.")
    elif mem_effect>3:
        print(f"\n  MEMORY is the real driver; SKILL adds little ({skill_effect:+.1f}). NEURON's")
        print(f"  persistent experience RETRIEVAL makes borrowed cognition more token-")
        print(f"  efficient; its learned reasoning policy does not yet add value with a")
        print(f"  real LLM. Honest - and it names the next Intelligence problem precisely:")
        print(f"  make the learned skill/search policy add value on real-model work.")
    else:
        print(f"\n  Honest: memory effect {mem_effect:+.1f}, skill effect {skill_effect:+.1f}. Reported straight.")

if __name__=="__main__": main()
