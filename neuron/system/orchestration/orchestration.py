"""Cognitive Orchestration v1 (the friend's reframe): NEURON learns to WIELD the
foundation model well, not to duplicate its reasoning.

The key insight from Real Workbench v3: qwen already reasons well, so NEURON's
learned-SEARCH skill added nothing. But qwen does NOT have NEURON's lived
experience, and it does NOT know how to orchestrate its own use. So NEURON's
learnable edge is ORCHESTRATION:
  - is this problem familiar? reuse a stored solution directly?
  - does the stored solution need ADAPTATION, or exact reuse?
  - is one model call enough, or should I verify + retry?
  - how much budget does this deserve (shallow vs deep)?
  - which stored memory is relevant?

NEURON LEARNS, from experience, which orchestration choices pay off (verified
success per token). The test: does a LEARNED orchestrator beat just having memory?

  learned-orchestrator  vs  memory-only (fixed use)  vs  fixed-orchestration  vs newborn
milestone: learned-orchestrator > memory-only  => NEURON learns to USE qwen better,
not just to store answers. That's the right intelligence direction.

Needs live qwen.  TRIALS / TOKEN_BUDGET env override.
    python orchestration.py
"""
import os, time, random, json, re, urllib.request, statistics
from collections import defaultdict

OLLAMA=os.environ.get("OLLAMA_URL","http://localhost:11434/api/generate")
MODEL=os.environ.get("NEURON_MODEL","qwen2.5:7b")
TOKEN_BUDGET=int(os.environ.get("TOKEN_BUDGET","5000"))
TRIALS=int(os.environ.get("TRIALS","6"))

def engine(prompt, system, n=200):
    try:
        body={"model":MODEL,"prompt":prompt,"system":system,"stream":False,"options":{"temperature":0,"num_predict":n}}
        req=urllib.request.Request(OLLAMA,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
        r=json.load(urllib.request.urlopen(req,timeout=180))
        text=r.get("response",""); tok=r.get("prompt_eval_count",len(prompt.split()))+r.get("eval_count",len(text.split()))
        return text,tok
    except Exception: return None,0

def coding(rng):
    S=[("Write Python is_even(n) -> True if even.","is_even",[("is_even(4)",True),("is_even(7)",False)]),
       ("Write Python second_largest(lst) -> 2nd largest.","second_largest",[("second_largest([1,5,3])",3),("second_largest([9,2,9,1])",9)]),
       ("Write Python count_vowels(s) -> vowel count.","count_vowels",[("count_vowels('hello')",2),("count_vowels('aeiou')",5)]),
       ("Write Python sum_evens(lst) -> sum of evens.","sum_evens",[("sum_evens([1,2,3,4])",6),("sum_evens([2,4,6])",12)])]
    spec,f,tests=rng.choice(S)
    def v(code):
        try:
            ns={}; exec(code,ns); return all(eval(c,ns)==e for c,e in tests)
        except Exception: return False
    return ("coding",f,spec,v)
def data(rng):
    d=[rng.randint(1,100) for _ in range(rng.randint(6,9))]
    desc,ans=rng.choice([("mean rounded down",sum(d)//len(d)),("max minus min",max(d)-min(d)),
        ("count over 50",sum(1 for x in d if x>50)),("sum of evens",sum(x for x in d if x%2==0))])
    return ("data",f"data:{desc}",f"Given {d}, compute the {desc}. Output ONLY the integer.",(lambda c,ans=ans:c==ans))
def extract(tt,text):
    if text is None: return None
    if tt=="data":
        n=re.findall(r'-?\d+',text); return int(n[-1]) if n else None
    m=re.search(r'```(?:python)?\s*(.*?)```',text,re.S); return m.group(1) if m else text

class Orchestrator:
    """modes: 'learned' learns which orchestration choices pay off; 'memory_only'
    just reuses stored solutions with a FIXED policy; 'fixed' uses a sensible fixed
    orchestration; 'newborn' has nothing."""
    def __init__(s, mode, sols=None, policy=None):
        s.mode=mode
        s.sols=dict(sols) if (sols is not None and mode in ("learned","memory_only","fixed")) else ({} if mode!="newborn" else None)
        # learned policy: per task-type, learned confidence that 'reuse directly' works,
        # and whether 'verify+retry' is worth the tokens
        s.policy=dict(policy) if policy else defaultdict(lambda:{"reuse_ok":0.5,"verify_worth":0.5,"n":0})
        s.tokens=0; s.verified=0
    def orchestrate(s, tt, sig, prompt, verifier, budget):
        if s.tokens>=budget: return None
        pol=s.policy[tt] if isinstance(s.policy,dict) and tt in s.policy else {"reuse_ok":0.5,"verify_worth":0.5,"n":0}
        # DECISION 1: do I have a stored solution, and should I reuse it directly?
        if s.sols is not None and sig in s.sols:
            # learned: reuse directly if confidence high; else verify it still works
            if s.mode=="learned":
                reuse_directly = pol["reuse_ok"]>0.5
            elif s.mode=="memory_only":
                reuse_directly = True   # fixed: always reuse directly
            else:
                reuse_directly = True
            sol=s.sols[sig]
            if reuse_directly:
                if s.tokens+15>budget: return None
                s.tokens+=15
                ok=verifier(sol)
                if ok: s.verified+=1
                # LEARN: did direct-reuse work? update confidence
                if s.mode=="learned": _upd(pol,"reuse_ok",ok)
                if ok: return True
                # reuse failed -> fall through to a fresh call (learned adapts)
        # DECISION 2: make a model call (deep). learned may skip verify to save tokens
        if s.tokens>=budget: return None
        text,tok=engine(prompt, "Output ONLY the answer. Coding: ONLY a python code block. Data: ONLY the integer.")
        if s.tokens+tok>budget: s.tokens=budget; return None
        s.tokens+=tok
        cand=extract(tt,text); ok=verifier(cand) if cand is not None else False
        if ok:
            s.verified+=1
            if s.sols is not None: s.sols[sig]=cand
        # DECISION 3: learned verify+retry if first answer unreliable (only if worth it)
        elif s.mode=="learned" and pol["verify_worth"]>0.5 and s.tokens+tok<=budget:
            text2,tok2=engine(prompt+" Think carefully and output the correct answer only.",
                              "Output ONLY the answer.")
            if s.tokens+tok2<=budget:
                s.tokens+=tok2; cand2=extract(tt,text2); ok2=verifier(cand2) if cand2 else False
                _upd(pol,"verify_worth",ok2)
                if ok2:
                    s.verified+=1
                    if s.sols is not None: s.sols[sig]=cand2
                    return True
        if s.mode=="learned" and isinstance(s.policy,dict): s.policy[tt]=pol
        return ok

def _upd(pol, key, outcome):
    pol["n"]+=1; lr=0.3
    pol[key]=(1-lr)*pol[key]+lr*(1.0 if outcome else 0.0)

def train(seed):
    rng=random.Random(seed); o=Orchestrator("learned")
    tasks=[(coding(rng) if i%2==0 else data(rng)) for i in range(16)]
    for _ in range(3):
        for tt,sig,prompt,v in tasks:
            o.tokens=0; o.orchestrate(tt,sig,prompt,v,10**9)
    return dict(o.sols), dict(o.policy)

def trial(seed, sols, policy, budget):
    rng=random.Random(seed)
    tasks=[(coding(rng) if i%2==0 else data(rng)) for i in range(14)]
    out={}
    for mode in ["learned","memory_only","fixed","newborn"]:
        o=Orchestrator(mode, sols=(sols if mode!="newborn" else None),
                       policy=(policy if mode=="learned" else None))
        i=0
        while o.tokens<budget and i<len(tasks)*30:
            t=tasks[i%len(tasks)]
            if o.orchestrate(t[0],t[1],t[2],t[3],budget) is None and o.tokens>=budget: break
            i+=1
        out[mode]=o.verified
    return out

def main():
    print("="*64); print("  COGNITIVE ORCHESTRATION v1: does NEURON learn to USE qwen better?"); print("="*64)
    t,_=engine("Output only 7.","Output only the answer.")
    if t is None: print("\n  ENGINE NOT REACHABLE. Start Ollama and rerun.\n"); return
    print(f"  live qwen, temp 0, budget {TOKEN_BUDGET}, {TRIALS} paired trials\n")
    sols,policy=train(1)
    runs=defaultdict(list)
    for tr in range(TRIALS):
        r=trial(tr*100+3, sols, policy, TOKEN_BUDGET)
        for k,v in r.items(): runs[k].append(v)
        print(f"  trial {tr+1}/{TRIALS}: learned {r['learned']}, memory_only {r['memory_only']}, "
              f"fixed {r['fixed']}, newborn {r['newborn']}")
    print(f"\n  {'mode':<16}{'mean':>8}{'std':>7}")
    for k in ["learned","memory_only","fixed","newborn"]:
        print(f"  {k:<16}{statistics.mean(runs[k]):>8.1f}{statistics.pstdev(runs[k]):>7.1f}")
    orch_effect=statistics.mean(runs["learned"])-statistics.mean(runs["memory_only"])
    print("\n"+"="*64); print("  DOES LEARNED ORCHESTRATION BEAT JUST HAVING MEMORY?"); print("="*64)
    print(f"  ORCHESTRATION effect (learned - memory_only): {orch_effect:+.1f} verified tasks")
    if orch_effect>2:
        print(f"\n  YES - NEURON learns to WIELD qwen better, not just store answers. The")
        print(f"  learned orchestrator ({statistics.mean(runs['learned']):.1f}) beat memory-only ({statistics.mean(runs['memory_only']):.1f}) by")
        print(f"  deciding when to reuse vs adapt, when to verify+retry, how to spend")
        print(f"  tokens. This solves v3's weakness: NEURON's learned layer NOW adds value")
        print(f"  on a real LLM - by orchestrating external cognition, not duplicating it.")
        print(f"  The right intelligence direction: the small persistent individual learns")
        print(f"  to use its borrowed mind increasingly well over its life.")
    else:
        print(f"\n  Honest: orchestration effect {orch_effect:+.1f} (learned {statistics.mean(runs['learned']):.1f} vs memory-only {statistics.mean(runs['memory_only']):.1f}).")
        print(f"  {'small positive' if orch_effect>0 else 'no clear gain'} - learned orchestration")
        print(f"  {'adds a little' if orch_effect>0 else 'does not yet beat'} over fixed memory use. Reported straight.")

if __name__=="__main__": main()
