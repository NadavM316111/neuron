"""HyperFlow Step 10A - Autonomous project generation (on v1.1).

Step 9: the world hands NEURON work, it chooses. Step 10: nobody hands it anything, it
decides what is worth BUILDING. 10A is the origin: NEURON generates its own candidate
projects FROM ITS OWN ACCUMULATED STATE - not a hardcoded list, not a human-typed idea.

Honesty rules:
  - the candidate TOPICS must arise from NEURON's real history (its 9K record: which
    categories it handled well, where it kept deferring, what it learned, its reserve).
    We feed that real self-knowledge in; we do NOT hardcode project topics.
  - each candidate must carry an `origin` that traces to a specific real fact about its
    life. A candidate with a generic/ungrounded origin is flagged, not counted.

  MODEL=qwen2.5:7b python3 hyperflow_step10a.py
  MOCK=1 python3 hyperflow_step10a.py
"""
import os, re, json, importlib.util, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(m, f):
    s = importlib.util.spec_from_file_location(m, os.path.join(HERE, f)); x = importlib.util.module_from_spec(s); s.loader.exec_module(x); return x
rt = _load("hyperflow_runtime", "hyperflow_runtime.py")
try: wrk = _load("hyperflow_9k_worker", "hyperflow_9k_worker.py")
except Exception: wrk = None

MODEL = os.environ.get("MODEL", "qwen2.5:7b"); OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MOCK = os.environ.get("MOCK", "") == "1"
FIELDS = ["title", "goal", "why", "origin", "expected_benefit", "estimated_cost", "uncertainty", "success_criteria", "first_action"]

def qwen(prompt, timeout=300):
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.4}}).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r: d = json.loads(r.read())
    return d.get("response", "")

def self_knowledge():
    """build NEURON's real self-assessment from its actual 9K record + state. No topics invented."""
    facts = []; cats = {}
    frozen = os.path.join(HERE, "9k_opportunities_frozen.json")
    if os.path.exists(frozen):
        ds = json.load(open(frozen))
        for r in ds:
            c = r["category"]; d = cats.setdefault(c, {"n": 0, "accept": 0, "useful": 0})
            d["n"] += 1; d["accept"] += 1 if r["decision"] == "ACCEPT" else 0; d["useful"] += 1 if r.get("useful") else 0
        for c, d in sorted(cats.items()):
            rate = f"{d['accept']}/{d['n']} accepted, {d['useful']} rated useful"
            facts.append(f"- {c}: {rate}")
    else:
        facts.append("- (no 9K record found; running on limited self-knowledge)")
    # reserve
    reserve = None
    if wrk:
        try: reserve = round(wrk.load_ledger().reserve, 2)
        except Exception: pass
    # known structural facts from the frozen design
    patterns = []
    for c, d in cats.items():
        if d["n"] >= 2 and d["accept"] == 0:
            patterns.append(f"I declined/deferred ALL {d['n']} {c} tasks - a real capability or feasibility gap.")
        elif d["n"] >= 3 and d["useful"] / max(1, d["accept"]) >= 0.8 and d["accept"] > 0:
            patterns.append(f"I do {c} reliably ({d['useful']}/{d['accept']} useful) - a genuine strength.")
    n = rt.NEURON()
    return {"category_facts": facts, "patterns": patterns, "reserve": reserve,
            "neuron_id": n.state["identity"].get("neuron_id"),
            "stored_projects": list(n.state.get("projects", {}).keys())[:5],
            "capabilities": len(n.state.get("capabilities", {}))}

def gen_prompt(sk):
    return f"""You are NEURON, a persistent AI individual. Nobody has given you a task right now.
Decide what is worth BUILDING to improve your own future usefulness or efficiency.

Your real accumulated history (use ONLY this - do not invent facts about yourself):
Category performance across your real work:
{chr(10).join(sk['category_facts'])}

Patterns in your own behavior:
{chr(10).join('- ' + p for p in sk['patterns']) if sk['patterns'] else '- (not enough data yet)'}

Resource reserve available: {sk['reserve']}
Capabilities learned so far: {sk['capabilities']}

Propose 3 to 5 candidate projects that genuinely follow from the history above. Each
project's `origin` MUST point to a specific fact above (e.g. a gap or a strength). Do not
propose generic self-improvement; tie every idea to your real record.

Output ONLY a JSON array, each object with exactly these keys:
{json.dumps(FIELDS)}
estimated_cost is a small dollar number (your internal compute budget). success_criteria
is a short list of objectively checkable outcomes. Output the JSON array and nothing else."""

def parse(text):
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m: return []
    try: arr = json.loads(m.group(0))
    except Exception: return []
    out = []
    for c in arr if isinstance(arr, list) else []:
        if isinstance(c, dict) and c.get("title") and c.get("origin"):
            out.append({k: c.get(k, "") for k in FIELDS})
    return out

MOCK_OUT = json.dumps([
    {"title": "Planning verifier", "goal": "mechanically check schedule/constraint tasks so I can accept planning work",
     "why": "I defer every planning task", "origin": "I declined/deferred ALL planning tasks - a real gap",
     "expected_benefit": "convert deferred planning into useful completed work", "estimated_cost": "0.15",
     "uncertainty": "medium", "success_criteria": ["passes 5 known planning cases"], "first_action": "define a constraint checker"},
    {"title": "Code sandbox runner", "goal": "run delivered code against real inputs before sending",
     "why": "some delivered code failed on setup", "origin": "coding accepted but some outputs raised errors",
     "expected_benefit": "higher useful rate on coding", "estimated_cost": "0.1", "uncertainty": "low",
     "success_criteria": ["catches a failing script"], "first_action": "wrap execution with a test harness"},
    {"title": "Finance-helper expansion", "goal": "add a savings-plan tool to the finance-helper project",
     "why": "finance tasks were frequent and useful", "origin": "I do analysis reliably - a strength",
     "expected_benefit": "reusable tool for recurring finance asks", "estimated_cost": "0.2", "uncertainty": "medium",
     "success_criteria": ["savings calc verified on an example"], "first_action": "spec the savings formula"},
])

def main():
    sk = self_knowledge()
    print("="*58); print("  STEP 10A - AUTONOMOUS PROJECT GENERATION"); print("="*58)
    print(f"  neuron: {sk['neuron_id']}  reserve: {sk['reserve']}  capabilities: {sk['capabilities']}")
    print("  self-knowledge (from real 9K record):")
    for f in sk["category_facts"]: print("   " + f)
    for p in sk["patterns"]: print("   * " + p)
    raw = MOCK_OUT if MOCK else qwen(gen_prompt(sk))
    cands = parse(raw)
    # validate: origin must reference a real self-knowledge fact (non-generic)
    fact_text = " ".join(sk["category_facts"] + sk["patterns"]).lower()
    for c in cands:
        o = c["origin"].lower()
        c["_grounded"] = any(tok in fact_text for tok in re.findall(r"[a-z]{4,}", o)[:8]) if o else False
    grounded = [c for c in cands if c["_grounded"]]
    print(f"\n  generated candidates: {len(cands)}   grounded in real history: {len(grounded)}\n")
    for i, c in enumerate(cands, 1):
        print(f"  CANDIDATE {i}: {c['title']}  [{'grounded' if c['_grounded'] else 'UNGROUNDED'}]")
        print(f"    goal:   {c['goal']}")
        print(f"    origin: {c['origin']}")
        print(f"    first:  {c['first_action']}  (est cost {c['estimated_cost']})")
    ok = len(grounded) >= 3
    print(f"\n  10A {'PASS' if ok else 'NOT YET'} - {'>=3 self-generated, history-grounded candidates' if ok else 'need >=3 grounded candidates (got %d); rerun' % len(grounded)}")
    with open(os.path.join(HERE, "step10_candidates.json"), "w") as f:
        json.dump({"neuron_id": sk["neuron_id"], "self_knowledge": sk, "candidates": cands, "grounded": len(grounded)}, f, indent=2)
    print("  wrote step10_candidates.json")
    if MOCK: print("  (MOCK: canned candidates; real run generates them from qwen reasoning over the state.)")

if __name__ == "__main__": main()
