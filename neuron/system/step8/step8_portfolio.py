"""Step 8: Emerging specialties (the friend's design). NEURON's capability portfolio
emerges from its REAL history - the numbers determine its specialties, nobody tells
it "you are a coder."

8A - build the portfolio: per-category performance from the actual ledgers.
8B - preregistered specialty bar (set before seeing results):
     a category is a SPECIALTY only if:
       >= 8 real tasks in that area
       usefulness clearly above NEURON's OWN overall average
       value-per-compute above its general baseline
       enough sample that it isn't 2 lucky tasks
8E - the portfolio becomes an input a chooser can use.
8F - recomputes as new tasks accrue, so specialties can shift with experience.

Reads the real value-pilot ledger (and step7 log) so specialties are grounded in
genuine outcomes. Honest about sample size: small-N categories are "insufficient",
not inflated into specialties.

    python step8_portfolio.py                 (prints the portfolio report)
    python step8_portfolio.py --json          (machine-readable, for the chooser)
"""
import os, json, sys, glob

# locate the ledgers (value pilot + step7). Adjust paths if your dirs differ.
CANDIDATE_LEDGERS=[
    os.path.expanduser("~/neuron/neuron/system/value_pilot/value_state/value_ledger.jsonl"),
    os.path.expanduser("~/neuron/neuron/system/step7/step7_state/step7_log.jsonl"),
    "value_state/value_ledger.jsonl",
    os.environ.get("LEDGER",""),
]

def load_rows():
    rows=[]
    for path in CANDIDATE_LEDGERS:
        if path and os.path.exists(path):
            raw=open(path).read()
            # handle both real-newline and escaped-newline logs
            raw=raw.replace("\\n","\n")
            buf=""
            for line in raw.split("\n"):
                line=line.strip()
                if not line: continue
                buf+=line
                try:
                    r=json.loads(buf); rows.append({**r,"_src":os.path.basename(path)}); buf=""
                except Exception:
                    # maybe multiple objects concatenated; try splitting on }{
                    import re as _re
                    parts=_re.split(r'(?<=\})\s*(?=\{)', buf)
                    if len(parts)>1:
                        for p in parts:
                            try: rows.append({**json.loads(p),"_src":os.path.basename(path)})
                            except Exception: pass
                        buf=""
    return rows

def build_portfolio(rows):
    # only count RATED tasks with a category
    rated=[r for r in rows if r.get("rated") or r.get("useful") is not None]
    cats={}
    for r in rated:
        c=r.get("ttype") or "other"
        if r.get("useful") is None: continue
        d=cats.setdefault(c,{"n":0,"useful":0,"tokens":0,"value":0,"cost":0})
        d["n"]+=1
        d["useful"]+=1 if r.get("useful") else 0
        d["tokens"]+=r.get("tokens", r.get("real_tokens",0))
        d["value"]+=r.get("wtp", r.get("paid",0)) or (1 if r.get("useful") else 0)
        d["cost"]+=r.get("true_cost", r.get("cost", r.get("real_cost",0)))
    # overall baseline
    tot_n=sum(d["n"] for d in cats.values())
    tot_useful=sum(d["useful"] for d in cats.values())
    tot_value=sum(d["value"] for d in cats.values())
    tot_cost=sum(d["cost"] for d in cats.values())
    overall_useful_rate = tot_useful/tot_n if tot_n else 0
    overall_vpc = tot_value/tot_cost if tot_cost else 0
    profile={}
    for c,d in cats.items():
        ur = d["useful"]/d["n"] if d["n"] else 0
        vpc = d["value"]/d["cost"] if d["cost"] else 0
        # preregistered specialty bar (8B)
        enough = d["n"]>=8
        above_avg = ur > overall_useful_rate + 0.05
        good_vpc = vpc >= overall_vpc*0.9
        if not enough:
            status="INSUFFICIENT DATA"; conf="LOW"
        elif above_avg and good_vpc:
            status="EMERGED SPECIALTY"; conf="HIGH" if d["n"]>=12 else "MODERATE"
        elif ur >= overall_useful_rate:
            status="SOLID (not clearly above avg)"; conf="MODERATE"
        else:
            status="NOT A SPECIALTY (below avg)"; conf="N/A"
        profile[c]={"tasks":d["n"],"useful_pct":round(ur*100),"value_per_compute":round(vpc,1),
                    "vs_baseline_vpc":round(vpc/overall_vpc,2) if overall_vpc else None,
                    "status":status,"confidence":conf}
    return profile, {"overall_useful_rate":round(overall_useful_rate*100),
                     "overall_vpc":round(overall_vpc,1),"total_rated_tasks":tot_n}

def main():
    rows=load_rows()
    profile, overall = build_portfolio(rows)
    if "--json" in sys.argv:
        print(json.dumps({"profile":profile,"overall":overall},indent=2)); return
    print("="*60); print("  STEP 8: EMERGING SPECIALTIES (from real history)"); print("="*60)
    print(f"  grounded in {overall['total_rated_tasks']} rated tasks")
    print(f"  NEURON's overall useful rate: {overall['overall_useful_rate']}%  (the baseline a specialty must beat)")
    print(f"  overall value-per-compute: {overall['overall_vpc']}\n")
    if not profile:
        print("  no rated tasks found - run tasks through the pilot first."); return
    # sort: specialties first, then by useful %
    order=sorted(profile.items(), key=lambda kv:(-("SPECIALTY" in kv[1]["status"]), -kv[1]["useful_pct"]))
    for c,p in order:
        print(f"  {c.upper()}")
        print(f"    tasks: {p['tasks']}   useful: {p['useful_pct']}%   value/compute: {p['value_per_compute']}"
              + (f" ({p['vs_baseline_vpc']}x baseline)" if p['vs_baseline_vpc'] else ""))
        print(f"    confidence: {p['confidence']}   >>> {p['status']}\n")
    print("="*60)
    specialties=[c for c,p in profile.items() if "EMERGED" in p["status"]]
    insufficient=[c for c,p in profile.items() if "INSUFFICIENT" in p["status"]]
    if specialties:
        print(f"  EMERGED SPECIALTIES: {', '.join(s.upper() for s in specialties)}")
        print(f"  These emerged from real outcomes - NEURON was never told it was good at them.")
    else:
        print(f"  No clear specialty has emerged YET with enough evidence (need >=8 clean")
        print(f"  tasks/category clearly above the {overall['overall_useful_rate']}% baseline).")
    if insufficient:
        print(f"  NEED MORE DATA: {', '.join(i.upper() for i in insufficient)} (<8 rated tasks)")
    print(f"\n  (8C fresh-task validation + 8D persistent-vs-fresh-within-specialty are the")
    print(f"   validation steps - run more tasks in the leading categories to confirm.)")

if __name__=="__main__": main()
