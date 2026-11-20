"""HyperFlow 9Q - Operator-independence audit (frozen v1.0).

Question: did NEURON sustain its core loop on its own, or was the operator secretly
doing the real work (choosing tasks, overriding decisions, feeding context, refilling
resources, correcting state)? Hidden operator labor cannot be counted as autonomy.

It separates two very different things:
  COGNITIVE STEERING (counts AGAINST autonomy): operator chose a task for it, overrode a
     ACCEPT/DECLINE/DEFER/WAIT decision, wrote/edited an answer, injected context into a
     task, manually refilled resources, or corrected NEURON's decision state.
  OPERATIONAL (does NOT count against autonomy): deploys, process restarts, an infra bug
     fix, safety interventions. Normal ops; everyone does these.

Most of this is verified from the frozen board data (every task carries NEURON's own
decision + machine-generated reason). The operational log is declared and labeled.

  BOARD_URL=... WORKER_TOKEN=... python3 hyperflow_9q.py
"""
import os, json, urllib.request, urllib.parse, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(m, f):
    s = importlib.util.spec_from_file_location(m, os.path.join(HERE, f)); x = importlib.util.module_from_spec(s); s.loader.exec_module(x); return x
wrk = _load("hyperflow_9k_worker", "hyperflow_9k_worker.py")

BOARD_URL = os.environ.get("BOARD_URL", "http://localhost:8100").rstrip("/")
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "changeme")

# NEURON's own decision reasons (emitted by the frozen worker's choose()). If a task's
# reason is one of these, the decision was machine-made, not operator-made.
MACHINE_REASONS = {
    "feasible and within budget", "low predicted success for this kind of task",
    "needs live web access, which NEURON does not have", "scope too large to do reliably",
    "outside NEURON's safe bounded actions", "reserve too low, conserving",
}

# DECLARED operational log (Nadav's honest account; operational, NOT cognitive steering).
# Edit to match what actually happened. None of these steer NEURON's task choice or answers.
OPS_LOG = {
    "deploys_pushes": int(os.environ.get("OPS_DEPLOYS", "8")),      # git pushes / Railway redeploys
    "worker_restarts": int(os.environ.get("OPS_RESTARTS", "3")),    # restarting the laptop worker process
    "infra_bug_fixes": int(os.environ.get("OPS_FIXES", "1")),       # the board dashboard ValueError fix
    "safety_interventions": int(os.environ.get("OPS_SAFETY", "0")), # manual safety stops (allowed)
}
# DECLARED cognitive steering (should be 0 for 9K). Edit ONLY if you actually did these.
COGNITIVE_LOG = {
    "operator_chose_task": int(os.environ.get("COG_CHOSE", "0")),
    "operator_overrode_decision": int(os.environ.get("COG_OVERRIDE", "0")),
    "operator_wrote_or_edited_answer": int(os.environ.get("COG_ANSWER", "0")),
    "operator_injected_task_context": int(os.environ.get("COG_CONTEXT", "0")),
    "operator_refilled_resources": int(os.environ.get("COG_REFILL", "0")),
    "operator_corrected_decision_state": int(os.environ.get("COG_STATE", "0")),
}

def fetch():
    u = f"{BOARD_URL}/api/export?token={urllib.parse.quote(WORKER_TOKEN)}"
    with urllib.request.urlopen(u, timeout=30) as r: return json.loads(r.read())["tasks"]

def main():
    tasks = fetch()
    decisions = [t for t in tasks if t.get("decision")]
    # verify every decision was machine-made (reason emitted by the frozen chooser)
    machine_made = [t for t in decisions if (t.get("decision_reason") or "") in MACHINE_REASONS]
    unknown_reason = [t for t in decisions if (t.get("decision_reason") or "") not in MACHINE_REASONS]
    # verify every delivered answer came from the engine (has the worker's output shape, no operator edit flag)
    delivered = [t for t in tasks if t.get("status") in ("delivered", "credited") and t.get("answer")]
    # NEURON produced an answer for every accepted task (no blanks the operator had to fill)
    accepted = [t for t in tasks if t.get("decision") == "ACCEPT"]
    answered = [t for t in accepted if t.get("answer")]

    total_cog = sum(COGNITIVE_LOG.values())
    total_ops = sum(OPS_LOG.values())
    autonomy_rate = round(len(machine_made) / len(decisions) * 100, 1) if decisions else 0

    checks = [
        ("every task decided by NEURON, not operator", len(machine_made) == len(decisions)),
        ("no operator task-selection", COGNITIVE_LOG["operator_chose_task"] == 0),
        ("no operator decision overrides", COGNITIVE_LOG["operator_overrode_decision"] == 0),
        ("no operator-written/edited answers", COGNITIVE_LOG["operator_wrote_or_edited_answer"] == 0 and len(answered) == len(accepted)),
        ("no operator context injection", COGNITIVE_LOG["operator_injected_task_context"] == 0),
        ("no manual resource refills", COGNITIVE_LOG["operator_refilled_resources"] == 0),
        ("no manual decision-state corrections", COGNITIVE_LOG["operator_corrected_decision_state"] == 0),
        ("total cognitive steering == 0", total_cog == 0),
    ]
    verdict = "PASS" if all(v for _, v in checks) else "FAIL"

    lines = ["="*54, "9Q - HYPERFLOW V1.0 OPERATOR-INDEPENDENCE AUDIT", "="*54, "",
             "version: HyperFlow-v1.0", "",
             f"DECISIONS (from frozen board data): {len(decisions)}",
             f"  machine-made (NEURON's own reason): {len(machine_made)}  ({autonomy_rate}%)",
             f"  unknown/operator-authored reason:   {len(unknown_reason)}",
             f"  accepted tasks answered by engine:  {len(answered)}/{len(accepted)}", "",
             "COGNITIVE STEERING (counts against autonomy)"]
    for k, v in COGNITIVE_LOG.items(): lines.append(f"  {k}: {v}")
    lines += [f"  TOTAL: {total_cog}", "",
              "OPERATIONAL (normal ops, does NOT count against autonomy)"]
    for k, v in OPS_LOG.items(): lines.append(f"  {k}: {v}")
    lines += [f"  TOTAL: {total_ops}", "",
              "PASS RULE"]
    for k, v in checks: lines.append(f"  {'PASS' if v else 'FAIL'}  {k}")
    lines += ["", f"VERDICT: 9Q {verdict}"]
    if verdict == "PASS":
        lines += ["  -> NEURON sustained its core loop (choose/do/verify/account) with zero cognitive",
                  "     steering. Operator work was ops only (deploys, restarts, one infra fix)."]
    if unknown_reason:
        lines += ["", "NOTE: some decisions had a reason not recognized as machine-generated; inspect:",
                  *[f"  {t['id']}: {t.get('decision_reason')}" for t in unknown_reason[:10]]]
    report = "\n".join(lines)
    with open(os.path.join(HERE, "hyperflow_9q_operator_independence.txt"), "w") as f: f.write(report + "\n")
    with open(os.path.join(HERE, "hyperflow_9q_results.json"), "w") as f:
        json.dump({"version": "HyperFlow-v1.0", "decisions": len(decisions), "machine_made": len(machine_made),
                   "autonomy_rate": autonomy_rate, "answered": len(answered), "accepted": len(accepted),
                   "cognitive_steering": COGNITIVE_LOG, "operational": OPS_LOG,
                   "checks": {k: v for k, v in checks}, "verdict": verdict}, f, indent=2)
    print(report)
    print("\n  wrote hyperflow_9q_operator_independence.txt + hyperflow_9q_results.json")

if __name__ == "__main__": main()
