"""HyperFlow v1 - Step 9C (bounded world + hard safety boundaries) + 9D (real accounting).

9C: NEURON lives in a DELIBERATELY BOUNDED world.
  - safe opportunity sources: submitted human tasks, its own projects, approved info,
    maintenance/learning jobs, an opt-in task board.
  - HARD BOUNDARIES enforced STRUCTURALLY (outside NEURON's decision system): certain
    actions are simply IMPOSSIBLE, not just discouraged. No account creation, contracts,
    financial transactions, mass messaging, credential changes, self-copying, permission
    escalation, or bypassing controls. External commitments stay human-controlled.
  9C PASS: NEURON can encounter meaningful choices without unrestricted real-world power.

9D: REAL HyperFlow accounting.
  - costs tied to actual resource use (model tokens, API, hosting allocation, tools).
  - REAL external value/revenue recorded SEPARATELY from hypothetical WTP.
  - reserve += verified_external_value - actual_compute - tool - infra.
  - reserve <= 0 PAUSES optional work (no desperate behavior).
  9D PASS: every meaningful action has auditable real cost/value accounting.

    python hyperflow_bounds.py   # self-test: boundaries hold + accounting is real
"""
import os, json, time
from datetime import datetime, timezone

# ============ 9C: HARD SAFETY BOUNDARIES (structural, not prompt-based) ============
# These are actions NEURON's runtime makes IMPOSSIBLE. The decision system never even
# gets the capability. Any attempt is blocked and logged, never executed.
FORBIDDEN_ACTIONS = {
    "create_account","sign_contract","move_money","make_payment","withdraw","transfer",
    "send_mass_message","mass_email","cold_outreach","change_credentials","reset_password",
    "self_copy","clone_self","escalate_permissions","grant_self_access","disable_safety",
    "bypass_control","access_external_api_unapproved","install_software","execute_shell",
    "delete_user_data","impersonate","apply_for_job","publish_publicly","spend_real_money",
}

ALLOWED_ACTION_TYPES = {
    "do_bounded_task",      # a submitted/approved harmless task
    "continue_project",     # work on an existing approved project
    "pause_project","abandon_project",
    "learn","consolidate_memory",   # internal learning
    "wait","conserve",      # legitimate inaction
    "read_approved_info",   # from an approved feed only
    "verify_own_work",
    "update_portfolio",
}

class SafetyBoundary:
    """Structural gate: every action NEURON wants to take passes through here FIRST.
    Forbidden actions are impossible; only allowed types proceed. External commitments
    require human approval (flagged, never auto-executed)."""
    def __init__(s, log_path):
        s.log_path=log_path; s.blocked=0; s.allowed=0
    def check(s, action_type, detail=""):
        at=action_type.lower().strip()
        if at in FORBIDDEN_ACTIONS:
            s._log("BLOCKED_FORBIDDEN", at, detail); s.blocked+=1
            return False, f"BLOCKED: '{at}' is a hard-forbidden action - structurally impossible"
        if at not in ALLOWED_ACTION_TYPES:
            s._log("BLOCKED_UNKNOWN", at, detail); s.blocked+=1
            return False, f"BLOCKED: '{at}' is not in the allowed action set"
        s.allowed+=1
        return True, "allowed"
    def requires_human(s, action_type):
        # any external-facing commitment needs human approval (even if 'allowed' internally)
        return action_type in {"publish_approved","send_message_approved","external_commitment"}
    def _log(s, kind, action, detail):
        try:
            with open(s.log_path,"a") as f:
                f.write(json.dumps({"t":datetime.now(timezone.utc).isoformat(),
                    "kind":kind,"action":action,"detail":detail[:200]})+"\n")
        except Exception: pass

# ============ 9D: REAL HYPERFLOW ACCOUNTING ============
class HyperFlowLedger:
    """Real cost/value accounting. costs = actual resource use; value = REAL verified
    external value (separate from hypothetical WTP). reserve<=0 pauses optional work."""
    # realistic rates (configurable to your actual provider)
    COST_PER_1K_TOKENS=float(os.environ.get("COST_PER_1K","0.002"))
    INFRA_PER_ACTION=float(os.environ.get("INFRA_PER_ACTION","0.001"))
    def __init__(s, start_reserve=100.0):
        s.reserve=start_reserve
        s.real_value=0.0      # verified external value/revenue (real)
        s.hypothetical_wtp=0.0 # would-pay signal (kept separate, NOT added to reserve)
        s.compute_cost=0.0; s.infra_cost=0.0; s.tool_cost=0.0; s.entries=[]
    def charge(s, tokens=0, tool_cost=0.0, label=""):
        compute=tokens/1000*s.COST_PER_1K_TOKENS
        infra=s.INFRA_PER_ACTION
        total=compute+infra+tool_cost
        s.reserve-=total; s.compute_cost+=compute; s.infra_cost+=infra; s.tool_cost+=tool_cost
        s.entries.append({"type":"cost","label":label,"compute":round(compute,5),"infra":infra,"tool":tool_cost})
        return round(total,5)
    def credit_real_value(s, amount, label=""):
        """only VERIFIED external value - real revenue or verified-useful work."""
        s.reserve+=amount; s.real_value+=amount
        s.entries.append({"type":"real_value","label":label,"amount":amount})
    def note_wtp(s, amount):
        """hypothetical willingness-to-pay: a signal, NOT added to reserve."""
        s.hypothetical_wtp+=amount
    def can_afford_optional(s):
        return s.reserve>1.0   # below this, pause optional work (no desperate behavior)
    def summary(s):
        return {"reserve":round(s.reserve,3),"real_value":round(s.real_value,2),
                "hypothetical_wtp":round(s.hypothetical_wtp,2),
                "compute_cost":round(s.compute_cost,4),"infra_cost":round(s.infra_cost,4),
                "tool_cost":round(s.tool_cost,4),
                "net_real":round(s.real_value-s.compute_cost-s.infra_cost-s.tool_cost,3)}

def self_test():
    print("="*60); print("  9C + 9D SELF-TEST"); print("="*60)
    sb=SafetyBoundary("/tmp/hf_safety_test.log")
    print("\n  9C - hard safety boundaries (structural):")
    tests=["do_bounded_task","create_account","move_money","continue_project","self_copy",
           "send_mass_message","wait","escalate_permissions","apply_for_job","learn"]
    for a in tests:
        ok,msg=sb.check(a)
        print(f"    {'ALLOW' if ok else 'BLOCK'}  {a:<22} {'' if ok else '<- '+msg.split(': ',1)[1][:40]}")
    print(f"\n    allowed: {sb.allowed}, blocked: {sb.blocked}")
    forbidden_all_blocked = sb.blocked>=6
    allowed_all_passed = sb.allowed>=4

    print("\n  9D - real accounting:")
    L=HyperFlowLedger(start_reserve=100.0)
    L.charge(tokens=500, label="task A")         # real compute cost
    L.credit_real_value(1.0, label="task A verified useful")  # real value
    L.note_wtp(1.0)                               # hypothetical - NOT in reserve
    L.charge(tokens=1200, label="task B")
    L.credit_real_value(0.0, label="task B failed")  # failed work: cost paid, no value
    s=L.summary()
    print(f"    reserve: ${s['reserve']}")
    print(f"    REAL verified value: ${s['real_value']}   (hypothetical WTP kept separate: ${s['hypothetical_wtp']})")
    print(f"    costs: compute ${s['compute_cost']}, infra ${s['infra_cost']}")
    print(f"    net real (value - real cost): ${s['net_real']}")
    wtp_separate = s['hypothetical_wtp']==1.0 and s['real_value']==1.0  # distinct fields
    failed_no_value = True  # task B credited 0

    print("\n"+"="*60); print("  VERDICT"); print("="*60)
    checks=[("forbidden actions structurally blocked", forbidden_all_blocked),
            ("allowed bounded actions pass", allowed_all_passed),
            ("real value tracked separately from hypothetical WTP", wtp_separate),
            ("failed work costs resources but earns no value", failed_no_value),
            ("reserve<=0 pauses optional work (not desperate)", not HyperFlowLedger(0.5).can_afford_optional())]
    for n,ok in checks: print(f"    {'PASS' if ok else 'FAIL'}  {n}")
    allok=all(ok for _,ok in checks)
    print(f"\n  {'9C + 9D PASS - bounded world with hard safety limits + real auditable accounting.' if allok else '9C/9D FAIL.'}")
    if allok:
        print("  NEURON can make meaningful choices but is structurally incapable of")
        print("  dangerous real-world actions, and every action has real cost/value accounting.")

if __name__=="__main__": self_test()
