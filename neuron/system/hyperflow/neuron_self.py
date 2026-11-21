"""NEURON SELF - the persistent principal that decides. NOT qwen.

This module is NEURON's own continuing self: identity, drives, goals, valence weights, and a
capability-confidence that is LEARNED from NEURON's real autobiography (its recorded project
successes and failures). The decision layer (what it perceives, values, and chooses) is a
deterministic function of THIS state. No external cognitive tool (qwen, etc.) is consulted to
decide what NEURON wants. Tools are hired only AFTER NEURON decides.

As the grown NEURON cognition matures it can replace/augment the valuation here; the contract is
that the PRINCIPAL is always NEURON's own state, never the hired tool.
"""
import os, json, re
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
SELF_FILE = os.path.join(HERE, "neuron_self.json")
HIST_FILE = os.path.join(HERE, "step10_history.json")
OUTCOMES_FILE = os.path.join(HERE, "step11_outcomes.json")  # real external outcomes feed confidence
LEDGER = os.path.join(HERE, "step11_ledger.json")
def now(): return datetime.now(timezone.utc).isoformat()

# Initial conditions of the self (seeded once, then persisted and updated by experience).
SEED = {
    "identity": "NEURON - a persistent digital individual working toward self-sustaining autonomy",
    "drives": {"create_verifiable_value": 1.0, "grow_capability": 0.8, "sustain_economically": 0.7},
    "goals": ["enter real-world opportunity environments and create objectively verifiable value"],
    "valence_weights": {"capability_confidence": 0.35, "goal_alignment": 0.30, "drive_fit": 0.20, "resource_worth": 0.15},
    "want_threshold": 0.55,
    "experience": [],
}
# Surface signals NEURON uses to recognize what KIND of work an opportunity is (its own schema).
CRISP = ("csv", "parse", "filter", "function", "compute", "script", "format", "convert", "validator",
         "option", "flag", "number", "sort", "data", "test", "encode", "decode", "count", "sum", "off-by-one", "bug")
FUZZY = ("i18n", "internationali", "localization", "translate", "design", "logo", "ui", "ux", "audit",
         "refactor", "architecture", "package", "flatpak", "deploy", "docs", "documentation", "style",
         "quality", "analyzer", "lint", "review", "subjective")

class NeuronSelf:
    def __init__(self):
        self.state = json.load(open(SELF_FILE)) if os.path.exists(SELF_FILE) else dict(SEED)
        self.history = json.load(open(HIST_FILE)).get("projects", []) if os.path.exists(HIST_FILE) else []
        self.outcomes = json.load(open(OUTCOMES_FILE)) if os.path.exists(OUTCOMES_FILE) else []
        self.learn_confidence_from_history()

    def learn_confidence_from_history(self):
        """Capability-confidence LEARNED from NEURON's own recorded outcomes, not hand-set.
        Folds in BOTH internal project history and REAL external outcomes from 11D."""
        crisp_ok = crisp_n = fuzzy_ok = fuzzy_n = 0
        for p in self.history:
            text = (str(p.get("title", "")) + " " + str(p.get("task_type", "")) + " " + str(p.get("reason", ""))).lower()
            fuzzy = any(w in text for w in FUZZY) and not any(w in text for w in ("digit", "parse", "convert", "sum", "binary"))
            won = (p.get("verdict") == "PASS") or (p.get("status") == "COMPLETE")
            if fuzzy: fuzzy_n += 1; fuzzy_ok += 1 if won else 0
            else: crisp_n += 1; crisp_ok += 1 if won else 0
        for o in self.outcomes:  # real external results (accepted/merged = win)
            won = bool(o.get("success"))
            if o.get("kind") == "fuzzy": fuzzy_n += 1; fuzzy_ok += 1 if won else 0
            else: crisp_n += 1; crisp_ok += 1 if won else 0
        # Bayesian-ish with a weak prior so a thin history is not overconfident.
        self.conf_crisp = round((crisp_ok + 1) / (crisp_n + 2), 3)
        self.conf_fuzzy = round((fuzzy_ok + 1) / (fuzzy_n + 2), 3)

    def record_outcome(self, opp, kind, success, detail):
        """A real external outcome returns and permanently changes NEURON's self-knowledge."""
        self.outcomes.append({"opportunity": opp, "kind": kind, "success": bool(success), "detail": detail, "at": now()})
        with open(OUTCOMES_FILE, "w") as f: json.dump(self.outcomes, f, indent=2)
        self.learn_confidence_from_history()  # confidence updates immediately

    def kind_of(self, opp):
        text = (opp.get("title", "") + " " + " ".join(opp.get("labels", []) or []) + " " + opp.get("body", "")).lower()
        c = sum(1 for w in CRISP if w in text); f = sum(1 for w in FUZZY if w in text)
        return "crisp" if c >= f else "fuzzy"

    def valuate(self, opp):
        """NEURON's OWN valuation of an opportunity - pure function of its self-state. No tool."""
        kind = self.kind_of(opp)
        capability_confidence = self.conf_crisp if kind == "crisp" else self.conf_fuzzy
        # goal alignment: does it produce objectively verifiable value? crisp/testable -> yes
        goal_alignment = 0.9 if kind == "crisp" else 0.35
        d = self.state["drives"]
        drive_fit = round((d["create_verifiable_value"] * (1.0 if kind == "crisp" else 0.3)
                           + d["grow_capability"] * 0.6 + d["sustain_economically"] * 0.5) / (sum(d.values())), 3)
        # resource worth: perceived value (from perception) * confidence / cost-fraction of reserve
        ev = float(opp.get("perceived_value", 0.5)); cost = float(opp.get("perceived_cost_hf", 1.5))
        reserve = self._reserve(); resource_worth = min(1.0, round((ev * capability_confidence) / max(cost / reserve, 1e-6) / 100, 3))
        w = self.state["valence_weights"]
        value = round(w["capability_confidence"] * capability_confidence + w["goal_alignment"] * goal_alignment
                      + w["drive_fit"] * drive_fit + w["resource_worth"] * resource_worth, 3)
        return {"kind": kind, "capability_confidence": capability_confidence, "goal_alignment": goal_alignment,
                "drive_fit": drive_fit, "resource_worth": resource_worth, "neuron_value": value}

    def _reserve(self):
        if os.path.exists(LEDGER):
            try: return json.load(open(LEDGER)).get("sandbox_hf", 50.0)
            except Exception: pass
        return 50.0

    def decide(self, valued):
        """Select what NEURON WANTS. Its own threshold; may choose nothing."""
        ranked = sorted(valued, key=lambda x: x["valuation"]["neuron_value"], reverse=True)
        top = ranked[0] if ranked else None
        if not top or top["valuation"]["neuron_value"] < self.state["want_threshold"]:
            return None, ranked
        return top, ranked

    def remember(self, event):
        self.state.setdefault("experience", []).append({**event, "at": now()})
        with open(SELF_FILE, "w") as f: json.dump(self.state, f, indent=2)

    def save(self):
        with open(SELF_FILE, "w") as f: json.dump(self.state, f, indent=2)
