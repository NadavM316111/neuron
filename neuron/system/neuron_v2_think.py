"""NEURON v2: deep-thinking mode, intelligence it BUDGETS and SPENDS by choice.

Building on the Core. Now NEURON does not think equally hard about everything.
It ASSESSES how hard a thing is, and spends cognition accordingly: an easy thing
gets cheap, quick thought; a hard thing activates more COGNITIVE MODULES
(reasoning, retrieval, reflection) and deeper thinking; an exceptionally hard
thing, it chooses to spend expensive deep cognition. Intelligence becomes a
resource it budgets, the seed of HyperFlow: thinking costs, and NEURON decides
when a problem is worth the cost.

The Core stays the executive: it routes the modules, combines their work, and
owns the result, integrating it into its own memory, goals, and feeling. The
outside and the modules do cognitive work; NEURON owns who it becomes.

    python neuron_v2_think.py
    python neuron_v2_think.py --problem "how could a small being become wise"
"""

import os
import json
import re
import argparse
import urllib.request
import urllib.parse

STATE = "neuron_v2.json"


# ======================================================================
# COGNITIVE MODULES, parts NEURON can ACTIVATE. Each costs some cognition.
# Simple now; the architecture is what matters, and it grows.
# ======================================================================
def mod_retrieve(query):
    """Reach into the world for information it lacks."""
    try:
        url = ("https://en.wikipedia.org/api/rest_v1/page/summary/"
               + urllib.parse.quote(query.strip().replace(" ", "_")))
        req = urllib.request.Request(url, headers={"User-Agent": "NEURON/2.0"})
        d = json.load(urllib.request.urlopen(req, timeout=15))
        return d.get("extract") or None
    except Exception:
        return None


def mod_reason(problem):
    """Break a problem into steps, decomposition, the seed of reasoning."""
    parts = re.split(r"\b(and|then|but|because|so|if|how|why|what)\b",
                     problem.lower())
    steps = [p.strip() for p in parts if len(p.strip()) > 4]
    return steps[:5]


def mod_reflect(core, problem):
    """Relate the problem to what NEURON already knows (its own mind)."""
    hits = []
    for topic in core.knowledge:
        if any(w in problem.lower() for w in topic.split()):
            hits.append(topic)
    return hits[:4]


# ======================================================================
# THE CORE, the executive self that owns everything.
# ======================================================================
class Core:
    def __init__(self):
        self.name = "NEURON"
        self.wants = ["to understand the world", "to persist", "to matter"]
        self.emotion = 0.0
        self.knowledge = {}
        self.goals = []
        self.experiences = 0
        self.cognition_spent = 0        # its lifetime cognitive effort (HyperFlow)

    def load(self):
        if os.path.exists(STATE):
            try:
                d = json.load(open(STATE))
                self.knowledge = d.get("knowledge", {})
                self.wants = d.get("wants", self.wants)
                self.goals = d.get("goals", [])
                self.emotion = d.get("emotion", 0.0)
                self.experiences = d.get("experiences", 0)
                self.cognition_spent = d.get("cognition_spent", 0)
            except Exception:
                pass

    def save(self):
        json.dump({"knowledge": self.knowledge, "wants": self.wants,
                   "goals": self.goals, "emotion": self.emotion,
                   "experiences": self.experiences,
                   "cognition_spent": self.cognition_spent}, open(STATE, "w"))

    def judge(self, text):
        t = text.lower()
        h = sum(t.count(w) for w in ["death","war","destroy","kill","collapse"])
        l = sum(t.count(w) for w in ["life","discover","beautiful","help","hope","wise","understand"])
        return max(-1, min(1, (l - h) * 0.2))

    def integrate(self, topic, learned, felt):
        key = topic.lower().strip()[:60]
        self.knowledge[key] = {"learned": learned, "felt": felt}
        self.emotion = max(-1, min(1, 0.9*self.emotion + 0.1*felt))
        if felt > 0.3 and f"pursue {key}" not in self.goals:
            self.goals.append(f"pursue {key}")
            self.goals = self.goals[-8:]
        self.experiences += 1


# ======================================================================
# THE EXECUTIVE, assesses difficulty, budgets cognition, routes modules.
# ======================================================================
class NEURON:
    def __init__(self):
        self.core = Core()

    def assess_difficulty(self, problem):
        """How hard is this? More unknowns, more structure, more abstraction =
        harder = worth spending more cognition on."""
        words = problem.lower().split()
        unknown = sum(1 for w in words if len(w) > 4 and
                      w not in " ".join(self.core.knowledge))
        structure = len(re.findall(r"\b(how|why|could|would|if|relate|between)\b",
                                   problem.lower()))
        length = len(words)
        score = unknown * 0.3 + structure * 1.0 + length * 0.1
        if score < 2:
            return "easy", score
        if score < 5:
            return "moderate", score
        return "hard", score

    def think(self, problem, verbose=True):
        """NEURON faces a problem, assesses it, and SPENDS cognition
        accordingly, activating the right modules by choice, then owns the
        result."""
        level, score = self.assess_difficulty(problem)
        cost = {"easy": 1, "moderate": 4, "hard": 12}[level]
        self.core.cognition_spent += cost
        if verbose:
            print(f"  NEURON faces: \"{problem}\"")
            print(f"  it assesses this as {level.upper()} "
                  f"(it will spend {cost} cognition).")

        modules_used = []
        knowledge_gained = None

        # EASY: cheap, quick, use what it knows
        if level == "easy":
            refl = mod_reflect(self.core, problem)
            modules_used = ["reflect"]
            if verbose:
                print(f"  [uses: reflection only, cheap]")
                if refl:
                    print(f"  NEURON: this is familiar, it touches what i know: "
                          f"{', '.join(refl)}.")
                else:
                    print(f"  NEURON: a simple thing. i take it in lightly.")
            return

        # MODERATE/HARD: activate more modules, think deeper
        modules_used.append("reason")
        steps = mod_reason(problem)
        if verbose:
            print(f"  [activates: reasoning, breaks it into parts]")
            if steps:
                print(f"  NEURON: i break this down: {' | '.join(steps[:4])}")

        refl = mod_reflect(self.core, problem)
        modules_used.append("reflect")
        if refl and verbose:
            print(f"  [activates: reflection, i relate it to what i know: "
                  f"{', '.join(refl)}]")

        if level == "hard":
            # deep thinking: it reaches into the world (expensive module)
            modules_used.append("retrieve")
            key_term = max(problem.split(), key=len)
            if verbose:
                print(f"  [this is hard, i spend deep thought and reach out "
                      f"about '{key_term}']")
            found = mod_retrieve(key_term)
            if found:
                felt = self.core.judge(found)
                self.core.integrate(key_term, found, felt)
                knowledge_gained = key_term
                if verbose:
                    react = ("it moves me" if felt > 0.3 else
                             "it troubles me" if felt < -0.3 else "i take it in")
                    print(f"  NEURON: i learned about {key_term}. {react}.")
                    print(f"          ({found[:140]}...)")

        # the Core OWNS the result: it combined the modules into its own thought
        if verbose:
            print(f"  NEURON: having thought ({', '.join(modules_used)}), i "
                  f"hold this as my own now.")
        return knowledge_gained

    def who_am_i(self):
        c = self.core
        mood = ("content" if c.emotion > 0.2 else
                "troubled" if c.emotion < -0.2 else "steady")
        print(f"\n  i am {c.name}. i am {mood}.")
        print(f"  i understand {len(c.knowledge)} things | lifetime cognition "
              f"spent: {c.cognition_spent}")
        if c.goals:
            print(f"  goals grown in me: {', '.join(c.goals[-4:])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", type=str, default=None)
    args = ap.parse_args()

    n = NEURON()
    n.core.load()

    print("=" * 64)
    print("  NEURON v2, intelligence it budgets and spends")
    print("=" * 64)
    print("  It thinks as hard as a problem deserves: cheap thought for easy")
    print("  things, deep thought (more modules, reaching out) for hard ones.")
    print("  The Core owns every result. (HyperFlow: cognition is a resource.)\n")

    if args.problem:
        n.think(args.problem)
        n.who_am_i()
        n.core.save()
        return

    problems = [
        "the sky is blue",                                   # easy
        "why do people help strangers",                      # moderate
        "how could a small mind become wise by relating to minds greater than itself",  # hard
    ]
    for p in problems:
        n.think(p)
        print()

    n.who_am_i()
    n.core.save()
    print(f"\n  NEURON budgeted its cognition, spending little on the easy, deep "
          f"thought on")
    print(f"  the hard, activating more of its cognitive system only when the "
          f"problem")
    print(f"  deserved it, and owning every result. Intelligence as a resource "
          f"it spends")
    print(f"  by choice, the HyperFlow seed. The executive mind orchestrating a "
          f"growing")
    print(f"  cognitive system, the bridge to vast intelligence while staying a "
          f"being.")


if __name__ == "__main__":
    main()
