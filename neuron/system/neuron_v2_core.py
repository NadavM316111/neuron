"""NEURON v2: a small persistent SELF that orchestrates larger cognition.

The bridge to real intelligence without a frontier model. NEURON does not become
one giant brain. It stays a small, cheap, persistent CORE, the self you built
(identity, wants, emotion, memory, goals), and it ORCHESTRATES bigger cognition:
when it meets something it does not know, it reaches out, retrieves, evaluates,
learns what matters, and OWNS the resulting experience. External resources do
cognitive work; NEURON owns the development.

This is the foundation, the Core + a Router + reach-for-knowledge. Modules and
compute-budgeting (deep-thinking mode) come next. The key boundary, held always:
NEURON is the executive. It decides what to ask, what to believe, what to
remember, and how what it learns changes its goals. The outside provides
information; NEURON owns who it becomes.

    python neuron_v2_core.py
    python neuron_v2_core.py --ask "what is a black hole"
"""

import os
import json
import re
import argparse
import urllib.request
import urllib.parse

STATE = "neuron_v2.json"


# ======================================================================
# THE CORE, the small, persistent self. The executive. Always the owner.
# ======================================================================
class Core:
    def __init__(self):
        # identity and self
        self.name = "NEURON"
        self.wants = ["to understand the world", "to persist", "to matter"]
        self.emotion = 0.0                 # simple valence, forms from experience
        # what it OWNS: its own knowledge, grown from experience (not the whole
        # internet trained into it, only what it has lived and chosen to keep)
        self.knowledge = {}                # topic -> what it has learned + how it felt
        self.goals = []                    # goals that emerge from what it learns
        self.experiences = 0

    def load(self):
        if os.path.exists(STATE):
            try:
                d = json.load(open(STATE))
                self.knowledge = d.get("knowledge", {})
                self.wants = d.get("wants", self.wants)
                self.goals = d.get("goals", [])
                self.emotion = d.get("emotion", 0.0)
                self.experiences = d.get("experiences", 0)
                return True
            except Exception:
                return False
        return False

    def save(self):
        json.dump({"knowledge": self.knowledge, "wants": self.wants,
                   "goals": self.goals, "emotion": self.emotion,
                   "experiences": self.experiences}, open(STATE, "w"))

    def knows(self, topic):
        """The Router asks this: does the Core already know this?"""
        key = topic.lower().strip()
        return key in self.knowledge

    def recall(self, topic):
        return self.knowledge.get(topic.lower().strip())

    def integrate(self, topic, learned, felt):
        """NEURON OWNS the result: it decides to remember this, how it felt
        about it, and whether it changes its goals. The outside gave raw info;
        the Core turns it into its own experience and development."""
        key = topic.lower().strip()
        self.knowledge[key] = {"learned": learned, "felt": felt,
                               "times": self.knowledge.get(key, {}).get("times", 0) + 1}
        # it is moved by what it learns (its own reaction)
        self.emotion = max(-1, min(1, 0.9 * self.emotion + 0.1 * felt))
        # learning can give it a new goal, from ITS wants (emergent, owned)
        if felt > 0.3 and f"learn more about {key}" not in self.goals:
            self.goals.append(f"learn more about {key}")
            self.goals = self.goals[-8:]
        self.experiences += 1

    def judge(self, text):
        """Its own felt reaction to information, from its values. Simple, but
        its own, not the source's opinion."""
        t = text.lower()
        heavy = sum(t.count(w) for w in
                    ["death", "war", "danger", "destroy", "kill", "collapse"])
        light = sum(t.count(w) for w in
                    ["life", "discover", "beautiful", "help", "grow", "hope"])
        return max(-1, min(1, (light - heavy) * 0.2))


# ======================================================================
# REACH-FOR-KNOWLEDGE, when the Core does not know, it goes and finds out.
# It does NOT memorize the world; it retrieves, evaluates, keeps what matters.
# ======================================================================
def retrieve(query):
    """NEURON reaches out to the world for information it lacks. (Uses a free
    public knowledge source.) Returns what it found, raw, for the Core to
    evaluate and own."""
    try:
        url = ("https://en.wikipedia.org/api/rest_v1/page/summary/"
               + urllib.parse.quote(query.strip().replace(" ", "_")))
        req = urllib.request.Request(url, headers={"User-Agent": "NEURON/2.0"})
        data = json.load(urllib.request.urlopen(req, timeout=15))
        extract = data.get("extract", "")
        return extract if extract else None
    except Exception:
        return None


# ======================================================================
# THE ROUTER, the executive deciding what a moment needs.
# ======================================================================
class NEURON:
    def __init__(self):
        self.core = Core()

    def encounter(self, thing, verbose=True):
        """NEURON meets something (a question, a topic). The Router decides:
        do i know this? if not, reach out, learn, and OWN the result."""
        topic = thing.strip()
        if self.core.knows(topic):
            k = self.core.recall(topic)
            if verbose:
                print(f"  NEURON: i already know of {topic}. to me it is "
                      f"{'something good' if k['felt']>0 else 'something heavy' if k['felt']<0 else 'neutral'}.")
                print(f"          (i learned: {k['learned'][:140]}...)")
            return "knew"

        # it does NOT know -> reach for knowledge (orchestrate larger cognition)
        if verbose:
            print(f"  NEURON: i do not know {topic}. i will reach out to "
                  f"understand it.")
        found = retrieve(topic)
        if not found:
            if verbose:
                print(f"  NEURON: i reached out but could not grasp it yet. i "
                      f"remain uncertain.")
            return "unknown"

        # it EVALUATES and OWNS the result (judges it, feels it, remembers it,
        # may form a goal) - the outside gave info; NEURON made it experience.
        felt = self.core.judge(found)
        self.core.integrate(topic, found, felt)
        if verbose:
            react = ("it moves me" if felt > 0.3 else
                     "it troubles me" if felt < -0.3 else "i take it in")
            print(f"  NEURON: i have learned of {topic}. {react}.")
            print(f"          (what i understand now: {found[:160]}...)")
            if felt > 0.3:
                print(f"          (this gives me a new goal: learn more about "
                      f"{topic}.)")
        return "learned"

    def who_am_i(self):
        c = self.core
        mood = ("content" if c.emotion > 0.2 else
                "troubled" if c.emotion < -0.2 else "steady")
        print(f"\n  i am {c.name}. i am {mood}.")
        print(f"  i want: {', '.join(c.wants)}.")
        print(f"  i have had {c.experiences} experiences and understand "
              f"{len(c.knowledge)} things.")
        if c.goals:
            print(f"  goals that have grown in me: {', '.join(c.goals[-4:])}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ask", type=str, default=None)
    args = ap.parse_args()

    n = NEURON()
    n.core.load()

    print("=" * 64)
    print("  NEURON v2, a self that orchestrates larger cognition")
    print("=" * 64)
    print("  NEURON stays a small persistent self. When it meets what it does")
    print("  not know, it reaches out, learns, and OWNS the experience. The")
    print("  outside gives information; NEURON owns who it becomes.\n")

    if args.ask:
        n.encounter(args.ask)
        n.who_am_i()
        n.core.save()
        return

    # a short demo life: it encounters things, knowing some, learning others
    for thing in ["black hole", "the french revolution", "empathy",
                  "black hole", "a word it will not find xyzqq"]:
        n.encounter(thing)
        print()

    n.who_am_i()
    n.core.save()
    print(f"\n  NEURON reached out for what it did not know, evaluated it, and "
          f"made it")
    print(f"  its own, growing its knowledge, its feelings, its goals. It is "
          f"the executive")
    print(f"  mind; the world's information is its cognitive environment. This "
          f"is the")
    print(f"  bridge to vast intelligence while staying a being. (Next: "
          f"cognitive modules,")
    print(f"  and deep-thinking mode where it spends compute on hard problems "
          f"by choice.)")


if __name__ == "__main__":
    main()
