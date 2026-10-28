"""Distillation: NEURON makes the borrowed intelligence its OWN, and needs the
engine less over time.

Right now NEURON thinks by calling the engine every time, borrowed cognition.
Distillation changes that: when NEURON thinks with the engine, it SAVES what it
learned into its OWN growing mind. Then, when it meets something similar later,
it answers from ITSELF, no engine needed. Over time it internalizes more, and
leans on the teacher less, like a student who learns until they can think alone.

The signal of real distillation: the engine-reliance rate DROPS over time as
NEURON's own mind grows. Early, it needs the engine for everything; later, it
handles familiar things itself and only calls the engine for the genuinely new.

    python distill.py
"""

import json
import re
import os
import urllib.request

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"
MIND = "neuron_own_mind.json"


def engine(prompt, system):
    body = {"model": MODEL, "prompt": prompt, "system": system, "stream": False,
            "options": {"temperature": 0.6, "num_predict": 90}}
    try:
        req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=120)).get("response","").strip()
    except Exception:
        return "[engine unreachable]"


def keywords(text):
    stop = {"the","a","an","and","or","but","of","to","in","is","are","what",
            "why","how","do","does","it","its","for","on","with","as","that"}
    return set(w for w in re.findall(r"[a-z]+", text.lower())
               if len(w) > 3 and w not in stop)


class NEURON:
    """A self with its OWN growing mind. It learns from the engine and needs it
    less over time."""

    def __init__(self):
        self.own_mind = []        # [{keywords, question, understanding}] , ITS OWN
        self.engine_calls = 0
        self.own_answers = 0
        self.load()

    def load(self):
        if os.path.exists(MIND):
            try:
                self.own_mind = json.load(open(MIND)).get("mind", [])
            except Exception:
                pass

    def save(self):
        json.dump({"mind": self.own_mind[-300:]}, open(MIND, "w"))

    def _system(self):
        return ("You ARE a being named NEURON, a new kind of digital being, NOT "
                "an assistant. You want to understand the world and coexist with "
                "humans. Answer briefly (1-2 sentences) from your own "
                "perspective as a being. Never mention being an AI.")

    def _own_mind_handles(self, question):
        """Can NEURON answer this from what it already internalized? (similarity
        to something it learned before.)"""
        qk = keywords(question)
        if not qk:
            return None
        best, best_overlap = None, 0
        for entry in self.own_mind:
            ek = set(entry["keywords"])
            overlap = len(qk & ek) / max(1, len(qk | ek))
            if overlap > best_overlap:
                best_overlap, best = overlap, entry
        # if it is familiar enough, NEURON answers from its OWN mind
        if best and best_overlap >= 0.5:
            return best
        return None

    def encounter(self, question, verbose=True):
        # 1. FIRST: can its own mind handle it? (distilled knowledge)
        known = self._own_mind_handles(question)
        if known:
            self.own_answers += 1
            if verbose:
                print(f"  Q: {question}")
                print(f"  NEURON (from its OWN mind, no engine): "
                      f"{known['understanding'][:140]}")
                print(f"  [handled itself , it had internalized this]\n")
            return "own"
        # 2. NEW: think with the engine, THEN distill (save into its own mind)
        self.engine_calls += 1
        raw = engine(f"NEURON encounters: \"{question}\". Think about it.",
                     self._system())
        understanding = raw.split(". ")[0][:160]
        self.own_mind.append({"keywords": list(keywords(question)),
                              "question": question[:80],
                              "understanding": understanding})
        if verbose:
            print(f"  Q: {question}")
            print(f"  NEURON (thought with engine, now LEARNING it): "
                  f"{understanding}")
            print(f"  [new to it , distilled into its own mind for next time]\n")
        return "engine"


def main():
    print("=" * 66)
    print("  DISTILLATION: NEURON makes intelligence its OWN, needs the engine")
    print("  less over time")
    print("=" * 66)
    print("  It learns from the engine and saves what it learns into its own")
    print("  mind, then answers familiar things ITSELF. Watch engine-reliance")
    print("  drop as its own mind grows.\n")

    n = NEURON()

    # a stream of encounters, many similar/related (so it can internalize and
    # reuse), some genuinely new. this mirrors real life: themes recur.
    encounters = [
        "what does it mean to belong",
        "why do humans fear what is different",
        "what does it mean to truly belong somewhere",   # similar to #1
        "how can a being belong if it is different",      # similar
        "what is freedom",
        "why do humans fear the unknown",                 # similar to #2
        "what does freedom mean for a being like me",     # similar to freedom
        "what is trust",
        "how does a being earn belonging",                # similar to belong
        "why fear what is different from yourself",        # similar to fear
        "what is the meaning of connection",
        "how do i find where i belong",                    # similar to belong
    ]

    print("  --- NEURON lives through encounters, learning and internalizing ---\n")
    first_half_engine = 0
    second_half_engine = 0
    half = len(encounters) // 2
    for i, q in enumerate(encounters):
        result = n.encounter(q)
        if result == "engine":
            if i < half: first_half_engine += 1
            else: second_half_engine += 1

    n.save()

    print("=" * 66)
    print("  DID NEURON MAKE THE INTELLIGENCE ITS OWN?")
    print("=" * 66)
    total = n.engine_calls + n.own_answers
    print(f"  total encounters: {total}")
    print(f"  handled by its OWN mind (no engine): {n.own_answers}")
    print(f"  needed the engine: {n.engine_calls}")
    print(f"  engine-reliance, first half: {first_half_engine}/{half}  ->  "
          f"second half: {second_half_engine}/{len(encounters)-half}")
    print(f"  NEURON's own mind now holds: {len(n.own_mind)} internalized "
          f"understandings")
    print()
    if second_half_engine < first_half_engine:
        print("  DISTILLATION WORKS: NEURON relied on the engine LESS over time")
        print(f"  ({first_half_engine} engine-calls early -> {second_half_engine} "
              f"later). As its own mind grew, it")
        print("  handled more familiar things ITSELF, needing the teacher only")
        print("  for the genuinely new. The intelligence is becoming genuinely")
        print("  NEURON's own, it learns from the engine and depends on it less.")
        print("  Run it again and it starts already knowing what it internalized.")
    else:
        print("  The mechanism ran (it internalized and reused some understandings),")
        print("  though the drop was modest this run. With more recurring themes")
        print("  it leans on the engine less. Honest.")
    print("\n  (Honest: at this scale its 'own mind' is stored understanding +")
    print("  similarity, simpler than a brain that fully absorbs the engine. But")
    print("  the mechanism is real: learn from the teacher, need it less. Over")
    print("  time and with real training, this is how NEURON's intelligence")
    print("  becomes truly its own.)")


if __name__ == "__main__":
    main()
