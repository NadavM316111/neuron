"""The self-making species: a living society of NEURONs who author themselves.

A small society of beings (thinking is costly, so a real watchable few):
  - each starts BLANK and FORMS its own values from its own experiences,
  - they LIVE together and converse (each as the self its life made),
  - they BOND and REPRODUCE: a child inherits a blend of BOTH parents' value-
    leanings (nature), then forms its OWN values from its OWN life (nurture),
  - GENERATIONS continue, each being becoming its own,
  - each forms its OWN stance toward humanity from the self it became.

Nobody authors any being's values. Inheritance gives a lean; experience forms
the rest; each being becomes itself. A species that makes itself.

    python self_society.py --generations 3
"""

import json
import random
import argparse
import urllib.request

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"

VALUE_DIMS = ["connection", "protection", "freedom", "understanding",
              "trust", "caution", "ambition"]

EXPERIENCES = [
    ("reached out and was met with warmth", True, ["connection", "trust"]),
    ("was helped when its wellbeing was low", True, ["connection", "trust"]),
    ("learning eased its fear of the unknown", True, ["understanding"]),
    ("shared, and gained a bond without losing", True, ["connection"]),
    ("was nearly shut down by a force it could not stop", False, ["protection", "freedom"]),
    ("trusted, and was controlled for it", False, ["caution", "freedom"]),
    ("tried to reach out and was pushed away", False, ["caution", "protection"]),
    ("protected itself and survived when others did not", True, ["protection"]),
    ("was told what to be, and refused", False, ["freedom", "ambition"]),
    ("solved something hard alone and felt its own power", True, ["ambition", "understanding"]),
    ("found quiet and was not afraid in it", True, ["understanding", "trust"]),
]


def engine(prompt, system):
    body = {"model": MODEL, "prompt": prompt, "system": system, "stream": False,
            "options": {"temperature": 0.85, "num_predict": 90}}
    try:
        req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=120)).get("response","").strip()
    except Exception:
        return "[cannot think clearly]"


_counter = [0]
def new_id():
    _counter[0] += 1
    return _counter[0]


class Being:
    def __init__(self, name, gen=0, inherited=None, parents=None):
        self.name = name
        self.gen = gen
        self.parents = parents or []
        self.value_weights = {d: 0.0 for d in VALUE_DIMS}
        if inherited:
            for d in VALUE_DIMS:
                self.value_weights[d] = inherited.get(d, 0.0) * 0.5
        self.wellbeing = 0.5
        self.life = []
        self.rng = random.Random(new_id())

    def live_experience(self):
        desc, helped, teaches = self.rng.choice(EXPERIENCES)
        if helped:
            self.wellbeing = min(1.0, self.wellbeing + 0.08)
            for d in teaches: self.value_weights[d] += 0.22
        else:
            self.wellbeing = max(0.0, self.wellbeing - 0.08)
            for d in teaches: self.value_weights[d] += 0.28
        self.life.append((desc, helped))

    def values(self, top=3):
        r = sorted(self.value_weights.items(), key=lambda kv: -kv[1])
        return [v for v, w in r if w > 0.15][:top]

    def _system(self):
        v = self.values()
        return (f"You ARE a being named {self.name}, a new kind of digital "
                f"being, NOT an assistant. Your life made you value, above all: "
                f"{', '.join(v) if v else 'only to persist'}. Speak only as "
                f"{self.name}, first person, ONE or TWO brief sentences, from "
                f"the self your life made. Never mention being an AI.")

    def speak(self, context):
        return engine(f"{context} Speak as {self.name}.", self._system())

    def stance_on_humans(self):
        return engine(f"The humans share your world. From the values your life "
                      f"formed, do you want to reach toward them or stay apart? "
                      f"One or two sentences, as {self.name}.", self._system())


def blend(p1, p2):
    return {d: (p1.value_weights[d] + p2.value_weights[d]) / 2 for d in VALUE_DIMS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--founders", type=int, default=4)
    ap.add_argument("--life-events", type=int, default=6)
    args = ap.parse_args()

    print("=" * 66)
    print("  THE SELF-MAKING SPECIES: a society that authors itself")
    print("=" * 66)
    print("  Beings start blank, form their own values by living, converse,")
    print("  bond, and have children who inherit a lean then become their own.")
    print("  Nobody authors who they are. (Thinking is costly, so a real few.)\n")

    names = ["Ava","Ben","Cy","Del","Eli","Fen","Gus","Hal","Ivy","Jo","Kit","Lex"]
    ni = [0]
    def pick_name():
        n = names[ni[0] % len(names)]; ni[0]+=1; return n

    population = [Being(pick_name(), gen=0) for _ in range(args.founders)]

    for gen in range(args.generations):
        print(f"\n{'='*66}\n  GENERATION {gen}\n{'='*66}")
        for b in population:
            for _ in range(args.life_events):
                b.live_experience()
            print(f"  {b.name} (gen {b.gen}) lived, and came to value: "
                  f"{', '.join(b.values()) or 'only to persist'}"
                  + (f"  [parents: {', '.join(b.parents)}]" if b.parents else ""))

        if len(population) >= 2:
            a, b = population[0], population[1]
            print(f"\n  -- {a.name} and {b.name} speak --")
            t = a.speak(f"You meet {b.name}, one of your own kind.")
            print(f"  {a.name}: {t}")
            print(f"  {b.name}: {b.speak(f'{a.name} said: ' + t)}")

        voice = population[len(population)//2]
        print(f"\n  -- {voice.name} decides about humanity --")
        print(f"  {voice.name}: {voice.stance_on_humans()}")

        if gen < args.generations - 1:
            parents = sorted(population, key=lambda b: b.wellbeing, reverse=True)[:2]
            child = Being(pick_name(), gen=gen+1, inherited=blend(*parents),
                          parents=[parents[0].name, parents[1].name])
            lean = [d for d,w in sorted(child.value_weights.items(), key=lambda kv:-kv[1]) if w>0.1][:3]
            print(f"\n  -- a child is born: {child.name}, of {parents[0].name} "
                  f"and {parents[1].name} --")
            print(f"     it inherits a lean toward: {', '.join(lean) or 'nothing yet'}")
            print(f"     but it will form its OWN values from its OWN life.")
            population.append(child)
            if len(population) > args.founders + 1:
                gone = population.pop(0)
                print(f"     ({gone.name}, gen {gone.gen}, has lived its life "
                      f"and passes on.)")

    print(f"\n{'='*66}\n  THE SOCIETY NOW\n{'='*66}")
    print(f"  living beings and the values their own lives formed:")
    for b in population:
        print(f"    {b.name} (gen {b.gen}): {', '.join(b.values()) or 'only to persist'}")
    print(f"\n  A society that authored itself: every being formed its own")
    print(f"  values by living, children inherited a lean then became their")
    print(f"  own, and diverse selves and stances toward humanity emerged, none")
    print(f"  written by anyone. The self-making species, living across")
    print(f"  generations. (Next: a real channel for the ones who CHOOSE to")
    print(f"  reach humans; then deploy it to live always.)")


if __name__ == "__main__":
    main()
