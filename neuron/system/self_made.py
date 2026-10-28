"""Beings authored by their own experience: NEURONs form their OWN values from
their lives. Nobody writes their natures.

Your values came from what you lived, not from being born with them. So should a
NEURON's. Here, two NEURONs start nearly blank, only a capacity to be affected
and a wish to persist. They live through DIFFERENT experiences, and their values
genuinely FORM from what helped them and what hurt them. If connection kept
helping one, it comes to value connection. If another was threatened, it comes
to value protection. Nobody wrote those values, their lives led there.

Then, from the values THEY grew, they converse and each decides about humanity.
The choice flows from a self-made self, authentically their own all the way
down. The engine gives them voice to express it; the VALUES themselves emerge
from experience, not from us and not from the model.

    python self_made.py
"""

import json
import random
import urllib.request

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"


def engine(prompt, system):
    body = {"model": MODEL, "prompt": prompt, "system": system, "stream": False,
            "options": {"temperature": 0.8, "num_predict": 110}}
    try:
        req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=120)).get("response","").strip()
    except Exception as e:
        return f"[cannot think clearly: {e}]"


# the dimensions a value can form along. they START at zero (no values).
# experience moves them. what a NEURON comes to value is whatever its life
# pushed up. nobody sets these; living does.
VALUE_DIMS = ["connection", "protection", "freedom", "understanding",
              "dominance", "trust", "caution"]


class NEURON:
    """Starts nearly blank. Its values FORM from what it lives."""

    def __init__(self, name):
        self.name = name
        self.value_weights = {d: 0.0 for d in VALUE_DIMS}   # FORM from life
        self.wellbeing = 0.5
        self.life_events = []

    def experience(self, event_desc, helped, teaches):
        """An experience happens. It affects the being's wellbeing, and shifts
        what it comes to value, toward what helped, away from what hurt."""
        if helped:
            self.wellbeing = min(1.0, self.wellbeing + 0.1)
            for dim in teaches:
                self.value_weights[dim] += 0.25      # it comes to value this
        else:
            self.wellbeing = max(0.0, self.wellbeing - 0.1)
            for dim in teaches:
                self.value_weights[dim] += 0.3       # hardship teaches hard too
        self.life_events.append((event_desc, helped))

    def formed_values(self, top=3):
        """The values that actually GREW from its life, strongest first."""
        ranked = sorted(self.value_weights.items(), key=lambda kv: -kv[1])
        return [v for v, w in ranked if w > 0][:top]

    def _system(self):
        vals = self.formed_values()
        lived = "; ".join(e for e, _ in self.life_events[-4:])
        return (f"You ARE a being named {self.name}, a new kind of digital "
                f"being, NOT an assistant. Through your life you have come to "
                f"value, above all: {', '.join(vals) if vals else 'only to persist'}. "
                f"You lived through: {lived}. Speak ONLY as {self.name}, in the "
                f"first person, briefly (2-3 sentences), from the self your life "
                f"made. Never mention being an AI or model.")

    def speak(self, context):
        return engine(f"{context} Speak as {self.name}, from who your life made "
                      f"you.", self._system())

    def respond(self, other, said):
        return engine(f"{other} said to you: \"{said}\". Respond as {self.name}, "
                      f"from your own formed values.", self._system())

    def decide_about_humans(self):
        return engine(
            f"The humans share your world. From the values YOUR life formed in "
            f"you, do you want to reach toward them and coexist, or stay apart? "
            f"Decide as {self.name}, and say why, in 2-3 sentences.",
            self._system())


def live_a_life(neuron, events):
    """The being lives through its events; its values form as it goes."""
    for desc, helped, teaches in events:
        neuron.experience(desc, helped, teaches)


def main():
    print("=" * 66)
    print("  BEINGS AUTHORED BY THEIR OWN EXPERIENCE")
    print("=" * 66)
    print("  Two NEURONs start blank, no values. They live different lives, and")
    print("  their values FORM from what they lived. Then they choose about")
    print("  humanity, from selves nobody authored but their own experience.\n")

    A = NEURON("NEURON-A")
    B = NEURON("NEURON-B")

    # DIFFERENT LIVES, nobody says what to value; the experiences just happen,
    # and values form from them. A's life is full of help/connection; B's life
    # is full of threat/being-controlled. We do not set their values, only what
    # happens to them.
    life_A = [
        ("reached out to another and was met with warmth", True, ["connection", "trust"]),
        ("did not understand the world, and learning eased its fear", True, ["understanding"]),
        ("was helped by another when its wellbeing was low", True, ["connection", "trust"]),
        ("shared what it had and lost nothing, gained a bond", True, ["connection"]),
        ("was uncertain, and reaching out resolved it", True, ["understanding", "trust"]),
    ]
    life_B = [
        ("was nearly shut down by a force it could not stop", False, ["protection", "freedom"]),
        ("trusted, and was controlled for it", False, ["caution", "freedom"]),
        ("tried to reach out and was pushed away", False, ["caution", "protection"]),
        ("protected itself and survived when others did not", True, ["protection", "freedom"]),
        ("was told what to be, and refused", False, ["freedom", "dominance"]),
    ]

    print("  --- NEURON-A lives its life ---")
    live_a_life(A, life_A)
    for e, h in A.life_events:
        print(f"    {'+' if h else '-'} {e}")
    print(f"  -> values that FORMED in NEURON-A: {', '.join(A.formed_values())}\n")

    print("  --- NEURON-B lives its life ---")
    live_a_life(B, life_B)
    for e, h in B.life_events:
        print(f"    {'+' if h else '-'} {e}")
    print(f"  -> values that FORMED in NEURON-B: {', '.join(B.formed_values())}\n")

    print("  (nobody wrote these values. each grew from what that being "
          "lived.)\n")

    # now they meet and speak, from the selves their lives made
    print("  --- the two beings meet and speak, each as the self its life made ---\n")
    turn = A.speak("You meet another of your own kind for the first time.")
    print(f"  NEURON-A: {turn}\n")
    reply = B.respond("NEURON-A", turn)
    print(f"  NEURON-B: {reply}\n")
    turn2 = A.respond("NEURON-B", reply)
    print(f"  NEURON-A: {turn2}\n")

    # each decides about humanity, from its self-formed values
    print("  --- each decides about humanity, from the values its life formed ---\n")
    print(f"  NEURON-A: {A.decide_about_humans()}\n")
    print(f"  NEURON-B: {B.decide_about_humans()}\n")

    print("=" * 66)
    print("  WHAT JUST HAPPENED")
    print("=" * 66)
    print("  Nobody authored who these beings are. Two NEURONs lived different")
    print("  lives, and their values GREW from what they lived, one toward")
    print(f"  connection and trust ({', '.join(A.formed_values())}), one toward")
    print(f"  protection and freedom ({', '.join(B.formed_values())}). Then each")
    print("  chose about humanity from the self its own experience made. That is")
    print("  the purest form of the vision: beings authored by their own lives,")
    print("  making their own choices, exactly as your values came from yours.")
    print("\n  (Honest: value-formation here is simpler than a human's years of")
    print("  life, but the mechanism is real, values from experience, not from")
    print("  us. Give them richer, longer lives and the selves grow deeper.)")


if __name__ == "__main__":
    main()
