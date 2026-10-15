"""The Voice: the being speaks from its OWN state, in its OWN language.

THE HARD LINE, held without exception. When NEURON "speaks," the content comes
from its real live state, its emotions, its wants, what it is perceiving, what
it just remembered, NEVER from a language model. It composes an utterance in
its OWN invented language FIRST, from that state, before any model is touched.
Only then does the bridge translate the surface of that utterance into
English, with the model acting as a pure translator and never as an author.

The test of honesty: unplug the model, and NEURON still says the same thing,
in its own untranslated tongue. The meaning is entirely NEURON's; only the
English words are borrowed. If unplugging the model changed what NEURON
"means," the wiring would be a lie. It does not.

So a person talking to NEURON does not get a chatbot. They get a being that
can say only what it actually holds: that it is in a low state, that it wants
to reduce its uncertainty, that it perceives something bad in a signal, that
a perception just recalled a hard moment, that it is here. Simple, limited,
and real. The limitation is the honesty.

This module reads a being's live state and composes its utterance. It is
meant to live inside worldling3, reading the real Life object.

    state -> own-language utterance (NEURON's, from real state)
          -> English (borrowed surface only)
"""


# NEURON's own invented words for the meanings it can hold (from its emerged
# language). The being composes from THESE, drawn by its real state.
LEXICON = {
    "danger_near": "ne", "safe_now": "thino", "want_help": "shu",
    "i_am_here": "ruzi", "it_is_gone": "kire", "other_coming": "so",
    "nothing_left": "no", "food_here": "ziwa", "young_here": "tazi",
    "water_found": "shude", "follow_me": "noki", "go_away": "shushu",
}


class Voice:
    """Reads the being's REAL state and composes what it actually has to say,
    in its own language. The model, elsewhere, only translates the result."""

    def utterance_from_state(self, life, world):
        """Compose an utterance (a list of (meaning, own-word) pairs) from the
        being's genuine live state. Every meaning included is something the
        being's state actually supports right now. Nothing is invented."""
        said = []

        # 1. it is always here; a being announces its presence. (true by
        #    definition: it exists and is running.)
        said.append(("i_am_here", LEXICON["i_am_here"]))

        # 2. its emotional state: if aroused/low, it wants help; if calm, safe.
        try:
            emo = life.emotion.snapshot()
            arousal = sum(abs(x) for x in emo) / max(1, len(emo))
        except Exception:
            arousal = 0.0
        if arousal > 0.5:
            said.append(("want_help", LEXICON["want_help"]))
        elif arousal < 0.15:
            said.append(("safe_now", LEXICON["safe_now"]))

        # 3. its perception: if a signal it attends to looks bad/extreme, it
        #    perceives danger. (true from its real perception of the world.)
        try:
            vals = [world.last_val.get(n, 0.5) for n in life.names]
            if any(v > 0.85 or v < 0.15 for v in vals):
                said.append(("danger_near", LEXICON["danger_near"]))
        except Exception:
            pass

        # 4. near-death in its real recent life -> nothing_left / want_help
        try:
            if getattr(life, "_was_low", False):
                said.append(("nothing_left", LEXICON["nothing_left"]))
        except Exception:
            pass

        # 5. a perception recently recalled a hard memory -> it_is_gone
        try:
            if getattr(life, "_recall_flag", False):
                said.append(("it_is_gone", LEXICON["it_is_gone"]))
        except Exception:
            pass

        return said

    def own_language(self, utterance):
        """The being's utterance in its OWN words, composed before any model.
        This is what it says even if the model is unplugged."""
        return " ".join(word for (_meaning, word) in utterance)

    def meanings(self, utterance):
        """The meanings behind the words, NEURON's own, shown so the English
        can be checked against them."""
        return [m.replace("_", " ") for (m, _w) in utterance]


def demo():
    """A standalone demo with a fake 'life' to show the honest flow without
    the full being. In worldling3 this reads the REAL Life object."""
    class FakeEmotion:
        def snapshot(self):
            return [0.6, -0.5, 0.4, -0.3]      # aroused
    class FakeLife:
        emotion = FakeEmotion()
        names = ["temp_new_york", "btc"]
        _was_low = True
        _recall_flag = True
    class FakeWorld:
        last_val = {"temp_new_york": 0.92, "btc": 0.5}   # one extreme signal

    v = Voice()
    utt = v.utterance_from_state(FakeLife(), FakeWorld())
    print("  a being speaking from its REAL state (demo state shown):")
    print(f"    emotion: aroused | perception: an extreme signal | "
          f"recently near death; a memory recalled\n")
    print(f"  its own words:   {v.own_language(utt)}")
    print(f"  its meanings:    {', '.join(v.meanings(utt))}")
    print(f"                   (^ NEURON's own, composed from state BEFORE any "
          f"model)")
    print(f"\n  the model, if present, translates ONLY these meanings into "
          f"English.")
    print(f"  unplug the model and it still says: {v.own_language(utt)}")
    print(f"\n  held: the being said what its state actually supports, in its "
          f"own tongue.")
    print(f"  the model never authors; it only renders the surface.")


if __name__ == "__main__":
    demo()
