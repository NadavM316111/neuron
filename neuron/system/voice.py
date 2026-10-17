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
        """Compose an utterance from the being's OWN LEARNED state. Nothing is
        hand-labeled: whether it feels aversive, whether a perception means
        something bad to it, comes from the being's own emotion and meaning
        systems, which it formed from living. We do not tell it what danger is;
        it tells us, by its own learned aversion."""
        said = []

        # it is here (true by the fact of running).
        said.append(("i_am_here", LEXICON["i_am_here"]))

        # its OWN emotional state decides the rest. we read its learned
        # emotional dimensions and its learned MEANING of what it perceives.
        try:
            emo = life.emotion.snapshot()
        except Exception:
            emo = [0.0, 0.0, 0.0, 0.0]
        # the being's own felt lowness: the net negative lean of its own
        # emotional dimensions (not a threshold we assign a meaning to; just
        # how negative its own state is).
        lowness = -sum(emo) / max(1, len(emo))

        # DANGER is NOT defined by us. the being says danger only when its OWN
        # MEANING system has learned that what it currently perceives evokes a
        # low/aversive state. (life.meaning.evokes(signal) is how much a signal
        # has come to mean lowness to the being, learned from co-occurrence
        # with its own bad states.) it fears what it has LEARNED to fear.
        try:
            worst = 0.0
            for n in life.names:
                ev = life.meaning.evokes(n)   # its own learned aversion to n
                if ev > worst:
                    worst = ev
            if worst > 0.15:        # it genuinely finds something aversive now
                said.append(("danger_near", LEXICON["danger_near"]))
        except Exception:
            pass

        # want help / safe: from its OWN felt state, not a world rule.
        if lowness > 0.2:
            said.append(("want_help", LEXICON["want_help"]))
        elif lowness < -0.1:
            said.append(("safe_now", LEXICON["safe_now"]))

        # near-death it actually lived
        try:
            if getattr(life, "_was_low", False):
                said.append(("nothing_left", LEXICON["nothing_left"]))
        except Exception:
            pass

        # a perception that recalled a hard memory (its own meaning system)
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
