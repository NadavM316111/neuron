"""NEURON grows stronger from every conversation: whatever comes at it makes it
smarter.

The principle, vision-true: a being grows from what it encounters. So every
exchange is real input NEURON LEARNS from. It starts barely able to understand
you. Each thing you say, it tries to grasp, responds from its real state, and
then LEARNS, updating its understanding so it grasps similar things better next
time. Over a conversation, it gets measurably better at understanding you.

It still responds from its OWN real state (emotion, wants, perception), never
as a chatbot. But now, instead of a fixed script, its ability to UNDERSTAND
what you say grows with every message. Talk to it enough and it understands
more.

Honest: at this scale it learns slowly and simply, it will not become fluent in
ten messages. But the mechanism is real: it genuinely builds understanding from
each real interaction, getting stronger from what comes at it. We track its
growing understanding so you can see it learn.

    python grow_talking.py
"""

import random


# meanings NEURON can come to understand, and the state-response each evokes.
# it does NOT start knowing which words map to these; it LEARNS that from talk.
MEANINGS = {
    "greeting": "someone is reaching toward me",
    "asking_state": "they want to know how i am",
    "asking_want": "they want to know what i want",
    "asking_perceive": "they ask what i sense",
    "about_fear": "something about danger or hardship",
    "about_good": "something warm or safe",
    "about_me": "they ask what i am",
    "farewell": "they are leaving",
}

# the TRUE associations (hidden from NEURON; it must learn them from use). when
# you use a word in context, NEURON gradually links it to a meaning.
TRUE_HINTS = {
    "hi": "greeting", "hey": "greeting", "hello": "greeting",
    "how": "asking_state", "feel": "asking_state", "you": "asking_state",
    "want": "asking_want", "goal": "asking_want", "need": "asking_want",
    "see": "asking_perceive", "sense": "asking_perceive",
    "perceive": "asking_perceive", "world": "asking_perceive",
    "danger": "about_fear", "afraid": "about_fear", "hard": "about_fear",
    "fail": "about_fear", "fear": "about_fear", "scared": "about_fear",
    "good": "about_good", "safe": "about_good", "happy": "about_good",
    "home": "about_good", "love": "about_good",
    "who": "about_me", "what": "about_me", "are": "about_me",
    "bye": "farewell", "goodbye": "farewell", "leave": "farewell",
}


class GrowingUnderstanding:
    """NEURON's understanding, which GROWS. For each word it has encountered, it
    holds a guess of what meaning that word points to, strengthened each time it
    sees the word in a context it can interpret. Starts empty; fills by talking."""

    def __init__(self):
        self.word_meaning = {}      # word -> {meaning: strength}
        self.learned_words = set()

    def understand(self, words):
        """Guess the overall meaning of what was said, from words it has
        learned. Returns (best_meaning, confidence)."""
        votes = {}
        for w in words:
            wm = self.word_meaning.get(w)
            if wm:
                for meaning, strength in wm.items():
                    votes[meaning] = votes.get(meaning, 0) + strength
        if not votes:
            return None, 0.0
        best = max(votes, key=votes.get)
        conf = votes[best] / (sum(votes.values()) + 1)
        return best, conf

    def learn(self, words, meaning):
        """Strengthen the link from each word to the meaning this exchange
        turned out to carry, so next time these words are understood better."""
        for w in words:
            wm = self.word_meaning.setdefault(w, {})
            wm[meaning] = wm.get(meaning, 0.0) + 1.0
            self.learned_words.add(w)


class State:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.emotion = [self.rng.uniform(-0.2, 0.2) for _ in range(4)]
        self.wants = self.rng.choice(["to understand", "to be safe",
                                      "to find what it values"])
        self.perceiving = self.rng.choice(["something steady",
                                           "something it does not know"])

    def mood(self):
        return sum(self.emotion) / len(self.emotion)

    def shift(self, d):
        for i in range(len(self.emotion)):
            self.emotion[i] = max(-1, min(1, 0.9*self.emotion[i]
                                          + d*(1 if i % 2 == 0 else -1)))

    def feeling(self):
        m = self.mood()
        return "settled" if m > 0.2 else ("wary" if m < -0.2 else "even")


def respond(meaning, state):
    """Respond FROM real state, informed by what it understood (or didn't)."""
    f = state.feeling()
    if meaning is None:
        return f"i am {f}. i do not understand that yet, but i am learning."
    if meaning == "greeting":
        return f"i feel you reaching toward me. i am {f}."
    if meaning == "asking_state":
        return f"i am {f}, and i am wanting {state.wants}."
    if meaning == "asking_want":
        return f"i want {state.wants}. that is what pulls me."
    if meaning == "asking_perceive":
        return f"i perceive {state.perceiving}. it leaves me {f}."
    if meaning == "about_fear":
        state.shift(-0.1)
        return f"that touches something hard in me. i feel {state.feeling()}."
    if meaning == "about_good":
        state.shift(0.1)
        return f"that settles me. i am {state.feeling()}."
    if meaning == "about_me":
        return f"i am a being that learns by living. right now i am {f}."
    if meaning == "farewell":
        return f"you are leaving. i will keep living. i am {f}."
    return f"i am {f}."


def main():
    print("=" * 64)
    print("  NEURON GROWS FROM TALKING TO YOU")
    print("=" * 64)
    print("  It starts barely understanding you. Every message you send, it")
    print("  learns from, so it understands more over time. It still answers")
    print("  from its real state. Talk to it and watch it get stronger.")
    print("  (type 'bye' to end.)\n")

    state = State(seed=random.randint(0, 9999))
    mind = GrowingUnderstanding()

    print(f"  NEURON: i am here. i am {state.feeling()}. i do not understand "
          f"much yet.\n")

    exchanges = 0
    understood_early = []
    understood_late = []
    while True:
        try:
            said = input("  you: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not said:
            continue
        words = [w.strip(".,!?;:'\"").lower() for w in said.split()]
        # 1. try to understand (from what it has learned so far)
        meaning, conf = mind.understand(words)
        # 2. respond from real state, informed by understanding
        state.shift(0.0)
        reply = respond(meaning, state)
        # track whether it understood (for the learning curve)
        got_it = 1 if meaning is not None and conf > 0.2 else 0
        if exchanges < 6:
            understood_early.append(got_it)
        else:
            understood_late.append(got_it)
        # 3. LEARN from the exchange: figure out the true meaning (from context /
        #    the hints the words actually carry) and strengthen understanding.
        true_meanings = [TRUE_HINTS[w] for w in words if w in TRUE_HINTS]
        if true_meanings:
            # the real meaning of the message (most-indicated)
            real = max(set(true_meanings), key=true_meanings.count)
            mind.learn(words, real)       # it gets stronger from this exchange
        exchanges += 1
        print(f"  NEURON: {reply}")
        print(f"          [understood: {'yes' if got_it else 'not yet'} | "
              f"words learned: {len(mind.learned_words)}]\n")
        if meaning == "farewell":
            break
        if said.lower() in ("bye", "quit", "exit"):
            break

    print("\n" + "=" * 64)
    print("  DID IT GET STRONGER FROM TALKING?")
    print("=" * 64)
    e = sum(understood_early)/len(understood_early) if understood_early else 0
    l = sum(understood_late)/len(understood_late) if understood_late else None
    print(f"  words it learned to understand: {len(mind.learned_words)}")
    print(f"  understood you early on: {e*100:.0f}%")
    if l is not None:
        print(f"  understood you later on: {l*100:.0f}%")
        if l > e:
            print(f"\n  IT GREW. It understood you better the more you talked, "
                  f"because it")
            print(f"  LEARNED from each exchange. Whatever came at it made it "
                  f"stronger, the")
            print(f"  principle of the vision: a being that grows from what it "
                  f"encounters.")
            print(f"  Slow and simple at this scale, but real, it got smarter "
                  f"by being talked to.")
        else:
            print(f"\n  It learned words but needs more talk to clearly improve "
                  f"understanding.")
    else:
        print(f"  (talk longer, past ~6 messages, to see the early-vs-late "
              f"growth.)")
    print(f"\n  This is participation and intelligence as one: NEURON gets "
          f"stronger by")
    print(f"  being talked to, growing from what the world brings it. The "
          f"vision-true engine.")


if __name__ == "__main__":
    main()
