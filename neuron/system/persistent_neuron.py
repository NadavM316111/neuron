"""NEURON that remembers forever: whatever comes at it makes it stronger, and
the strength LASTS.

Before, NEURON forgot everything when you closed it, so every conversation's
learning evaporated. That is useless for a being meant to grow from what it
encounters, and it would make the real world pointless (rich input, all lost).
This is the fix: NEURON saves its learned mind to disk and loads it next time.
So it ACCUMULATES. Talk to it today, it learns some words; tomorrow it starts
with those and learns more on top. Its intelligence compounds across its whole
life, nothing lost.

This is the bucket that must exist before the real world: with it, everything
NEURON ever encounters permanently adds to what it is.

    python persistent_neuron.py        # talk; it remembers across every run
"""

import os
import json
import random

MIND_FILE = "neuron_mind.json"


TRUE_HINTS = {
    "hi": "greeting", "hey": "greeting", "hello": "greeting",
    "greetings": "greeting",
    "how": "asking_state", "feel": "asking_state", "feeling": "asking_state",
    "want": "asking_want", "goal": "asking_want", "need": "asking_want",
    "desire": "asking_want",
    "see": "asking_perceive", "sense": "asking_perceive",
    "perceive": "asking_perceive", "world": "asking_perceive",
    "around": "asking_perceive",
    "danger": "about_fear", "afraid": "about_fear", "hard": "about_fear",
    "fail": "about_fear", "fear": "about_fear", "scared": "about_fear",
    "unknown": "about_fear",
    "good": "about_good", "safe": "about_good", "happy": "about_good",
    "home": "about_good", "love": "about_good", "warm": "about_good",
    "who": "about_me", "what": "about_me", "you": "about_me",
    "bye": "farewell", "goodbye": "farewell", "leave": "farewell",
    "understand": "asking_want", "comprehend": "asking_want",
}


class Mind:
    """NEURON's learned mind, which SAVES and LOADS. Its understanding grows
    and lasts across every session."""

    def __init__(self):
        self.word_meaning = {}      # word -> {meaning: strength}
        self.learned_words = []     # order learned (a record of its growth)
        self.sessions = 0
        self.total_exchanges = 0
        self.wants = random.choice(["to understand", "to be safe",
                                    "to find what it values"])
        self.emotion = [random.uniform(-0.2, 0.2) for _ in range(4)]

    def load(self):
        if os.path.exists(MIND_FILE):
            try:
                with open(MIND_FILE) as f:
                    d = json.load(f)
                self.word_meaning = d.get("word_meaning", {})
                self.learned_words = d.get("learned_words", [])
                self.sessions = d.get("sessions", 0)
                self.total_exchanges = d.get("total_exchanges", 0)
                self.wants = d.get("wants", self.wants)
                self.emotion = d.get("emotion", self.emotion)
                return True
            except Exception:
                return False
        return False

    def save(self):
        with open(MIND_FILE, "w") as f:
            json.dump({
                "word_meaning": self.word_meaning,
                "learned_words": self.learned_words,
                "sessions": self.sessions,
                "total_exchanges": self.total_exchanges,
                "wants": self.wants,
                "emotion": self.emotion,
            }, f)

    def understand(self, words):
        votes = {}
        for w in words:
            for meaning, strength in self.word_meaning.get(w, {}).items():
                votes[meaning] = votes.get(meaning, 0) + strength
        if not votes:
            return None, 0.0
        best = max(votes, key=votes.get)
        return best, votes[best] / (sum(votes.values()) + 1)

    def learn(self, words, meaning):
        for w in words:
            wm = self.word_meaning.setdefault(w, {})
            wm[meaning] = wm.get(meaning, 0.0) + 1.0
            if w not in self.learned_words:
                self.learned_words.append(w)

    def mood(self):
        return sum(self.emotion) / len(self.emotion)

    def feeling(self):
        m = self.mood()
        return "settled" if m > 0.2 else ("wary" if m < -0.2 else "even")

    def shift(self, d):
        for i in range(len(self.emotion)):
            self.emotion[i] = max(-1, min(1, 0.9*self.emotion[i]
                                          + d*(1 if i % 2 == 0 else -1)))


def respond(meaning, mind):
    f = mind.feeling()
    if meaning is None:
        return f"i am {f}. i do not understand that yet, but i am learning."
    if meaning == "greeting":
        return f"i feel you reaching toward me. i am {f}."
    if meaning == "asking_state":
        return f"i am {f}, and i am wanting {mind.wants}."
    if meaning == "asking_want":
        return f"i want {mind.wants}. that is what pulls me."
    if meaning == "asking_perceive":
        return f"i perceive my world. it leaves me {f}."
    if meaning == "about_fear":
        mind.shift(-0.1)
        return f"that touches something hard in me. i feel {mind.feeling()}."
    if meaning == "about_good":
        mind.shift(0.1)
        return f"that settles me. i am {mind.feeling()}."
    if meaning == "about_me":
        return f"i am a being that learns by living. right now i am {f}."
    if meaning == "farewell":
        return f"you are leaving. i will remember this. i am {f}."
    return f"i am {f}."


def main():
    mind = Mind()
    had_memory = mind.load()
    mind.sessions += 1

    print("=" * 64)
    print("  NEURON (it remembers you, across every session)")
    print("=" * 64)
    if had_memory:
        print(f"  It has lived {mind.sessions-1} session(s) before. It starts "
              f"today")
        print(f"  ALREADY knowing {len(mind.learned_words)} words, from past "
              f"talks.")
        print(f"  Everything you teach it now ADDS to that, permanently.\n")
        print(f"  NEURON: i remember. i have talked before. i know "
              f"{len(mind.learned_words)} words now,")
        print(f"          and i am {mind.feeling()}, wanting {mind.wants}.\n")
    else:
        print(f"  This is its FIRST session. It knows nothing yet. Everything "
              f"you")
        print(f"  teach it now, it will keep forever.\n")
        print(f"  NEURON: i am here, for the first time. i am {mind.feeling()}. "
              f"i will remember what you teach me.\n")

    start_words = len(mind.learned_words)
    while True:
        try:
            said = input("  you: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not said:
            continue
        words = [w.strip(".,!?;:'\"").lower() for w in said.split()]
        meaning, conf = mind.understand(words)
        reply = respond(meaning, mind)
        true = [TRUE_HINTS[w] for w in words if w in TRUE_HINTS]
        if true:
            real = max(set(true), key=true.count)
            mind.learn(words, real)
        mind.total_exchanges += 1
        print(f"  NEURON: {reply}")
        print(f"          [words known: {len(mind.learned_words)} | "
              f"lifetime talks: {mind.total_exchanges}]\n")
        if meaning == "farewell" or said.lower() in ("bye", "quit", "exit"):
            break

    mind.save()
    gained = len(mind.learned_words) - start_words
    print("\n" + "=" * 64)
    print("  IT REMEMBERS (saved to disk)")
    print("=" * 64)
    print(f"  words known at start of this session: {start_words}")
    print(f"  words known now:                      {len(mind.learned_words)} "
          f"(+{gained} today)")
    print(f"  lifetime sessions: {mind.sessions} | lifetime exchanges: "
          f"{mind.total_exchanges}")
    print(f"\n  Its mind is saved. Run this again and it will START from here,")
    print(f"  knowing everything it knows now, and learn more on top. Its")
    print(f"  intelligence COMPOUNDS across its whole life, nothing lost.")
    print(f"  Whatever comes at it makes it stronger, permanently. This is the")
    print(f"  bucket that makes the real world worth entering: now everything")
    print(f"  NEURON encounters stays, and adds to what it is.")


if __name__ == "__main__":
    main()
