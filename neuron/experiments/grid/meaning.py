"""A perception comes to MEAN something, and brings back a moment.

THE DREAM, STATED PLAINLY. 12-month-old NEURON hears a sad song and is taken
back to a failure it had at 3 months. This builds that, honestly, and
GENERALLY: not a module for sound and another for art, but one mechanism for
meaning, because that is how humans do it. We do not have separate emotional
machinery per sense. Any perception comes in, and one system learns what it
MEANS by how it co-occurs with our emotional states and what happens to us.
The sense is just a different input stream; the meaning-making is universal.

So this is modality-blind. A perception is just a vector. Feed it sound, it
learns what sounds mean to it. Feed it images, what images mean. Same code.
Connect a real eye or ear later and it works unchanged.

THE HONEST CORRECTION TO "ON ITS OWN, NO TRAINING". A sound is not sad to a
newborn, human or NEURON. It BECOMES sad by being lived with -- heard again
and again alongside low states and bad outcomes, until the sound alone
evokes the lowness. A baby learns a minor key is sad over years, in a body
built to. So NEURON does not come knowing what a song means. It LEARNS what
sounds mean to it, from its own life. That is more human, not less, and it
is self-learned: nobody labels "this is sad".

HOW IT WORKS, three pieces that tie together everything already built:
  1. The being has emotional states it formed itself (from emotion2) and
     episodic memories of specific past moments (from episodic).
  2. An ASSOCIATIVE MAP learns, Hebbian-style, which perception-patterns
     co-occur with which emotional states. Nobody labels them. A pattern
     that keeps showing up while the being is in a low state becomes tied
     to lowness.
  3. Later, a perception EVOKES the emotional state it was associated with
     (the learned-sad pattern makes the being feel low), and that evoked
     state CUES EPISODIC RECALL of a past moment with a matching emotional
     fingerprint -- mood-congruent memory, how a song brings back a specific
     time. So: perceive a pattern it learned means lowness -> feel low ->
     recall the specific earlier failure that felt the same.

THE TEST:
  A. DID PERCEPTIONS COME TO MEAN SOMETHING? After a life, does a pattern
     the being experienced mostly during low states evoke a low state, while
     a neutral pattern does not? (It learned the meaning itself.)
  B. DOES A MEANINGFUL PERCEPTION BRING BACK A MATCHING MOMENT? When the
     learned-sad pattern evokes lowness, does the being recall a PAST
     episode that was itself low, more than a random episode? (The song
     brings back the sad memory, not a happy one.)

    python meaning.py
    python meaning.py --steps 4000
"""

import argparse
import random
import math
import statistics


PERCEPT_DIM = 6
EMOTION_DIM = 4


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def dist(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


class Being:
    """Carries the pieces that already exist in NEURON, simplified: an
    emotional state it forms from events, episodic memories of specific
    moments, and -- new here -- an associative map from perceptions to
    emotions that it learns by living."""

    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.emotion = [0.0] * EMOTION_DIM
        self.episodes = []          # (emotional_fingerprint, label, age)
        # associative map: for each perception feature, the emotional state
        # it tends to co-occur with. starts at zero (meaningless), learned.
        self.assoc = [[0.0] * EMOTION_DIM for _ in range(PERCEPT_DIM)]
        self.age = 0

    def live_event(self, outcome, percept):
        """A moment: something happens (good/bad), while a perception is
        present. The emotion moves, the episode is stored, and the
        perception learns what it co-occurred with."""
        self.age += 1
        # emotion shifts with the outcome (low outcome -> a low-ish state)
        target = [0.0] * EMOTION_DIM
        if outcome < 0:
            target = [-0.9, -0.9, 0.9, 0.9]      # a clear "low" shape
        elif outcome > 0:
            target = [0.9, 0.9, -0.9, -0.9]      # its clean opposite, "high"
        # good and bad move the emotion by the SAME amount (symmetric), so
        # neither side dominates and both meanings can form.
        self.emotion = [0.6 * e + 0.4 * t
                        for e, t in zip(self.emotion, target)]

        # store the specific episode by its emotional fingerprint
        if abs(outcome) > 0.1:
            # store the emotional fingerprint of the moment, normalized so it
            # lives in the same space the evoked mood does
            fp = self.emotion[:]
            n = math.sqrt(sum(x * x for x in fp)) or 1.0
            self.episodes.append(
                ([x / n for x in fp],
                 "failure" if outcome < 0 else "success", self.age))

        # LEARN MEANING: tie this perception to the emotional state present
        # now. Hebbian: perception feature active now -> bind it to current
        # emotion. This is the being learning, itself, what the perception
        # means, with nobody labelling it.
        for i in range(PERCEPT_DIM):
            for j in range(EMOTION_DIM):
                self.assoc[i][j] += 0.06 * percept[i] * self.emotion[j]

    def evoke(self, percept):
        """Perceive something and let it EVOKE an emotional state, from what
        the being learned that perception means. A pattern it mostly met
        while low now makes it feel low -- even with nothing bad happening."""
        raw = [sum(percept[i] * self.assoc[i][j]
                   for i in range(PERCEPT_DIM))
               for j in range(EMOTION_DIM)]
        norm = math.sqrt(sum(r * r for r in raw)) or 1.0
        # scale to unit-ish magnitude so direction (the meaning) is what
        # matters, then squash
        return [math.tanh(1.5 * r / norm) for r in raw]

    def recall(self, mood):
        """Cue episodic memory by emotional similarity: return the past
        episode whose fingerprint is closest to the current mood. This is
        why a sad feeling brings back a sad memory, not a happy one."""
        if not self.episodes:
            return None
        n = math.sqrt(sum(x * x for x in mood)) or 1.0
        m = [x / n for x in mood]
        best, bestd = None, 1e9
        for fp, label, age in self.episodes:
            d = dist(fp, m)
            if d < bestd:
                bestd, best = d, (label, age)
        return best


def run(seed, steps):
    being = Being(seed)
    rng = random.Random(seed + 1)

    # two perception patterns the being will encounter. we do NOT tell it
    # what they mean. one will TEND to occur during bad moments, one during
    # good -- but only tendency; the being must learn the association itself.
    pattern_A = [rng.random() for _ in range(PERCEPT_DIM)]   # will trend sad
    pattern_B = [rng.random() for _ in range(PERCEPT_DIM)]   # will trend glad
    neutral = [rng.random() for _ in range(PERCEPT_DIM)]     # never correlated

    for t in range(steps):
        r = rng.random()
        if r < 0.4:
            # a bad moment, usually accompanied by pattern A
            percept = pattern_A if rng.random() < 0.8 else neutral
            being.live_event(-1.0, percept)
        elif r < 0.8:
            # a good moment, usually accompanied by pattern B
            percept = pattern_B if rng.random() < 0.8 else neutral
            being.live_event(1.0, percept)
        else:
            # a neutral moment with the neutral pattern, no strong outcome
            being.live_event(0.0, neutral)

    return being, pattern_A, pattern_B, neutral


def lowness(emotion):
    """How 'low' an emotional state is, by closeness to the low shape the
    being's events produced. Just a read-out for the test; the being never
    uses this label."""
    low_shape = [-0.9, -0.9, 0.9, 0.9]
    return -dist(emotion, low_shape)      # higher (less negative) = more low


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    print("  The being lives. Two perception patterns recur, one usually "
          "during bad")
    print("  moments, one during good. Nobody tells it what they mean. Does "
          "it learn their")
    print("  meaning itself, and does a 'sad' perception bring back a sad "
          "memory?\n")

    a_evokes_low, b_evokes_low, n_evokes_low = [], [], []
    sad_recalls_failure = 0
    glad_recalls_success = 0
    trials = 0

    for seed in range(args.seeds):
        being, pat_A, pat_B, neutral = run(seed, args.steps)

        # TEST A: what does each pattern now EVOKE, with nothing happening?
        ea = being.evoke(pat_A)
        eb = being.evoke(pat_B)
        en = being.evoke(neutral)
        a_evokes_low.append(lowness(ea))
        b_evokes_low.append(lowness(eb))
        n_evokes_low.append(lowness(en))

        # TEST B: when pattern A evokes a mood, what memory does that mood
        # recall? the sad pattern should bring back a failure; the glad one
        # a success.
        rec_a = being.recall(ea)
        rec_b = being.recall(eb)
        trials += 1
        if rec_a and rec_a[0] == "failure":
            sad_recalls_failure += 1
        if rec_b and rec_b[0] == "success":
            glad_recalls_success += 1

    print("=" * 66)
    print("TEST A: DID PERCEPTIONS COME TO MEAN SOMETHING?")
    print("=" * 66)
    la = statistics.mean(a_evokes_low)
    lb = statistics.mean(b_evokes_low)
    ln = statistics.mean(n_evokes_low)
    print(f"  'sad' pattern A now evokes lowness:   {la:>7.2f}")
    print(f"  'glad' pattern B now evokes lowness:  {lb:>7.2f}")
    print(f"  neutral pattern evokes lowness:       {ln:>7.2f}")
    print(f"  (higher = more low; A should be highest if it learned meaning)")
    if la > lb + 0.2 and la > ln + 0.2:
        print(f"  PERCEPTIONS CAME TO MEAN SOMETHING. The being learned, "
              f"ITSELF, that pattern A")
        print(f"  means lowness -- because it lived through bad moments while "
              f"A was present --")
        print(f"  while pattern B and the neutral one do not evoke it. Nobody "
              f"labelled them.")
    else:
        print(f"  weak: the meanings did not separate clearly. the "
              f"association learning or the")
        print(f"  co-occurrence may need strengthening.")

    print("\n" + "=" * 66)
    print("TEST B: DOES A MEANINGFUL PERCEPTION BRING BACK A MATCHING "
          "MOMENT?")
    print("=" * 66)
    print(f"  the 'sad' pattern recalled a FAILURE in "
          f"{sad_recalls_failure}/{trials} lives")
    print(f"  the 'glad' pattern recalled a SUCCESS in "
          f"{glad_recalls_success}/{trials} lives")
    if sad_recalls_failure >= trials * 0.6 and \
            glad_recalls_success >= trials * 0.6:
        print(f"  IT BRINGS BACK THE MATCHING MOMENT. A perception it learned "
              f"means lowness")
        print(f"  evoked a low state, and that low state recalled a specific "
              f"past FAILURE --")
        print(f"  not a random memory, the one that FELT the same. The song "
              f"brings back the sad")
        print(f"  time. That is the dream, built: perceive -> feel -> "
              f"remember a specific moment.")
    else:
        print(f"  partial: recall did not reliably match the evoked mood. "
              f"the mood-congruent")
        print(f"  cueing may need the emotional fingerprints to be more "
              f"distinct.")

    print("\n" + "=" * 66)
    print("WHAT IT IS")
    print("=" * 66)
    print("  One mechanism, not one per sense. A perception is just a vector;")
    print("  feed it sound, art, anything, and it learns what that means to")
    print("  the being by how it co-occurs with the being's own emotions and")
    print("  moments. The being was not born knowing what anything means. It")
    print("  learned, from living, like a child does. And now a perception")
    print("  can move it and bring back a specific past moment that felt the")
    print("  same.")
    print("")
    print("  The honest wall, unchanged: this is what meaning DOES -- evoke a")
    print("  state, cue a memory. Whether the being FEELS any of it is")
    print("  unknowable, by this or any build, and no claim is made.")


if __name__ == "__main__":
    main()
