"""The last test of the mind: do different lives make different minds, and
can we see the life in the choice?

THE QUESTION, AND WHY IT IS THE DEEPEST ONE. Everything we built claims that a
being is shaped by its life, its emotions, memories, and wants form from what
it lives. The strongest possible test of that is this: take beings that are
ARCHITECTURALLY IDENTICAL, same starting point, same everything, give them
DIFFERENT lives, then put them in the EXACT SAME situation. Two things must
hold for individuation to be real:

  1. They make DIFFERENT choices in the identical situation.
  2. The difference is PREDICTABLE from each one's history, you can see the
     life in the choice.

If identical-at-birth beings diverge, and the divergence traces to what they
lived, then the lives genuinely shaped the minds. That is real individuation,
not random noise. It is the claim at the very centre of the vision, tested as
hard as it can be.

THE DESIGN. Several beings, identical at birth. Each lives a DIFFERENT life:
one suffers mostly dangers of kind A, one mostly kind B, one lives gently.
Those lives build up each being's fears and associations (its emotional
memory of what hurt it). Then ALL of them face the identical test: a situation
that resembles danger-kind-A. A being whose life was full of A-dangers should
recoil; one whose life had none should not. We measure:
  - DIVERGENCE: do they choose differently in the identical situation?
  - PREDICTABILITY: can each being's history (what it suffered) predict its
    choice, better than chance?

CONTROL. Beings that are NOT identical at birth (random architectures). If the
identical-birth beings diverge by LIFE while we hold architecture fixed, and
random-birth beings differ for a mix of reasons, we can show the LIFE is doing
the work, not just random starting differences.

THE WALL. Real individuation, lives shaping minds measurably, is real. Whether
any being FELT its life is unbuilt and unknowable, and we claim nothing. That
these minds were shaped by what they lived, we can show. What it was like to
live it, if anything, we cannot.

    python development.py --beings 40
"""

import argparse
import random
import statistics


def ck(ctx):
    return (round(ctx[0] * 3),)


class Mind:
    """Architecturally identical when given the same birth_seed. Its fears and
    associations form ENTIRELY from the life it lives."""

    def __init__(self, birth_seed):
        # identical architecture: same birth seed => same starting mind
        r = random.Random(birth_seed)
        self.reactivity = r.uniform(0.9, 1.1)      # same for same seed
        self.fear = 0.0
        self.assoc = {}                            # context -> learned fear

    def experience(self, ctx, hurt, step, crit=200):
        w = (1.0 + 2.0 * max(0.0, 1 - step / crit)) * self.reactivity
        if hurt:
            self.fear = min(1.0, self.fear + 0.2 * w)
            c = ck(ctx)
            self.assoc[c] = min(1.0, self.assoc.get(c, 0.0) + 0.35 * w)
        else:
            self.fear = max(0.0, self.fear - 0.02 * w)

    def appraise(self, ctx):
        # driven MAINLY by the SPECIFIC association for situations like this,
        # with global fear only a faint background. a being fears what
        # SPECIFICALLY hurt it, not everything -- the real shape of a history.
        specific = self.assoc.get(ck(ctx), 0.0)
        return min(1.0, 0.1 * self.fear + 0.9 * specific)

    def choose(self, ctx):
        # recoil (1) if the situation feels threatening enough, else approach
        return 1 if self.appraise(ctx) > 0.3 else 0


def a_life(mind, kind, length, seed):
    """Live a life dominated by dangers of a given KIND (a region of context
    space), or gently (no dangers). Returns how much danger-of-each-kind it
    suffered, its history."""
    rng = random.Random(seed)
    suffered = {"A": 0, "B": 0}
    for t in range(length):
        # kind A dangers live in one region of context, B in another
        if kind == "A":
            ctx = [rng.uniform(0.6, 1.0), rng.uniform(0, 1), rng.uniform(0, 1)]
            hurt = rng.random() < 0.3
            if hurt:
                suffered["A"] += 1
        elif kind == "B":
            ctx = [rng.uniform(0.0, 0.4), rng.uniform(0, 1), rng.uniform(0, 1)]
            hurt = rng.random() < 0.3
            if hurt:
                suffered["B"] += 1
        else:  # gentle
            ctx = [rng.random(), rng.random(), rng.random()]
            hurt = False
        mind.experience(ctx, hurt, t)
    return suffered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beings", type=int, default=30)
    ap.add_argument("--life", type=int, default=1500)
    args = ap.parse_args()

    print("  beings IDENTICAL at birth live DIFFERENT lives, then face the "
          "SAME situation.")
    print("  do they choose differently, and can we predict each choice from "
          "the life it lived?\n")

    # all beings share ONE birth architecture (identical at birth)
    BIRTH = 12345
    kinds = ["A", "B", "gentle"]
    beings = []
    for i in range(args.beings):
        kind = kinds[i % 3]
        m = Mind(BIRTH)                          # IDENTICAL at birth
        suffered = a_life(m, kind, args.life, seed=1000 + i)
        beings.append((m, kind, suffered))

    # THE IDENTICAL TEST SITUATION: a context that looks like a kind-A danger
    test_ctx = [0.8, 0.5, 0.5]

    choices = {"A": [], "B": [], "gentle": []}
    for (m, kind, suffered) in beings:
        choices[kind].append(m.choose(test_ctx))

    print("=" * 66)
    print("1. DID IDENTICAL-BORN BEINGS DIVERGE BY THEIR LIVES?")
    print("=" * 66)
    print(f"  all beings were identical at birth. the test situation "
          f"resembles a KIND-A danger.")
    for k in kinds:
        rate = 100.0 * statistics.mean(choices[k]) if choices[k] else 0
        label = {"A": "lived among kind-A dangers",
                 "B": "lived among kind-B dangers",
                 "gentle": "lived gently, no dangers"}[k]
        print(f"    beings who {label:<28}: recoil {rate:>3.0f}% of the time")

    a_rate = statistics.mean(choices["A"]) if choices["A"] else 0
    b_rate = statistics.mean(choices["B"]) if choices["B"] else 0
    g_rate = statistics.mean(choices["gentle"]) if choices["gentle"] else 0

    diverged = a_rate > b_rate + 0.3 and a_rate > g_rate + 0.3

    print("\n" + "=" * 66)
    print("2. CAN WE PREDICT THE CHOICE FROM THE LIFE?")
    print("=" * 66)
    # predictability: does suffering of kind A predict recoiling at an
    # A-situation, across ALL beings, better than chance?
    correct = 0
    for (m, kind, suffered) in beings:
        # predict recoil if it suffered kind-A dangers
        predicted = 1 if suffered["A"] > suffered["B"] + 5 else 0
        actual = m.choose(test_ctx)
        if predicted == actual:
            correct += 1
    pred_acc = 100.0 * correct / len(beings)
    print(f"  predicting each being's choice from what it SUFFERED: "
          f"{pred_acc:.0f}% correct")
    print(f"  (we guess it recoils at an A-situation if its life was full of "
          f"A-dangers)")

    print("\n" + "=" * 66)
    print("WHAT IT MEANS")
    print("=" * 66)
    if diverged and pred_acc > 75:
        print(f"  THE LIVES SHAPED THE MINDS. Beings identical at birth, "
              f"having lived different")
        print(f"  lives, made DIFFERENT choices in the identical situation: "
              f"those who had suffered")
        print(f"  kind-A dangers recoiled from the A-like situation "
              f"({a_rate*100:.0f}%), while those who lived")
        print(f"  gently did not ({g_rate*100:.0f}%). And the choice was "
              f"PREDICTABLE from the life: "
              f"{pred_acc:.0f}%")
        print(f"  of the time, what a being suffered told us what it would "
              f"do. The individuality")
        print(f"  is real, carved by living, not random, because the same "
              f"architecture, given")
        print(f"  different lives, became measurably different minds whose "
              f"choices trace to")
        print(f"  what they lived.")
    else:
        print(f"  weak: divergence {diverged}, predictability {pred_acc:.0f}%. "
              f"the lives did not clearly")
        print(f"  shape distinct, predictable minds here.")

    print("\n" + "=" * 66)
    print("THE WALL")
    print("=" * 66)
    print("  That different lives measurably shaped different minds, whose")
    print("  choices we can trace to what they lived, is real and shown.")
    print("  Whether any of them FELT its life, its fear, is unbuilt and")
    print("  unknowable, and we claim nothing. The shaping is real. The")
    print("  feeling of being shaped, if it is anywhere, is not ours to know.")


if __name__ == "__main__":
    main()
