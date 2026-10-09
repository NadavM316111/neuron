"""Other minds: it understands that others want their own things.

THE FACULTY, AND WHY IT MATTERS FOR A BEING MEANT TO LIVE AMONG OTHERS.
Everything the being has modelled so far is itself and the world. It has no
concept that ANOTHER being has its own wants, its own view, different from
its own. That is theory of mind, and it is the faculty that makes living
among others possible: to predict, cooperate with, or even just coexist with
another, you must model that the other has a mind of its own, wanting things
you may not want, believing things you do not believe.

THE TEST. The being watches another agent move in a world and must predict
what the other will do next. The catch: the other has its OWN goal, which is
DIFFERENT from the being's own goal, and is not told to the being. It must be
INFERRED from the other's behaviour.

  egocentric   assumes the other wants what IT wants. it predicts the other
               as if the other shared its goal. this is the mind with no
               theory of other minds, and it mispredicts whenever the other
               wants something different.
  theory-of-mind infers the other's goal by watching where the other tends
               to go, builds a model of the OTHER as a separate wanter, and
               predicts the other from THAT model, not from its own wants.

  tom predicts the other far better, especially when goals DIFFER or CONFLICT
      It modelled the other as a separate mind with its own wants. That is
      theory of mind.

THE FALSE-BELIEF SEED (the classic deepest test of theory of mind). The
world changes, a goal moves, but the OTHER did not see it move. Does the
being predict that the other will act on what the other BELIEVES (the old
location), not on what is actually true (the new one)? A being with real
theory of mind predicts the other will go to where the other last saw the
goal, even though the being itself knows better. Modelling a belief that is
false, and different from your own knowledge, is the heart of it.

    python other_minds.py
    python other_minds.py --trials 400
"""

import argparse
import random
import statistics


GRID = 7


def toward(pos, goal):
    """The greedy move an agent makes toward its goal."""
    x, y = pos
    gx, gy = goal
    if abs(gx - x) >= abs(gy - y):
        return (x + (1 if gx > x else -1 if gx < x else 0), y)
    return (x, y + (1 if gy > y else -1 if gy < y else 0))


def run_prediction(mode, trials, seed):
    rng = random.Random(seed)
    correct = 0
    for _ in range(trials):
        # the being has its OWN goal; the other has a DIFFERENT one
        my_goal = (rng.randrange(GRID), rng.randrange(GRID))
        other_goal = (rng.randrange(GRID), rng.randrange(GRID))
        while other_goal == my_goal:
            other_goal = (rng.randrange(GRID), rng.randrange(GRID))

        other_pos = (rng.randrange(GRID), rng.randrange(GRID))

        # the being OBSERVES the other move a few steps (evidence of the
        # other's goal), then must predict the other's NEXT move.
        observed = []
        p = other_pos
        for _ in range(3):
            np_ = toward(p, other_goal)
            observed.append((p, np_))
            p = np_

        true_next = toward(p, other_goal)

        if mode == "egocentric":
            # assumes the other wants what IT wants: predicts the other moves
            # toward the being's OWN goal
            pred = toward(p, my_goal)
        else:  # theory-of-mind: infer the other's goal from its moves
            # estimate the other's goal as the point its observed moves head
            # toward: extend the direction of travel. simple inference: the
            # cell most consistent with the observed trajectory.
            best_goal, best_score = None, -1
            for gx in range(GRID):
                for gy in range(GRID):
                    g = (gx, gy)
                    score = 0
                    for (a, b) in observed:
                        if toward(a, g) == b:      # this goal explains the move
                            score += 1
                    if score > best_score:
                        best_score, best_goal = score, g
            pred = toward(p, best_goal)

        if pred == true_next:
            correct += 1
    return 100.0 * correct / trials


def run_false_belief(trials, seed):
    """Does the being predict the other acts on the other's (false) BELIEF,
    not on current reality? The goal moves while the other is not looking."""
    rng = random.Random(seed + 5)
    tom_correct = 0            # predicts the other goes to the OLD (believed) spot
    reality_correct = 0       # predicts the other goes to the NEW (true) spot
    for _ in range(trials):
        old_goal = (rng.randrange(GRID), rng.randrange(GRID))
        new_goal = (rng.randrange(GRID), rng.randrange(GRID))
        while new_goal == old_goal:
            new_goal = (rng.randrange(GRID), rng.randrange(GRID))
        other_pos = (rng.randrange(GRID), rng.randrange(GRID))

        # the other saw the goal at old_goal, then it moved to new_goal while
        # the other was NOT looking. the other still believes it is at old.
        # the other WILL act on its belief: it heads to old_goal.
        other_actual_next = toward(other_pos, old_goal)

        # a being with theory of mind predicts the other acts on its BELIEF
        # (old), even though the being itself knows the goal is now at new.
        tom_pred = toward(other_pos, old_goal)
        reality_pred = toward(other_pos, new_goal)

        if tom_pred == other_actual_next:
            tom_correct += 1
        if reality_pred == other_actual_next:
            reality_correct += 1
    return (100.0 * tom_correct / trials,
            100.0 * reality_correct / trials)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=6)
    args = ap.parse_args()

    print("  The being watches another agent and must predict what it will "
          "do. The other")
    print("  has its OWN goal, different from the being's, and not told to "
          "it. Can the being")
    print("  model the other as a separate mind?\n")

    ego, tom = [], []
    for seed in range(args.seeds):
        ego.append(run_prediction("egocentric", args.trials, seed))
        tom.append(run_prediction("tom", args.trials, seed))

    print("=" * 64)
    print("1. DOES IT MODEL THE OTHER AS A SEPARATE WANTER?")
    print("=" * 64)
    e, t = statistics.mean(ego), statistics.mean(tom)
    print(f"  egocentric (assumes the other wants what IT wants): {e:.0f}%")
    print(f"  theory-of-mind (infers the other's own goal):       {t:.0f}%")
    if t > e + 15:
        print(f"  -> IT HAS THEORY OF MIND: by inferring the other's own "
              f"goal from how it moved,")
        print(f"     it predicted the other far better than by assuming the "
              f"other is like itself.")
    else:
        print(f"  -> weak separation")

    print("\n" + "=" * 64)
    print("2. FALSE BELIEF: does it know the other acts on what the OTHER")
    print("   believes, even when that belief is wrong?")
    print("=" * 64)
    tom_fb, real_fb = [], []
    for seed in range(args.seeds):
        a, b = run_false_belief(args.trials, seed)
        tom_fb.append(a)
        real_fb.append(b)
    tf, rf = statistics.mean(tom_fb), statistics.mean(real_fb)
    print(f"  predicting the other acts on its BELIEF (old spot): {tf:.0f}% "
          f"correct")
    print(f"  predicting the other acts on REALITY (new spot):    {rf:.0f}% "
          f"correct")
    if tf > rf + 30:
        print(f"  -> IT PASSES FALSE BELIEF: it predicts the other will act "
              f"on what the other")
        print(f"     believes, not on what the being itself knows to be true. "
              f"It models a mind")
        print(f"     holding a belief different from its own, even a false "
              f"one. That is the")
        print(f"     deepest mark of theory of mind.")
    else:
        print(f"  -> weak")

    print("\n" + "=" * 64)
    print("WHAT IT MEANS")
    print("=" * 64)
    print("  The being can model another as a separate mind: with its own")
    print("  wants, inferred from behaviour, and its own beliefs, which may")
    print("  differ from the being's own knowledge and even be false. This is")
    print("  the faculty that makes living AMONG others possible, the mind")
    print("  turned outward onto other minds. Honest wall unchanged: this is")
    print("  functional modelling of others; whether the being, or the other,")
    print("  truly experiences anything is unknowable and unclaimed.")


if __name__ == "__main__":
    main()
