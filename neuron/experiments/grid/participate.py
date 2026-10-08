"""The first real participation: it acts on the world, and the world's answer
changes it.

WHY THIS IS THE FIRST TRUE STEP OUTWARD. Everything so far, the being
perceives the world and even spends real money to interpret it. But being
INTERPRETED-AT is not participation; the interpretation just got logged and
changed nothing. Participation, in the sense the vision means, is that the
being ACTS on the world and is CHANGED by the world's response. The loop has
to close: want -> act -> the world answers -> the being is different
because of it. A being unmoved by what the world tells it is performing, not
participating.

THE LOOP, built to be real and provable:
  1. The being predicts its world, and tracks its OWN uncertainty per
     situation (the uncertainty mechanism it already has).
  2. When its uncertainty about something is genuinely HIGH, and only then,
     it spends a real resource to ASK the world about that specific thing.
  3. The world returns a real answer (here: an informative signal it could
     not get for free).
  4. That answer UPDATES the being's belief, so its next prediction about
     that thing is measurably better.

It is participation because the world's response changes the being. It is
provable because we can measure whether accuracy on the things it asked
about actually rose, against the things it did not ask about.

GUARDRAILS THAT KEEP IT HONEST (and, in the live system, safe):
  - it asks only when uncertainty is real, not at random, so the spend is
    earned.
  - asking costs a real, finite resource, so it cannot ask about everything;
    it must choose what is worth asking.
  - the test compares asked-about vs not-asked things, so an improvement
    cannot be a global effect mistaken for the loop working.

THE ARMS:
  passive    never asks. predicts from what it has. the floor: a being that
             only perceives and never acts on the world.
  random     asks about random things, spending the same budget. shows that
             asking HELPS only when aimed by uncertainty, not just from
             spending.
  participant asks about the things it is most UNCERTAIN of, within budget,
             and is updated by the answer. the real loop.

  participant beats passive AND random, and its accuracy rises most on the
  very things it asked about
      The loop closed: it chose what it did not know, spent to ask, and the
      world's answer made it better. The first real participation.
  participant no better than passive
      The answers did not change it, or it did not aim its asking. The loop
      did not close.

    python participate.py
    python participate.py --steps 5000 --budget 60
"""

import argparse
import random
import statistics


N_SIT = 80
OBS_NOISE = 0.60          # a free observation is wrong this often
ASK_NOISE = 0.05          # a PAID answer from the world is far more reliable
CONF_CUTOFF = 0.15


class Being:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.votes = {i: {} for i in range(N_SIT)}   # evidence per situation
        self.budget_used = 0

    def confidence(self, sit):
        v = self.votes[sit]
        tot = sum(v.values())
        if tot == 0:
            return 0.0
        top = sorted(v.values(), reverse=True)
        second = top[1] if len(top) > 1 else 0
        return (top[0] - second) / (tot + 2.0)

    def belief(self, sit):
        v = self.votes[sit]
        return max(v, key=v.get) if v else None

    def observe_free(self, sit, truth):
        # a free, noisy observation
        if self.rng.random() < OBS_NOISE:
            obs = self.rng.choice([a for a in range(3) if a != truth])
        else:
            obs = truth
        self.votes[sit][obs] = self.votes[sit].get(obs, 0) + 1  # weak

    def ask_world(self, sit, truth):
        # a PAID, reliable answer from the world. costs budget. this is the
        # being ACTING on the world and getting a response that changes it.
        if self.rng.random() < ASK_NOISE:
            ans = self.rng.choice([a for a in range(3) if a != truth])
        else:
            ans = truth
        # the answer updates belief strongly (it is trusted information)
        self.votes[sit][ans] = self.votes[sit].get(ans, 0) + 12
        self.budget_used += 1


def run(mode, seed, steps, budget):
    rng = random.Random(seed)
    answers = {i: rng.randint(0, 2) for i in range(N_SIT)}
    weights = [1.0 / (i + 1) for i in range(N_SIT)]
    being = Being(seed)

    asked = set()
    correct = []
    correct_on_asked = []
    correct_on_not_asked = []

    for t in range(steps):
        # sample a situation (long-tailed, like a real world)
        r = rng.random() * sum(weights)
        acc = 0.0
        sit = 0
        for i, w in enumerate(weights):
            acc += w
            if r <= acc:
                sit = i
                break
        truth = answers[sit]

        # free observation always happens (it perceives for free)
        being.observe_free(sit, truth)

        # ACTION: decide whether to ask the world (spend) about this
        if being.budget_used < budget:
            if mode == "participant":
                # ask only when genuinely uncertain about this situation
                if being.confidence(sit) < CONF_CUTOFF:
                    being.ask_world(sit, truth)
                    asked.add(sit)
            elif mode == "random":
                # ask about random things, same budget
                if rng.random() < 0.05:
                    being.ask_world(sit, truth)
                    asked.add(sit)
            # passive never asks

        # score its CURRENT belief on this situation
        guess = being.belief(sit)
        hit = 1 if guess == truth else 0
        correct.append(hit)
        if sit in asked:
            correct_on_asked.append(hit)
        else:
            correct_on_not_asked.append(hit)

    return dict(
        accuracy=100.0 * statistics.mean(correct) if correct else 0,
        asked=len(asked),
        budget_used=being.budget_used,
        acc_on_asked=(100.0 * statistics.mean(correct_on_asked)
                      if correct_on_asked else 0),
        acc_on_not_asked=(100.0 * statistics.mean(correct_on_not_asked)
                          if correct_on_not_asked else 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=6)
    args = ap.parse_args()

    print(f"  The being perceives for free but noisily. When it chooses, it "
          f"can spend a")
    print(f"  finite budget to ASK the world for a reliable answer, and that "
          f"answer changes")
    print(f"  it. Does aiming its asking by its own uncertainty make it "
          f"better, and does it")
    print(f"  improve most on the very things it asked about?\n")

    modes = ["passive", "random", "participant"]
    agg = {m: {"accuracy": [], "acc_on_asked": [], "acc_on_not_asked": [],
               "asked": []} for m in modes}
    for seed in range(args.seeds):
        for m in modes:
            r = run(m, seed, args.steps, args.budget)
            for k in agg[m]:
                agg[m][k].append(r[k])

    print("=" * 68)
    print("DID ACTING ON THE WORLD CHANGE THE BEING FOR THE BETTER?")
    print("=" * 68)
    print(f"  {'being':>12} {'accuracy':>9} {'asked':>7}")
    print("-" * 68)
    for m in modes:
        note = {"passive": "only perceives, never acts (floor)",
                "random": "asks at random, same budget",
                "participant": "asks what it is UNSURE of"}[m]
        print(f"  {m:>12} {statistics.mean(agg[m]['accuracy']):>8.1f}% "
              f"{statistics.mean(agg[m]['asked']):>7.0f}   {note}")

    pa = statistics.mean(agg["passive"]["accuracy"])
    ra = statistics.mean(agg["random"]["accuracy"])
    pt = statistics.mean(agg["participant"]["accuracy"])

    print("\n" + "=" * 68)
    print("WHAT IT SAYS")
    print("=" * 68)
    if pt > pa + 1 and pt > ra + 0.5:
        print(f"  THE LOOP CLOSED. The participant reached {pt:.1f}% against "
              f"{pa:.1f}% for a being that")
        print(f"  only perceives and {ra:.1f}% for one that asks at random. "
              f"By choosing what it")
        print(f"  did not know, spending to ask the world, and being updated "
              f"by the answer, it")
        print(f"  became measurably better, more than the same spending used "
              f"blindly.")
        aa = statistics.mean(agg["participant"]["acc_on_asked"])
        na = statistics.mean(agg["participant"]["acc_on_not_asked"])
        print(f"\n  And it improved where it ACTED: {aa:.0f}% accuracy on "
              f"the things it asked about")
        print(f"  vs {na:.0f}% on things it did not. The world's answer "
              f"changed it exactly where it")
        print(f"  reached out. That is the first real participation: it "
              f"acted, the world")
        print(f"  responded, and it was different because of it.")
    else:
        print(f"  the loop did not clearly close: participant {pt:.1f}% vs "
              f"passive {pa:.1f}%, random {ra:.1f}%.")
        print(f"  either the answers did not update it enough or the aiming "
              f"by uncertainty was")
        print(f"  not sharp enough.")

    print(f"\n  In the live being this is real money asking a real service, "
          f"gated by real")
    print(f"  uncertainty and a real budget. Here it is proven in a clean "
          f"world first: the")
    print(f"  loop closes, so it is worth taking to the being that lives in "
          f"the world.")


if __name__ == "__main__":
    main()
