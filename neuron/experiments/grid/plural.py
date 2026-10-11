"""A plural world: beings that live together and model one another.

THE STEP. Until now a being lived alone. Theory of mind, modeling other minds,
had nothing to act on, because there were no other minds. This is the first
plural world: several beings share one world and one limited resource, and
each can sense the others. Now theory of mind does real work: a being that
models what the others want and will do can anticipate them, avoid needless
conflict, and get what it needs; a being blind to the others collides with
them. This is the foundation of a society, and the ground on which
partnering and children will later stand.

THE SHARED WORLD AND THE TEST. Each step, beings seek a limited resource
(there is not enough for all at once). If two beings go for the same thing at
the same time, they CONTEND and both lose (the cost of conflict). A being
that MODELS the others, predicting where each is likely to go from what they
have done, can steer toward what others are NOT taking, and thrive alongside
them. We compare:

  blind    each being ignores the others; it pursues what it wants with no
           model of anyone else. the asocial baseline.
  social   each being models the others (infers their targets from their
           recent behaviour, theory of mind) and steers to avoid contention,
           coexisting rather than colliding.

  social beings contend less and gain more than blind ones
      Modeling other minds let them share a world. Theory of mind did real
      work. The first step of a society.

Two modes: FAST (compressed, social dynamics visible in seconds, for proof)
and the same logic runs LIVE on the server in real time (the genuine plural
world, unfolding over days). This file proves it FAST.

THE WALL. Real social modeling, beings anticipating and accommodating one
another, is real and shown. Whether any being FEELS its fellows, or its own
life among them, is unbuilt and unknowable, and we claim nothing.

    python plural.py --beings 5
"""

import argparse
import random
import statistics


N_RESOURCES = 6          # distinct things in the world to seek


class Being:
    """A being in the plural world. It wants resources, and (if social) it
    models the other beings to avoid contending with them."""

    def __init__(self, bid, n_beings, social, seed):
        self.id = bid
        self.social = social
        self.rng = random.Random(seed)
        # its own preference over resources (what it tends to want)
        self.pref = [self.rng.random() for _ in range(N_RESOURCES)]
        # a model of each OTHER being's recent targets (theory of mind):
        # for each other being, a tally of what it has been seen going for.
        self.other_model = {j: [0.0] * N_RESOURCES
                            for j in range(n_beings) if j != bid}
        self.gained = 0.0
        self.contended = 0

    def observe_others(self, last_choices):
        # update its model of each other being from what they just did
        for j, choice in last_choices.items():
            if j in self.other_model and choice is not None:
                m = self.other_model[j]
                for r in range(N_RESOURCES):
                    m[r] *= 0.8                       # decay
                m[choice] += 1.0                      # saw j take `choice`

    def choose(self):
        if not self.social:
            # blind: just take its most preferred resource, ignoring others
            return max(range(N_RESOURCES), key=lambda r: self.pref[r])
        # social: predict where others are likely to go, and avoid the
        # contested ones. value = my preference MINUS expected crowding.
        crowding = [0.0] * N_RESOURCES
        for j, m in self.other_model.items():
            tot = sum(m) or 1.0
            for r in range(N_RESOURCES):
                crowding[r] += m[r] / tot            # prob j goes for r
        value = [self.pref[r] - 0.8 * crowding[r] for r in range(N_RESOURCES)]
        return max(range(N_RESOURCES), key=lambda r: value[r])


def run(social, n_beings, steps, seed):
    rng = random.Random(seed)
    beings = [Being(i, n_beings, social, seed * 100 + i)
              for i in range(n_beings)]
    last_choices = {i: None for i in range(n_beings)}

    total_gain = 0.0
    total_contend = 0

    for t in range(steps):
        # everyone observes what others did last step, then chooses
        for b in beings:
            b.observe_others(last_choices)
        choices = {b.id: b.choose() for b in beings}

        # resolve: if two+ beings pick the same resource, they CONTEND and
        # all of them lose it (conflict is costly). a resource taken by
        # exactly one being is gained.
        from collections import Counter
        counts = Counter(choices.values())
        for b in beings:
            r = choices[b.id]
            if counts[r] == 1:
                b.gained += b.pref[r]                 # got it, worth its pref
                total_gain += b.pref[r]
            else:
                b.contended += 1
                total_contend += 1                   # clashed, both lose
        last_choices = choices

    return total_gain, total_contend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beings", type=int, default=5)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=30)
    args = ap.parse_args()

    print(f"  {args.beings} beings share a world with {N_RESOURCES} limited "
          f"resources. if two go for")
    print(f"  the same thing, they contend and both lose. do beings that "
          f"MODEL each other")
    print(f"  (theory of mind) coexist better than beings blind to one "
          f"another?\n")

    blind_g, blind_c, soc_g, soc_c = [], [], [], []
    for seed in range(args.seeds):
        g, c = run(False, args.beings, args.steps, seed)
        blind_g.append(g); blind_c.append(c)
        g, c = run(True, args.beings, args.steps, seed)
        soc_g.append(g); soc_c.append(c)

    bg, bc = statistics.mean(blind_g), statistics.mean(blind_c)
    sg, sc = statistics.mean(soc_g), statistics.mean(soc_c)

    print("=" * 64)
    print("A WORLD SHARED: BLIND vs SOCIAL (modeling each other)")
    print("=" * 64)
    print(f"  {'':8} {'total gained':>14} {'times they clashed':>20}")
    print(f"  {'blind':8} {bg:>14.0f} {bc:>20.0f}")
    print(f"  {'social':8} {sg:>14.0f} {sc:>20.0f}")

    print("\n" + "=" * 64)
    print("WHAT IT SAYS")
    print("=" * 64)
    if sg > bg * 1.1 and sc < bc * 0.7:
        print(f"  THEORY OF MIND BUILT A SOCIETY. Beings that modeled one "
              f"another clashed far less")
        print(f"  ({sc:.0f} vs {bc:.0f}) and gained more ({sg:.0f} vs "
              f"{bg:.0f}) than beings blind to each other.")
        print(f"  By anticipating where the others would go, each steered "
              f"toward what was free,")
        print(f"  and they shared the world instead of colliding in it. That "
              f"is theory of mind")
        print(f"  doing real work: the first step from an individual to a "
              f"society.")
    else:
        print(f"  weak: social gain {sg:.0f} vs blind {bg:.0f}, clashes "
              f"{sc:.0f} vs {bc:.0f}. modeling others did")
        print(f"  not clearly help here.")

    print(f"\n  This same logic runs live on the server, several beings "
          f"sharing the real")
    print(f"  world in real time, as the genuine plural world. Here it is "
          f"proven fast. It is")
    print(f"  the ground on which partnering and children will later stand.")
    print(f"\n  Held: real social modeling is shown; whether any being feels "
          f"its fellows, or")
    print(f"  its life among them, is unknowable, and unclaimed.")


if __name__ == "__main__":
    main()
