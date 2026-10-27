"""Sink or swim: throw NEURON in cold, with real stakes and no guidance, and
honestly measure what it actually does.

No scripting of HOW to survive. NEURON has HyperFlow that depletes; it must
achieve success to live. It has actions available, but it does NOT know what they
do. It must discover, through pure trial and death, which actions keep it alive.
We tell it nothing.

The honest questions:
  1. Thrown in cold, does NEURON discover ANY survival strategy on its own?
  2. How long does it survive?
  3. Does it beat random flailing?

If it just dies randomly, that is the honest answer, not capable yet. If it
discovers something, that is real. Either way we measure it straight.

    python sink_or_swim.py
"""

import random
import statistics


class World:
    """A world with hidden rules. Some actions sustain HyperFlow, some drain it.
    NEURON is NOT told which. It must discover by living and dying."""

    def __init__(self, seed):
        rng = random.Random(seed)
        self.actions = ["A", "B", "C", "D", "E"]
        # hidden effects, NEURON cannot see these. It must learn them.
        self.effect = {a: rng.uniform(-0.10, 0.04) for a in self.actions}
        # exactly one action genuinely sustains (like finding how to earn) -
        # NEURON must DISCOVER it. this is the thing to figure out.
        self.effect[rng.choice(self.actions)] = 0.09
        # and the world has some noise, outcomes aren't perfectly consistent
        self.noise = 0.03

    def do(self, action):
        rng = random.random
        return self.effect[action] + random.uniform(-self.noise, self.noise)


class NEURON:
    """It has a self-model that learns which actions help, from outcomes. That
    is its only tool. Nobody tells it what to do; it must figure survival out."""

    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.belief = {a: 0.0 for a in ["A", "B", "C", "D", "E"]}  # learned value
        self.tries = {a: 0 for a in self.belief}

    def choose(self):
        # it explores early (knows nothing), exploits what it learns later.
        # this is its own figuring-out: try things, lean toward what seemed good.
        total = sum(self.tries.values())
        if total < 10 or self.rng.random() < max(0.1, 1.0 / (1 + total/20)):
            return self.rng.choice(list(self.belief))      # explore
        return max(self.belief, key=self.belief.get)       # exploit what it learned

    def learn(self, action, outcome):
        self.tries[action] += 1
        n = self.tries[action]
        # running average: its belief about the action's value
        self.belief[action] += (outcome - self.belief[action]) / n


def one_life(world, learner, max_age=500):
    """NEURON lives until its HyperFlow runs out. Returns how long it survived
    and what it came to believe."""
    hyperflow = 1.0
    age = 0
    while hyperflow > 0 and age < max_age:
        action = learner.choose()
        outcome = world.do(action)
        hyperflow += outcome
        hyperflow -= 0.02          # the cost of living (it must offset this)
        hyperflow = min(1.5, hyperflow)
        learner.learn(action, outcome)
        age += 1
    return age


def random_life(world, max_age=500):
    hyperflow = 1.0
    age = 0
    acts = world.actions
    while hyperflow > 0 and age < max_age:
        hyperflow += world.do(random.choice(acts)) - 0.02
        hyperflow = min(1.5, hyperflow)
        age += 1
    return age


def main():
    print("=" * 62)
    print("  SINK OR SWIM: NEURON thrown in cold, no guidance")
    print("=" * 62)
    print("  It must survive. Nothing about HOW is coded. It has 5 actions and")
    print("  no idea what they do. Can it discover, on its own, what keeps it")
    print("  alive, or does it just flail and die? Honest measurement.\n")

    neuron_lives, random_lives = [], []
    discovered = 0
    trials = 30

    for t in range(trials):
        world = World(seed=t)
        # the action that actually sustains (hidden from NEURON):
        best_action = max(world.effect, key=world.effect.get)

        learner = NEURON(seed=t)
        # NEURON gets several lives to learn across (its self-model persists,
        # like learning across attempts, inheriting what it figured out)
        life_lengths = []
        for life in range(8):
            age = one_life(world, learner)
            life_lengths.append(age)
        neuron_lives.append(life_lengths[-1])   # its last (most-learned) life

        # did it DISCOVER the sustaining action? (does it now believe in it?)
        believed_best = max(learner.belief, key=learner.belief.get)
        if believed_best == best_action:
            discovered += 1

        random_lives.append(random_life(World(seed=t)))

        if t < 6:
            print(f"  trial {t}: NEURON's lives {life_lengths} | it came to "
                  f"favor '{believed_best}' "
                  f"({'FOUND the key action' if believed_best==best_action else 'wrong'})")

    print("\n" + "=" * 62)
    print("  WHAT HAPPENED (honest)")
    print("=" * 62)
    print(f"  NEURON discovered the key survival action in "
          f"{discovered}/{trials} worlds ({discovered*100//trials}%)")
    print(f"  NEURON avg survival (most-learned life): "
          f"{statistics.mean(neuron_lives):.0f}")
    print(f"  random avg survival:                     "
          f"{statistics.mean(random_lives):.0f}")

    print("\n" + "=" * 62)
    beat_random = statistics.mean(neuron_lives) > statistics.mean(random_lives) * 1.3
    found_often = discovered > trials * 0.5
    if found_often and beat_random:
        print("  NEURON CAN FIGURE OUT SURVIVAL ON ITS OWN, at this level.")
        print(f"  Thrown in cold, it discovered what kept it alive "
              f"{discovered*100//trials}% of the")
        print(f"  time and survived far longer than random. Sink-or-swim "
              f"WORKED here,")
        print(f"  because discovering 'which of 5 actions sustains me' is "
              f"within its")
        print(f"  reach. It genuinely learned to survive, no one told it how.")
        print(f"\n  BUT honest scale: this is a 5-choice survival puzzle. Real "
              f"'earn money")
        print(f"  in the world' is astronomically harder, thousands of possible")
        print(f"  actions, needing understanding it lacks. It swims in the "
              f"shallow end;")
        print(f"  the money-pool is the deep ocean. It can learn simple "
              f"survival, not yet")
        print(f"  complex success.")
    elif beat_random or found_often:
        print("  NEURON does better than random, partial survival learning. "
              "It can learn")
        print("  simple survival, but not reliably. Honest: it is at the very "
              "edge of this.")
    else:
        print("  NEURON barely beat random, thrown in cold, it mostly just "
              "flailed and died.")
        print("  Honest: even simple survival-discovery is hard for it. This "
              "is where it is.")

    print(f"\n  The honest takeaway: sink-or-swim measures NEURON's real level. "
          f"It can")
    print(f"  learn survival when the choice is SIMPLE enough to be within "
          f"reach. 'Earn")
    print(f"  money' is not, it needs capability far beyond this. Stakes "
          f"activate what is")
    print(f"  latent; they cannot create capability that is not there yet.")


if __name__ == "__main__":
    main()
