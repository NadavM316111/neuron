"""The seed of abstract thought: forming concepts, and combining them.

THE FACULTY, AND ITS HONEST LIMIT. A mind does not just memorise specific
experiences. It carves the continuous mess of perception into discrete
CONCEPTS, warm, bright, heavy, nobody labels them, it forms them itself, and
then it COMBINES them to think about things it never actually met: a
warm-bright-heavy thing it has never seen, understood by composing concepts
it has. That composition is the root of abstract thought, and it is what
lets a mind generalise beyond its experience.

The honest limit, stated up front: full symbolic reasoning likely needs more
than this toy and more scale than is on hand. But the SEED, forming one's own
symbols from raw experience and composing them to grasp the never-seen, is
real and buildable, and that is what this demonstrates.

THE TEST, built so compositionality cannot be faked by memorising. The world
has things described by several FEATURES (each feature takes a few values).
Some feature-combinations the being experiences; a set of combinations is
HELD OUT, it never sees them. Each thing has an outcome determined
COMPOSITIONALLY by its features (e.g. good if warm AND light). The question:
on the held-out combinations it never experienced, can the being still judge
them correctly, by having formed concepts for the individual features and
composing them?

  memoriser     learns each specific thing it meets as a whole. on a novel
                combination it has no memory of, it is lost. the floor:
                experience without concepts.
  concept-former forms a concept for each feature-value from across all the
                things that share it (so 'warm' is learned from every warm
                thing, whatever else it was), and judges a new thing by
                COMPOSING the relevant concepts. it can judge a combination
                it never saw, because it has the parts.

  concept-former handles held-out combinations, memoriser cannot
      It formed its own symbols and composed them to grasp the never-seen.
      The seed of abstract thought.
  no difference
      The concepts did not factor cleanly, or the outcome was not really
      compositional.

    python symbols.py
    python symbols.py --features 4 --values 4
"""

import argparse
import random
import statistics


def make_world(n_features, n_values, seed):
    """Each feature has a hidden 'goodness' per value; a thing's outcome is
    good if the SUM of its features' goodness crosses a threshold. This is
    compositional: the outcome is built from the parts."""
    rng = random.Random(seed)
    goodness = [[rng.uniform(-1, 1) for _ in range(n_values)]
                for _ in range(n_features)]
    return goodness


def outcome(thing, goodness):
    s = sum(goodness[f][v] for f, v in enumerate(thing))
    return 1 if s > 0 else 0


def all_things(n_features, n_values):
    things = [[]]
    for f in range(n_features):
        things = [t + [v] for t in things for v in range(n_values)]
    return [tuple(t) for t in things]


def run(mode, n_features, n_values, seed):
    rng = random.Random(seed)
    goodness = make_world(n_features, n_values, seed)
    things = all_things(n_features, n_values)
    rng.shuffle(things)

    # hold out ~30% of specific combinations: the being NEVER experiences
    # these. the rest it can learn from.
    cut = int(len(things) * 0.7)
    seen_things = things[:cut]
    held_out = things[cut:]

    if mode == "memoriser":
        # learns each specific seen thing's outcome as a whole
        memory = {}
        for thing in seen_things:
            memory[thing] = outcome(thing, goodness)
        # a TRUE memoriser: it knows only the exact things it saw. a
        # never-seen combination has no entry, so it can only fall back to
        # the base rate it observed. no concepts, no composition, no
        # similarity cheat -- experience without abstraction.
        base = sum(memory.values()) / len(memory)
        base_guess = 1 if base > 0.5 else 0
        correct = 0
        for thing in held_out:
            if thing in memory:
                guess = memory[thing]
            else:
                guess = base_guess          # it genuinely does not know
            if guess == outcome(thing, goodness):
                correct += 1
        return 100.0 * correct / len(held_out)

    else:  # concept-former: form a concept per feature-value, then compose
        # a CONCEPT for (feature f, value v) = the average outcome across all
        # SEEN things that had that feature-value. this abstracts the part
        # from the wholes. nobody labels the concepts; they emerge from
        # co-occurrence with outcomes.
        concept_sum = [[0.0] * n_values for _ in range(n_features)]
        concept_cnt = [[0] * n_values for _ in range(n_features)]
        for thing in seen_things:
            o = outcome(thing, goodness)
            for f, v in enumerate(thing):
                # center the outcome so a concept captures whether this
                # feature-value pushes toward good (+) or bad (-)
                concept_sum[f][v] += (o - 0.5)
                concept_cnt[f][v] += 1
        concept = [[(concept_sum[f][v] / concept_cnt[f][v]
                     if concept_cnt[f][v] else 0.0)
                    for v in range(n_values)] for f in range(n_features)]
        # judge a NOVEL thing by COMPOSING its features' concepts
        correct = 0
        for thing in held_out:
            score = sum(concept[f][v] for f, v in enumerate(thing))
            guess = 1 if score > 0 else 0
            if guess == outcome(thing, goodness):
                correct += 1
        return 100.0 * correct / len(held_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=int, default=4)
    ap.add_argument("--values", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=8)
    args = ap.parse_args()

    total = args.values ** args.features
    print(f"  things described by {args.features} features, each with "
          f"{args.values} values ({total} possible things).")
    print(f"  outcomes are COMPOSITIONAL (built from the features). the being "
          f"sees 70% of the")
    print(f"  combinations and is tested on the 30% it NEVER saw. can it "
          f"judge the never-seen")
    print(f"  by forming concepts for the parts and composing them?\n")

    mem, con = [], []
    for seed in range(args.seeds):
        mem.append(run("memoriser", args.features, args.values, seed))
        con.append(run("concept-former", args.features, args.values, seed))

    print("=" * 64)
    print("CAN IT GRASP THE NEVER-SEEN BY COMPOSING CONCEPTS?")
    print("=" * 64)
    m, c = statistics.mean(mem), statistics.mean(con)
    print(f"  {'being':>16} {'accuracy on combinations it NEVER saw':>40}")
    print("-" * 64)
    print(f"  {'memoriser':>16} {m:>38.0f}%   (experience, no concepts)")
    print(f"  {'concept-former':>16} {c:>38.0f}%   (forms & composes its own "
          f"symbols)")

    print("\n" + "=" * 64)
    print("WHAT IT SAYS")
    print("=" * 64)
    if c > m + 8:
        print(f"  IT FORMED SYMBOLS AND COMPOSED THEM. On combinations it had "
              f"NEVER experienced,")
        print(f"  the concept-former judged {c:.0f}% correctly against the "
              f"memoriser's {m:.0f}%. By")
        print(f"  carving its experience into a concept for each feature, "
              f"learned across every")
        print(f"  thing that shared it, and composing those concepts for a "
              f"new thing, it")
        print(f"  reasoned about what it had never met. That is the seed of "
              f"abstract thought:")
        print(f"  from specific experiences to reusable symbols to the "
              f"never-seen.")
    else:
        print(f"  composition did not clearly help: {c:.0f}% vs {m:.0f}%. the "
              f"features may not factor")
        print(f"  cleanly, or the outcome was not compositional enough to "
              f"reward concepts.")

    print(f"\n  The concepts are the being's own, formed from co-occurrence "
          f"with outcomes, named")
    print(f"  by nobody. Composing them to grasp the never-seen is the root "
          f"of generalising")
    print(f"  beyond experience. Honest limit: this is the SEED at toy scale; "
          f"full symbolic")
    print(f"  reasoning likely needs more than this and more scale. But the "
          f"seed is real.")


if __name__ == "__main__":
    main()
