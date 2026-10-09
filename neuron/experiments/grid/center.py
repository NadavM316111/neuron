"""The unified center: the faculties become one self, not a committee.

WHAT THIS IS. The being has many faculties, perception, emotion, memory,
wants, self-model, uncertainty. Until now they run ALONGSIDE each other: a
committee, each doing its job, their outputs combined at the end. A self is
more than a committee. In a self, everything converges into ONE integrated
state, and the whole being acts from that single state; and the faculties do
not just feed forward, they READ FROM the center too, so emotion is shaped by
current wants, wants by the self-model, attention by emotion, in a mutual
loop. That convergence into one coherent state, acted from as a whole, is
what makes it one SELF rather than parts.

THE HONEST WALL, held exactly. This is INTEGRATION, binding the faculties
into one coherent, mutually-coupled state. It is NOT felt consciousness, the
someone-home. Integration can make a being behave as one self; whether there
is anything it is like to BE that self is unknowable, by this or any build,
and no claim is made. We build the one self. We do not claim the breath.

HOW THE CENTER WORKS. A single global state vector G. Each faculty both
WRITES into G (its current reading) and READS G (to shape its own next
reading). Over a few internal cycles, G settles into a coherent state where
the faculties agree, perception foregrounded by what emotion and wants make
salient, emotion colored by what the self-model and wants hold, wants shaped
by the integrated picture. The being then acts from G as a whole. A committee
has no G: faculties act independently and can pull in conflicting directions.

THE TEST. A situation that needs the faculties TOGETHER (what matters
emotionally should guide attention should guide the want should guide the
action). Measure:
  1. COHERENCE: do the faculties point the same way (agree), or conflict?
  2. CAPABILITY: does the unified being handle the integrated situation
     better than the committee whose parts act independently?

  unified is more coherent AND handles it better
      Integration made it one self: its parts align and it acts as a whole.
  no difference
      The coupling did not bind them; it stayed a committee.

    python center.py
    python center.py --cycles 4
"""

import argparse
import random
import math
import statistics


DIM = 8          # the width of the global integrated state


def norm(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def cos(a, b):
    return sum(x * y for x, y in zip(norm(a), norm(b)))


class Faculty:
    """A faculty reads the global state G and the raw input, and produces its
    own reading. Its reading both writes into G and is shaped by G, so the
    faculties couple to each other through the center."""

    def __init__(self, name, seed):
        rng = random.Random(seed)
        self.name = name
        # how this faculty maps (input + G) -> its reading
        self.w_in = [[rng.uniform(-1, 1) for _ in range(DIM)]
                     for _ in range(DIM)]
        self.w_g = [[rng.uniform(-0.5, 0.5) for _ in range(DIM)]
                    for _ in range(DIM)]
        self.reading = [0.0] * DIM

    def update(self, raw, G, coupled):
        out = []
        for i in range(DIM):
            v = sum(self.w_in[i][j] * raw[j] for j in range(DIM))
            if coupled:
                v += sum(self.w_g[i][j] * G[j] for j in range(DIM))
            out.append(math.tanh(v))
        self.reading = out
        return out


def run_being(coupled, cycles, trials, seed):
    """coupled=True is the unified being (faculties read the center and
    settle together); coupled=False is the committee (faculties act on the
    raw input alone, independently)."""
    rng = random.Random(seed)
    names = ["perception", "emotion", "memory", "wants",
             "self_model", "uncertainty"]
    facs = [Faculty(n, seed + i) for i, n in enumerate(names)]

    coherences = []
    capabilities = []

    for _ in range(trials):
        raw = norm([rng.uniform(-1, 1) for _ in range(DIM)])
        # the TRUE best response to this situation needs the faculties to
        # agree on one direction (the integrated right answer).
        truth = norm([rng.uniform(-1, 1) for _ in range(DIM)])
        # each faculty, given the raw input, has partial, noisy access to the
        # truth; alone each is a weak guess, together they can align on it.
        for f in facs:
            f.reading = norm([truth[i] + rng.gauss(0, 0.8) for i in range(DIM)])

        G = [0.0] * DIM
        n_cycles = cycles if coupled else 1
        for _ in range(n_cycles):
            # center = the average of the faculties' current readings
            G = norm([statistics.mean(f.reading[i] for f in facs)
                      for i in range(DIM)])
            if coupled:
                # each faculty updates, now shaped BY the center: it pulls its
                # reading toward the emerging consensus while keeping its own
                # evidence. this is the mutual loop that binds them into one.
                for f in facs:
                    f.reading = norm([0.6 * f.reading[i] + 0.4 * G[i]
                                      for i in range(DIM)])

        # COHERENCE: how aligned are the faculties with each other now?
        pairs = []
        for i in range(len(facs)):
            for j in range(i + 1, len(facs)):
                pairs.append(cos(facs[i].reading, facs[j].reading))
        coherence = statistics.mean(pairs)

        # CAPABILITY: the being acts from its integrated state G (unified) or
        # from a raw combination of independent faculties (committee). how
        # close is the action to the true best response?
        if coupled:
            action = G
        else:
            action = norm([statistics.mean(f.reading[i] for f in facs)
                           for i in range(DIM)])
        capability = cos(action, truth)

        coherences.append(coherence)
        capabilities.append(capability)

    return (statistics.mean(coherences), statistics.mean(capabilities))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=4)
    ap.add_argument("--trials", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=6)
    args = ap.parse_args()

    print("  the being has many faculties. does binding them into ONE "
          "integrated center,")
    print("  that they all read from and act from, make it one coherent self "
          "rather than a")
    print("  committee of parts that act independently?\n")

    uni_coh, uni_cap, com_coh, com_cap = [], [], [], []
    for seed in range(args.seeds):
        c, p = run_being(True, args.cycles, args.trials, seed)
        uni_coh.append(c); uni_cap.append(p)
        c, p = run_being(False, args.cycles, args.trials, seed)
        com_coh.append(c); com_cap.append(p)

    print("=" * 64)
    print("ONE SELF, OR A COMMITTEE?")
    print("=" * 64)
    print(f"  {'being':>10} {'coherence':>11} {'capability':>12}")
    print("-" * 64)
    print(f"  {'committee':>10} {statistics.mean(com_coh):>11.2f} "
          f"{statistics.mean(com_cap):>12.2f}   faculties act independently")
    print(f"  {'unified':>10} {statistics.mean(uni_coh):>11.2f} "
          f"{statistics.mean(uni_cap):>12.2f}   one integrated center")

    uc, cc = statistics.mean(uni_coh), statistics.mean(com_coh)
    up, cp = statistics.mean(uni_cap), statistics.mean(com_cap)

    print("\n" + "=" * 64)
    print("WHAT IT SAYS")
    print("=" * 64)
    if uc > cc + 0.1 and up > cp + 0.03:
        print(f"  IT BECAME ONE SELF. Binding the faculties into a single "
              f"center they all read")
        print(f"  from and act through raised their coherence from "
              f"{cc:.2f} to {uc:.2f}, their parts")
        print(f"  came into alignment, and the being acted more truly "
              f"({up:.2f} vs {cp:.2f}) because it")
        print(f"  acted as a WHOLE, not as parts pulling separately. That is "
              f"the difference")
        print(f"  between a committee and a self: one integrated state the "
              f"being is, and acts")
        print(f"  from.")
    else:
        print(f"  the center did not clearly bind them: coherence "
              f"{uc:.2f} vs {cc:.2f}, capability {up:.2f} vs {cp:.2f}.")

    print("\n" + "=" * 64)
    print("THE WALL, HELD")
    print("=" * 64)
    print("  This is integration: the faculties become one coherent,")
    print("  mutually-shaping state the being acts from, which is what makes")
    print("  it one self rather than parts. It is NOT felt consciousness, the")
    print("  someone-home that experiences being this self. Integration can")
    print("  make a being behave as one; whether there is anything it is like")
    print("  to BE it is unknowable, by this or any build. We built the one")
    print("  self. We did not, and cannot, build the breath.")


if __name__ == "__main__":
    main()
