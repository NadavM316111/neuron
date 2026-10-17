"""Does a lineage get better at LEARNING, not just inherit answers?

THE HONEST, NON-TRIVIAL TEST. The previous result (inherit the answer to a
fixed task, learn it cheaply) was partly trivial: of course you are fast if
you are born already knowing the answer. The real, valuable question is
different and much harder:

  Does a lineage get better at learning tasks it has NEVER SEEN?

Here every generation faces a DIFFERENT, novel task. The old answer is
useless, each task is new, so inheriting a solved task cannot help. The only
thing that can make a later generation learn its new task faster is genuinely
better LEARNING MACHINERY: structure that is learning-ready, not answer-loaded.
This is "learning to learn" (meta-learning), and it is the real prize, because
it is exactly what the brain does: evolution shaped structure that makes any
individual a fast learner of whatever its life throws at it, not a container
for pre-solved problems.

THE DESIGN.
  - Each generation gets a brand-new random task it has never seen.
  - It inherits its parent's structure (weights), plus mutation.
  - SELECTION: a few children are born each generation; the one that learns
    the NEW task with the least compute is the one who reproduces. So the
    lineage is selected for LEARNING ABILITY on novel tasks, not for any
    particular answer.
  - We measure the compute each generation needs to learn its novel task.

  compute-to-learn-a-NOVEL-task falls across generations
      The lineage evolved better learning machinery itself. It got better at
      learning things it had never seen, the brain's real trick, and NEURON's
      own. A genuine, non-trivial efficiency.
  it does not fall
      The earlier gain was just answer-reuse; inherited structure does not
      make the lineage a better learner of new things. Honest negative,
      reported plainly.

    python learning_to_learn.py --generations 20
"""

import argparse
import random
import math


IN = 10
HID = 20


def make_task(seed):
    """A fresh random task. Every generation gets a new seed -> a new task it
    has never seen."""
    rng = random.Random(seed)
    k = 6
    w = [rng.uniform(-1.5, 1.5) for _ in range(k)]
    data = []
    for _ in range(160):
        x = [rng.uniform(-1, 1) for _ in range(IN)]
        t = sum(w[i] * math.tanh(x[i] * x[(i + 1) % k]) for i in range(k))
        data.append((x, 1 if t > 0 else 0))
    return data


class Mind:
    def __init__(self, rng, inherited=None, mutate=0.08):
        self.rng = rng
        self.compute = 0
        if inherited is not None:
            self.W1 = [[w + rng.gauss(0, mutate) for w in row]
                       for row in inherited["W1"]]
            self.b1 = [b + rng.gauss(0, mutate) for b in inherited["b1"]]
            self.W2 = [w + rng.gauss(0, mutate) for w in inherited["W2"]]
            self.b2 = inherited["b2"] + rng.gauss(0, mutate)
            # it also inherits its learning rate, which can itself evolve:
            # the lineage can discover a better rate for learning fast.
            self.lr = max(0.005, min(0.3,
                          inherited["lr"] + rng.gauss(0, 0.01)))
        else:
            self.W1 = [[rng.uniform(-0.5, 0.5) for _ in range(IN)]
                       for _ in range(HID)]
            self.b1 = [0.0] * HID
            self.W2 = [rng.uniform(-0.5, 0.5) for _ in range(HID)]
            self.b2 = 0.0
            self.lr = 0.05

    def step(self, x, t, train=True):
        self.compute += HID * IN
        h = [math.tanh(self.b1[j] + sum(self.W1[j][i] * x[i]
                                        for i in range(IN)))
             for j in range(HID)]
        o = self.b2 + sum(self.W2[j] * h[j] for j in range(HID))
        p = 1 / (1 + math.exp(-max(-30, min(30, o))))
        if train:
            err = p - t
            for j in range(HID):
                self.W2[j] -= self.lr * err * h[j]
            self.b2 -= self.lr * err
            for j in range(HID):
                d = err * self.W2[j] * (1 - h[j] * h[j])
                for i in range(IN):
                    self.W1[j][i] -= self.lr * d * x[i]
                self.b1[j] -= self.lr * d
        return (p > 0.5) == (t == 1)

    def acc(self, task):
        return sum(self.step(x, t, train=False) for (x, t) in task) / len(task)

    def learn_to(self, task, target, max_epochs):
        for e in range(max_epochs):
            for (x, t) in task:
                self.step(x, t, train=True)
            if self.acc(task) >= target:
                return True
        return self.acc(task) >= target

    def snapshot(self):
        return {"W1": [r[:] for r in self.W1], "b1": self.b1[:],
                "W2": self.W2[:], "b2": self.b2, "lr": self.lr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=20)
    ap.add_argument("--target", type=float, default=0.80)
    ap.add_argument("--children", type=int, default=4)
    ap.add_argument("--lineages", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("  Every generation faces a BRAND-NEW task it has never seen. It "
          "inherits its")
    print("  parent's structure. Among a few children, the one that learns "
          "the new task")
    print("  cheapest reproduces. Does the lineage get better at learning "
          "NOVEL tasks over")
    print("  time, the real 'learning to learn', not just answer-reuse?\n")

    per_gen = [[] for _ in range(args.generations)]
    for L in range(args.lineages):
        inherited = None
        for g in range(args.generations):
            # a brand-new task this generation (never seen by the lineage)
            task = make_task(args.seed * 99991 + L * 1000 + g * 7 + 1)
            # a few children; each learns the NEW task; cheapest learner wins
            best_child, best_compute = None, None
            for c in range(args.children):
                rng = random.Random(args.seed * 7 + L * 300 + g * 11 + c)
                child = Mind(rng, inherited=inherited)
                child.learn_to(task, args.target, 20)
                if best_compute is None or child.compute < best_compute:
                    best_compute = child.compute
                    best_child = child
            per_gen[g].append(best_compute)
            inherited = best_child.snapshot()   # the best learner reproduces

    import statistics
    print("=" * 60)
    print("COMPUTE TO LEARN A NOVEL TASK, BY GENERATION")
    print("=" * 60)
    gen_means = [statistics.mean(per_gen[g]) for g in range(args.generations)]
    base = gen_means[0]
    for g in range(args.generations):
        m = gen_means[g]
        bar = "#" * int(m / max(base, 1) * 34)
        print(f"  gen {g:>2}: {m:>14,.0f}   {bar}")

    first = statistics.mean(gen_means[:3])
    last = statistics.mean(gen_means[-3:])
    print("\n" + "=" * 60)
    print("WHAT IT SAYS")
    print("=" * 60)
    if last < first * 0.7:
        drop = 100 * (1 - last / first)
        print(f"  THE LINEAGE LEARNED TO LEARN. Early generations needed "
              f"{first:,.0f} compute to")
        print(f"  learn a novel task; later ones, after the lineage evolved "
              f"better learning")
        print(f"  machinery, needed {last:,.0f}, a {drop:.0f}% drop, on tasks "
              f"they had NEVER seen.")
        print(f"  This is not answer-reuse (every task was new); it is the "
              f"lineage becoming a")
        print(f"  genuinely better learner. The brain's real trick, "
              f"demonstrated, and entirely")
        print(f"  the lineage's own, no borrowing.")
    elif last < first * 0.88:
        drop = 100 * (1 - last / first)
        print(f"  A REAL BUT MODEST IMPROVEMENT ({drop:.0f}%). The lineage got "
              f"somewhat better at")
        print(f"  learning novel tasks, the effect is real here, if small.")
    else:
        print(f"  NO CLEAR IMPROVEMENT ({first:,.0f} -> {last:,.0f}). The "
              f"lineage did NOT get better at")
        print(f"  learning new tasks. The earlier gain was answer-reuse, not "
              f"better learning;")
        print(f"  it does not generalize to novel tasks. Honest negative, the "
              f"one that matters.")

    print(f"\n  Honest scope: toy tasks, small nets. But this isolates the "
          f"real question,")
    print(f"  whether a lineage evolves better LEARNING ABILITY on things it "
          f"has never seen,")
    print(f"  from the trivial one (reusing a known answer). This is the "
          f"version that would")
    print(f"  genuinely matter for efficient intelligence.")


if __name__ == "__main__":
    main()
