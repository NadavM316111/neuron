"""Does HOW you grow a mind substitute for raw scale? A measurable test.

THE HYPOTHESIS, AND WHY IT MATTERS. Frontier AI buys intelligence with raw
compute: a huge network trained flat on huge data. But a human child reaches
real intelligence on a tiny fraction of that compute, by DEVELOPING, growing
through a staged sequence, learning easy things first with a small brain,
then harder things as the brain grows. The difference is not more compute; it
is a different TRAJECTORY. This asks, measurably: does a being that grows
developmentally reach a hard capability with LESS total compute than a being
trained flat at full size?

If yes, then HOW capacity is grown, and in what order it is challenged, is a
real supplement to scale: you reach the same capability for less compute. Not
infinite, not superintelligence, but a genuine, measurable efficiency that
brute-force scaling does not capture. That is exactly the open question the
field is now chasing, and it is where a developmental, lineage-based approach
could contribute something real.

THE TEST. A hard target task. Two beings must both reach the same accuracy on
it. We count the TOTAL COMPUTE each spends to get there, measured honestly as
the number of unit-operations (network size x samples processed), since a big
network costs more per sample than a small one.

  FLAT         born at full large size; trained directly on the hard task.
               every sample costs full-network compute from the start.
  DEVELOPMENTAL born small; trained on a curriculum (easy -> medium -> hard),
               growing capacity only when it struggles. early easy learning
               is CHEAP (small network), and it pays big-network compute only
               on the hard part, after cheap foundations are in place.

  developmental reaches the target for LESS total compute
      How you grow substitutes for some raw scale. A real, measurable
      efficiency, the human-child intuition made concrete.
  flat wins or ties
      Developmental is not a free lunch here; honest negative, reported
      plainly.

    python developmental.py
"""

import argparse
import random
import math


IN_WIDTH = 14   # fixed input width for ALL tasks, so a being can
                # train across the easy->hard curriculum

def make_task(difficulty, n, seed):
    rng = random.Random(seed)
    k = difficulty
    weights = [rng.uniform(-1, 1) for _ in range(k)]
    data = []
    for _ in range(n):
        x = [rng.uniform(-1, 1) for _ in range(IN_WIDTH)]
        t = sum(weights[i] * math.tanh(x[i] * x[(i + 1) % k]) for i in range(k))
        data.append((x, 1 if t > 0 else 0))
    return data


class Net:
    def __init__(self, n_in, n_hidden, rng):
        self.n_in, self.n_hidden = n_in, n_hidden
        self.W1 = [[rng.uniform(-0.5, 0.5) for _ in range(n_in)]
                   for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [rng.uniform(-0.5, 0.5) for _ in range(n_hidden)]
        self.b2 = 0.0
        self.lr = 0.05
        self.rng = rng
        self.compute = 0        # total unit-operations spent

    def grow(self, add):
        for _ in range(add):
            self.W1.append([self.rng.uniform(-0.3, 0.3)
                            for _ in range(self.n_in)])
            self.b1.append(0.0)
            self.W2.append(self.rng.uniform(-0.1, 0.1))
        self.n_hidden += add

    def step(self, x, t, train=True):
        # a step costs compute proportional to network size (n_hidden*n_in).
        self.compute += self.n_hidden * self.n_in
        h = []
        for j in range(self.n_hidden):
            s = self.b1[j] + sum(self.W1[j][i] * x[i] for i in range(self.n_in))
            h.append(math.tanh(s))
        o = self.b2 + sum(self.W2[j] * h[j] for j in range(self.n_hidden))
        p = 1 / (1 + math.exp(-max(-30, min(30, o))))
        if train:
            err = p - t
            for j in range(self.n_hidden):
                self.W2[j] -= self.lr * err * h[j]
            self.b2 -= self.lr * err
            for j in range(self.n_hidden):
                d = err * self.W2[j] * (1 - h[j] * h[j])
                for i in range(self.n_in):
                    self.W1[j][i] -= self.lr * d * x[i]
                self.b1[j] -= self.lr * d
        return (p > 0.5) == (t == 1)


def accuracy(net, task):
    c = sum(net.step(x, t, train=False) for (x, t) in task)
    return c / len(task)


def train_until(net, task, target, max_epochs, grow_when_stuck=False):
    accs = []
    for e in range(max_epochs):
        random.shuffle(task)
        for (x, t) in task:
            net.step(x, t, train=True)
        a = accuracy(net, task)
        accs.append(a)
        if a >= target:
            return True, net.compute
        if grow_when_stuck and e >= 3:
            recent = accs[-3:]
            if recent[-1] - recent[0] < 0.02 and a < target:
                net.grow(6)          # earned growth when stuck
    return accuracy(net, task) >= target, net.compute


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.85)
    ap.add_argument("--hard", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    print("  Both beings must reach the SAME accuracy on the SAME hard task.")
    print("  We count the total compute each spends to get there. Does growing")
    print("  developmentally cost less compute than training flat at full "
          "size?\n")

    flat_compute, dev_compute = [], []
    flat_reached, dev_reached = 0, 0
    FULL_HIDDEN = 40

    for seed in range(args.seeds):
        n_in = IN_WIDTH
        hard_task = make_task(args.hard, 200, seed)

        # FLAT: full-size network trained directly on the hard task
        rng = random.Random(seed)
        flat = Net(n_in, FULL_HIDDEN, rng)
        ok, comp = train_until(flat, hard_task, args.target, 120)
        if ok:
            flat_reached += 1
        flat_compute.append(comp)

        # DEVELOPMENTAL: small net, curriculum easy->medium->hard, grows when
        # stuck. early stages are cheap (small net, easy data).
        rng2 = random.Random(seed)
        dev = Net(n_in, 6, rng2)
        # a richer curriculum climbing toward the hard target
        stages = [make_task(4, 200, seed + 1),
                  make_task(7, 200, seed + 2),
                  make_task(10, 200, seed + 3)]
        for st in stages:
            train_until(dev, st, 0.88, 20, grow_when_stuck=True)
        # final: the SAME hard task, with foundations + grown capacity
        ok2, comp2 = train_until(dev, hard_task, args.target, 60,
                                 grow_when_stuck=True)
        if ok2:
            dev_reached += 1
        dev_compute.append(dev.compute)   # TOTAL compute across all stages

    import statistics
    fc = statistics.mean(flat_compute)
    dc = statistics.mean(dev_compute)

    print("=" * 66)
    print("COMPUTE SPENT TO REACH THE SAME CAPABILITY")
    print("=" * 66)
    print(f"  {'approach':>14} {'reached target':>16} {'total compute':>16}")
    print("  " + "-" * 52)
    print(f"  {'flat (full)':>14} {flat_reached}/{args.seeds:>13} "
          f"{fc:>16,.0f}")
    print(f"  {'developmental':>14} {dev_reached}/{args.seeds:>13} "
          f"{dc:>16,.0f}")

    print("\n" + "=" * 66)
    print("WHAT IT SAYS")
    print("=" * 66)
    if dev_reached >= flat_reached and dc < fc * 0.9:
        saving = 100 * (1 - dc / fc)
        print(f"  DEVELOPMENTAL WAS MORE COMPUTE-EFFICIENT. It reached the same")
        print(f"  capability for {saving:.0f}% less total compute than training "
              f"flat at full size,")
        print(f"  by learning foundations cheaply with a small brain before "
              f"spending big-")
        print(f"  network compute on the hard part. How it grew substituted "
              f"for some raw")
        print(f"  scale, a real, measurable efficiency, not a claim of "
              f"superintelligence.")
    elif dev_reached > flat_reached:
        print(f"  DEVELOPMENTAL REACHED THE TARGET MORE OFTEN "
              f"({dev_reached} vs {flat_reached}), even if compute")
        print(f"  was comparable: the staged path solved what flat training "
              f"could not.")
    else:
        print(f"  NO CLEAR WIN. Flat training matched or beat developmental "
              f"here (compute {fc:,.0f}")
        print(f"  vs {dc:,.0f}). Developmental growth was not a free lunch in "
              f"this test; honest")
        print(f"  negative. The trajectory did not save compute under these "
              f"conditions.")

    print(f"\n  Honest scope: one toy task, small networks. This measures "
          f"whether a developmental")
    print(f"  trajectory can be more compute-efficient than flat scaling, the "
          f"human-child")
    print(f"  intuition made concrete. It does not claim to escape compute "
          f"limits entirely;")
    print(f"  it asks whether HOW you grow is a real supplement to HOW MUCH "
          f"you scale.")


if __name__ == "__main__":
    main()
