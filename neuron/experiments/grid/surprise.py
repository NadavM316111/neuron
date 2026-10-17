"""The brain's energy trick: only pay for surprise.

THE PRINCIPLE. The brain does not spend energy on what it already expects. It
is a prediction machine that only responds to prediction ERROR, to surprise.
What it correctly predicts, it ignores; it spends its limited energy only on
what violates its model. This is predictive processing, and it is a large part
of how the brain runs on ~20 watts: most of the stream is expected, and the
expected is nearly free.

THE TEST, MEASURABLE. A being learns a stream in which much is predictable and
some is surprising. Two strategies, same task, same target accuracy:

  process-all     updates on every observation, expected or not. pays full
                  compute for the whole stream.
  surprise-gated  predicts first; if it was right (no surprise), it skips the
                  costly update, spending almost nothing; it pays the full
                  update only when it was WRONG (surprised), because that is
                  the only thing it needs to learn from.

We measure the total compute (updates performed) each spends to reach the same
accuracy. If surprise-gating reaches the same capability for less compute, the
brain's energy trick is real and NEURON's own, a genuine efficiency lever that
makes later scaling cheaper.

  surprise-gated reaches the same accuracy for less compute
      Only paying for surprise is a real efficiency: the being spends its
      compute where it actually needs to learn, skipping the expected.
  it costs the same/more, or hurts accuracy
      Surprise-gating did not help here; honest negative, reported plainly.

    python surprise.py
"""

import argparse
import random
import math


IN = 8


def make_stream(n, predictable_frac, seed):
    """A stream where a fraction of situations are EASY/predictable (recur, so
    once learned they are expected) and the rest are varied/surprising."""
    rng = random.Random(seed)
    # a small set of recurring 'expected' situations, plus varied ones
    common = [([rng.uniform(-1, 1) for _ in range(IN)], rng.randint(0, 1))
              for _ in range(6)]
    stream = []
    for _ in range(n):
        if rng.random() < predictable_frac:
            x, y = common[rng.randrange(len(common))]
            x = [v + rng.gauss(0, 0.02) for v in x]   # slight noise
            stream.append((x, y))
        else:
            x = [rng.uniform(-1, 1) for _ in range(IN)]
            # a harder target from the input
            y = 1 if sum(math.sin(3 * xi) for xi in x) > 0 else 0
            stream.append((x, y))
    return stream


class Net:
    def __init__(self, rng):
        self.H = 16
        self.W1 = [[rng.uniform(-0.5, 0.5) for _ in range(IN)]
                   for _ in range(self.H)]
        self.b1 = [0.0] * self.H
        self.W2 = [rng.uniform(-0.5, 0.5) for _ in range(self.H)]
        self.b2 = 0.0
        self.lr = 0.1
        self.compute = 0

    def forward(self, x):
        h = [math.tanh(self.b1[j] + sum(self.W1[j][i] * x[i]
                                        for i in range(IN)))
             for j in range(self.H)]
        o = self.b2 + sum(self.W2[j] * h[j] for j in range(self.H))
        p = 1 / (1 + math.exp(-max(-30, min(30, o))))
        return p, h

    def predict(self, x):
        # a prediction costs a little (forward pass), always paid.
        self.compute += self.H
        p, h = self.forward(x)
        return p, h

    def update(self, x, h, p, t):
        # the costly part: the full backward update. paid only when we choose.
        self.compute += self.H * IN
        err = p - t
        for j in range(self.H):
            self.W2[j] -= self.lr * err * h[j]
        self.b2 -= self.lr * err
        for j in range(self.H):
            d = err * self.W2[j] * (1 - h[j] * h[j])
            for i in range(IN):
                self.W1[j][i] -= self.lr * d * x[i]
            self.b1[j] -= self.lr * d


def run(mode, stream, target, seed):
    rng = random.Random(seed)
    net = Net(rng)
    correct_window = []
    for pass_num in range(8):
        for (x, t) in stream:
            p, h = net.predict(x)
            right = (p > 0.5) == (t == 1)
            correct_window.append(1 if right else 0)
            if len(correct_window) > 300:
                correct_window.pop(0)
            if mode == "process_all":
                net.update(x, h, p, t)          # always update
            else:  # surprise_gated
                # only update when WRONG (surprised). if already right, the
                # prediction was expected - skip the costly update.
                if not right:
                    net.update(x, h, p, t)
        acc = sum(correct_window) / len(correct_window)
        if acc >= target:
            return acc, net.compute, pass_num + 1
    return (sum(correct_window) / len(correct_window)), net.compute, 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.85)
    ap.add_argument("--predictable", type=float, default=0.6)
    ap.add_argument("--seeds", type=int, default=6)
    args = ap.parse_args()

    print("  The brain only pays for surprise. A being predicts first; if it "
          "was right")
    print("  (expected), it skips the costly update. Does paying only for "
          "surprise reach")
    print(f"  the same accuracy for less compute? (stream is "
          f"{args.predictable*100:.0f}% predictable)\n")

    all_c, gate_c, all_a, gate_a = [], [], [], []
    for s in range(args.seeds):
        stream = make_stream(400, args.predictable, s)
        a1, c1, _ = run("process_all", stream, args.target, s)
        a2, c2, _ = run("surprise_gated", stream, args.target, s)
        all_a.append(a1); all_c.append(c1)
        gate_a.append(a2); gate_c.append(c2)

    import statistics
    aa, ac = statistics.mean(all_a), statistics.mean(all_c)
    ga, gc = statistics.mean(gate_a), statistics.mean(gate_c)

    print("=" * 60)
    print("COMPUTE TO REACH THE SAME CAPABILITY")
    print("=" * 60)
    print(f"  {'strategy':>16} {'accuracy':>10} {'compute':>14}")
    print(f"  {'process-all':>16} {aa*100:>9.0f}% {ac:>14,.0f}")
    print(f"  {'surprise-gated':>16} {ga*100:>9.0f}% {gc:>14,.0f}")

    print("\n" + "=" * 60)
    print("WHAT IT SAYS")
    print("=" * 60)
    # must reach comparable accuracy AND spend clearly less compute
    if ga >= aa - 0.03 and gc < ac * 0.8:
        saving = 100 * (1 - gc / ac)
        print(f"  PAYING ONLY FOR SURPRISE WORKED. The surprise-gated being "
              f"reached the same")
        print(f"  accuracy ({ga*100:.0f}% vs {aa*100:.0f}%) for {saving:.0f}% "
              f"less compute, by skipping the costly")
        print(f"  update on everything it already predicted correctly and "
              f"spending only on")
        print(f"  what surprised it. The brain's energy trick, real and "
              f"NEURON's own, a")
        print(f"  genuine efficiency lever that makes later scaling cheaper.")
    elif ga < aa - 0.03:
        print(f"  SURPRISE-GATING HURT ACCURACY ({ga*100:.0f}% vs "
              f"{aa*100:.0f}%). Skipping updates on correct")
        print(f"  predictions cost it some learning; not a clean win. Honest "
              f"negative.")
    else:
        print(f"  NO CLEAR COMPUTE SAVING ({gc:,.0f} vs {ac:,.0f}). "
              f"Surprise-gating did not reduce")
        print(f"  compute enough here; honest negative.")

    print(f"\n  Honest scope: toy stream, tiny net. This tests the brain's "
          f"predictive-processing")
    print(f"  energy trick, pay only for surprise, as a real efficiency "
          f"lever. It does not")
    print(f"  raise the capability ceiling; it makes reaching capability "
          f"cheaper, which is")
    print(f"  exactly what makes later scaling more affordable.")


if __name__ == "__main__":
    main()
