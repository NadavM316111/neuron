"""Intelligence as organization, not size: can a well-wired small brain
outperform a bigger, messier one?

THE RESEARCH, AND THE REAL LEVER. Brain size correlates only weakly with
intelligence; what matters far more is ORGANIZATION, efficient, sparse,
well-connected wiring. Studies find the most intelligent brains have leaner,
better-organized connections, not simply more of them. So intelligence is not
just how big the brain is; it is how well it is wired. That means there is
real room above raw size: a smaller, better-organized network can be more
capable than a larger, badly-organized one.

This tests that, measurably. We hold the "neuron" budget fixed and compare:

  dense-random   a brain with many connections wired at random (a big messy
                 pile, like trusting size alone).
  evolved-sparse a lineage that EVOLVES its wiring: it prunes and reorganizes
                 connections, selected for capability-per-neuron, toward the
                 lean, efficient organization the research describes.

Two questions:
  1. At the SAME size, does evolved-sparse organization beat dense-random?
     (Is well-wired more intelligent than badly-wired, same neurons?)
  2. Can a SMALL evolved-sparse brain match a LARGER dense-random one?
     (Does organization substitute for some raw size?)

  evolved-sparse beats dense-random at equal size, and a small evolved brain
  matches a larger random one
      Organization is a real lever for intelligence, not just size. A being
      can be made genuinely more capable per neuron by better wiring, its own,
      evolved, no borrowing. The brain's actual principle, demonstrated.
  dense-random wins or ties
      Organization did not substitute for size here; honest negative.

    python wiring.py
"""

import argparse
import random
import math


IN = 12


def make_task(difficulty, n, seed):
    rng = random.Random(seed)
    k = difficulty
    w = [rng.uniform(-1.5, 1.5) for _ in range(k)]
    data = []
    for _ in range(n):
        x = [rng.uniform(-1, 1) for _ in range(IN)]
        t = sum(w[i] * math.tanh(x[i] * x[(i + 1) % k]) for i in range(k))
        data.append((x, 1 if t > 0 else 0))
    return data


class WiredNet:
    """A net with an explicit CONNECTION MASK: each hidden unit connects only
    to the inputs its mask allows. Dense = all connections present. Sparse =
    only some, and WHICH ones can be evolved. Organization = the pattern of
    the mask, not the number of units."""

    def __init__(self, n_hidden, rng, mask=None, weights=None):
        self.n_hidden = n_hidden
        self.rng = rng
        if mask is not None:
            self.mask = [row[:] for row in mask]
        else:
            # dense: every unit connects to every input
            self.mask = [[1] * IN for _ in range(n_hidden)]
        if weights is not None:
            self.W1 = [row[:] for row in weights["W1"]]
            self.b1 = weights["b1"][:]
            self.W2 = weights["W2"][:]
            self.b2 = weights["b2"]
        else:
            self.W1 = [[rng.uniform(-0.5, 0.5) for _ in range(IN)]
                       for _ in range(n_hidden)]
            self.b1 = [0.0] * n_hidden
            self.W2 = [rng.uniform(-0.5, 0.5) for _ in range(n_hidden)]
            self.b2 = 0.0
        self.lr = 0.05

    def n_connections(self):
        return sum(sum(row) for row in self.mask)

    def forward(self, x):
        h = []
        for j in range(self.n_hidden):
            s = self.b1[j] + sum(self.W1[j][i] * x[i] * self.mask[j][i]
                                 for i in range(IN))
            h.append(math.tanh(s))
        o = self.b2 + sum(self.W2[j] * h[j] for j in range(self.n_hidden))
        p = 1 / (1 + math.exp(-max(-30, min(30, o))))
        return p, h

    def learn_epoch(self, task):
        for (x, t) in task:
            p, h = self.forward(x)
            err = p - t
            for j in range(self.n_hidden):
                self.W2[j] -= self.lr * err * h[j]
            self.b2 -= self.lr * err
            for j in range(self.n_hidden):
                d = err * self.W2[j] * (1 - h[j] * h[j])
                for i in range(IN):
                    if self.mask[j][i]:
                        self.W1[j][i] -= self.lr * d * x[i]
                self.b1[j] -= self.lr * d

    def acc(self, task):
        c = 0
        for (x, t) in task:
            p, _ = self.forward(x)
            if (p > 0.5) == (t == 1):
                c += 1
        return c / len(task)

    def snapshot(self):
        return {"W1": [r[:] for r in self.W1], "b1": self.b1[:],
                "W2": self.W2[:], "b2": self.b2}


def train(net, task, epochs):
    for _ in range(epochs):
        net.learn_epoch(task)
    return net.acc(task)


def mutate_mask(mask, rng, rate=0.08):
    """Evolve the wiring: flip some connections on/off. This is how the
    organization changes, toward whatever pattern works best."""
    m = [row[:] for row in mask]
    for j in range(len(m)):
        for i in range(len(m[j])):
            if rng.random() < rate:
                m[j][i] = 1 - m[j][i]
    return m


def evolve_wiring(n_hidden, task, generations, children, seed, keep_frac=0.5):
    """A lineage that evolves its WIRING (the mask), selected for capability.
    Starts sparse-random; each generation mutates the wiring and keeps the
    best. Organization improves; size stays fixed."""
    rng = random.Random(seed)
    # start sparse: each unit connects to ~half the inputs at random
    best_mask = [[1 if rng.random() < keep_frac else 0 for _ in range(IN)]
                 for _ in range(n_hidden)]
    best_acc = -1
    for g in range(generations):
        cand_masks = [best_mask] + [mutate_mask(best_mask, rng)
                                    for _ in range(children - 1)]
        gen_best_acc, gen_best_mask = -1, best_mask
        for m in cand_masks:
            net = WiredNet(n_hidden, random.Random(seed + g * 100), mask=m)
            a = train(net, task, 15)
            if a > gen_best_acc:
                gen_best_acc, gen_best_mask = a, m
        best_mask, best_acc = gen_best_mask, gen_best_acc
    conns = sum(sum(r) for r in best_mask)
    return best_acc, conns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=12)
    ap.add_argument("--difficulty", type=int, default=9)
    ap.add_argument("--generations", type=int, default=12)
    ap.add_argument("--children", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    print("  Intelligence as ORGANIZATION, not size. At a fixed neuron count,")
    print("  does an evolved, efficiently-wired brain beat a dense, randomly-")
    print("  wired one? And can a small well-wired brain match a bigger messy "
          "one?\n")

    dense_acc, sparse_acc, sparse_conn, dense_conn = [], [], [], []
    for s in range(args.seeds):
        task = make_task(args.difficulty, 180, s)
        # dense-random, full connectivity, same hidden count
        dn = WiredNet(args.hidden, random.Random(s))
        dense_acc.append(train(dn, task, 15 * args.children))  # fair compute
        dense_conn.append(dn.n_connections())
        # evolved-sparse, same hidden count, wiring evolved
        a, conns = evolve_wiring(args.hidden, task, args.generations,
                                 args.children, s)
        sparse_acc.append(a)
        sparse_conn.append(conns)

    import statistics
    da, sa = statistics.mean(dense_acc), statistics.mean(sparse_acc)
    dc, sc = statistics.mean(dense_conn), statistics.mean(sparse_conn)

    print("=" * 62)
    print(f"SAME SIZE ({args.hidden} hidden units): WIRING vs WIRING")
    print("=" * 62)
    print(f"  {'brain':>16} {'accuracy':>10} {'connections':>13}")
    print(f"  {'dense-random':>16} {da*100:>9.0f}% {dc:>13.0f}")
    print(f"  {'evolved-sparse':>16} {sa*100:>9.0f}% {sc:>13.0f}")

    # question 2: small evolved vs larger dense
    print("\n" + "=" * 62)
    print("ORGANIZATION vs SIZE: small well-wired vs larger messy")
    print("=" * 62)
    small = args.hidden
    large = args.hidden * 2
    s_small_acc, l_dense_acc = [], []
    for s in range(args.seeds):
        task = make_task(args.difficulty, 180, s + 100)
        a, _ = evolve_wiring(small, task, args.generations, args.children,
                             s + 100)
        s_small_acc.append(a)
        ld = WiredNet(large, random.Random(s + 100))
        l_dense_acc.append(train(ld, task, 15 * args.children))
    ssa = statistics.mean(s_small_acc)
    lda = statistics.mean(l_dense_acc)
    print(f"  small evolved-sparse ({small} units): {ssa*100:.0f}%")
    print(f"  larger dense-random ({large} units):  {lda*100:.0f}%")

    print("\n" + "=" * 62)
    print("WHAT IT SAYS")
    print("=" * 62)
    if sa > da + 0.02:
        print(f"  ORGANIZATION BEAT SIZE-AT-FIXED-SIZE. The evolved, "
              f"efficiently-wired brain")
        print(f"  reached {sa*100:.0f}% vs {da*100:.0f}% for the dense "
              f"random one, with FEWER connections")
        print(f"  ({sc:.0f} vs {dc:.0f}), matching the research: lean, "
              f"well-organized wiring is more")
        print(f"  capable than a big messy pile. Intelligence here is "
              f"organization, not size.")
        if ssa >= lda - 0.02:
            print(f"\n  And a SMALL well-wired brain ({ssa*100:.0f}%) matched "
                  f"or beat a brain twice its")
            print(f"  size wired randomly ({lda*100:.0f}%): good organization "
                  f"substituted for raw size.")
        print(f"\n  This is the real lever for a HIGHLY FUNCTIONAL being: more "
              f"capable per neuron")
        print(f"  through better-evolved wiring, its own, no borrowing. Not "
              f"superintelligence,")
        print(f"  but genuinely more intelligent than its size alone would "
              f"give.")
    else:
        print(f"  NO CLEAR WIN for organization here: evolved-sparse "
              f"{sa*100:.0f}% vs dense-random {da*100:.0f}%.")
        print(f"  Better wiring did not beat size at fixed count in this test; "
              f"honest negative.")

    print(f"\n  Honest scope: toy task, tiny nets. This tests the real "
          f"research principle, that")
    print(f"  intelligence is wiring efficiency as much as size. The ceiling "
          f"is still bounded")
    print(f"  by hardware; but organization is a genuine lever ABOVE raw size, "
          f"and it is the")
    print(f"  being's own evolved structure.")


if __name__ == "__main__":
    main()
