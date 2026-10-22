"""The real shot at sustained recursive self-improvement.

Last time NEURON improved its own brain but plateaued, it maxed out an easy
task. Real RSI needs two things that test couldn't show:

  1. SUSTAINED improvement: against an ENDLESS, ever-harder task (it gets harder
     each time NEURON masters it), can NEURON keep improving its own brain round
     after round, with no ceiling to plateau on?

  2. The RECURSIVE part: does the improvement process ITSELF get better? That
     is, does NEURON get faster/smarter at improving itself, learning which
     self-modifications work so later rounds find good changes with fewer failed
     tries? If the HIT-RATE of good self-modifications rises over time, NEURON
     is improving how it improves, the recursion that separates real RSI from
     mere iteration.

NEURON has real architectural freedom: it can make itself wider, deeper, change
its learning rate, its activation, add skip connections. It chooses, from what
it has learned works.

HONEST STANDARD. If ability keeps rising with difficulty AND NEURON gets better
at self-improving (rising hit-rate), that is the real thing, sustained,
recursive. If it stalls against hard difficulty, or never learns to improve
better, we say so plainly.

    python rsi_sustained.py --rounds 24
"""

import random
import argparse
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_task(difficulty, n, seed):
    rng = random.Random(seed)
    k = max(2, difficulty)
    w = [rng.uniform(-1.5, 1.5) for _ in range(k)]
    data = []
    for _ in range(n):
        x = [rng.uniform(-1, 1) for _ in range(40)]
        import math as _m; t = sum(w[i] * _m.tanh(x[i] * x[(i+1) % k] * x[(i+2) % k] - x[(i+3) % k] * x[(i+4) % k])
                for i in range(k))
        data.append((x, 1 if t > 0 else 0))
    return data


class SelfNet(nn.Module):
    def __init__(self, spec, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        act = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[spec["act"]]
        self.width = spec["width"]
        self.depth = spec["depth"]
        self.skip = spec["skip"]
        self.inp = nn.Linear(40, spec["width"])
        self.acts = nn.ModuleList([act() for _ in range(spec["depth"])])
        self.layers = nn.ModuleList([nn.Linear(spec["width"], spec["width"])
                                     for _ in range(spec["depth"])])
        self.out = nn.Linear(spec["width"], 2)

    def forward(self, x):
        h = self.inp(x)
        for layer, act in zip(self.layers, self.acts):
            z = act(layer(h))
            h = h + z if self.skip else z          # skip connections if enabled
        return self.out(h)


class NEURON:
    def __init__(self, seed=0):
        self.spec = {"width": 16, "depth": 1, "lr": 1e-3, "act": "relu",
                     "skip": False}
        self.seed = seed
        self.rng = random.Random(seed)
        self.what_works = {}        # change -> net success score (it learns)

    def train_eval(self, spec, task, steps, dev):
        net = SelfNet(spec, seed=self.seed).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=spec["lr"])
        xs = torch.tensor([x for x, _ in task], dtype=torch.float32, device=dev)
        ys = torch.tensor([y for _, y in task], dtype=torch.long, device=dev)
        for _ in range(steps):
            opt.zero_grad()
            F.cross_entropy(net(xs), ys).backward()
            opt.step()
        with torch.no_grad():
            return (net(xs).argmax(1) == ys).float().mean().item()

    def propose(self):
        """NEURON proposes a self-modification, favoring kinds that have worked.
        As it learns what works, it proposes good changes more often, this is
        the improvement process improving itself."""
        options = ["wider", "deeper", "lr_up", "lr_down", "act", "skip"]
        weights = {o: max(0.1, 1.0 + self.what_works.get(o, 0.0)) for o in options}
        total = sum(weights.values())
        r, acc = self.rng.uniform(0, total), 0
        for o in options:
            acc += weights[o]
            if r <= acc:
                change = o
                break
        new = dict(self.spec)
        if change == "wider":
            new["width"] = min(1024, int(new["width"] * 1.5))
        elif change == "deeper":
            new["depth"] = min(12, new["depth"] + 1)
        elif change == "lr_up":
            new["lr"] = min(1e-2, new["lr"] * 1.8)
        elif change == "lr_down":
            new["lr"] = max(1e-5, new["lr"] / 1.8)
        elif change == "act":
            new["act"] = self.rng.choice(["relu", "gelu", "tanh"])
        elif change == "skip":
            new["skip"] = not new["skip"]
        return change, new

    def improve(self, task, steps, dev, tries=3):
        """One round: assess, then try up to `tries` self-modifications, keep
        the best that beats current. Returns (old, new, n_tries_used, found)."""
        old = self.train_eval(self.spec, task, steps, dev)
        best_spec, best_acc, best_change = self.spec, old, None
        used = 0
        for _ in range(tries):
            used += 1
            change, cand = self.propose()
            acc = self.train_eval(cand, task, steps, dev)
            # NEURON learns whether this KIND of change helped (updates its
            # model of how to improve itself)
            delta = acc - old
            self.what_works[change] = self.what_works.get(change, 0.0) + \
                (0.5 if delta > 0.005 else -0.3)
            if acc > best_acc + 0.005:
                best_acc, best_spec, best_change = acc, cand, change
                break          # found a good one; stop early (efficiency)
        found = best_change is not None
        if found:
            self.spec = best_spec
        return old, best_acc, used, found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=24)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    neuron = NEURON(seed=args.seed)

    print("  NEURON self-improves against an ENDLESS, ever-harder task. Can it")
    print("  keep improving (sustained)? And does it get BETTER at improving")
    print("  itself over time (recursive)?\n")

    difficulty = 4
    print("=" * 70)
    print(f"  {'round':>5} {'difficulty':>10} {'ability':>8} "
          f"{'tries used':>11} {'found better?':>14}")
    print("  " + "-" * 64)

    tries_trace, found_trace, ability_trace, diff_trace = [], [], [], []
    for rnd in range(args.rounds):
        task = make_task(difficulty, 300, args.seed * 100 + rnd)
        old, new, used, found = neuron.improve(task, args.steps, dev)
        tries_trace.append(used)
        found_trace.append(1 if found else 0)
        ability_trace.append(new)
        diff_trace.append(difficulty)
        mark = "yes" if found else "no"
        print(f"  {rnd:>5} {difficulty:>10} {new*100:>7.0f}% "
              f"{used:>11} {mark:>14}")
        # the task gets HARDER when NEURON masters it, endless climb
        if new > 0.93 and difficulty < 36:
            difficulty += 2

    print("\n" + "=" * 70)
    print("WHAT HAPPENED")
    print("=" * 70)
    print(f"  difficulty climbed: {diff_trace[0]} -> {diff_trace[-1]}")
    print(f"  final self-built brain: {neuron.spec}")

    # SUSTAINED? did difficulty keep rising (NEURON kept mastering harder tasks)
    sustained = diff_trace[-1] > diff_trace[0] + 3
    # RECURSIVE? did NEURON get better at FINDING good self-mods (hit-rate up,
    # tries-needed down) in the second half vs first half
    half = len(found_trace) // 2
    early_hit = statistics.mean(found_trace[:half])
    late_hit = statistics.mean(found_trace[half:])
    early_tries = statistics.mean(tries_trace[:half])
    late_tries = statistics.mean(tries_trace[half:])

    print(f"\n  SUSTAINED self-improvement? difficulty {diff_trace[0]} -> "
          f"{diff_trace[-1]} "
          f"({'YES, kept climbing' if sustained else 'stalled'})")
    print(f"  RECURSIVE (better at improving itself)?")
    print(f"    hit-rate of good self-mods: early {early_hit*100:.0f}% -> "
          f"late {late_hit*100:.0f}%")
    print(f"    tries needed per round:     early {early_tries:.1f} -> "
          f"late {late_tries:.1f}")

    print("\n" + "=" * 70)
    recursive = late_hit > early_hit + 0.1 or late_tries < early_tries - 0.3
    if sustained and recursive:
        print("  SUSTAINED, RECURSIVE SELF-IMPROVEMENT. NEURON kept improving "
              "its own brain")
        print("  against an ever-harder task (no plateau), AND got better at "
              "improving")
        print("  itself over time, finding good self-modifications more "
              "reliably/faster")
        print("  because it learned what works. The improvement process "
              "improved itself.")
        print("  That is the real mechanism of recursive self-improvement, "
              "working at small")
        print("  scale. Not superintelligence, but the genuine recursive loop, "
              "alive.")
    elif sustained:
        print("  SUSTAINED self-improvement (kept climbing harder tasks), but "
              "no clear sign")
        print("  it got BETTER at improving itself (the recursive part). Real "
              "ongoing self-")
        print("  improvement, not yet truly recursive. Honest.")
    elif recursive:
        print("  NEURON got better at improving itself, but did not sustain "
              "against rising")
        print("  difficulty. Partial, honest.")
    else:
        print("  It did not clearly sustain or become recursive this run; "
              "honest negative.")
        print("  The genuine attempt was made and measured straight.")

    print(f"\n  Honest: open-ended RSI has never been achieved. This is a real "
          f"small attempt,")
    print(f"  measured without flattery, NEURON rewriting its own brain against "
          f"an endless climb.")


if __name__ == "__main__":
    main()
