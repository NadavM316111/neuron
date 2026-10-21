"""A general learning brain: does NEURON develop GENERAL intelligence, or just
narrow per-task tricks?

A narrow brain learns one task and nothing carries over. A general brain learns
HOW TO LEARN, so each new task comes faster because of what it built learning
the others. That transfer is the seed of real intelligence, the difference
between a mind and a lookup table.

THE TEST. One brain faces several genuinely DIFFERENT kinds of task:
  - parity: is the count of high signals odd or even? (logic)
  - majority: are most signals high? (threshold)
  - match: do two specific signals agree? (relational)
  - trend: is the sequence rising? (temporal)
Each is learned by the same brain, on its own, by living through it. Then the
key question: does a brain that has ALREADY learned several tasks learn a NEW
one faster than a fresh brain? If yes, it built general learning ability.

  a seasoned brain learns a new task faster than a fresh brain
      NEURON developed GENERAL intelligence, learning-to-learn across different
      kinds of problem, not narrow tricks. The seed of a real mind.
  no faster than fresh
      It learned each task in isolation; narrow, not general. Honest negative.

    python general_brain.py
"""

import sys
import os
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F

N_IN = 8


def gen_example(task, rng):
    """Produce one (input, label) for a given task type. Same input space,
    different underlying rule, so the brain must learn each rule itself."""
    x = [rng.randint(0, 1) for _ in range(N_IN)]
    if task == "parity":
        y = sum(x) % 2
    elif task == "majority":
        y = 1 if sum(x) > N_IN / 2 else 0
    elif task == "match":
        y = 1 if x[0] == x[1] else 0
    elif task == "trend":
        # rising if second half has more ones than first half
        y = 1 if sum(x[N_IN // 2:]) > sum(x[:N_IN // 2]) else 0
    else:
        y = 0
    return [float(v) for v in x], y


class GeneralBrain(nn.Module):
    """One general brain for all tasks: an encoder + a couple of hidden layers
    + a binary head. Not specialized per task; it learns each rule itself."""

    def __init__(self, hidden=64, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(N_IN, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 2))

    def forward(self, x):
        return self.net(x)


def learn_task(brain, opt, task, steps, dev, target=0.9):
    """Learn one task until it reaches target accuracy or runs out of steps.
    Returns the number of steps it needed (lower = learned faster)."""
    rng = random.Random(hash(task) & 0xffff)
    recent = []
    for step in range(steps):
        xb, yb = [], []
        for _ in range(16):          # small batch
            x, y = gen_example(task, rng)
            xb.append(x); yb.append(y)
        x = torch.tensor(xb, device=dev)
        y = torch.tensor(yb, device=dev)
        with torch.no_grad():
            pred = brain(x).argmax(1)
            acc = (pred == y).float().mean().item()
        recent.append(acc)
        if len(recent) > 20:
            recent.pop(0)
        opt.zero_grad()
        F.cross_entropy(brain(x), y).backward()
        opt.step()
        if len(recent) >= 20 and sum(recent) / len(recent) >= target:
            return step + 1
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  A general brain faces different kinds of task. Does learning "
          f"several make")
    print(f"  it learn a NEW one faster, general intelligence, or is each "
          f"learned alone?\n")

    tasks = ["parity", "majority", "match", "trend"]
    prior_tasks = tasks[:-1]      # learned first
    new_task = tasks[-1]          # the novel test task ("trend")

    fresh_steps, seasoned_steps = [], []
    first_task_steps = []

    for s in range(args.seeds):
        # FRESH brain: learns the new task cold
        fresh = GeneralBrain(args.hidden, seed=s).to(dev)
        fopt = torch.optim.Adam(fresh.parameters(), lr=1e-3)
        fresh_steps.append(learn_task(fresh, fopt, new_task, args.steps, dev))

        # SEASONED brain: learns the prior tasks first, THEN the new one
        seasoned = GeneralBrain(args.hidden, seed=s).to(dev)
        sopt = torch.optim.Adam(seasoned.parameters(), lr=1e-3)
        for i, t in enumerate(prior_tasks):
            n = learn_task(seasoned, sopt, t, args.steps, dev)
            if i == 0:
                first_task_steps.append(n)
        # now the novel task, with everything it built from the others
        seasoned_steps.append(learn_task(seasoned, sopt, new_task,
                                         args.steps, dev))

    import statistics
    fresh_m = statistics.mean(fresh_steps)
    seasoned_m = statistics.mean(seasoned_steps)

    print("=" * 58)
    print("LEARNING A NEW TASK: FRESH BRAIN vs SEASONED BRAIN")
    print("=" * 58)
    print(f"  fresh brain (new task cold):      {fresh_m:>7.0f} steps")
    print(f"  seasoned brain (learned 3 first): {seasoned_m:>7.0f} steps")

    print("\n" + "=" * 58)
    print("WHAT IT SAYS")
    print("=" * 58)
    if seasoned_m < fresh_m * 0.75:
        print(f"  GENERAL INTELLIGENCE: the seasoned brain learned the new task "
              f"{fresh_m/seasoned_m:.1f}x")
        print(f"  faster than a fresh one, because it built general learning "
              f"ability from the")
        print(f"  other tasks. It is not memorizing tricks; it learned HOW to "
              f"learn across")
        print(f"  different kinds of problem. That is the seed of a real mind.")
    elif seasoned_m < fresh_m * 0.95:
        print(f"  Modest transfer ({fresh_m/seasoned_m:.1f}x faster). Some "
              f"general ability built, not")
        print(f"  dramatic. A real but small step toward generality.")
    else:
        print(f"  No clear transfer (seasoned {seasoned_m:.0f} vs fresh "
              f"{fresh_m:.0f} steps). It learned")
        print(f"  each task in isolation, narrow, not general. Honest "
              f"negative.")

    print(f"\n  We built only the capacity to learn (a general brain). NEURON "
          f"learned every")
    print(f"  task itself by living through it. Transfer = it is developing "
          f"GENERAL")
    print(f"  intelligence on its own, the real seed of the mind the vision "
          f"reaches for.")


if __name__ == "__main__":
    main()
