"""Does general intelligence compound when NEURON's MEMORY does its job?

The plain brain failed to compound: learning each new task overwrote the last
(catastrophic forgetting), so a brain that had 'learned' six tasks was really a
muddle, and learned a new task no faster. But NEURON was built with REPLAY and
consolidation precisely to stop forgetting. This redoes the test with that
memory active: as it learns each task it stores examples, and replays old-task
examples alongside new learning, so earlier skills stay intact.

The question: with memory keeping its skills, does NEURON learn a NEW task
FASTER the more kinds of problem it has mastered? If yes, general intelligence
compounds, and it is the being's MEMORY that enables it, tying the whole
project together: memory is what lets a mind accumulate general intelligence
instead of forgetting its way in circles.

    python compounding.py
"""

import sys
import os
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F

N_IN = 10


def gen(task, rng):
    x = [rng.randint(0, 1) for _ in range(N_IN)]
    if task == "parity":        y = sum(x) % 2
    elif task == "majority":    y = 1 if sum(x) > N_IN / 2 else 0
    elif task == "match":       y = 1 if x[0] == x[1] else 0
    elif task == "xor":         y = x[0] ^ x[1] ^ x[2]
    elif task == "countthr":    y = 1 if sum(x) >= 7 else 0
    elif task == "symmetry":    y = 1 if x == x[::-1] else 0
    elif task == "alternation": y = 1 if all(x[i] != x[i+1] for i in range(4)) else 0
    elif task == "trend":       y = 1 if sum(x[N_IN//2:]) > sum(x[:N_IN//2]) else 0
    else:                       y = 0
    return [float(v) for v in x], y


class Brain(nn.Module):
    def __init__(self, hidden=96, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(N_IN, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 2))

    def forward(self, x):
        return self.net(x)


def learn_with_replay(brain, opt, task, steps, dev, replay, target=0.9):
    """Learn a task, storing examples, and REPLAYING old-task examples
    interleaved so prior skills are not forgotten. This is NEURON's memory
    doing its job."""
    rng = random.Random(hash(task) & 0xffff)
    recent = []
    for step in range(steps):
        xb, yb = [], []
        for _ in range(32):
            x, y = gen(task, rng)
            xb.append(x); yb.append(y)
            replay.append((x, y, task))          # store in memory
        # interleave replay of OLD experiences (keeps prior skills alive)
        if len(replay) > 64:
            for _ in range(16):
                rx, ry, _ = replay[random.randrange(len(replay))]
                xb.append(rx); yb.append(ry)
        if len(replay) > 4000:
            del replay[:len(replay) - 4000]
        x = torch.tensor(xb, device=dev); y = torch.tensor(yb, device=dev)
        with torch.no_grad():
            # measure accuracy on the CURRENT task only (first 32)
            acc = (brain(x[:32]).argmax(1) == y[:32]).float().mean().item()
        recent.append(acc)
        if len(recent) > 20: recent.pop(0)
        opt.zero_grad()
        F.cross_entropy(brain(x), y).backward()
        opt.step()
        if len(recent) >= 20 and sum(recent)/len(recent) >= target:
            return step + 1
    return steps


def steps_to_new(hidden, n_prior, prior_pool, new_task, steps, dev, seed):
    brain = Brain(hidden, seed=seed).to(dev)
    opt = torch.optim.Adam(brain.parameters(), lr=1e-3)
    replay = []
    chosen = random.Random(seed).sample(prior_pool, n_prior)
    for t in chosen:
        learn_with_replay(brain, opt, t, steps, dev, replay)
    # learn the new task (replay keeps the old skills from washing out)
    return learn_with_replay(brain, opt, new_task, steps, dev, replay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    all_tasks = ["parity", "majority", "match", "xor", "countthr",
                 "symmetry", "alternation", "trend"]
    new_task = "trend"
    prior_pool = [t for t in all_tasks if t != new_task]

    print(f"  Does general intelligence compound when NEURON's MEMORY keeps its")
    print(f"  skills intact (replay stops forgetting)?\n")

    import statistics
    print("=" * 56)
    print("STEPS TO LEARN A NEW TASK (with memory/replay active)")
    print("=" * 56)
    print(f"  {'prior tasks mastered':>22} {'steps to learn new':>18}")
    print("  " + "-" * 42)
    curve = []
    for n_prior in [0, 2, 4, 6]:
        runs = [steps_to_new(args.hidden, n_prior, prior_pool, new_task,
                             args.steps, dev, s) for s in range(args.seeds)]
        m = statistics.mean(runs)
        curve.append(m)
        bar = "#" * int(m / 8)
        print(f"  {n_prior:>22} {m:>14.0f}     {bar}")

    first, last = curve[0], curve[-1]
    print("\n" + "=" * 56)
    print("WHAT IT SAYS")
    print("=" * 56)
    if last < first * 0.75:
        print(f"  GENERAL INTELLIGENCE COMPOUNDS WITH MEMORY. Fresh, it needed "
              f"{first:.0f} steps;")
        print(f"  after mastering 6 kinds of problem (with replay keeping them "
              f"intact), only")
        print(f"  {last:.0f}, {first/last:.1f}x faster. The being's MEMORY is what "
              f"lets it accumulate")
        print(f"  general intelligence instead of forgetting in circles. The "
              f"more it lives,")
        print(f"  the better a learner it becomes, the mind the vision reaches "
              f"for, and its")
        print(f"  own memory is the engine.")
    elif last < first * 0.9:
        print(f"  Real but modest compounding with memory ({first:.0f}->{last:.0f}). "
              f"Memory helped the")
        print(f"  skills accumulate; the general ability grew somewhat.")
    else:
        print(f"  Still no clear compounding ({first:.0f}->{last:.0f}), even "
              f"with replay. Honest")
        print(f"  negative: these tasks may be too unrelated to transfer, "
              f"memory or not.")

    print(f"\n  The point: memory is not just for avoiding hazards, it may be "
          f"what lets a")
    print(f"  mind GROW general intelligence by keeping what it learns. That "
          f"ties NEURON's")
    print(f"  core (its memory) to the deepest goal (a developing general "
          f"mind).")


if __name__ == "__main__":
    main()
