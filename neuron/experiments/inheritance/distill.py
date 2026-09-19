"""Can an old instance teach a new one?

Merging answered the cold start: a newcomer seeded from a 16-way merge needs
500 steps to reach what a blank one takes 6,000 to reach. But merging has a
hard requirement — every instance must descend from the same initialisation,
because that is what keeps them linearly connected. Change the architecture,
change the initialisation, and you cannot merge across the boundary.

DISTILLATION has no such requirement. The teacher does not hand over weights;
it hands over BEHAVIOUR. It sees a situation, says what it expects to happen,
and the student learns from that. Architectures can differ. Sizes can differ.
The teacher can be retired afterwards.

That makes it the route that survives a version change, which merging cannot.

Three ways a student can learn, all from the same number of steps:

  alone        lives its own life from random weights. The floor.
  merged       seeded from the teacher's weights, then lives its own life.
               The method that already worked, for comparison.
  distilled    lives its own life, but at each step ALSO learns from what
               the teacher predicted for that moment. The teacher is never
               copied, only consulted.
  both         seeded from the teacher AND distilled from it.

WHAT THE TEACHER KNOWS AND THE STUDENT DOES NOT: the teacher has lived a long
"keyed" life. The student lives a fresh keyed life on a DIFFERENT layout, so
nothing transfers by memorising a map — only the RULE can transfer.

A CROSS-SIZE ARM is included, where the teacher is 128 units and the student
is 64. Merging cannot do that at all, so if distillation works there it has
a capability merging lacks, which is the point of testing it.
"""

import copy
import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from world import GridWorld, ACTIONS, EVENTS, DELTA, LOCKED


RULES_USED = "keyed"        # the rule that requires memory
TEACHER_STEPS = 40000
STUDENT_STEPS = 6000
EPISODE = 200
TEACHER_HIDDEN = 128
STUDENT_HIDDEN = 128
SMALL_HIDDEN = 64
LR = 3e-4
SEEDS = [0, 1, 2]
N_TERRAIN = 5
PATCH = 9
TEST_STEPS = 5000
CHECKPOINTS = [500, 1000, 2000, 4000, 6000]
DISTILL_WEIGHT = 1.0        # how much the teacher's opinion counts
DISTILL_TEMP = 2.0          # softens the teacher's distribution


def obs_dim():
    return PATCH * N_TERRAIN + len(ACTIONS)


def encode(obs, action):
    v = torch.zeros(obs_dim())
    for i, cell in enumerate(obs["patch"]):
        v[i * N_TERRAIN + cell] = 1.0
    v[PATCH * N_TERRAIN + ACTIONS.index(action)] = 1.0
    return v.unsqueeze(0)


class GRU(nn.Module):
    def __init__(self, hidden, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.enc = nn.Linear(obs_dim(), hidden)
        self.cell = nn.GRUCell(hidden, hidden)
        self.head = nn.Linear(hidden, len(EVENTS))

    def forward(self, x, h=None):
        e = F.relu(self.enc(x))
        h = self.cell(e, h)
        return self.head(h), h


def walk(layout_seed, walk_seed, n, rules=RULES_USED):
    rng = random.Random(walk_seed)
    world = GridWorld(layout_seed, rules=rules)
    out = []
    for i in range(n):
        if i % EPISODE == 0:
            world.reset()
        obs = world.observe()
        action = rng.choice(ACTIONS)
        dr, dc = DELTA[action]
        r, c = world.r + dr, world.c + dc
        at_door = (0 <= r < world.h and 0 <= c < world.w
                   and world.grid[r][c] == LOCKED)
        _, event = world.step(action)
        out.append((obs, action, event, at_door))
    return out


def train(net, data, opt=None, teacher=None):
    """Live a life. If a teacher is present, also learn from its opinion.

    The teacher runs its own hidden state over the same stream, so its
    predictions carry its memory of what it has seen — which is the thing
    worth transferring. It is never copied, only consulted.
    """
    if opt is None:
        opt = torch.optim.Adam(net.parameters(), lr=LR)
    h = None
    th = None
    net.train()
    if teacher is not None:
        teacher.eval()

    for i, item in enumerate(data):
        if i % EPISODE == 0:
            h = None
            th = None
        x = encode(item[0], item[1])
        logits, h = net(x, h)
        loss = F.cross_entropy(
            logits, torch.tensor([EVENTS.index(item[2])]))

        if teacher is not None:
            with torch.no_grad():
                t_logits, th = teacher(x, th)
                th = th.detach()
            # match the teacher's whole distribution, not just its top
            # choice — the shape of its uncertainty is part of what it knows
            soft = F.kl_div(
                F.log_softmax(logits / DISTILL_TEMP, dim=1),
                F.softmax(t_logits / DISTILL_TEMP, dim=1),
                reduction="batchmean") * (DISTILL_TEMP ** 2)
            loss = loss + DISTILL_WEIGHT * soft

        opt.zero_grad()
        loss.backward()
        opt.step()
        h = h.detach()
    return net


def evaluate(net, test):
    """Accuracy at locked doors, where the rule actually matters."""
    net.eval()
    h = None
    hit = seen = 0
    with torch.no_grad():
        for i, item in enumerate(test):
            if i % EPISODE == 0:
                h = None
            logits, h = net(encode(item[0], item[1]), h)
            if item[3]:
                seen += 1
                hit += int(logits.argmax(1).item()) == EVENTS.index(item[2])
    return 100.0 * hit / seen if seen else None


def curve(net, data, test, teacher=None):
    """Accuracy at each checkpoint through a single life."""
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    out = {}
    done = 0
    for cp in CHECKPOINTS:
        chunk = data[done:cp]
        if chunk:
            train(net, chunk, opt, teacher)
        done = cp
        out[cp] = evaluate(net, test)
    return out


if __name__ == "__main__":
    print("Can an old instance teach a new one?")
    print(f"teacher lives {TEACHER_STEPS} steps, student gets "
          f"{STUDENT_STEPS} on a DIFFERENT layout\n")

    def mean(xs):
        xs = [x for x in xs if x is not None and x == x]
        return sum(xs) / len(xs) if xs else float("nan")

    arms = ["alone", "merged", "distilled", "both", "small-alone",
            "small-distilled"]
    curves = {a: [] for a in arms}
    teacher_scores = []

    for seed in SEEDS:
        # The teacher: a long life on its own layout.
        t_data = walk(100, seed * 10 + 1, TEACHER_STEPS)
        teacher = train(GRU(TEACHER_HIDDEN, seed), t_data)

        # The student's world: a different layout, same rule.
        s_data = walk(200, seed * 10 + 2, STUDENT_STEPS)
        s_test = walk(200, 70000 + seed, TEST_STEPS)
        teacher_scores.append(evaluate(teacher, s_test))

        # same-size arms
        curves["alone"].append(
            curve(GRU(STUDENT_HIDDEN, seed + 900), s_data, s_test))
        curves["merged"].append(
            curve(copy.deepcopy(teacher), s_data, s_test))
        curves["distilled"].append(
            curve(GRU(STUDENT_HIDDEN, seed + 900), s_data, s_test, teacher))
        curves["both"].append(
            curve(copy.deepcopy(teacher), s_data, s_test, teacher))

        # cross-size arms: merging cannot do this at all
        curves["small-alone"].append(
            curve(GRU(SMALL_HIDDEN, seed + 900), s_data, s_test))
        curves["small-distilled"].append(
            curve(GRU(SMALL_HIDDEN, seed + 900), s_data, s_test, teacher))

        del teacher
        print(f"  seed {seed} done", flush=True)

    print(f"\nthe teacher scores {mean(teacher_scores):.2f}% on the "
          f"student's world\n(it never trained there, so this is the rule "
          f"transferring, not the map)\n")

    print("=" * 78)
    print("ACCURACY THROUGH THE STUDENT'S LIFE")
    print(f"{'steps':>8} " + "  ".join(f"{a[:13]:>13}" for a in arms[:4]))
    print("-" * 78)
    for cp in CHECKPOINTS:
        print(f"{cp:>8} " + "  ".join(
            f"{mean([c[cp] for c in curves[a]]):>12.2f}%" for a in arms[:4]))
    print("=" * 78)

    print("\nCROSS-SIZE: teacher 128 units, student 64. "
          "Merging cannot do this.")
    print(f"{'steps':>8} {'small-alone':>14} {'small-distilled':>17} "
          f"{'advantage':>11}")
    print("-" * 78)
    for cp in CHECKPOINTS:
        a = mean([c[cp] for c in curves["small-alone"]])
        d = mean([c[cp] for c in curves["small-distilled"]])
        print(f"{cp:>8} {a:>13.2f}% {d:>16.2f}% {d - a:>+10.2f}")
    print("=" * 78)

    first, last = CHECKPOINTS[0], CHECKPOINTS[-1]
    base_first = mean([c[first] for c in curves["alone"]])
    base_last = mean([c[last] for c in curves["alone"]])

    print(f"\nADVANTAGE OVER LEARNING ALONE:")
    for a in arms[1:4]:
        f = mean([c[first] for c in curves[a]]) - base_first
        l = mean([c[last] for c in curves[a]]) - base_last
        print(f"  {a:>10}: {f:+7.2f} at {first} steps, "
              f"{l:+7.2f} at {last}")

    sa = mean([c[first] for c in curves["small-alone"]])
    sd = mean([c[first] for c in curves["small-distilled"]])
    print(f"\n  cross-size at {first} steps: {sd - sa:+.2f}")

    print("""
Two things this settles.

  DOES BEHAVIOUR TRANSFER WITHOUT WEIGHTS? If distilled beats alone, a
      retiring instance can pass on what it learned without handing over its
      parameters. That matters because merging requires a shared
      initialisation, so it cannot cross a version change. Distillation can.

  IS IT BETTER OR WORSE THAN MERGING? Merged is the method already shown to
      work. If distilled matches it, there are two independent routes and
      the vision is not dependent on either. If "both" beats each alone, they
      are complementary — seed from the merge, then keep consulting the
      teacher while the student settles in.

The cross-size table is the capability merging simply does not have. A 64-unit
student cannot be seeded from a 128-unit teacher by averaging, because the
weight shapes do not match. It can be taught by one. If that column shows a
real advantage, distillation is the more general mechanism even where it is
not the stronger one.
""")
    with open("distill.json", "w") as f:
        json.dump(curves, f, indent=2, default=str)
    print("wrote distill.json")