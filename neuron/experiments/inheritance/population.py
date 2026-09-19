"""Does merging survive many instances, not two?

Merging two lives worked and beat joint training. But the vision needs
merging MANY — a new user seeded from thousands of predecessors, not one.
Averaging sixteen diverged networks is a much harder ask than averaging two,
and it is where the approach most plausibly breaks.

Each instance lives its own life: its own layout, its own wander, and one of
the three rule sets. So the population genuinely disagrees — a third of them
learned that doors need keys, a third that doors always open, a third that
doors never open.

Two questions.

  DOES THE MERGE HOLD UP AS N GROWS? Merge 2, 4, 8, 16 instances and test the
      result on all three rule sets. If quality collapses with N, merging
      only works for small groups and the cold-start story needs rethinking.

  IS A MERGE A BETTER STARTING POINT THAN NOTHING? This is the actual
      product question. Take a fresh instance, seed it either from random
      weights or from the population merge, then let it live a NEW life and
      measure how fast it learns. A merge is only useful if it accelerates
      the newcomer.

That second test is the one that matters. A merge that scores well on the
lives it came from but does not help a newcomer is a curiosity.

Merging methods, since two seeds is not enough to distinguish them: plain
averaging only, which the two-instance run showed to be as good as Fisher
weighting or rehearsal repair.
"""

import copy
import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from world import GridWorld, ACTIONS, EVENTS, DELTA, LOCKED, RULES


POPULATION = 16
MERGE_SIZES = [2, 4, 8, 16]
LIFE_STEPS = 20000
NEW_LIFE_STEPS = 6000       # how long the newcomer gets
EPISODE = 200
HIDDEN = 128
LR = 3e-4
SEEDS = [0, 1]
N_TERRAIN = 5
PATCH = 9
TEST_STEPS = 5000
CHECKPOINTS = [500, 1000, 2000, 4000, 6000]


def obs_dim():
    """Terrain patch and action. has_key excluded, so the keyed rule set
    genuinely requires memory."""
    return PATCH * N_TERRAIN + len(ACTIONS)


def encode(obs, action):
    v = torch.zeros(obs_dim())
    for i, cell in enumerate(obs["patch"]):
        v[i * N_TERRAIN + cell] = 1.0
    v[PATCH * N_TERRAIN + ACTIONS.index(action)] = 1.0
    return v.unsqueeze(0)


class GRU(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.enc = nn.Linear(obs_dim(), HIDDEN)
        self.cell = nn.GRUCell(HIDDEN, HIDDEN)
        self.head = nn.Linear(HIDDEN, len(EVENTS))

    def forward(self, x, h=None):
        e = F.relu(self.enc(x))
        h = self.cell(e, h)
        return self.head(h), h


def walk(layout_seed, walk_seed, rules, n):
    """A stream, with locked-door arrivals marked — the only moments where
    the rule sets disagree."""
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


def train(net, data, opt=None):
    if opt is None:
        opt = torch.optim.Adam(net.parameters(), lr=LR)
    h = None
    net.train()
    for i, item in enumerate(data):
        if i % EPISODE == 0:
            h = None
        logits, h = net(encode(item[0], item[1]), h)
        loss = F.cross_entropy(
            logits, torch.tensor([EVENTS.index(item[2])]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        h = h.detach()
    return net


def evaluate(net, test):
    """Accuracy at locked doors, the contested moments."""
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


def merge_many(nets):
    """Plain average of N weight sets.

    The two-instance run showed Fisher weighting and rehearsal repair are
    within 1.2 points of this, so the simplest method is used.
    """
    out = copy.deepcopy(nets[0])
    states = [n.state_dict() for n in nets]
    merged = {k: sum(s[k] for s in states) / len(states)
              for k in states[0]}
    out.load_state_dict(merged)
    return out


def learning_curve(net, data, test, checkpoints):
    """How fast does this network pick up a life it has never seen?

    The product question: a merge is only useful if it makes a NEWCOMER
    learn faster than starting blank.
    """
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    out = {}
    done = 0
    for cp in checkpoints:
        chunk = data[done:cp]
        if chunk:
            train(net, chunk, opt)
        done = cp
        out[cp] = evaluate(net, test)
    return out


if __name__ == "__main__":
    print(f"Does merging survive many instances?")
    print(f"{POPULATION} instances, each with its own layout, its own "
          f"wander,\nand one of {RULES}\n")

    def mean(xs):
        xs = [x for x in xs if x is not None and x == x]
        return sum(xs) / len(xs) if xs else float("nan")

    quality = {n: [] for n in MERGE_SIZES}
    solo_scores = []
    curves = {"blank": [], "merged": []}

    for seed in SEEDS:
        # A population. Same initialisation for everyone, which is what
        # makes merging possible and is automatic in a shipped product.
        pop = []
        rules_of = []
        for i in range(POPULATION):
            rules = RULES[i % len(RULES)]
            rules_of.append(rules)
            data = walk(1000 + i, seed * 1000 + i, rules, LIFE_STEPS)
            net = train(GRU(seed), data)
            pop.append(net)

        # Held-out tests, one per rule set, on layouts nobody trained on.
        tests = {r: walk(7000 + j, 80000 + seed, r, TEST_STEPS)
                 for j, r in enumerate(RULES)}

        solo_scores.append(mean([
            evaluate(pop[i], tests[rules_of[i]])
            for i in range(min(4, POPULATION))]))

        for n in MERGE_SIZES:
            m = merge_many(pop[:n])
            quality[n].append(
                {r: evaluate(m, tests[r]) for r in RULES})
            del m

        # THE PRODUCT TEST: a newcomer with a life nobody has lived.
        new_rules = RULES[0]
        new_data = walk(9999, seed * 77 + 5, new_rules, NEW_LIFE_STEPS)
        new_test = walk(9999, 88888 + seed, new_rules, TEST_STEPS)

        blank = GRU(seed + 999)
        curves["blank"].append(
            learning_curve(blank, new_data, new_test, CHECKPOINTS))
        del blank

        seeded = merge_many(pop)
        curves["merged"].append(
            learning_curve(seeded, new_data, new_test, CHECKPOINTS))
        del seeded

        for n in pop:
            del n
        print(f"  seed {seed} done", flush=True)

    print("\n" + "=" * 72)
    print("MERGE QUALITY AS THE POPULATION GROWS")
    print(f"{'merged':>8} " + "  ".join(f"{r:>10}" for r in RULES) +
          f" {'mean':>9}")
    print("-" * 72)
    for n in MERGE_SIZES:
        per = {r: mean([q[r] for q in quality[n]]) for r in RULES}
        print(f"{n:>8} " + "  ".join(f"{per[r]:>9.2f}%" for r in RULES) +
              f" {mean(list(per.values())):>8.2f}%")
    print("-" * 72)
    print(f"{'solo':>8} {mean(solo_scores):>9.2f}%   "
          f"(each instance on its own rule set, for reference)")
    print("=" * 72)

    print("\nDOES A MERGE HELP A NEWCOMER? "
          f"accuracy after N steps of a new life")
    print(f"{'steps':>8} {'from blank':>12} {'from merge':>12} "
          f"{'advantage':>11}")
    print("-" * 72)
    for cp in CHECKPOINTS:
        b = mean([c[cp] for c in curves["blank"]])
        m = mean([c[cp] for c in curves["merged"]])
        print(f"{cp:>8} {b:>11.2f}% {m:>11.2f}% {m - b:>+10.2f}")
    print("=" * 72)

    first = CHECKPOINTS[0]
    last = CHECKPOINTS[-1]
    b0 = mean([c[first] for c in curves["blank"]])
    m0 = mean([c[first] for c in curves["merged"]])
    bl = mean([c[last] for c in curves["blank"]])
    ml = mean([c[last] for c in curves["merged"]])

    print(f"\n  at {first} steps: merge is {m0 - b0:+.2f} ahead")
    print(f"  at {last} steps: merge is {ml - bl:+.2f} ahead")

    # how many steps does blank need to reach what the merge starts near
    reached = None
    for cp in CHECKPOINTS:
        if mean([c[cp] for c in curves["blank"]]) >= m0:
            reached = cp
            break
    if reached:
        print(f"  a blank instance needs {reached} steps to reach what a "
              f"seeded one\n  has after {first}")
    else:
        print(f"  a blank instance never reaches the seeded one's "
              f"{first}-step score\n  within {last} steps")

    print("""
Two results, and the second is the one that matters commercially.

  MERGE QUALITY AS N GROWS. If the mean holds steady from 2 to 16, merging
      scales and a new user can be seeded from the whole population. If it
      falls, averaging many diverged networks produces mush and the cold
      start needs a different answer — most likely distillation, where an
      old instance TEACHES a new one rather than being averaged into it.

  THE NEWCOMER CURVE. A merge that scores well on the lives it came from but
      does not accelerate a newcomer is a curiosity. The advantage at the
      EARLIEST checkpoint is the cold-start number: it is how much better
      than nothing the software is on day one. If that advantage vanishes by
      the last checkpoint, the merge is a head start rather than a
      permanent gain, which is exactly what the cold start needs.

Note the newcomer lives a "keyed" life, which requires memory and is the
hardest of the three. A third of the population lived that rule set, so the
merge contains relevant experience diluted by two thirds of contradictory
experience — a realistic version of the problem.
""")
    with open("population.json", "w") as f:
        json.dump(dict(quality={str(k): v for k, v in quality.items()},
                       curves=curves), f, indent=2, default=str)
    print("wrote population.json")