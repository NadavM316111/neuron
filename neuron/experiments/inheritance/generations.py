"""Does merged knowledge accumulate across generations, or decay?

Merging once works: a newcomer seeded from a 16-way merge reaches in 500
steps what a blank one takes 6,000 to reach. But a product with users has
GENERATIONS, not a single merge. The real cycle is:

  everyone lives a life  ->  merge them  ->  everyone starts from the merge
  and lives a NEW life   ->  merge again  ->  and so on

Two things could happen and they point opposite ways.

  ACCUMULATION. Each generation starts better informed, so it learns more in
      the same number of steps, so the next merge is better still. Knowledge
      compounds and the population gets collectively smarter over time. That
      is what the vision needs.

  DECAY. Repeated averaging is a low-pass filter. Each merge blurs whatever
      the individuals learned, and starting every generation from a blurred
      point means the blur compounds. The population converges to something
      bland that knows the shared structure and nothing else.

The second is the default expectation. Averaging is lossy, and doing it
repeatedly usually loses more.

MEASURED EACH GENERATION:
  merge quality     the merged network's accuracy on held-out tests from
                    every rule set
  newcomer speed    how fast a fresh instance seeded from this generation's
                    merge learns a life nobody has lived. This is the
                    product number and the one that must not decay.
  individual peak   how good the best individual gets, to see whether
                    individuals still learn or whether starting from a merge
                    traps them

A CONTROL runs alongside: instances that never merge, each living the same
total number of steps. If merging accumulates, the merged line should pull
ahead; if it decays, the never-merged control should overtake it.
"""

import copy
import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from world import GridWorld, ACTIONS, EVENTS, DELTA, LOCKED, RULES


GENERATIONS = 6
POPULATION = 8
LIFE_STEPS = 8000            # per instance per generation
NEWCOMER_STEPS = 1000        # how long the cold-start probe gets
EPISODE = 200
HIDDEN = 128
LR = 3e-4
SEEDS = [0, 1]
N_TERRAIN = 5
PATCH = 9
TEST_STEPS = 4000


def obs_dim():
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


def train(net, data):
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
    out = copy.deepcopy(nets[0])
    states = [n.state_dict() for n in nets]
    out.load_state_dict({k: sum(s[k] for s in states) / len(states)
                         for k in states[0]})
    return out


def newcomer_probe(seed_net, gen, seed):
    """How fast does a fresh instance learn a life nobody has lived?

    The product number. A new user arrives, is seeded from the current
    merge, and gets a short time to become useful. If this decays across
    generations, repeated merging is destroying the thing it exists to
    provide.
    """
    rules = RULES[0]
    data = walk(9000 + gen, 55000 + gen * 7 + seed, rules, NEWCOMER_STEPS)
    test = walk(9000 + gen, 66000 + gen * 7 + seed, rules, TEST_STEPS)
    net = copy.deepcopy(seed_net) if seed_net is not None \
        else GRU(seed * 100 + gen + 31)
    train(net, data)
    score = evaluate(net, test)
    del net
    return score


if __name__ == "__main__":
    print("Does merged knowledge accumulate across generations?")
    print(f"{GENERATIONS} generations, {POPULATION} instances, "
          f"{LIFE_STEPS} steps each per generation")
    print(f"cold-start probe gets {NEWCOMER_STEPS} steps\n")

    def mean(xs):
        xs = [x for x in xs if x is not None and x == x]
        return sum(xs) / len(xs) if xs else float("nan")

    history = {"merged_quality": [], "newcomer": [], "best_individual": [],
               "control_quality": [], "control_newcomer": []}

    for seed in SEEDS:
        tests = {r: walk(7000 + j, 80000 + seed, r, TEST_STEPS)
                 for j, r in enumerate(RULES)}

        # the merging population, and a control that never merges
        parent = None
        control = [GRU(seed * 50 + i) for i in range(POPULATION)]

        mq, nc, bi, cq, cn = [], [], [], [], []

        for gen in range(GENERATIONS):
            pop = []
            for i in range(POPULATION):
                rules = RULES[i % len(RULES)]
                data = walk(2000 + gen * 100 + i,
                            seed * 10000 + gen * 100 + i,
                            rules, LIFE_STEPS)
                net = copy.deepcopy(parent) if parent is not None \
                    else GRU(seed * 50 + i)
                pop.append(train(net, data))

                # the control lives the same life but never inherits
                control[i] = train(control[i], data)

            merged = merge_many(pop)
            mq.append(mean([evaluate(merged, tests[r]) for r in RULES]))
            bi.append(max(
                evaluate(pop[i], tests[RULES[i % len(RULES)]])
                for i in range(POPULATION)))
            nc.append(newcomer_probe(merged, gen, seed))

            cmerged = merge_many(control)
            cq.append(mean([evaluate(cmerged, tests[r]) for r in RULES]))
            cn.append(newcomer_probe(cmerged, gen, seed))
            del cmerged

            parent = merged
            for n in pop:
                del n
            print(f"  seed {seed} gen {gen}: merged {mq[-1]:.1f}%  "
                  f"newcomer {nc[-1]:.1f}%  best {bi[-1]:.1f}%", flush=True)

        history["merged_quality"].append(mq)
        history["newcomer"].append(nc)
        history["best_individual"].append(bi)
        history["control_quality"].append(cq)
        history["control_newcomer"].append(cn)
        del parent, control

    print("\n" + "=" * 84)
    print("ACROSS GENERATIONS")
    print(f"{'gen':>5} {'merged':>9} {'newcomer':>10} {'best indiv':>12} "
          f"{'ctrl merged':>13} {'ctrl newcomer':>15}")
    print("-" * 84)
    for g in range(GENERATIONS):
        print(f"{g:>5} "
              f"{mean([h[g] for h in history['merged_quality']]):>8.2f}% "
              f"{mean([h[g] for h in history['newcomer']]):>9.2f}% "
              f"{mean([h[g] for h in history['best_individual']]):>11.2f}% "
              f"{mean([h[g] for h in history['control_quality']]):>12.2f}% "
              f"{mean([h[g] for h in history['control_newcomer']]):>14.2f}%")
    print("=" * 84)

    def trend(key):
        first = mean([h[0] for h in history[key]])
        last = mean([h[-1] for h in history[key]])
        return first, last, last - first

    print("\nTREND FROM FIRST TO LAST GENERATION:")
    for key, label in [("merged_quality", "merge quality"),
                       ("newcomer", "newcomer speed"),
                       ("best_individual", "best individual"),
                       ("control_newcomer", "control newcomer")]:
        f, l, d = trend(key)
        print(f"  {label:>18}: {f:6.2f}% -> {l:6.2f}%  ({d:+6.2f})")

    nf, nl, nd = trend("newcomer")
    cf, cl, cd = trend("control_newcomer")
    print(f"\n  merging vs never merging, final generation: "
          f"{nl - cl:+.2f}")

    print("""
The newcomer row is the product number. It is how good a brand new user's
instance is after a short time, seeded from whatever the population has
collectively become.

  newcomer RISES across generations -> knowledge compounds. Each cohort
      starts better informed, learns more in the same time, and hands on
      more. That is the thing the vision needs and it is not the default
      expectation.

  newcomer FLAT -> merging preserves but does not accumulate. Still useful,
      since the cold start is solved, but the population does not get
      collectively smarter and there is no reason to keep merging beyond
      the first generation.

  newcomer FALLS -> repeated averaging is a low-pass filter and the blur
      compounds. The population converges to something bland. Merging would
      then need to happen once, from a diverse founding cohort, and never
      again.

Watch "best individual" alongside it. If merge quality holds but individuals
stop improving, starting every life from a merge is trapping them in the
consensus — the population would be stable and incapable of learning
anything new, which is a subtler failure than decay.
""")
    with open("generations.json", "w") as f:
        json.dump(history, f, indent=2, default=str)
    print("wrote generations.json")