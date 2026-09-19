"""Merge instances that did NOT start from the same weights.

Merging works, compounds across generations, and answers the cold start. But
it has one hard requirement: every instance must descend from the same
initialisation. That is fine on day one — you ship one thing and everyone
starts identical. It fails the moment you ship version 2. Every upgrade would
strand all the learning from every user.

Distillation was the obvious fix and it failed: the student converged to the
teacher's ceiling (65.99% against the teacher's 67.43%) while a student
learning alone reached 80.08%.

THE KNOWN ANSWER IS GIT RE-BASIN. Two networks from different starting points
look incompatible because their hidden units are in a different ORDER — unit
7 in one does what unit 34 does in the other. Averaging mixes unrelated
things. Permute one network so matching units line up, and they merge. The
permutation does not change what the network computes at all; it is the same
function with the units relabelled.

The literature reports that loss landscapes contain nearly a single basin
once permutation symmetries are accounted for, and that weight matching finds
the permutation in seconds.

FIXED FROM THE FIRST VERSION, which the sanity check caught. Permuting must
move BOTH SIDES of any weight matrix whose rows and columns both index
permuted quantities:

  enc.weight      rows only     (columns are the raw observation)
  enc.bias        entries
  cell.weight_ih  rows AND columns — columns index the encoder output,
                  which has itself just been permuted
  cell.weight_hh  rows AND columns — columns index the hidden state
  cell biases     entries, in three gate blocks
  head.weight     columns only  (rows are the output classes)

The first version permuted only the rows of weight_ih, so the network no
longer computed the same function and every merge number after it was void.

Arms:
  solo           each network on its own life. The per-network ceiling.
  same-init      both from identical weights, averaged. The existing method.
  diff-naive     different initialisations, averaged directly. The
                 version-change problem.
  diff-rebasin   different initialisations, permuted then averaged. The
                 claim.
"""

import copy
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from world import GridWorld, ACTIONS, EVENTS, DELTA, LOCKED


LIFE_A = "keyed"
LIFE_B = "open"
STEPS = 25000
EPISODE = 200
HIDDEN = 128
LR = 3e-4
SEEDS = [0, 1, 2]
N_TERRAIN = 5
PATCH = 9
TEST_STEPS = 5000
MATCH_ROUNDS = 20


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


# ---------- the permutation machinery ----------

def assign(cost):
    """Find the permutation maximising total cost.

    Uses scipy's Hungarian algorithm when available; otherwise a greedy
    fallback, which is close enough for a first test.
    """
    try:
        from scipy.optimize import linear_sum_assignment
        _, col = linear_sum_assignment(-cost)
        return col
    except ImportError:
        n = cost.shape[0]
        perm = np.full(n, -1, dtype=int)
        c = cost.copy()
        for _ in range(n):
            i, j = np.unravel_index(np.argmax(c), c.shape)
            perm[i] = j
            c[i, :] = -np.inf
            c[:, j] = -np.inf
        return perm


def blocks_of(w, h):
    """A GRU's stacked weights are three gate blocks of h rows each."""
    return [w[i * h:(i + 1) * h] for i in range(3)]


def find_permutation(a, b, h=HIDDEN, rounds=MATCH_ROUNDS):
    """Which unit of b corresponds to which unit of a?

    Maximises the alignment of every weight touching the hidden units. The
    recurrent matrices carry the permutation on both sides, so the current
    guess is applied to the columns while solving for the rows — coordinate
    descent, as in the weight-matching algorithm.
    """
    sa = {k: v.detach().numpy() for k, v in a.state_dict().items()}
    sb = {k: v.detach().numpy() for k, v in b.state_dict().items()}

    perm = np.arange(h)
    for _ in range(rounds):
        cost = np.zeros((h, h))

        # encoder rows and bias
        cost += sa["enc.weight"] @ sb["enc.weight"].T
        cost += np.outer(sa["enc.bias"], sb["enc.bias"])

        # readout columns
        cost += sa["head.weight"].T @ sb["head.weight"]

        # gate biases
        for name in ["cell.bias_ih", "cell.bias_hh"]:
            for ba, bb in zip(blocks_of(sa[name], h), blocks_of(sb[name], h)):
                cost += np.outer(ba, bb)

        # both recurrent matrices have the permutation on rows AND columns,
        # so apply the current guess to the columns and solve for the rows
        for name in ["cell.weight_ih", "cell.weight_hh"]:
            for ga, gb in zip(blocks_of(sa[name], h), blocks_of(sb[name], h)):
                cost += ga @ gb[:, perm].T

        new = assign(cost)
        if np.array_equal(new, perm):
            break
        perm = new
    return perm


def permute(net, perm, h=HIDDEN):
    """Relabel the hidden units.

    The network computes exactly the same function afterwards. Every place a
    hidden-unit index appears has to move together, which is what the first
    version got wrong.
    """
    out = copy.deepcopy(net)
    s = {k: v.clone() for k, v in net.state_dict().items()}
    p = torch.tensor(perm, dtype=torch.long)

    # encoder output is indexed by hidden unit: rows only
    s["enc.weight"] = s["enc.weight"][p]
    s["enc.bias"] = s["enc.bias"][p]

    # readout reads from hidden units: columns only
    s["head.weight"] = s["head.weight"][:, p]

    # weight_ih rows are gate blocks; its COLUMNS index the encoder output,
    # which was just permuted, so both sides move
    blks = blocks_of(s["cell.weight_ih"], h)
    s["cell.weight_ih"] = torch.cat([b[p][:, p] for b in blks], dim=0)

    # weight_hh rows are gate blocks; its COLUMNS index the hidden state,
    # so both sides move here too
    blks = blocks_of(s["cell.weight_hh"], h)
    s["cell.weight_hh"] = torch.cat([b[p][:, p] for b in blks], dim=0)

    for name in ["cell.bias_ih", "cell.bias_hh"]:
        blks = blocks_of(s[name], h)
        s[name] = torch.cat([b[p] for b in blks], dim=0)

    out.load_state_dict(s)
    return out


# ---------- the experiment ----------

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


def average(a, b, alpha=0.5):
    out = copy.deepcopy(a)
    sa, sb = a.state_dict(), b.state_dict()
    out.load_state_dict({k: alpha * sa[k] + (1 - alpha) * sb[k]
                         for k in sa})
    return out


if __name__ == "__main__":
    print("Can instances that started differently be merged?")
    print(f"life A = {LIFE_A}, life B = {LIFE_B}, {STEPS} steps each\n")

    def mean(xs):
        xs = [x for x in xs if x is not None and x == x]
        return sum(xs) / len(xs) if xs else float("nan")

    rows = {}
    sanity = []

    for seed in SEEDS:
        tr_a = walk(10, seed * 100 + 1, LIFE_A, STEPS)
        tr_b = walk(10, seed * 100 + 2, LIFE_B, STEPS)
        te_a = walk(10, 90000 + seed, LIFE_A, TEST_STEPS)
        te_b = walk(10, 91000 + seed, LIFE_B, TEST_STEPS)

        a_same = train(GRU(seed), tr_a)
        b_same = train(GRU(seed), tr_b)
        a_diff = train(GRU(seed), tr_a)
        b_diff = train(GRU(seed + 7777), tr_b)

        rows.setdefault("solo", []).append(
            (evaluate(a_diff, te_a), evaluate(b_diff, te_b)))

        m = average(a_same, b_same)
        rows.setdefault("same-init", []).append(
            (evaluate(m, te_a), evaluate(m, te_b)))
        del m

        m = average(a_diff, b_diff)
        rows.setdefault("diff-naive", []).append(
            (evaluate(m, te_a), evaluate(m, te_b)))
        del m

        perm = find_permutation(a_diff, b_diff)
        b_perm = permute(b_diff, perm)

        # A permutation relabels units without changing the function, so the
        # permuted network MUST score exactly what it scored before. This is
        # what caught the first version's bug.
        before = evaluate(b_diff, te_b)
        after = evaluate(b_perm, te_b)
        sanity.append(abs(before - after))

        m = average(a_diff, b_perm)
        rows.setdefault("diff-rebasin", []).append(
            (evaluate(m, te_a), evaluate(m, te_b)))
        del m, b_perm

        n_moved = int((perm != np.arange(HIDDEN)).sum())
        print(f"  seed {seed}: permutation moved {n_moved}/{HIDDEN} units, "
              f"function changed by {sanity[-1]:.4f} points", flush=True)

        del a_same, b_same, a_diff, b_diff

    print(f"\n  permutation sanity: worst change to the permuted network's "
          f"own score is {max(sanity):.4f} points")
    void = max(sanity) > 0.5
    if void:
        print("  WARNING: permuting changed what the network computes, so "
              "the\n  implementation is still wrong and everything below is "
              "void.")
    else:
        print("  good: the permutation is function-preserving, so the merge "
              "numbers are real")

    ORDER = ["solo", "same-init", "diff-naive", "diff-rebasin"]
    print("\n" + "=" * 66)
    print(f"{'method':>14} {'on life A':>11} {'on life B':>11} {'mean':>9}")
    print("-" * 66)
    for name in ORDER:
        a = mean([r[0] for r in rows[name]])
        b = mean([r[1] for r in rows[name]])
        print(f"{name:>14} {a:>10.2f}% {b:>10.2f}% {(a + b) / 2:>8.2f}%")
    print("=" * 66)

    if not void:
        naive = mean([x for r in rows["diff-naive"] for x in r])
        rebas = mean([x for r in rows["diff-rebasin"] for x in r])
        same = mean([x for r in rows["same-init"] for x in r])
        print(f"\n  different inits, averaged directly : {naive:.2f}%")
        print(f"  different inits, permuted first    : {rebas:.2f}%  "
              f"({rebas - naive:+.2f})")
        print(f"  same init (the existing method)    : {same:.2f}%")
        if same - naive > 1:
            print(f"\n  re-basin recovers "
                  f"{100.0 * (rebas - naive) / (same - naive):.0f}% of the "
                  f"gap between naive cross-init\n  merging and same-init "
                  f"merging")

    print("""
The version-change problem is that merging needs a shared initialisation, so
an upgrade strands every user's learning.

  diff-rebasin near same-init -> SOLVED. Permuting units before averaging
      lets instances from different starting points merge, so a new version
      can inherit from the old one and nothing is lost on upgrade.

  diff-rebasin near diff-naive -> the permutation did not find the
      correspondence. Plausible for a recurrent network: the literature's
      results are on feedforward and residual nets, and a GRU's three coupled
      gate blocks are a harder matching problem. The next thing would be
      REPAIR, which renormalises activation statistics after permuting,
      since averaging permuted networks is known to disturb them.

Read the sanity line first. If the permutation is not function-preserving,
nothing below it means anything.
""")
    with open("rebasin.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print("wrote rebasin.json")