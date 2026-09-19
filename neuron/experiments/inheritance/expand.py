"""Merge across a size change by growing the small model, not shrinking the
big one.

The previous attempt shrank the teacher: it picked the 64 of its 128 units
that best matched the student's, and merged those. Nothing transferred (life
A went 59.09 -> 58.58) because half the teacher was thrown away before
merging.

The literature does it the other way. Width mismatch is considered the
manageable case, and the recipe is to transform both models into a COMMON
architecture first — CLAFusion, for instance, pads the smaller model with
identity-valued blocks to match the larger.

So: expand the 64-unit student to 128 units by adding 64 DEAD units whose
weights are all zero. A dead unit outputs nothing and is read by nothing, so
the expanded model computes exactly the same function as the original — a
sanity check verifies this. Then align the teacher to the expanded student
and average at 128.

Nothing is discarded. The teacher's knowledge has somewhere to land: the
dead slots. And the result is a 128-unit network, which is what an upgrade
would want anyway.

Arms:
  student-64        the small model on its own. What an upgrade would throw
                    away.
  student-expanded  the same model padded to 128. Must score identically.
  teacher-128       the big model on its own.
  expand-naive      expanded student averaged with the teacher, no
                    alignment.
  expand-rebasin    aligned first. The claim.
  shrink-rebasin    the previous approach, for comparison.
"""

import copy
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from world import GridWorld, ACTIONS, EVENTS, DELTA, LOCKED


LIFE_TEACHER = "keyed"     # the teacher's life, requires memory
LIFE_STUDENT = "open"      # the student's life
STEPS = 25000
EPISODE = 200
BIG = 128
SMALL = 64
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
    def __init__(self, hidden, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.hidden = hidden
        self.enc = nn.Linear(obs_dim(), hidden)
        self.cell = nn.GRUCell(hidden, hidden)
        self.head = nn.Linear(hidden, len(EVENTS))

    def forward(self, x, h=None):
        e = F.relu(self.enc(x))
        h = self.cell(e, h)
        return self.head(h), h


def blocks_of(w, h):
    return [w[i * h:(i + 1) * h] for i in range(3)]


# ---------- expansion ----------

def expand(net, new_h):
    """Grow a network to a larger width by adding dead units.

    A dead unit has zero incoming weights, zero bias, and is read by nothing,
    so it contributes nothing and the expanded network computes exactly the
    same function. Its slots are where a merged partner's knowledge can land.

    One subtlety: the GRU's update gate uses a sigmoid, so a zero
    preactivation gives 0.5, not 0. That would make dead units carry half
    their previous state forward. Their bias is therefore set strongly
    negative so the gates close and the unit stays at zero.
    """
    old_h = net.hidden
    out = GRU(new_h, seed=0)
    s = net.state_dict()
    new = {k: torch.zeros_like(v) for k, v in out.state_dict().items()}

    new["enc.weight"][:old_h] = s["enc.weight"]
    new["enc.bias"][:old_h] = s["enc.bias"]
    new["head.weight"][:, :old_h] = s["head.weight"]
    new["head.bias"] = s["head.bias"].clone()

    for name in ["cell.weight_ih", "cell.weight_hh"]:
        parts = []
        for blk in blocks_of(s[name], old_h):
            big = torch.zeros(new_h, new_h)
            big[:old_h, :old_h] = blk
            parts.append(big)
        new[name] = torch.cat(parts, dim=0)

    for name in ["cell.bias_ih", "cell.bias_hh"]:
        parts = []
        for gi, blk in enumerate(blocks_of(s[name], old_h)):
            big = torch.zeros(new_h)
            big[:old_h] = blk
            # gates are [reset, update, new]. Force the dead units' update
            # gate closed so they do not carry state forward through the
            # sigmoid's 0.5 default.
            if gi == 1 and name == "cell.bias_ih":
                big[old_h:] = -10.0
            parts.append(big)
        new[name] = torch.cat(parts, dim=0)

    out.load_state_dict(new)
    return out


# ---------- alignment ----------

def assign(cost):
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


def find_permutation(a, b, h, rounds=MATCH_ROUNDS):
    sa = {k: v.detach().numpy() for k, v in a.state_dict().items()}
    sb = {k: v.detach().numpy() for k, v in b.state_dict().items()}
    perm = np.arange(h)
    for _ in range(rounds):
        cost = np.zeros((h, h))
        cost += sa["enc.weight"] @ sb["enc.weight"].T
        cost += np.outer(sa["enc.bias"], sb["enc.bias"])
        cost += sa["head.weight"].T @ sb["head.weight"]
        for name in ["cell.bias_ih", "cell.bias_hh"]:
            for ba, bb in zip(blocks_of(sa[name], h), blocks_of(sb[name], h)):
                cost += np.outer(ba, bb)
        for name in ["cell.weight_ih", "cell.weight_hh"]:
            for ga, gb in zip(blocks_of(sa[name], h), blocks_of(sb[name], h)):
                cost += ga @ gb[:, perm].T
        new = assign(cost)
        if np.array_equal(new, perm):
            break
        perm = new
    return perm


def permute(net, perm, h):
    """Relabel hidden units. Both sides of the recurrent matrices move."""
    out = copy.deepcopy(net)
    s = {k: v.clone() for k, v in net.state_dict().items()}
    p = torch.tensor(perm, dtype=torch.long)
    s["enc.weight"] = s["enc.weight"][p]
    s["enc.bias"] = s["enc.bias"][p]
    s["head.weight"] = s["head.weight"][:, p]
    for name in ["cell.weight_ih", "cell.weight_hh"]:
        s[name] = torch.cat([b[p][:, p] for b in blocks_of(s[name], h)],
                            dim=0)
    for name in ["cell.bias_ih", "cell.bias_hh"]:
        s[name] = torch.cat([b[p] for b in blocks_of(s[name], h)], dim=0)
    out.load_state_dict(s)
    return out


def average(a, b, alpha=0.5):
    out = copy.deepcopy(a)
    sa, sb = a.state_dict(), b.state_dict()
    out.load_state_dict({k: alpha * sa[k] + (1 - alpha) * sb[k] for k in sa})
    return out


def shrink_merge(small, big, h_small, h_big, alpha=0.5):
    """The previous approach: pick the big model's best-matching units and
    merge into the small model, discarding the rest."""
    sa = {k: v.detach().numpy() for k, v in small.state_dict().items()}
    sb = {k: v.detach().numpy() for k, v in big.state_dict().items()}
    cost = sa["enc.weight"] @ sb["enc.weight"].T
    cost += sa["head.weight"].T @ sb["head.weight"]
    cols = assign(cost)[:h_small] if cost.shape[0] <= cost.shape[1] \
        else np.arange(h_small)
    c = torch.tensor(np.asarray(cols), dtype=torch.long)

    out = copy.deepcopy(small)
    ss, sB = small.state_dict(), big.state_dict()
    new = {}
    new["enc.weight"] = alpha * ss["enc.weight"] + \
        (1 - alpha) * sB["enc.weight"][c]
    new["enc.bias"] = alpha * ss["enc.bias"] + (1 - alpha) * sB["enc.bias"][c]
    new["head.weight"] = alpha * ss["head.weight"] + \
        (1 - alpha) * sB["head.weight"][:, c]
    new["head.bias"] = alpha * ss["head.bias"] + (1 - alpha) * sB["head.bias"]
    for name in ["cell.weight_ih", "cell.weight_hh"]:
        new[name] = torch.cat(
            [alpha * bs + (1 - alpha) * bb[c][:, c]
             for bs, bb in zip(blocks_of(ss[name], h_small),
                               blocks_of(sB[name], h_big))], dim=0)
    for name in ["cell.bias_ih", "cell.bias_hh"]:
        new[name] = torch.cat(
            [alpha * bs + (1 - alpha) * bb[c]
             for bs, bb in zip(blocks_of(ss[name], h_small),
                               blocks_of(sB[name], h_big))], dim=0)
    out.load_state_dict(new)
    return out


# ---------- experiment ----------

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
        loss = F.cross_entropy(logits,
                               torch.tensor([EVENTS.index(item[2])]))
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


if __name__ == "__main__":
    print("Merging across a size change: grow the small model, do not "
          "shrink the big one.\n")
    print(f"teacher lives '{LIFE_TEACHER}' at {BIG} units, "
          f"student lives '{LIFE_STUDENT}' at {SMALL}\n")

    def mean(xs):
        xs = [x for x in xs if x is not None and x == x]
        return sum(xs) / len(xs) if xs else float("nan")

    rows = {}
    expand_ok = []

    for seed in SEEDS:
        tr_t = walk(10, seed * 100 + 1, LIFE_TEACHER, STEPS)
        tr_s = walk(10, seed * 100 + 2, LIFE_STUDENT, STEPS)
        te_t = walk(10, 90000 + seed, LIFE_TEACHER, TEST_STEPS)
        te_s = walk(10, 91000 + seed, LIFE_STUDENT, TEST_STEPS)

        teacher = train(GRU(BIG, seed), tr_t)
        student = train(GRU(SMALL, seed + 4242), tr_s)

        def rec(name, net):
            rows.setdefault(name, []).append(
                (evaluate(net, te_t), evaluate(net, te_s)))

        rec("teacher-128", teacher)
        rec("student-64", student)

        grown = expand(student, BIG)
        rec("student-expanded", grown)
        # expansion must be function-preserving
        expand_ok.append(
            abs(evaluate(student, te_s) - evaluate(grown, te_s)))

        rec("expand-naive", average(grown, teacher))
        perm = find_permutation(grown, teacher, BIG)
        rec("expand-rebasin", average(grown, permute(teacher, perm, BIG)))
        rec("shrink-rebasin", shrink_merge(student, teacher, SMALL, BIG))

        del teacher, student, grown
        print(f"  seed {seed}: expansion changed the function by "
              f"{expand_ok[-1]:.4f} points", flush=True)

    print(f"\n  expansion sanity: worst {max(expand_ok):.4f} points")
    if max(expand_ok) > 0.5:
        print("  WARNING: padding changed what the network computes, so the "
              "expansion\n  is wrong and everything below is void.")

    ORDER = ["teacher-128", "student-64", "student-expanded",
             "shrink-rebasin", "expand-naive", "expand-rebasin"]
    print("\n" + "=" * 74)
    print(f"{'method':>18} {'teacher life':>14} {'student life':>14} "
          f"{'mean':>9}")
    print("-" * 74)
    for name in ORDER:
        a = mean([r[0] for r in rows[name]])
        b = mean([r[1] for r in rows[name]])
        print(f"{name:>18} {a:>13.2f}% {b:>13.2f}% {(a + b) / 2:>8.2f}%")
    print("=" * 74)

    st = mean([r[0] for r in rows["student-64"]])
    sh = mean([r[0] for r in rows["shrink-rebasin"]])
    ex = mean([r[0] for r in rows["expand-rebasin"]])
    tt = mean([r[0] for r in rows["teacher-128"]])

    print(f"\nDID THE TEACHER'S KNOWLEDGE TRANSFER?")
    print(f"  (accuracy on the TEACHER'S life, which the student never "
          f"lived)")
    print(f"  student alone      {st:6.2f}%")
    print(f"  shrink and merge   {sh:6.2f}%  ({sh - st:+.2f})")
    print(f"  expand and merge   {ex:6.2f}%  ({ex - st:+.2f})")
    print(f"  teacher itself     {tt:6.2f}%  (the ceiling)")
    if tt - st > 1:
        print(f"\n  expansion recovers "
              f"{100.0 * (ex - st) / (tt - st):.0f}% of what the teacher "
              f"knew")

    print("""
The teacher-life column is the whole question. The student never lived that
life, so anything above its solo score came from the merge.

  expand-rebasin well above student-64 -> knowledge crossed an architecture
      change. An upgrade to a larger model can inherit from the old one,
      which is the last unsolved piece of inheritance.

  expand-rebasin near student-64 -> growing the model gives the teacher
      somewhere to land but the average still cancels what it knew. The
      remaining option in the literature is representation-level fusion
      (ZipIt-style feature pairing) rather than weight averaging.

  expand-naive already works -> alignment was not the obstacle and the
      earlier failure was entirely about discarding half the teacher.

The student-expanded row is the check that matters most. Padding with dead
units must not change what the network computes. If that row differs from
student-64, the expansion is wrong and nothing else means anything.
""")
    with open("expand.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print("wrote expand.json")