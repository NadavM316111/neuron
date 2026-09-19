"""Fix the other half of the merge gap, and try merging across sizes.

Re-basin closed 51% of the gap between naive cross-init merging (32.97%) and
same-init merging (80.11%), reaching 56.93%. The literature says alignment
alone is not enough, and names the reason: VARIANCE COLLAPSE. Averaging two
networks squashes each unit's activity toward the middle — the peaks cancel
— so units end up quieter than in either parent and downstream layers get a
weaker signal than they were trained on.

REPAIR rescales the merged network's units so their statistics match the
parents'. Reported to give 60-100% relative barrier reduction on top of
alignment.

Implemented here in two places, which is everywhere the hidden units appear:

  encoder preactivation   folded directly into enc.weight and enc.bias, so
                          it is exact and free at run time
  hidden state            applied as an explicit per-unit scale and shift,
                          because h feeds back into itself and cannot be
                          folded into a weight

The target statistics are the interpolation of the two parents' statistics,
not the statistics of the interpolated network — that distinction is the
whole point.

ALSO TESTED: merging across a SIZE change. A 128-unit network and a 64-unit
one cannot be averaged, since the shapes do not match. But the matching
machinery can pick WHICH 64 of the teacher's 128 units correspond to the
student's 64, and merge into those. Distillation already failed at this, so
if it works it is the only route across an architecture change.

Arms:
  solo            each network on its own life
  same-init       the existing method
  diff-naive      different inits, averaged
  diff-rebasin    permuted, then averaged
  diff-repair     permuted, averaged, then rescaled. The claim.
  size-naive      128 and 64, subset-averaged without alignment
  size-rebasin    128 and 64, aligned then subset-averaged
  size-repair     the same plus rescaling
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
BIG = 128
SMALL = 64
LR = 3e-4
SEEDS = [0, 1, 2]
N_TERRAIN = 5
PATCH = 9
TEST_STEPS = 5000
STAT_STEPS = 4000        # how much data the statistics are measured on
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
    """Carries optional per-unit rescaling of the hidden state, which is
    what REPAIR needs and what cannot be folded into a weight."""

    def __init__(self, hidden, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.hidden = hidden
        self.enc = nn.Linear(obs_dim(), hidden)
        self.cell = nn.GRUCell(hidden, hidden)
        self.head = nn.Linear(hidden, len(EVENTS))
        self.register_buffer("h_scale", torch.ones(hidden))
        self.register_buffer("h_shift", torch.zeros(hidden))

    def forward(self, x, h=None, collect=None):
        pre = self.enc(x)
        if collect is not None:
            collect["enc"].append(pre.detach().squeeze(0).clone())
        e = F.relu(pre)
        h = self.cell(e, h)
        h = h * self.h_scale + self.h_shift
        if collect is not None:
            collect["h"].append(h.detach().squeeze(0).clone())
        return self.head(h), h


# ---------- alignment ----------

def assign(cost):
    try:
        from scipy.optimize import linear_sum_assignment
        row, col = linear_sum_assignment(-cost)
        return row, col
    except ImportError:
        n, m = cost.shape
        c = cost.copy()
        rows, cols = [], []
        for _ in range(min(n, m)):
            i, j = np.unravel_index(np.argmax(c), c.shape)
            rows.append(i)
            cols.append(j)
            c[i, :] = -np.inf
            c[:, j] = -np.inf
        order = np.argsort(rows)
        return np.array(rows)[order], np.array(cols)[order]


def blocks_of(w, h):
    return [w[i * h:(i + 1) * h] for i in range(3)]


def match_cost(a, b, ha, hb, perm_b):
    """Alignment score between every unit of a and every unit of b."""
    sa = {k: v.detach().numpy() for k, v in a.state_dict().items()}
    sb = {k: v.detach().numpy() for k, v in b.state_dict().items()}
    cost = np.zeros((ha, hb))
    cost += sa["enc.weight"] @ sb["enc.weight"].T
    cost += np.outer(sa["enc.bias"], sb["enc.bias"])
    cost += sa["head.weight"].T @ sb["head.weight"]
    for name in ["cell.bias_ih", "cell.bias_hh"]:
        for ba, bb in zip(blocks_of(sa[name], ha), blocks_of(sb[name], hb)):
            cost += np.outer(ba, bb)
    for name in ["cell.weight_ih", "cell.weight_hh"]:
        for ga, gb in zip(blocks_of(sa[name], ha), blocks_of(sb[name], hb)):
            cost += ga[:, :min(ha, hb)] @ gb[:, perm_b][:, :min(ha, hb)].T \
                if ha != hb else ga @ gb[:, perm_b].T
    return cost


def find_permutation(a, b, h, rounds=MATCH_ROUNDS):
    """Same-size matching: which unit of b corresponds to which unit of a."""
    perm = np.arange(h)
    for _ in range(rounds):
        _, new = assign(match_cost(a, b, h, h, perm))
        if np.array_equal(new, perm):
            break
        perm = new
    return perm


def permute(net, perm, h):
    """Relabel hidden units. Function-preserving: every place a hidden index
    appears must move, including BOTH sides of the recurrent matrices."""
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
    s["h_scale"] = s["h_scale"][p]
    s["h_shift"] = s["h_shift"][p]
    out.load_state_dict(s)
    return out


# ---------- REPAIR ----------

def stats(net, data):
    """Per-unit mean and standard deviation of the encoder preactivation
    and the hidden state."""
    net.eval()
    collect = {"enc": [], "h": []}
    h = None
    with torch.no_grad():
        for i, item in enumerate(data[:STAT_STEPS]):
            if i % EPISODE == 0:
                h = None
            _, h = net(encode(item[0], item[1]), h, collect)
    out = {}
    for k in ["enc", "h"]:
        A = torch.stack(collect[k])
        out[k] = (A.mean(0), A.std(0) + 1e-6)
    return out


def repair(merged, sa, sb, data, alpha=0.5):
    """Rescale so the merged network's unit statistics match the parents'.

    The target is the INTERPOLATION OF THE PARENTS' statistics, not the
    statistics of the interpolated network. Averaging weights squashes each
    unit's variance; this puts it back.
    """
    out = copy.deepcopy(merged)
    target = {k: (alpha * sa[k][0] + (1 - alpha) * sb[k][0],
                  alpha * sa[k][1] + (1 - alpha) * sb[k][1])
              for k in ["enc", "h"]}
    cur = stats(out, data)

    # encoder: fold the correction into the weights, which is exact
    scale = target["enc"][1] / cur["enc"][1]
    shift = target["enc"][0] - scale * cur["enc"][0]
    with torch.no_grad():
        out.enc.weight.mul_(scale.unsqueeze(1))
        out.enc.bias.mul_(scale).add_(shift)

    # hidden state: cannot be folded, because h feeds back into itself
    cur = stats(out, data)
    hs = target["h"][1] / cur["h"][1]
    hsh = target["h"][0] - hs * cur["h"][0]
    with torch.no_grad():
        out.h_scale.copy_(out.h_scale * hs)
        out.h_shift.copy_(out.h_shift * hs + hsh)
    return out


# ---------- merging ----------

def average(a, b, alpha=0.5):
    out = copy.deepcopy(a)
    sa, sb = a.state_dict(), b.state_dict()
    out.load_state_dict({k: alpha * sa[k] + (1 - alpha) * sb[k]
                         for k in sa})
    return out


def subset_merge(small, big, cols, alpha=0.5):
    """Merge a big network into a small one, using only the big network's
    units that correspond to the small one's.

    Averaging is impossible across sizes because the shapes differ. But if
    unit j of the big network does what unit i of the small one does, the
    big network's row j can be averaged into the small network's row i.
    """
    out = copy.deepcopy(small)
    ss = small.state_dict()
    sb = big.state_dict()
    hs, hb = small.hidden, big.hidden
    c = torch.tensor(cols, dtype=torch.long)
    new = {}

    new["enc.weight"] = alpha * ss["enc.weight"] + \
        (1 - alpha) * sb["enc.weight"][c]
    new["enc.bias"] = alpha * ss["enc.bias"] + \
        (1 - alpha) * sb["enc.bias"][c]
    new["head.weight"] = alpha * ss["head.weight"] + \
        (1 - alpha) * sb["head.weight"][:, c]
    new["head.bias"] = alpha * ss["head.bias"] + (1 - alpha) * sb["head.bias"]

    for name in ["cell.weight_ih", "cell.weight_hh"]:
        parts = []
        for blk_s, blk_b in zip(blocks_of(ss[name], hs),
                                blocks_of(sb[name], hb)):
            parts.append(alpha * blk_s + (1 - alpha) * blk_b[c][:, c])
        new[name] = torch.cat(parts, dim=0)
    for name in ["cell.bias_ih", "cell.bias_hh"]:
        parts = []
        for blk_s, blk_b in zip(blocks_of(ss[name], hs),
                                blocks_of(sb[name], hb)):
            parts.append(alpha * blk_s + (1 - alpha) * blk_b[c])
        new[name] = torch.cat(parts, dim=0)

    new["h_scale"] = ss["h_scale"]
    new["h_shift"] = ss["h_shift"]
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


if __name__ == "__main__":
    print("REPAIR, and merging across a size change\n")

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
        both = tr_a[:STAT_STEPS // 2] + tr_b[:STAT_STEPS // 2]

        a_same = train(GRU(BIG, seed), tr_a)
        b_same = train(GRU(BIG, seed), tr_b)
        a_diff = train(GRU(BIG, seed), tr_a)
        b_diff = train(GRU(BIG, seed + 7777), tr_b)
        b_small = train(GRU(SMALL, seed + 4242), tr_b)

        def rec(name, net):
            rows.setdefault(name, []).append(
                (evaluate(net, te_a), evaluate(net, te_b)))

        rec("solo", a_diff)
        rec("same-init", average(a_same, b_same))
        rec("diff-naive", average(a_diff, b_diff))

        perm = find_permutation(a_diff, b_diff, BIG)
        b_perm = permute(b_diff, perm, BIG)
        sanity.append(abs(evaluate(b_diff, te_b) - evaluate(b_perm, te_b)))
        merged = average(a_diff, b_perm)
        rec("diff-rebasin", merged)
        rec("diff-repair", repair(merged, stats(a_diff, both),
                                  stats(b_perm, both), both))
        del b_perm, merged

        # --- across a size change: small student, big teacher ---
        cost = match_cost(b_small, a_diff, SMALL, BIG, np.arange(BIG))
        _, cols = assign(cost)
        rec("size-naive", subset_merge(b_small, a_diff,
                                       np.arange(SMALL)))
        sm = subset_merge(b_small, a_diff, cols)
        rec("size-rebasin", sm)
        rec("size-repair", repair(sm, stats(b_small, both),
                                  stats(b_small, both), both))
        rows.setdefault("small-solo", []).append(
            (evaluate(b_small, te_a), evaluate(b_small, te_b)))
        del sm, a_same, b_same, a_diff, b_diff, b_small

        print(f"  seed {seed} done, permutation changed the function by "
              f"{sanity[-1]:.4f} points", flush=True)

    print(f"\n  permutation sanity: worst {max(sanity):.4f} points")
    if max(sanity) > 0.5:
        print("  WARNING: permutation is not function-preserving, results "
              "void.")

    ORDER = ["solo", "same-init", "diff-naive", "diff-rebasin",
             "diff-repair", "small-solo", "size-naive", "size-rebasin",
             "size-repair"]
    print("\n" + "=" * 66)
    print(f"{'method':>14} {'on life A':>11} {'on life B':>11} {'mean':>9}")
    print("-" * 66)
    for name in ORDER:
        a = mean([r[0] for r in rows[name]])
        b = mean([r[1] for r in rows[name]])
        print(f"{name:>14} {a:>10.2f}% {b:>10.2f}% {(a + b) / 2:>8.2f}%")
    print("=" * 66)

    g = lambda n: mean([x for r in rows[n] for x in r])
    naive, rebas, rep, same = (g("diff-naive"), g("diff-rebasin"),
                               g("diff-repair"), g("same-init"))
    span = same - naive
    print(f"\nDIFFERENT INITIALISATIONS, same size:")
    print(f"  naive average  {naive:6.2f}%")
    print(f"  + re-basin     {rebas:6.2f}%  "
          f"({100.0 * (rebas - naive) / span:+4.0f}% of the gap)")
    print(f"  + REPAIR       {rep:6.2f}%  "
          f"({100.0 * (rep - naive) / span:+4.0f}% of the gap)")
    print(f"  same-init      {same:6.2f}%  (the target)")

    print(f"\nACROSS A SIZE CHANGE (64-unit student, 128-unit teacher):")
    print(f"  student alone  {g('small-solo'):6.2f}%")
    print(f"  naive subset   {g('size-naive'):6.2f}%")
    print(f"  + re-basin     {g('size-rebasin'):6.2f}%")
    print(f"  + REPAIR       {g('size-repair'):6.2f}%")

    print("""
REPAIR addresses variance collapse: averaging two networks squashes each
unit's activity toward the middle, so units end up quieter than in either
parent and downstream layers receive a weaker signal than they were trained
on. Rescaling puts the variance back.

  REPAIR clearly above re-basin -> variance collapse was the remaining
      obstacle, and the version-change problem is substantially solved.

  no change -> variance collapse is not what is limiting a GRU merge. The
      literature's results are on convolutional and residual networks with
      normalisation layers, and a recurrent network without them may fail
      for a different reason.

The size-change block is the harder question, and merging cannot do it at all
without alignment — the shapes do not match. If size-rebasin beats the student
alone, an upgrade to a LARGER architecture can inherit, which distillation
could not deliver.
""")
    with open("repair.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print("wrote repair.json")