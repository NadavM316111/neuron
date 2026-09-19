"""Can two lives be combined into one?

Stage 4 is inheritance. The version that matters for the vision is MERGING,
because it answers the cold start: software that arrives blank and is useless
for three weeks gets deleted in one. If instances that lived different lives
can be combined, a new user starts from the merge of everyone else's and then
diverges into their own.

BUILT ON world.py, NOT bigworld.py. The first two attempts used bigworld,
whose GridWorld accepts a `rules` argument and never reads it — the rule
variants only ever existed in world.py. So both lives were secretly identical
and the run reported a ceiling BELOW its floor. No earlier result is affected;
every phased experiment already used world.py.

  life A   "keyed" — a locked door opens only while carrying a key
  life B   "open"  — a locked door always opens, key or not

has_key is NOT given to the network, so the keyed life genuinely requires
memory and the two lives demand different answers at the same observation.

SCORED ONLY AT LOCKED DOORS, which is the only place the rules differ.
Everything else is identical in both lives and would dilute the measure to
noise — the mistake that made the first attempt unreadable.

Merge methods:
  average      the arithmetic mean of the two weight sets
  interpolate  a sweep along the line between them
  fisher       weight each parameter by how much each life cared about it,
               from squared gradients. The standard fix for naive averaging.
  rehearsed    average, then let the stability layer replay a little of each
               life to repair the damage

References:
  solo         each network on its own life. The per-life ceiling.
  cross        network A on life B. The floor.
  joint        one network trained on both lives interleaved. What a perfect
               merge would achieve.

Two validity checks run before anything is reported: the lives must actually
disagree at the door, and joint training must clearly beat the cross floor.
"""

import copy
import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from world import GridWorld, ACTIONS, EVENTS, DELTA, LOCKED, KEY, EMPTY, WALL
from stability import StabilityLayer


LIFE_A = "keyed"
LIFE_B = "open"
STEPS = 30000
EPISODE = 200
HIDDEN = 128
LR = 3e-4
SEEDS = [0, 1]
N_TERRAIN = 5
PATCH = 9
TEST_STEPS = 6000
REPAIR_STEPS = 4000
SEQ_LEN = 10
SEQ_COUNT = 2


def obs_dim():
    """Terrain patch and action only. has_key is deliberately excluded, so
    the keyed life requires memory."""
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


class Backend:
    """Wraps a network so the stability layer can repair a merge."""

    def __init__(self, net):
        self.net = net
        self.opt = torch.optim.Adam(net.parameters(), lr=LR)
        self.h = None
        self._saved = None

    def begin_sequence(self):
        self._saved = self.h
        self.h = None

    def _xy(self, item):
        return encode(item[0], item[1]), \
            torch.tensor([EVENTS.index(item[2])])

    def score(self, item):
        self.net.eval()
        with torch.no_grad():
            x, y = self._xy(item)
            logits, _ = self.net(x, self.h)
            return float(F.cross_entropy(logits, y).item()), None

    def update(self, item, steps):
        self.net.train()
        last = None
        for _ in range(steps):
            x, y = self._xy(item)
            logits, h = self.net(x, self.h)
            loss = F.cross_entropy(logits, y)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.h = h.detach()
            last = float(loss.item())
        return last

    def reset_state(self):
        self.h = None
        self._saved = None

    def snapshot(self):
        return {k: v.detach().clone()
                for k, v in self.net.state_dict().items()}

    def restore(self, state):
        self.net.load_state_dict(state)

    def on_rollback(self):
        for g in self.opt.param_groups:
            g["lr"] = max(1e-6, g["lr"] * 0.5)


def walk(layout_seed, walk_seed, rules, n):
    """A stream of (obs, action, event, at_door). at_door marks the moments
    where the two lives can disagree."""
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


def door_stats(rules, seeds=SEEDS, n=8000):
    """How often a locked door opens under these rules."""
    opened = total = 0
    for s in seeds:
        for _, _, e, at in walk(s, 5000 + s, rules, n):
            if at:
                total += 1
                opened += (e == "unlocked")
    return (100.0 * opened / total if total else 0.0), total


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


def fisher(net, data, n=3000):
    """How much this life cared about each parameter, from squared
    gradients. Large means load-bearing for this life."""
    f = {k: torch.zeros_like(v) for k, v in net.named_parameters()}
    h = None
    net.train()
    for i, item in enumerate(data[:n]):
        if i % EPISODE == 0:
            h = None
        logits, h = net(encode(item[0], item[1]), h)
        loss = F.cross_entropy(
            logits, torch.tensor([EVENTS.index(item[2])]))
        net.zero_grad()
        loss.backward()
        for k, v in net.named_parameters():
            if v.grad is not None:
                f[k] += v.grad.detach() ** 2
        h = h.detach()
    for k in f:
        f[k] /= max(1, min(n, len(data)))
    return f


def merge_average(a, b, alpha=0.5):
    out = copy.deepcopy(a)
    sa, sb = a.state_dict(), b.state_dict()
    out.load_state_dict({k: alpha * sa[k] + (1 - alpha) * sb[k]
                         for k in sa})
    return out


def merge_fisher(a, b, fa, fb, eps=1e-10):
    """Keep whichever life cared more about each parameter. A parameter
    critical to A and irrelevant to B should end near A's value, not
    halfway between."""
    out = copy.deepcopy(a)
    sa, sb = a.state_dict(), b.state_dict()
    merged = {}
    for k in sa:
        if k in fa and k in fb:
            wa, wb = fa[k] + eps, fb[k] + eps
            merged[k] = (wa * sa[k] + wb * sb[k]) / (wa + wb)
        else:
            merged[k] = 0.5 * sa[k] + 0.5 * sb[k]
    out.load_state_dict(merged)
    return out


def repair(net, data_a, data_b, seed):
    """Average, then replay a little of each life through the layer."""
    b = Backend(net)
    mixed = []
    for i in range(REPAIR_STEPS // 2):
        mixed.append(data_a[i % len(data_a)])
        mixed.append(data_b[i % len(data_b)])

    layer = StabilityLayer(
        b, canary=mixed[:40], seed=seed,
        window=200, warmup=30, top_fraction=0.50,
        coherence_veto=1e9, loss_floor=0.02, steps_per_update=1,
        rehearse_per_item=5, rehearse_count=SEQ_COUNT, rehearse_steps=1,
        anchor_size=100, buffer_size=500, sequence_len=SEQ_LEN,
        replay_policy="uniform",
        guard=True, guard_per_item=1000, canary_tolerance=0.5)

    for i, item in enumerate(mixed):
        if i % EPISODE == 0:
            b.reset_state()
        layer.observe(item)
    del layer, b
    return net


def evaluate(net, test):
    """Accuracy at locked doors only — the one place the lives differ."""
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
    print("Can two lives be combined into one?")
    print(f"life A = {LIFE_A}, life B = {LIFE_B}, scored at locked doors\n")

    pa, na = door_stats(LIFE_A)
    pb, nb = door_stats(LIFE_B)
    print(f"  life A: doors open {pa:5.1f}% of {na} arrivals")
    print(f"  life B: doors open {pb:5.1f}% of {nb} arrivals")
    if abs(pa - pb) < 15:
        print("\n  ABORT: the lives barely differ at the door, so there is\n"
              "  nothing for a merge to reconcile. Check that GridWorld is\n"
              "  actually reading its rules argument.")
        raise SystemExit
    print(f"  good: they disagree by {abs(pa - pb):.1f} points\n")

    def mean(xs):
        xs = [x for x in xs if x is not None and x == x]
        return sum(xs) / len(xs) if xs else float("nan")

    rows = {}
    interp = {}
    for seed in SEEDS:
        tr_a = walk(seed, seed * 100 + 1, LIFE_A, STEPS)
        tr_b = walk(seed, seed * 100 + 2, LIFE_B, STEPS)
        te_a = walk(seed, 90000 + seed, LIFE_A, TEST_STEPS)
        te_b = walk(seed, 91000 + seed, LIFE_B, TEST_STEPS)

        net_a = train(GRU(seed), tr_a)
        net_b = train(GRU(seed), tr_b)

        rows.setdefault("solo", []).append(
            (evaluate(net_a, te_a), evaluate(net_b, te_b)))
        rows.setdefault("cross", []).append(
            (evaluate(net_b, te_a), evaluate(net_a, te_b)))

        joint_data = []
        for i in range(STEPS // 2):
            joint_data.append(tr_a[i])
            joint_data.append(tr_b[i])
        net_j = train(GRU(seed + 500), joint_data)
        rows.setdefault("joint", []).append(
            (evaluate(net_j, te_a), evaluate(net_j, te_b)))
        del net_j

        m = merge_average(net_a, net_b)
        rows.setdefault("average", []).append(
            (evaluate(m, te_a), evaluate(m, te_b)))
        del m

        fa, fb = fisher(net_a, tr_a), fisher(net_b, tr_b)
        m = merge_fisher(net_a, net_b, fa, fb)
        rows.setdefault("fisher", []).append(
            (evaluate(m, te_a), evaluate(m, te_b)))
        del m, fa, fb

        m = repair(merge_average(net_a, net_b), tr_a, tr_b, seed)
        rows.setdefault("rehearsed", []).append(
            (evaluate(m, te_a), evaluate(m, te_b)))
        del m

        if seed == SEEDS[0]:
            for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
                mm = merge_average(net_a, net_b, alpha)
                interp[alpha] = (evaluate(mm, te_a), evaluate(mm, te_b))
                del mm

        del net_a, net_b
        print(f"  seed {seed} done", flush=True)

    ORDER = ["solo", "cross", "joint", "average", "fisher", "rehearsed"]

    print("\n" + "=" * 62)
    print(f"{'method':>12} {'on life A':>11} {'on life B':>11} {'mean':>9}")
    print("-" * 62)
    for name in ORDER:
        a = mean([r[0] for r in rows[name]])
        b = mean([r[1] for r in rows[name]])
        print(f"{name:>12} {a:>10.2f}% {b:>10.2f}% {(a + b) / 2:>8.2f}%")
    print("=" * 62)

    cross = mean([x for r in rows["cross"] for x in r])
    joint = mean([x for r in rows["joint"] for x in r])
    span = joint - cross

    print(f"\nfloor (a network on the OTHER life): {cross:.2f}%")
    print(f"ceiling (one network trained on both): {joint:.2f}%")

    if span < 5:
        print(f"\n  RUN IS VOID. The ceiling is only {span:+.2f} above the\n"
              f"  floor, so there is no span for a merge to recover.")
        raise SystemExit

    print(f"\nINTERPOLATION between the networks, seed 0:")
    print(f"{'alpha':>7} {'on life A':>11} {'on life B':>11} {'mean':>9}")
    print("-" * 42)
    for alpha in sorted(interp):
        a, b = interp[alpha]
        print(f"{alpha:>7.2f} {a:>10.2f}% {b:>10.2f}% {(a + b) / 2:>8.2f}%")
    print("  alpha 1.0 is pure A, 0.0 is pure B")

    print(f"\nHOW MUCH OF THE SPAN EACH MERGE RECOVERS:")
    for name in ["average", "fisher", "rehearsed"]:
        m = mean([x for r in rows[name] for x in r])
        print(f"  {name:>10}: {m:6.2f}%  "
              f"({100.0 * (m - cross) / span:+5.0f}% of the span)")

    print("""
The floor is a network tested on a life it never lived. The ceiling is one
network trained on both lives interleaved — what a perfect merge achieves.

  a merge near the ceiling -> inheritance works, and the cold start has an
      answer: a new instance begins from the merge of others rather than
      from nothing.

  merges near the floor -> combining lives destroys both, which is the
      documented behaviour for averaging diverged networks. Inheritance
      would then need distillation, where an old instance TEACHES a new one
      rather than being averaged into it.

  rehearsed beats average -> the stability layer repairs merge damage, a
      second use for something already built and measured.

The interpolation sweep matters independently. If the middle of the line is
worse than both ends, the two networks sit in different basins and no
averaging will ever work.
""")
    with open("merge.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print("wrote merge.json")