"""Does the network use all its units, or do a few carry the signal?

Stage 3 of the build plan is dynamic compute routing — activating only the
part of the network a given input needs, the way a brain fires a fraction of
its neurons per thought and runs on twenty watts.

Routing only pays if there is something to skip. So before building any
routing machinery, measure whether the network is already concentrating its
work in a few units, or spreading it across all of them.

What is measured, on trained networks at several widths:

  dead units        never meaningfully active on any input. Free to remove.
  concentration     what fraction of units carry 90% of the total activity.
                    Low means a few units do the work.
  input specificity how much the ACTIVE SET changes between inputs. This is
                    the one that matters for routing: a network where every
                    input lights up the same units has nothing to route,
                    even if most units are dead. Routing needs DIFFERENT
                    inputs to need DIFFERENT parts.
  ablation          zero the least-active half and measure what accuracy
                    costs. The direct test of whether the quiet units are
                    doing anything.

Widths 64 through 512, because the answer probably depends on size — a small
network may be forced to use everything while a larger one has room to
specialise. If concentration rises with width, routing pays at scale and
Stage 3 is worth building. If it stays flat, the network genuinely needs all
of itself and routing buys nothing here.
"""

import json
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from bigworld import BigWorld, ACTIONS, EVENTS, FORKS


WIDTHS = [64, 128, 256, 512]
STEPS = 40000
EPISODE = 400
LR = 3e-4
SEEDS = [0, 1]
N_TERRAIN = 14
PATCH = 25
TEST_STEPS = 6000
ACTIVE_THRESHOLD = 0.01     # below this counts as inactive

FORK_OF = {}
for name, (yes, no) in FORKS.items():
    for e in yes + no:
        FORK_OF[e] = name


def obs_dim():
    return PATCH * N_TERRAIN + len(ACTIONS)


def encode(obs, action):
    v = torch.zeros(obs_dim())
    for i, cell in enumerate(obs["patch"]):
        v[i * N_TERRAIN + cell] = 1.0
    v[PATCH * N_TERRAIN + ACTIONS.index(action)] = 1.0
    return v.unsqueeze(0)


class GRU(nn.Module):
    """Exposes the hidden activations so they can be inspected."""

    def __init__(self, hidden, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.hidden = hidden
        self.enc = nn.Linear(obs_dim(), hidden)
        self.cell = nn.GRUCell(hidden, hidden)
        self.head = nn.Linear(hidden, len(EVENTS))

    def forward(self, x, h=None, mask=None):
        e = F.relu(self.enc(x))
        h_new = self.cell(e, h)
        if mask is not None:
            h_new = h_new * mask
        return self.head(h_new), h_new


def walk(layout_seed, walk_seed, n):
    rng = random.Random(walk_seed)
    world = BigWorld(layout_seed)
    out = []
    for i in range(n):
        if i % EPISODE == 0:
            world.reset()
        obs = world.observe(hide_state=True)
        action = rng.choice(ACTIONS)
        _, event = world.step(action)
        out.append((obs, action, event))
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


def evaluate(net, test, mask=None):
    net.eval()
    h = None
    hit = seen = 0
    with torch.no_grad():
        for i, item in enumerate(test):
            if i % EPISODE == 0:
                h = None
            logits, h = net(encode(item[0], item[1]), h, mask)
            hit += int(logits.argmax(1).item()) == EVENTS.index(item[2])
            seen += 1
    return 100.0 * hit / seen


def profile(net, test, sample=3000):
    """Collect hidden activations and describe how the work is distributed.

    The key quantity is not how many units are quiet — it is whether
    DIFFERENT inputs use DIFFERENT units. A network where every input
    lights up the same half has nothing to route.
    """
    net.eval()
    h = None
    acts = []
    with torch.no_grad():
        for i, item in enumerate(test[:sample]):
            if i % EPISODE == 0:
                h = None
            _, h = net(encode(item[0], item[1]), h)
            acts.append(h.abs().squeeze(0).clone())
    A = torch.stack(acts)                       # (n, hidden)

    mean_act = A.mean(0)                        # per-unit average activity
    n_units = mean_act.numel()

    # dead: essentially never active
    dead = int((mean_act < ACTIVE_THRESHOLD).sum())

    # concentration: how many units hold 90% of the total activity
    sorted_act, _ = mean_act.sort(descending=True)
    cum = sorted_act.cumsum(0) / sorted_act.sum()
    n90 = int((cum < 0.90).sum()) + 1
    concentration = 100.0 * n90 / n_units

    # input specificity: per input, which units are in the top quarter?
    # Then how much do those sets differ between inputs?
    k = max(1, n_units // 4)
    top = A.topk(k, dim=1).indices                # (n, k)
    onehot = torch.zeros(A.shape[0], n_units)
    onehot.scatter_(1, top, 1.0)
    usage = onehot.mean(0)                        # how often each unit is
    # a unit used in every input's top set is not routable; one used in
    # half is. Variance of usage across units measures that.
    specificity = float(usage.std())
    # fraction of units that are in the top set for almost every input
    always_on = float((usage > 0.9).float().mean())

    return dict(dead=dead, n_units=n_units,
                concentration=concentration,
                specificity=specificity,
                always_on=100.0 * always_on,
                mean_act=mean_act)


def ablate(net, test, mean_act, fraction):
    """Zero the least-active `fraction` of units and see what it costs."""
    n = mean_act.numel()
    k = int(n * fraction)
    if k == 0:
        return evaluate(net, test)
    order = mean_act.argsort()                    # quietest first
    mask = torch.ones(1, n)
    mask[0, order[:k]] = 0.0
    return evaluate(net, test, mask)


if __name__ == "__main__":
    print("Does the network use all its units?")
    print(f"widths {WIDTHS}, {STEPS} steps, {len(SEEDS)} seeds\n")

    def mean(xs):
        return sum(xs) / len(xs)

    results = {}
    for hidden in WIDTHS:
        runs = []
        for seed in SEEDS:
            tr = walk(seed, seed * 100 + 1, STEPS)
            te = walk(seed, 90000 + seed, TEST_STEPS)
            net = train(GRU(hidden, seed), tr)
            base = evaluate(net, te)
            p = profile(net, te)
            abl = {f: ablate(net, te, p["mean_act"], f)
                   for f in [0.25, 0.5, 0.75, 0.9]}
            runs.append(dict(base=base, **{k: v for k, v in p.items()
                                           if k != "mean_act"},
                             abl=abl))
            del net
        results[hidden] = runs
        r0 = runs[0]
        print(f"  {hidden:>4} units: acc {mean([r['base'] for r in runs]):5.2f}%  "
              f"dead {mean([r['dead'] for r in runs]):5.1f}  "
              f"90% of activity in {mean([r['concentration'] for r in runs]):5.1f}% of units  "
              f"always-on {mean([r['always_on'] for r in runs]):5.1f}%",
              flush=True)

    print("\n" + "=" * 92)
    print("HOW THE WORK IS DISTRIBUTED")
    print(f"{'width':>7} {'accuracy':>9} {'dead':>7} "
          f"{'90% activity in':>16} {'always-on':>11} {'specificity':>12}")
    print("-" * 92)
    for hidden in WIDTHS:
        runs = results[hidden]
        print(f"{hidden:>7} {mean([r['base'] for r in runs]):>8.2f}% "
              f"{mean([r['dead'] for r in runs]):>6.1f} "
              f"{mean([r['concentration'] for r in runs]):>15.1f}% "
              f"{mean([r['always_on'] for r in runs]):>10.1f}% "
              f"{mean([r['specificity'] for r in runs]):>12.4f}")
    print("=" * 92)

    print("\nWHAT REMOVING THE QUIETEST UNITS COSTS")
    print(f"{'width':>7} {'full':>8} " +
          "  ".join(f"{'-' + str(int(f * 100)) + '%':>8}"
                    for f in [0.25, 0.5, 0.75, 0.9]))
    print("-" * 92)
    for hidden in WIDTHS:
        runs = results[hidden]
        cells = "  ".join(
            f"{mean([r['abl'][f] for r in runs]):>7.2f}%"
            for f in [0.25, 0.5, 0.75, 0.9])
        print(f"{hidden:>7} {mean([r['base'] for r in runs]):>7.2f}% {cells}")
    print("=" * 92)

    print("\nHALF-ABLATION COST BY WIDTH (the routing headroom):")
    for hidden in WIDTHS:
        runs = results[hidden]
        b = mean([r["base"] for r in runs])
        a = mean([r["abl"][0.5] for r in runs])
        print(f"  {hidden:>4}: {b:.2f}% -> {a:.2f}%  ({a - b:+.2f})")

    print("""
Three things to read, and the third is the one that decides Stage 3.

  DEAD UNITS are free to remove but say nothing about routing. A network can
      have half its units dead and still be unroutable, if the live half is
      used identically by every input.

  ABLATION COST says how much slack there is. If removing the quietest half
      costs little, the network is not using its capacity and could simply
      be smaller — which is cheaper than routing and was already found to
      work (128 to 64 was free on weather).

  ALWAYS-ON is the routing test. It is the share of units that are in the
      active set for almost every input. Routing requires different inputs
      to need different parts, so:

        always-on high -> nothing to route. Every input needs the same
            units, and dynamic routing would skip nothing. Stage 3 does not
            pay at this scale and probably not at any scale for this
            architecture.

        always-on low, and falling with width -> larger networks specialise,
            different inputs use different units, and routing has something
            to skip. Stage 3 is worth building when the model is big.

If concentration and always-on are flat across a 8x width range, that is a
strong signal that this architecture spreads work evenly by nature and
routing is the wrong lever.
""")
    with open("sparsity.json", "w") as f:
        json.dump({str(k): [{kk: vv for kk, vv in r.items()
                             if kk != "mean_act"} for r in v]
                   for k, v in results.items()}, f, indent=2, default=str)
    print("wrote sparsity.json")