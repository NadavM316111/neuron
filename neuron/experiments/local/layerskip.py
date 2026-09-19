"""Can a local rule skip layers that do not need updating?

The field's entire speed story for local learning is PARALLELISM. Backward
locking means no layer can update until the full forward-backward pass
completes, so the fix is to decouple layers and run them on separate
devices. Every paper measures multi-GPU throughput.

That is unavailable on one laptop. But it is not the only thing local
learning makes possible, and the other thing has not been measured.

THE ASYMMETRY THIS EXPLOITS. With backpropagation you must traverse EVERY
layer to update ANY layer, because the gradient reaching layer 1 comes
through layers 2..N. There is no way to skip layer 3 and still update layer
1. With a local rule each layer has its OWN loss, so each layer can decide
independently whether it needs updating at all.

THE HYPOTHESIS: on a real stream most inputs are ordinary. A layer that has
already learned to handle ordinary input has a low local loss and gains
almost nothing from another update. If each layer skips when its own loss is
below a threshold, most layers skip most of the time, and the saving is in
WALL CLOCK on a single device.

TWO VOID RUNS BEFORE THIS ONE, both caught by checks rather than by luck:

  The network is feedforward with no recurrence, but the task hid the key
  bit, so no arm could learn whether a door opens. Every local arm scored
  exactly 51.67% — the majority-class rate — and identical numbers across
  different configurations mean COLLAPSE, not agreement. The key bit is now
  an input.

  The learnability check then failed at 64.3%, but the probe was trained on
  a third of the data the real arms get. The check was stricter than the
  experiment. It now gets the same total training.

Arms:
  bp              ordinary backpropagation. The reference.
  local-full      local losses, every layer updated every step. Isolates
                  the cost of the local rule with no skipping.
  local-skip-XX   a layer updates only when its own loss is above the XXth
                  percentile of its recent history.
"""

import json
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from world import GridWorld, ACTIONS, EVENTS, DELTA, LOCKED

PHASES = ["keyed"]          # ONE phase. Three contradictory phases made
PHASE_STEPS = 12000         # this a retention test, not a speed test, and
                            # every arm collapsed to the majority floor.
EPISODE = 200
DEPTH = 6                 # layers, so there is something to skip
WIDTH = 128
LR = 3e-4
SEEDS = [0, 1]
N_TERRAIN = 5
PATCH = 9
TEST_STEPS = 4000
HISTORY = 200             # window for each layer's own loss percentile


def obs_dim():
    return PATCH * N_TERRAIN + len(ACTIONS) + 1      # +1 for has_key


def encode(obs, action):
    """The key bit is INCLUDED here, unlike elsewhere in this project.

    This experiment compares layer-skipping against backpropagation, and
    that comparison is only meaningful if both can learn the task. The
    network is feedforward with no recurrence, so a hidden key bit makes
    the task unlearnable for every arm and the comparison measures nothing.
    """
    v = torch.zeros(obs_dim())
    for i, cell in enumerate(obs["patch"]):
        v[i * N_TERRAIN + cell] = 1.0
    v[PATCH * N_TERRAIN + ACTIONS.index(action)] = 1.0
    v[-1] = 1.0 if obs.get("has_key") else 0.0
    return v.unsqueeze(0)


class Net(nn.Module):
    """A deep stack where every layer can be trained on its own loss.

    Each layer has an auxiliary head predicting the target from that
    layer's representation. Under the local rule a layer trains on its own
    head's loss and nothing flows between layers, so each has an
    independent decision about whether to update.
    """

    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.layers = nn.ModuleList(
            [nn.Linear(obs_dim() if i == 0 else WIDTH, WIDTH)
             for i in range(DEPTH)])
        self.heads = nn.ModuleList(
            [nn.Linear(WIDTH, len(EVENTS)) for _ in range(DEPTH)])

    def predict(self, x):
        """The final head is the network's answer."""
        h = x
        for layer in self.layers:
            h = F.relu(layer(h))
        return self.heads[-1](h)


class Percentiles:
    """Each layer's recent local losses, so 'unusually high for this layer'
    is judged per layer rather than by a shared absolute number.

    A shared threshold would be wrong: early and late layers sit at
    completely different loss scales.
    """

    def __init__(self, depth, window=HISTORY):
        self.hist = [[] for _ in range(depth)]
        self.window = window

    def push(self, i, v):
        h = self.hist[i]
        h.append(v)
        if len(h) > self.window:
            h.pop(0)

    def above(self, i, v, pct):
        h = self.hist[i]
        if len(h) < 30:
            return True          # warm up before skipping anything
        k = int(len(h) * pct / 100.0)
        return v >= sorted(h)[min(k, len(h) - 1)]


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


def train_bp(net, data):
    """Ordinary backpropagation through the whole stack. Every layer is
    touched on every step, because the gradient to layer 1 must traverse
    all of them."""
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    t0 = time.perf_counter()
    net.train()
    for item in data:
        x = encode(item[0], item[1])
        y = torch.tensor([EVENTS.index(item[2])])
        loss = F.cross_entropy(net.predict(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return time.perf_counter() - t0, DEPTH * len(data)


def train_local(net, data, skip_pct=None):
    """Layer-local losses, with optional skipping.

    Each layer is optimised separately, so a skipped layer costs nothing —
    no gradient computed, no optimiser state touched. That is the thing
    backpropagation cannot do, since its gradient must pass through every
    layer regardless.

    The input to layer i+1 is the representation layer i produced BEFORE
    its own update. That one step of staleness is inherent to local
    learning; recomputing after each update would cost a second forward
    pass per layer and defeat the point.
    """
    opts = [torch.optim.Adam(
        list(net.layers[i].parameters()) + list(net.heads[i].parameters()),
        lr=LR) for i in range(DEPTH)]
    pct = Percentiles(DEPTH)
    updates = 0
    t0 = time.perf_counter()
    net.train()

    for item in data:
        x = encode(item[0], item[1])
        y = torch.tensor([EVENTS.index(item[2])])

        h = x
        for i in range(DEPTH):
            # detach the input so no gradient crosses a layer boundary —
            # this is what makes the loss local
            h_in = h.detach()
            rep = F.relu(net.layers[i](h_in))
            loss = F.cross_entropy(net.heads[i](rep), y)
            v = float(loss.item())

            do_update = True
            if skip_pct is not None:
                do_update = pct.above(i, v, skip_pct)
            pct.push(i, v)

            if do_update:
                opts[i].zero_grad()
                loss.backward()
                opts[i].step()
                updates += 1

            h = rep

    return time.perf_counter() - t0, updates


def evaluate(net, test):
    """Accuracy at locked doors, where the rule actually decides
    something."""
    net.eval()
    hit = seen = 0
    with torch.no_grad():
        for item in test:
            logits = net.predict(encode(item[0], item[1]))
            if item[3]:
                seen += 1
                hit += int(logits.argmax(1).item()) == \
                    EVENTS.index(item[2])
    return 100.0 * hit / seen if seen else None


ARMS = [("bp", None), ("local-full", None),
        ("local-skip-50", 50), ("local-skip-70", 70),
        ("local-skip-90", 90)]


if __name__ == "__main__":
    print("Can a local rule skip layers that do not need updating?\n")
    print(f"{DEPTH} layers of {WIDTH}, {len(PHASES)} phases x "
          f"{PHASE_STEPS} steps, {len(SEEDS)} seeds")
    print("backprop must touch every layer every step. A local rule need "
          "not.\n")

    # Backprop must be able to learn the task, or nothing else means
    # anything. The probe gets the SAME total training the arms get — a
    # first version gave it a third as much and failed its own threshold
    # at 64.3%, which said nothing about the task and everything about
    # the check.
    probe = Net(0)
    train_bp(probe, walk(0, 1, PHASES[0], PHASE_STEPS * len(PHASES)))
    probe_acc = evaluate(probe, walk(0, 90000, PHASES[0], 2000))
    del probe
    print(f"  learnability check: backprop reaches {probe_acc:.1f}% "
          f"on one phase with full training")
    if probe_acc < 70:
        print("  ABORT: backprop cannot learn this task even with full\n"
              "  training, so comparing anything to it is meaningless.\n"
              "  The majority-class rate is about 52%.")
        raise SystemExit
    print("  good: the task is learnable, so the comparison is real\n")

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    results = {}
    for name, pct in ARMS:
        accs, secs, upds = [], [], []
        for seed in SEEDS:
            stream = []
            for r in PHASES:
                stream += walk(seed, seed * 100 + PHASES.index(r),
                               r, PHASE_STEPS)
            test = walk(seed, 90000 + seed, PHASES[0], TEST_STEPS)

            net = Net(seed)
            if name == "bp":
                t, u = train_bp(net, stream)
            else:
                t, u = train_local(net, stream, pct)
            accs.append(evaluate(net, test))
            secs.append(t)
            upds.append(u)
            del net

        total = DEPTH * (len(PHASES) * PHASE_STEPS)
        results[name] = dict(acc=mean(accs), secs=mean(secs),
                             updates=mean(upds),
                             frac=100.0 * mean(upds) / total)
        print(f"  {name:>15}: acc {mean(accs):5.2f}%  "
              f"{mean(secs):6.1f}s  "
              f"layer-updates {results[name]['frac']:5.1f}% of possible",
              flush=True)

    print("\n" + "=" * 78)
    print(f"{'arm':>15} {'accuracy':>10} {'seconds':>9} {'speedup':>9} "
          f"{'updates done':>14}")
    print("-" * 78)
    ref = results["bp"]["secs"]
    for name, _ in ARMS:
        r = results[name]
        print(f"{name:>15} {r['acc']:>9.2f}% {r['secs']:>8.1f} "
              f"{ref / r['secs']:>8.2f}x {r['frac']:>13.1f}%")
    print("=" * 78)

    # Identical accuracies across different configurations mean collapse,
    # not agreement. This project has hit that twice.
    accs = [round(results[n]["acc"], 2) for n, _ in ARMS if n != "bp"]
    if len(set(accs)) == 1:
        print("\n  WARNING: every local arm scored identically, which means "
              "they\n  collapsed to the same trivial answer rather than "
              "agreeing.")

    bp_acc = results["bp"]["acc"]
    full = results["local-full"]
    print(f"\nWHAT THE LOCAL RULE COSTS BEFORE ANY SKIPPING:")
    print(f"  accuracy {full['acc'] - bp_acc:+.2f} against backprop, "
          f"time {ref / full['secs']:.2f}x")

    print(f"\nWHAT SKIPPING BUYS, against local-full:")
    for name, pct in ARMS:
        if pct is None:
            continue
        r = results[name]
        print(f"  {name:>15}: {r['acc'] - full['acc']:+6.2f} accuracy, "
              f"{full['secs'] / r['secs']:.2f}x faster, "
              f"{r['frac']:.0f}% of updates performed")

    best = None
    for name, pct in ARMS:
        if pct is None:
            continue
        r = results[name]
        if r["acc"] >= bp_acc - 2 and (best is None
                                       or r["secs"] < results[best]["secs"]):
            best = name
    print()
    if best:
        r = results[best]
        print(f"  FASTEST ARM WITHIN 2 POINTS OF BACKPROP: {best}, "
              f"{ref / r['secs']:.2f}x faster than backprop, doing "
              f"{r['frac']:.0f}% of the layer updates.")
    else:
        print("  No skipping arm stayed within 2 points of backprop.")

    print("""
Three numbers, and the middle one is the point.

  WHAT THE LOCAL RULE COSTS. Layer-local losses are known to underperform
      backpropagation, and local-full measures that here with no skipping.
      If the gap is large, skipping cannot rescue it.

  WHAT SKIPPING BUYS. This is the claim. Backpropagation cannot skip a
      layer, because the gradient to layer 1 passes through every layer
      above it. A local rule can. If most layers skip most of the time and
      accuracy holds, that is a wall-clock saving on ONE device, which is
      not what the field's parallelism results are about.

  THE UPDATE FRACTION. The hypothesis predicts this is low on a real
      stream, because most inputs are ordinary and a layer that already
      handles them gains little from another update. If it stays near
      100%, layers do not converge independently and the premise is wrong.

The local arms carry real overhead: separate optimisers per layer, and a
per-layer loss computed every step whether or not it is used. So local-full
may be SLOWER than backprop, and the honest comparison for skipping is
against local-full rather than against bp.
""")
    with open("layerskip.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote layerskip.json")