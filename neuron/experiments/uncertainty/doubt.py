"""Can the model learn to know when it is unsure?

Rung 3 found something specific: noise does not hurt PERCEPTION, it hurts
MEMORY. The ceiling arm, which reads the hidden state directly, actually
scored BETTER under noise than on clean data. The memory arms fell 12 to 14
points.

The reason is that a misread cell is a transient error, but a misread cell
that gets STORED is a persistent one. The agent walks onto an ambiguous cell,
the network decides it was a key, stores that, and carries it forward as
fact. Every later prediction depending on it is wrong until something
accidentally corrects it.

The hidden state can say "I have a key". It has nothing that says "how sure
am I".

THE IDEA, deliberately the smallest version: the model already produces a
distribution over outcomes every step, and when it is torn between two
possibilities that distribution is spread out. That spread IS a confidence
signal, and it is already being computed and thrown away.

So feed it back. Each step, compute the entropy of the model's own
prediction and hand it back as an input on the NEXT step, alongside a running
average of recent entropy. The model can then learn things like "the last few
readings were ambiguous, so trust the stored state less".

No new losses, no new machinery, no extra forward passes. Just stop
discarding a signal that already exists.

Four arms:
  clean          the discrete world, as a reference point
  noisy          the noisy world, no doubt signal. The rung 3 result.
  noisy-doubt    the same plus the entropy feedback. The test.
  noisy-ceiling  reads the true hidden state. The upper bound.

If noisy-doubt recovers any of the 12-14 points that noise cost, the
approach works and the principled version (precision weighting, tracking a
running variance alongside the state) is worth building. If it recovers
nothing, knowing-you-are-unsure needs more than the model's own output
distribution.
"""

import json
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from bigworld import BigWorld, ACTIONS, EVENTS, FORKS
from noisy import NoisyWorld, FEATURES


STEPS = 60000
EPISODE = 400
HIDDEN = 128
LR = 3e-4
SEEDS = [0, 1]
N_TERRAIN = 14
PATCH = 25
N_STATE = 7
TEST_STEPS = 8000
DOUBT_DECAY = 0.9        # how fast the running average of doubt forgets

SIMPLE = ["locked door", "dark cell", "ice", "heavy door"]
CONJ = ["chasm", "cold room"]

FORK_OF = {}
for name, (yes, no) in FORKS.items():
    for e in yes + no:
        FORK_OF[e] = name

MAX_ENTROPY = math.log(len(EVENTS))


def obs_dim(noisy, see_state, doubt):
    per_cell = FEATURES if noisy else N_TERRAIN
    return (PATCH * per_cell + len(ACTIONS)
            + (N_STATE if see_state else 0)
            + (2 if doubt else 0))


def encode(obs, action, noisy, see_state, doubt_now=None, doubt_avg=None):
    """The observation, plus optionally two doubt features.

    doubt_now  the model's own uncertainty on the PREVIOUS step, normalised
    doubt_avg  a decaying average of recent uncertainty

    Both come from the model's own output distribution, which was already
    being computed and discarded.
    """
    doubt = doubt_now is not None
    v = torch.zeros(obs_dim(noisy, see_state, doubt))
    i = 0
    if noisy:
        for cell in obs["patch"]:
            for f in cell:
                v[i] = f
                i += 1
    else:
        for j, cell in enumerate(obs["patch"]):
            v[j * N_TERRAIN + cell] = 1.0
        i = PATCH * N_TERRAIN
    v[i + ACTIONS.index(action)] = 1.0
    i += len(ACTIONS)
    if see_state:
        for j, s in enumerate(obs["state"]):
            v[i + j] = float(s)
        i += N_STATE
    if doubt:
        v[i] = doubt_now
        v[i + 1] = doubt_avg
    return v.unsqueeze(0)


class GRU(nn.Module):
    def __init__(self, noisy, see_state, doubt, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.enc = nn.Linear(obs_dim(noisy, see_state, doubt), HIDDEN)
        self.cell = nn.GRUCell(HIDDEN, HIDDEN)
        self.head = nn.Linear(HIDDEN, len(EVENTS))

    def forward(self, x, h=None):
        e = F.relu(self.enc(x))
        h = self.cell(e, h)
        return self.head(h), h


def entropy_of(logits):
    """How spread out the model's guess is, normalised to 0..1.

    Near 0 means confident in one outcome. Near 1 means torn between many.
    This is the doubt signal, and it costs nothing because the logits are
    already computed.
    """
    p = F.softmax(logits, dim=-1)
    ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1)
    return float(ent.item()) / MAX_ENTROPY


def walk(layout_seed, walk_seed, n, noisy, see_state):
    rng = random.Random(walk_seed)
    world = (NoisyWorld(layout_seed) if noisy else BigWorld(layout_seed))
    out = []
    for i in range(n):
        if i % EPISODE == 0:
            world.reset()
        obs = world.observe(hide_state=not see_state)
        action = rng.choice(ACTIONS)
        _, event = world.step(action)
        out.append((obs, action, event))
    return out


def run(noisy, see_state, doubt, train, test, seed):
    net = GRU(noisy, see_state, doubt, seed)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    t0 = time.perf_counter()

    h = None
    d_now, d_avg = 0.0, 0.0
    net.train()
    for i, item in enumerate(train):
        if i % EPISODE == 0:
            h = None
            d_now, d_avg = 0.0, 0.0
        x = encode(item[0], item[1], noisy, see_state,
                   d_now if doubt else None, d_avg if doubt else None)
        logits, h = net(x, h)
        loss = F.cross_entropy(
            logits, torch.tensor([EVENTS.index(item[2])]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        h = h.detach()
        if doubt:
            d_now = entropy_of(logits.detach())
            d_avg = DOUBT_DECAY * d_avg + (1 - DOUBT_DECAY) * d_now
    secs = time.perf_counter() - t0

    net.eval()
    h = None
    d_now, d_avg = 0.0, 0.0
    hit = seen = 0
    fh = {k: 0 for k in FORKS}
    fs = {k: 0 for k in FORKS}
    with torch.no_grad():
        for i, item in enumerate(test):
            if i % EPISODE == 0:
                h = None
                d_now, d_avg = 0.0, 0.0
            x = encode(item[0], item[1], noisy, see_state,
                       d_now if doubt else None, d_avg if doubt else None)
            logits, h = net(x, h)
            right = int(logits.argmax(1).item()) == EVENTS.index(item[2])
            seen += 1
            hit += right
            fork = FORK_OF.get(item[2])
            if fork:
                fs[fork] += 1
                fh[fork] += right
            if doubt:
                d_now = entropy_of(logits)
                d_avg = DOUBT_DECAY * d_avg + (1 - DOUBT_DECAY) * d_now

    acc = 100.0 * hit / seen
    forks = {k: (100.0 * fh[k] / fs[k] if fs[k] else None) for k in FORKS}
    del net, opt
    return dict(acc=acc, forks=forks, seconds=secs)


# name -> (noisy, sees true state, doubt feedback)
ARMS = {
    "clean":         (False, False, False),
    "noisy":         (True, False, False),
    "noisy-doubt":   (True, False, True),
    "noisy-ceiling": (True, True, False),
}


if __name__ == "__main__":
    print("Can the model learn to know when it is unsure?")
    print(f"{STEPS} steps, {len(SEEDS)} seeds")
    print("doubt = the model's own prediction entropy, fed back next "
          "step\n")

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    results = {}
    for name, (noisy, see_state, doubt) in ARMS.items():
        runs = []
        for seed in SEEDS:
            train = walk(seed, seed * 100 + 1, STEPS, noisy, see_state)
            test = walk(seed, 90000 + seed, TEST_STEPS, noisy, see_state)
            runs.append(run(noisy, see_state, doubt, train, test, seed))
        results[name] = runs
        s = mean([mean([r["forks"][k] for k in SIMPLE]) for r in runs])
        c = mean([mean([r["forks"][k] for k in CONJ]) for r in runs])
        print(f"  {name:>14}: overall "
              f"{mean([r['acc'] for r in runs]):5.2f}%  "
              f"simple {s:5.1f}%  conj {c:5.1f}%  "
              f"{mean([r['seconds'] for r in runs]) / 60:.1f}m", flush=True)

    print("\n" + "=" * 88)
    print(f"{'arm':>14} {'overall':>9} {'simple':>8} {'conj':>8}   " +
          "  ".join(f"{k[:8]:>8}" for k in FORKS))
    print("-" * 88)
    for name in ARMS:
        runs = results[name]
        cells = "  ".join(
            f"{mean([r['forks'][k] for r in runs]):>7.1f}%" for k in FORKS)
        print(f"{name:>14} {mean([r['acc'] for r in runs]):>8.2f}% "
              f"{mean([mean([r['forks'][k] for k in SIMPLE]) for r in runs]):>7.1f}% "
              f"{mean([mean([r['forks'][k] for k in CONJ]) for r in runs]):>7.1f}%   "
              f"{cells}")
    print("=" * 88)

    def pair(name):
        runs = results[name]
        return (mean([mean([r["forks"][k] for k in SIMPLE]) for r in runs]),
                mean([mean([r["forks"][k] for k in CONJ]) for r in runs]))

    cl_s, cl_c = pair("clean")
    no_s, no_c = pair("noisy")
    db_s, db_c = pair("noisy-doubt")
    ce_s, ce_c = pair("noisy-ceiling")

    print(f"\nWHAT NOISE COST (clean minus noisy):")
    print(f"  simple {cl_s - no_s:+6.2f}   conj {cl_c - no_c:+6.2f}")
    print(f"\nWHAT DOUBT RECOVERED (noisy-doubt minus noisy):")
    print(f"  simple {db_s - no_s:+6.2f}   conj {db_c - no_c:+6.2f}")
    cost_s = cl_s - no_s
    cost_c = cl_c - no_c
    if cost_s > 0.5:
        print(f"  that is {100.0 * (db_s - no_s) / cost_s:.0f}% of the "
              f"simple-fork damage")
    if cost_c > 0.5:
        print(f"  that is {100.0 * (db_c - no_c) / cost_c:.0f}% of the "
              f"conjunctive damage")
    print(f"\nHEADROOM REMAINING (ceiling minus noisy-doubt):")
    print(f"  simple {ce_s - db_s:+6.2f}   conj {ce_c - db_c:+6.2f}")

    print("""
The ceiling arm reads the true hidden state, so it never has to remember
anything and never stores a wrong belief. The gap between it and the noisy
arms is the cost of remembering under uncertainty. That gap is what this is
trying to close.

  doubt recovers a real fraction -> the model CAN learn to use its own
      uncertainty, and the principled version is worth building: track a
      running variance alongside the hidden state and scale updates by it,
      so ambiguous readings move the state less.

  doubt recovers nothing -> knowing you are unsure needs more than the
      model's own output distribution. The entropy of a prediction may
      reflect task difficulty rather than perceptual ambiguity, in which
      case the signal has to come from somewhere else — disagreement
      between readings, or an explicit variance estimate.

  doubt makes it worse -> two extra inputs is two more things to overfit,
      and the signal is noise.
""")
    with open("doubt.json", "w") as f:
        json.dump({k: [dict(acc=r["acc"], forks=r["forks"]) for r in v]
                   for k, v in results.items()}, f, indent=2)
    print("wrote doubt.json")