"""Should an ambiguous reading move the memory less?

Approach A fed the model's own prediction entropy back as an input and
recovered nothing (-0.96 on simple forks). The diagnosis: entropy measures
TASK difficulty, not perceptual ambiguity. The model is uncertain at a locked
door because the outcome depends on hidden state it cannot see, not because
it misread the cell. Wrong question.

Two corrections follow, and this tests both.

  MEASURE THE RIGHT THING. Ambiguity is a property of the READING, not of
      the prediction. Each cell arrives as a point in 8-dimensional sensor
      space, and the terrain types sit at known signatures. A reading close
      to one signature is clean; a reading sitting between two is ambiguous.
      That distance is computable from the input alone — no truth needed,
      nothing leaked.

  USE IT IN THE RIGHT PLACE. Approach A handed doubt to the model as a
      feature and hoped it would learn what to do. This instead GATES THE
      STATE UPDATE directly: an ambiguous reading moves the hidden state
      less, so a bad reading cannot overwrite a good belief. That is
      precision weighting, the mechanism from the closed-form predictive
      coding work.

Arms:
  noisy               no protection. The rung 3 result.
  noisy-feature       clarity as an extra INPUT. Isolates whether the
                      benefit comes from the signal or from how it is used.
  noisy-gated         clarity GATES the state update. The claim.
  noisy-gated-learn   the same, but the gate strength is learned rather
                      than fixed, so the model can decide how much to
                      trust clean readings.
  noisy-ceiling       reads the true hidden state. The upper bound.

The feature arm matters: if it works as well as the gated one, the signal
was the whole story and the gating mechanism is unnecessary complexity.
"""

import json
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from bigworld import BigWorld, ACTIONS, EVENTS, FORKS
from noisy import NoisyWorld, SIGNATURES, FEATURES, NOISE


STEPS = 60000
EPISODE = 400
HIDDEN = 128
LR = 3e-4
SEEDS = [0, 1]
PATCH = 25
N_STATE = 7
TEST_STEPS = 8000

SIMPLE = ["locked door", "dark cell", "ice", "heavy door"]
CONJ = ["chasm", "cold room"]

FORK_OF = {}
for name, (yes, no) in FORKS.items():
    for e in yes + no:
        FORK_OF[e] = name

# The known signatures, as a tensor, for the clarity computation.
SIG = torch.tensor([SIGNATURES[k] for k in sorted(SIGNATURES)],
                   dtype=torch.float32)


def clarity_of(patch):
    """How unambiguous this reading is, from 0 (torn) to 1 (clean).

    For each cell, find the two nearest terrain signatures. If the nearest
    is much closer than the second, the reading is clean. If they are
    equidistant, it could be either and the reading is ambiguous.

    Uses only the observed features and the known signature table. No truth
    is consulted, so nothing is leaked — a real deployment would have the
    same information, since the signatures are what the sensor was
    calibrated against.
    """
    x = torch.tensor(patch, dtype=torch.float32)          # (25, 8)
    d = torch.cdist(x, SIG)                                # (25, 14)
    nearest, _ = torch.topk(d, 2, dim=1, largest=False)
    gap = nearest[:, 1] - nearest[:, 0]
    # Normalise by the noise scale: a gap of a few standard deviations is
    # decisive, a gap near zero is a coin flip.
    return float(torch.tanh(gap.mean() / (2.0 * NOISE)).item())


def obs_dim(see_state, feature):
    return (PATCH * FEATURES + len(ACTIONS)
            + (N_STATE if see_state else 0)
            + (1 if feature else 0))


def encode(obs, action, see_state, feature, clar=None):
    v = torch.zeros(obs_dim(see_state, feature))
    i = 0
    for cell in obs["patch"]:
        for f in cell:
            v[i] = f
            i += 1
    v[i + ACTIONS.index(action)] = 1.0
    i += len(ACTIONS)
    if see_state:
        for j, s in enumerate(obs["state"]):
            v[i + j] = float(s)
        i += N_STATE
    if feature:
        v[i] = clar
    return v.unsqueeze(0)


class GatedGRU(nn.Module):
    """A GRU whose state update can be scaled by how clear the reading is.

    A standard GRU cell computes a new state from the old one and the input.
    Here the result is blended back toward the OLD state in proportion to
    how ambiguous the reading was:

        h_new = h_old + clarity * (h_gru - h_old)

    With clarity 1 this is an ordinary GRU. With clarity 0 the state does not
    move at all, so a reading that could mean anything cannot overwrite what
    is already believed.

    learn_gate lets the model scale the effect, so it can decide how much a
    clean reading is worth rather than taking the fixed measure literally.
    """

    def __init__(self, in_dim, gated, learn_gate, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.enc = nn.Linear(in_dim, HIDDEN)
        self.cell = nn.GRUCell(HIDDEN, HIDDEN)
        self.head = nn.Linear(HIDDEN, len(EVENTS))
        self.gated = gated
        self.learn_gate = learn_gate
        if learn_gate:
            # scale and bias on the clarity value, learned
            self.gain = nn.Parameter(torch.tensor(1.0))
            self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, h=None, clar=1.0):
        e = F.relu(self.enc(x))
        h_new = self.cell(e, h)
        if self.gated and h is not None:
            g = clar
            if self.learn_gate:
                g = torch.sigmoid(self.gain * clar + self.bias)
            h_new = h + g * (h_new - h)
        return self.head(h_new), h_new


def walk(layout_seed, walk_seed, n, see_state):
    rng = random.Random(walk_seed)
    world = NoisyWorld(layout_seed)
    out = []
    for i in range(n):
        if i % EPISODE == 0:
            world.reset()
        obs = world.observe(hide_state=not see_state)
        action = rng.choice(ACTIONS)
        _, event = world.step(action)
        out.append((obs, action, event, clarity_of(obs["patch"])))
    return out


def run(see_state, feature, gated, learn_gate, train, test, seed):
    net = GatedGRU(obs_dim(see_state, feature), gated, learn_gate, seed)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    t0 = time.perf_counter()

    h = None
    net.train()
    for i, item in enumerate(train):
        if i % EPISODE == 0:
            h = None
        x = encode(item[0], item[1], see_state, feature, item[3])
        logits, h = net(x, h, item[3])
        loss = F.cross_entropy(
            logits, torch.tensor([EVENTS.index(item[2])]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        h = h.detach()
    secs = time.perf_counter() - t0

    net.eval()
    h = None
    hit = seen = 0
    fh = {k: 0 for k in FORKS}
    fs = {k: 0 for k in FORKS}
    with torch.no_grad():
        for i, item in enumerate(test):
            if i % EPISODE == 0:
                h = None
            x = encode(item[0], item[1], see_state, feature, item[3])
            logits, h = net(x, h, item[3])
            right = int(logits.argmax(1).item()) == EVENTS.index(item[2])
            seen += 1
            hit += right
            fork = FORK_OF.get(item[2])
            if fork:
                fs[fork] += 1
                fh[fork] += right

    acc = 100.0 * hit / seen
    forks = {k: (100.0 * fh[k] / fs[k] if fs[k] else None) for k in FORKS}
    gate_params = None
    if learn_gate:
        gate_params = (float(net.gain.item()), float(net.bias.item()))
    del net, opt
    return dict(acc=acc, forks=forks, seconds=secs, gate=gate_params)


# name -> (sees true state, clarity as feature, gates state, learned gate)
ARMS = {
    "noisy":             (False, False, False, False),
    "noisy-feature":     (False, True, False, False),
    "noisy-gated":       (False, False, True, False),
    "noisy-gated-learn": (False, False, True, True),
    "noisy-ceiling":     (True, False, False, False),
}


if __name__ == "__main__":
    print("Should an ambiguous reading move the memory less?\n")

    # Sanity check on the clarity measure before spending any training.
    w = NoisyWorld(0)
    rng = random.Random(0)
    clean_w = BigWorld(0)
    vals = []
    for _ in range(300):
        obs = w.observe()
        vals.append(clarity_of(obs["patch"]))
        w.step(rng.choice(ACTIONS))
    print(f"  clarity on noisy readings: mean {sum(vals) / len(vals):.3f}, "
          f"min {min(vals):.3f}, max {max(vals):.3f}")
    if max(vals) - min(vals) < 0.05:
        print("  WARNING: clarity barely varies, so it carries almost no "
              "information\n  and no arm using it can help.")
    print()

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    results = {}
    for name, (see_state, feature, gated, learn_gate) in ARMS.items():
        runs = []
        for seed in SEEDS:
            train = walk(seed, seed * 100 + 1, STEPS, see_state)
            test = walk(seed, 90000 + seed, TEST_STEPS, see_state)
            runs.append(run(see_state, feature, gated, learn_gate,
                            train, test, seed))
        results[name] = runs
        s = mean([mean([r["forks"][k] for k in SIMPLE]) for r in runs])
        c = mean([mean([r["forks"][k] for k in CONJ]) for r in runs])
        extra = ""
        if runs[0]["gate"] is not None:
            g = runs[0]["gate"]
            extra = f"  gain {g[0]:+.2f} bias {g[1]:+.2f}"
        print(f"  {name:>18}: overall "
              f"{mean([r['acc'] for r in runs]):5.2f}%  "
              f"simple {s:5.1f}%  conj {c:5.1f}%{extra}", flush=True)

    print("\n" + "=" * 92)
    print(f"{'arm':>18} {'overall':>9} {'simple':>8} {'conj':>8}   " +
          "  ".join(f"{k[:8]:>8}" for k in FORKS))
    print("-" * 92)
    for name in ARMS:
        runs = results[name]
        cells = "  ".join(
            f"{mean([r['forks'][k] for r in runs]):>7.1f}%" for k in FORKS)
        print(f"{name:>18} {mean([r['acc'] for r in runs]):>8.2f}% "
              f"{mean([mean([r['forks'][k] for k in SIMPLE]) for r in runs]):>7.1f}% "
              f"{mean([mean([r['forks'][k] for k in CONJ]) for r in runs]):>7.1f}%   "
              f"{cells}")
    print("=" * 92)

    def pair(name):
        runs = results[name]
        return (mean([mean([r["forks"][k] for k in SIMPLE]) for r in runs]),
                mean([mean([r["forks"][k] for k in CONJ]) for r in runs]))

    base_s, base_c = pair("noisy")
    ce_s, ce_c = pair("noisy-ceiling")

    print(f"\nRECOVERY, against the unprotected noisy arm "
          f"({base_s:.1f}% / {base_c:.1f}%):")
    for name in ["noisy-feature", "noisy-gated", "noisy-gated-learn"]:
        s, c = pair(name)
        gap_s = ce_s - base_s
        gap_c = ce_c - base_c
        print(f"  {name:>18}: simple {s - base_s:+6.2f} "
              f"({100.0 * (s - base_s) / gap_s:+5.0f}% of the gap)   "
              f"conj {c - base_c:+6.2f} "
              f"({100.0 * (c - base_c) / gap_c:+5.0f}%)")

    print(f"\nheadroom to the ceiling: simple {ce_s - base_s:.1f}, "
          f"conj {ce_c - base_c:.1f}")

    print("""
Three questions, and the middle one is the point.

  DOES THE SIGNAL HELP AT ALL? noisy-feature hands clarity to the model as
      an input, the same way approach A handed it entropy. If this works,
      the signal was the whole story.

  DOES GATING BEAT FEEDING? noisy-gated uses clarity to scale how far the
      state moves, so an ambiguous reading cannot overwrite a good belief.
      If gating beats the feature arm, the mechanism matters and not just
      the information — which is the claim precision weighting makes.

  SHOULD THE GATE BE LEARNED? noisy-gated-learn lets the model scale the
      measure rather than taking it literally. Watch the printed gain and
      bias: a large positive gain means the model wanted sharper gating
      than the fixed version gave it.

If all three fail, the honest conclusion is that this architecture cannot
hold a belief loosely, and doing so needs an explicit memory with confidence
attached rather than a hidden state.
""")
    with open("precision.json", "w") as f:
        json.dump({k: [dict(acc=r["acc"], forks=r["forks"], gate=r["gate"])
                       for r in v] for k, v in results.items()}, f, indent=2)
    print("wrote precision.json")