"""Particles that are genuinely different hypotheses, not one plus jitter.

Attempt D carried K hidden states and weighted them by a learned observation
head. It recovered +3.23 on conjunctive forks, monotonic in K — the first
real signal in four attempts. But the effective sample size came out at
almost exactly K, meaning the weights were uniform and the observation head
was discriminating nothing. With uniform weights the output is just the mean
of K predictions, so the gain was ENSEMBLING, not filtering.

The reason the head learned nothing: the particles differed only by random
Gaussian jitter in hidden space. Jitter does not correlate with being right,
so there was no signal to learn. They were one hypothesis plus noise.

THE FIX: make the particles disagree about something that MATTERS.

When a reading is ambiguous between a key and a rope, the useful thing is
not to average them — it is for some particles to believe "key" and others
"rope", and let the future sort them out. If a later locked door opens, the
key-believers were right and should gain weight.

So instead of adding noise to the hidden state, each particle is given a
DIFFERENT INTERPRETATION of the same ambiguous reading:

  each cell's reading is scored against every terrain signature, giving a
  soft distribution over what it might be
  each particle SAMPLES its own reading from that distribution
  clean readings produce the same interpretation for everyone; ambiguous
  ones split the particles

Now the particles are genuine competing hypotheses about the world, the
observation head has something real to discriminate on, and the weighting
can do work.

Arms:
  base-1        one particle, ordinary GRU. The control.
  jitter-8      attempt D: eight particles, random noise diversity
  hypoth-8      eight particles, interpretation diversity. The claim.
  hypoth-16     sixteen
  ceiling       reads the true hidden state. The upper bound.

jitter-8 is kept so the comparison isolates WHERE the diversity comes from,
holding the particle count fixed. If hypoth beats jitter at the same K, the
source of diversity is what matters and not the ensembling.
"""

import json
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from bigworld import ACTIONS, EVENTS, FORKS
from noisy import NoisyWorld, SIGNATURES, FEATURES, NOISE


STEPS = 60000
EPISODE = 400
HIDDEN = 128
LR = 3e-4
SEEDS = [0, 1]
PATCH = 25
N_STATE = 7
TEST_STEPS = 8000
PROCESS_NOISE = 0.1
RESAMPLE_ALPHA = 0.5
TEMP = 0.4              # softness of the interpretation distribution

SIMPLE = ["locked door", "dark cell", "ice", "heavy door"]
CONJ = ["chasm", "cold room"]

FORK_OF = {}
for name, (yes, no) in FORKS.items():
    for e in yes + no:
        FORK_OF[e] = name

SIG_KEYS = sorted(SIGNATURES)
SIG = torch.tensor([SIGNATURES[k] for k in SIG_KEYS], dtype=torch.float32)
N_TERRAIN = len(SIG_KEYS)


def interpret(patch, k, rng):
    """Turn one noisy reading into K different interpretations of it.

    Each cell is scored against every known signature, giving a soft
    distribution over what that cell might be. Each particle then samples
    its own reading from that distribution.

    A clean reading has almost all its mass on one signature, so every
    particle interprets it the same way. An ambiguous reading splits them —
    some particles believe key, others rope. That is the diversity that
    means something.

    Returns (k, 25 * n_terrain), a one-hot terrain guess per cell per
    particle.
    """
    x = torch.tensor(patch, dtype=torch.float32)          # (25, 8)
    d = torch.cdist(x, SIG)                                # (25, 14)
    logits = -d / (TEMP * NOISE)
    probs = F.softmax(logits, dim=1)                       # (25, 14)

    # one independent sample per particle per cell
    idx = torch.multinomial(probs, k, replacement=True)    # (25, k)
    out = torch.zeros(k, PATCH, N_TERRAIN)
    for p in range(k):
        out[p].scatter_(1, idx[:, p].unsqueeze(1), 1.0)
    return out.reshape(k, PATCH * N_TERRAIN), probs


def obs_dim(mode, see_state):
    """raw = the 8 continuous features per cell.
    hypoth = a one-hot terrain guess per cell, sampled per particle."""
    per_cell = N_TERRAIN if mode == "hypoth" else FEATURES
    return PATCH * per_cell + len(ACTIONS) + (N_STATE if see_state else 0)


class ParticleGRU(nn.Module):
    def __init__(self, mode, see_state, k, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.mode = mode
        self.k = k
        self.see_state = see_state
        d = obs_dim(mode, see_state)
        self.enc = nn.Linear(d, HIDDEN)
        self.cell = nn.GRUCell(HIDDEN, HIDDEN)
        self.head = nn.Linear(HIDDEN, len(EVENTS))
        self.obs_head = nn.Sequential(
            nn.Linear(HIDDEN * 2, 64), nn.ReLU(), nn.Linear(64, 1))

    def init_state(self):
        h = torch.zeros(self.k, HIDDEN)
        if self.k > 1 and self.mode == "jitter":
            h = h + PROCESS_NOISE * torch.randn_like(h)
        logw = torch.full((self.k,),
                          -float(torch.log(torch.tensor(float(self.k)))))
        return h, logw

    def forward(self, x_per_particle, state=None):
        """x_per_particle is (k, d) — each particle may see a DIFFERENT
        interpretation of the same reading."""
        if state is None:
            state = self.init_state()
        h, logw = state

        e = F.relu(self.enc(x_per_particle))         # (k, hidden)
        h_new = self.cell(e, h)

        if self.k > 1:
            if self.mode == "jitter":
                # attempt D's diversity: random and meaningless
                h_new = h_new + PROCESS_NOISE * torch.randn_like(h_new)

            score = self.obs_head(
                torch.cat([h_new, e], dim=1)).squeeze(-1)
            logw = logw + score
            logw = logw - torch.logsumexp(logw, dim=0)

            if RESAMPLE_ALPHA < 1.0:
                w = logw.exp()
                mixed = RESAMPLE_ALPHA * w + (1 - RESAMPLE_ALPHA) / self.k
                mixed = mixed / mixed.sum()
                # gather, never scale
                idx = torch.multinomial(mixed, self.k, replacement=True)
                h_new = h_new[idx]
                logw = logw[idx] - torch.log(mixed[idx] + 1e-9)
                logw = logw - torch.logsumexp(logw, dim=0)

        w = logw.exp()
        per = self.head(h_new)
        out = (w.unsqueeze(1) * per).sum(0, keepdim=True)
        return out, (h_new, logw)


def build_input(obs, action, mode, see_state, k, rng):
    """Assemble the per-particle input tensor."""
    tail = torch.zeros(len(ACTIONS) + (N_STATE if see_state else 0))
    tail[ACTIONS.index(action)] = 1.0
    if see_state:
        for j, s in enumerate(obs["state"]):
            tail[len(ACTIONS) + j] = float(s)

    if mode == "hypoth":
        body, _ = interpret(obs["patch"], k, rng)          # (k, 25*14)
    else:
        flat = torch.tensor(
            [f for cell in obs["patch"] for f in cell], dtype=torch.float32)
        body = flat.unsqueeze(0).expand(k, -1)             # identical

    return torch.cat([body, tail.unsqueeze(0).expand(k, -1)], dim=1)


def diagnostics(state):
    h, logw = state
    if h.shape[0] == 1:
        return 0.0, 1.0
    w = logw.exp()
    mean = (w.unsqueeze(1) * h).sum(0, keepdim=True)
    spread = float((w.unsqueeze(1) * (h - mean) ** 2).sum(0).mean())
    ess = float(1.0 / (w ** 2).sum())
    return spread, ess


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
        out.append((obs, action, event))
    return out


def run(mode, see_state, k, train, test, seed):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    net = ParticleGRU(mode, see_state, k, seed)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    t0 = time.perf_counter()

    state = None
    net.train()
    for i, item in enumerate(train):
        if i % EPISODE == 0:
            state = None
        x = build_input(item[0], item[1], mode, see_state, k, rng)
        logits, state = net(x, state)
        loss = F.cross_entropy(
            logits, torch.tensor([EVENTS.index(item[2])]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        state = (state[0].detach(), state[1].detach())
    secs = time.perf_counter() - t0

    net.eval()
    state = None
    hit = seen = 0
    fh = {f: 0 for f in FORKS}
    fs = {f: 0 for f in FORKS}
    spreads, esss = [], []
    with torch.no_grad():
        for i, item in enumerate(test):
            if i % EPISODE == 0:
                state = None
            x = build_input(item[0], item[1], mode, see_state, k, rng)
            logits, state = net(x, state)
            right = int(logits.argmax(1).item()) == EVENTS.index(item[2])
            seen += 1
            hit += right
            fork = FORK_OF.get(item[2])
            if fork:
                fs[fork] += 1
                fh[fork] += right
            if i % 50 == 0:
                sp, es = diagnostics(state)
                spreads.append(sp)
                esss.append(es)

    out = dict(acc=100.0 * hit / seen,
               forks={f: (100.0 * fh[f] / fs[f] if fs[f] else None)
                      for f in FORKS},
               seconds=secs,
               spread=sum(spreads) / len(spreads) if spreads else 0.0,
               ess=sum(esss) / len(esss) if esss else 1.0)
    del net, opt
    return out


# name -> (mode, sees true state, particles)
ARMS = {
    "base-1":    ("raw", False, 1),
    "jitter-8":  ("jitter", False, 8),
    "hypoth-8":  ("hypoth", False, 8),
    "hypoth-16": ("hypoth", False, 16),
    "ceiling":   ("raw", True, 1),
}


if __name__ == "__main__":
    print("Particles as genuine competing hypotheses.\n")

    # Check the interpretation actually splits on ambiguous readings,
    # before spending an hour on it.
    w = NoisyWorld(0)
    rng = random.Random(0)
    agree = []
    for _ in range(200):
        obs = w.observe()
        body, probs = interpret(obs["patch"], 8, rng)
        top = probs.max(dim=1).values
        # how often do all 8 particles agree on a cell
        guesses = body.reshape(8, PATCH, N_TERRAIN).argmax(-1)
        same = (guesses == guesses[0:1]).all(0).float().mean()
        agree.append(float(same))
        w.step(rng.choice(ACTIONS))
    mean_agree = sum(agree) / len(agree)
    print(f"  particles agree on {100 * mean_agree:.1f}% of cells")
    if mean_agree > 0.98:
        print("  WARNING: almost no disagreement, so this is jitter-8 with\n"
              "  extra steps. Raise TEMP.")
    elif mean_agree < 0.4:
        print("  WARNING: particles disagree on most cells, so each one is\n"
              "  seeing near-random terrain. Lower TEMP.")
    else:
        print("  good: clean cells agreed on, ambiguous ones split")
    print()

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    results = {}
    for name, (mode, see_state, k) in ARMS.items():
        runs = []
        for seed in SEEDS:
            train = walk(seed, seed * 100 + 1, STEPS, see_state)
            test = walk(seed, 90000 + seed, TEST_STEPS, see_state)
            runs.append(run(mode, see_state, k, train, test, seed))
        results[name] = runs
        s = mean([mean([r["forks"][f] for f in SIMPLE]) for r in runs])
        c = mean([mean([r["forks"][f] for f in CONJ]) for r in runs])
        print(f"  {name:>11}: overall "
              f"{mean([r['acc'] for r in runs]):5.2f}%  "
              f"simple {s:5.1f}%  conj {c:5.1f}%  "
              f"spread {mean([r['spread'] for r in runs]):.4f}  "
              f"ess {mean([r['ess'] for r in runs]):5.2f}  "
              f"{mean([r['seconds'] for r in runs]) / 60:.1f}m", flush=True)

    print("\n" + "=" * 100)
    print(f"{'arm':>11} {'overall':>9} {'simple':>8} {'conj':>8} "
          f"{'spread':>9} {'ess':>7}   " +
          "  ".join(f"{f[:8]:>8}" for f in FORKS))
    print("-" * 100)
    for name in ARMS:
        runs = results[name]
        cells = "  ".join(
            f"{mean([r['forks'][f] for r in runs]):>7.1f}%" for f in FORKS)
        print(f"{name:>11} {mean([r['acc'] for r in runs]):>8.2f}% "
              f"{mean([mean([r['forks'][f] for f in SIMPLE]) for r in runs]):>7.1f}% "
              f"{mean([mean([r['forks'][f] for f in CONJ]) for r in runs]):>7.1f}% "
              f"{mean([r['spread'] for r in runs]):>8.4f} "
              f"{mean([r['ess'] for r in runs]):>6.2f}   {cells}")
    print("=" * 100)

    def pair(name):
        runs = results[name]
        return (mean([mean([r["forks"][f] for f in SIMPLE]) for r in runs]),
                mean([mean([r["forks"][f] for f in CONJ]) for r in runs]))

    b_s, b_c = pair("base-1")
    ce_s, ce_c = pair("ceiling")
    gap_s, gap_c = ce_s - b_s, ce_c - b_c

    print(f"\nRECOVERY against the single-particle control:")
    for name in ["jitter-8", "hypoth-8", "hypoth-16"]:
        s, c = pair(name)
        print(f"  {name:>11}: simple {s - b_s:+6.2f} "
              f"({100.0 * (s - b_s) / gap_s:+5.0f}%)   "
              f"conj {c - b_c:+6.2f} "
              f"({100.0 * (c - b_c) / gap_c:+5.0f}%)")

    j_s, j_c = pair("jitter-8")
    h_s, h_c = pair("hypoth-8")
    print(f"\nDOES THE SOURCE OF DIVERSITY MATTER? "
          f"hypoth-8 minus jitter-8, same K:")
    print(f"  simple {h_s - j_s:+6.2f}   conj {h_c - j_c:+6.2f}")

    print("""
The comparison that decides this is hypoth-8 against jitter-8. Same particle
count, same ensembling benefit, same everything — the only difference is
whether the particles disagree about something meaningful or only by random
noise.

  hypoth beats jitter -> the source of diversity matters. The particles are
      genuine competing hypotheses and the weighting can discriminate
      between them, which is what a belief actually is.

  no difference -> the entire gain was ensembling all along, and holding
      several hypotheses buys nothing beyond averaging several guesses.

Watch ESS. In attempt D it sat at almost exactly K, meaning uniform weights
and an inert observation head. If it drops meaningfully below K here, the
head has learned to tell hypotheses apart, which is the thing that was
missing.
""")
    with open("hypotheses.json", "w") as f:
        json.dump({n: [dict(acc=r["acc"], forks=r["forks"],
                            spread=r["spread"], ess=r["ess"]) for r in v]
                   for n, v in results.items()}, f, indent=2)
    print("wrote hypotheses.json")