"""Hold several hypotheses instead of one.

Three attempts at uncertainty failed. Feeding the model its own prediction
entropy recovered nothing, because entropy measures TASK difficulty rather
than perceptual ambiguity. Measuring input ambiguity and using it to gate
the state update recovered 2%.

The research explains why. A recurrent network on a partially observable
task ALREADY approximates a belief — its hidden state becomes correlated
with the posterior over the hidden variables as it learns. Bolting a
confidence number onto it was redundant. It has a belief; the belief is just
deterministic and single-valued.

The real distinction is between remembering the past by deterministic
feature computation, and inferring a DISTRIBUTION over the latent state.
Only the second can represent uncertainty.

Particle filter RNNs do the second without lengthening the latent vector.
Carry K hidden states rather than one — K hypotheses — each weighted by how
well it explains what is observed. Ambiguous reading, the particles
disagree. Clear reading, they converge. That is holding a belief loosely in
the literal sense.

TWO EARLIER VERSIONS OF THIS FILE WERE BROKEN, and both failures are
instructive:

  no process noise    every particle received the same input from the same
                      state and computed the same result, so sixteen
                      particles were one particle copied sixteen times.
                      Spread was exactly zero.
  scaling not gathering  the resample step multiplied particle states by
                      their weights. That is not resampling. It amplified
                      the dominant particle every step until everything
                      went NaN. Resampling SELECTS — it duplicates good
                      particles and drops bad ones, leaving state values
                      untouched.

pf_check.py verifies both before this runs: states finite, particles
differing, weights spread across more than one particle.

Arms:
  noisy-1        one particle, which is exactly an ordinary GRU
  noisy-4/8/16   four, eight, sixteen hypotheses
  noisy-ceiling  reads the true hidden state. The upper bound.
"""

import json
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from bigworld import ACTIONS, EVENTS, FORKS
from noisy import NoisyWorld, FEATURES


STEPS = 60000
EPISODE = 400
HIDDEN = 128
LR = 3e-4
SEEDS = [0, 1]
PATCH = 25
N_STATE = 7
TEST_STEPS = 8000
PROCESS_NOISE = 0.1      # what makes particles differ. Without it, K=1.
RESAMPLE_ALPHA = 0.5     # soft resampling mix, 1.0 = no resampling

SIMPLE = ["locked door", "dark cell", "ice", "heavy door"]
CONJ = ["chasm", "cold room"]

FORK_OF = {}
for name, (yes, no) in FORKS.items():
    for e in yes + no:
        FORK_OF[e] = name


def obs_dim(see_state):
    return PATCH * FEATURES + len(ACTIONS) + (N_STATE if see_state else 0)


def encode(obs, action, see_state):
    v = torch.zeros(obs_dim(see_state))
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
    return v.unsqueeze(0)


class ParticleGRU(nn.Module):
    """K hypotheses about the hidden state, carried in parallel.

    With K=1 this is exactly an ordinary GRU, which makes the control exact
    rather than approximate.

    The observation head is what makes it a filter: it scores how compatible
    each particle is with the current observation, and those scores become
    the weights. A particle that keeps failing to explain what is seen loses
    weight and stops influencing predictions.
    """

    def __init__(self, see_state, n_particles, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.k = n_particles
        d = obs_dim(see_state)
        self.enc = nn.Linear(d, HIDDEN)
        self.cell = nn.GRUCell(HIDDEN, HIDDEN)
        self.head = nn.Linear(HIDDEN, len(EVENTS))
        self.obs_head = nn.Sequential(
            nn.Linear(HIDDEN * 2, 64), nn.ReLU(),
            nn.Linear(64, 1))

    def init_state(self):
        h = torch.zeros(self.k, HIDDEN)
        if self.k > 1:
            # Start apart, so the first step's noise is not the only thing
            # distinguishing them.
            h = h + PROCESS_NOISE * torch.randn_like(h)
        logw = torch.full((self.k,),
                          -float(torch.log(torch.tensor(float(self.k)))))
        return h, logw

    def forward(self, x, state=None):
        if state is None:
            state = self.init_state()
        h, logw = state

        e = F.relu(self.enc(x))
        e_rep = e.expand(self.k, -1)
        h_new = self.cell(e_rep, h)

        if self.k > 1:
            # Diversity. Without this every particle computes the same
            # thing from the same input and the same state, forever.
            h_new = h_new + PROCESS_NOISE * torch.randn_like(h_new)

            score = self.obs_head(
                torch.cat([h_new, e_rep], dim=1)).squeeze(-1)
            logw = logw + score
            logw = logw - torch.logsumexp(logw, dim=0)

            if RESAMPLE_ALPHA < 1.0:
                w = logw.exp()
                mixed = RESAMPLE_ALPHA * w + (1 - RESAMPLE_ALPHA) / self.k
                mixed = mixed / mixed.sum()
                # GATHER, never scale. Resampling selects particles; it
                # does not multiply their values.
                idx = torch.multinomial(mixed, self.k, replacement=True)
                h_new = h_new[idx]
                # importance correction, because we sampled from the mixed
                # distribution rather than from the weights themselves
                logw = logw[idx] - torch.log(mixed[idx] + 1e-9)
                logw = logw - torch.logsumexp(logw, dim=0)

        w = logw.exp()
        per_particle = self.head(h_new)
        out = (w.unsqueeze(1) * per_particle).sum(0, keepdim=True)
        return out, (h_new, logw)


def diagnostics(state):
    """How much the particles disagree, and how many are doing any work.

    spread near zero means they have converged on one story.
    effective near 1 means one particle carries everything and the rest are
    dead, which is degeneracy — the classic particle filter failure.
    """
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


def run(see_state, k, train, test, seed):
    net = ParticleGRU(see_state, k, seed)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    t0 = time.perf_counter()

    state = None
    net.train()
    for i, item in enumerate(train):
        if i % EPISODE == 0:
            state = None
        x = encode(item[0], item[1], see_state)
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
            x = encode(item[0], item[1], see_state)
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

    acc = 100.0 * hit / seen
    forks = {f: (100.0 * fh[f] / fs[f] if fs[f] else None) for f in FORKS}
    out = dict(acc=acc, forks=forks, seconds=secs,
               spread=sum(spreads) / len(spreads) if spreads else 0.0,
               ess=sum(esss) / len(esss) if esss else 1.0,
               finite=all(torch.isfinite(p).all()
                          for p in net.parameters()))
    del net, opt
    return out


ARMS = {
    "noisy-1":       (False, 1),
    "noisy-4":       (False, 4),
    "noisy-8":       (False, 8),
    "noisy-16":      (False, 16),
    "noisy-ceiling": (True, 1),
}


if __name__ == "__main__":
    print("Hold several hypotheses instead of one.")
    print(f"{STEPS} steps, {len(SEEDS)} seeds, process noise "
          f"{PROCESS_NOISE}, resample alpha {RESAMPLE_ALPHA}\n")

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    results = {}
    for name, (see_state, k) in ARMS.items():
        runs = []
        for seed in SEEDS:
            train = walk(seed, seed * 100 + 1, STEPS, see_state)
            test = walk(seed, 90000 + seed, TEST_STEPS, see_state)
            runs.append(run(see_state, k, train, test, seed))
        results[name] = runs
        s = mean([mean([r["forks"][f] for f in SIMPLE]) for r in runs])
        c = mean([mean([r["forks"][f] for f in CONJ]) for r in runs])
        print(f"  {name:>14}: overall "
              f"{mean([r['acc'] for r in runs]):5.2f}%  "
              f"simple {s:5.1f}%  conj {c:5.1f}%  "
              f"spread {mean([r['spread'] for r in runs]):.4f}  "
              f"ess {mean([r['ess'] for r in runs]):5.2f}  "
              f"{mean([r['seconds'] for r in runs]) / 60:.1f}m", flush=True)

    print("\n" + "=" * 100)
    print(f"{'arm':>14} {'overall':>9} {'simple':>8} {'conj':>8} "
          f"{'spread':>9} {'ess':>7}   " +
          "  ".join(f"{f[:8]:>8}" for f in FORKS))
    print("-" * 100)
    for name in ARMS:
        runs = results[name]
        cells = "  ".join(
            f"{mean([r['forks'][f] for r in runs]):>7.1f}%" for f in FORKS)
        print(f"{name:>14} {mean([r['acc'] for r in runs]):>8.2f}% "
              f"{mean([mean([r['forks'][f] for f in SIMPLE]) for r in runs]):>7.1f}% "
              f"{mean([mean([r['forks'][f] for f in CONJ]) for r in runs]):>7.1f}% "
              f"{mean([r['spread'] for r in runs]):>8.4f} "
              f"{mean([r['ess'] for r in runs]):>6.2f}   {cells}")
    print("=" * 100)

    def pair(name):
        runs = results[name]
        return (mean([mean([r["forks"][f] for f in SIMPLE]) for r in runs]),
                mean([mean([r["forks"][f] for f in CONJ]) for r in runs]))

    b_s, b_c = pair("noisy-1")
    ce_s, ce_c = pair("noisy-ceiling")
    gap_s, gap_c = ce_s - b_s, ce_c - b_c

    print(f"\nRECOVERY against the single-particle control "
          f"({b_s:.1f}% / {b_c:.1f}%):")
    for name in ["noisy-4", "noisy-8", "noisy-16"]:
        s, c = pair(name)
        print(f"  {name:>10}: simple {s - b_s:+6.2f} "
              f"({100.0 * (s - b_s) / gap_s:+5.0f}% of the gap)   "
              f"conj {c - b_c:+6.2f} "
              f"({100.0 * (c - b_c) / gap_c:+5.0f}%)")
    print(f"\nheadroom to the ceiling: simple {gap_s:.1f}, conj {gap_c:.1f}")

    broken = [n for n in ["noisy-4", "noisy-8", "noisy-16"]
              if mean([r["spread"] for r in results[n]]) < 1e-6
              or not all(r["finite"] for r in results[n])]
    print()
    if broken:
        print(f"  MECHANISM NOT RUNNING in {broken} — particles identical\n"
              f"  or states non-finite. The accuracy numbers there mean "
              f"nothing.")
    else:
        print("  Mechanism ran: particles differ and states stayed finite.")

    print("""
Read the diagnostics before the accuracy.

  SPREAD is how much the particles disagree. Zero means they collapsed onto
      one hypothesis and this is an ordinary GRU wearing a costume.

  ESS is effective sample size. Near K means every particle contributes.
      Near 1 means one particle carries all the weight and the rest are
      dead — degeneracy, the classic particle filter failure. On random
      data ESS sits at exactly K because nothing discriminates; on a real
      task it should fall below K as the observation head learns to tell
      particles apart. ESS staying at exactly K here would mean the
      observation head learned nothing and the weighting is inert.

Then the recovery. If accuracy climbs with K, the problem was never a
missing confidence number — it was that a single deterministic state cannot
represent "it might be either".
""")
    with open("particles.json", "w") as f:
        json.dump({n: [dict(acc=r["acc"], forks=r["forks"],
                            spread=r["spread"], ess=r["ess"]) for r in v]
                   for n, v in results.items()}, f, indent=2)
    print("wrote particles.json")