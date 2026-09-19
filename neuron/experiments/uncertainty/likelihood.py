"""Make the weighting earn its place.

Interpretation diversity works: particles that each sample their own reading
of an ambiguous cell recover 47% of the simple-fork gap and 23% of the
conjunctive one, and beat random-jitter particles by +8.24 at the same
particle count.

But the effective sample size stayed at 7.78 of 8, meaning the weights are
nearly uniform and the learned observation head is discriminating almost
nothing. The gain comes from carrying diverse interpretations, not from
selecting between them. So it is a diverse ensemble, not yet a belief.

WHY THE HEAD LEARNED NOTHING. It is trained only through the final
prediction loss, so the gradient reaching it runs through a weighted sum and
is weak and indirect. Nothing ever tells it "this particle was right".

THE FIX, and it is the standard Bayesian update rather than an invention:
weight each particle by the LIKELIHOOD it assigned to what actually
happened. A hypothesis that predicted the outcome well gains weight; one
that predicted badly loses it. No learning required — it is arithmetic on
the particle's own output.

Legitimate here because this is a streaming prediction task: the model
predicts, and then the outcome is revealed. Reweighting on the PREVIOUS
step's outcome uses only the past, exactly as a filter should.

Arms:
  base-1        one particle. The control.
  head-8        the current version: learned compatibility head
  lik-8         likelihood weighting instead. The claim.
  both-8        both signals together
  lik-16        sixteen, to see whether the count still matters
  ceiling       reads the true hidden state. The upper bound.
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
RESAMPLE_ALPHA = 0.5
TEMP = 0.4               # tuned: particles split only on ambiguous cells
LIK_SCALE = 1.0          # how sharply the likelihood moves the weights

SIMPLE = ["locked door", "dark cell", "ice", "heavy door"]
CONJ = ["chasm", "cold room"]

FORK_OF = {}
for name, (yes, no) in FORKS.items():
    for e in yes + no:
        FORK_OF[e] = name

SIG_KEYS = sorted(SIGNATURES)
SIG = torch.tensor([SIGNATURES[k] for k in SIG_KEYS], dtype=torch.float32)
N_TERRAIN = len(SIG_KEYS)


def interpret(patch, k):
    """K different readings of the same ambiguous observation.

    Clean cells look the same to every particle. Ambiguous ones split them
    into genuine competing hypotheses.
    """
    x = torch.tensor(patch, dtype=torch.float32)
    d = torch.cdist(x, SIG)
    probs = F.softmax(-d / (TEMP * NOISE), dim=1)
    idx = torch.multinomial(probs, k, replacement=True)     # (25, k)
    out = torch.zeros(k, PATCH, N_TERRAIN)
    for p in range(k):
        out[p].scatter_(1, idx[:, p].unsqueeze(1), 1.0)
    return out.reshape(k, PATCH * N_TERRAIN)


def obs_dim(hypoth, see_state):
    per_cell = N_TERRAIN if hypoth else FEATURES
    return PATCH * per_cell + len(ACTIONS) + (N_STATE if see_state else 0)


def build_input(obs, action, hypoth, see_state, k):
    tail = torch.zeros(len(ACTIONS) + (N_STATE if see_state else 0))
    tail[ACTIONS.index(action)] = 1.0
    if see_state:
        for j, s in enumerate(obs["state"]):
            tail[len(ACTIONS) + j] = float(s)
    if hypoth:
        body = interpret(obs["patch"], k)
    else:
        flat = torch.tensor(
            [f for cell in obs["patch"] for f in cell], dtype=torch.float32)
        body = flat.unsqueeze(0).expand(k, -1)
    return torch.cat([body, tail.unsqueeze(0).expand(k, -1)], dim=1)


class ParticleGRU(nn.Module):
    """K hypotheses, weighted by a learned head, by likelihood, or both."""

    def __init__(self, hypoth, see_state, k, use_head, use_lik, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.k = k
        self.use_head = use_head
        self.use_lik = use_lik
        d = obs_dim(hypoth, see_state)
        self.enc = nn.Linear(d, HIDDEN)
        self.cell = nn.GRUCell(HIDDEN, HIDDEN)
        self.head = nn.Linear(HIDDEN, len(EVENTS))
        self.obs_head = nn.Sequential(
            nn.Linear(HIDDEN * 2, 64), nn.ReLU(), nn.Linear(64, 1))

    def init_state(self):
        h = torch.zeros(self.k, HIDDEN)
        logw = torch.full((self.k,),
                          -float(torch.log(torch.tensor(float(self.k)))))
        return h, logw

    def step(self, x, state):
        """Advance and predict. Returns the blended prediction, the
        per-particle predictions (needed for the likelihood update), and
        the new state."""
        if state is None:
            state = self.init_state()
        h, logw = state

        e = F.relu(self.enc(x))
        h_new = self.cell(e, h)

        if self.k > 1 and self.use_head:
            score = self.obs_head(
                torch.cat([h_new, e], dim=1)).squeeze(-1)
            logw = logw + score
            logw = logw - torch.logsumexp(logw, dim=0)

        per = self.head(h_new)                       # (k, events)
        w = logw.exp()
        out = (w.unsqueeze(1) * per).sum(0, keepdim=True)
        return out, per, (h_new, logw)

    def reweight(self, per, y, state):
        """The Bayesian update: weight each hypothesis by the likelihood it
        assigned to what actually happened, then resample.

        This is the step that makes it a filter rather than an ensemble. A
        particle that keeps predicting the wrong outcome loses weight and
        stops influencing anything.
        """
        h, logw = state
        if self.k == 1:
            return (h, logw)

        if self.use_lik:
            loglik = F.log_softmax(per, dim=1)[:, y]     # (k,)
            logw = logw + LIK_SCALE * loglik
            logw = logw - torch.logsumexp(logw, dim=0)

        if RESAMPLE_ALPHA < 1.0:
            w = logw.exp()
            mixed = RESAMPLE_ALPHA * w + (1 - RESAMPLE_ALPHA) / self.k
            mixed = mixed / mixed.sum()
            idx = torch.multinomial(mixed, self.k, replacement=True)
            h = h[idx]
            logw = logw[idx] - torch.log(mixed[idx] + 1e-9)
            logw = logw - torch.logsumexp(logw, dim=0)
        return (h, logw)


def diagnostics(state):
    h, logw = state
    if h.shape[0] == 1:
        return 0.0, 1.0
    w = logw.exp()
    mean = (w.unsqueeze(1) * h).sum(0, keepdim=True)
    spread = float((w.unsqueeze(1) * (h - mean) ** 2).sum(0).mean())
    return spread, float(1.0 / (w ** 2).sum())


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


def run(hypoth, see_state, k, use_head, use_lik, train, test, seed):
    torch.manual_seed(seed)
    net = ParticleGRU(hypoth, see_state, k, use_head, use_lik, seed)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    t0 = time.perf_counter()

    state = None
    net.train()
    for i, item in enumerate(train):
        if i % EPISODE == 0:
            state = None
        x = build_input(item[0], item[1], hypoth, see_state, k)
        logits, per, state = net.step(x, state)
        y = EVENTS.index(item[2])
        loss = F.cross_entropy(logits, torch.tensor([y]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            state = net.reweight(per.detach(), y, state)
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
            x = build_input(item[0], item[1], hypoth, see_state, k)
            logits, per, state = net.step(x, state)
            y = EVENTS.index(item[2])
            right = int(logits.argmax(1).item()) == y
            seen += 1
            hit += right
            fork = FORK_OF.get(item[2])
            if fork:
                fs[fork] += 1
                fh[fork] += right
            # the outcome is revealed after predicting, so reweighting on
            # it uses only the past
            state = net.reweight(per, y, state)
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


# name -> (hypoth interpretations, sees state, k, learned head, likelihood)
ARMS = {
    "base-1":  (False, False, 1, False, False),
    "head-8":  (True, False, 8, True, False),
    "lik-8":   (True, False, 8, False, True),
    "both-8":  (True, False, 8, True, True),
    "lik-16":  (True, False, 16, False, True),
    "ceiling": (False, True, 1, False, False),
}


if __name__ == "__main__":
    print("Make the weighting earn its place.")
    print(f"TEMP {TEMP}, likelihood scale {LIK_SCALE}, "
          f"resample alpha {RESAMPLE_ALPHA}\n")

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    results = {}
    for name, (hyp, ss, k, uh, ul) in ARMS.items():
        runs = []
        for seed in SEEDS:
            train = walk(seed, seed * 100 + 1, STEPS, ss)
            test = walk(seed, 90000 + seed, TEST_STEPS, ss)
            runs.append(run(hyp, ss, k, uh, ul, train, test, seed))
        results[name] = runs
        s = mean([mean([r["forks"][f] for f in SIMPLE]) for r in runs])
        c = mean([mean([r["forks"][f] for f in CONJ]) for r in runs])
        print(f"  {name:>9}: overall "
              f"{mean([r['acc'] for r in runs]):5.2f}%  "
              f"simple {s:5.1f}%  conj {c:5.1f}%  "
              f"spread {mean([r['spread'] for r in runs]):.4f}  "
              f"ess {mean([r['ess'] for r in runs]):5.2f}  "
              f"{mean([r['seconds'] for r in runs]) / 60:.1f}m", flush=True)

    print("\n" + "=" * 100)
    print(f"{'arm':>9} {'overall':>9} {'simple':>8} {'conj':>8} "
          f"{'spread':>9} {'ess':>7}   " +
          "  ".join(f"{f[:8]:>8}" for f in FORKS))
    print("-" * 100)
    for name in ARMS:
        runs = results[name]
        cells = "  ".join(
            f"{mean([r['forks'][f] for r in runs]):>7.1f}%" for f in FORKS)
        print(f"{name:>9} {mean([r['acc'] for r in runs]):>8.2f}% "
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
    for name in ["head-8", "lik-8", "both-8", "lik-16"]:
        s, c = pair(name)
        print(f"  {name:>9}: simple {s - b_s:+6.2f} "
              f"({100.0 * (s - b_s) / gap_s:+5.0f}%)   "
              f"conj {c - b_c:+6.2f} ({100.0 * (c - b_c) / gap_c:+5.0f}%)")

    h_s, h_c = pair("head-8")
    l_s, l_c = pair("lik-8")
    print(f"\nDOES LIKELIHOOD WEIGHTING BEAT THE LEARNED HEAD? "
          f"lik-8 minus head-8:")
    print(f"  simple {l_s - h_s:+6.2f}   conj {l_c - h_c:+6.2f}")
    print(f"  ESS: head-8 {mean([r['ess'] for r in results['head-8']]):.2f}"
          f"  lik-8 {mean([r['ess'] for r in results['lik-8']]):.2f}"
          f"  (out of 8)")

    print("""
The ESS column is the direct test of whether the weighting is doing
anything. With the learned head it sat at 7.78 of 8 — uniform weights, an
inert filter, a diverse ensemble wearing a filter's clothes.

  ESS drops well below K with likelihood weighting -> the filter is
      selecting between hypotheses. Particles that predict badly lose
      weight and stop influencing the answer. That is a belief being
      updated by evidence rather than an average being taken.

  ESS drops but accuracy does not improve -> the selection is happening and
      is not useful, which would mean the particles' interpretations were
      already good enough that choosing between them adds nothing.

  ESS collapses toward 1 -> degeneracy. One particle takes all the weight
      and the others die, which is the classic particle filter failure and
      would need a lower LIK_SCALE or more aggressive resampling.
""")
    with open("likelihood.json", "w") as f:
        json.dump({n: [dict(acc=r["acc"], forks=r["forks"],
                            spread=r["spread"], ess=r["ess"]) for r in v]
                   for n, v in results.items()}, f, indent=2)
    print("wrote likelihood.json")