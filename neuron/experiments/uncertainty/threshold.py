"""Let the weights accumulate: resample only when they degenerate.

Six attempts in. Interpretation diversity works — particles that each sample
their own reading of an ambiguous cell recover 45-48% of the simple-fork gap
and 30-35% of the conjunctive one, and beat random-jitter particles by +8.24
at the same particle count.

But the WEIGHTING has never done anything. Effective sample size sat at 7.75
of 8 with a learned head, and 7.95 with likelihood weighting — more uniform,
not less. So the system is a diverse ensemble, not yet a belief being updated
by evidence.

THE CAUSE, and it is a design error rather than a property of the problem:
soft resampling ran EVERY step. Resampling resets weights toward uniform, so
each likelihood update was immediately wiped by the resample that followed
it. The weights never got a chance to accumulate across steps.

THE FIX, which is standard practice in particle filtering: resample ONLY when
the effective sample size falls below a threshold. Between resamples the
weights accumulate evidence, so a hypothesis that keeps predicting badly
steadily loses influence. Resampling then rescues the population only when it
has genuinely degenerated.

Arms:
  base-1        one particle. The control.
  every-8       resample every step. The previous behaviour, for comparison.
  thresh-8      resample only below the threshold. The claim.
  thresh-8-hi   the same with a sharper likelihood, in case the signal is
                too weak to move the weights at all.
  thresh-16     sixteen particles.
  ceiling       reads the true hidden state. The upper bound.

The number that decides it is ESS. Below 8 means the weights are finally
carrying information.
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
TEMP = 0.4                # tuned: particles split only on ambiguous cells

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
    idx = torch.multinomial(probs, k, replacement=True)
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
    """K hypotheses, reweighted by evidence, resampled only on demand."""

    def __init__(self, hypoth, see_state, k, lik_scale, ess_threshold,
                 seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.k = k
        self.lik_scale = lik_scale
        self.ess_threshold = ess_threshold
        d = obs_dim(hypoth, see_state)
        self.enc = nn.Linear(d, HIDDEN)
        self.cell = nn.GRUCell(HIDDEN, HIDDEN)
        self.head = nn.Linear(HIDDEN, len(EVENTS))
        self.resamples = 0
        self.steps_seen = 0

    def init_state(self):
        h = torch.zeros(self.k, HIDDEN)
        logw = torch.full((self.k,),
                          -float(torch.log(torch.tensor(float(self.k)))))
        return h, logw

    def step(self, x, state):
        """Advance and predict. Returns the weighted prediction, the
        per-particle predictions (needed for the evidence update), and the
        new state."""
        if state is None:
            state = self.init_state()
        h, logw = state
        e = F.relu(self.enc(x))
        h_new = self.cell(e, h)
        per = self.head(h_new)
        w = logw.exp()
        out = (w.unsqueeze(1) * per).sum(0, keepdim=True)
        return out, per, (h_new, logw)

    def reweight(self, per, y, state):
        """Weight each hypothesis by the likelihood it gave to what actually
        happened, then resample ONLY if the population has degenerated.

        The threshold is the whole point. Resampling every step resets the
        weights toward uniform and destroys the evidence they carry, which
        is why every earlier version had an effective sample size of nearly
        K and a filter that filtered nothing.
        """
        h, logw = state
        self.steps_seen += 1
        if self.k == 1:
            return (h, logw)

        loglik = F.log_softmax(per, dim=1)[:, y]
        logw = logw + self.lik_scale * loglik
        logw = logw - torch.logsumexp(logw, dim=0)

        w = logw.exp()
        ess = float(1.0 / (w ** 2).sum())
        if ess < self.ess_threshold * self.k and RESAMPLE_ALPHA < 1.0:
            self.resamples += 1
            mixed = RESAMPLE_ALPHA * w + (1 - RESAMPLE_ALPHA) / self.k
            mixed = mixed / mixed.sum()
            # gather, never scale: resampling selects particles, it does
            # not multiply their state values
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


def run(hypoth, see_state, k, lik_scale, ess_threshold, train, test, seed):
    torch.manual_seed(seed)
    net = ParticleGRU(hypoth, see_state, k, lik_scale, ess_threshold, seed)
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
    net.resamples = 0
    net.steps_seen = 0
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
            if i % 50 == 0:
                sp, es = diagnostics(state)
                spreads.append(sp)
                esss.append(es)
            # the outcome is revealed after predicting, so reweighting on
            # it uses only the past
            state = net.reweight(per, y, state)

    out = dict(acc=100.0 * hit / seen,
               forks={f: (100.0 * fh[f] / fs[f] if fs[f] else None)
                      for f in FORKS},
               seconds=secs,
               spread=sum(spreads) / len(spreads) if spreads else 0.0,
               ess=sum(esss) / len(esss) if esss else 1.0,
               resample_rate=(100.0 * net.resamples / net.steps_seen
                              if net.steps_seen else 0.0))
    del net, opt
    return out


# name -> (hypoth, sees state, k, likelihood scale, ESS threshold)
# threshold 1.0 means "always resample", which is the old behaviour
ARMS = {
    "base-1":      (False, False, 1, 0.0, 1.0),
    "every-8":     (True, False, 8, 1.0, 1.0),
    "thresh-8":    (True, False, 8, 1.0, 0.5),
    "thresh-8-hi": (True, False, 8, 3.0, 0.5),
    "thresh-16":   (True, False, 16, 1.0, 0.5),
    "ceiling":     (False, True, 1, 0.0, 1.0),
}


if __name__ == "__main__":
    print("Let the weights accumulate: resample only on degeneracy.")
    print(f"TEMP {TEMP}, resample alpha {RESAMPLE_ALPHA}\n")

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    results = {}
    for name, (hyp, ss, k, lik, thr) in ARMS.items():
        runs = []
        for seed in SEEDS:
            train = walk(seed, seed * 100 + 1, STEPS, ss)
            test = walk(seed, 90000 + seed, TEST_STEPS, ss)
            runs.append(run(hyp, ss, k, lik, thr, train, test, seed))
        results[name] = runs
        s = mean([mean([r["forks"][f] for f in SIMPLE]) for r in runs])
        c = mean([mean([r["forks"][f] for f in CONJ]) for r in runs])
        print(f"  {name:>12}: overall "
              f"{mean([r['acc'] for r in runs]):5.2f}%  "
              f"simple {s:5.1f}%  conj {c:5.1f}%  "
              f"ess {mean([r['ess'] for r in runs]):5.2f}  "
              f"resampled {mean([r['resample_rate'] for r in runs]):5.1f}%  "
              f"{mean([r['seconds'] for r in runs]) / 60:.1f}m", flush=True)

    print("\n" + "=" * 104)
    print(f"{'arm':>12} {'overall':>9} {'simple':>8} {'conj':>8} "
          f"{'ess':>7} {'resamp':>8}   " +
          "  ".join(f"{f[:8]:>8}" for f in FORKS))
    print("-" * 104)
    for name in ARMS:
        runs = results[name]
        cells = "  ".join(
            f"{mean([r['forks'][f] for r in runs]):>7.1f}%" for f in FORKS)
        print(f"{name:>12} {mean([r['acc'] for r in runs]):>8.2f}% "
              f"{mean([mean([r['forks'][f] for f in SIMPLE]) for r in runs]):>7.1f}% "
              f"{mean([mean([r['forks'][f] for f in CONJ]) for r in runs]):>7.1f}% "
              f"{mean([r['ess'] for r in runs]):>6.2f} "
              f"{mean([r['resample_rate'] for r in runs]):>7.1f}%   {cells}")
    print("=" * 104)

    def pair(name):
        runs = results[name]
        return (mean([mean([r["forks"][f] for f in SIMPLE]) for r in runs]),
                mean([mean([r["forks"][f] for f in CONJ]) for r in runs]))

    b_s, b_c = pair("base-1")
    ce_s, ce_c = pair("ceiling")
    gap_s, gap_c = ce_s - b_s, ce_c - b_c

    print(f"\nRECOVERY against the single-particle control:")
    for name in ["every-8", "thresh-8", "thresh-8-hi", "thresh-16"]:
        s, c = pair(name)
        print(f"  {name:>12}: simple {s - b_s:+6.2f} "
              f"({100.0 * (s - b_s) / gap_s:+5.0f}%)   "
              f"conj {c - b_c:+6.2f} ({100.0 * (c - b_c) / gap_c:+5.0f}%)")

    e_s, e_c = pair("every-8")
    t_s, t_c = pair("thresh-8")
    print(f"\nDOES LETTING WEIGHTS ACCUMULATE HELP? "
          f"thresh-8 minus every-8:")
    print(f"  simple {t_s - e_s:+6.2f}   conj {t_c - e_c:+6.2f}")
    print(f"  ESS: every-8 {mean([r['ess'] for r in results['every-8']]):.2f}"
          f"  thresh-8 {mean([r['ess'] for r in results['thresh-8']]):.2f}"
          f"  thresh-8-hi "
          f"{mean([r['ess'] for r in results['thresh-8-hi']]):.2f}"
          f"  (out of 8)")

    print("""
ESS is the direct test and it has failed twice: 7.75 with a learned head,
7.95 with per-step likelihood weighting. Both mean uniform weights and a
filter that filters nothing.

  ESS now sits well below 8 -> the weights are carrying evidence across
      steps. Particles that keep predicting badly lose influence, which is
      a belief being updated rather than an average being taken.

  ESS still near 8 with a high likelihood scale -> the per-particle
      predictions are too similar for the likelihood to separate them, and
      the diversity lives in the interpretation rather than in the outputs.
      That would be a real finding: the ensemble is the mechanism and the
      filter is decoration.

  ESS collapses toward 1 -> degeneracy. One particle wins everything and
      the others die. Lower the threshold so resampling fires sooner.

The resample column shows how often it actually fired. Near 0% means the
threshold never triggers and this is pure weight accumulation; near 100%
means it is the old behaviour under a new name.
""")
    with open("threshold.json", "w") as f:
        json.dump({n: [dict(acc=r["acc"], forks=r["forks"], ess=r["ess"],
                            resample_rate=r["resample_rate"]) for r in v]
                   for n, v in results.items()}, f, indent=2)
    print("wrote threshold.json")