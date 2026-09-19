"""Ten seconds of sanity before another forty minutes of running.

Two broken versions so far. First had no process noise, so every particle
computed the same thing and spread was exactly zero. Second scaled particle
states by their weights, which is not resampling — it amplified the dominant
particle every step until everything went NaN.

Resampling SELECTS particles. It duplicates the ones that explain the
observation and drops the ones that do not, leaving their state VALUES
untouched. Gather, never scale.

This checks three things on a few hundred random steps, before any real
experiment:

  the states stay finite
  the particles genuinely differ from each other
  the weights do not collapse onto a single particle
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


HIDDEN = 32
IN_DIM = 40
N_EVENTS = 5
PROCESS_NOISE = 0.1
RESAMPLE_ALPHA = 0.5


class PF(nn.Module):
    def __init__(self, k, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.k = k
        self.enc = nn.Linear(IN_DIM, HIDDEN)
        self.cell = nn.GRUCell(HIDDEN, HIDDEN)
        self.head = nn.Linear(HIDDEN, N_EVENTS)
        self.obs_head = nn.Sequential(
            nn.Linear(HIDDEN * 2, 32), nn.ReLU(), nn.Linear(32, 1))

    def init_state(self):
        h = torch.zeros(self.k, HIDDEN)
        if self.k > 1:
            # Particles must start apart, or the first step's noise is the
            # only thing distinguishing them.
            h = h + PROCESS_NOISE * torch.randn_like(h)
        logw = torch.full((self.k,), -float(torch.log(torch.tensor(
            float(self.k)))))
        return h, logw

    def forward(self, x, state=None):
        if state is None:
            state = self.init_state()
        h, logw = state

        e = F.relu(self.enc(x))
        e_rep = e.expand(self.k, -1)
        h_new = self.cell(e_rep, h)

        if self.k > 1:
            # diversity: without it, k particles are one particle k times
            h_new = h_new + PROCESS_NOISE * torch.randn_like(h_new)

            score = self.obs_head(
                torch.cat([h_new, e_rep], dim=1)).squeeze(-1)
            logw = logw + score
            logw = logw - torch.logsumexp(logw, dim=0)

            if RESAMPLE_ALPHA < 1.0:
                w = logw.exp()
                mixed = RESAMPLE_ALPHA * w + (1 - RESAMPLE_ALPHA) / self.k
                mixed = mixed / mixed.sum()
                # GATHER, do not scale. This is the fix.
                idx = torch.multinomial(mixed, self.k, replacement=True)
                h_new = h_new[idx]
                # importance correction, so the weights stay valid after
                # sampling from the mixed distribution rather than from w
                logw = logw[idx] - torch.log(mixed[idx] + 1e-9)
                logw = logw - torch.logsumexp(logw, dim=0)

        w = logw.exp()
        per = self.head(h_new)
        out = (w.unsqueeze(1) * per).sum(0, keepdim=True)
        return out, (h_new, logw)


def check(k, steps=400):
    net = PF(k)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    state = None
    spreads, effective = [], []

    for i in range(steps):
        if i % 100 == 0:
            state = None
        x = torch.randn(1, IN_DIM)
        logits, state = net(x, state)
        loss = F.cross_entropy(logits, torch.tensor([i % N_EVENTS]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        state = (state[0].detach(), state[1].detach())

        h, logw = state
        if not torch.isfinite(h).all():
            return dict(ok=False, why=f"non-finite state at step {i}")
        if k > 1:
            w = logw.exp()
            mean = (w.unsqueeze(1) * h).sum(0, keepdim=True)
            spreads.append(
                float((w.unsqueeze(1) * (h - mean) ** 2).sum(0).mean()))
            # effective sample size: k means all particles matter equally,
            # 1 means one particle carries everything
            effective.append(float(1.0 / (w ** 2).sum()))

    return dict(ok=True,
                spread=sum(spreads) / len(spreads) if spreads else 0.0,
                ess=sum(effective) / len(effective) if effective else 1.0,
                finite=bool(torch.isfinite(state[0]).all()))


if __name__ == "__main__":
    print("particle filter sanity check, before spending real time\n")
    print(f"  {'k':>4} {'finite':>8} {'spread':>10} {'effective':>11}")
    print("  " + "-" * 38)
    bad = []
    for k in [1, 4, 8, 16]:
        r = check(k)
        if not r["ok"]:
            print(f"  {k:>4}   FAILED: {r['why']}")
            bad.append(k)
            continue
        flag = ""
        if k > 1:
            if r["spread"] < 1e-6:
                flag = "  <-- particles identical"
                bad.append(k)
            elif r["ess"] < 1.5:
                flag = "  <-- weights collapsed onto one"
                bad.append(k)
        print(f"  {k:>4} {str(r['finite']):>8} {r['spread']:>10.5f} "
              f"{r['ess']:>10.2f}{flag}")

    print()
    if bad:
        print(f"  NOT READY: {bad}. Fix before running the experiment.")
    else:
        print("  READY. States finite, particles differ, weights spread\n"
              "  across more than one particle. The mechanism is running.")
    print("""
  spread     how much the particles disagree. Zero means they are one
             particle copied k times and the filter is not doing anything.
  effective  effective sample size. Near k means every particle
             contributes; near 1 means one particle carries all the weight
             and the others are dead, which is degeneracy.
""")