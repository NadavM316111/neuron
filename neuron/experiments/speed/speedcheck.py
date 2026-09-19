"""Does the speed result hold on a task it was not tuned on?

Found on the weather task: gradient accumulation plus a graph spanning the
accumulation window gives 4.16x the speed of stepping every sample, WITH
better accuracy. Adam's cost is per-step and batch-independent, so stepping
every eighth sample amortises the dominant cost; calling backward once per
window instead of eight times amortises the second-largest cost; and
averaging eight noisy gradients cancels noise, which is why accuracy rose
rather than fell.

But that was one task, two seeds, one weather station. Weather is a smooth
physical process with three outcome classes. If the result is real it should
transfer to something structurally different.

The grid world is that: 21 outcomes, six hidden variables, discrete terrain,
rules someone wrote rather than physics. If accumulation wins there too, the
mechanism generalises. If it does not, the weather result was a property of
smooth continuous data and should be reported that way.

Arms, mirroring the weather sweep:
  step-1-128      the original: step every sample, detach every step
  accum-8-128     accumulate 8, detach every step
  accum-8-128-deep    accumulate 8, graph spans the window
  accum-8-64-deep     the same at half width, the weather winner
  accum-8-32-deep     half again, to find this task's floor

Measured on wall clock AND on the six memory-dependent forks, because a
faster arm that stops solving the hard part is not faster at the same task.
"""

import json
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from bigworld import BigWorld, ACTIONS, EVENTS, FORKS


STEPS = 60000
EPISODE = 400
LR = 3e-4
SEEDS = [0, 1]
N_TERRAIN = 14
PATCH = 25
TEST_STEPS = 8000

SIMPLE = ["locked door", "dark cell", "ice", "heavy door"]
CONJ = ["chasm", "cold room"]

FORK_OF = {}
for name, (yes, no) in FORKS.items():
    for e in yes + no:
        FORK_OF[e] = name


def obs_dim():
    """No hidden state. The model must remember."""
    return PATCH * N_TERRAIN + len(ACTIONS)


def encode(obs, action):
    v = torch.zeros(obs_dim())
    for i, cell in enumerate(obs["patch"]):
        v[i * N_TERRAIN + cell] = 1.0
    v[PATCH * N_TERRAIN + ACTIONS.index(action)] = 1.0
    return v.unsqueeze(0)


class GRU(nn.Module):
    def __init__(self, hidden, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.enc = nn.Linear(obs_dim(), hidden)
        self.cell = nn.GRUCell(hidden, hidden)
        self.head = nn.Linear(hidden, len(EVENTS))

    def forward(self, x, h=None):
        e = F.relu(self.enc(x))
        h = self.cell(e, h)
        return self.head(h), h


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


def evaluate(net, test):
    """Overall plus per-fork accuracy. The forks are where memory decides."""
    net.eval()
    h = None
    hit = seen = 0
    fh = {k: 0 for k in FORKS}
    fs = {k: 0 for k in FORKS}
    with torch.no_grad():
        for i, item in enumerate(test):
            if i % EPISODE == 0:
                h = None
            logits, h = net(encode(item[0], item[1]), h)
            right = int(logits.argmax(1).item()) == EVENTS.index(item[2])
            seen += 1
            hit += right
            fork = FORK_OF.get(item[2])
            if fork:
                fs[fork] += 1
                fh[fork] += right
    return (100.0 * hit / seen,
            {k: (100.0 * fh[k] / fs[k] if fs[k] else None) for k in FORKS})


def run(hidden, accum, deep, train, test, seed):
    """One pass, in order, never revisited.

    deep=False detaches the hidden state every step, so each backward call
    covers one step. deep=True collects the losses and calls backward once
    per window, so the graph spans all `accum` samples — which on weather
    was both cheaper and more accurate, because one large backward beats
    eight small ones.
    """
    net = GRU(hidden, seed)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    t0 = time.perf_counter()
    h = None
    net.train()

    if deep:
        losses = []
        for i, item in enumerate(train):
            if i % EPISODE == 0:
                if losses:
                    opt.zero_grad()
                    torch.stack(losses).mean().backward()
                    opt.step()
                    losses = []
                h = None
            logits, h = net(encode(item[0], item[1]), h)
            losses.append(F.cross_entropy(
                logits, torch.tensor([EVENTS.index(item[2])])))
            if len(losses) >= accum:
                opt.zero_grad()
                torch.stack(losses).mean().backward()
                opt.step()
                losses = []
                h = h.detach()
        if losses:
            opt.zero_grad()
            torch.stack(losses).mean().backward()
            opt.step()
    else:
        pending = 0
        opt.zero_grad()
        for i, item in enumerate(train):
            if i % EPISODE == 0:
                h = None
            logits, h = net(encode(item[0], item[1]), h)
            loss = F.cross_entropy(
                logits, torch.tensor([EVENTS.index(item[2])]))
            (loss / accum).backward()
            h = h.detach()
            pending += 1
            if pending >= accum:
                opt.step()
                opt.zero_grad()
                pending = 0
        if pending:
            opt.step()
            opt.zero_grad()

    secs = time.perf_counter() - t0
    acc, forks = evaluate(net, test)
    del net, opt
    return dict(acc=acc, forks=forks, seconds=secs)


# name -> (hidden, accumulation, spanning graph)
ARMS = {
    "step-1-128":       (128, 1, False),
    "accum-8-128":      (128, 8, False),
    "accum-8-128-deep": (128, 8, True),
    "accum-8-64-deep":  (64, 8, True),
    "accum-8-32-deep":  (32, 8, True),
}


if __name__ == "__main__":
    print("Does the weather speed result transfer to the grid world?")
    print(f"{STEPS} steps, {len(SEEDS)} seeds, 21 outcomes, "
          f"6 hidden variables\n")

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    results = {}
    for name, (hidden, accum, deep) in ARMS.items():
        runs = []
        for seed in SEEDS:
            train = walk(seed, seed * 100 + 1, STEPS)
            test = walk(seed, 90000 + seed, TEST_STEPS)
            runs.append(run(hidden, accum, deep, train, test, seed))
        results[name] = runs
        a = mean([r["acc"] for r in runs])
        s = mean([mean([r["forks"][k] for k in SIMPLE]) for r in runs])
        c = mean([mean([r["forks"][k] for k in CONJ]) for r in runs])
        us = 1e6 * mean([r["seconds"] for r in runs]) / STEPS
        print(f"  {name:>18}: overall {a:5.2f}%  simple {s:5.1f}%  "
              f"conj {c:5.1f}%  {us:6.1f} us/step", flush=True)

    ref = 1e6 * mean([r["seconds"] for r in results["step-1-128"]]) / STEPS

    print("\n" + "=" * 84)
    print(f"{'arm':>18} {'overall':>9} {'simple':>8} {'conj':>8} "
          f"{'us/step':>9} {'speedup':>9}")
    print("-" * 84)
    for name in ARMS:
        runs = results[name]
        a = mean([r["acc"] for r in runs])
        s = mean([mean([r["forks"][k] for k in SIMPLE]) for r in runs])
        c = mean([mean([r["forks"][k] for k in CONJ]) for r in runs])
        us = 1e6 * mean([r["seconds"] for r in runs]) / STEPS
        print(f"{name:>18} {a:>8.2f}% {s:>7.1f}% {c:>7.1f}% "
              f"{us:>8.1f} {ref / us:>8.2f}x")
    print("=" * 84)

    print("\nper-fork detail:")
    print(f"{'arm':>18} " + "  ".join(f"{k[:9]:>9}" for k in FORKS))
    print("-" * 84)
    for name in ARMS:
        runs = results[name]
        cells = "  ".join(
            f"{mean([r['forks'][k] for r in runs]):>8.1f}%" for k in FORKS)
        print(f"{name:>18} {cells}")

    base = results["step-1-128"]
    base_s = mean([mean([r["forks"][k] for k in SIMPLE]) for r in base])
    base_c = mean([mean([r["forks"][k] for k in CONJ]) for r in base])
    print(f"\nagainst step-1-128 (simple {base_s:.1f}%, conj {base_c:.1f}%):")
    for name in ARMS:
        if name == "step-1-128":
            continue
        runs = results[name]
        s = mean([mean([r["forks"][k] for k in SIMPLE]) for r in runs])
        c = mean([mean([r["forks"][k] for k in CONJ]) for r in runs])
        us = 1e6 * mean([r["seconds"] for r in runs]) / STEPS
        verdict = ""
        if s >= base_s - 1 and c >= base_c - 1:
            verdict = "  NO ACCURACY COST"
        print(f"  {name:>18}: simple {s - base_s:+5.1f}  "
              f"conj {c - base_c:+5.1f}  at {ref / us:.2f}x{verdict}")

    print("""
The question is whether the weather result was about the mechanism or about
smooth continuous data.

  accumulation wins here too -> the mechanism generalises. Adam's per-step
      cost is the dominant expense in any single-sample online setting, and
      amortising it is a general fix rather than a property of one dataset.

  accuracy falls here -> the weather gain came from noise-averaging on a
      smooth physical signal, and discrete symbolic streams do not have the
      same noise to average away. Worth reporting as a scope condition
      rather than a general result.

Watch the conjunctive forks especially. They were the hardest thing in this
world and the most sensitive to how credit is assigned, so if spanning the
graph helps anywhere it should help there.
""")
    with open("speedcheck.json", "w") as f:
        json.dump({k: [dict(acc=r["acc"], forks=r["forks"],
                            seconds=r["seconds"]) for r in v]
                   for k, v in results.items()}, f, indent=2)
    print("wrote speedcheck.json")