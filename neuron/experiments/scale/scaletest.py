"""Do the mechanisms survive a hundredfold increase in network size?

Everything in this project was built and measured on roughly 100,000
parameters. That is tiny. Mechanisms that work at that size routinely break
larger, and the failures are usually specific: a gate calibrated for one
loss scale fires wrong at another; a rehearsal buffer sized for a small
model cannot anchor a big one; instability that never appeared at 128 units
appears at 2048.

This sweeps hidden width from 64 to 1024 on the grid world, which is roughly
25,000 to 6.5 million parameters — about a hundredfold range.

Two arms at every size:
  online     no protection. The control. Shows what the raw network does.
  guarded    through stability.py with sequence replay. Shows whether the
             protection still functions.

What is measured, and why each matters at scale:

  accuracy on the six memory forks   does it still learn the hard part
  gate fire rate                     the gate takes the top 30% of a rolling
                                     surprise window. Loss scale changes with
                                     model size, so the rate may drift
  rehearsals and buffer fill         does the anchor still fill and get used
  rollbacks                          does the canary guard start firing, or
                                     stop being able to catch anything
  health                             the canary drift measure. If this grows
                                     with size, damage is scaling too
  alive                              parameters still finite. The floor.
  us/step                            cost, which now scales with size

The interesting outcome is not "bigger is better". It is whether the SHAPE
holds: does guarded still beat online, does the gate still fire at a sane
rate, does the buffer still fill, does anything collapse.
"""

import json
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from bigworld import BigWorld, ACTIONS, EVENTS, FORKS
from stability import StabilityLayer


WIDTHS = [64, 128, 256, 512, 1024]
STEPS = 30000
EPISODE = 400
LR = 3e-4
SEEDS = [0, 1]
N_TERRAIN = 14
PATCH = 25
TEST_STEPS = 6000
SEQ_LEN = 10
SEQ_COUNT = 2

SIMPLE = ["locked door", "dark cell", "ice", "heavy door"]
CONJ = ["chasm", "cold room"]

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


class Backend:
    def __init__(self, hidden, seed=0):
        self.net = GRU(hidden, seed)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.h = None
        self._saved = None

    def begin_sequence(self):
        self._saved = self.h
        self.h = None

    def _xy(self, item):
        return encode(item[0], item[1]), \
            torch.tensor([EVENTS.index(item[2])])

    def score(self, item):
        self.net.eval()
        with torch.no_grad():
            x, y = self._xy(item)
            logits, _ = self.net(x, self.h)
            return float(F.cross_entropy(logits, y).item()), None

    def update(self, item, steps):
        self.net.train()
        last = None
        for _ in range(steps):
            x, y = self._xy(item)
            logits, h = self.net(x, self.h)
            loss = F.cross_entropy(logits, y)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.h = h.detach()
            last = float(loss.item())
        return last

    def reset_state(self):
        self.h = None
        self._saved = None

    def snapshot(self):
        return {k: v.detach().clone()
                for k, v in self.net.state_dict().items()}

    def restore(self, state):
        self.net.load_state_dict(state)

    def on_rollback(self):
        for g in self.opt.param_groups:
            g["lr"] = max(1e-6, g["lr"] * 0.5)


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


def run(hidden, guarded, train, test, seed):
    b = Backend(hidden, seed)
    n_params = sum(p.numel() for p in b.net.parameters())
    layer = None
    if guarded:
        layer = StabilityLayer(
            b, canary=train[:40], seed=seed,
            window=200, warmup=30, top_fraction=0.50,
            coherence_veto=1e9, loss_floor=0.02, steps_per_update=1,
            rehearse_per_item=5, rehearse_count=SEQ_COUNT, rehearse_steps=1,
            anchor_size=100, buffer_size=500, sequence_len=SEQ_LEN,
            replay_policy="uniform",
            guard=True, guard_per_item=2000, canary_tolerance=0.5)

    t0 = time.perf_counter()
    for i, item in enumerate(train):
        if i % EPISODE == 0:
            b.reset_state()
        if layer is not None:
            layer.observe(item)
        else:
            b.update(item, 1)
    secs = time.perf_counter() - t0

    b.reset_state()
    acc, forks = evaluate(b.net, test)
    alive = all(torch.isfinite(p).all() for p in b.net.parameters())
    st = layer.summary() if layer else {}
    out = dict(acc=acc, forks=forks, seconds=secs, alive=alive,
               params=n_params,
               fired=st.get("updates", STEPS),
               rehearsals=st.get("rehearsals", 0),
               buffer=st.get("buffer", 0),
               rollbacks=st.get("rollbacks", 0),
               health=st.get("health", 0.0))
    del b, layer
    return out


if __name__ == "__main__":
    print("Do the mechanisms survive a hundredfold size increase?")
    print(f"widths {WIDTHS}, {STEPS} steps, {len(SEEDS)} seeds\n")

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    results = {}
    for hidden in WIDTHS:
        for guarded in [False, True]:
            name = f"{'guarded' if guarded else 'online'}-{hidden}"
            runs = []
            for seed in SEEDS:
                train = walk(seed, seed * 100 + 1, STEPS)
                test = walk(seed, 90000 + seed, TEST_STEPS)
                runs.append(run(hidden, guarded, train, test, seed))
            results[name] = runs
            s = mean([mean([r["forks"][k] for k in SIMPLE]) for r in runs])
            c = mean([mean([r["forks"][k] for k in CONJ]) for r in runs])
            us = 1e6 * mean([r["seconds"] for r in runs]) / STEPS
            extra = ""
            if guarded:
                extra = (f"  fired {mean([r['fired'] for r in runs]):.0f}"
                         f"  buf {mean([r['buffer'] for r in runs]):.0f}"
                         f"  rb {mean([r['rollbacks'] for r in runs]):.1f}")
            print(f"  {name:>14} ({runs[0]['params']:>8} params): "
                  f"simple {s:5.1f}%  conj {c:5.1f}%  "
                  f"alive {all(r['alive'] for r in runs)}  "
                  f"{us:7.1f} us/step{extra}", flush=True)

    print("\n" + "=" * 92)
    print("DOES THE SHAPE HOLD? guarded minus online at each size")
    print(f"{'width':>7} {'params':>10} {'online s':>9} {'guard s':>9} "
          f"{'diff':>7} {'online c':>9} {'guard c':>9} {'diff':>7}")
    print("-" * 92)
    for hidden in WIDTHS:
        on = results[f"online-{hidden}"]
        gu = results[f"guarded-{hidden}"]
        os_ = mean([mean([r["forks"][k] for k in SIMPLE]) for r in on])
        gs = mean([mean([r["forks"][k] for k in SIMPLE]) for r in gu])
        oc = mean([mean([r["forks"][k] for k in CONJ]) for r in on])
        gc = mean([mean([r["forks"][k] for k in CONJ]) for r in gu])
        print(f"{hidden:>7} {on[0]['params']:>10} {os_:>8.1f}% {gs:>8.1f}% "
              f"{gs - os_:>+6.1f} {oc:>8.1f}% {gc:>8.1f}% {gc - oc:>+6.1f}")
    print("=" * 92)

    print("\nDO THE MECHANISMS STILL FUNCTION? guarded arm only")
    print(f"{'width':>7} {'fire rate':>10} {'rehearsals':>11} "
          f"{'buffer':>8} {'rollbacks':>10} {'health':>9} {'alive':>6}")
    print("-" * 92)
    for hidden in WIDTHS:
        gu = results[f"guarded-{hidden}"]
        rate = 100.0 * mean([r["fired"] for r in gu]) / STEPS
        print(f"{hidden:>7} {rate:>9.1f}% "
              f"{mean([r['rehearsals'] for r in gu]):>10.0f} "
              f"{mean([r['buffer'] for r in gu]):>7.0f} "
              f"{mean([r['rollbacks'] for r in gu]):>9.1f} "
              f"{mean([r['health'] for r in gu]):>+8.3f} "
              f"{str(all(r['alive'] for r in gu)):>6}")
    print("=" * 92)

    print("\nCOST, which is expected to rise:")
    for hidden in WIDTHS:
        gu = results[f"guarded-{hidden}"]
        us = 1e6 * mean([r["seconds"] for r in gu]) / STEPS
        print(f"  {hidden:>5} units: {us:7.1f} us/step  "
              f"({gu[0]['params']:>8} params)")

    print("""
Three things to read.

  DOES THE SHAPE HOLD? guarded should keep beating online at every size. If
      the advantage shrinks as the network grows, the protection is a
      small-model effect and will not survive to anything useful.

  DO THE MECHANISMS STILL FUNCTION? The gate takes the top 30% of a rolling
      surprise window, and loss scale changes with model size, so the fire
      rate may drift. The buffer should still fill. Rollbacks appearing at
      large sizes would mean instability that never existed at 128 units.

  WHERE DOES IT BREAK? If something fails at a specific width, that is the
      most useful possible answer — it names the thing to fix and the size
      it matters at, before any money goes on a 7B run.

Note accuracy may fall at large widths simply from being undertrained: 30,000
steps is few for 6 million parameters. Read the guarded-minus-online DIFF
rather than the absolute numbers, since both arms share that handicap.
""")
    with open("scaletest.json", "w") as f:
        json.dump({k: [dict(acc=r["acc"], forks=r["forks"],
                            params=r["params"], fired=r["fired"],
                            buffer=r["buffer"], rollbacks=r["rollbacks"],
                            health=r["health"], alive=r["alive"],
                            seconds=r["seconds"]) for r in v]
                   for k, v in results.items()}, f, indent=2)
    print("wrote scaletest.json")