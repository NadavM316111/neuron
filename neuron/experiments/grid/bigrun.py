"""Do the mechanisms survive a world fifteen times bigger?

Everything proven so far: 9x9 grid, 4 outcomes, 1 hidden bit, 64 units.
This is 625 cells, 21 live outcomes, 6 hidden variables, and two rules that
need CONJUNCTIONS of hidden state rather than single flags.

Nothing about the mechanisms changes. Same network size, same learning
rate, same stability settings. If performance collapses, the world did it.

Five arms:
  frozen      random weights, never trained. The floor.
  ceiling     GRU that can SEE the six hidden variables. The upper bound,
              since it reads what the others must remember.
  online      GRU, hidden state hidden, no protection.
  moments     the same, with single-moment replay.
  sequences   the same, with sequence replay. The mechanism that won small.

Scored per fork, not just overall. Overall accuracy is dominated by moved,
blocked and waited, which are 72% of the stream and need no memory at all.
The six forks are where memory decides the answer.

The interesting question is not whether accuracy drops. It will. It is
whether sequence replay still beats the alternatives, and whether the two
CONJUNCTIVE forks (chasm, cold room) fall further than the simple ones. If
they do, the system is learning flags and not combinations, which would be
a real limit worth knowing.
"""

import json
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from bigworld import BigWorld, ACTIONS, EVENTS, FORKS, fairness_check
from stability import StabilityLayer


STEPS = 60000
EPISODE = 400
HIDDEN = 64
LR = 3e-4
SEEDS = [0, 1, 2]
N_TERRAIN = 14
PATCH = 25                 # 5x5
N_STATE = 7
TEST_STEPS = 8000
CHECK_EVERY = 15000
SEQ_LEN = 10
SEQ_COUNT = 2

# Which fork each outcome belongs to, for scoring.
FORK_OF = {}
for name, (yes, no) in FORKS.items():
    for e in yes + no:
        FORK_OF[e] = name


def obs_dim(see_state):
    return PATCH * N_TERRAIN + len(ACTIONS) + (N_STATE if see_state else 0)


def encode(obs, action, see_state):
    v = torch.zeros(obs_dim(see_state))
    for i, cell in enumerate(obs["patch"]):
        v[i * N_TERRAIN + cell] = 1.0
    i = PATCH * N_TERRAIN
    v[i + ACTIONS.index(action)] = 1.0
    if see_state:
        i += len(ACTIONS)
        for j, s in enumerate(obs["state"]):
            v[i + j] = float(s)
    return v


class GRU(nn.Module):
    def __init__(self, see_state, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.enc = nn.Linear(obs_dim(see_state), HIDDEN)
        self.cell = nn.GRUCell(HIDDEN, HIDDEN)
        self.head = nn.Linear(HIDDEN, len(EVENTS))

    def forward(self, x, h=None):
        e = F.relu(self.enc(x))
        h = self.cell(e, h)
        return self.head(h), h


class Backend:
    def __init__(self, see_state, seed=0):
        self.see_state = see_state
        self.net = GRU(see_state, seed)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.grad_steps = 0
        self.h = None
        self._saved = None

    def begin_sequence(self):
        """A replayed run is starting. Park the live state, start clean."""
        self._saved = self.h
        self.h = None

    def _xy(self, item):
        return encode(item[0], item[1], self.see_state).unsqueeze(0), \
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
            self.grad_steps += 1
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
            g["lr"] = max(1e-5, g["lr"] * 0.5)

    def evaluate(self, test, episode=EPISODE):
        """Overall accuracy plus per-fork accuracy.

        Overall is dominated by moved/blocked/waited, which are 72% of the
        stream and need no memory. The forks are the real measurement.
        """
        self.net.eval()
        h = None
        hit = seen = 0
        fh = {k: 0 for k in FORKS}
        fs = {k: 0 for k in FORKS}
        with torch.no_grad():
            for i, item in enumerate(test):
                if i % episode == 0:
                    h = None
                x, y = self._xy(item)
                logits, h = self.net(x, h)
                right = int(logits.argmax(1).item()) == int(y.item())
                seen += 1
                hit += right
                fork = FORK_OF.get(item[2])
                if fork:
                    fs[fork] += 1
                    fh[fork] += right
        return (100.0 * hit / seen,
                {k: (100.0 * fh[k] / fs[k] if fs[k] else None)
                 for k in FORKS})


def walk(layout_seed, walk_seed, n, see_state, episode=EPISODE):
    rng = random.Random(walk_seed)
    world = BigWorld(layout_seed)
    out = []
    for i in range(n):
        if i % episode == 0:
            world.reset()
        obs = world.observe(hide_state=not see_state)
        action = rng.choice(ACTIONS)
        _, event = world.step(action)
        out.append((obs, action, event))
    return out


ARMS = [
    ("frozen",    False, None),
    ("ceiling",   True,  None),
    ("online",    False, None),
    ("moments",   False, 1),
    ("sequences", False, SEQ_LEN),
]


def run(name, see_state, seq_len, seed):
    train = walk(seed, seed * 100 + 1, STEPS, see_state)
    test = walk(seed, 90000 + seed, TEST_STEPS, see_state)

    b = Backend(see_state, seed)
    t0 = time.time()

    if name == "frozen":
        acc, forks = b.evaluate(test)
        return dict(acc=acc, forks=forks, grad_steps=0, minutes=0.0,
                    alive=True)

    layer = None
    if seq_len is not None:
        # moments does the same number of replay updates as sequences, so
        # only the ORDERING differs. The compute-matched comparison from the
        # small world found ordering worth +15.5 and volume only +4.0.
        count = SEQ_COUNT if seq_len > 1 else SEQ_LEN * SEQ_COUNT
        layer = StabilityLayer(
            b, canary=train[:40], seed=seed,
            window=200, warmup=30, top_fraction=0.50,
            coherence_veto=1e9, loss_floor=0.02, steps_per_update=1,
            rehearse_per_item=5, rehearse_count=count, rehearse_steps=1,
            anchor_size=100, buffer_size=500, sequence_len=seq_len,
            replay_policy="uniform",
            guard=True, guard_per_item=2000, canary_tolerance=0.5)

    for i, item in enumerate(train):
        if i % EPISODE == 0:
            b.reset_state()
        if layer is not None:
            layer.observe(item)
        else:
            b.update(item, 1)

    b.reset_state()
    acc, forks = b.evaluate(test)
    alive = all(torch.isfinite(p).all() for p in b.net.parameters())
    out = dict(acc=acc, forks=forks, grad_steps=b.grad_steps,
               rollbacks=layer.summary()["rollbacks"] if layer else 0,
               minutes=(time.time() - t0) / 60, alive=alive)
    del b, layer
    return out


if __name__ == "__main__":
    print("checking the world is still fair before spending any training\n")
    worst, worst_name, _, _ = fairness_check(quiet=True)
    print(f"  worst fork: {worst_name} at {worst:.1f}%")
    if worst > 75:
        print("\nABORT: a memoryless model could score well without "
              "remembering.")
        raise SystemExit
    print()

    print(f"{STEPS} steps, {len(SEEDS)} seeds, 25x25 world, "
          f"{len(EVENTS)} outcomes, 6 hidden variables")
    print(f"network {obs_dim(False)}->{HIDDEN}->{len(EVENTS)}, "
          f"random init, no pretraining\n")

    results = {}
    for name, see_state, seq_len in ARMS:
        print(f"--- {name} ---")
        runs = []
        for seed in SEEDS:
            r = run(name, see_state, seq_len, seed)
            runs.append(r)
            print(f"  seed {seed}: overall {r['acc']:5.2f}%  "
                  f"alive {r['alive']}  grad {r['grad_steps']}  "
                  f"{r['minutes']:.1f}m")
        results[name] = runs
        print()

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    fork_names = list(FORKS)
    print("=" * 96)
    print(f"{'arm':>11} {'overall':>8}   " +
          "  ".join(f"{k[:9]:>9}" for k in fork_names))
    print("-" * 96)
    for name, _, _ in ARMS:
        runs = results[name]
        cells = "  ".join(
            f"{mean([r['forks'][k] for r in runs]):>8.1f}%"
            for k in fork_names)
        print(f"{name:>11} {mean([r['acc'] for r in runs]):>7.1f}%   {cells}")
    print("=" * 96)

    simple = ["locked door", "dark cell", "ice", "heavy door"]
    conj = ["chasm", "cold room"]

    print("\nsimple forks (one hidden variable) vs conjunctive forks (two):")
    print(f"{'arm':>11} {'simple':>9} {'conjunctive':>13} {'gap':>7}")
    print("-" * 44)
    for name, _, _ in ARMS:
        runs = results[name]
        s = mean([mean([r["forks"][k] for r in runs]) for k in simple])
        c = mean([mean([r["forks"][k] for r in runs]) for k in conj])
        print(f"{name:>11} {s:>8.1f}% {c:>12.1f}% {s - c:>+6.1f}")

    seq = [mean([r["forks"][k] for k in fork_names])
           for r in results["sequences"]]
    mom = [mean([r["forks"][k] for k in fork_names])
           for r in results["moments"]]
    on = [mean([r["forks"][k] for k in fork_names])
          for r in results["online"]]
    wins = lambda a, b: sum(1 for x, y in zip(a, b) if x > y)

    print(f"\nmean across all six forks:")
    print(f"  sequences minus moments: "
          f"{mean(seq) - mean(mom):+6.1f}  (wins {wins(seq, mom)}/{len(SEEDS)})")
    print(f"  sequences minus online : "
          f"{mean(seq) - mean(on):+6.1f}  (wins {wins(seq, on)}/{len(SEEDS)})")

    print("""
Three questions.

  DOES IT SURVIVE? Compare every arm against frozen. If the learning arms
      are near the floor, the world broke them and the mechanisms do not
      scale as they stand.

  DOES ORDERING STILL WIN? sequences against moments. In the small world
      that was +15.5 at matched replay budget. If it holds here, the
      mechanism generalises. If it vanishes, it was a property of a task
      with one thing to remember.

  CONJUNCTIONS. If the gap between simple and conjunctive forks is large,
      the system learns flags but not combinations. That is a real limit
      and the most likely thing to be genuinely new here, since the small
      world never tested it.
""")
    with open("bigrun.json", "w") as f:
        json.dump({k: [dict(acc=r["acc"], forks=r["forks"]) for r in v]
                   for k, v in results.items()}, f, indent=2)
    print("wrote bigrun.json")