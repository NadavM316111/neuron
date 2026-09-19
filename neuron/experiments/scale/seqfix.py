"""Does sequence replay get better once its own bug is fixed?

stability.py replays a stored run by calling backend.update() once per item.
Every backend detaches the hidden state after each update, so a ten-step
trajectory was replayed as ten unconnected single steps and the gradient
could never link step one to step ten.

That defeats the entire point of storing a SEQUENCE rather than snapshots.
And it is the same defect found on 23 Aug in the live training loop, where
fixing it closed 23% of the memory gap in the grid world.

Every number sequence replay has produced — the +15.5 in the small world,
the disappearance in the big world, the weather wins, the 7B result — was
measured with this in place. They understate it by an unknown amount.

Three arms on the grid world, which is where sequence replay was first
measured:

  moments      single-moment replay. The control.
  seq-broken   sequence replay, per-step updates. What has been measured
               all along.
  seq-fixed    sequence replay, losses accumulated across the run and
               applied once, so credit flows through the trajectory.

Replay budgets are matched: moments replays 20 isolated items, the sequence
arms replay 2 runs of 10. The only difference between the two sequence arms
is whether the gradient can span the run.
"""

import json
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from bigworld import BigWorld, ACTIONS, EVENTS, FORKS
from stability import StabilityLayer


STEPS = 40000
EPISODE = 400
HIDDEN = 128
LR = 3e-4
SEEDS = [0, 1, 2]
N_TERRAIN = 14
PATCH = 25
TEST_STEPS = 8000
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
    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.enc = nn.Linear(obs_dim(), HIDDEN)
        self.cell = nn.GRUCell(HIDDEN, HIDDEN)
        self.head = nn.Linear(HIDDEN, len(EVENTS))

    def forward(self, x, h=None):
        e = F.relu(self.enc(x))
        h = self.cell(e, h)
        return self.head(h), h


class Backend:
    """Exposes update_sequence(), which is what the fix uses.

    `spanning` controls whether that method actually spans the run or
    falls back to per-step updates, so the two sequence arms differ in
    exactly one thing.
    """

    def __init__(self, seed=0, spanning=True):
        self.net = GRU(seed)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.spanning = spanning
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

    def update_sequence(self, items, steps):
        """Replay a run so the gradient spans it.

        With spanning=False this deliberately reproduces the old broken
        behaviour, so the comparison isolates the fix.
        """
        if not self.spanning:
            n = 0
            for it in items:
                if self.update(it, steps) is not None:
                    n += 1
            return n

        self.net.train()
        for _ in range(steps):
            h = None
            losses = []
            for it in items:
                x, y = self._xy(it)
                logits, h = self.net(x, h)
                losses.append(F.cross_entropy(logits, y))
            self.opt.zero_grad()
            torch.stack(losses).mean().backward()
            self.opt.step()
        return len(items)

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


# name -> (sequence_len, rehearse_count, spanning backend)
ARMS = {
    "moments":    (1, SEQ_LEN * SEQ_COUNT, False),
    "seq-broken": (SEQ_LEN, SEQ_COUNT, False),
    "seq-fixed":  (SEQ_LEN, SEQ_COUNT, True),
}


def run(name, seed):
    seq_len, count, spanning = ARMS[name]
    train = walk(seed, seed * 100 + 1, STEPS)
    test = walk(seed, 90000 + seed, TEST_STEPS)

    b = Backend(seed, spanning)
    layer = StabilityLayer(
        b, canary=train[:40], seed=seed,
        window=200, warmup=30, top_fraction=0.50,
        coherence_veto=1e9, loss_floor=0.02, steps_per_update=1,
        rehearse_per_item=5, rehearse_count=count, rehearse_steps=1,
        anchor_size=100, buffer_size=500, sequence_len=seq_len,
        replay_policy="uniform",
        guard=True, guard_per_item=2000, canary_tolerance=0.5)

    t0 = time.perf_counter()
    for i, item in enumerate(train):
        if i % EPISODE == 0:
            b.reset_state()
        layer.observe(item)
    secs = time.perf_counter() - t0

    b.reset_state()
    acc, forks = evaluate(b.net, test)
    st = layer.summary()
    out = dict(acc=acc, forks=forks, seconds=secs,
               rehearsals=st["rehearsals"], rollbacks=st["rollbacks"])
    del b, layer
    return out


if __name__ == "__main__":
    print("Does sequence replay improve once its own bug is fixed?")
    print(f"{STEPS} steps, {len(SEEDS)} seeds, matched replay budgets\n")

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    results = {}
    for name in ARMS:
        runs = [run(name, s) for s in SEEDS]
        results[name] = runs
        s = mean([mean([r["forks"][k] for k in SIMPLE]) for r in runs])
        c = mean([mean([r["forks"][k] for k in CONJ]) for r in runs])
        secs = mean([r["seconds"] for r in runs])
        print(f"  {name:>11}: simple {s:5.1f}%  conj {c:5.1f}%  "
              f"reh {mean([r['rehearsals'] for r in runs]):.0f}  "
              f"{secs / 60:.1f}m", flush=True)

    print("\n" + "=" * 80)
    print(f"{'arm':>11} {'overall':>9} {'simple':>8} {'conj':>8} "
          f"{'minutes':>9}")
    print("-" * 80)
    for name in ARMS:
        runs = results[name]
        print(f"{name:>11} "
              f"{mean([r['acc'] for r in runs]):>8.2f}% "
              f"{mean([mean([r['forks'][k] for k in SIMPLE]) for r in runs]):>7.1f}% "
              f"{mean([mean([r['forks'][k] for k in CONJ]) for r in runs]):>7.1f}% "
              f"{mean([r['seconds'] for r in runs]) / 60:>8.1f}")
    print("=" * 80)

    print("\nper-fork detail:")
    print(f"{'arm':>11} " + "  ".join(f"{k[:9]:>9}" for k in FORKS))
    print("-" * 80)
    for name in ARMS:
        runs = results[name]
        print(f"{name:>11} " + "  ".join(
            f"{mean([r['forks'][k] for r in runs]):>8.1f}%" for k in FORKS))

    br_s = mean([mean([r["forks"][k] for k in SIMPLE])
                 for r in results["seq-broken"]])
    fx_s = mean([mean([r["forks"][k] for k in SIMPLE])
                 for r in results["seq-fixed"]])
    br_c = mean([mean([r["forks"][k] for k in CONJ])
                 for r in results["seq-broken"]])
    fx_c = mean([mean([r["forks"][k] for k in CONJ])
                 for r in results["seq-fixed"]])
    mo_s = mean([mean([r["forks"][k] for k in SIMPLE])
                 for r in results["moments"]])

    wins = sum(1 for a, b in zip(
        [mean([r["forks"][k] for k in SIMPLE]) for r in results["seq-fixed"]],
        [mean([r["forks"][k] for k in SIMPLE])
         for r in results["seq-broken"]]) if a > b)

    print(f"\nTHE FIX (seq-fixed minus seq-broken):")
    print(f"  simple forks      {fx_s - br_s:+6.2f}  "
          f"(wins {wins}/{len(SEEDS)} seeds)")
    print(f"  conjunctive forks {fx_c - br_c:+6.2f}")
    print(f"\nSEQUENCE REPLAY vs SINGLE MOMENTS, with the fix in:")
    print(f"  simple forks      {fx_s - mo_s:+6.2f}")

    print("""
Every result sequence replay has produced was measured with the bug in
place. This says by how much they were wrong.

  seq-fixed clearly above seq-broken -> the mechanism was being undersold,
      and the weather and 7B numbers should be re-run before publishing,
      since they will look better.
  no difference -> the per-step replay was already doing the work, the
      trajectory structure was never load-bearing, and the mechanism is
      simpler than described. Also worth knowing, and it would mean nothing
      needs re-running.
  seq-fixed worse -> spanning the replay introduces something harmful, and
      the current behaviour should stay as it is.
""")
    with open("seqfix.json", "w") as f:
        json.dump({k: [dict(acc=r["acc"], forks=r["forks"],
                            seconds=r["seconds"]) for r in v]
                   for k, v in results.items()}, f, indent=2)
    print("wrote seqfix.json")