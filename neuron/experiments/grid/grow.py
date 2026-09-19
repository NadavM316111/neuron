"""Milestone one: a network from random weights that stays alive.

No pretraining. No training run. A small network starts from noise and
learns to predict what the world does, one step at a time, from a stream
it cannot revisit.

Prediction error is the importance signal. Nothing inside a sentence tells
you it matters, which defeated two attempts at this in text. Here the world
tells you by surprising you.

Fixed 20 Aug: the previous version set warmup=50 against the default
window=40, which meant the layer never left warmup and the gate fired zero
times in 10,000 steps. All the "guarded" learning came from rehearsal.
stability.py now raises on that configuration; here the window is widened
so the gate has real history to judge against.

Three arms:
  frozen   random weights, never trained. The floor.
  raw      learns from every step. No protection.
  guarded  the same stream through stability.py.

Milestone 1: runs 10,000 steps without collapsing.
Milestone 2: beats the frozen network.

Then read the per-event columns. Overall accuracy is dominated by "moved"
and "blocked", which are 99.5% of the stream. The rare events are the
informative ones.
"""

import json
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from world import GridWorld, ACTIONS, EVENTS
from stability import StabilityLayer


STEPS = 10000
EPISODE = 200                 # reset this often so rare events recur
HIDDEN = 64
LR = 3e-4
SEEDS = [0, 1, 2]
N_CELLS = 5                   # empty, wall, door, key, locked
CHECK_EVERY = 1000

WINDOW = 200                  # rolling history the gate judges against
WARMUP = 30                   # must be < WINDOW or the gate never fires


def encode(obs, action):
    """Observation and action as a flat vector. 9 cells one-hot, plus the
    key flag, plus the action one-hot."""
    v = torch.zeros(9 * N_CELLS + 1 + len(ACTIONS))
    for i, cell in enumerate(obs["patch"]):
        v[i * N_CELLS + cell] = 1.0
    v[9 * N_CELLS] = float(obs["has_key"])
    v[9 * N_CELLS + 1 + ACTIONS.index(action)] = 1.0
    return v


INPUT_DIM = 9 * N_CELLS + 1 + len(ACTIONS)


class Predictor(nn.Module):
    """Random weights. Nothing has ever been trained into this."""

    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, len(EVENTS)),
        )

    def forward(self, x):
        return self.net(x)


class WorldBackend:
    """The four callbacks stability.py needs. An item is one moment:
    (observation, action, what actually happened)."""

    def __init__(self, seed=0):
        self.net = Predictor(seed)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.grad_steps = 0

    def _xy(self, item):
        obs, action, event = item
        return encode(obs, action).unsqueeze(0), \
            torch.tensor([EVENTS.index(event)])

    def score(self, item):
        """Surprise is how wrong the prediction was. No coherence signal
        here, so the veto is off."""
        self.net.eval()
        with torch.no_grad():
            x, y = self._xy(item)
            loss = F.cross_entropy(self.net(x), y)
        return float(loss.item()), None

    def update(self, item, steps):
        self.net.train()
        last = None
        for _ in range(steps):
            x, y = self._xy(item)
            loss = F.cross_entropy(self.net(x), y)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            last = float(loss.item())
            self.grad_steps += 1
        return last

    def snapshot(self):
        return {k: v.detach().clone()
                for k, v in self.net.state_dict().items()}

    def restore(self, state):
        self.net.load_state_dict(state)

    def on_rollback(self):
        for g in self.opt.param_groups:
            g["lr"] = max(1e-5, g["lr"] * 0.5)

    def predict(self, item):
        self.net.eval()
        with torch.no_grad():
            x, _ = self._xy(item)
            return EVENTS[int(self.net(x).argmax(1).item())]


def make_stream(seed, n, episode=EPISODE):
    """Random wandering with periodic resets, so keys respawn and the rare
    events recur instead of happening once."""
    rng = random.Random(seed)
    world = GridWorld(seed)
    out = []
    for i in range(n):
        if i % episode == 0:
            world.reset()
        obs = world.observe()
        action = rng.choice(ACTIONS)
        _, event = world.step(action)
        out.append((obs, action, event))
    return out


def evaluate(backend, test):
    """Overall and per-event accuracy on a held-out stream."""
    correct = 0
    per = {e: [0, 0] for e in EVENTS}
    for item in test:
        truth = item[2]
        guess = backend.predict(item)
        per[truth][1] += 1
        if guess == truth:
            correct += 1
            per[truth][0] += 1
    acc = 100.0 * correct / len(test)
    per_acc = {e: (100.0 * c / n if n else None) for e, (c, n) in per.items()}
    counts = {e: n for e, (_, n) in per.items()}
    return acc, per_acc, counts


def run(mode, seed, train, test):
    backend = WorldBackend(seed)
    t0 = time.time()

    if mode == "frozen":
        acc, per, _ = evaluate(backend, test)
        return dict(acc=acc, per=per, checks=[], grad_steps=0,
                    gate_fires=0, rehearsals=0, rollbacks=0,
                    minutes=0.0, alive=True)

    layer = None
    if mode == "guarded":
        canary = train[:40]           # early moments, replayed as an anchor
        layer = StabilityLayer(
            backend, canary=canary, seed=seed,
            window=WINDOW, warmup=WARMUP,
            top_fraction=0.50, coherence_veto=1e9,
            loss_floor=0.02, steps_per_update=1,
            rehearse_per_item=5, rehearse_count=2, rehearse_steps=1,
            anchor_size=100, buffer_size=500,
            guard=True, guard_per_item=500, canary_tolerance=0.5)

    checks = []
    for i, item in enumerate(train):
        if layer is not None:
            layer.observe(item)
        else:
            backend.update(item, 1)

        if (i + 1) % CHECK_EVERY == 0:
            acc, per, _ = evaluate(backend, test)
            checks.append(dict(step=i + 1, acc=acc,
                               got_key=per.get("got_key"),
                               unlocked=per.get("unlocked")))

    acc, per, _ = evaluate(backend, test)
    s = layer.summary() if layer else {}
    alive = all(torch.isfinite(p).all() for p in backend.net.parameters())

    out = dict(acc=acc, per=per, checks=checks,
               grad_steps=backend.grad_steps,
               gate_fires=s.get("updates", len(train)),
               rehearsals=s.get("rehearsals", 0),
               rollbacks=s.get("rollbacks", 0),
               warmed=s.get("warmed", 0),
               floored=s.get("floored", 0),
               unremarkable=s.get("unremarkable", 0),
               minutes=(time.time() - t0) / 60,
               alive=alive)
    del backend, layer
    return out


if __name__ == "__main__":
    print(f"{STEPS} steps, reset every {EPISODE}, "
          f"network {INPUT_DIM}->{HIDDEN}->{HIDDEN}->{len(EVENTS)}, "
          f"random init, no pretraining")
    print(f"gate: window {WINDOW}, warmup {WARMUP}, top 50%\n")

    # A held-out stream from a different seed, never learned from.
    test = make_stream(999, 3000)
    _, _, counts = evaluate(WorldBackend(0), test)
    print("held-out stream event counts:")
    for e in EVENTS:
        print(f"  {e:>10}: {counts[e]:>5}")
    thin = [e for e in EVENTS if counts[e] < 20]
    if thin:
        print(f"\n  WARNING: too few {', '.join(thin)} to measure. Those "
              f"columns are noise.")
    print()

    results = {}
    for mode in ["frozen", "raw", "guarded"]:
        print(f"--- {mode} ---")
        runs = []
        for seed in SEEDS:
            train = make_stream(seed, STEPS)
            r = run(mode, seed, train, test)
            runs.append(r)
            extra = ""
            if mode == "guarded":
                extra = (f"  gate {r['gate_fires']}  grad {r['grad_steps']}  "
                         f"reh {r['rehearsals']}  warm {r['warmed']}  "
                         f"floor {r['floored']}  unrem {r['unremarkable']}  "
                         f"rb {r['rollbacks']}")
            elif mode == "raw":
                extra = f"  grad {r['grad_steps']}"
            print(f"  seed {seed}: acc {r['acc']:5.2f}%  "
                  f"alive {r['alive']}{extra}  {r['minutes']:.1f}m")

        avg = sum(r["acc"] for r in runs) / len(runs)
        per_avg = {}
        for e in EVENTS:
            vals = [r["per"][e] for r in runs if r["per"][e] is not None]
            per_avg[e] = sum(vals) / len(vals) if vals else None
        results[mode] = dict(acc=avg, per=per_avg,
                             alive=all(r["alive"] for r in runs),
                             checks=runs[0]["checks"],
                             gate_fires=sum(r["gate_fires"] for r in runs) / len(runs),
                             grad_steps=sum(r["grad_steps"] for r in runs) / len(runs))
        print()

    print("=" * 76)
    print(f"{'arm':>9} {'accuracy':>9} {'alive':>6} {'grad':>7}   " +
          "  ".join(f"{e:>9}" for e in EVENTS))
    print("-" * 76)
    for mode in ["frozen", "raw", "guarded"]:
        r = results[mode]
        cells = "  ".join(
            f"{r['per'][e]:>8.0f}%" if r["per"][e] is not None else f"{'-':>9}"
            for e in EVENTS)
        print(f"{mode:>9} {r['acc']:>8.2f}% {str(r['alive']):>6} "
              f"{r['grad_steps']:>7.0f}   {cells}")
    print("=" * 76)

    print("\nprogress of the guarded arm, seed 0:")
    for c in results["guarded"]["checks"]:
        g = f"{c['got_key']:.0f}%" if c["got_key"] is not None else "-"
        u = f"{c['unlocked']:.0f}%" if c["unlocked"] is not None else "-"
        print(f"  step {c['step']:>6}: acc {c['acc']:5.2f}%  "
              f"got_key {g:>5}  unlocked {u:>5}")

    print(f"""
The gate fired {results['guarded']['gate_fires']:.0f} times on average this
run. Last time it was zero, because warmup exceeded window and the layer
never left warmup. If it is still zero, something else is wrong.

MILESTONE 1: the alive column. A network grown from random weights
surviving 10,000 sequential steps.

MILESTONE 2: raw or guarded beating frozen.

THE REAL QUESTION: the got_key and unlocked columns. Overall accuracy is
dominated by moved and blocked, which are 99.5% of the stream, so a system
can score 99% while understanding nothing rare. The gate exists to catch
rare informative moments. If guarded does no better than raw on those
columns, it is not doing its job here.
""")
    with open("grow.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote grow.json")