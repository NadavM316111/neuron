"""Rung 3: does the system survive imperfect senses?

The world's rules are unchanged. Only the seeing is different: continuous
readings, gaussian noise, overlapping signatures, occasional dropout. A
perfect reader misidentifies about 16% of cells.

Two questions, and the second is the one that could end the approach.

  1. DOES IT STILL LEARN? Compare clean against noisy on the same rules.
     Some drop is expected and fine. Collapse is not.

  2. DOES THE GATE SURVIVE? The gate fires on SURPRISE, and noise IS
     surprise. If the gate saturates and fires on everything, it stops
     selecting. Worse, items enter the rehearsal buffer ONLY when the gate
     REJECTS them, so a saturated gate starves rehearsal too. That is a
     specific, project-threatening failure that has never been tested.

Arms:
  clean-online     the discrete world, no protection. The reference.
  noisy-ceiling    noisy senses, but reads the hidden state. Shows how much
                   of any drop is the SEEING rather than the remembering.
  noisy-online     noisy senses, no protection.
  noisy-guarded    noisy senses through stability.py, with sequence replay.

The gate's own statistics are printed for the guarded arm. That is the real
measurement here, more than the accuracy: how often it fired, how often it
rejected, and whether the rehearsal buffer filled at all.
"""

import json
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from bigworld import BigWorld, ACTIONS, EVENTS, FORKS
from noisy import NoisyWorld, FEATURES, NOISE, DROPOUT, overlap_check
from stability import StabilityLayer


STEPS = 60000
EPISODE = 400
HIDDEN = 128
LR = 3e-4
SEEDS = [0, 1]
N_TERRAIN = 14
PATCH = 25
N_STATE = 7
TEST_STEPS = 8000
SEQ_LEN = 10
SEQ_COUNT = 2

SIMPLE = ["locked door", "dark cell", "ice", "heavy door"]
CONJ = ["chasm", "cold room"]

FORK_OF = {}
for name, (yes, no) in FORKS.items():
    for e in yes + no:
        FORK_OF[e] = name


def obs_dim(noisy, see_state):
    per_cell = FEATURES if noisy else N_TERRAIN
    return PATCH * per_cell + len(ACTIONS) + (N_STATE if see_state else 0)


def encode(obs, action, noisy, see_state):
    v = torch.zeros(obs_dim(noisy, see_state))
    if noisy:
        i = 0
        for cell in obs["patch"]:
            for f in cell:
                v[i] = f
                i += 1
    else:
        for j, cell in enumerate(obs["patch"]):
            v[j * N_TERRAIN + cell] = 1.0
        i = PATCH * N_TERRAIN
    v[i + ACTIONS.index(action)] = 1.0
    if see_state:
        i += len(ACTIONS)
        for j, s in enumerate(obs["state"]):
            v[i + j] = float(s)
    return v


class GRU(nn.Module):
    def __init__(self, noisy, see_state, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.enc = nn.Linear(obs_dim(noisy, see_state), HIDDEN)
        self.cell = nn.GRUCell(HIDDEN, HIDDEN)
        self.head = nn.Linear(HIDDEN, len(EVENTS))

    def forward(self, x, h=None):
        e = F.relu(self.enc(x))
        h = self.cell(e, h)
        return self.head(h), h


class Backend:
    def __init__(self, noisy, see_state, seed=0):
        self.noisy = noisy
        self.see_state = see_state
        self.net = GRU(noisy, see_state, seed)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.h = None
        self._saved = None

    def begin_sequence(self):
        self._saved = self.h
        self.h = None

    def _xy(self, item):
        return encode(item[0], item[1], self.noisy,
                      self.see_state).unsqueeze(0), \
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
            g["lr"] = max(1e-5, g["lr"] * 0.5)

    def evaluate(self, test):
        self.net.eval()
        h = None
        fh = {k: 0 for k in FORKS}
        fs = {k: 0 for k in FORKS}
        hit = seen = 0
        with torch.no_grad():
            for i, item in enumerate(test):
                if i % EPISODE == 0:
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


def walk(layout_seed, walk_seed, n, noisy, see_state, episode=EPISODE):
    rng = random.Random(walk_seed)
    world = (NoisyWorld(layout_seed) if noisy else BigWorld(layout_seed))
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
    ("clean-online",  False, False, False),
    ("noisy-ceiling", True,  True,  False),
    ("noisy-online",  True,  False, False),
    ("noisy-guarded", True,  False, True),
]


def run(name, noisy, see_state, guarded, seed):
    train = walk(seed, seed * 100 + 1, STEPS, noisy, see_state)
    test = walk(seed, 90000 + seed, TEST_STEPS, noisy, see_state)

    b = Backend(noisy, see_state, seed)
    t0 = time.time()
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

    for i, item in enumerate(train):
        if i % EPISODE == 0:
            b.reset_state()
        if layer is not None:
            layer.observe(item)
        else:
            b.update(item, 1)

    b.reset_state()
    acc, forks = b.evaluate(test)
    stats = layer.summary() if layer else {}
    out = dict(acc=acc, forks=forks, stats=stats,
               minutes=(time.time() - t0) / 60)
    del b, layer
    return out


if __name__ == "__main__":
    print("Rung 3: imperfect senses.\n")
    print(f"  noise {NOISE}, dropout {DROPOUT:.0%}, "
          f"{FEATURES} features per cell")
    rate = overlap_check(NOISE, quiet=True)
    print(f"  a perfect reader misidentifies {rate:.1f}% of cells\n")

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    results = {}
    for name, noisy, see_state, guarded in ARMS:
        runs = []
        for seed in SEEDS:
            runs.append(run(name, noisy, see_state, guarded, seed))
        results[name] = runs
        s = mean([mean([r["forks"][k] for k in SIMPLE]) for r in runs])
        c = mean([mean([r["forks"][k] for k in CONJ]) for r in runs])
        extra = ""
        if runs[0]["stats"]:
            st = runs[0]["stats"]
            extra = (f"  gate {st['updates']}  reh {st['rehearsals']}  "
                     f"buffer {st['buffer']}")
        print(f"  {name:>14}: overall "
              f"{mean([r['acc'] for r in runs]):5.1f}%  "
              f"simple {s:5.1f}%  conj {c:5.1f}%{extra}  "
              f"{sum(r['minutes'] for r in runs):.1f}m", flush=True)

    print("\n" + "=" * 92)
    print(f"{'arm':>14} {'overall':>8} " +
          "  ".join(f"{k[:9]:>9}" for k in FORKS))
    print("-" * 92)
    for name, _, _, _ in ARMS:
        runs = results[name]
        cells = "  ".join(
            f"{mean([r['forks'][k] for r in runs]):>8.1f}%" for k in FORKS)
        print(f"{name:>14} {mean([r['acc'] for r in runs]):>7.1f}% {cells}")
    print("=" * 92)

    print("\nDID THE GATE SURVIVE THE NOISE?\n")
    st = results["noisy-guarded"][0]["stats"]
    seen = st["seen"]
    fired = st["updates"]
    print(f"  items seen          {seen}")
    print(f"  gate fired          {fired}  ({100.0 * fired / seen:.1f}%)")
    print(f"  rejected as ordinary {st['unremarkable'] + st['below_bar']}")
    print(f"  skipped as known    {st['floored']}")
    print(f"  rehearsals          {st['rehearsals']}")
    print(f"  sequences stored    {st['sequences']}")
    print(f"  buffer at end       {st['buffer']} of 500")
    print(f"  anchor at end       {st['anchor']} of 100")
    print(f"  rollbacks           {st['rollbacks']}")

    fire_pct = 100.0 * fired / seen
    print()
    if fire_pct > 80:
        print("  THE GATE SATURATED. Noise reads as surprise, so almost "
              "everything\n  looks novel and the gate has stopped "
              "selecting. And because items\n  enter the buffer only when "
              "the gate REJECTS them, rehearsal starves\n  with it. This is "
              "the failure mode this rung was built to find.")
    elif st["buffer"] < 50:
        print("  THE BUFFER STARVED. The gate is rejecting too little, so "
              "almost\n  nothing is stored to rehearse. Same coupling "
              "problem, milder form.")
    else:
        print(f"  THE GATE HELD. It fired on {fire_pct:.0f}% of items and "
              f"the buffer filled\n  to {st['buffer']}, so selection and "
              f"rehearsal both still function\n  under noise.")

    print("""
Read the three noisy arms against clean-online.

  noisy-ceiling near clean-online -> the drop is about SEEING, not
      remembering. The senses got harder; the mechanisms are fine.
  noisy-online well below noisy-ceiling -> noise makes REMEMBERING harder
      too, on top of making seeing harder.
  noisy-guarded below noisy-online -> the stability layer is actively
      hurting under noise, which would be the clearest possible signal that
      a surprise-driven gate does not survive a noisy stream.

The gate statistics matter more than the accuracies here. A gate that fires
on everything has stopped being a gate, whatever the score says.
""")
    with open("noisyrun.json", "w") as f:
        json.dump({k: [dict(acc=r["acc"], forks=r["forks"],
                            stats=r["stats"]) for r in v]
                   for k, v in results.items()}, f, indent=2)
    print("wrote noisyrun.json")