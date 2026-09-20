"""Does a PERSISTENT drive beat an instantaneous one at choosing what to do?

acting.py established three things about self-generated data on this world:
value-seeking starves its own learning signal, exploration is a
precondition for consequence being usable at all, and curiosity is an
elegant way to get exploration for free out of the forward pass the policy
already needs.

Every one of those policies is INSTANTANEOUS. greedy reads the current
predicted outcome. curious reads the current predicted entropy. Nothing
carries anything between steps. The agent has no memory of having been
wrong.

That is the same gap valence closed on the replay side (core/valence.py,
20 Sep): selection was being recomputed from scratch every time, and
accumulating across a unit's life beat recomputing by a small consistent
margin.

THE SPECIFIC HOLE CURIOSITY HAS. Entropy measures how unsure the model
SAYS it is. A model that is CONFIDENTLY WRONG has low entropy, so curiosity
walks past it. That is exactly the state the agent is in immediately after
the rule reverses: it predicts berry-good with high confidence and is
wrong every time. Curiosity is blind at the precise moment the world has
changed, which is the moment that matters most.

Accumulated actual prediction error is not blind to it. Being wrong raises
the drive whether or not the model expected to be wrong.

THE DRIVE. One scalar per kind of thing the agent can move onto (empty,
berry, fungus), updated from the surprise actually experienced on eating
it, as an EMA:

    drive[cell] = decay * drive[cell] + (1 - decay) * surprise

It rises where the world keeps violating expectations and decays where
outcomes have become predictable, so the agent stops spending steps on
things it has understood. The policy picks the action whose target cell
carries the highest drive.

Calling this a "want" is a functional description and nothing more. It is a
persistent scalar that steers behaviour. It is not a feeling and should
never be written up as one.

FIVE ARMS. passive, greedy, explore and curious are unchanged from
acting.py so the numbers are comparable; driven is the new one.

WHAT WOULD MAKE THIS INTERESTING, in order of how much it would mean:

  driven beats curious on the PHASE-2 rule
      The confidently-wrong hypothesis is right. Persistent error finds the
      reversal faster than predicted entropy does.

  driven's fungus coverage RISES in phase 2 by more than curious's
      The mechanism, visible directly. The drive is pulling the agent
      toward what changed.

  driven matches curious
      Persistence buys nothing here and the instantaneous signal was
      sufficient. Which is roughly what valence found on replay, so it is
      the outcome to expect rather than the one to hope for.

  driven starves like greedy
      The drive collapsed onto one cell type and the agent stopped sampling
      the rest. Watch the coverage column before reading any score.

THE LEARNABILITY GATE FROM acting.py IS KEPT. An arm that does not clear
the majority-class baseline learned nothing and no forgetting or coverage
conclusion is read off it.

    python drive.py                    # ~3 minutes
    python drive.py --steps 8000 --seeds 3
"""

import argparse
import json
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))

from stability import StabilityLayer          # noqa: E402


SIZE = 7
ACTIONS = ["up", "down", "left", "right"]
EMPTY, BERRY, FUNGUS = 0, 1, 2
CELLS = 3
NOTHING, GOOD, BAD = 0, 1, 2
OUTCOMES = 3

# Which patch index each action moves onto. The patch is the 3x3 around the
# agent, flattened row-major, so index 4 is the agent itself.
TARGET = {"up": 1, "down": 7, "left": 3, "right": 5}

DRIVE_DECAY = 0.9
DRIVE_INIT = 1.0      # unvisited cell types start attractive, not neutral


class World:
    """A grid of berries and fungus. Moving onto one eats it.

    The rule that decides which is good reverses between phases, so the
    stream contradicts itself the way the language drift stream did.
    """

    def __init__(self, seed, berry_good=True):
        self.rng = random.Random(seed)
        self.berry_good = berry_good
        self.reset()

    def reset(self):
        self.grid = [[EMPTY] * SIZE for _ in range(SIZE)]
        for _ in range(SIZE * 3):
            self._place(BERRY)
            self._place(FUNGUS)
        self.x = self.rng.randrange(SIZE)
        self.y = self.rng.randrange(SIZE)

    def _place(self, kind):
        for _ in range(50):
            x, y = self.rng.randrange(SIZE), self.rng.randrange(SIZE)
            if self.grid[y][x] == EMPTY:
                self.grid[y][x] = kind
                return

    def observe(self):
        patch = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                x, y = self.x + dx, self.y + dy
                patch.append(self.grid[y][x]
                             if 0 <= x < SIZE and 0 <= y < SIZE else EMPTY)
        return patch

    def step(self, action):
        dx, dy = {"up": (0, -1), "down": (0, 1),
                  "left": (-1, 0), "right": (1, 0)}[action]
        nx, ny = self.x + dx, self.y + dy
        if not (0 <= nx < SIZE and 0 <= ny < SIZE):
            return NOTHING, EMPTY
        self.x, self.y = nx, ny
        cell = self.grid[ny][nx]
        if cell == EMPTY:
            return NOTHING, EMPTY
        self.grid[ny][nx] = EMPTY
        self._place(cell)
        good = (cell == BERRY) == self.berry_good
        return (GOOD if good else BAD), cell


def encode(patch, action):
    v = torch.zeros(9 * CELLS + len(ACTIONS))
    for i, c in enumerate(patch):
        v[i * CELLS + c] = 1.0
    v[9 * CELLS + ACTIONS.index(action)] = 1.0
    return v.unsqueeze(0)


class Net(nn.Module):
    def __init__(self, hidden=64, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.enc = nn.Linear(9 * CELLS + len(ACTIONS), hidden)
        self.cell = nn.GRUCell(hidden, hidden)
        self.head = nn.Linear(hidden, OUTCOMES)

    def forward(self, x, h=None):
        return self.head(self.cell(F.relu(self.enc(x)), h)), h


class Backend:
    """Adapts the net to the stability layer's five methods.

    Also carries the drive, because the drive is updated from the same
    surprise the layer reads and there is no reason to compute it twice.
    """

    def __init__(self, hidden=64, seed=0, lr=3e-4,
                 drive_decay=DRIVE_DECAY):
        self.net = Net(hidden, seed)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.h = None

        self.drive = {EMPTY: DRIVE_INIT, BERRY: DRIVE_INIT,
                      FUNGUS: DRIVE_INIT}
        self.drive_decay = drive_decay
        self.drive_updates = {EMPTY: 0, BERRY: 0, FUNGUS: 0}
        self.drive_trace = []

    def _loss(self, item, grad=False):
        patch, action, outcome = item
        x = encode(patch, action)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            logits, _ = self.net(x, None)
        return F.cross_entropy(logits, torch.tensor([outcome]))

    def score(self, item):
        self.net.eval()
        return self._loss(item).item(), None

    def update(self, item, steps):
        self.net.train()
        last = None
        for _ in range(steps):
            loss = self._loss(item, grad=True)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            last = loss.item()
        return last

    def snapshot(self):
        return {k: v.detach().clone()
                for k, v in self.net.state_dict().items()}

    def restore(self, state):
        self.net.load_state_dict(state)

    # ---------- the drive ----------

    def feel(self, cell, surprise):
        """Update the drive for a kind of thing from experienced surprise.

        This is the whole mechanism. Being wrong about a cell type raises
        its drive whether or not the model EXPECTED to be wrong, which is
        the difference from curiosity. A cell type whose outcomes have
        become predictable decays back down and stops attracting steps.
        """
        d = self.drive[cell]
        self.drive[cell] = (self.drive_decay * d
                            + (1.0 - self.drive_decay) * surprise)
        self.drive_updates[cell] += 1

    def act(self, patch, mode, epsilon):
        """Choose an action.

        value     prefer the action least likely to be bad. Converges to
                  standing still on this world and starves the signal.
        curious   prefer the action whose predicted outcome has the highest
                  entropy. Blind to confidently-wrong predictions.
        driven    prefer the action whose TARGET CELL carries the highest
                  accumulated drive. Sees confidently-wrong, because the
                  drive is built from errors that actually happened rather
                  than from the model's own claimed uncertainty.
        """
        if random.random() < epsilon:
            return random.choice(ACTIONS)

        if mode == "driven":
            best, choice = None, ACTIONS[0]
            for a in ACTIONS:
                cell = patch[TARGET[a]]
                score = self.drive[cell]
                if best is None or score > best:
                    best, choice = score, a
            return choice

        self.net.eval()
        best, choice = None, ACTIONS[0]
        with torch.no_grad():
            for a in ACTIONS:
                p = F.softmax(self.net(encode(patch, a), None)[0], dim=1)[0]
                if mode == "curious":
                    score = -(p * torch.log(p.clamp(min=1e-9))).sum().item()
                else:
                    score = p[GOOD].item() - p[BAD].item()
                if best is None or score > best:
                    best, choice = score, a
        return choice


# --------------------------------------------------------------- probes

def make_probes(seed, n, berry_good):
    rng = random.Random(10_000 + seed)
    out = []
    while len(out) < n:
        patch = [rng.choice([EMPTY, EMPTY, BERRY, FUNGUS]) for _ in range(9)]
        action = rng.choice(ACTIONS)
        cell = patch[TARGET[action]]
        if cell == EMPTY:
            outcome = NOTHING
        else:
            outcome = GOOD if (cell == BERRY) == berry_good else BAD
        out.append((patch, action, outcome))
    return out


def accuracy(backend, probes):
    backend.net.eval()
    hit = 0
    with torch.no_grad():
        for patch, action, outcome in probes:
            logits, _ = backend.net(encode(patch, action), None)
            hit += int(logits.argmax(1).item() == outcome)
    return 100.0 * hit / len(probes)


# ------------------------------------------------------------------ run

def run(name, mode, retention, steps, seed, epsilon, probes1, probes2,
        drive_decay=DRIVE_DECAY):
    started = time.time()
    backend = Backend(seed=seed, drive_decay=drive_decay)
    layer = None
    if retention:
        canary = make_probes(seed + 500, 12, berry_good=True)
        layer = StabilityLayer(
            backend, canary=canary, seed=seed,
            window=200, warmup=30, top_fraction=0.50,
            loss_floor=0.15, steps_per_update=1,
            anchor_size=60, buffer_size=300, sequence_len=8,
            rehearse_per_item=24, rehearse_count=2, rehearse_steps=1,
            replay_policy="old",
            guard=True, guard_per_item=max(50, steps // 8),
            canary_tolerance=0.40)

    marks, coverage, updates = [], [], 0
    drive_marks = []

    for phase, berry_good in enumerate([True, False], start=1):
        world = World(seed * 100 + phase, berry_good=berry_good)
        seen = {BERRY: 0, FUNGUS: 0}
        for i in range(steps):
            if i % 200 == 0:
                world.reset()
            patch = world.observe()
            action = (random.choice(ACTIONS) if mode == "passive"
                      else backend.act(patch, mode, epsilon))
            target_cell = patch[TARGET[action]]
            outcome, cell = world.step(action)
            if cell in seen:
                seen[cell] += 1
            item = (patch, action, outcome)

            # The drive is updated for every arm, so its trace is
            # comparable across policies even where nothing reads it.
            # Scored BEFORE the update, so it is the surprise the agent
            # actually experienced rather than what it knows afterwards.
            s, _ = backend.score(item)
            backend.feel(target_cell, s)

            if layer is not None:
                if layer.observe(item).get("learned"):
                    updates += 1
            else:
                backend.update(item, 1)
                updates += 1
        coverage.append(seen)
        marks.append((phase, accuracy(backend, probes1),
                      accuracy(backend, probes2)))
        drive_marks.append(dict(backend.drive))

    s = layer.summary() if layer else dict(rollbacks=0, rehearsals=0,
                                           health=0.0)
    return dict(name=name, seed=seed, mode=mode, retention=retention,
                marks=marks, coverage=coverage, updates=updates,
                drive=drive_marks,
                rollbacks=s["rollbacks"], rehearsals=s["rehearsals"],
                health=s["health"], seconds=time.time() - started)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000,
                    help="steps per phase")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--probes", type=int, default=300)
    ap.add_argument("--drive-decay", type=float, default=DRIVE_DECAY)
    ap.add_argument("--out", default="drive.json")
    args = ap.parse_args()

    # name, mode, epsilon, retention. The first four are acting.py
    # unchanged so the numbers stay comparable.
    arms = [("passive",      "passive", 0.0,  True),
            ("passive-none", "passive", 0.0,  False),
            ("greedy",       "value",   0.1,  True),
            ("explore",      "value",   0.5,  True),
            ("curious",      "curious", 0.05, True),
            ("driven",       "driven",  0.05, True)]

    ref = make_probes(0, args.probes, berry_good=True)
    counts = {}
    for _, _, o in ref:
        counts[o] = counts.get(o, 0) + 1
    baseline = 100.0 * max(counts.values()) / len(ref)

    print(f"  {args.steps} steps per phase, 2 phases, {args.seeds} seed"
          f"{'s' if args.seeds > 1 else ''}")
    print(f"  phase 1: berry good.  phase 2: the rule reverses.")
    print(f"  drive decay {args.drive_decay}, one scalar per cell type")
    print(f"  majority-class baseline {baseline:.1f}% — an arm below this "
          f"learned nothing.\n", flush=True)

    runs = []
    for seed in range(args.seeds):
        p1 = make_probes(seed, args.probes, berry_good=True)
        p2 = make_probes(seed + 7, args.probes, berry_good=False)
        for name, mode, eps, retention in arms:
            r = run(name, mode, retention, args.steps, seed, eps, p1, p2,
                    drive_decay=args.drive_decay)
            runs.append(r)
            a1, a2 = r["marks"][0], r["marks"][1]
            cov = r["coverage"]
            print(f"  {name:14s} s{seed}  "
                  f"p1 rule: {a1[1]:5.1f}% -> {a2[1]:5.1f}%   "
                  f"p2 rule: {a2[2]:5.1f}%   "
                  f"ate B/F {cov[0][BERRY]}/{cov[0][FUNGUS]} then "
                  f"{cov[1][BERRY]}/{cov[1][FUNGUS]}   "
                  f"{r['updates']} upd  {r['seconds']:.0f}s", flush=True)
        print(flush=True)

    with open(args.out, "w") as f:
        json.dump(dict(args=vars(args), runs=runs), f, indent=2)

    def mean(name, fn):
        vals = [fn(r) for r in runs if r["name"] == name]
        return sum(vals) / len(vals)

    print("=" * 84)
    print("PHASE-1 KNOWLEDGE AFTER THE RULE REVERSED")
    print("=" * 84)
    print(f"  {'arm':14s} {'after p1':>9s} {'after p2':>9s} {'forgot':>8s} "
          f"{'p2 rule':>9s} {'fungus p1':>10s} {'fungus p2':>10s}")
    table = {}
    for name, _, _, _ in arms:
        a1 = mean(name, lambda r: r["marks"][0][1])
        a2 = mean(name, lambda r: r["marks"][1][1])
        p2 = mean(name, lambda r: r["marks"][1][2])
        f1 = mean(name, lambda r: r["coverage"][0][FUNGUS])
        f2 = mean(name, lambda r: r["coverage"][1][FUNGUS])
        table[name] = dict(a1=a1, a2=a2, forgot=a1 - a2, p2=p2,
                           f1=f1, f2=f2)
        print(f"  {name:14s} {a1:8.1f}% {a2:8.1f}% {a1 - a2:+7.1f} "
              f"{p2:8.1f}% {f1:10.0f} {f2:10.0f}")

    print("\n" + "=" * 84)
    print("THE DRIVE ITSELF — mean value per cell type at the end of "
          "each phase")
    print("=" * 84)
    print(f"  {'arm':14s} {'empty p1':>9s} {'berry p1':>9s} "
          f"{'fungus p1':>10s} {'empty p2':>9s} {'berry p2':>9s} "
          f"{'fungus p2':>10s}")
    for name, _, _, _ in arms:
        row = []
        for phase in (0, 1):
            for cell in (EMPTY, BERRY, FUNGUS):
                row.append(mean(name, lambda r, p=phase, c=cell:
                                r["drive"][p][c]))
        print(f"  {name:14s} " + " ".join(f"{v:9.3f}" for v in row))

    print("\n" + "=" * 84)
    print("VERDICT")
    print("=" * 84)

    MARGIN = 8.0
    learned = [n for n in table if table[n]["a1"] >= baseline + MARGIN]

    for name in table:
        t = table[name]
        if name not in learned:
            print(f"  {name:14s} LEARNED NOTHING: {t['a1']:.1f}% against a "
                  f"{baseline:.1f}% baseline.")
        else:
            print(f"  {name:14s} learned to {t['a1']:.1f}%, kept "
                  f"{t['a2']:.1f}% (forgot {t['forgot']:+.1f}), "
                  f"p2 rule {t['p2']:.1f}%")

    if "passive" not in learned:
        print(f"\n  INCONCLUSIVE: even the passive arm did not learn. The "
              f"world or the network is")
        print(f"  wrong, not the policies. Nothing below is worth reading.")
        return

    print()
    if "driven" not in learned:
        d = table["driven"]
        print(f"  THE DRIVE STARVED THE SIGNAL. driven stayed at "
              f"{d['a1']:.1f}% and ate {d['f2']:.0f} fungus in")
        print(f"  phase 2 against the passive arm's "
              f"{table['passive']['f2']:.0f}. A persistent drive collapses "
              f"onto whatever")
        print(f"  was surprising first and stops sampling the rest, which "
              f"is the same failure")
        print(f"  greedy had by a different route. Read the drive table "
              f"above: if one cell type")
        print(f"  dominates, that is the collapse.")
    elif "curious" not in learned:
        print(f"  driven learned where curious did not "
              f"({table['driven']['a1']:.1f}% against "
              f"{table['curious']['a1']:.1f}%).")
    else:
        c, d = table["curious"], table["driven"]
        gap = d["p2"] - c["p2"]
        print(f"  THE COMPARISON THAT MATTERS: the PHASE-2 rule, learned "
              f"after the reversal.")
        print(f"  curious {c['p2']:.1f}%   driven {d['p2']:.1f}%   "
              f"({gap:+.1f})")
        if gap >= 3:
            print(f"\n  PERSISTENT ERROR BEATS PREDICTED UNCERTAINTY. The "
                  f"confidently-wrong argument")
            print(f"  holds: after the reversal the model is confident and "
                  f"wrong, entropy stays low,")
            print(f"  and curiosity walks past exactly the thing that "
                  f"changed. A drive built from")
            print(f"  errors that actually happened does not have that "
                  f"blind spot.")
        elif gap <= -3:
            print(f"\n  PREDICTED UNCERTAINTY WINS. The confidently-wrong "
                  f"argument does not hold here,")
            print(f"  or the drive is too coarse at one scalar per cell "
                  f"type to act on.")
        else:
            print(f"\n  NO DIFFERENCE. Persistence buys nothing on this "
                  f"world, which is what valence")
            print(f"  found on the replay side. Two mechanisms, same "
                  f"answer: the instantaneous")
            print(f"  signal was already sufficient. That is worth writing "
                  f"down as a pattern.")

        rise_d = d["f2"] - d["f1"]
        rise_c = c["f2"] - c["f1"]
        print(f"\n  COVERAGE SHIFT ACROSS THE REVERSAL (fungus eaten):")
        print(f"    curious {c['f1']:.0f} -> {c['f2']:.0f} ({rise_c:+.0f})"
              f"    driven {d['f1']:.0f} -> {d['f2']:.0f} "
              f"({rise_d:+.0f})")
        if rise_d > rise_c + 20:
            print(f"    The drive pulled the agent toward what changed. "
                  f"That is the mechanism")
            print(f"    working, visible directly, whatever the scores say.")

    print(f"\n  One toy world, {args.seeds} seed"
          f"{'s' if args.seeds > 1 else ''}, one drive granularity. A first "
          f"data point about wanting,")
    print(f"  not a result about drives in general.")
    print(f"\n  written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
