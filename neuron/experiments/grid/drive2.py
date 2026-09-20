"""Second attempt at a drive, fixing both failures of the first.

drive.py tested whether a PERSISTENT value built from experienced error
steers action better than an INSTANTANEOUS one. It did not answer the
question, and the two reasons are separable.

FAILURE ONE: NO GRADIENT TO ACT ON. The drive was one scalar per cell type,
three buckets updated on every one of 8,000 steps. Over that many samples an
EMA is a running mean, and the measured values confirm it: at decay 0.9 the
driven arm ended at empty 0.293, berry 0.607, fungus 0.631, and at decay
0.99 it was flatter still, 0.375 / 0.719 / 0.713. The policy was choosing
between numbers that were nearly equal, which is a random walk with extra
arithmetic. The hypothesis was never tested because the mechanism had
nothing to steer by.

Raising the decay made it WORSE, which is the clue: the problem was
resolution, not timescale. Twelve buckets keyed on (target cell, action)
give the drive somewhere to develop structure. An action that moves onto
fungus and an action that moves onto berry are different situations even
when the agent has learned both.

FAILURE TWO: TEMPORALLY CLUSTERED SAMPLING. Argmax over a drive means the
agent chases whatever is currently highest until it decays, so its
experience arrives in runs of similar items. The driven arm ate MORE than
any other arm (1,487 fungus in phase 2 against passive's 944) and learned
LESS (50.3% against 91.4%). Coverage was never its problem. Order was.

That is worth stating as a finding on its own, because no passive
experiment in this repo could have produced it: for a single-pass online
learner, WHEN things arrive matters more than how many of them do. A random
walk interleaves by construction. Any policy that steers is, by
construction, doing the opposite.

The fix is to keep the steering and lose the clustering: sample from a
softmax over the drive rather than taking the argmax. High-drive targets
are still preferred, but which one is chosen on any given step is random,
so consecutive steps decorrelate. Temperature controls how hard it steers;
--drive-temp 0 falls back to argmax so the old behaviour is still runnable.

WHAT IS MEASURED NOW. The drive's SPREAD (max minus min across buckets) is
reported at the end of each phase. That is the precondition the first run
failed and it should be read before any score:

  spread near zero    the drive is still a running mean, the policy is
                      still effectively random, and no conclusion about
                      persistent versus instantaneous can be drawn. The
                      granularity is still wrong.
  spread large        there is a gradient, the policy is genuinely steering
                      by it, and the comparison against curious is real.

THE COMPARISON, unchanged: the phase-2 rule, learned after the reversal.
Curiosity reads the model's claimed uncertainty, which is low when the
model is confidently wrong, which is exactly its state right after a
reversal. A drive built from errors that actually happened has no such
blind spot. That is the hypothesis. It still has not been tested.

    python drive2.py
    python drive2.py --drive-temp 0        # argmax, the old behaviour
    python drive2.py --steps 8000 --seeds 5
"""

import argparse
import json
import math
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

TARGET = {"up": 1, "down": 7, "left": 3, "right": 5}

DRIVE_DECAY = 0.9
DRIVE_INIT = 1.0
DRIVE_TEMP = 0.3


class World:
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
    """The net, plus the drive.

    The drive is keyed on (target cell, action) rather than on cell alone.
    Twelve buckets instead of three, which is the whole point: three
    buckets updated thousands of times is a running mean and cannot hold
    a gradient.
    """

    def __init__(self, hidden=64, seed=0, lr=3e-4,
                 drive_decay=DRIVE_DECAY, drive_temp=DRIVE_TEMP):
        self.net = Net(hidden, seed)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.h = None

        self.drive = {(c, a): DRIVE_INIT
                      for c in (EMPTY, BERRY, FUNGUS) for a in ACTIONS}
        self.drive_decay = drive_decay
        self.drive_temp = drive_temp
        self.rng = random.Random(seed + 991)

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

    def feel(self, cell, action, surprise):
        k = (cell, action)
        d = self.drive[k]
        self.drive[k] = (self.drive_decay * d
                         + (1.0 - self.drive_decay) * surprise)

    def spread(self):
        """max minus min across buckets. The precondition, not a result.

        If this is near zero the drive is a running mean, the policy is
        effectively random, and nothing downstream is interpretable.
        """
        v = list(self.drive.values())
        return max(v) - min(v)

    def by_cell(self):
        """Mean drive per cell type, for reporting against drive.py."""
        out = {}
        for c in (EMPTY, BERRY, FUNGUS):
            vals = [self.drive[(c, a)] for a in ACTIONS]
            out[c] = sum(vals) / len(vals)
        return out

    def act(self, patch, mode, epsilon):
        """Choose an action.

        driven    SAMPLE from a softmax over the drive of each action's
                  target bucket. Argmax was what produced temporally
                  clustered experience in drive.py: the agent chased one
                  bucket until it decayed, so its stream arrived in runs of
                  similar items and a single-pass online learner cannot
                  cope with that. Sampling keeps the preference and
                  decorrelates consecutive steps.

                  drive_temp 0 restores argmax, so the old behaviour is
                  still runnable for comparison.
        """
        if random.random() < epsilon:
            return random.choice(ACTIONS)

        if mode == "driven":
            scores = [self.drive[(patch[TARGET[a]], a)] for a in ACTIONS]
            if self.drive_temp <= 0:
                return ACTIONS[max(range(len(ACTIONS)),
                                   key=lambda i: scores[i])]
            hi = max(scores)
            w = [math.exp((s - hi) / self.drive_temp) for s in scores]
            total = sum(w)
            r = self.rng.random() * total
            acc = 0.0
            for a, weight in zip(ACTIONS, w):
                acc += weight
                if r <= acc:
                    return a
            return ACTIONS[-1]

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
        drive_decay, drive_temp):
    started = time.time()
    backend = Backend(seed=seed, drive_decay=drive_decay,
                      drive_temp=drive_temp)
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
    drive_marks, spreads, runlengths = [], [], []

    for phase, berry_good in enumerate([True, False], start=1):
        world = World(seed * 100 + phase, berry_good=berry_good)
        seen = {BERRY: 0, FUNGUS: 0}

        # How temporally clustered the experience is. Counts how often two
        # consecutive steps landed on the same cell type. This is the
        # variable drive.py failed on and could not see.
        same, prev = 0, None

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
            if prev is not None and prev == target_cell:
                same += 1
            prev = target_cell

            item = (patch, action, outcome)
            s, _ = backend.score(item)
            backend.feel(target_cell, action, s)

            if layer is not None:
                if layer.observe(item).get("learned"):
                    updates += 1
            else:
                backend.update(item, 1)
                updates += 1

        coverage.append(seen)
        runlengths.append(100.0 * same / max(1, steps - 1))
        marks.append((phase, accuracy(backend, probes1),
                      accuracy(backend, probes2)))
        drive_marks.append(backend.by_cell())
        spreads.append(backend.spread())

    s = layer.summary() if layer else dict(rollbacks=0, rehearsals=0,
                                           health=0.0)
    return dict(name=name, seed=seed, mode=mode, retention=retention,
                marks=marks, coverage=coverage, updates=updates,
                drive=drive_marks, spread=spreads, repeat=runlengths,
                rollbacks=s["rollbacks"], rehearsals=s["rehearsals"],
                health=s["health"], seconds=time.time() - started)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--probes", type=int, default=300)
    ap.add_argument("--drive-decay", type=float, default=DRIVE_DECAY)
    ap.add_argument("--drive-temp", type=float, default=DRIVE_TEMP,
                    help="0 restores argmax, the drive.py behaviour")
    ap.add_argument("--out", default="drive2.json")
    args = ap.parse_args()

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

    print(f"  {args.steps} steps per phase, 2 phases, {args.seeds} seeds")
    print(f"  phase 1: berry good.  phase 2: the rule reverses.")
    print(f"  drive: 12 buckets on (cell, action), decay "
          f"{args.drive_decay}, temp {args.drive_temp}"
          f"{'  [ARGMAX]' if args.drive_temp <= 0 else ''}")
    print(f"  majority-class baseline {baseline:.1f}%\n", flush=True)

    runs = []
    for seed in range(args.seeds):
        p1 = make_probes(seed, args.probes, berry_good=True)
        p2 = make_probes(seed + 7, args.probes, berry_good=False)
        for name, mode, eps, retention in arms:
            r = run(name, mode, retention, args.steps, seed, eps, p1, p2,
                    args.drive_decay, args.drive_temp)
            runs.append(r)
            a1, a2 = r["marks"][0], r["marks"][1]
            cov = r["coverage"]
            print(f"  {name:14s} s{seed}  "
                  f"p1: {a1[1]:5.1f}% -> {a2[1]:5.1f}%   "
                  f"p2: {a2[2]:5.1f}%   "
                  f"B/F {cov[0][BERRY]}/{cov[0][FUNGUS]} then "
                  f"{cov[1][BERRY]}/{cov[1][FUNGUS]}   "
                  f"spread {r['spread'][1]:.2f}  "
                  f"repeat {r['repeat'][1]:.0f}%  "
                  f"{r['seconds']:.0f}s", flush=True)
        print(flush=True)

    with open(args.out, "w") as f:
        json.dump(dict(args=vars(args), runs=runs), f, indent=2,
                  default=str)

    def mean(name, fn):
        vals = [fn(r) for r in runs if r["name"] == name]
        return sum(vals) / len(vals)

    print("=" * 88)
    print("THE PRECONDITION — is there a gradient to steer by, and is the "
          "stream interleaved?")
    print("=" * 88)
    print(f"  {'arm':14s} {'spread p1':>10s} {'spread p2':>10s} "
          f"{'repeat p1':>10s} {'repeat p2':>10s}")
    print("  spread = max minus min across the 12 drive buckets. "
          "Near zero means no gradient.")
    print("  repeat = % of consecutive steps landing on the same cell "
          "type. Higher is more clustered.")
    print("-" * 88)
    for name, _, _, _ in arms:
        print(f"  {name:14s} "
              f"{mean(name, lambda r: r['spread'][0]):10.3f} "
              f"{mean(name, lambda r: r['spread'][1]):10.3f} "
              f"{mean(name, lambda r: r['repeat'][0]):9.1f}% "
              f"{mean(name, lambda r: r['repeat'][1]):9.1f}%")

    print("\n" + "=" * 88)
    print("PHASE-1 KNOWLEDGE AFTER THE RULE REVERSED")
    print("=" * 88)
    print(f"  {'arm':14s} {'after p1':>9s} {'after p2':>9s} {'forgot':>8s} "
          f"{'p2 rule':>9s} {'fungus p1':>10s} {'fungus p2':>10s}")
    table = {}
    for name, _, _, _ in arms:
        a1 = mean(name, lambda r: r["marks"][0][1])
        a2 = mean(name, lambda r: r["marks"][1][1])
        p2 = mean(name, lambda r: r["marks"][1][2])
        f1 = mean(name, lambda r: r["coverage"][0][FUNGUS])
        f2 = mean(name, lambda r: r["coverage"][1][FUNGUS])
        sp = mean(name, lambda r: r["spread"][1])
        rp = mean(name, lambda r: r["repeat"][1])
        table[name] = dict(a1=a1, a2=a2, forgot=a1 - a2, p2=p2, f1=f1,
                           f2=f2, spread=sp, repeat=rp)
        print(f"  {name:14s} {a1:8.1f}% {a2:8.1f}% {a1 - a2:+7.1f} "
              f"{p2:8.1f}% {f1:10.0f} {f2:10.0f}")

    print("\n" + "=" * 88)
    print("VERDICT")
    print("=" * 88)

    MARGIN = 8.0
    learned = [n for n in table if table[n]["a1"] >= baseline + MARGIN]

    d = table["driven"]
    c = table["curious"]
    p = table["passive"]

    # The precondition is checked BEFORE anything else, because a flat
    # drive makes every downstream number uninterpretable and drive.py
    # printed a confident diagnosis off exactly that situation.
    FLAT = 0.15
    if d["spread"] < FLAT:
        print(f"  STILL NO GRADIENT. The drive's spread across 12 buckets "
              f"is {d['spread']:.3f}, so the")
        print(f"  policy is choosing between near-equal numbers and is "
              f"effectively random. Twelve")
        print(f"  buckets was not enough resolution either. NOTHING BELOW "
              f"IS INTERPRETABLE, and the")
        print(f"  hypothesis about persistent versus instantaneous signals "
              f"remains untested.")
        print(f"\n  Next thing to try: key the drive on a hash of the local "
              f"patch rather than on")
        print(f"  the target cell, or update it only on steps that "
              f"actually ate something.")
    else:
        print(f"  THE DRIVE HAS A GRADIENT: spread {d['spread']:.3f} "
              f"across 12 buckets.")
        print(f"  Sampling rather than argmax left it at "
              f"{d['repeat']:.1f}% repeated cell types against")
        print(f"  the passive walk's {p['repeat']:.1f}%.")

        if "driven" not in learned:
            print(f"\n  AND IT STILL DID NOT LEARN: {d['a1']:.1f}% against "
                  f"a {baseline:.1f}% baseline, eating")
            print(f"  {d['f2']:.0f} fungus against passive's {p['f2']:.0f}. "
                  f"Coverage is not the constraint and")
            print(f"  neither is the gradient, so steering itself is "
                  f"costing something the random")
            print(f"  walk does not pay.")
        else:
            gap = d["p2"] - c["p2"]
            print(f"\n  THE COMPARISON: the phase-2 rule, learned after the "
                  f"reversal.")
            print(f"  curious {c['p2']:.1f}%   driven {d['p2']:.1f}%   "
                  f"({gap:+.1f})")
            if gap >= 3:
                print(f"\n  PERSISTENT ERROR BEATS PREDICTED UNCERTAINTY. "
                      f"After a reversal the model is")
                print(f"  confident and wrong, entropy stays low, and "
                      f"curiosity walks past the thing that")
                print(f"  changed. A drive built from errors that actually "
                      f"happened does not.")
            elif gap <= -3:
                print(f"\n  PREDICTED UNCERTAINTY STILL WINS, with a real "
                      f"gradient this time. The")
                print(f"  confidently-wrong argument does not hold on this "
                      f"world.")
            else:
                print(f"\n  NO DIFFERENCE. Persistence buys nothing here, "
                      f"which is what valence found on")
                print(f"  the replay side. Two mechanisms, same answer, and "
                      f"that pattern is worth more")
                print(f"  than either result alone.")

    print(f"\n  CLUSTERING, the finding drive.py produced by accident: "
          f"driven ate the most and")
    print(f"  learned the least there, at {100.0:.0f}% argmax steering. "
          f"Compare repeat rates above")
    print(f"  against scores to see whether sampling fixed it.")

    print(f"\n  One toy world, {args.seeds} seeds. A data point about "
          f"wanting, not a result about")
    print(f"  drives in general.")
    print(f"\n  written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
