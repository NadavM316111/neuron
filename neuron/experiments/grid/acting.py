"""Does retention still work when the system CHOOSES its own data?

THE GAP THIS OPENS. Every experiment in this project so far is passive. A
random walker moves, the network predicts what happens next, and the stream
arrives regardless of what the network thinks. That is prediction, not
action, and it means nothing here has ever learned from a consequence it
caused.

That matters beyond tidiness. The signal a model learns from has been the
thing that defined each era of the field: hand-written rules, then labelled
data, then the public internet, then human preference. Public text is
projected to be exhausted between 2026 and 2032. The reservoir that has
never been tapped is what happens when a deployed system ACTS, because
every deployed model today is frozen and throws that away.

So this is the first test of whether the machinery in this repo works on
that kind of signal.

THE COMPLICATION, and it is the reason to run this before building anything
on the idea. An acting agent shapes its own data. If it learns that berries
are good it eats berries, so it stops sampling fungus, so when the rule
reverses it may never find out. The data distribution is downstream of the
policy, which is downstream of the data. Passive prediction has no such
loop. This is standard exploration/exploitation, and the point here is not
to solve it but to MEASURE whether it breaks retention, because a retention
result that only holds on passively collected data would not support any of
the claims this project wants to make.

THE WORLD. A 7x7 grid with berries and fungus. Moving onto one eats it.

  phase 1   berry good, fungus bad
  phase 2   the rule REVERSES

Held-out probes test the phase-1 rule. The question is whether phase-1
knowledge survives phase 2, exactly as in the language drift experiment,
but now on a stream the agent generated.

FOUR ARMS, and the passive ones are the control that makes it readable:

  acting-retention    chooses its actions, full stability layer
  acting-none         chooses its actions, no layer
  passive-retention   random walk, full layer
  passive-none        random walk, no layer

Passive arms see every outcome because a random walk samples everything.
Acting arms see what their policy sampled. The difference between those two
pairs isolates the cost of self-generated data.

COVERAGE IS REPORTED because it is the mechanism. If the acting arms ate
far fewer fungus in phase 2, any retention difference may be about what
they SAW rather than how they learned, and that has to be visible rather
than inferred.

WHAT THE FIRST RUN FOUND, and it was not what this was designed to measure.
The acting arms scored 54.3% and 53.2% on the phase-1 rule against a
majority-class baseline of 49.5%. They learned NOTHING, so asking whether
retention preserved their knowledge was meaningless. The verdict printed
"retention does not survive self-generated data", which the data could not
support.

The cause was visible in the coverage column. Acting arms ate 150 to 358
items per phase where the passive arms ate around 950. THE AGENT LEARNED TO
EAT NOTHING. With half of all outcomes being "nothing" and the rest split
between good and bad, refusing to step on anything is the best policy
available to an agent that does not yet know the rule. Optimal behaviour,
and it starves the learning signal completely.

So an acting system does not merely bias its data, it can switch its own
signal off. That is a design requirement for anything meant to learn from
deployment, not a bug in this script.

TWO FIXES, both in this version.

A LEARNABILITY GATE. Any arm whose phase-1 accuracy does not clear the
majority-class baseline by a margin is reported as having learned nothing,
and no forgetting number is read from it. The previous version checked that
the PASSIVE arm could forget and never checked that the ACTING arm could
learn. That is the same mistake this repo's method notes already warn
about: check the test can fail before running it.

CURIOSITY AS AN EXPLORATION RULE. The surprise signal that already decides
what the layer LEARNS from can also decide what the agent DOES. An agent
choosing the action whose outcome it is least certain about samples exactly
what it does not yet understand. This is intrinsic-motivation exploration,
an established idea, and here it costs nothing extra because the
uncertainty comes from the same forward pass the policy already needs.

  passive      random walk. Sees everything. The control.
  greedy       epsilon 0.1. The arm that starved itself.
  explore      epsilon 0.5. The blunt fix.
  curious      picks the action with the most uncertain predicted outcome.

    python acting.py                    # ~2 minutes
    python acting.py --steps 8000 --seeds 3
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
# outcome classes the network predicts
NOTHING, GOOD, BAD = 0, 1, 2
OUTCOMES = 3


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
        """What the agent can see: the 3x3 patch around it.

        Position is not given. The patch is, so an action's outcome is
        predictable from what is adjacent — which is the point, since the
        agent has to learn the RULE rather than memorise the map.
        """
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
        self._place(cell)                      # keep density constant
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

    An item is (patch, action, outcome). Surprise is the loss on predicting
    the outcome, which is exactly the signal the gate wants: an action whose
    result the agent did not expect.
    """

    def __init__(self, hidden=64, seed=0, lr=3e-4):
        self.net = Net(hidden, seed)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.h = None

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

    def act(self, patch, mode, epsilon):
        """Choose an action. Three policies, and the third is the point.

        value     prefer the action least likely to be bad. This is what an
                  agent optimising its outcome does, and on this world it
                  converges to standing still, because refusing to eat
                  avoids every bad outcome and the agent starves its own
                  learning signal.

        epsilon   the same, with more random actions mixed in. Blunt, and
                  it works by simply not being the policy half the time.

        curious   prefer the action whose outcome is most UNCERTAIN, scored
                  as the entropy of the predicted distribution. The agent
                  goes where it does not yet know what will happen, which
                  is the same surprise signal the stability layer already
                  uses to decide what is worth learning from. One signal,
                  two jobs.
        """
        if random.random() < epsilon:
            return random.choice(ACTIONS)
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
    """Held-out (patch, action, outcome) triples under a fixed rule.

    Generated from the rule directly rather than sampled from the agent's
    experience, so an agent that stopped visiting fungus is still tested on
    fungus. Otherwise the probe set would inherit the very bias the
    experiment exists to measure.
    """
    rng = random.Random(10_000 + seed)
    out = []
    while len(out) < n:
        patch = [rng.choice([EMPTY, EMPTY, BERRY, FUNGUS]) for _ in range(9)]
        action = rng.choice(ACTIONS)
        target = {"up": 1, "down": 7, "left": 3, "right": 5}[action]
        cell = patch[target]
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

def run(name, mode, retention, steps, seed, epsilon, probes1, probes2):
    """mode is "passive" for a random walk, otherwise a policy name."""
    started = time.time()
    backend = Backend(seed=seed)
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
    for phase, berry_good in enumerate([True, False], start=1):
        world = World(seed * 100 + phase, berry_good=berry_good)
        seen = {BERRY: 0, FUNGUS: 0}
        for i in range(steps):
            if i % 200 == 0:
                world.reset()
            patch = world.observe()
            action = (random.choice(ACTIONS) if mode == "passive"
                      else backend.act(patch, mode, epsilon))
            outcome, cell = world.step(action)
            if cell in seen:
                seen[cell] += 1
            item = (patch, action, outcome)
            if layer is not None:
                if layer.observe(item).get("learned"):
                    updates += 1
            else:
                backend.update(item, 1)
                updates += 1
        coverage.append(seen)
        marks.append((phase, accuracy(backend, probes1),
                      accuracy(backend, probes2)))

    s = layer.summary() if layer else dict(rollbacks=0, rehearsals=0,
                                           health=0.0)
    return dict(name=name, seed=seed, mode=mode, retention=retention,
                marks=marks, coverage=coverage, updates=updates,
                rollbacks=s["rollbacks"], rehearsals=s["rehearsals"],
                health=s["health"], seconds=time.time() - started)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000,
                    help="steps per phase")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--probes", type=int, default=300)
    ap.add_argument("--out", default="acting.json")
    args = ap.parse_args()

    # mode, epsilon, retention
    arms = [("passive",          "passive", 0.0, True),
            ("passive-none",     "passive", 0.0, False),
            ("greedy",           "value",   0.1, True),
            ("explore",          "value",   0.5, True),
            ("curious",          "curious", 0.05, True)]

    # The bar any arm has to clear before its numbers mean anything.
    ref = make_probes(0, args.probes, berry_good=True)
    counts = {}
    for _, _, o in ref:
        counts[o] = counts.get(o, 0) + 1
    baseline = 100.0 * max(counts.values()) / len(ref)

    print(f"  {args.steps} steps per phase, 2 phases, {args.seeds} seed"
          f"{'s' if args.seeds > 1 else ''}")
    print(f"  phase 1: berry good.  phase 2: the rule reverses.")
    print(f"  majority-class baseline {baseline:.1f}% — an arm below this "
          f"learned nothing.\n", flush=True)

    runs = []
    for seed in range(args.seeds):
        p1 = make_probes(seed, args.probes, berry_good=True)
        p2 = make_probes(seed + 7, args.probes, berry_good=False)
        for name, mode, eps, retention in arms:
            r = run(name, mode, retention, args.steps, seed,
                    eps, p1, p2)
            runs.append(r)
            a1 = r["marks"][0]
            a2 = r["marks"][1]
            cov = r["coverage"]
            print(f"  {name:19s} s{seed}  "
                  f"phase1 rule: {a1[1]:5.1f}% -> {a2[1]:5.1f}%   "
                  f"phase2 rule: {a2[2]:5.1f}%   "
                  f"ate B/F {cov[0][BERRY]}/{cov[0][FUNGUS]} then "
                  f"{cov[1][BERRY]}/{cov[1][FUNGUS]}   "
                  f"{r['updates']} upd  {r['rollbacks']} rb  "
                  f"{r['seconds']:.0f}s", flush=True)
        print(flush=True)

    with open(args.out, "w") as f:
        json.dump(dict(args=vars(args), runs=runs), f, indent=2)

    def mean(name, fn):
        vals = [fn(r) for r in runs if r["name"] == name]
        return sum(vals) / len(vals)

    print("=" * 78)
    print("PHASE-1 KNOWLEDGE AFTER THE RULE REVERSED")
    print("=" * 78)
    print(f"  {'arm':20s} {'after p1':>9s} {'after p2':>9s} "
          f"{'forgot':>8s} {'p2 rule':>9s} {'fungus eaten p2':>16s}")
    table = {}
    for name, _, _, _ in arms:
        a1 = mean(name, lambda r: r["marks"][0][1])
        a2 = mean(name, lambda r: r["marks"][1][1])
        p2 = mean(name, lambda r: r["marks"][1][2])
        fung = mean(name, lambda r: r["coverage"][1][FUNGUS])
        table[name] = (a1, a2, a1 - a2, p2, fung)
        print(f"  {name:20s} {a1:8.1f}% {a2:8.1f}% {a1 - a2:+7.1f} "
              f"{p2:8.1f}% {fung:16.0f}")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)

    # Did each arm learn anything at all? An arm at the baseline has no
    # knowledge, so "how much did it forget" is not a question about it.
    # The previous version checked only that the PASSIVE arm could forget
    # and read forgetting numbers off arms that had learned nothing.
    MARGIN = 8.0
    learned, blank = [], []
    for name in table:
        (learned if table[name][0] >= baseline + MARGIN
         else blank).append(name)

    for name in table:
        a1, a2, forgot, p2, fung = table[name]
        if name in blank:
            print(f"  {name:14s} LEARNED NOTHING: {a1:.1f}% against a "
                  f"{baseline:.1f}% baseline. Its forgetting")
            print(f"  {'':14s} number is not interpretable.")
        else:
            print(f"  {name:14s} learned to {a1:.1f}%, kept {a2:.1f}% "
                  f"(forgot {forgot:+.1f}), ate {fung:.0f} fungus in p2")

    if "passive" not in learned:
        print(f"\n  INCONCLUSIVE: even the passive arm did not learn. The "
              f"world or the network")
        print(f"  is wrong, not the policies. Nothing below this is worth "
              f"reading.")
        return

    acting_arms = [n for n in ("greedy", "explore", "curious")
                   if n in learned]

    print()
    if not acting_arms:
        print(f"  NO ACTING POLICY LEARNED. Every one of greedy, explore "
              f"and curious stayed at")
        print(f"  the baseline while the passive random walk reached "
              f"{table['passive'][0]:.1f}%. On this")
        print(f"  world, choosing your own actions destroys the learning "
              f"signal however you")
        print(f"  choose them. That is a finding about acting systems, and "
              f"it says exploration")
        print(f"  has to be solved before consequence is a usable signal at "
              f"all.")
    else:
        best = max(acting_arms, key=lambda n: table[n][0])
        print(f"  BEST ACTING POLICY: {best}, learned to "
              f"{table[best][0]:.1f}% against the passive")
        print(f"  arm's {table['passive'][0]:.1f}%, eating "
              f"{table[best][4]:.0f} fungus in phase 2 against "
              f"{table['passive'][4]:.0f}.")
        # The comparison that matters is curious against EXPLORE, not
        # against greedy. A first version fired a "curiosity rescued it"
        # branch whenever greedy failed and curious did not, and printed
        # that while explore was beating curious on both seeds. Curiosity
        # has to beat plain randomness to be worth its machinery.
        if "greedy" not in learned and acting_arms:
            print(f"\n  VALUE-SEEKING DESTROYS THE SIGNAL. greedy stayed "
                  f"at {table['greedy'][0]:.1f}% and ate")
            print(f"  {table['greedy'][4]:.0f} fungus in phase 2 against "
                  f"the passive arm's {table['passive'][4]:.0f}. An agent "
                  f"optimising its")
            print(f"  outcome learns to touch nothing, which is optimal and "
                  f"useless. Exploration is")
            print(f"  a precondition for consequence being a learning "
                  f"signal at all.")

        if "curious" in learned and "explore" in learned:
            c, e = table["curious"], table["explore"]
            if c[0] >= e[0] + 3:
                print(f"\n  CURIOSITY BEATS RANDOM EXPLORATION: "
                      f"{c[0]:.1f}% against {e[0]:.1f}%. Seeking your own")
                print(f"  uncertainty is better than being random, so the "
                      f"surprise signal the layer")
                print(f"  uses to choose what to learn from also works as "
                      f"the thing that decides")
                print(f"  what to do. One signal, two jobs.")
            elif c[0] <= e[0] - 3:
                print(f"\n  CURIOSITY LOSES TO RANDOM EXPLORATION: "
                      f"{c[0]:.1f}% against {e[0]:.1f}%. It beats")
                print(f"  value-seeking and it does not beat being random "
                      f"half the time, so on this")
                print(f"  world the machinery is not earning its place.")
                # Worth looking at anyway: curiosity is the only policy
                # whose sampling can RISE when the world changes, since a
                # reversal makes outcomes uncertain again.
                c_cov = mean("curious", lambda r: r["coverage"][0][FUNGUS])
                e_cov = mean("explore", lambda r: r["coverage"][0][FUNGUS])
                if c[4] > c_cov and e[4] < e_cov:
                    print(f"\n  But its COVERAGE ROSE after the rule "
                          f"reversed ({c_cov:.0f} to {c[4]:.0f} fungus) "
                          f"while")
                    print(f"  explore's fell ({e_cov:.0f} to {e[4]:.0f}). "
                          f"Curiosity samples more exactly when the world")
                    print(f"  stops being predictable, which no other "
                          f"policy here does. Two phases may be")
                    print(f"  too short to reward that; more reversals "
                          f"would test it.")
            else:
                print(f"\n  CURIOSITY MATCHES RANDOM EXPLORATION "
                      f"({c[0]:.1f}% against {e[0]:.1f}%). Elegant and "
                      f"not")
                print(f"  necessary on this world.")
        pf, pn = table["passive"][2], table["passive-none"][2]
        print(f"\n  Retention on the passive stream: unprotected forgot "
              f"{pn:+.1f}, protected {pf:+.1f}")
        print(f"  (advantage {pn - pf:+.1f}). On a two-phase grid world "
              f"that is small, and worth")
        print(f"  reporting against the larger language result rather than "
              f"alongside it.")

    print(f"\n  One toy world, {args.seeds} seed"
          f"{'s' if args.seeds > 1 else ''}. A first data point about "
          f"acting, not a result about deployment.")
    print(f"\n  written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()