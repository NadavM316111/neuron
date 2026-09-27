"""The first world where the system can lose.

WHAT IS MISSING FROM EVERY EXPERIMENT IN THIS REPO. Nothing can go wrong.
A run ends when the stream ends, the numbers are read off, and a bad model
and a good one both reach the end. Retention, drives and sleep have all
been measured as accuracy at a fixed horizon, which is a proxy for
mattering rather than mattering itself.

Here the agent has energy. Every step costs some. Eating the right thing
restores it. Eating the wrong thing costs more. At zero the life ENDS, the
accumulated weights are discarded, and the only number that comes out is
how long it lasted.

That single change alters what everything else is for. Prediction stops
being the goal and becomes the means: knowing which cell is food is worth
exactly as much energy as it saves. Memory matters because forgetting
where the rule stands gets the agent killed. Exploration becomes a
genuine risk rather than a free bonus, because the steps spent looking are
steps not spent eating.

SLEEP COSTS SOMETHING, AND THAT IS THE POINT. In the grid-world sleep
sweep (20 Sep) consolidation was free: the schedule changed but the agent
paid nothing for it. Here a sleep block burns energy and gathers no food,
so consolidating is a bet — the agent spends life now to predict better
later. An animal makes that bet every night. Nothing in this repo has ever
had to.

THE WORLD. A grid of berries and fungus that regrow. One of the two is
food and the other is poison, and WHICH ONE REVERSES every REVERSAL_STEPS.
So the agent cannot learn the rule once and coast: a system that stops
learning dies at the first reversal, and a system that learns but cannot
retain dies at the second.

THE ARMS, in order of how much machinery they carry:

  reflex        no learning at all, random movement. The floor. If a
                learning arm cannot beat this, the learning is worthless
                here however good its probe losses were.
  online        learns every step, no layer, no replay. The naive system.
  continuous    the layer with interleaved consolidation, which is what
                neuron_system.py did before 20 Sep.
  sleeping      the layer with offline consolidation in blocks, which is
                what it does now. Sleep costs energy.
  driven        sleeping, plus action chosen by accumulated prediction
                error rather than by predicted value. The drive that beat
                every other policy on 20 Sep, now with something at stake.

WHAT THIS CAN SHOW THAT NOTHING ELSE HERE CAN.

  A survival curve. Not accuracy at step 4,000 but how long each kind of
  system stays alive, across many lives, with the spread visible. That is
  the first number in this repo that answers "does any of this matter"
  rather than "does any of this work".

  Whether sleep survives being expensive. It won on accuracy when it was
  free. If it loses here, the 20 Sep result is real and irrelevant, which
  is worth knowing before more is built on it.

  Whether the drive pays for itself. Curiosity-like exploration always
  looks good when exploring is free. Here every exploratory step is
  energy, and greedy value-seeking — which starved the learning signal in
  acting.py — is now the policy that eats.

HONEST LIMITS, stated before any number is read. This is a toy: a 9x9
grid, a GRU, and a rule with two states. Lifespan here is not lifespan
anywhere else. What transfers, if anything, is the ORDERING of the arms,
and only if it is stable across lives.

    python survive.py                    # 12 lives per arm
    python survive.py --lives 30         # tighter, slower
"""

import argparse
import json
import os
import random
import statistics
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))

from sleeping import SleepLayer          # noqa: E402


SIZE = 9
ACTIONS = ["up", "down", "left", "right"]
EMPTY, BERRY, FUNGUS = 0, 1, 2
CELLS = 3
NOTHING, GOOD, BAD = 0, 1, 2
OUTCOMES = 3
TARGET = {"up": 1, "down": 7, "left": 3, "right": 5}
DELTA = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}

# The energy economy. These numbers decide whether the world is survivable
# at all, so they are stated together rather than scattered: a world where
# a random walker survives forever measures nothing, and one where a
# perfect predictor still starves measures nothing either. The reflex arm
# is the check that this is set sensibly.
START_ENERGY = 200.0
MAX_ENERGY = 260.0
STEP_COST = 1.0
FOOD_GAIN = 18.0
POISON_COST = 16.0
SLEEP_COST_PER_ROUND = 0.15

# How often the rule flips. Long enough to learn, short enough that a
# system which stops learning dies.
REVERSAL_STEPS = 300
MAX_STEPS = 10_000              # a life this long counts as survival

# A DESIGN CONSTRAINT, FOUND THE EXPENSIVE WAY. Poison cost must stay BELOW
# food gain in any world where the agent has to eat in order to learn what
# is edible.
#
# An agent choosing by expected energy eats when p(good)*FOOD exceeds
# p(bad)*POISON, so the accuracy it needs before eating is worthwhile is
# POISON / (FOOD + POISON). At 18 and 22 that is 0.55, which is ABOVE
# chance: an untrained agent is correctly better off eating nothing, and it
# can never bootstrap, because the only way to learn which cell is food is
# to eat some. Four configurations of this file starved for that reason and
# all four looked like a policy bug. They were not. The agent was right.
#
# At 18 and 12 the threshold is 0.40, below chance, so eating pays from the
# first step and accuracy still buys lifespan: 615 steps at 50% accuracy,
# 1,850 at 60%, past 10,000 above 70%.
#
# The same trap is already in acting.py from August, where value-seeking
# agents "learned to eat nothing, which is optimal and useless". That note
# described the symptom. This is the arithmetic behind it.

# Density of food, kept constant by replacing what is eaten.
N_BERRY = 30
N_FUNGUS = 30

HIDDEN = 64
LR = 3e-4
SEQ_LEN = 8

# Sleep schedule. 20 Sep found a threshold on the grid: separation below
# ~100 items bought little. This is the smallest separation that cleared
# it, and the rounds match what interleaved consolidation would have spent
# over the same stretch, so the arms differ in schedule and not in volume.
SLEEP_EVERY = 300
SLEEP_ROUNDS = SLEEP_EVERY // 6
REHEARSE_PER_ITEM = 6

DRIVE_DECAY = 0.9
DRIVE_TEMP = 0.3
EPSILON = 0.2


class World:
    """Berries and fungus that regrow. Which one is food reverses."""

    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.berry_good = True
        self.grid = [[EMPTY] * SIZE for _ in range(SIZE)]
        for _ in range(N_BERRY):
            self._place(BERRY)
        for _ in range(N_FUNGUS):
            self._place(FUNGUS)
        self.x = self.rng.randrange(SIZE)
        self.y = self.rng.randrange(SIZE)

    def _place(self, kind):
        for _ in range(80):
            x, y = self.rng.randrange(SIZE), self.rng.randrange(SIZE)
            if self.grid[y][x] == EMPTY:
                self.grid[y][x] = kind
                return

    def reverse(self):
        self.berry_good = not self.berry_good

    def observe(self):
        patch = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                x, y = self.x + dx, self.y + dy
                patch.append(self.grid[y][x]
                             if 0 <= x < SIZE and 0 <= y < SIZE else EMPTY)
        return patch

    def step(self, action):
        """Returns (outcome, cell, energy_delta) BEFORE the step cost."""
        dx, dy = DELTA[action]
        nx, ny = self.x + dx, self.y + dy
        if not (0 <= nx < SIZE and 0 <= ny < SIZE):
            return NOTHING, EMPTY, 0.0
        self.x, self.y = nx, ny
        cell = self.grid[ny][nx]
        if cell == EMPTY:
            return NOTHING, EMPTY, 0.0
        self.grid[ny][nx] = EMPTY
        self._place(cell)
        good = (cell == BERRY) == self.berry_good
        if good:
            return GOOD, cell, FOOD_GAIN
        return BAD, cell, -POISON_COST


def encode(patch, action):
    v = torch.zeros(9 * CELLS + len(ACTIONS))
    for i, c in enumerate(patch):
        v[i * CELLS + c] = 1.0
    v[9 * CELLS + ACTIONS.index(action)] = 1.0
    return v.unsqueeze(0)


class Net(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.enc = nn.Linear(9 * CELLS + len(ACTIONS), HIDDEN)
        self.cell = nn.GRUCell(HIDDEN, HIDDEN)
        self.head = nn.Linear(HIDDEN, OUTCOMES)

    def forward(self, x, h=None):
        return self.head(self.cell(F.relu(self.enc(x)), h)), h


class Backend:
    def __init__(self, seed=0):
        self.net = Net(seed)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.h = None
        self.drive = {(c, a): 1.0
                      for c in (EMPTY, BERRY, FUNGUS) for a in ACTIONS}
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

    def update_sequence(self, items, steps):
        self.net.train()
        done = 0
        for _ in range(steps):
            h = None
            total = 0.0
            self.opt.zero_grad()
            for patch, action, outcome in items:
                logits, h = self.net(encode(patch, action), h)
                total = total + F.cross_entropy(
                    logits, torch.tensor([outcome]))
                done += 1
            (total / max(1, len(items))).backward()
            self.opt.step()
        return done

    def begin_sequence(self):
        self.h = None

    def reset_state(self):
        self.h = None

    def snapshot(self):
        return {k: v.detach().clone()
                for k, v in self.net.state_dict().items()}

    def restore(self, state):
        self.net.load_state_dict(state)

    def feel(self, cell, action, surprise):
        k = (cell, action)
        self.drive[k] = (DRIVE_DECAY * self.drive[k]
                         + (1.0 - DRIVE_DECAY) * surprise)

    def act(self, patch, mode):
        """reflex is random. value eats what it predicts is good. driven
        samples by accumulated error, which is what won on 20 Sep when
        exploring was free and is now being charged for."""
        if mode == "reflex" or random.random() < EPSILON:
            return random.choice(ACTIONS)

        if mode == "driven":
            scores = [self.drive[(patch[TARGET[a]], a)] for a in ACTIONS]
            hi = max(scores)
            w = [pow(2.718281828, (s - hi) / DRIVE_TEMP) for s in scores]
            total = sum(w)
            r = self.rng.random() * total
            acc = 0.0
            for a, weight in zip(ACTIONS, w):
                acc += weight
                if r <= acc:
                    return a
            return ACTIONS[-1]

        self.net.eval()
        order = list(ACTIONS)
        self.rng.shuffle(order)
        best, choice = None, order[0]
        with torch.no_grad():
            for a in order:
                p = F.softmax(self.net(encode(patch, a), None)[0], dim=1)[0]
                # EXPECTED ENERGY, not a probability difference. The
                # difference scores an empty cell at ~0 and a coin-flip
                # food cell at ~0 too, so the agent walks into nothing and
                # starves -- which is the failure acting.py documented in
                # August. Weighting by what is at stake makes an uncertain
                # food cell worth +4 and empty worth 0, so eating beats
                # abstaining whenever food gain exceeds poison cost.
                score = (p[GOOD].item() * FOOD_GAIN
                         - p[BAD].item() * POISON_COST)
                if best is None or score > best:
                    best, choice = score, a
        return choice


CANARY_ITEMS = None


def make_canary(seed):
    """Ordinary situations the model should always handle. Built from the
    rule directly rather than from the agent's experience, so an agent
    that stopped visiting fungus is still watched on fungus."""
    rng = random.Random(7000 + seed)
    out = []
    for _ in range(12):
        patch = [rng.choice([EMPTY, EMPTY, BERRY, FUNGUS])
                 for _ in range(9)]
        action = rng.choice(ACTIONS)
        cell = patch[TARGET[action]]
        outcome = NOTHING if cell == EMPTY else (
            GOOD if cell == BERRY else BAD)
        out.append((patch, action, outcome))
    return out


def live(arm, seed, max_steps=MAX_STEPS):
    """One life. Returns how long it lasted and what it did.

    Death is the end of the function. Nothing is saved, nothing is
    resumed, and the weights go out of scope with the agent. That is the
    whole point: a failure that can be restarted from a snapshot is not a
    failure.
    """
    random.seed(seed * 31 + 7)
    torch.manual_seed(seed)
    world = World(seed)
    b = Backend(seed)

    layered = arm in ("continuous", "sleeping", "driven")
    layer = None
    if layered:
        layer = SleepLayer(
            b, canary=make_canary(seed), seed=seed, contiguous=True,
            window=200, warmup=30, top_fraction=1.00,
            coherence_veto=1e9, loss_floor=0.02, steps_per_update=1,
            rehearse_per_item=REHEARSE_PER_ITEM, rehearse_count=2,
            rehearse_steps=1, anchor_size=100, buffer_size=500,
            sequence_len=SEQ_LEN, replay_policy="uniform",
            guard=True, guard_per_item=1000, canary_tolerance=0.5)

    policy = {"reflex": "reflex", "online": "value", "continuous": "value",
              "sleeping": "value", "driven": "driven"}[arm]
    sleeps_offline = arm in ("sleeping", "driven")

    energy = START_ENERGY
    since_sleep = 0
    eaten, poisoned, sleeps, reversals = 0, 0, 0, 0
    slept_energy = 0.0

    for step in range(max_steps):
        if step and step % REVERSAL_STEPS == 0:
            world.reverse()
            reversals += 1

        patch = world.observe()
        action = b.act(patch, policy)
        target_cell = patch[TARGET[action]]
        outcome, cell, delta = world.step(action)

        energy += delta - STEP_COST
        if outcome == GOOD:
            eaten += 1
        elif outcome == BAD:
            poisoned += 1
        energy = min(energy, MAX_ENERGY)
        if energy <= 0:
            return dict(arm=arm, seed=seed, steps=step + 1, died=True,
                        eaten=eaten, poisoned=poisoned, sleeps=sleeps,
                        reversals=reversals, slept_energy=slept_energy)

        item = (patch, action, outcome)

        if arm != "reflex":
            s, _ = b.score(item)
            b.feel(target_cell, action, s)

            if layer is None:
                b.update(item, 1)
            else:
                layer.observe(item)
                since_sleep += 1

                if sleeps_offline:
                    if since_sleep >= SLEEP_EVERY:
                        # Sleeping costs life and gathers no food. The
                        # agent bets energy now on predicting better later,
                        # which is the trade nothing in this repo has ever
                        # had to make.
                        cost = SLEEP_ROUNDS * SLEEP_COST_PER_ROUND
                        if energy > cost + 5.0:
                            layer.sleep(SLEEP_ROUNDS)
                            energy -= cost
                            slept_energy += cost
                            sleeps += 1
                            since_sleep = 0
                        else:
                            # Too poor to sleep. Skipped rather than
                            # forced, because an agent that consolidates
                            # itself to death is measuring the harness
                            # rather than the mechanism.
                            since_sleep = 0
                else:
                    # Interleaved: one short consolidation every N items,
                    # same total rounds over the same stretch, same cost
                    # per round. Only the shape differs.
                    if since_sleep >= REHEARSE_PER_ITEM:
                        cost = SLEEP_COST_PER_ROUND
                        if energy > cost + 5.0:
                            layer.sleep(1)
                            energy -= cost
                            slept_energy += cost
                            sleeps += 1
                        since_sleep = 0

        if energy <= 0:
            return dict(arm=arm, seed=seed, steps=step + 1, died=True,
                        eaten=eaten, poisoned=poisoned, sleeps=sleeps,
                        reversals=reversals, slept_energy=slept_energy)

    return dict(arm=arm, seed=seed, steps=max_steps, died=False,
                eaten=eaten, poisoned=poisoned, sleeps=sleeps,
                reversals=reversals, slept_energy=slept_energy)


ARMS = ["reflex", "online", "continuous", "sleeping", "driven"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lives", type=int, default=12,
                    help="lives per arm")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--out", default="survive.json")
    args = ap.parse_args()

    print(f"  {args.lives} lives per arm, {len(ARMS)} arms")
    print(f"  energy {START_ENERGY} start / {MAX_ENERGY} cap, "
          f"{STEP_COST} per step")
    print(f"  food +{FOOD_GAIN}, poison -{POISON_COST}, "
          f"sleep -{SLEEP_COST_PER_ROUND} per round")
    print(f"  the rule reverses every {REVERSAL_STEPS} steps")
    print(f"  a life reaching {args.max_steps} steps counts as survival\n",
          flush=True)

    runs = []
    for arm in ARMS:
        t0 = time.time()
        for seed in range(args.lives):
            r = live(arm, seed, args.max_steps)
            runs.append(r)
        lens = [r["steps"] for r in runs if r["arm"] == arm]
        print(f"  {arm:>11}  median {statistics.median(lens):>7.0f}  "
              f"mean {statistics.mean(lens):>7.0f}  "
              f"max {max(lens):>7.0f}  "
              f"survived {sum(1 for r in runs if r['arm'] == arm and not r['died'])}"
              f"/{args.lives}  {time.time() - t0:.0f}s", flush=True)

    with open(args.out, "w") as f:
        json.dump(dict(args=vars(args), runs=runs), f, indent=2)

    def get(arm, key):
        return [r[key] for r in runs if r["arm"] == arm]

    print("\n" + "=" * 78)
    print("SURVIVAL")
    print("=" * 78)
    print(f"  {'arm':>11} {'median':>8} {'mean':>8} {'worst':>7} "
          f"{'best':>7} {'ate':>7} {'poisoned':>9} {'sleeps':>7}")
    print("-" * 78)
    table = {}
    for arm in ARMS:
        lens = get(arm, "steps")
        table[arm] = statistics.median(lens)
        print(f"  {arm:>11} {statistics.median(lens):>8.0f} "
              f"{statistics.mean(lens):>8.0f} {min(lens):>7} "
              f"{max(lens):>7} "
              f"{statistics.mean(get(arm, 'eaten')):>7.0f} "
              f"{statistics.mean(get(arm, 'poisoned')):>9.0f} "
              f"{statistics.mean(get(arm, 'sleeps')):>7.0f}")

    print("\n" + "=" * 78)
    print("CHECKS, READ BEFORE THE ORDERING")
    print("=" * 78)

    reflex = table["reflex"]
    best = max(table.values())
    ok = True

    if reflex >= args.max_steps * 0.9:
        ok = False
        print(f"  WORLD TOO EASY: the random walker survived to "
              f"{reflex:.0f} steps. Nothing has to")
        print(f"  learn anything here, so no arm's number means anything. "
              f"Raise STEP_COST or")
        print(f"  lower FOOD_GAIN.")
    elif best <= reflex * 1.2:
        ok = False
        print(f"  NO ARM BEAT THE FLOOR: the best median was "
              f"{best:.0f} against the random")
        print(f"  walker's {reflex:.0f}. Either the world is too hard to "
              f"learn or the learning is")
        print(f"  worth nothing in it. Check the poisoned column: if every "
              f"arm eats as much")
        print(f"  poison as the reflex arm, nothing learned the rule.")
    else:
        print(f"  ok: the random walker died at {reflex:.0f} and the best "
              f"arm reached {best:.0f}.")
        print(f"  Learning is worth something in this world, so the "
              f"ordering below is readable.")

    # An arm that barely ate did not learn to discriminate, it learned to
    # abstain, and then starved. Its lifespan says nothing about the
    # mechanism it was meant to test. This is the same class of check as
    # the learnability gate in acting.py and it exists for the same reason.
    starved = [a for a in ARMS if a != "reflex"
               and statistics.mean(get(a, "eaten")) < 10]
    if starved:
        ok = False
        print(f"  ARMS THAT LEARNED TO ABSTAIN: "
              f"{', '.join(starved)}. Each ate fewer than 10 things in a")
        print(f"  whole life against the random walker's "
              f"{statistics.mean(get('reflex', 'eaten')):.0f}. They did not "
              f"learn which cell is food,")
        print(f"  they learned not to eat, and then starved. Their "
              f"lifespans measure the")
        print(f"  policy collapsing, not the mechanism under test.")

    spread = [max(get(a, "steps")) / max(1, min(get(a, "steps")))
              for a in ARMS]
    if max(spread) > 20:
        print(f"  NOTE: lifespans vary by more than 20x within an arm. "
              f"With {args.lives} lives")
        print(f"  the medians are not stable. Read the ordering as a hint "
              f"and rerun with more.")

    if ok:
        print("\n" + "=" * 78)
        print("WHAT IT SAYS")
        print("=" * 78)
        online, cont = table["online"], table["continuous"]
        sleep, driven = table["sleeping"], table["driven"]

        print(f"  learning over reflex        "
              f"{online - reflex:>+8.0f}")
        print(f"  the layer over bare online  {cont - online:>+8.0f}")
        print(f"  sleeping over interleaved   {sleep - cont:>+8.0f}")
        print(f"  the drive over value        {driven - sleep:>+8.0f}")

        if sleep > cont:
            print(f"\n  SLEEP SURVIVES BEING EXPENSIVE. It won on accuracy "
                  f"on 20 Sep when")
            print(f"  consolidating was free. Here it costs energy and "
                  f"gathers no food, and the")
            print(f"  bet still pays: {sleep:.0f} against {cont:.0f}.")
        else:
            print(f"\n  SLEEP DOES NOT PAY FOR ITSELF. It won on accuracy "
                  f"when it was free and")
            print(f"  loses here at {sleep:.0f} against {cont:.0f}. The "
                  f"20 Sep result is real and")
            print(f"  irrelevant, which is worth knowing before anything "
                  f"more is built on it.")

        if driven > sleep:
            print(f"\n  AND THE DRIVE PAYS. Exploration chosen by "
                  f"accumulated error beat eating")
            print(f"  what the model predicts is good, {driven:.0f} "
                  f"against {sleep:.0f}, with every")
            print(f"  exploratory step costing energy.")
        else:
            print(f"\n  THE DRIVE DOES NOT PAY. It beat every policy on "
                  f"20 Sep when exploring was")
            print(f"  free. Charged for, it loses at {driven:.0f} against "
                  f"{sleep:.0f}: curiosity is a")
            print(f"  luxury an agent with stakes cannot always afford.")

    print(f"\n  A 9x9 grid and a GRU. Lifespan here is not lifespan "
          f"anywhere else; what")
    print(f"  might transfer is the ORDERING, and only if it holds across "
          f"more lives.")
    print(f"\n  written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
