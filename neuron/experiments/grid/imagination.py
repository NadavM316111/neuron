"""Can it imagine a future before it lives it?

WHAT IMAGINATION IS, FUNCTIONALLY. Not reacting to what is in front of it, but
running possible futures INTERNALLY, in its own world-model, and choosing
based on what it pictures. You imagine eating the whole cake and feeling
sick, and you stop, without having to actually get sick first. That is the
power: getting it right the FIRST time by simulating the consequence, rather
than learning it by suffering it.

The being already has the pieces: a world-model (what my actions lead to)
and wants (what states I prefer). Imagination is running the model forward
in the mind, several steps deep, across several possible actions, and
judging each imagined future against what it wants. It then takes the action
whose IMAGINED future is best. Nothing happened in the world yet; it all
happened inside.

THE TEST, built so imagination cannot be faked by luck or trial-and-error. A
world with a TRAP OF FORESIGHT: in certain states, the immediately appealing
action (the one with the best next-step reward) leads, a few steps later, to
a bad place; the wise action looks worse now but leads somewhere good. A
being that only reacts to immediate value walks into the trap every time. A
being that IMAGINES the next several steps sees the bad place coming and
avoids it -- on the FIRST encounter, because it simulated rather than
experienced.

  reactor    picks the action with the best IMMEDIATE predicted value. no
             lookahead. the purely reactive floor.
  imaginer   simulates several steps ahead through its LEARNED world-model
             across candidate actions, scores each imagined future by its
             wants, and picks the best. real imagination.
  oracle     imagines through the TRUE world instead of a learned model. the
             ceiling: how well foresight does with a perfect imagination.

  FIRST-ENCOUNTER success is the key measure: on states it has never been
  trapped by before, does the imaginer already avoid the trap? If yes, it is
  simulating the future, not remembering a past mistake.

  imaginer beats reactor, and avoids traps on first encounter
      It imagined the consequence and chose against the tempting now. Real
      foresight from internal simulation.
  imaginer no better than reactor
      Its imagination (the learned model it simulates through) is too weak
      to see far enough, or the horizon is too short.

    python imagination.py
    python imagination.py --horizon 5 --steps 4000
"""

import argparse
import random
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F


GRID = 6
ACTIONS = ["up", "down", "left", "right"]
DELTA = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


class World:
    """A grid with a REAL TRAP OF FORESIGHT. The sweet cells give +2 now, but
    stepping onto a sweet cell FORCES you onto a pit cell next (a one-way
    slide, built into step()). So taking the tempting reward now genuinely
    costs -5 the following step. A reactor grabbing +2 cannot escape the
    slide; only a being that imagines the next step refuses the +2 and takes
    the safe path to a modest, real goal."""

    def __init__(self, seed=0):
        rng = random.Random(seed)
        self.reward = {}
        self.pit = set()
        self.sweet = set()
        px = GRID - 2
        for y in range(GRID):
            self.pit.add((px, y))
            self.reward[(px, y)] = -5.0
        for y in range(GRID):
            self.sweet.add((px - 1, y))
            self.reward[(px - 1, y)] = 2.0
        self.goal = (0, GRID - 1)
        self.reward[self.goal] = 3.0

    def step(self, pos, action):
        x, y = pos
        # THE SLIDE: if you are on a sweet cell, you are pulled into the pit
        # next no matter what you choose. this is what makes the +2 a trap.
        if pos in self.sweet:
            return (pos[0] + 1, pos[1])          # into the pit
        dx, dy = DELTA[action]
        nx, ny = max(0, min(GRID - 1, x + dx)), max(0, min(GRID - 1, y + dy))
        return (nx, ny)

    def rew(self, pos):
        return self.reward.get(pos, -0.1)


def encode(pos):
    v = torch.zeros(GRID * GRID)
    v[pos[1] * GRID + pos[0]] = 1.0
    return v


class Model(nn.Module):
    """The being's IMAGINATION substrate: a learned world-model predicting
    the next cell from (cell, action). Imagining = rolling this forward."""

    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(GRID * GRID + len(ACTIONS), 128), nn.ReLU(),
            nn.Linear(128, GRID * GRID))

    def forward(self, pos, action):
        a = torch.zeros(len(ACTIONS))
        a[ACTIONS.index(action)] = 1.0
        x = torch.cat([encode(pos), a])
        return self.net(x.unsqueeze(0))[0]

    def predict(self, pos, action):
        with torch.no_grad():
            i = int(self.forward(pos, action).argmax().item())
            return (i % GRID, i // GRID)


def learn_model(world, steps, seed):
    model = Model(seed)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    rng = random.Random(seed + 1)
    pos = (rng.randrange(GRID), rng.randrange(GRID))
    for _ in range(steps):
        a = rng.choice(ACTIONS)
        nxt = world.step(pos, a)
        pred = model.forward(pos, a)
        target = torch.tensor([nxt[1] * GRID + nxt[0]])
        loss = F.cross_entropy(pred.unsqueeze(0), target)
        opt.zero_grad(); loss.backward(); opt.step()
        pos = nxt
    return model


def imagine(world, model, pos, first_action, horizon, step_fn):
    """Roll a future forward in the MIND: take first_action, then follow a
    greedy-reward rollout for `horizon` steps using step_fn (the learned
    model for the imaginer, the true world for the oracle). Return the total
    imagined reward of that future."""
    p = step_fn(pos, first_action)
    total = world.rew(p)          # reward judged by the being's wants
    for _ in range(horizon - 1):
        # within the imagined rollout, continue toward imagined reward
        best_a, best_r, best_p = None, -1e9, p
        for a in ACTIONS:
            np_ = step_fn(p, a)
            if world.rew(np_) > best_r:
                best_r, best_a, best_p = world.rew(np_), a, np_
        p = best_p
        total += best_r
    return total


def run(mode, world, model, steps, horizon, seed):
    rng = random.Random(seed + 2)
    pos = (0, 0)
    total = 0.0
    trap_hits = 0
    first_trap_encounters = 0
    first_trap_avoided = 0
    seen_trap_states = set()

    for t in range(steps):
        # is this a state from which the TEMPTING action leads to the trap?
        # (adjacent to a sweet cell). used only to score first-encounter.
        tempting_here = any(world.step(pos, a) in world.sweet
                            for a in ACTIONS) and pos not in world.sweet

        if mode == "reactor":
            # purely immediate: pick action with best next-step reward
            a = max(ACTIONS, key=lambda a: world.rew(world.step(pos, a)))
        elif mode == "oracle":
            a = max(ACTIONS, key=lambda a: imagine(
                world, model, pos, a, horizon, world.step))
        else:  # imaginer: simulate through the LEARNED model
            a = max(ACTIONS, key=lambda a: imagine(
                world, model, pos, a, horizon, model.predict))

        nxt = world.step(pos, a)
        r = world.rew(nxt)
        total += r
        if nxt in world.pit:
            trap_hits += 1

        # first-encounter test: the first time it faces a tempting state,
        # does it avoid stepping into the sweet-then-pit trap?
        if tempting_here and pos not in seen_trap_states:
            seen_trap_states.add(pos)
            first_trap_encounters += 1
            # did it avoid going to the sweet cell (which leads to pit)?
            if nxt not in world.sweet:
                first_trap_avoided += 1

        pos = nxt
        if rng.random() < 0.05:      # occasional reset to vary states
            pos = (rng.randrange(GRID), rng.randrange(GRID))

    foresight = (100.0 * first_trap_avoided / first_trap_encounters
                 if first_trap_encounters else 0.0)
    return dict(reward=total, trap_hits=trap_hits,
                first_encounter_foresight=foresight)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--learn", type=int, default=6000)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    print(f"  a world with a trap of foresight: a tempting cell (+2 now) "
          f"sits right before")
    print(f"  a pit (-5 soon). reacting to the tempting reward walks into "
          f"the pit. does a")
    print(f"  being that IMAGINES {args.horizon} steps ahead see it coming "
          f"and avoid it?\n")

    modes = ["reactor", "imaginer", "oracle"]
    agg = {m: {"reward": [], "trap_hits": [], "first_encounter_foresight": []}
           for m in modes}
    accs = []
    for seed in range(args.seeds):
        world = World(seed)
        model = learn_model(world, args.learn, seed)
        for m in modes:
            r = run(m, world, model, args.steps, args.horizon, seed)
            for k in agg[m]:
                agg[m][k].append(r[k])

    print("=" * 68)
    print("DID IT IMAGINE THE FUTURE AND CHOOSE WELL?")
    print("=" * 68)
    print(f"  {'being':>10} {'reward':>9} {'trap hits':>10} "
          f"{'foresight(1st)':>15}")
    print("-" * 68)
    for m in modes:
        note = {"reactor": "reacts to now (floor)",
                "imaginer": "simulates the future",
                "oracle": "perfect imagination (ceiling)"}[m]
        print(f"  {m:>10} {statistics.mean(agg[m]['reward']):>9.0f} "
              f"{statistics.mean(agg[m]['trap_hits']):>10.0f} "
              f"{statistics.mean(agg[m]['first_encounter_foresight']):>13.0f}%"
              f"  {note}")

    rr = statistics.mean(agg["reactor"]["reward"])
    ir = statistics.mean(agg["imaginer"]["reward"])
    orc = statistics.mean(agg["oracle"]["reward"])
    rtr = statistics.mean(agg["reactor"]["trap_hits"])
    itr = statistics.mean(agg["imaginer"]["trap_hits"])
    ifs = statistics.mean(agg["imaginer"]["first_encounter_foresight"])

    print("\n" + "=" * 68)
    print("WHAT IT SAYS")
    print("=" * 68)
    if ir > rr + 0.05 * abs(rr) and itr < rtr * 0.7:
        print(f"  IT IMAGINED, AND CHOSE WELL. The imaginer scored {ir:.0f} "
              f"against the reactor's")
        print(f"  {rr:.0f}, and walked into the trap {itr:.0f} times against "
              f"{rtr:.0f}. By running the future")
        print(f"  forward in its own model before acting, it saw the pit "
              f"beyond the tempting")
        print(f"  cell and chose against the appealing now.")
        if ifs > 55:
            print(f"\n  AND IT WAS FORESIGHT, NOT HINDSIGHT: on states it had "
                  f"NEVER been trapped by,")
            print(f"  it already avoided the trap {ifs:.0f}% of the time. It "
                  f"did not learn the pit by")
            print(f"  falling in. It imagined falling in, and did not. That "
                  f"is imagination.")
        gap = orc - ir
        print(f"\n  Its imagination cost {gap:.0f} against a perfect one "
              f"(oracle {orc:.0f}); the learned")
        print(f"  model is good enough to foresee through, not perfect.")
    else:
        print(f"  imagination did not clearly help: imaginer {ir:.0f} vs "
              f"reactor {rr:.0f}, traps {itr:.0f} vs {rtr:.0f}.")
        print(f"  likely the learned model is too weak to simulate through, "
              f"or the horizon too short.")

    print(f"\n  Imagination here is the being running its own world-model "
          f"forward, in the")
    print(f"  mind, and choosing by what it pictures. It is the difference "
          f"between learning")
    print(f"  from pain and foreseeing it. The substrate is the model and "
          f"wants it already")
    print(f"  has; imagination is using them inwardly, before acting.")


if __name__ == "__main__":
    main()
