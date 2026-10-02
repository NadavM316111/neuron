"""Can it want a state it was never rewarded for reaching?

THE CLAIM BEING TESTED, AND THE TRAP IT HAS TO AVOID. A goal is a state of
the world the system prefers, acts to bring about, and keeps working toward
even when nothing external rewards it. The last clause is the whole thing.
If the system is REWARDED for reaching the state, this is reinforcement
learning and the "want" is the reward function someone wrote, not the
system's own goal. So the experiment withholds reward entirely. The state
is preferred internally; nothing ever pays the system for getting there.

The drive (drive_scale.py, 2 Oct) was a want to reduce uncertainty, and it
collapsed. This is a different and harder kind of want: a want for a STATE
rather than a want for novelty. It is the kind the vision actually needs.

HOW A GOAL CAN WORK WITHOUT REWARD. In three stages, and the separation is
the point:

  1. LEARN THE WORLD, no goal, no reward. The agent moves at random and
     learns a model: given where I am and what I do, where do I end up.
     This is pure prediction, the one thing this project does well.

  2. HOLD A PREFERRED STATE. A particular cell is designated as wanted.
     Not rewarded. Just carried, as the target the planner aims at.

  3. PLAN TOWARD IT. At each step the agent uses its learned model to
     imagine where each action leads, and over a short horizon picks the
     action sequence whose predicted end is closest to the wanted state.
     It chooses the action because the MODEL says it leads toward what is
     wanted, not because anything rewards the choice.

If step 3 reaches the wanted state using only the model from step 1, the
system pursued a goal it was never rewarded for. That is goal-directed
behaviour from a world model plus a preference, which is the honest minimum
version of wanting a state.

THE ARMS, so the result cannot be mistaken for something cheaper:

  random      ignores the goal, moves at random. The floor. If planning
              does not beat this, the plan is doing nothing.
  reactive    greedy one step: take the action whose IMMEDIATE predicted
              next cell is closest to the goal. No lookahead. This is the
              cheapest thing that uses the goal at all, and it will get
              stuck wherever a wall needs going AROUND.
  planner     imagines HORIZON steps ahead with the learned model and
              picks the best rollout. Lookahead is what lets it route
              around obstacles toward a state it cannot reach in one move.
  oracle      plans with the TRUE world instead of the learned model. The
              ceiling: how well planning would do if the model were
              perfect. The gap between planner and oracle is how much the
              learned model costs.

WHAT EACH OUTCOME MEANS.

  planner reaches the goal, random does not
      Goal-directed behaviour from a learned model and a held preference,
      with no reward. The first real want-for-a-state in this project.
  planner no better than reactive
      One-step greed was enough on this world; lookahead bought nothing.
      Use a world with obstacles that force detours (this one has them) so
      that is a real statement rather than an artefact of an easy map.
  planner far below oracle
      The idea works but the learned model is too weak to plan through.
      That points at the model, not at goals.
  planner no better than random
      The model is not good enough to plan with at all, or planning over
      it is broken. A real negative about this specific attempt.

HONEST SCOPE. A preferred cell on a grid is the smallest possible goal. It
is not a rich want, and "reach this cell" is a long way from "bring about a
state of the world I care about". But it is the first version where the
preference is internal, never rewarded, and pursued through a learned
model, which are exactly the three properties a goal needs.

    python goal.py
    python goal.py --learn 8000 --trials 200
"""

import argparse
import random
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F


SIZE = 7
ACTIONS = ["up", "down", "left", "right", "stay"]
DELTA = {"up": (0, -1), "down": (0, 1), "left": (-1, 0),
         "right": (1, 0), "stay": (0, 0)}


class World:
    """A grid with walls that force going AROUND. wall: one divide, one gap
    (easy). spiral: two combs with gaps at opposite ends, forcing a long
    detour where greedy fails and planning does not."""

    def __init__(self, seed=0, kind="wall"):
        rng = random.Random(seed)
        self.walls = set()
        self.kind = kind
        if kind == "wall":
            gap = rng.randrange(SIZE)
            for y in range(SIZE):
                if y != gap:
                    self.walls.add((SIZE // 2, y))
        elif kind == "spiral":
            for y in range(SIZE - 1):
                self.walls.add((2, y))
            for y in range(1, SIZE):
                self.walls.add((4, y))

    def step(self, pos, action):
        x, y = pos
        dx, dy = DELTA[action]
        nx, ny = x + dx, y + dy
        if not (0 <= nx < SIZE and 0 <= ny < SIZE):
            return pos
        if (nx, ny) in self.walls:
            return pos
        return (nx, ny)

    def free(self):
        return [(x, y) for x in range(SIZE) for y in range(SIZE)
                if (x, y) not in self.walls]


def encode(pos, action):
    v = torch.zeros(SIZE * SIZE + len(ACTIONS))
    v[pos[1] * SIZE + pos[0]] = 1.0
    v[SIZE * SIZE + ACTIONS.index(action)] = 1.0
    return v


class Model(nn.Module):
    """The learned world model: given a cell and an action, predict the
    next cell as a distribution over all cells. No reward anywhere in it.
    It only ever learns where actions lead."""

    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(SIZE * SIZE + len(ACTIONS), 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, SIZE * SIZE))

    def forward(self, pos, action):
        return self.net(encode(pos, action).unsqueeze(0))

    def predict(self, pos, action):
        with torch.no_grad():
            logits = self.forward(pos, action)[0]
            i = int(logits.argmax().item())
            return (i % SIZE, i // SIZE)


def learn_world(world, steps, seed):
    """Stage 1: move at random, learn where actions lead. No goal, no
    reward. Returns a model that knows the world's dynamics."""
    model = Model(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = random.Random(seed + 1)
    pos = rng.choice(world.free())
    for _ in range(steps):
        action = rng.choice(ACTIONS)
        nxt = world.step(pos, action)
        logits = model(pos, action)
        target = torch.tensor([nxt[1] * SIZE + nxt[0]])
        loss = F.cross_entropy(logits, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        pos = nxt
    return model


def model_accuracy(world, model):
    """How often the learned model predicts the true next cell. The plan is
    only as good as this."""
    hit = tot = 0
    for pos in world.free():
        for a in ACTIONS:
            if model.predict(pos, a) == world.step(pos, a):
                hit += 1
            tot += 1
    return 100.0 * hit / tot


def dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def plan_action(pos, goal, step_fn, horizon, rng):
    """Pick the first action of the shortest MODEL-path to the goal.

    reactive (horizon 1) just takes the action whose predicted next cell is
    closest to the goal, which gets stuck where a wall needs going around.

    planner (horizon > 1) does real search: it explores, via the learned
    model, where sequences of actions lead, up to `horizon` deep, and
    returns the first action of the shortest path that reaches the goal. If
    none reaches it within the horizon, it falls back to the first action
    of the path that got closest. Tracking visited cells is what stops the
    oscillation that made the first version of this worse than random: a
    greedy rollout with no memory walks toward the goal and back forever.

    The goal enters ONLY here, as what the search aims at. Nothing rewards
    reaching it.
    """
    if horizon <= 1:
        best_a, best_d = None, 1e9
        for a in ACTIONS:
            d = dist(step_fn(pos, a), goal)
            if d < best_d or (d == best_d and rng.random() < 0.5):
                best_d, best_a = d, a
        return best_a

    # breadth-first search over the model, shortest path wins
    from collections import deque as _dq
    frontier = _dq([(pos, None, 0)])     # cell, first-action, depth
    seen = {pos}
    best_first, best_d = None, dist(pos, goal)
    order = list(ACTIONS)
    while frontier:
        cell, first, depth = frontier.popleft()
        if cell == goal and first is not None:
            return first                 # shortest path, found first
        d = dist(cell, goal)
        if d < best_d:
            best_d, best_first = d, first
        if depth >= horizon:
            continue
        rng.shuffle(order)
        for a in order:
            nxt = step_fn(cell, a)
            if nxt not in seen:
                seen.add(nxt)
                frontier.append((nxt, first if first is not None else a,
                                 depth + 1))
    return best_first if best_first is not None else rng.choice(ACTIONS)


def run_trial(world, model, mode, goal, start, horizon, rng, max_steps=40):
    """One attempt to reach the goal. Returns whether it arrived and in how
    many steps. No reward is given at any point; arrival is only measured,
    never fed back."""
    if mode == "oracle":
        step_fn = world.step
    else:
        step_fn = model.predict
    pos = start
    for t in range(max_steps):
        if pos == goal:
            return True, t
        if mode == "random":
            a = rng.choice(ACTIONS)
        elif mode == "reactive":
            a = plan_action(pos, goal, step_fn, 1, rng)
        else:
            a = plan_action(pos, goal, step_fn, horizon, rng)
        pos = world.step(pos, a)      # the REAL world moves the agent
    return pos == goal, max_steps


ARMS = ["random", "reactive", "planner", "oracle"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--learn", type=int, default=5000,
                    help="random steps to learn the world model")
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--trials", type=int, default=120)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--world", choices=["wall", "spiral"], default="wall")
    args = ap.parse_args()

    print(f"  a {SIZE}x{SIZE} grid, world={args.world!r}")
    print(f"  stage 1: learn the world from {args.learn} random steps, no "
          f"goal, no reward")
    print(f"  stage 2: hold a preferred cell and try to reach it using "
          f"only the model")
    print(f"  nothing rewards arrival; it is measured, never fed back\n")

    reached = {a: [] for a in ARMS}
    steps_taken = {a: [] for a in ARMS}
    accs = []

    for seed in range(args.seeds):
        world = World(seed, kind=args.world)
        model = learn_world(world, args.learn, seed)
        acc = model_accuracy(world, model)
        accs.append(acc)
        print(f"  seed {seed}: learned world model is {acc:.0f}% accurate",
              flush=True)

        rng = random.Random(seed + 100)
        free = world.free()
        for _ in range(args.trials):
            start = rng.choice(free)
            goal = rng.choice(free)
            while goal == start:
                goal = rng.choice(free)
            for arm in ARMS:
                ok, t = run_trial(world, model, arm, goal, start,
                                  args.horizon, rng)
                reached[arm].append(1 if ok else 0)
                if ok:
                    steps_taken[arm].append(t)

    print("\n" + "=" * 68)
    print("DID IT REACH A GOAL IT WAS NEVER REWARDED FOR?")
    print("=" * 68)
    print(f"  learned model accuracy: {statistics.mean(accs):.0f}%\n")
    print(f"  {'arm':>10} {'reached':>9} {'avg steps':>11}")
    print("-" * 68)
    for arm in ARMS:
        r = 100.0 * statistics.mean(reached[arm])
        st = statistics.mean(steps_taken[arm]) if steps_taken[arm] else 0
        print(f"  {arm:>10} {r:>8.0f}% {st:>11.1f}")

    print("\n" + "=" * 68)
    print("WHAT IT SAYS")
    print("=" * 68)

    rnd = 100.0 * statistics.mean(reached["random"])
    rea = 100.0 * statistics.mean(reached["reactive"])
    pla = 100.0 * statistics.mean(reached["planner"])
    orc = 100.0 * statistics.mean(reached["oracle"])

    if pla <= rnd + 5:
        print(f"  NO GOAL PURSUIT. The planner reached the goal {pla:.0f}% "
              f"against random's {rnd:.0f}%.")
        print(f"  Either the learned model is too weak to plan through "
              f"(it was "
              f"{statistics.mean(accs):.0f}% accurate), or planning over it "
              f"is broken. A real")
        print(f"  negative about this attempt, not a statement that goals "
              f"are impossible.")
    else:
        print(f"  IT PURSUED A GOAL WITH NO REWARD. The planner reached a "
              f"preferred cell")
        print(f"  {pla:.0f}% of the time against random's {rnd:.0f}%, using "
              f"only a world model it")
        print(f"  learned by moving at random and a preference it was "
              f"never paid to satisfy.")
        print(f"  That is goal-directed behaviour from a model plus a "
              f"held state: the first")
        print(f"  want-for-a-state in this project, and a different thing "
              f"from the drive.")

        if pla > rea + 5:
            print(f"\n  AND LOOKAHEAD MATTERS: the planner ({pla:.0f}%) "
                  f"beat one-step reactive")
            print(f"  ({rea:.0f}%), which gets stuck where the wall needs "
                  f"going around. Planning,")
            print(f"  not just greedy pull toward the goal, is what reaches "
                  f"it.")
        else:
            print(f"\n  Lookahead added little here ({pla:.0f}% vs "
                  f"{rea:.0f}% reactive). On a world")
            print(f"  with harder detours it would matter more.")

        gap = orc - pla
        print(f"\n  The learned model cost {gap:.0f} points against the "
              f"oracle ({orc:.0f}% with the")
        print(f"  true world). That gap is how much a better model would "
              f"buy.")

    print(f"\n  A preferred cell on a grid is the smallest possible goal. "
          f"What makes it")
    print(f"  the real thing: the preference is internal, never rewarded, "
          f"and pursued")
    print(f"  through a model the system learned itself.")


if __name__ == "__main__":
    main()
