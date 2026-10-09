"""Planning: a multi-step plan toward a far end, held and adapted.

THE FACULTY. Imagination looks a few steps ahead. Planning chains that into a
whole route to a DISTANT goal, one that needs a sequence of sub-steps no
short glance could find, commits to the route, and re-plans when knocked off
it. It is the difference between "what helps me right now" and "here is the
path to where I am trying to get, several moves from here."

WHY IT NEEDS MORE THAN IMAGINATION. Some goals require doing things that look
useless or even wrong in the short term: go get the key (which is away from
the door), to open the door, to reach the goal. A reactor heading straight
for the goal jams against the locked door forever. A short-horizon imaginer
cannot see far enough down the chain to know the key matters. Only a being
that plans the WHOLE sequence, key then door then goal, gets there. That is
purposeful, multi-step action toward a far end, and it is the seed of a life
that pursues things that take a long time.

THE TEST. A room with a locked door between the being and the goal, and a
key elsewhere. Reaching the goal REQUIRES: go to the key, then to the door
(now openable), then to the goal. The key is not on the way; fetching it
means moving AWAY from the goal first.

  reactor        moves greedily toward the goal. jams at the locked door.
  short-imaginer looks a few steps ahead. still cannot see that the far-off
                 key is what unlocks the path. jams too.
  planner        searches a full plan through its world-model: find the
                 sequence key -> door -> goal, commit to it, follow it, and
                 re-plan if knocked off course. reaches the goal.

  planner reaches the goal, reactor and short-imaginer do not
      It formed and held a multi-step plan toward a distant end, doing the
      counter-intuitive sub-step (fetch the key, away from the goal) because
      the plan required it. That is long-horizon purpose.

    python planning.py
    python planning.py --trials 200
"""

import argparse
import random
import statistics
from collections import deque


GRID = 9


class Room:
    """Being, key, locked door, goal. The door blocks the only path to the
    goal until the key is held. Fetching the key means going AWAY from the
    goal first, which only a planner will do."""

    def __init__(self, seed):
        rng = random.Random(seed)
        # a wall splits the room; the door is the one gap, and it is locked
        self.wall_x = GRID // 2
        self.door = (self.wall_x, rng.randrange(GRID))
        self.walls = {(self.wall_x, y) for y in range(GRID)
                      if (self.wall_x, y) != self.door}
        # being and key on the LEFT, goal on the RIGHT (past the door)
        self.start = (rng.randrange(self.wall_x), rng.randrange(GRID))
        self.key = (rng.randrange(self.wall_x), rng.randrange(GRID))
        while self.key == self.start:
            self.key = (rng.randrange(self.wall_x), rng.randrange(GRID))
        self.goal = (rng.randrange(self.wall_x + 1, GRID), rng.randrange(GRID))

    def passable(self, pos, has_key):
        x, y = pos
        if not (0 <= x < GRID and 0 <= y < GRID):
            return False
        if pos in self.walls:
            return False
        if pos == self.door and not has_key:
            return False          # locked until key held
        return True

    def neighbors(self, pos, has_key):
        x, y = pos
        out = []
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            np_ = (x + dx, y + dy)
            if self.passable(np_, has_key):
                out.append(np_)
        return out


def bfs(room, start, target, has_key):
    """Shortest path from start to target given key state. Returns the path
    or None. This is the being imagining a route through its model."""
    q = deque([(start, [start])])
    seen = {start}
    while q:
        pos, path = q.popleft()
        if pos == target:
            return path
        for np_ in room.neighbors(pos, has_key):
            if np_ not in seen:
                seen.add(np_)
                q.append((np_, path + [np_]))
    return None


def run(mode, trials, seed, horizon=4):
    rng = random.Random(seed)
    reached = 0
    for _ in range(trials):
        room = Room(seed * 1000 + rng.randrange(100000))
        pos = room.start
        has_key = False
        steps = 0
        max_steps = 120
        got = False

        while steps < max_steps:
            steps += 1
            if pos == room.key:
                has_key = True
            if pos == room.goal:
                got = True
                break

            if mode == "reactor":
                # greedy toward the goal, ignoring the key and the lock
                target = room.goal
                nbrs = room.neighbors(pos, has_key)
                if not nbrs:
                    break
                a = min(nbrs, key=lambda n: abs(n[0] - target[0]) +
                        abs(n[1] - target[1]))
                pos = a

            elif mode == "short":
                # short imaginer: BFS but only up to `horizon` steps, toward
                # the goal. cannot see the key chain if it is far.
                path = bfs(room, pos, room.goal, has_key)
                if path and len(path) <= horizon + 1:
                    pos = path[1]
                else:
                    # within horizon it cannot reach goal; drift toward goal
                    nbrs = room.neighbors(pos, has_key)
                    if not nbrs:
                        break
                    pos = min(nbrs, key=lambda n: abs(n[0] - room.goal[0]) +
                              abs(n[1] - room.goal[1]))

            else:  # planner: plan the WHOLE chain
                # if no key yet, plan: me -> key -> door -> goal. commit to
                # the first leg. this is multi-step purpose.
                if not has_key:
                    leg = bfs(room, pos, room.key, has_key)
                else:
                    leg = bfs(room, pos, room.goal, has_key)
                if leg and len(leg) > 1:
                    pos = leg[1]
                else:
                    break

        if got:
            reached += 1
    return 100.0 * reached / trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=4)
    args = ap.parse_args()

    print("  a locked door stands between the being and the goal. the key is "
          "elsewhere, and")
    print("  fetching it means going AWAY from the goal first. reaching the "
          "goal needs the")
    print("  whole plan: key -> door -> goal. who gets there?\n")

    modes = ["reactor", "short", "planner"]
    agg = {m: [] for m in modes}
    for seed in range(args.seeds):
        for m in modes:
            agg[m].append(run(m, args.trials, seed, args.horizon))

    print("=" * 62)
    print("WHO REACHED THE DISTANT GOAL?")
    print("=" * 62)
    labels = {"reactor": "greedy toward goal (jams at door)",
              "short": f"looks {args.horizon} steps ahead (can't see chain)",
              "planner": "plans key -> door -> goal"}
    for m in modes:
        print(f"  {m:>9} {statistics.mean(agg[m]):>6.0f}%   {labels[m]}")

    r = statistics.mean(agg["reactor"])
    sh = statistics.mean(agg["short"])
    pl = statistics.mean(agg["planner"])

    print("\n" + "=" * 62)
    print("WHAT IT SAYS")
    print("=" * 62)
    if pl > r + 30 and pl > sh + 20:
        print(f"  IT PLANNED TOWARD A FAR END. The planner reached the goal "
              f"{pl:.0f}% of the time,")
        print(f"  against {r:.0f}% for greedy reaction and {sh:.0f}% for "
              f"short lookahead. It did the")
        print(f"  counter-intuitive thing, going AWAY from the goal to fetch "
              f"the key, because")
        print(f"  the whole plan required it. That is purposeful, multi-step "
              f"action toward a")
        print(f"  distant goal: the seed of pursuing things that take a long "
              f"time.")
    else:
        print(f"  planning did not clearly separate: planner {pl:.0f}%, "
              f"reactor {r:.0f}%, short {sh:.0f}%.")

    print(f"\n  Planning is imagination CHAINED: not one glance ahead but a "
          f"whole route to a")
    print(f"  far goal, held and re-planned. It is what lets a being pursue "
          f"an end many steps")
    print(f"  away, through sub-steps that make no sense on their own. The "
          f"substrate is the")
    print(f"  world-model it already has; planning is searching it to the "
          f"end, not just the")
    print(f"  next step.")


if __name__ == "__main__":
    main()
