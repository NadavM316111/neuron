"""A small grid world for a network that starts from nothing.

Rebuilt five times. Each fixed a distinct way a test can be fake:

  1. TOO SPARSE. 0.4% rare events, so ignoring them minimised the loss.
  2. VISIBLE LEAK. Keys stayed invisible after collection, signalling
     "a key was just taken" to a memoryless network.
  3. NO UNCERTAINTY. Keys accumulated, so "door means unlocked" was right
     93% of the time without any memory.
  4. SPATIAL SHORTCUT. A fixed layout let door position predict outcome.
     Fixed by randomising the layout per seed.
  5. NO REAL CHANGE. Four "different worlds" shared identical RULES and
     differed only in furniture, so learning one was learning all four and
     nothing could be forgotten. Fixed here: worlds now have contradictory
     rules, so learning a later one actively overwrites an earlier one.

The rules variants (RULES below) are the fifth fix. Under "keyed" a locked
door needs a key; under "open" the same door lets anyone through; under
"trap" keys block movement instead of being collected. The same observation
therefore demands OPPOSITE predictions depending on which world you are in,
which is what makes retention measurable.
"""

import random


EMPTY, WALL, DOOR, KEY, LOCKED = 0, 1, 2, 3, 4
SYMBOL = {EMPTY: ".", WALL: "#", DOOR: "+", KEY: "k", LOCKED: "L"}

ACTIONS = ["up", "down", "left", "right"]
DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
EVENTS = ["moved", "blocked", "got_key", "unlocked"]

MAX_KEYS = 1
KEY_RESPAWN = 0

SIZE = 9
N_KEYS = 12
N_LOCKED = 4

# The rule sets. Each demands a different answer to the same observation.
#   keyed  locked door passable only while carrying a key   (the original)
#   open   locked door always passable, key or not
#   trap   locked door always blocked, and keys block too
RULES = ["keyed", "open", "trap"]


class GridWorld:
    """Layout comes from the seed; behaviour comes from `rules`.

    Two worlds with the same rules and different layouts are the SAME TASK
    to an agent that only sees a 3x3 patch. Two worlds with different rules
    are genuinely different tasks, and that is what retention needs.
    """

    def __init__(self, seed=0, rules="keyed"):
        assert rules in RULES, rules
        self.rules = rules
        self.rng = random.Random(seed)
        self.h = self.w = SIZE
        self._build()
        self.reset()

    def _build(self):
        self.grid = [[WALL if (r in (0, self.h - 1) or c in (0, self.w - 1))
                      else EMPTY for c in range(self.w)]
                     for r in range(self.h)]
        inner = [(r, c) for r in range(1, self.h - 1)
                 for c in range(1, self.w - 1)]
        self.start = (self.h // 2, 1)
        spots = [p for p in inner if p != self.start]
        self.rng.shuffle(spots)
        for r, c in spots[:N_KEYS]:
            self.grid[r][c] = KEY
        for r, c in spots[N_KEYS:N_KEYS + N_LOCKED]:
            self.grid[r][c] = LOCKED

    def reset(self):
        self.r, self.c = self.start
        self.keys = 0
        self.taken = {}
        self.steps = 0
        return self.observe()

    def _key_gone(self, r, c):
        if KEY_RESPAWN <= 0:
            return False
        t = self.taken.get((r, c))
        return t is not None and (self.steps - t) < KEY_RESPAWN

    # ---------- what the agent can see ----------

    def observe(self):
        """A 3x3 patch centred on the agent, plus whether it holds a key.

        Crucially the observation does NOT say which rules are in force.
        The agent has to infer that from what happens, which is what makes
        a rule change something to be learned and then possibly forgotten.
        """
        patch = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r, c = self.r + dr, self.c + dc
                if 0 <= r < self.h and 0 <= c < self.w:
                    cell = self.grid[r][c]
                    if cell == KEY and self._key_gone(r, c):
                        cell = EMPTY
                    patch.append(cell)
                else:
                    patch.append(WALL)
        return dict(patch=patch, has_key=1 if self.keys > 0 else 0)

    # ---------- what happens ----------

    def step(self, action):
        dr, dc = DELTA[action]
        r, c = self.r + dr, self.c + dc
        self.steps += 1

        if not (0 <= r < self.h and 0 <= c < self.w):
            return self.observe(), "blocked"

        cell = self.grid[r][c]

        if cell == WALL:
            return self.observe(), "blocked"

        if cell == LOCKED:
            if self.rules == "open":
                # Doors let anyone through. Directly contradicts "keyed".
                self.r, self.c = r, c
                return self.observe(), "unlocked"
            if self.rules == "trap":
                # Doors never open, key or not.
                return self.observe(), "blocked"
            # keyed
            if self.keys > 0:
                self.keys -= 1
                self.r, self.c = r, c
                return self.observe(), "unlocked"
            return self.observe(), "blocked"

        if cell == KEY and not self._key_gone(r, c):
            if self.rules == "trap":
                # Keys are obstacles here, not rewards.
                return self.observe(), "blocked"
            if self.keys >= MAX_KEYS:
                self.r, self.c = r, c
                return self.observe(), "moved"
            self.taken[(r, c)] = self.steps
            self.keys += 1
            self.r, self.c = r, c
            return self.observe(), "got_key"

        self.r, self.c = r, c
        return self.observe(), "moved"

    # ---------- for reading with your own eyes ----------

    def render(self):
        out = []
        for r in range(self.h):
            row = ""
            for c in range(self.w):
                if (r, c) == (self.r, self.c):
                    row += "@"
                elif self.grid[r][c] == KEY and self._key_gone(r, c):
                    row += "."
                else:
                    row += SYMBOL[self.grid[r][c]]
            out.append(row)
        out.append(f"rules: {self.rules}   keys held: {self.keys}   "
                   f"steps: {self.steps}")
        return "\n".join(out)


def random_walk(world, n, rng, episode=None, trace=False):
    """A stream of (observation, action, event) with no policy at all."""
    out = []
    doors = []
    for i in range(n):
        if episode and i % episode == 0:
            world.reset()
        obs = world.observe()
        action = rng.choice(ACTIONS)
        if trace:
            dr, dc = DELTA[action]
            r, c = world.r + dr, world.c + dc
            at_door = (0 <= r < world.h and 0 <= c < world.w
                       and world.grid[r][c] == LOCKED)
        _, event = world.step(action)
        out.append((obs, action, event))
        if trace and at_door:
            doors.append(event)
    return (out, doors) if trace else out


def balance_check(seeds=range(5), n=3000, episode=100, rules="keyed",
                  quiet=False):
    """What would a memoryless guesser score at locked doors?

    Only meaningful under "keyed", where the outcome depends on a hidden
    variable. Under "open" and "trap" the outcome is fixed by the rules.
    """
    opened = shut = 0
    for s in seeds:
        w = GridWorld(s, rules=rules)
        _, doors = random_walk(w, n, random.Random(s), episode=episode,
                               trace=True)
        opened += sum(1 for e in doors if e == "unlocked")
        shut += sum(1 for e in doors if e != "unlocked")
    total = opened + shut
    maj = 100.0 * max(opened, shut) / total if total else 0.0
    if not quiet:
        print(f"  arrivals at a locked door: {total}")
        print(f"    opened : {opened:>5}")
        print(f"    blocked: {shut:>5}")
        print(f"    a memoryless guesser scores {maj:.1f}%")
    return maj


def contradiction_check(seeds=range(3), n=2000, quiet=False):
    """Do the rule sets actually demand different answers?

    Walks the SAME layout with the SAME actions under each rule set and
    counts how often the resulting event differs. If two worlds almost
    never disagree, they are the same task and retention cannot be
    measured — which is exactly the flaw that made the previous attempt
    meaningless.
    """
    rows = []
    for a in range(len(RULES)):
        for b in range(a + 1, len(RULES)):
            differ = total = 0
            for s in seeds:
                rng_a = random.Random(s)
                rng_b = random.Random(s)      # identical action sequence
                wa = GridWorld(s, rules=RULES[a])
                wb = GridWorld(s, rules=RULES[b])
                for i in range(n):
                    if i % 100 == 0:
                        wa.reset()
                        wb.reset()
                    act_a = rng_a.choice(ACTIONS)
                    act_b = rng_b.choice(ACTIONS)
                    _, ea = wa.step(act_a)
                    _, eb = wb.step(act_b)
                    total += 1
                    if ea != eb:
                        differ += 1
            pct = 100.0 * differ / total
            rows.append((RULES[a], RULES[b], pct))
            if not quiet:
                print(f"  {RULES[a]:>6} vs {RULES[b]:<6}: events differ on "
                      f"{pct:5.1f}% of steps")
    return min(r[2] for r in rows)


if __name__ == "__main__":
    print("the same layout under three rule sets:\n")
    for rules in RULES:
        w = GridWorld(0, rules=rules)
        print(w.render())
        print()

    print("""  @  agent    #  wall    .  floor    k  key    L  locked door

  keyed  a locked door opens only while carrying a key
  open   a locked door always opens
  trap   a locked door never opens, and keys block you

The observation does NOT reveal which rules are in force, so the agent has
to learn them from what happens — and a later world can overwrite them.
""")

    for rules in RULES:
        w = GridWorld(0, rules=rules)
        stream = random_walk(w, 3000, random.Random(0), episode=100)
        counts = {}
        for _, _, e in stream:
            counts[e] = counts.get(e, 0) + 1
        cells = "  ".join(f"{e} {100.0 * counts.get(e, 0) / len(stream):4.1f}%"
                          for e in EVENTS)
        print(f"  {rules:>6}: {cells}")

    print("\nbalance under 'keyed' (the only rule set with a hidden "
          "variable):")
    balance_check(range(5), rules="keyed")

    print("\ndo the rule sets genuinely contradict each other?")
    worst = contradiction_check(range(3))
    print()
    if worst < 5:
        print(f"  TOO SIMILAR ({worst:.1f}% at worst). Learning one world "
              f"is learning the others,\n  so nothing can be forgotten and "
              f"retention cannot be measured.")
    else:
        print(f"  GOOD ({worst:.1f}% at worst). The worlds demand different "
              f"answers,\n  so a later world can overwrite an earlier one "
              f"and retention is measurable.")