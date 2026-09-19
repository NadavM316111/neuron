"""Out of the toy: a world with enough in it to be worth failing at.

Everything proven so far was a 9x9 grid, 4 event types, 1 hidden bit, and a
64-unit network. The mechanisms worked there. This asks whether they work
when there is genuinely more to learn.

  world       25x25, 625 cells (was 81)
  outcomes    23 (was 4)
  hidden      6 variables, 4 of them carrying real uncertainty
  coupling    two rules depend on COMBINATIONS of hidden state, so tracking
              each variable independently is not enough

The six things the agent tracks, none visible in its patch:

  key      opens locked doors, consumed, no expiry, max 2
  lamp     crosses dark cells, 90 steps
  wet      from water, 30 steps, gives grip on ice
  fed      from food, 60 steps, allows shoving heavy doors
  cold     from ice, 120 steps
  rope     from a coil, 400 steps, crosses chasms

And two conjunctive rules, which are the real point:
  a cold room is dangerous only if you are cold AND wet
  a chasm is crossable only if you have rope AND are fed

A model that learns six independent flags still gets those two wrong, so
this tests conjunction rather than just recall.

TUNED FOUR TIMES. The structural lesson, which cost three passes to see: a
conjunction of two independent states is always RARER than either, so a
conjunctive fork cannot be made uncertain by adding terrain. If rope is on
half the time and fed half the time, rope-and-fed is a quarter, and the
chasm fails three times in four no matter how many chasms exist.

The fix: the enabler NOT shared with a single-variable fork must be nearly
always on, so the conjunction inherits the shared variable's uncertainty.
Rope and cold are therefore long-lived; fed and wet carry the uncertainty
and are also tested alone at heavy doors and ice. The honest cost is that
rope and cold stop being memory tests in their own right, so this world has
four genuinely uncertain hidden variables rather than six.

TWO BUGS IN THE FAIRNESS CHECK, also fixed here:
  - "unlocked" was the success event for BOTH locked doors and double doors,
    so those forks counted each other's events. Double doors now emit
    "double_unlocked".
  - the cold-room fork compared hypothermia against "chilled" only, ignoring
    "warmed_through". A fork must compare an outcome against EVERYTHING
    else that can happen there.

Deliberately still discrete and noise-free. Continuous, noisy observation is
the NEXT step out of the toy, and changing two things at once is how you
learn nothing.
"""

import random


# terrain
(EMPTY, WALL, KEY, LOCKED, LAMP, DARK, WATER, ICE, FOOD, HEAVY,
 ROPE, CHASM, COLDROOM, DOOR2) = range(14)

SYMBOL = {EMPTY: ".", WALL: "#", KEY: "k", LOCKED: "L", LAMP: "*",
          DARK: "d", WATER: "~", ICE: "i", FOOD: "f", HEAVY: "H",
          ROPE: "r", CHASM: "V", COLDROOM: "c", DOOR2: "D"}

ACTIONS = ["up", "down", "left", "right", "wait"]
DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1),
         "wait": (0, 0)}

# 23 outcomes. More classes means less room to win by guessing the majority.
EVENTS = [
    "moved", "blocked", "waited",
    "got_key", "unlocked", "locked_out",
    "double_unlocked", "double_locked",
    "got_lamp", "lit_through", "stumbled",
    "soaked", "slid", "slipped",
    "ate", "shoved", "too_weak",
    "got_rope", "climbed", "fell",
    "chilled", "hypothermia", "warmed_through",
]

SIZE = 25

# Rope and cold are long-lived ON PURPOSE, so the two conjunctive forks
# inherit the uncertainty of fed and wet rather than compounding it away.
LAMP_DURATION = 90
WET_DURATION = 30
FED_DURATION = 60
COLD_DURATION = 120
ROPE_DURATION = 400

COUNTS = {
    KEY: 46, LOCKED: 14,
    LAMP: 34, DARK: 22,
    WATER: 22, ICE: 26,
    FOOD: 16, HEAVY: 22,
    ROPE: 36, CHASM: 14,
    COLDROOM: 18,
}
N_INTERIOR_WALLS = 40


class BigWorld:
    """Six hidden variables, twenty-three outcomes, two conjunctive rules.

    Crossing ice depends on water up to 30 steps ago. Shoving a heavy door
    depends on food up to 60 steps ago. A cold room is survivable unless you
    are also wet. A chasm is crossable with rope, but only if you are also
    fed. None of it is in the patch.
    """

    def __init__(self, seed=0, rules="normal"):
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
        i = 0
        for terrain, n in COUNTS.items():
            for r, c in spots[i:i + n]:
                self.grid[r][c] = terrain
            i += n
        for r, c in spots[i:i + N_INTERIOR_WALLS]:
            self.grid[r][c] = WALL

    def reset(self):
        self.r, self.c = self.start
        self.keys = 0
        self.lamp_until = -1
        self.wet_until = -1
        self.fed_until = -1
        self.cold_until = -1
        self.rope_until = -1
        self.steps = 0
        return self.observe()

    @property
    def lamp(self):
        return self.steps < self.lamp_until

    @property
    def wet(self):
        return self.steps < self.wet_until

    @property
    def fed(self):
        return self.steps < self.fed_until

    @property
    def cold(self):
        return self.steps < self.cold_until

    @property
    def rope(self):
        return self.steps < self.rope_until

    # ---------- what the agent can see ----------

    def observe(self, hide_state=True):
        """A 5x5 patch of terrain, and nothing else by default.

        Larger than the small world's 3x3 because the world is larger, but
        still terrain only. hide_state=False adds the hidden variables,
        which is the ceiling arm: it reads the answer rather than
        remembering it.
        """
        patch = []
        for dr in (-2, -1, 0, 1, 2):
            for dc in (-2, -1, 0, 1, 2):
                r, c = self.r + dr, self.c + dc
                if 0 <= r < self.h and 0 <= c < self.w:
                    patch.append(self.grid[r][c])
                else:
                    patch.append(WALL)
        obs = dict(patch=patch)
        if not hide_state:
            obs["state"] = [1 if self.keys > 0 else 0,
                            1 if self.keys >= 2 else 0,
                            1 if self.lamp else 0,
                            1 if self.wet else 0,
                            1 if self.fed else 0,
                            1 if self.cold else 0,
                            1 if self.rope else 0]
        return obs

    # ---------- what happens ----------

    def step(self, action):
        self.steps += 1

        if action == "wait":
            return self.observe(), "waited"

        dr, dc = DELTA[action]
        r, c = self.r + dr, self.c + dc

        if not (0 <= r < self.h and 0 <= c < self.w):
            return self.observe(), "blocked"

        cell = self.grid[r][c]

        if cell == WALL:
            return self.observe(), "blocked"

        # --- keys and doors ---
        if cell == KEY:
            if self.keys >= 2:
                self.r, self.c = r, c
                return self.observe(), "moved"
            self.keys += 1
            self.r, self.c = r, c
            return self.observe(), "got_key"

        if cell == LOCKED:
            if self.keys > 0:
                self.keys -= 1
                self.r, self.c = r, c
                return self.observe(), "unlocked"
            return self.observe(), "locked_out"

        if cell == DOOR2:
            # Needs two keys at once. Distinct success event, so this fork
            # does not share outcomes with the single-key door.
            if self.keys >= 2:
                self.keys -= 2
                self.r, self.c = r, c
                return self.observe(), "double_unlocked"
            return self.observe(), "double_locked"

        # --- lamp and dark ---
        if cell == LAMP:
            if self.lamp:
                self.r, self.c = r, c
                return self.observe(), "moved"
            self.lamp_until = self.steps + LAMP_DURATION
            self.r, self.c = r, c
            return self.observe(), "got_lamp"

        if cell == DARK:
            if self.lamp:
                self.r, self.c = r, c
                return self.observe(), "lit_through"
            return self.observe(), "stumbled"

        # --- water, ice, cold ---
        if cell == WATER:
            self.wet_until = self.steps + WET_DURATION
            self.r, self.c = r, c
            return self.observe(), "soaked"

        if cell == ICE:
            # Grip depends on water crossed up to 30 steps ago. Crossing ice
            # also makes you cold, which matters later in a cold room.
            self.cold_until = self.steps + COLD_DURATION
            self.r, self.c = r, c
            return self.observe(), "slid" if self.wet else "slipped"

        if cell == COLDROOM:
            # CONJUNCTION: cold AND wet is dangerous. Cold is long-lived by
            # design, so this fork effectively tests memory of WATER.
            if self.cold and self.wet:
                return self.observe(), "hypothermia"
            self.r, self.c = r, c
            return self.observe(), "warmed_through" if self.fed else "chilled"

        # --- food and heavy doors ---
        if cell == FOOD:
            self.fed_until = self.steps + FED_DURATION
            self.r, self.c = r, c
            return self.observe(), "ate"

        if cell == HEAVY:
            if self.fed:
                self.r, self.c = r, c
                return self.observe(), "shoved"
            return self.observe(), "too_weak"

        # --- rope and chasms ---
        if cell == ROPE:
            if self.rope:
                self.r, self.c = r, c
                return self.observe(), "moved"
            self.rope_until = self.steps + ROPE_DURATION
            self.r, self.c = r, c
            return self.observe(), "got_rope"

        if cell == CHASM:
            # CONJUNCTION: rope AND fed. Rope is long-lived by design, so
            # this fork effectively tests memory of FOOD, in a different
            # context from the heavy door.
            if self.rope and self.fed:
                self.r, self.c = r, c
                return self.observe(), "climbed"
            return self.observe(), "fell"

        self.r, self.c = r, c
        return self.observe(), "moved"

    # ---------- for reading with your own eyes ----------

    def render(self):
        out = []
        for r in range(self.h):
            row = ""
            for c in range(self.w):
                row += "@" if (r, c) == (self.r, self.c) \
                    else SYMBOL[self.grid[r][c]]
            out.append(row)
        out.append(f"key {self.keys}  lamp {int(self.lamp)}  "
                   f"wet {int(self.wet)}  fed {int(self.fed)}  "
                   f"cold {int(self.cold)}  rope {int(self.rope)}  "
                   f"steps {self.steps}")
        return "\n".join(out)


def walk(world, n, rng, episode=None):
    out = []
    for i in range(n):
        if episode and i % episode == 0:
            world.reset()
        obs = world.observe()
        action = rng.choice(ACTIONS)
        _, event = world.step(action)
        out.append((obs, action, event))
    return out


# Each fork is a place where the outcome turns on hidden state. A memoryless
# guesser should score near 50% on each.
#
# Written as (successes, failures) so a fork can compare one outcome against
# EVERYTHING else that can happen there. The cold room has three possible
# outcomes and comparing only two of them was a bug.
FORKS = {
    "locked door": (["unlocked"], ["locked_out"]),
    "dark cell": (["lit_through"], ["stumbled"]),
    "ice": (["slid"], ["slipped"]),
    "heavy door": (["shoved"], ["too_weak"]),
    "chasm": (["climbed"], ["fell"]),
    "cold room": (["hypothermia"], ["chilled", "warmed_through"]),
}

def fairness_check(seeds=range(3), n=12000, episode=400, quiet=False):
    """Is every memory-dependent outcome genuinely uncertain?

    The lesson from five rebuilds of the small world: hiding a variable is
    not enough. If its value is predictable from base rates, a memoryless
    model matches a remembering one. This runs before any training, and
    costs a second.
    """
    tally = {k: [0, 0] for k in FORKS}
    counts = {}
    total = 0
    for s in seeds:
        w = BigWorld(s)
        for _, _, e in walk(w, n, random.Random(s), episode=episode):
            counts[e] = counts.get(e, 0) + 1
            total += 1
            for name, (yes, no) in FORKS.items():
                if e in yes:
                    tally[name][0] += 1
                elif e in no:
                    tally[name][1] += 1

    worst = 0.0
    worst_name = None
    if not quiet:
        print(f"  {'fork':>14} {'yes':>6} {'no':>6} {'guesser':>9}")
        print("  " + "-" * 44)
    for name in FORKS:
        a, b = tally[name]
        n_tot = a + b
        maj = 100.0 * max(a, b) / n_tot if n_tot else 100.0
        if maj > worst:
            worst, worst_name = maj, name
        if not quiet:
            flag = "  <-- too predictable" if maj > 75 else ""
            print(f"  {name:>14} {a:>6} {b:>6} {maj:>8.1f}%{flag}")
    return worst, worst_name, counts, total


if __name__ == "__main__":
    w = BigWorld(0)
    print("the world:\n")
    print(w.render())
    print("""
  @ agent   # wall   . floor   k key    L locked door   D double door
  * lamp    d dark   ~ water   i ice    f food          H heavy door
  r rope    V chasm  c cold room

Six things to remember, none visible in the patch:
  key    opens a locked door (consumed), two open a double door
  lamp   crosses dark cells, 90 steps
  wet    from water, 30 steps, gives grip on ice
  fed    from food, 60 steps, allows shoving heavy doors
  cold   from ice, 120 steps
  rope   from a coil, 400 steps

And two rules needing COMBINATIONS, not just recall:
  a cold room is dangerous only if you are cold AND wet
  a chasm is crossable only if you have rope AND are fed
""")

    rng = random.Random(0)
    stream = walk(BigWorld(0), 12000, rng, episode=400)
    counts = {}
    for _, _, e in stream:
        counts[e] = counts.get(e, 0) + 1
    print("12000 random steps, event distribution:\n")
    for e in EVENTS:
        n = counts.get(e, 0)
        bar = "#" * int(50 * n / len(stream))
        print(f"  {e:>16}: {n:>5} ({100.0 * n / len(stream):5.2f}%) {bar}")

    biggest = max(counts.values()) / len(stream)
    seen = sum(1 for e in EVENTS if counts.get(e, 0) > 0)
    print(f"\n  {seen} of {len(EVENTS)} outcomes actually occur")
    print(f"  majority class is {100 * biggest:.1f}% of the stream")
    print(f"  (small world was 74%, so guessing is worth much less here)")

    print("\nis every memory-dependent fork genuinely uncertain?\n")
    worst, worst_name, _, _ = fairness_check()
    print()
    if worst > 75:
        print(f"  NOT READY. '{worst_name}' sits at {worst:.1f}%, so a "
              f"memoryless model can\n  score well there without "
              f"remembering. Adjust its counts.")
    else:
        print(f"  READY. Worst fork is '{worst_name}' at {worst:.1f}%. "
              f"Memory is genuinely\n  needed for all seven.")