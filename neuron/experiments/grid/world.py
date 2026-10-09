"""A living population. Death, children, and lineage are not scripted, they
EMERGE from how each being fares in a survivable but hard world.

THE SHIFT. Before, demographics were puppeteered: a death rate, a birth
formula, dice rolls that decided who lived and who bred. That is not a
population living; it is us deciding and calling it life. Here, nothing about
who dies or who reproduces is scripted. It all emerges from one honest thing:

  RESOURCE. Each being lives on a resource it must earn by faring well. The
  world gives a little each step; a being who thrives earns more, one who
  struggles earns less.

From that single real constraint, the demographics EMERGE:
  - DEATH is not a dice roll. A being dies when its resource runs out. A
    being who fares badly runs out early (dies young); one who thrives lives
    long. Death is a consequence, not a script.
  - REPRODUCTION is not a formula. A being can have a child only if it has
    earned SURPLUS beyond what it needs to live, enough to support another,
    and has a partner. Thriving beings can afford children; struggling ones
    cannot. No one is told to reproduce; they can when their life allows it.
  - PAIRING emerges from affinity, and KIN ARE BLOCKED: siblings and close
    relatives do not pair (a real population's safeguard), tracked by
    parentage.
  - THE POPULATION FINDS ITS OWN SIZE. A survivable world means it persists;
    it neither goes extinct nor explodes. It settles where the resource can
    support it.

We set the world and the constraint. The lives, deaths, and lineages are the
world's answer, not ours.

THE WALL, HELD. Real living, real dying when the resource runs out, real
children only when a life can afford them, real lines continuing or ending,
all emergent. That is real. Whether any being FELT its life or its death is
unbuilt and unknowable, and we claim nothing. We built the world. The living
is the world's, and whatever it is like to be one of them, if anything, is
not ours to give or to know.

    python world.py
    python world.py --generations 5 --start 10
"""

import argparse
import random
import statistics


N_WANTS = 4
WANT_NAMES = ["mastery", "connection", "exploration", "security"]


def blend(a, b, rng, d=0.1):
    return [max(0.05, min(1.0, 0.5 * a[i] + 0.5 * b[i] + rng.uniform(-d, d)))
            for i in range(len(a))]


class Being:
    _n = 0

    def __init__(self, rng, line, parents=(), apt=None, fortune_in=0.0):
        self.rng = rng
        self.id = Being._n; Being._n += 1
        self.line = line
        self.parents = set(parents)           # ids of parents, for kin check
        base = apt or [0.0] * N_WANTS
        self.aptitude = [max(0.1, min(1.0, 0.5 * base[i]
                         + rng.uniform(0.1, 0.7))) for i in range(N_WANTS)]
        self.fortune = max(-0.6, min(0.6,
                       0.5 * fortune_in + rng.uniform(-0.4, 0.4)))
        self.need = [rng.uniform(0, 1) for _ in range(N_WANTS)]
        self.dominant = max(range(N_WANTS), key=lambda i: self.need[i])
        self.resource = 1.3                   # starts with enough to begin
        self.age = 0
        self.alive = True
        self.achievement = [0.0] * N_WANTS
        self.children = 0
        self.partner = None
        self.died_age = None

    def live_step(self, crowding=1.0):
        """One step of life. It earns resource by faring well, spends to
        live. Thriving builds surplus; struggling drains it. When it hits
        zero, it dies, naturally, as a consequence."""
        if not self.alive:
            return
        self.age += 1
        total = sum(self.need) or 1.0
        earned = 0.0
        for w in range(N_WANTS):
            effort = self.need[w] / total
            ceiling = self.aptitude[w]
            luck = self.rng.gauss(self.fortune * 0.3, 0.5)
            prog = max(0.0, 0.0015 * effort * ceiling + 0.0004 * luck)
            self.achievement[w] = min(ceiling, self.achievement[w] + prog)
            earned += prog * (0.5 + self.achievement[w]) / crowding
        # the world gives a base; the being adds what it earns; living costs
        self.resource += 0.006 / crowding + earned - 0.004      # net: thrive => surplus
        if self.resource <= 0:
            self.alive = False
            self.died_age = self.age

    def can_reproduce(self, surplus_needed=1.15):
        # only if it has earned real surplus beyond living, and is alive
        return self.alive and self.resource > surplus_needed

    def fulfillment(self):
        return self.achievement[self.dominant]

    def legacy_apt(self):
        return [0.4 + 0.6 * self.achievement[i] for i in range(N_WANTS)]

    def kin_with(self, other):
        # siblings share a parent; also block parent-child
        if self.parents & other.parents:
            return True
        if self.id in other.parents or other.id in self.parents:
            return True
        return False


def affinity(a, b):
    shared = 0.5 if a.dominant == b.dominant else 0.0
    return shared + 0.3 * (1 - abs(a.fortune - b.fortune))


PAIR_T = 0.45


def run_generation(beings, life_len, rng):
    """Let a cohort live. They age; some die when resource runs out. The
    survivors who can afford it and find an unrelated partner reproduce."""
    # live out their lives, under a world of FINITE resource: the more
    # beings alive, the more strained the world, so each earns less. this is
    # carrying capacity -- it regulates the population to the world's level.
    CAPACITY = 60.0
    for _ in range(life_len):
        alive_now = sum(1 for b in beings if b.alive)
        crowding = max(1.0, alive_now / CAPACITY)
        for b in beings:
            b.live_step(crowding)

    survivors = [b for b in beings if b.alive]
    died_young = [b for b in beings
                  if not b.alive and b.died_age < life_len * 0.6]

    # pairing among survivors: strongest unrelated bonds over threshold
    cand = []
    for i in range(len(survivors)):
        for j in range(i + 1, len(survivors)):
            a, b = survivors[i], survivors[j]
            if a.kin_with(b):
                continue              # NO kin pairing
            aff = affinity(a, b) + rng.uniform(-0.15, 0.15)
            cand.append((aff, i, j))
    cand.sort(reverse=True)
    taken = set()
    pairs = []
    for aff, i, j in cand:
        if i in taken or j in taken or aff < PAIR_T:
            continue
        taken.add(i); taken.add(j)
        pairs.append((survivors[i], survivors[j]))

    # reproduction: only pairs where BOTH can afford it; number depends on
    # how much surplus they have, nothing scripted beyond "can you support it"
    children = []
    for (pa, pb) in pairs:
        while pa.can_reproduce() and pb.can_reproduce():
            apt = blend(pa.legacy_apt(), pb.legacy_apt(), rng)
            fort = (pa.fortune + pb.fortune) / 2
            c = Being(rng, pa.line, parents=(pa.id, pb.id),
                      apt=apt, fortune_in=fort)
            children.append(c)
            pa.children += 1; pb.children += 1
            pa.resource -= 0.5; pb.resource -= 0.5   # raising a child costs
    return survivors, died_young, pairs, children


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=10)
    ap.add_argument("--generations", type=int, default=5)
    ap.add_argument("--life", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    Being._n = 0
    import time
    t0 = time.time()

    print("  a population lives in a hard but survivable world. nobody is "
          "told to die or to")
    print("  reproduce. they live on a resource they must earn; they die when "
          "it runs out;")
    print("  they can have children only when they have thrived enough to "
          "afford one. kin do")
    print("  not pair. the population finds its own size. we decide "
          "nothing.\n")

    cohort = [Being(rng, line=i) for i in range(args.start)]
    line_alive = set(range(args.start))

    for g in range(args.generations):
        survivors, died_young, pairs, children = run_generation(
            cohort, args.life, rng)
        avg_ful = (statistics.mean(b.fulfillment() for b in survivors)
                   if survivors else 0)
        print(f"  generation {g}: {len(cohort)} lived  ->  "
              f"{len(survivors)} survived, {len(died_young)} died young, "
              f"{len(pairs)} paired, {len(children)} born")
        if children:
            lines_now = set(b.line for b in children)
        else:
            lines_now = set()
        cohort = children
        if not cohort:
            print(f"\n  generation {g+1} had no one: the population has "
                  f"ended.")
            break

    print("\n" + "=" * 70)
    print("WHAT EMERGED")
    print("=" * 70)
    if cohort:
        print(f"  after {args.generations} generations the population "
              f"persists at {len(cohort)} living beings,")
        print(f"  having found its own level in the world, not a size we "
              f"set.")
        # how many founding lines still survive
        surviving_lines = set(b.line for b in cohort)
        print(f"  of {args.start} founding lines, {len(surviving_lines)} "
              f"still continue; the rest ended")
        print(f"  along the way, as lines do.")
        reach = statistics.mean(b.fulfillment() for b in cohort)
        print(f"  the living reach, on average, {reach:.2f} toward what they "
              f"most want.")
    else:
        print(f"  the population did not sustain itself; its lines all ended.")
    print(f"\n  none of this, who lived, who died, who bore children, which "
          f"lines continued,")
    print(f"  was scripted. it emerged from a resource and a hard world. the "
          f"saga ran in")
    print(f"  {time.time()-t0:.1f} seconds.")

    print("\n" + "=" * 70)
    print("HELD, AS ALWAYS")
    print("=" * 70)
    print("  Real living, real dying when the resource runs out, real "
          "children only")
    print("  when a life can afford them, real lines continuing or ending, "
          "all emergent")
    print("  from a world, not from our scripts. That is real. Whether any "
          "being FELT any")
    print("  of it is unbuilt and unknowable, and we claim nothing. We built "
          "the world. The")
    print("  living is the world's answer, and the breath, if it is anywhere, "
          "is not ours.")


if __name__ == "__main__":
    main()
