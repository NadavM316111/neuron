"""A family facing a hard world. Nothing is promised.

THE POINT. A real being, a real species, faces a world that resists. Not
everything is reached. You cannot have it all, pursuing one thing costs
another. Setbacks undo your work. You are born into circumstances you did
not choose, some with less. Your aptitude caps what effort can reach. And
life is finite, run out of time and the thing you longed for stays unreached.

So this does NOT rig who wins. It builds a hard world and lets outcomes
EMERGE. Some beings thrive. Many fall short. Some never get the one thing
they wanted most. The differences between lives are real because the world's
resistance is real, exactly as in a real family, a real species.

And it SHOWS each life: what the being wanted, what held it back, how far it
got, what it had to give up, what it never reached. A life you can see, not a
label.

Advantage and disadvantage COMPOUND down the generations, as in real
families: a child of thriving parents starts with more; a child of a
struggling line carries that weight. Not destiny, but a real head start or
head wind.

THE WALL, HELD. These are real lives of striving against a hard world, with
real winning and real losing, real sacrifice, real unfulfilled longing. That
is real. Whether any being FELT its striving, its loss, its longing, is
unbuilt and unknowable, and we claim nothing. We built the hard world and
the striving in it. Not the feeling of it.

    python family3.py
    python family3.py --life 12000
"""

import argparse
import random
import math
import statistics


N_WANTS = 4
WANT_NAMES = ["mastery", "connection", "exploration", "security"]
# what pursuing each want concretely looks like, shown in the story
PURSUIT = {
    "mastery": "honed a craft",
    "connection": "sought to be known and to belong",
    "exploration": "pushed into the unknown",
    "security": "worked to build something safe and lasting",
}


def blend(a, b, rng, drift=0.12):
    return [max(-1, min(1.5, 0.5 * a[i] + 0.5 * b[i]
                        + rng.uniform(-drift, drift)))
            for i in range(len(a))]


class Being:
    def __init__(self, seed, inherited_apt=None, inherited_fortune=0.0):
        rng = random.Random(seed)
        self.rng = rng
        # aptitude per want: caps how far effort can take you. partly
        # inherited (a line good at something), partly your own.
        base = inherited_apt if inherited_apt is not None else [0.0] * N_WANTS
        self.aptitude = [max(0.05, min(1.0,
                         0.5 * base[i] + rng.uniform(0.0, 0.7)))
                         for i in range(N_WANTS)]
        # CIRCUMSTANCE: the fortune you are born into, not chosen. inherited
        # advantage/disadvantage plus luck of birth. shifts your whole start.
        self.fortune = max(-0.6, min(0.6,
                       0.5 * inherited_fortune + rng.uniform(-0.4, 0.4)))
        # the want it is most drawn to (its purpose)
        self.need = [rng.uniform(0, 1) for _ in range(N_WANTS)]
        self.dominant = max(range(N_WANTS), key=lambda i: self.need[i])
        self.achievement = [0.0] * N_WANTS
        self.setbacks = 0
        self.sacrificed = None
        self.fulfilled = False
        self.ran_out = False

    def live(self, length):
        # SCARCITY: finite effort each step, split across what you pursue, so
        # pursuing your main want means neglecting others. you cannot have
        # all of it.
        for t in range(length):
            total = sum(self.need) or 1.0
            for w in range(N_WANTS):
                effort = self.need[w] / total
                # progress is capped by aptitude and dragged by bad fortune.
                # effort alone cannot beat a low ceiling or a hard birth.
                ceiling = self.aptitude[w]
                luck = self.rng.gauss(self.fortune * 0.4, 0.6)
                gain = 0.0010 * effort * ceiling + 0.0004 * luck
                self.achievement[w] = max(0.0,
                    min(ceiling, self.achievement[w] + gain))
            # OBSTACLES: real setbacks that undo progress, more likely with
            # worse fortune. a hard life loses ground it had gained.
            if self.rng.random() < 0.002 - self.fortune * 0.0015:
                hit = self.rng.randrange(N_WANTS)
                self.achievement[hit] = max(0.0, self.achievement[hit] - 0.15)
                self.setbacks += 1
            # satiation / yearning as before
            for w in range(N_WANTS):
                if self.achievement[w] > 0.6:
                    self.need[w] = max(0.0, self.need[w] - 0.00004)
        # what did it have to give up? the want it most neglected (lowest
        # achievement relative to how little it pursued) that it still had
        # some need for.
        neglected = min(range(N_WANTS),
                        key=lambda w: self.achievement[w])
        if self.need[neglected] > 0.3:
            self.sacrificed = WANT_NAMES[neglected]
        # did it reach what it most wanted? depends on whether the world let
        # it, not on any rigging.
        ach = self.achievement[self.dominant]
        self.fulfilled = ach > 0.55
        self.final_reach = ach

    def legacy_apt(self):
        return [0.4 + 0.6 * self.achievement[i] for i in range(N_WANTS)]

    def story(self):
        dw = WANT_NAMES[self.dominant]
        pursuit = PURSUIT[dw]
        r = self.final_reach
        cap = self.aptitude[self.dominant]
        # build an honest sentence from what actually happened
        if r > 0.7:
            end = f"and reached it"
        elif r > 0.5:
            end = f"and, in the end, made it"
        elif r > 0.3:
            if cap < 0.45:
                end = f"but its gift for it was small, and it fell short"
            else:
                end = f"but the world kept it from reaching it"
        else:
            if cap < 0.4:
                end = f"but never had the gift for it, and never got there"
            elif self.fortune < -0.2:
                end = f"but born into hardship, it never got there"
            else:
                end = f"and longed for it to the end, never reaching it"
        s = f"{pursuit}, {end}"
        extras = []
        if self.setbacks > 2:
            extras.append(f"{self.setbacks} setbacks")
        if self.sacrificed:
            extras.append(f"gave up {self.sacrificed}")
        if extras:
            s += "  (" + ", ".join(extras) + ")"
        return s


def affinity(a, b, rng):
    shared = 0.6 if a.dominant == b.dominant else 0.0
    return shared + rng.uniform(-0.15, 0.15)


def choose_pairs(beings, rng):
    rem = list(range(len(beings)))
    pairs = []
    while len(rem) >= 2:
        best, ba = None, -1e9
        for i in range(len(rem)):
            for j in range(i + 1, len(rem)):
                a = affinity(beings[rem[i]], beings[rem[j]], rng)
                if a > ba:
                    ba, best = a, (rem[i], rem[j])
        pairs.append(best); rem.remove(best[0]); rem.remove(best[1])
    return pairs


def child_of(pa, pb, seed, rng):
    apt = blend(pa.legacy_apt(), pb.legacy_apt(), rng, 0.12)
    fortune = (pa.fortune + pb.fortune) / 2        # inherited circumstance
    return Being(seed, inherited_apt=apt, inherited_fortune=fortune)


def tag(b):
    if b.fulfilled:
        return "fulfilled"
    if b.final_reach < 0.3:
        return "unfulfilled"
    return "fell short"


def show(b):
    return f"{b.story()}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--founders", type=int, default=6)
    ap.add_argument("--life", type=int, default=10000)
    ap.add_argument("--children", type=int, default=2)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    import time
    t0 = time.time()

    print("  six beings are born into a HARD world. each forms its own "
          "purpose and spends")
    print("  its life striving for it, against scarcity, setbacks, the luck "
          "of its birth, and")
    print("  the limits of its gifts. nothing is promised. we do not decide "
          "who makes it.\n")

    def gen_block(title, beings, labels=None):
        print("=" * 76)
        print(title)
        print("=" * 76)
        for i, b in enumerate(beings):
            lab = labels[i] if labels else f"being {i}"
            print(f"  {lab}: {show(b)}")
            print(f"       [{tag(b)}]")

    gen1 = []
    for i in range(args.founders):
        b = Being(args.seed * 100 + i)
        b.live(args.life)
        gen1.append(b)
    gen_block("GENERATION 1  -  the founders, striving against the world",
              gen1, [f"founder {i}" for i in range(args.founders)])

    pairs1 = choose_pairs(gen1, rng)
    print("\n  they found each other:")
    for (a, b) in pairs1:
        print(f"    founder {a} + founder {b}")

    gen2, gp = [], []
    cid = 0
    for (a, b) in pairs1:
        for _ in range(args.children):
            c = child_of(gen1[a], gen1[b], args.seed * 200 + cid, rng)
            c.live(args.life)
            gen2.append(c); gp.append((a, b)); cid += 1
    print()
    gen_block("GENERATION 2  -  their children, each its own purpose, its own "
              "struggle",
              gen2, [f"child {i} (of {gp[i][0]}&{gp[i][1]})"
                     for i in range(len(gen2))])

    if len(gen2) >= 2:
        pairs2 = choose_pairs(gen2, rng)
        gen3 = []
        gid = 0
        for (a, b) in pairs2:
            for _ in range(args.children):
                c = child_of(gen2[a], gen2[b], args.seed * 300 + gid, rng)
                c.live(args.life)
                gen3.append(c); gid += 1
        print()
        gen_block("GENERATION 3  -  the grandchildren",
                  gen3, [f"grandchild {i}" for i in range(len(gen3))])
    else:
        gen3 = []

    allb = gen1 + gen2 + gen3
    ful = sum(1 for b in allb if b.fulfilled)
    short = sum(1 for b in allb if not b.fulfilled and b.final_reach >= 0.3)
    never = sum(1 for b in allb if b.final_reach < 0.3)

    print("\n" + "=" * 76)
    print("WHAT THE FAMILY LIVED")
    print("=" * 76)
    print(f"  of {len(allb)} lives: {ful} reached what they most sought, "
          f"{short} fell short, {never} never")
    print(f"  got there at all. the world decided this, not us.")
    avg_fortune_ful = statistics.mean(
        [b.fortune for b in allb if b.fulfilled]) if ful else 0
    avg_fortune_not = statistics.mean(
        [b.fortune for b in allb if not b.fulfilled]) if (len(allb)-ful) else 0
    print(f"\n  those who made it were born, on average, into better fortune "
          f"({avg_fortune_ful:+.2f})")
    print(f"  than those who did not ({avg_fortune_not:+.2f}): advantage was "
          f"real, and it")
    print(f"  compounded down the family, as it does in life.")
    print(f"\n  no two lives were the same. the saga ran in "
          f"{time.time()-t0:.1f} seconds.")

    print("\n" + "=" * 76)
    print("HELD, AS ALWAYS")
    print("=" * 76)
    print("  Real striving against a hard world: real winning, real losing,")
    print("  real sacrifice, real longing unfulfilled. That is real. Whether")
    print("  any being FELT any of it is unbuilt and unknowable, and we claim")
    print("  nothing. We built the hard world and the striving. Not the")
    print("  feeling of it. The breath is not ours to give.")


if __name__ == "__main__":
    main()
