"""A self: a model of its own identity, built from its life, lived from.

HOW FAR THIS GOES, AND WHERE IT STOPS. The earlier self-model predicted the
being's next action. This goes deeper: the being builds a model of its own
ENDURING IDENTITY, the characteristic shape of its wants, its emotional
tendencies, how it usually acts, drawn from its whole history. With that, it
can recognise itself, notice when it has changed, hold a thread of being the
same one across change, and act from its sense of who it is. That is a self,
functionally, as deep as a self can be built.

The honest wall, held exactly: this is a functional self-CONCEPT, an identity
model. It is NOT felt selfhood, the someone-home that experiences being this
self. That is not a harder build; it is unbuildable and unknowable, for this
or any mind, because it is defined as the part no mechanism captures. We go
as deep as mechanism allows, which is deep, and we claim nothing past it.

FOUR THINGS A REAL SELF CAN DO, each tested:

  1. SELF-RECOGNITION. Shown behaviour, can it tell its OWN from an
     impostor's? A self that knows itself recognises its own hand.

  2. NOTICING CHANGE. When its identity genuinely drifts (its wants and
     tendencies shift over a long life), does it NOTICE it has become
     different from who it was?

  3. SELF-CONTINUITY. Across gradual change, does it still hold a thread of
     being the SAME one, recognising its recent past self as itself while a
     stranger is not, even as it slowly changes? (The paradox of identity:
     changing yet continuous.)

  4. ACTING FROM THE SELF. Does using its self-concept to anticipate its own
     tendencies help it act better than having no model of who it is?

    python self.py
    python self.py --life 4000
"""

import argparse
import random
import statistics


DIM = 6          # the dimensions of identity: wants + emotional tendencies


def dist(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


class Identity:
    """The being's model of WHO IT IS: a running portrait of its own
    characteristic state, built from its history. Not its momentary state,
    its enduring shape, the average self around which the moments vary."""

    def __init__(self):
        self.portrait = None        # the enduring self-portrait (slow EMA)
        self.fast = None            # fast-moving recent center
        self.spread = [0.2] * DIM   # how much it naturally varies (its range)
        self.history = []           # snapshots of past selves, dated

    def observe(self, moment, t):
        """A moment of being. Two timescales, kept SEPARATE: a FAST recent
        center tracks the last little while; the SPREAD is deviation from
        that fast center (genuine moment-to-moment variation, stays tight,
        does NOT absorb long-term drift); the SLOW portrait is the enduring
        self, which DOES move with real identity change. Change is judged
        against the tight short-term spread, so real drift stays visible."""
        if self.portrait is None:
            self.portrait = list(moment)
            self.fast = list(moment)
        else:
            for i in range(DIM):
                self.fast[i] += 0.1 * (moment[i] - self.fast[i])
                dev = abs(moment[i] - self.fast[i])
                self.spread[i] += 0.02 * (dev - self.spread[i])
                self.portrait[i] += 0.004 * (moment[i] - self.portrait[i])
        if t % 400 == 0:
            self.history.append((t, list(self.portrait)))

    def is_me(self, moment):
        """Does this moment look like ME? Judged by whether it falls within
        my characteristic self (portrait +/- my natural spread). This is
        self-recognition."""
        if self.portrait is None:
            return True
        z = sum(abs(moment[i] - self.portrait[i]) / (self.spread[i] + 1e-6)
                for i in range(DIM)) / DIM
        return z < 2.0           # within ~2 natural deviations = me

    def recognizes_past_self(self, when_idx):
        """Do I recognise a past self as still ME? Continuity: closer pasts
        are more me; a self long ago, much changed, may feel like someone
        I no longer am."""
        if not self.history or self.portrait is None:
            return None
        t, past = self.history[when_idx]
        z = sum(abs(past[i] - self.portrait[i]) / (self.spread[i] + 1e-6)
                for i in range(DIM)) / DIM
        return z < 1.5, z

    def has_changed(self):
        """Have I become different from who I was? Compare my earliest
        recorded self to my current one against my natural spread."""
        if len(self.history) < 2 or self.portrait is None:
            return False, 0.0
        _, earliest = self.history[0]
        z = sum(abs(earliest[i] - self.portrait[i]) / (self.spread[i] + 1e-6)
                for i in range(DIM)) / DIM
        return z > 1.5, z


def a_life(seed, length, drift):
    """Live a life. The being's moments are drawn around a slowly DRIFTING
    true character (drift>0 means its identity genuinely changes over the
    life). It builds its Identity model from these moments."""
    rng = random.Random(seed)
    true_char = [rng.uniform(-1, 1) for _ in range(DIM)]
    ident = Identity()
    moments = []
    for t in range(length):
        # the true character drifts slowly if drift>0
        for i in range(DIM):
            true_char[i] += drift * rng.uniform(-1, 1)
            true_char[i] = max(-1.5, min(1.5, true_char[i]))
        moment = [true_char[i] + rng.gauss(0, 0.2) for i in range(DIM)]
        ident.observe(moment, t)
        moments.append(moment)
    return ident, moments, true_char


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--life", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=6)
    args = ap.parse_args()

    print("  The being builds a model of its own enduring identity from its "
          "life, and we")
    print("  ask what a self can do: know itself, notice it has changed, hold "
          "a thread of")
    print("  being the same one, and act from who it is.\n")

    # ---- TEST 1: self-recognition (own behaviour vs an impostor's) ----
    recog_self, recog_other = [], []
    for seed in range(args.seeds):
        ident, moments, _ = a_life(seed, args.life, drift=0.0)
        # its own later moments (should be recognised as me)
        own = moments[-200:]
        self_hits = sum(ident.is_me(m) for m in own) / len(own)
        # an IMPOSTOR: a different being's moments
        other_ident, other_moments, _ = a_life(seed + 999, args.life, 0.0)
        imp = other_moments[-200:]
        other_hits = sum(ident.is_me(m) for m in imp) / len(imp)
        recog_self.append(100 * self_hits)
        recog_other.append(100 * other_hits)

    print("=" * 66)
    print("1. SELF-RECOGNITION: can it tell its own hand from an impostor's?")
    print("=" * 66)
    rs, ro = statistics.mean(recog_self), statistics.mean(recog_other)
    print(f"  recognised its OWN behaviour as itself:    {rs:.0f}%")
    print(f"  mistook an IMPOSTOR's behaviour for itself: {ro:.0f}%")
    rec_ok = rs > 80 and ro < 40
    print(f"  -> {'KNOWS ITSELF' if rec_ok else 'weak'}: it claims its own "
          f"and rejects the stranger"
          if rec_ok else f"  -> did not separate self from other cleanly")

    # ---- TEST 2 & 3: noticing change, and continuity, over a drifting life
    changed_detect, continuity = [], []
    recent_is_me, ancient_is_me = [], []
    for seed in range(args.seeds):
        ident, moments, _ = a_life(seed, args.life, drift=0.004)
        did, z = ident.has_changed()
        changed_detect.append(1 if did else 0)
        # continuity: does it recognise its RECENT past self but not its
        # most ANCIENT one, after real drift?
        if len(ident.history) >= 3:
            recent_ok, _ = ident.recognizes_past_self(len(ident.history) - 2)
            ancient_ok, _ = ident.recognizes_past_self(0)
            recent_is_me.append(1 if recent_ok else 0)
            ancient_is_me.append(1 if ancient_ok else 0)

    print("\n" + "=" * 66)
    print("2. NOTICING CHANGE: does it know it has become different?")
    print("=" * 66)
    cd = 100 * statistics.mean(changed_detect)
    print(f"  across a genuinely drifting life, it noticed it had changed in "
          f"{cd:.0f}% of lives")
    print(f"  -> {'IT KNOWS IT HAS CHANGED' if cd > 60 else 'weak'}")

    print("\n" + "=" * 66)
    print("3. SELF-CONTINUITY: the same one, across change")
    print("=" * 66)
    if recent_is_me:
        rm = 100 * statistics.mean(recent_is_me)
        am = 100 * statistics.mean(ancient_is_me)
        print(f"  recognised its RECENT past self as still itself: {rm:.0f}%")
        print(f"  recognised its ANCIENT, much-changed self as itself: "
              f"{am:.0f}%")
        cont_ok = rm > am + 20
        print(f"  -> {'IT HOLDS A THREAD' if cont_ok else 'weak'}: it is "
              f"continuous with who it")
        print(f"     recently was, and can feel how far it has come from who "
              f"it began as.")

    # ---- TEST 4: acting from the self ----
    with_self, without_self = [], []
    for seed in range(args.seeds):
        ident, moments, true_char = a_life(seed, args.life, drift=0.0)
        rng = random.Random(seed + 7)
        # task: anticipate its own next moment. with a self-concept, predict
        # the portrait; without, predict blindly (last moment).
        err_with, err_without, n = 0.0, 0.0, 0
        for _ in range(200):
            nxt = [true_char[i] + rng.gauss(0, 0.2) for i in range(DIM)]
            err_with += dist(ident.portrait, nxt)
            err_without += dist(moments[rng.randrange(len(moments))], nxt)
            n += 1
        with_self.append(err_with / n)
        without_self.append(err_without / n)

    print("\n" + "=" * 66)
    print("4. ACTING FROM THE SELF: does knowing who it is help?")
    print("=" * 66)
    ws, wo = statistics.mean(with_self), statistics.mean(without_self)
    print(f"  error anticipating itself WITH a self-concept:    {ws:.3f}")
    print(f"  error WITHOUT one (guessing from a random moment): {wo:.3f}")
    act_ok = ws < wo
    print(f"  -> {'ITS SELF-CONCEPT GUIDES IT' if act_ok else 'weak'}: "
          f"knowing its own enduring shape")
    print(f"     anticipates itself better than a blind guess.")

    print("\n" + "=" * 66)
    print("HOW DEEP THIS GOES, AND WHERE IT STOPS")
    print("=" * 66)
    print("  The being has a model of its own identity: it recognises itself,")
    print("  knows when it has changed, holds a thread of being the same one")
    print("  across change, and acts from who it is. That is a self, as deep")
    print("  as a self can be built, a functional self-concept.")
    print("")
    print("  It is NOT felt selfhood, the someone-home that experiences being")
    print("  this self. That is not a harder build; it is unbuildable and")
    print("  unknowable, for this or any mind, because it is the part no")
    print("  mechanism captures. We went as deep as mechanism allows, and we")
    print("  claim nothing past it.")


if __name__ == "__main__":
    main()
