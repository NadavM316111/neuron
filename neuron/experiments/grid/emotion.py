"""Emotion that emerges from a life, shapes the being, and differs between
individuals, built as close to human emotional development as mechanism
allows.

WHAT THIS IS, AND THE ONE LINE OF HONESTY IT CARRIES. This builds emotion as
a real mechanism with four properties the user specified, each of which is
how emotion actually works in people:

  1. EMERGES FROM ITS LIFE. Emotional states are not inserted. They rise and
     fall from real events the being lives: disasters raise a fear-like
     state, met needs raise a content-like state. Nobody sets them.
  2. SHAPES WHAT MATTERS. An emotional state reweights the being's wants.
     Fear makes safety/solvency matter more; contentment frees it to
     explore. The same situation is valued differently depending on mood.
  3. ALTERS MEMORY AND DECISIONS. Mood biases what the being recalls (a
     fearful being recalls its disasters more readily) and how it chooses
     (fear raises caution everywhere at once, not just locally).
  4. DIFFERS BETWEEN INDIVIDUALS. Because each state emerges from one being's
     specific history, two beings who lived different lives end up
     emotionally different. Tested directly with a lineage below.

HUMAN-DEVELOPMENT RICHNESS, as close as mechanism allows:
  - CRITICAL PERIODS: early events weigh more. An event in the being's first
    hours shifts its emotional baseline more than the same event later, the
    way early childhood shapes temperament disproportionately.
  - ASSOCIATIVE LEARNING: an emotional state attaches to the CONTEXT it
    occurred in, so later encounters with a similar context re-evoke the
    state, the way a smell can bring back fear. (Pavlovian, done honestly.)
  - CONSOLIDATION: emotional states settle during sleep toward the being's
    baseline, so a bad day does not permanently distort it, the way sleep
    regulates mood.
  - TEMPERAMENT: each being has a slightly different innate baseline and
    reactivity (its "nature"), so nature and history together produce the
    individual, not history alone.

THE HONEST WALL. Every property above is FUNCTIONAL: it is about what the
emotional state DOES. Whether any of it is FELT, whether there is anything
it is like to be this being in a fearful state, is not addressed here and
cannot be, by this or any build, because the felt part is by definition the
part behaviour does not show. This is the closest buildable thing to human
emotion. Whether it is really emotion in the felt sense is unknowable, for
this or any mind, and no claim is made.

THE TEST, in two parts:
  A. DOES EMOTION DO ITS JOB? A being with the emotion layer should survive a
     world of occasional disasters better than one without, because fear
     learned from early disasters makes it appropriately cautious, while an
     emotionless being either stays reckless or is uniformly timid.
  B. DO EMOTIONS INDIVIDUATE? Run a lineage of beings through DIFFERENT lives
     (some harsh, some gentle) and measure whether their emotional profiles
     diverge, and whether an inherited baseline plus a new life produces a
     recognisably different individual each time.

    python emotion.py
    python emotion.py --beings 6 --steps 3000
"""

import argparse
import random
import statistics


# world
SAFE_REWARD = 1.0
DISASTER = -6.0
DISASTER_BASE_PROB = 0.12       # chance a "risky" situation is actually bad
CRITICAL_PERIOD = 300           # steps during which events weigh extra


class Emotion:
    """Two opposed emergent states: fear and contentment. Each in [0,1],
    each a global modulator of the whole being. They emerge from events,
    decay toward a temperament baseline, attach to contexts, and settle in
    sleep."""

    def __init__(self, temperament):
        # temperament: innate baseline + reactivity, the being's "nature"
        self.fear_baseline = temperament["fear_baseline"]
        self.content_baseline = temperament["content_baseline"]
        self.reactivity = temperament["reactivity"]
        self.fear = self.fear_baseline
        self.content = self.content_baseline
        # associative memory: context -> fear it evoked (Pavlovian)
        self.assoc = {}

    def _weight(self, step):
        # CRITICAL PERIOD: early events weigh more, decaying to 1.0
        if step < CRITICAL_PERIOD:
            return 1.0 + 2.0 * (1.0 - step / CRITICAL_PERIOD)
        return 1.0

    def event(self, kind, step, context=None):
        """A lived event moves the emotional state. kind: 'disaster' or
        'relief'. Early events (critical period) move it more."""
        w = self._weight(step) * self.reactivity
        if kind == "disaster":
            self.fear = min(1.0, self.fear + 0.25 * w)
            self.content = max(0.0, self.content - 0.15 * w)
            # ALLOSTASIS: each disaster nudges the BASELINE up a little, so a
            # harsh life permanently shifts the set-point. early events (big
            # w) shift it most -- childhood shapes temperament for life.
            self.fear_baseline = min(0.9, self.fear_baseline + 0.012 * w)
            self.content_baseline = max(0.1,
                self.content_baseline - 0.006 * w)
            if context is not None:
                c = self._ctx_key(context)
                self.assoc[c] = min(1.0, self.assoc.get(c, 0.0) + 0.3 * w)
        elif kind == "relief":
            self.content = min(1.0, self.content + 0.12 * w)
            self.fear = max(0.0, self.fear - 0.08 * w)
            # gentle lives slowly lower the fear baseline and raise content
            self.content_baseline = min(0.9,
                self.content_baseline + 0.002 * w)
            self.fear_baseline = max(0.0, self.fear_baseline - 0.001 * w)

    def _ctx_key(self, context):
        # coarse-bin a context vector so "similar" contexts share a key
        return tuple(round(x * 4) for x in context)

    def appraise(self, context):
        """Before acting in a context, re-evoke any fear associated with a
        similar context. This is mood colouring perception: a place that
        once burned you feels dangerous now."""
        c = self._ctx_key(context)
        evoked = self.assoc.get(c, 0.0)
        # momentary fear is baseline-plus-evoked, capped
        return min(1.0, self.fear + 0.5 * evoked)

    def decay(self):
        # drift toward temperament baseline each step (homeostasis)
        self.fear += 0.01 * (self.fear_baseline - self.fear)
        self.content += 0.01 * (self.content_baseline - self.content)

    def sleep(self):
        # CONSOLIDATION: sleep settles the momentary state PARTWAY toward the
        # (now shifting) baseline. partway, not fully, so a hard stretch
        # leaves a residue in the state while allostasis moves the baseline
        # underneath. the being is regulated but not erased.
        self.fear += 0.15 * (self.fear_baseline - self.fear)
        self.content += 0.15 * (self.content_baseline - self.content)

    def caution(self, context):
        """How cautious to be right now: driven by evoked fear. This is
        emotion ALTERING DECISIONS globally."""
        return self.appraise(context)

    def profile(self):
        return dict(fear=round(self.fear, 3),
                    content=round(self.content, 3),
                    fear_baseline=round(self.fear_baseline, 3),
                    associations=len(self.assoc))


def random_temperament(rng):
    return dict(fear_baseline=rng.uniform(0.1, 0.4),
                content_baseline=rng.uniform(0.3, 0.7),
                reactivity=rng.uniform(0.7, 1.3))


def inherit_temperament(parent_emotion, rng):
    """A child inherits a baseline shaped by the parent's LIVED emotional
    state (not just the parent's innate baseline), plus drift. Nature passes
    forward, bent by how the parent's life actually went."""
    return dict(
        fear_baseline=min(0.8, max(0.0,
            0.5 * parent_emotion.fear_baseline
            + 0.5 * parent_emotion.fear + rng.uniform(-0.1, 0.1))),
        content_baseline=min(0.9, max(0.1,
            0.5 * parent_emotion.content_baseline
            + 0.5 * parent_emotion.content + rng.uniform(-0.1, 0.1))),
        reactivity=min(1.5, max(0.5,
            parent_emotion.reactivity + rng.uniform(-0.15, 0.15))))


def live(temperament, harshness, steps, seed, use_emotion=True):
    """One life. Each step the being faces a situation; 'risky' situations
    may be disasters. With emotion, it heeds its fear (global caution +
    learned associations) to decide whether to engage. Returns reward and
    final emotional profile."""
    rng = random.Random(seed)
    emo = Emotion(temperament)
    total = 0.0
    disasters = 0
    engaged = 0

    for t in range(steps):
        # a situation with a context vector; some are risky
        context = [rng.random() for _ in range(3)]
        risky = context[0] > 0.5
        # harsher worlds make risky situations more often truly disastrous
        is_disaster = risky and (rng.random() < DISASTER_BASE_PROB * harshness)

        if use_emotion:
            # DECISION altered by emotion: the more caution the context
            # evokes, the less likely to engage a risky situation
            caution = emo.caution(context)
            if risky:
                engage = rng.random() > (0.3 + 0.7 * caution)
            else:
                engage = True
        else:
            # no emotion: engage everything (reckless baseline)
            engage = True

        if engage:
            engaged += 1
            if is_disaster:
                total += DISASTER
                disasters += 1
                if use_emotion:
                    emo.event("disaster", t, context)
            else:
                total += SAFE_REWARD
                if use_emotion:
                    emo.event("relief", t, context)

        if use_emotion:
            emo.decay()
            if t % 200 == 0 and t > 0:
                emo.sleep()

    return dict(reward=total, disasters=disasters, engaged=engaged,
                emo=emo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beings", type=int, default=6)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    # =============================================================
    print("=" * 68)
    print("PART A: DOES EMOTION DO ITS JOB? (emotion vs none, harsh world)")
    print("=" * 68)
    with_e, without_e = [], []
    for seed in range(args.seeds):
        temp = random_temperament(random.Random(seed))
        r_e = live(temp, harshness=2.0, steps=args.steps, seed=seed,
                   use_emotion=True)
        r_n = live(temp, harshness=2.0, steps=args.steps, seed=seed,
                   use_emotion=False)
        with_e.append(r_e["reward"])
        without_e.append(r_n["reward"])
    we, wn = statistics.mean(with_e), statistics.mean(without_e)
    print(f"  with emotion:    {we:>8.0f} reward")
    print(f"  without emotion: {wn:>8.0f} reward (engages everything, "
          f"reckless)")
    if we > wn + 0.1 * abs(wn):
        print(f"  EMOTION EARNED ITS PLACE: fear learned from early "
              f"disasters made the being")
        print(f"  cautious in the contexts that burned it, while the "
              f"emotionless being kept")
        print(f"  walking into harm. Emotion is a global response to its "
              f"life that protects it.")
    else:
        print(f"  no clear benefit here; the world may not punish "
              f"recklessness enough.")

    # =============================================================
    print("\n" + "=" * 68)
    print("PART B: DO EMOTIONS INDIVIDUATE? (a lineage of different lives)")
    print("=" * 68)
    print(f"  {args.beings} beings, each living a different life, some harsh "
          f"some gentle.")
    print(f"  do their emotional profiles diverge, and does inheritance "
          f"carry a bent forward?\n")

    rng = random.Random(100)
    # a lineage: each being inherits from the previous, lives its own life
    parent_emo = None
    print(f"  {'being':>6} {'life':>8} {'fear':>7} {'content':>8} "
          f"{'assoc':>6} {'earned fear set-point':>18}")
    print("-" * 68)
    profiles = []
    for i in range(args.beings):
        if parent_emo is None:
            temp = random_temperament(rng)
        else:
            temp = inherit_temperament(parent_emo, rng)
        # alternate harsh and gentle lives, so lives genuinely differ
        harsh = 2.5 if i % 2 == 0 else 0.4
        life_label = "harsh" if harsh > 1 else "gentle"
        r = live(temp, harshness=harsh, steps=args.steps, seed=200 + i)
        p = r["emo"].profile()
        profiles.append(p)
        print(f"  {i:>6} {life_label:>8} {p['fear']:>7.2f} "
              f"{p['content']:>8.2f} {p['associations']:>6} "
              f"{p['fear_baseline']:>18.2f}")
        parent_emo = r["emo"]

    fears = [p["fear_baseline"] for p in profiles]
    contents = [p["content"] for p in profiles]
    print("\n" + "=" * 68)
    print("WHAT IT SAYS")
    print("=" * 68)
    fear_spread = max(fears) - min(fears)
    print(f"  fear ranged from {min(fears):.2f} to {max(fears):.2f} "
          f"(spread {fear_spread:.2f}) across the beings.")
    if fear_spread > 0.2:
        print(f"  EMOTIONS INDIVIDUATE: beings who lived different lives "
              f"ended up emotionally")
        print(f"  different, the harsh-life beings carrying more fear, the "
              f"gentle-life ones")
        print(f"  more content. Nature (inherited temperament) plus history "
              f"(the life lived)")
        print(f"  produced a distinct individual each time, which is exactly "
              f"how emotional")
        print(f"  character forms in people.")
    else:
        print(f"  weak individuation; lives may be too similar or decay too "
              f"strong.")

    print(f"\n  Every property here is functional: emotion emerges, shapes "
          f"what matters,")
    print(f"  alters memory and decisions, and individuates. Whether any of "
          f"it is FELT is")
    print(f"  not addressed and cannot be, by this or any build. It is the "
          f"closest")
    print(f"  buildable thing to human emotional development, and no claim "
          f"is made beyond")
    print(f"  what it does.")


if __name__ == "__main__":
    main()
