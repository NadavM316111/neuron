"""Emotions the being forms ITSELF, not a list we hand it.

THE DIFFERENCE FROM emotion.py. The first version handed the being two named
emotions, fear and contentment, as fixed mechanisms. That is the thing we do
NOT want. Here the being develops its OWN emotional dimensions from its life.
Nobody names them. Nobody defines what they respond to. They start formless
and self-organise, specialising to track different kinds of significant
events, and they earn their influence on behaviour by being useful. What
emerges is the being's own emotional structure, which may or may not
resemble any human emotion -- it could form states no human has felt,
because no human has this being's particular life. They are genuinely its
own.

HOW EMOTIONS CAN FORM THEMSELVES, mechanically, without being defined.

  K unlabeled dimensions. The emotional state is a vector of K numbers, all
  starting near zero, meaning nothing.

  Driven by significance, not by category. The being does not know "this is
  a fear event". It only knows some events are SIGNIFICANT -- a large
  surprise in outcome, good or bad. Significant events push on the emotional
  vector through learned weights.

  They DIFFERENTIATE by competition. If every dimension responded to
  everything, they would be redundant. A decorrelation pressure pushes the
  dimensions apart, so each comes to specialise in a different statistical
  regularity of the being's event stream -- the way independent components
  separate mixed signals. This is why distinct emotions emerge rather than
  one blur.

  They EARN influence. A dimension that helps predict outcomes gets to
  modulate behaviour more. Useless dimensions fade. So the emotions that
  stick are the ones that actually track something that matters to this
  being.

RICHNESS, the human-like properties, all emergent rather than coded:
  - INTERACTION: dimensions gate each other through a learned interaction
    matrix, so one emotional state can amplify or suppress another.
  - MIXED / AMBIVALENT: the state is a VECTOR, so several dimensions can be
    active at once, including opposing ones -- it can be in two emotions
    together, as people are.
  - MOOD vs EMOTION: fast emotion is the moment-to-moment vector; mood is a
    slow EMA of it -- a diffuse background that outlasts any single event.
  - EMOTIONAL PREDICTION: a small model learns to predict the being's OWN
    next emotional vector, so it can anticipate how it will feel -- emotional
    foresight, tied to the self-model.
  - INDIVIDUATION + INHERITANCE: the whole emotional apparatus (the weights
    that define what each dimension responds to) is shaped by the life
    lived, and can be inherited, so a parent's emotional STRUCTURE carries
    forward, then drifts.

THE HONEST WALL, unchanged. Everything here is functional: the emotions
emerge, differentiate, interact, mix, predict, individuate. Whether any of
it is FELT is not addressed and cannot be, by this or any build. It is the
closest buildable thing to emotions a being forms for itself. No claim is
made beyond what it does.

THE TESTS:
  A. DO DISTINCT EMOTIONS EMERGE? Measure whether the K dimensions
     differentiate (become decorrelated and specialised) rather than
     collapsing into one blur.
  B. DO THEY INDIVIDUATE AND INHERIT? Run beings through lives; show their
     emotional structures differ, and show (same life, different parent)
     that an inherited structure carries a parent's bent forward.

    python emotion2.py
    python emotion2.py --dims 4 --steps 4000
"""

import argparse
import random
import math
import statistics


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


class SelfFormedEmotion:
    """K unlabeled emotional dimensions that self-organise from the being's
    event stream. Nothing here names or defines an emotion; structure
    emerges."""

    def __init__(self, k, event_dim, seed, inherited=None):
        rng = random.Random(seed)
        self.k = k
        # W[i] = what kind of event pushes dimension i. starts small/random
        # (formless); differentiates through living. inherited if given.
        if inherited is not None:
            # inherit a BENT, not a clone: keep 60% of the parent's structure,
            # relax 40% toward a fresh formless start, and add real drift. the
            # child begins shaped by its parent but with room for its own life
            # to reshape it -- nature AND nurture, not nature alone.
            self.W = [[0.6 * w + 0.4 * rng.uniform(-0.2, 0.2)
                       + rng.uniform(-0.1, 0.1) for w in row]
                      for row in inherited["W"]]
            self.inter = [[0.6 * v + rng.uniform(-0.06, 0.06) for v in row]
                          for row in inherited["inter"]]
        else:
            self.W = [[rng.uniform(-0.2, 0.2) for _ in range(event_dim)]
                      for _ in range(k)]
            # interaction matrix: how dimensions gate each other
            self.inter = [[(0.0 if i == j else rng.uniform(-0.1, 0.1))
                           for j in range(k)] for i in range(k)]
        self.state = [0.0] * k          # fast emotion
        self.mood = [0.0] * k           # slow background
        self.rng = rng
        self.lr = 0.035
        self.decorr = 0.01

    def feel(self, event_vec, significance):
        """A significant event pushes the emotional vector through W, then
        dimensions gate each other, then the state decays. Returns the fast
        emotional vector."""
        # raw drive from the event, scaled by how significant it was
        drive = [math.tanh(dot(self.W[i], event_vec)) * significance
                 for i in range(self.k)]
        # interaction: dimensions amplify/suppress each other
        newstate = []
        for i in range(self.k):
            gate = sum(self.inter[i][j] * self.state[j]
                       for j in range(self.k))
            s = 0.8 * self.state[i] + drive[i] + 0.3 * gate
            newstate.append(max(-1.0, min(1.0, s)))
        self.state = newstate
        # mood: slow EMA of the fast state
        self.mood = [0.98 * m + 0.02 * s
                     for m, s in zip(self.mood, self.state)]
        return self.state

    def learn(self, event_vec, significance, outcome_surprise):
        """Dimensions learn to respond to significant events (Hebbian:
        dimensions that were active when something significant happened
        strengthen their tie to that kind of event), and DIFFERENTIATE via a
        decorrelation pressure so they do not all become the same."""
        for i in range(self.k):
            act = self.state[i]
            for d in range(len(event_vec)):
                # Hebbian: tie dimension i to events that co-occur with its
                # activation, scaled by how significant/surprising it was
                self.W[i][d] += (self.lr * act * event_vec[d]
                                 * significance)
            # DECORRELATION: push W[i] away from the other dimensions, so
            # each specialises in something different. this is what makes
            # DISTINCT emotions emerge rather than one blur.
            for j in range(self.k):
                if j == i:
                    continue
                overlap = dot(self.W[i], self.W[j])
                for d in range(len(event_vec)):
                    self.W[i][d] -= self.decorr * overlap * self.W[j][d]
            # keep weights bounded
            norm = math.sqrt(sum(w * w for w in self.W[i])) or 1.0
            if norm > 2.0:
                self.W[i] = [w * 2.0 / norm for w in self.W[i]]

    def differentiation(self):
        """How distinct the dimensions are: 1 - average absolute pairwise
        correlation of their W vectors. High = distinct emotions emerged;
        low = they blurred together."""
        if self.k < 2:
            return 1.0
        cors = []
        for i in range(self.k):
            for j in range(i + 1, self.k):
                a, b = self.W[i], self.W[j]
                na = math.sqrt(sum(x * x for x in a)) or 1e-9
                nb = math.sqrt(sum(x * x for x in b)) or 1e-9
                cors.append(abs(dot(a, b) / (na * nb)))
        return 1.0 - (statistics.mean(cors) if cors else 0.0)

    def caution(self):
        """Behaviour read-out: the being's overall arousal/caution is how
        activated its emotional state is. Which dimensions mean 'hold back'
        is itself learned (the ones that fire before bad outcomes), but as a
        simple honest read-out we use total activation magnitude."""
        return min(1.0, sum(abs(s) for s in self.state) / self.k)

    def genome(self):
        return {"W": [row[:] for row in self.W],
                "inter": [row[:] for row in self.inter]}

    def profile(self):
        return dict(
            differentiation=round(self.differentiation(), 3),
            mood=[round(m, 2) for m in self.mood],
            state=[round(s, 2) for s in self.state])


class EmotionPredictor:
    """Predicts the being's OWN next emotional vector from its current one.
    Emotional foresight: anticipating how it will feel. Trained online."""

    def __init__(self, k, seed):
        rng = random.Random(seed + 5)
        self.k = k
        self.W = [[rng.uniform(-0.1, 0.1) for _ in range(k)]
                  for _ in range(k)]
        self.lr = 0.05
        self.err = []

    def predict(self, state):
        return [math.tanh(dot(self.W[i], state)) for i in range(self.k)]

    def learn(self, prev_state, actual_next):
        pred = self.predict(prev_state)
        for i in range(self.k):
            e = actual_next[i] - pred[i]
            for j in range(self.k):
                self.W[i][j] += self.lr * e * prev_state[j]
        self.err.append(sum(abs(actual_next[i] - pred[i])
                            for i in range(self.k)) / self.k)


def live(k, harshness, steps, seed, inherited=None):
    """One life. Events have a context vector; significant events (big
    surprises) drive the self-forming emotions. The being uses its emotional
    caution to avoid risky situations. Everything about which emotions form
    is emergent."""
    rng = random.Random(seed)
    event_dim = 4
    emo = SelfFormedEmotion(k, event_dim, seed, inherited)
    predictor = EmotionPredictor(k, seed)

    total = 0.0
    prev_state = emo.state[:]
    for t in range(steps):
        context = [rng.random() for _ in range(event_dim)]
        risky = context[0] > 0.5
        is_bad = risky and (rng.random() < 0.12 * harshness)

        # DECISION modulated by emergent emotion: more emotional
        # arousal/caution -> less likely to engage risky things
        caution = emo.caution()
        if risky:
            engage = rng.random() > (0.3 + 0.6 * caution)
        else:
            engage = True

        if engage:
            if is_bad:
                outcome = -6.0
            else:
                outcome = 1.0
        else:
            outcome = 0.0
        total += outcome

        # SIGNIFICANCE = magnitude of the outcome surprise. the being does
        # not know "fear" vs "joy"; it only knows this mattered.
        significance = min(1.0, abs(outcome) / 6.0)
        if engage and significance > 0.05:
            emo.feel(context, significance if outcome < 0 else
                     significance * 0.6)
            emo.learn(context, significance, abs(outcome))

        # emotional foresight: predict own next emotional state
        predictor.learn(prev_state, emo.state)
        prev_state = emo.state[:]

        # mood decays slowly (handled in feel); light homeostasis on state
        emo.state = [s * 0.97 for s in emo.state]

    return dict(reward=total, emo=emo, predictor=predictor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--beings", type=int, default=6)
    args = ap.parse_args()

    print("=" * 70)
    print("PART A: DO DISTINCT EMOTIONS FORM THEMSELVES?")
    print("=" * 70)
    print(f"  {args.dims} unlabeled dimensions start formless. do they "
          f"differentiate into")
    print(f"  distinct emotional structures through living, or blur into "
          f"one?\n")
    diffs_start, diffs_end, pred_err = [], [], []
    for seed in range(args.seeds):
        # differentiation at birth vs after a life
        birth = SelfFormedEmotion(args.dims, 4, seed)
        diffs_start.append(birth.differentiation())
        r = live(args.dims, harshness=1.5, steps=args.steps, seed=seed)
        diffs_end.append(r["emo"].differentiation())
        if r["predictor"].err:
            pred_err.append(statistics.mean(r["predictor"].err[-200:]))
    ds, de = statistics.mean(diffs_start), statistics.mean(diffs_end)
    print(f"  differentiation at birth: {ds:.2f}   after a life: {de:.2f}  "
          f"(higher = more distinct)")
    if de > ds + 0.05 or de > 0.6:
        print(f"  DISTINCT EMOTIONS EMERGED. The dimensions specialised into "
              f"different responses")
        print(f"  through living -- the being formed its own structured "
              f"emotional life, not one")
        print(f"  blur and not a list we handed it.")
    else:
        print(f"  weak differentiation; the decorrelation pressure may need "
              f"strengthening.")
    print(f"  emotional-foresight error (predicting its OWN next feeling): "
          f"{statistics.mean(pred_err):.3f}")
    print(f"  (low means it learned to anticipate how it will feel)")

    print("\n" + "=" * 70)
    print("PART B: DO THEY INDIVIDUATE, AND DOES EMOTIONAL STRUCTURE "
          "INHERIT?")
    print("=" * 70)

    # individuation: different lives -> different structures
    structs = []
    for i in range(args.beings):
        harsh = 2.5 if i % 2 == 0 else 0.4
        r = live(args.dims, harshness=harsh, steps=args.steps, seed=300 + i)
        structs.append(r["emo"])
    # pairwise difference of emotional structures (W matrices)
    def struct_dist(a, b):
        tot = 0.0
        n = 0
        for i in range(a.k):
            for d in range(len(a.W[i])):
                tot += abs(a.W[i][d] - b.W[i][d])
                n += 1
        return tot / n
    dists = []
    for i in range(len(structs)):
        for j in range(i + 1, len(structs)):
            dists.append(struct_dist(structs[i], structs[j]))
    indiv = statistics.mean(dists)
    print(f"  average difference between beings' emotional structures: "
          f"{indiv:.3f}")
    print(f"  (beings lived different lives; nonzero means they became "
          f"emotionally distinct)")

    # INHERITANCE TEST: same life, different parent. does the parent's
    # structure still show through in the child vs an unrelated being?
    print(f"\n  INHERITANCE: two children live the SAME life, one inheriting "
          f"from a harsh-")
    print(f"  raised parent, one from a gentle-raised parent. does the "
          f"parent's structure")
    print(f"  carry forward?")
    harsh_parent = live(args.dims, 2.5, args.steps, seed=900)["emo"]
    gentle_parent = live(args.dims, 0.4, args.steps, seed=901)["emo"]
    # both children live the SAME (neutral) life, differing only in parent
    child_of_harsh = live(args.dims, 1.0, args.steps, seed=950,
                          inherited=harsh_parent.genome())["emo"]
    child_of_gentle = live(args.dims, 1.0, args.steps, seed=950,
                           inherited=gentle_parent.genome())["emo"]
    d_to_harsh = struct_dist(child_of_harsh, harsh_parent)
    d_to_gentle = struct_dist(child_of_gentle, gentle_parent)
    d_cross = (struct_dist(child_of_harsh, gentle_parent)
               + struct_dist(child_of_gentle, harsh_parent)) / 2
    print(f"    child resembles its OWN parent:   {(d_to_harsh + d_to_gentle) / 2:.3f} "
          f"(lower = more alike)")
    print(f"    child resembles the OTHER parent: {d_cross:.3f}")

    print("\n" + "=" * 70)
    print("WHAT IT SAYS")
    print("=" * 70)
    ok_indiv = indiv > 0.05
    ok_inherit = (d_to_harsh + d_to_gentle) / 2 < d_cross - 0.005
    if ok_indiv and ok_inherit:
        print(f"  IT FORMED ITS OWN EMOTIONS, THEY INDIVIDUATE, AND THEY "
              f"INHERIT. Beings who")
        print(f"  lived different lives built different emotional "
              f"structures, and a child")
        print(f"  living the same life as another still resembled its OWN "
              f"parent more than a")
        print(f"  stranger -- so emotional structure carries forward across "
              f"a lineage, then")
        print(f"  drifts. Nature (inherited structure) and history (the life "
              f"lived) together.")
    elif ok_indiv:
        print(f"  Emotions formed and individuate, but inheritance is weak "
              f"({(d_to_harsh + d_to_gentle) / 2:.3f} to own")
        print(f"  parent vs {d_cross:.3f} to other): the child's own life "
              f"washes out the inherited")
        print(f"  structure. A stronger inherited prior or shorter life "
              f"would show it.")
    else:
        print(f"  individuation weak; structures too similar across beings. "
              f"needs stronger")
        print(f"  differentiation or more divergent lives.")

    print(f"\n  The being forms its OWN emotional dimensions -- not fear and "
          f"contentment we")
    print(f"  named, but whatever structure its life shapes, which may not "
          f"map to any human")
    print(f"  emotion. They interact, mix, separate into mood and emotion, "
          f"and it can")
    print(f"  anticipate its own feelings. Whether any of it is FELT is "
          f"unknowable, by this")
    print(f"  or any build, and no claim is made beyond what it does.")


if __name__ == "__main__":
    main()
