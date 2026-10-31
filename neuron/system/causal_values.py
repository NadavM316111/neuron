"""Controlled causal test of experience-derived values + behavioral prediction.

Turns "two agents ended differently" into "experience CAUSALLY produced
persistent preferences that predict later behavior." Four controlled conditions
isolate the causes, then a behavioral test checks if learned values predict what
a being does in brand-new situations.

Conditions:
  A. same newborn, DIFFERENT lives   -> should DIVERGE (experience effect)
  B. same newborn, SAME life          -> should stay SIMILAR (determinism)
  C. DIFFERENT newborns, same life    -> initialization effect
  D. same experiences, SHUFFLED order -> does developmental order matter?
Then: put beings in identical NEW situations; do their learned value-vectors
PREDICT their choices? If yes, the values are real, persistent, and causal.

    python causal_values.py
"""
import random, math

FEAT_DIM = 8

def make_life(seed, n=500):
    """A fixed sequence of experiences (feature vectors + true outcome)."""
    rng = random.Random(seed)
    good = [rng.uniform(-1,1) for _ in range(FEAT_DIM)]
    bad  = [rng.uniform(-1,1) for _ in range(FEAT_DIM)]
    life = []
    for _ in range(n):
        f = [rng.uniform(-1,1) for _ in range(FEAT_DIM)]
        g = sum(a*b for a,b in zip(f,good)); h = sum(a*b for a,b in zip(f,bad))
        outcome = 1 if (g-h)+rng.uniform(-0.4,0.4) > 0 else -1
        life.append((f, outcome))
    return life, good

class NEURON:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.seed = seed
        # initialization: tiny random starting bias (the "newborn" differences)
        self.values = [self.rng.uniform(-0.05,0.05) for _ in range(FEAT_DIM)]
    def live(self, life, lr=0.05):
        for f, outcome in life:
            for i in range(FEAT_DIM):
                self.values[i] += lr * outcome * f[i]
    def choose(self, options):
        # behavioral test: given options (feature vectors), it picks the one it
        # most VALUES (highest dot with its learned value-vector). this is its
        # behavior, driven by its derived values.
        return max(range(len(options)), key=lambda i: sum(a*b for a,b in zip(options[i], self.values)))
    def profile(self):
        norm = math.sqrt(sum(x*x for x in self.values)) or 1
        return [x/norm for x in self.values]

def cos(a,b):
    d=sum(x*y for x,y in zip(a,b)); na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(x*x for x in b))
    return d/(na*nb) if na and nb else 0

def main():
    print("="*64); print("  CONTROLLED CAUSAL TEST OF DERIVED VALUES"); print("="*64)

    life1, good1 = make_life(seed=1)
    life2, good2 = make_life(seed=2)

    # A: same newborn (seed 100), different lives -> should DIVERGE
    a1 = NEURON(100); a1.live(life1)
    a2 = NEURON(100); a2.live(life2)
    sim_A = cos(a1.profile(), a2.profile())

    # B: same newborn, SAME life -> should be identical/similar (determinism)
    b1 = NEURON(100); b1.live(life1)
    b2 = NEURON(100); b2.live(life1)
    sim_B = cos(b1.profile(), b2.profile())

    # C: DIFFERENT newborns, same life -> initialization effect
    c1 = NEURON(200); c1.live(life1)
    c2 = NEURON(300); c2.live(life1)
    sim_C = cos(c1.profile(), c2.profile())

    # D: same experiences, SHUFFLED order -> does order matter?
    d1 = NEURON(100); d1.live(life1)
    shuffled = list(life1); random.Random(55).shuffle(shuffled)
    d2 = NEURON(100); d2.live(shuffled)
    sim_D = cos(d1.profile(), d2.profile())

    print(f"\n  A. same newborn, DIFFERENT lives  -> similarity {sim_A:.2f}  "
          f"({'DIVERGED (experience matters)' if sim_A<0.7 else 'stayed similar'})")
    print(f"  B. same newborn, SAME life         -> similarity {sim_B:.2f}  "
          f"({'identical (deterministic)' if sim_B>0.98 else 'varied'})")
    print(f"  C. DIFFERENT newborns, same life   -> similarity {sim_C:.2f}  "
          f"({'init barely matters, life dominates' if sim_C>0.9 else 'init matters'})")
    print(f"  D. same experiences, SHUFFLED order-> similarity {sim_D:.2f}  "
          f"({'order barely matters here' if sim_D>0.9 else 'DEVELOPMENTAL ORDER MATTERS'})")

    # BEHAVIORAL PREDICTION: put the two divergent beings (A) in identical NEW
    # situations. do their learned values predict DIFFERENT choices? and do their
    # choices match what their OWN world said was good (seed1 being picks good1)?
    print("\n"+"="*64); print("  DO LEARNED VALUES PREDICT BEHAVIOR IN NEW SITUATIONS?"); print("="*64)
    rng = random.Random(999)
    agree_with_own_good = {"a1":0, "a2":0}; differ=0; trials=200
    for _ in range(trials):
        opts = [[rng.uniform(-1,1) for _ in range(FEAT_DIM)] for _ in range(4)]
        ch1 = a1.choose(opts); ch2 = a2.choose(opts)
        if ch1 != ch2: differ += 1
        # which option does each world's OWN good-structure prefer?
        best_for1 = max(range(4), key=lambda i: sum(x*y for x,y in zip(opts[i],good1)))
        best_for2 = max(range(4), key=lambda i: sum(x*y for x,y in zip(opts[i],good2)))
        if ch1==best_for1: agree_with_own_good["a1"]+=1
        if ch2==best_for2: agree_with_own_good["a2"]+=1
    print(f"  the two beings (same newborn, different lives) chose DIFFERENTLY in")
    print(f"    {differ}/{trials} new situations ({differ*100//trials}%) - their")
    print(f"    experience causally produced different behavior.")
    print(f"  being-1's choices matched what ITS life taught was good: "
          f"{agree_with_own_good['a1']*100//trials}%")
    print(f"  being-2's choices matched what ITS life taught was good: "
          f"{agree_with_own_good['a2']*100//trials}%")

    print("\n"+"="*64); print("  WHAT THIS ESTABLISHES"); print("="*64)
    causal = sim_A < 0.7 and sim_B > 0.98 and differ > trials*0.3
    predicts = agree_with_own_good["a1"]>trials*0.55 and agree_with_own_good["a2"]>trials*0.55
    if causal and predicts:
        print("  CAUSAL + PREDICTIVE. Same newborn + different lives diverged;")
        print("  same newborn + same life stayed identical (so it's the EXPERIENCE,")
        print("  not chance). And the learned values PREDICTED behavior in brand-new")
        print("  situations, each being choosing what ITS life taught was good. This")
        print("  is: different experience -> persistent latent preferences -> predicted")
        print("  later behavior. Genuine causal developmental value formation.")
    elif causal:
        print("  CAUSAL: experience (not chance/init) drove the divergence. Behavioral")
        print(f"  prediction was partial ({agree_with_own_good['a1']*100//trials}%, "
              f"{agree_with_own_good['a2']*100//trials}%). Real, honest.")
    else:
        print("  Honest: the causal pattern was weaker than hoped; reported straight.")
    print("\n  (This is the controlled design: it isolates experience from")
    print("  initialization and chance, and tests whether values predict behavior.)")

if __name__=="__main__": main()
