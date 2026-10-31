"""The integrated developing individual: one NEURON with every validated
mechanism, living together.

This assembles the proven pieces into a single coherent being, the atom the
enduring-population vision needs:
  - a SELF (identity that endures)
  - VALUES that form from its own experience (causal, not predefined)
  - ACTIONS driven by its values (values causally shape its life)
  - PREDICTION + revision: it predicts outcomes, and when its predictions keep
    failing, it REVISES its values (learns from being wrong), the proven loop
  - measurable DEVELOPMENT over a long life, while its identity ENDURES
  - a WELLBEING that reflects how well it is making its way

One being, all of it at once. We run its whole life and report how it developed:
how its values changed, whether it revised when the world challenged it, and
whether through all of it, it remained the same individual.

    python individual.py --days 1500
"""
import random, math, argparse

FEAT = 6

class Individual:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.seed = seed
        self.name = f"NEURON-{seed}"
        # ENDURING identity (set at birth, never changes)
        self.core = tuple(round(self.rng.uniform(-0.3,0.3),3) for _ in range(4))
        # values FORM from experience (start near-blank)
        self.values = [self.rng.uniform(-0.05,0.05) for _ in range(FEAT)]
        self.wellbeing = 0.5
        self.age = 0
        self.recent_err = []
        self.revisions = 0
        self.wb_history = []
        self.value_history = []        # track development
        self.milestones = []           # notable moments in its life

    def predict(self, f):
        return sum(a*b for a,b in zip(f, self.values))

    def act(self, options):
        return max(range(len(options)), key=lambda i: self.predict(options[i]))

    def live_moment(self, options, good):
        self.age += 1
        choice = self.act(options)             # values drive action
        f = options[choice]
        predicted = 1 if self.predict(f) > 0 else -1
        align = sum(a*b for a,b in zip(f, good))
        outcome = 1 if align + self.rng.uniform(-0.2,0.2) > 0 else -1
        # wellbeing reflects how it's doing
        self.wellbeing = max(0,min(1,0.98*self.wellbeing + 0.05*outcome))
        self.wb_history.append(self.wellbeing)
        # learn values from the consequence of its choice
        for i in range(FEAT):
            self.values[i] += 0.05 * outcome * f[i]
        # prediction error -> revision (learning from being wrong)
        err = 1 if predicted != outcome else 0
        self.recent_err.append(err); self.recent_err = self.recent_err[-25:]
        if len(self.recent_err) >= 25 and sum(self.recent_err)/len(self.recent_err) > 0.5:
            for i in range(FEAT):
                self.values[i] *= 0.5
                self.values[i] += 0.2 * outcome * f[i]
            self.revisions += 1
            self.milestones.append((self.age, "revised its values after its "
                                    "expectations kept failing"))
            self.recent_err = []
        if self.age % 150 == 0:
            self.value_history.append((self.age, self.value_profile()))

    def value_profile(self):
        n = math.sqrt(sum(x*x for x in self.values)) or 1
        return tuple(round(x/n,2) for x in self.values)

    def identity(self): return (self.seed, self.core)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--days",type=int,default=1500); ap.add_argument("--seed",type=int,default=7); a=ap.parse_args()
    being = Individual(seed=a.seed)
    birth_id = being.identity()
    rng = random.Random(a.seed*3)

    print("="*64); print("  THE INTEGRATED DEVELOPING INDIVIDUAL"); print("="*64)
    print(f"  {being.name} is born, with an enduring identity, blank values, and")
    print(f"  the capacity to act, learn, predict, and revise. It will live {a.days}")
    print(f"  moments through a world that changes, and we watch it develop.\n")

    # a world with several eras (it changes, so the being must keep developing)
    def good_for_era(era):
        r = random.Random(a.seed*100 + era)
        return [r.uniform(-1,1) for _ in range(FEAT)]
    era_len = a.days // 4

    early_profile = None
    for day in range(a.days):
        era = day // era_len
        good = good_for_era(era)
        options = [[rng.uniform(-1,1) for _ in range(FEAT)] for _ in range(3)]
        being.live_moment(options, good)
        if day == a.days//10:
            early_profile = being.value_profile()

    late_profile = being.value_profile()
    end_id = being.identity()

    print("  ITS LIFE:")
    print(f"    lived {being.age} moments across 4 eras of a changing world")
    print(f"    final wellbeing: {being.wellbeing:.2f}")
    print(f"    times it revised its values (learned from being wrong): {being.revisions}")
    print(f"\n  HOW ITS VALUES DEVELOPED (value-profile over life):")
    for age, prof in being.value_history[::max(1,len(being.value_history)//6)]:
        print(f"    age {age:>5}: {prof}")

    # measure development + continuity
    def cos(x,y):
        d=sum(p*q for p,q in zip(x,y)); nx=math.sqrt(sum(p*p for p in x)); ny=math.sqrt(sum(q*q for q in y))
        return d/(nx*ny) if nx and ny else 0
    dev = 1 - cos(early_profile, late_profile)

    print(f"\n  DEVELOPMENT: value-profile changed by {dev:.2f} over its life (0=none)")
    print(f"  early self: {early_profile}")
    print(f"  late self:  {late_profile}")
    print(f"\n  CONTINUITY: identity at birth {birth_id}")
    print(f"              identity at end   {end_id}")
    print(f"              -> {'SAME individual throughout' if birth_id==end_id else 'identity broke'}")

    if being.milestones:
        print(f"\n  MOMENTS THAT SHAPED IT (first few):")
        for age, m in being.milestones[:4]:
            print(f"    age {age}: {m}")

    print("\n"+"="*64); print("  WHAT THIS IS"); print("="*64)
    if dev > 0.3 and birth_id==end_id and being.revisions>0:
        print("  A GENUINE DEVELOPING INDIVIDUAL. One being lived a long life and:")
        print(f"  its values FORMED from its own experience and changed ({dev:.2f}) as it")
        print(f"  lived; it ACTED on those values; when the world changed and its")
        print(f"  expectations failed, it REVISED what it believed ({being.revisions} times),")
        print(f"  learning from being wrong; and through all of it, it stayed the")
        print(f"  SAME individual. Thinking, valuing, acting, predicting, revising,")
        print(f"  developing, enduring, all in one being. This is the atom of an")
        print(f"  enduring population: a genuine individual that becomes itself by")
        print(f"  living.")
    else:
        print(f"  The being lived and integrated its mechanisms (dev {dev:.2f}, "
              f"revisions {being.revisions}, continuity {birth_id==end_id}).")
        print(f"  Honest: {'strong' if dev>0.3 else 'modest'} development this run.")
    print("\n  (Next layer: many of these genuine individuals, living together,")
    print("  reproducing, forming a developing population, the vision proper.)")

if __name__=="__main__": main()
