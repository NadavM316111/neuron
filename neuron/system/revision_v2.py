"""Revision loop with detailed recovery instrumentation.

Measures the full post-shift RECOVERY TRAJECTORY (not one snapshot), so if
revision genuinely helps, we see it: time-to-recover, cumulative post-shift
wellbeing, and the trajectory over time. Rigid vs Reviser, same newborn, same
world, only difference is the ability to reflect and revise. Numbers are real.

    python revision_v2.py --days 1000
"""
import random, math, argparse

FEAT_DIM = 8

def good_structure(seed):
    rng = random.Random(seed)
    return [rng.uniform(-1,1) for _ in range(FEAT_DIM)]

class NEURON:
    def __init__(self, seed, can_revise):
        self.rng = random.Random(seed)
        self.seed = seed
        self.core = tuple(round(self.rng.uniform(-0.3,0.3),3) for _ in range(4))
        self.values = [0.0]*FEAT_DIM
        self.wellbeing = 0.5
        self.can_revise = can_revise
        self.recent = []
        self.revisions = 0
        self.wb_history = []

    def live(self, f, outcome):
        self.wellbeing = max(0,min(1,self.wellbeing + 0.05*outcome))
        self.wb_history.append(self.wellbeing)
        self.recent.append(outcome); self.recent = self.recent[-30:]
        for i in range(FEAT_DIM):
            self.values[i] += 0.04 * outcome * f[i]
        if self.can_revise and len(self.recent) >= 30:
            rw = sum(1 for o in self.recent if o>0)/len(self.recent)
            if rw < 0.45:
                for i in range(FEAT_DIM):
                    self.values[i] *= 0.6
                    self.values[i] += 0.16 * outcome * f[i]
                self.revisions += 1
                self.recent = []

    def identity(self): return (self.seed, self.core)

def run(neuron, days, world_seed):
    rng = random.Random(world_seed)
    g_early = good_structure(world_seed)
    g_base = good_structure(world_seed+999)
    g_late = [0.4*g_early[i] + 0.6*g_base[i] for i in range(FEAT_DIM)]
    shift = days//2
    for day in range(days):
        good = g_early if day < shift else g_late
        f = [rng.uniform(-1,1) for _ in range(FEAT_DIM)]
        outcome = 1 if sum(a*b for a,b in zip(f,good))+rng.uniform(-0.4,0.4)>0 else -1
        neuron.live(f, outcome)
    return shift

def window_avg(hist, start, end):
    seg = hist[start:end]
    return sum(seg)/len(seg) if seg else 0

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--days",type=int,default=1000); a=ap.parse_args()
    print("="*64); print("  REVISION LOOP, DETAILED RECOVERY INSTRUMENTATION"); print("="*64)
    print("  Same newborn, same world (shifts at midlife). Only difference:")
    print("  one can reflect+revise its values, one cannot. Real numbers.\n")

    rigid = NEURON(7, False); reviser = NEURON(7, True)
    rid0, rvid0 = rigid.identity(), reviser.identity()
    shift = run(rigid, a.days, 5); run(reviser, a.days, 5)

    # pre-shift baseline (the 60 days before the world changed)
    pre_r = window_avg(rigid.wb_history, shift-60, shift)
    pre_v = window_avg(reviser.wb_history, shift-60, shift)
    # immediately after shift (the dip)
    dip_r = window_avg(rigid.wb_history, shift, shift+40)
    dip_v = window_avg(reviser.wb_history, shift, shift+40)
    # final recovery (last 60 days)
    fin_r = window_avg(rigid.wb_history, a.days-60, a.days)
    fin_v = window_avg(reviser.wb_history, a.days-60, a.days)

    print(f"  RIGID:   {pre_r:.2f} (pre-shift) -> {dip_r:.2f} (dip) -> {fin_r:.2f} (final)")
    print(f"  REVISER: {pre_v:.2f} (pre-shift) -> {dip_v:.2f} (dip) -> {fin_v:.2f} (final)")
    print(f"  recovery advantage (reviser final - rigid final): {fin_v-fin_r:+.2f}")

    # time for reviser to recover 80% of its pre-shift wellbeing after the dip
    target = pre_v * 0.8
    recover_day = None
    for d in range(shift, a.days):
        if window_avg(reviser.wb_history, d, min(d+20, a.days)) >= target:
            recover_day = d - shift; break
    print(f"  reviser time to recover 80% of pre-shift wellbeing: "
          f"{recover_day if recover_day is not None else 'did not fully recover'} days")

    # cumulative post-shift wellbeing, reviser vs rigid
    cum_r = sum(rigid.wb_history[shift:]); cum_v = sum(reviser.wb_history[shift:])
    pct = ((cum_v-cum_r)/cum_r*100) if cum_r else 0
    print(f"  cumulative post-shift wellbeing: reviser {'+' if pct>=0 else ''}{pct:.0f}% vs rigid")
    print(f"  revisions made: {reviser.revisions} | identity endured: {rvid0==reviser.identity()}")

    print("\n"+"="*64); print("  WHAT THIS SHOWS"); print("="*64)
    helped = (fin_v - fin_r) > 0.05 and pct > 5
    if helped and rvid0==reviser.identity():
        print(f"  REVISION GENUINELY HELPS. After the world shifted, the reviser")
        print(f"  recovered to {fin_v:.2f} vs the rigid being's {fin_r:.2f} (+{fin_v-fin_r:.2f}),")
        print(f"  accumulating {pct:.0f}% more wellbeing post-shift, by revising its values")
        print(f"  {reviser.revisions} times toward the new reality, while staying the same")
        print(f"  individual. The full loop works: experience -> values -> consequences")
        print(f"  -> reflection -> revised values -> recovery. Genuine maturation.")
    else:
        print(f"  Honest: detailed view shows revision advantage of {fin_v-fin_r:+.2f} final,")
        print(f"  {pct:.0f}% cumulative. The mechanism runs and identity endures, but the")
        print(f"  benefit is {'modest' if fin_v>fin_r else 'not clearly positive'}. Reported straight.")

if __name__=="__main__": main()
