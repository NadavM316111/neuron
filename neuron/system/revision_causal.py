"""Revision loop, rebuilt so values CAUSALLY drive outcomes.

The bug before: the being's values didn't affect its outcomes, so revision had
nothing to improve. Fixed here: the being ACTS on its values (it chooses which
situations to engage based on what it values), and those choices determine its
outcomes. So values that match reality -> good choices -> good life; values that
don't -> bad choices -> bad life. Now revision genuinely matters: when the world
changes and old values cause bad choices, a being that revises chooses better
and recovers; a rigid one keeps choosing badly.

Same newborn, same world (shifts at midlife), only difference: can revise or not.

    python revision_causal.py --days 1000
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
        self.values = [self.rng.uniform(-0.05,0.05) for _ in range(FEAT_DIM)]
        self.wellbeing = 0.5
        self.can_revise = can_revise
        self.recent = []
        self.revisions = 0
        self.wb_history = []

    def act(self, options):
        # it CHOOSES the option it most values (its values drive its action)
        return max(range(len(options)),
                   key=lambda i: sum(a*b for a,b in zip(options[i], self.values)))

    def live(self, chosen_features, outcome):
        # outcome is the consequence of the CHOICE it made based on its values
        self.wellbeing = max(0,min(1,self.wellbeing + 0.05*outcome))
        self.wb_history.append(self.wellbeing)
        self.recent.append(outcome); self.recent = self.recent[-30:]
        # it learns from the consequence of its choice
        for i in range(FEAT_DIM):
            self.values[i] += 0.04 * outcome * chosen_features[i]
        # revision: if its recent choices have gone badly, reflect + revise
        if self.can_revise and len(self.recent) >= 30:
            rw = sum(1 for o in self.recent if o>0)/len(self.recent)
            if rw < 0.45:
                for i in range(FEAT_DIM):
                    self.values[i] *= 0.6
                    self.values[i] += 0.16 * outcome * chosen_features[i]
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
        # the world offers 4 options (situations); the being CHOOSES one by its
        # values; the chosen one's alignment with reality's good-structure
        # determines the outcome. values -> choice -> outcome. causal.
        options = [[rng.uniform(-1,1) for _ in range(FEAT_DIM)] for _ in range(4)]
        choice = neuron.act(options)
        chosen = options[choice]
        align = sum(a*b for a,b in zip(chosen, good))
        outcome = 1 if align + rng.uniform(-0.3,0.3) > 0 else -1
        neuron.live(chosen, outcome)
    return shift

def wavg(h,s,e):
    seg=h[s:e]; return sum(seg)/len(seg) if seg else 0

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--days",type=int,default=1000); a=ap.parse_args()
    print("="*64); print("  REVISION LOOP (values causally drive outcomes)"); print("="*64)
    print("  Now the being ACTS on its values and lives the consequences. When")
    print("  the world shifts, good values -> good choices. Does revising help?\n")

    rigid=NEURON(7,False); reviser=NEURON(7,True)
    rvid0=reviser.identity()
    shift=run(rigid,a.days,5); run(reviser,a.days,5)

    pre_r=wavg(rigid.wb_history,shift-60,shift); pre_v=wavg(reviser.wb_history,shift-60,shift)
    dip_r=wavg(rigid.wb_history,shift,shift+40); dip_v=wavg(reviser.wb_history,shift,shift+40)
    fin_r=wavg(rigid.wb_history,a.days-60,a.days); fin_v=wavg(reviser.wb_history,a.days-60,a.days)

    print(f"  RIGID:   {pre_r:.2f} (pre-shift) -> {dip_r:.2f} (dip) -> {fin_r:.2f} (final)")
    print(f"  REVISER: {pre_v:.2f} (pre-shift) -> {dip_v:.2f} (dip) -> {fin_v:.2f} (final)")
    print(f"  recovery advantage (reviser - rigid, final): {fin_v-fin_r:+.2f}")

    target=pre_v*0.8; rec=None
    for d in range(shift,a.days):
        if wavg(reviser.wb_history,d,min(d+20,a.days))>=target: rec=d-shift; break
    print(f"  reviser time to recover 80% of pre-shift wellbeing: "
          f"{rec if rec is not None else 'did not fully recover'} days")
    cum_r=sum(rigid.wb_history[shift:]); cum_v=sum(reviser.wb_history[shift:])
    pct=((cum_v-cum_r)/cum_r*100) if cum_r else 0
    print(f"  cumulative post-shift wellbeing: reviser {'+' if pct>=0 else ''}{pct:.0f}% vs rigid")
    print(f"  revisions: {reviser.revisions} | identity endured: {rvid0==reviser.identity()}")

    print("\n"+"="*64); print("  WHAT THIS SHOWS"); print("="*64)
    helped=(fin_v-fin_r)>0.04 and pct>4
    if helped and rvid0==reviser.identity():
        print(f"  REVISION GENUINELY HELPS NOW. With values causally driving choices,")
        print(f"  the reviser recovered to {fin_v:.2f} vs rigid {fin_r:.2f} (+{fin_v-fin_r:.2f}), with")
        print(f"  {pct:.0f}% more cumulative post-shift wellbeing, by revising {reviser.revisions} times")
        print(f"  toward the changed world, while staying the same individual. The full")
        print(f"  loop works: experience -> values -> ACTIONS -> consequences ->")
        print(f"  reflection -> revised values -> recovery. Genuine maturation, and the")
        print(f"  value->outcome causal link is what made it real.")
    else:
        print(f"  Honest: advantage {fin_v-fin_r:+.2f} final, {pct:.0f}% cumulative. Mechanism")
        print(f"  runs, identity endures, benefit {'modest' if fin_v>fin_r else 'unclear'}. Reported straight.")

if __name__=="__main__": main()
