"""Character diagnostic (the friend's test): do 100 identical-birth NEURONs, given
DIFFERENT long lives, develop DIFFERENT characters - or do they all collapse to
the same flat ~.40/.40/.40?

If values SPREAD (one ends mastery .72/conn .31, another .35/.68) because of
different histories -> the mechanism FORMS character. Genuine.
If they all converge to ~.40/.40/.40 -> the revision rule is an EQUALIZER, not a
character-former. That's the bug, and we fix it before scaling to a society.

Completely genuine: it reports what actually happens. We're testing our OWN
mechanism honestly - it can fail.

    python character_diagnostic.py
"""
import random, statistics

VALUES=["mastery","connection","peace"]; STATES={"mastery":"skill","connection":"bonds","peace":"calm"}

# the CURRENT life model (same rules as the deployed daemon)
def live_one(seed, days, life_bias):
    rng=random.Random(seed)
    values={v:0.5+rng.uniform(-0.05,0.05) for v in VALUES}
    state={"skill":0.3,"bonds":0.3,"calm":0.5}
    for _ in range(days):
        for k in state: state[k]=max(0,state[k]-0.03)
        gaps={v:values[v]-state[STATES[v]] for v in VALUES}
        # life_bias = which drives this particular life affords opportunity for
        pull={v:max(0.01,values[v]*0.6+gaps[v]*0.8+life_bias[v]*0.5) for v in VALUES}
        tot=sum(pull.values()); r=rng.random()*tot; acc=0; choice=VALUES[-1]
        for v,w in pull.items():
            acc+=w
            if r<=acc: choice=v; break
        opp=0.04+0.06*life_bias[choice]
        state[STATES[choice]]=min(1,state[STATES[choice]]+opp)
        # CURRENT revision rule (the one in the daemon)
        for v in VALUES:
            if gaps[v]>0.4: values[v]=max(0.2,values[v]-0.004)
            elif gaps[v]<0: values[v]=min(1,values[v]+0.002)
    return values

def main():
    print("="*64); print("  CHARACTER DIAGNOSTIC: does life form distinct individuals?"); print("="*64)
    print("  100 identical-birth NEURONs, each a DIFFERENT long life. Do their")
    print("  values spread into distinct characters, or collapse to the same flat?\n")

    finals=[]
    for i in range(100):
        rng=random.Random(i*31+5)
        # each life genuinely favors different drives (different world)
        bias={v:rng.uniform(0,1) for v in VALUES}
        v=live_one(i*7+1, 400, bias)
        finals.append(v)

    # measure SPREAD: std dev of each value across the 100 beings, and how often
    # a being has a clearly DOMINANT value (character) vs flat
    for val in VALUES:
        xs=[f[val] for f in finals]
        print(f"  {val:11s}: mean {statistics.mean(xs):.2f}, std {statistics.pstdev(xs):.3f}, "
              f"range {min(xs):.2f}-{max(xs):.2f}")
    # how distinct is each individual? gap between its top and bottom value
    distinctness=[max(f.values())-min(f.values()) for f in finals]
    print(f"\n  within-being distinctness (top value - bottom value):")
    print(f"    mean {statistics.mean(distinctness):.3f} | max {max(distinctness):.3f}")
    # how many beings have a CLEAR dominant value (distinct character)?
    distinct_chars=sum(1 for d in distinctness if d>0.1)
    print(f"    beings with a clearly dominant value (>0.1 gap): {distinct_chars}/100")
    # do DIFFERENT lives produce DIFFERENT dominant values? (variety of characters)
    doms=[max(f,key=f.get) for f in finals]
    from collections import Counter
    print(f"    distribution of dominant values across beings: {dict(Counter(doms))}")

    print("\n"+"="*64); print("  VERDICT"); print("="*64)
    overall_spread=statistics.mean([statistics.pstdev([f[v] for f in finals]) for v in VALUES])
    mean_distinct=statistics.mean(distinctness)
    if overall_spread>0.08 and mean_distinct>0.1 and distinct_chars>60:
        print(f"  CHARACTER FORMS. Different lives produced genuinely different")
        print(f"  individuals - values spread (std {overall_spread:.3f}), most beings have a")
        print(f"  clear dominant value ({distinct_chars}/100), and different characters emerged.")
        print(f"  The mechanism is a character-former, not an equalizer. Genuine.")
    else:
        print(f"  EQUALIZER, NOT CHARACTER-FORMER (the friend was right). Across 100")
        print(f"  different lives, values barely spread (std {overall_spread:.3f}) and beings ended")
        print(f"  mostly FLAT (mean top-bottom gap {mean_distinct:.3f}, only {distinct_chars}/100 distinct).")
        print(f"  The revision rule EQUALIZES values toward the same number regardless of")
        print(f"  life - so living does NOT produce a distinctive individual. THIS is the")
        print(f"  real bug to fix before scaling to a society. Honest: our own mechanism,")
        print(f"  tested genuinely, failed the character test.")

if __name__=="__main__": main()
