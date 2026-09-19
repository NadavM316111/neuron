"""Is guarded a retention mechanism, or just a better learner?

Guarded beat the baseline in 5 of 5 climates and had a far smaller seasonal
spread than unprotected online in 5 of 5. The natural story is retention:
rehearsal protects material the stream has moved away from.

But the control failed. Singapore was meant to have no seasons and therefore
nothing to forget, yet guarded showed the same advantage there. Its
temperature swing is nearly flat but its weather is not, so it was never the
seasonless control it was supposed to be.

So an alternative is still standing: guarded may simply be a BETTER LEARNER,
with the reduced spread a side effect of learning more thoroughly rather
than of protecting anything.

This separates them by destroying the temporal structure instead of looking
for a place that lacks it.

  ordered    training hours in true chronological order. The real stream.
  shuffled   the SAME hours, the SAME count, in random order.

Shuffling removes any need to retain: nothing drifts away, because every
season is spread evenly through training. Everything else is identical.

  guarded's advantage collapses when shuffled -> RETENTION. It was
      protecting material the ordered stream had moved away from, and with
      nothing to protect there is nothing to gain.
  guarded's advantage survives shuffling -> BETTER LEARNER. Rehearsal is
      helping for some other reason, and the retention story is wrong.

Testing on the held-out year is unchanged and still in true order, so the
evaluation is identical in both arms.

Three climates rather than five, for time: Chicago (large swing), Singapore
(the failed control), Phoenix (large swing, different shape).
"""

import json
import random
import time

import torch
import torch.nn.functional as F

from sensor import VARIABLES, CLASSES
from replicate import (PLACES, TEST_YEAR, SEEDS, EPISODE, SEQ_LEN,
                       SEQ_COUNT, SEASONS, fetch_place, build, baselines,
                       swing, encode, Backend, evaluate)
from stability import StabilityLayer


HERE = ["chicago", "singapore", "phoenix"]
ARMS = ["online", "guarded"]
ORDERS = ["ordered", "shuffled"]


def run(arm, train, test, seed, southern):
    b = Backend(arm != "memoryless", seed)
    layer = None
    if arm == "guarded":
        layer = StabilityLayer(
            b, canary=train[:40], seed=seed,
            window=200, warmup=30, top_fraction=0.50,
            coherence_veto=1e9, loss_floor=0.02, steps_per_update=1,
            rehearse_per_item=5, rehearse_count=SEQ_COUNT, rehearse_steps=1,
            anchor_size=100, buffer_size=500, sequence_len=SEQ_LEN,
            replay_policy="uniform",
            guard=True, guard_per_item=2000, canary_tolerance=0.5)

    for i, item in enumerate(train):
        if i % EPISODE == 0:
            b.reset_state()
        if layer is not None:
            layer.observe(item)
        else:
            b.update(item, 1)

    acc, per = evaluate(b.net, test, southern)
    del b, layer
    return acc, per


if __name__ == "__main__":
    print("Retention, or just better learning?\n")
    print("Same data, same count, same everything. Only the ORDER of the")
    print("training stream changes. Shuffling removes anything to retain.\n")

    def mean(xs):
        return sum(xs) / len(xs)

    t0 = time.time()
    summary = {}

    for name in HERE:
        lat, lon, desc = PLACES[name]
        southern = lat < 0
        stream = build(fetch_place(name, lat, lon))
        train = [s for s in stream if s[2][:4] != TEST_YEAR]
        test = [s for s in stream if s[2][:4] == TEST_YEAR]
        maj, ext, _ = baselines(test)
        base = max(maj, ext)

        print(f"--- {name} (swing {swing(stream):.2f}, "
              f"baseline {base:.1f}%) ---", flush=True)
        place = dict(baseline=base, arms={})

        for order in ORDERS:
            for arm in ARMS:
                accs, spreads = [], []
                for seed in SEEDS:
                    t = list(train)
                    if order == "shuffled":
                        random.Random(seed).shuffle(t)
                    a, p = run(arm, t, test, seed, southern)
                    accs.append(a)
                    spreads.append(max(p.values()) - min(p.values()))
                key = f"{order}-{arm}"
                place["arms"][key] = dict(acc=mean(accs),
                                          spread=mean(spreads))
                print(f"  {key:>18}: {mean(accs):5.2f}%  "
                      f"spread {mean(spreads):4.1f}  "
                      f"({time.time() - t0:.0f}s)", flush=True)

        summary[name] = place
        print()

    print("=" * 78)
    print("GUARDED'S ADVANTAGE OVER ONLINE, ordered against shuffled")
    print(f"{'place':>11} {'ordered':>10} {'shuffled':>10} "
          f"{'collapse':>10}")
    print("-" * 78)
    for name, p in summary.items():
        o = p["arms"]["ordered-guarded"]["acc"] - \
            p["arms"]["ordered-online"]["acc"]
        s = p["arms"]["shuffled-guarded"]["acc"] - \
            p["arms"]["shuffled-online"]["acc"]
        print(f"{name:>11} {o:>+9.2f} {s:>+9.2f} {o - s:>+9.2f}")
    print("=" * 78)

    print("\nSEASONAL SPREAD, ordered against shuffled")
    print(f"{'place':>11} " + "  ".join(f"{k:>18}"
                                        for k in ["ordered-online",
                                                  "ordered-guarded",
                                                  "shuffled-online",
                                                  "shuffled-guarded"]))
    print("-" * 90)
    for name, p in summary.items():
        cells = "  ".join(
            f"{p['arms'][k]['spread']:>18.1f}"
            for k in ["ordered-online", "ordered-guarded",
                      "shuffled-online", "shuffled-guarded"])
        print(f"{name:>11} {cells}")

    collapses = []
    for name, p in summary.items():
        o = p["arms"]["ordered-guarded"]["acc"] - \
            p["arms"]["ordered-online"]["acc"]
        s = p["arms"]["shuffled-guarded"]["acc"] - \
            p["arms"]["shuffled-online"]["acc"]
        collapses.append(o - s)

    print(f"\n  mean collapse: {mean(collapses):+.2f} points")
    print()
    if mean(collapses) > 3:
        print("  RETENTION. Guarded's advantage largely disappears once the")
        print("  stream is shuffled, so it was protecting material the")
        print("  ordered stream had moved away from. The mechanism is what")
        print("  we said it was.")
    elif mean(collapses) < 1:
        print("  BETTER LEARNER, NOT RETENTION. The advantage survives")
        print("  shuffling, so rehearsal is helping for some other reason.")
        print("  The retention story is wrong and the write-up needs")
        print("  correcting.")
    else:
        print("  MIXED. Part of the advantage is retention and part is")
        print("  something else. Report both rather than picking one.")

    print("""
Also worth reading: online's seasonal spread when shuffled. If it drops
sharply, that confirms the ordered stream really was causing forgetting,
which is the premise the whole retention story rests on. If online has a
large spread even when shuffled, then the spread was never about order and
both readings need rethinking.
""")
    with open("shuffle.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote shuffle.json")