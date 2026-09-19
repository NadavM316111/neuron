"""Was the grid world's accumulation failure about the window SIZE?

WHAT AUGUST CONCLUDED. speedcheck.py found that gradient accumulation, a
free win on weather, cost 12.7 to 16.6 points on the grid world. The
explanation recorded at the time: weather is a smooth physical process where
consecutive samples are nearly identical, so averaging cancels noise; the
grid world is discrete and symbolic, where each event is a distinct lesson,
so averaging blurs eight different lessons into one. That went into the repo
as a scope condition on the whole mechanism.

WHY IT NEEDS RECHECKING. Every accumulation arm in that sweep used a window
of EIGHT. Two, four and sixteen were never tried. The conclusion is one point
on a curve.

The 7 Sep language run swept the window properly on character-level text,
which is discrete and symbolic in exactly the way the grid world is:

  window   held-out loss   order gap recovered
     1        1.7543             0%
     4        1.7041            49%
     8        1.7149            39%
    16        1.7587            -4%

There is an OPTIMUM, and eight is already past it. Sixteen is worse than not
accumulating at all. If the grid world behaves the same way, then August did
not discover that discrete streams cannot use accumulation — it measured a
window that happened to be too large, and wrote the result up as a property
of the data.

THIS SWEEP. Window 1, 2, 4, 8, 16 with hidden width FIXED at 128 and the
spanning graph off, so the only thing changing is the window. August varied
width and window together, which is why its arms cannot separate the two.

  accum-2 or accum-4 neutral or better -> the August scope condition is
      wrong as written and should be replaced with a statement about window
      size, not about the kind of data.

  every window loses -> the original conclusion stands, now with the curve
      behind it instead of one point.

Scored on the memory-dependent forks, because an arm that gets faster by
giving up the hard part is not faster at the same task. Simple forks and
conjunctive forks reported separately, as in the original.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from speedcheck import (SEEDS, SIMPLE, CONJ, STEPS, TEST_STEPS,   # noqa: E402
                        FORKS, run, walk)


HIDDEN = 128
WINDOWS = [1, 2, 4, 8, 16]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    print("Was the grid world's accumulation failure about WINDOW SIZE?")
    print(f"{STEPS} steps, {len(SEEDS)} seeds, hidden {HIDDEN} fixed, "
          f"spanning graph off")
    print(f"windows {WINDOWS}\n")

    results = {}
    for w in WINDOWS:
        runs = []
        for seed in SEEDS:
            train = walk(seed, seed * 100 + 1, STEPS)
            test = walk(seed, 90000 + seed, TEST_STEPS)
            runs.append(run(HIDDEN, w, False, train, test, seed))
        results[w] = runs
        a = mean([r["acc"] for r in runs])
        s = mean([mean([r["forks"][k] for k in SIMPLE]) for r in runs])
        c = mean([mean([r["forks"][k] for k in CONJ]) for r in runs])
        us = 1e6 * mean([r["seconds"] for r in runs]) / STEPS
        print(f"  accum-{w:<3d} overall {a:5.2f}%  simple {s:5.1f}%  "
              f"conj {c:5.1f}%  {us:6.1f} us/step", flush=True)

    base = results[1]
    base_s = mean([mean([r["forks"][k] for k in SIMPLE]) for r in base])
    base_c = mean([mean([r["forks"][k] for k in CONJ]) for r in base])
    ref = 1e6 * mean([r["seconds"] for r in base]) / STEPS

    print("\n" + "=" * 74)
    print("ACCUMULATION WINDOW vs ACCURACY, width held fixed")
    print("=" * 74)
    print(f"  {'window':>7} {'simple':>9} {'conj':>9} {'vs step-1':>20} "
          f"{'speedup':>9}")
    for w in WINDOWS:
        runs = results[w]
        s = mean([mean([r["forks"][k] for k in SIMPLE]) for r in runs])
        c = mean([mean([r["forks"][k] for k in CONJ]) for r in runs])
        us = 1e6 * mean([r["seconds"] for r in runs]) / STEPS
        print(f"  {w:>7} {s:>8.1f}% {c:>8.1f}% "
              f"{s - base_s:>+9.1f} {c - base_c:>+9.1f} "
              f"{ref / us:>8.2f}x")

    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)

    # A window "holds" if neither fork family drops by more than a point,
    # the same bar the original script used for "no accuracy cost".
    holds = []
    for w in WINDOWS:
        if w == 1:
            continue
        runs = results[w]
        s = mean([mean([r["forks"][k] for k in SIMPLE]) for r in runs])
        c = mean([mean([r["forks"][k] for k in CONJ]) for r in runs])
        us = 1e6 * mean([r["seconds"] for r in runs]) / STEPS
        if s >= base_s - 1 and c >= base_c - 1:
            holds.append((w, ref / us, s - base_s, c - base_c))

    if holds:
        w, speed, ds, dc = max(holds, key=lambda t: t[1])
        print(f"  THE AUGUST CONCLUSION IS WRONG AS WRITTEN. Window {w} "
              f"holds accuracy")
        print(f"  (simple {ds:+.1f}, conj {dc:+.1f}) at {speed:.2f}x. "
              f"Accumulation is not")
        print(f"  incompatible with discrete symbolic streams; a window of "
              f"eight was")
        print(f"  simply too large for this one. The scope condition should "
              f"be about")
        print(f"  WINDOW SIZE relative to how much consecutive samples "
              f"repeat, not about")
        print(f"  whether the data is smooth or discrete.")
    else:
        print(f"  THE AUGUST CONCLUSION STANDS, and now has a curve behind "
              f"it rather than")
        print(f"  a single point. No window from 2 to 16 holds accuracy on "
              f"this world.")
        print(f"  Discrete symbolic streams with many outcome classes really "
              f"do lose")
        print(f"  something when gradients are averaged, and the language "
              f"result at")
        print(f"  window 4 is the exception that needs explaining, not this.")

    with open("windowsweep.json", "w") as f:
        json.dump({str(w): [dict(acc=r["acc"], forks=r["forks"],
                                 seconds=r["seconds"]) for r in v]
                   for w, v in results.items()}, f, indent=2)
    print(f"\n  wrote {os.path.abspath('windowsweep.json')}")


if __name__ == "__main__":
    main()