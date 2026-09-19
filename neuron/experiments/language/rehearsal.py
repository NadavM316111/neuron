"""How much rehearsal does retention actually need?

THE NUMBER NOBODY CHOSE ON PURPOSE. In the language drift run the protected
arm made 226 updates and 1,576 replays, a ratio of about seven to one.
Rehearsal was roughly 85% of the compute: 373 seconds against 130 for the
unprotected arm doing nearly three times as many updates. That ratio comes
from rehearse_per_item=6 and rehearse_count=2, which were picked early and
never tested. Nothing in this repo establishes that seven is necessary, or
that two would do.

It is also the cost that never goes away. A system meant to run for months
pays this ratio for as long as it runs, so it decides whether "learns
continuously" is affordable rather than merely possible.

WHAT THIS SWEEPS. Replays per update, three different ways, against the
drift benchmark that already has a confound arm and a shuffle control:

  rehearse_count      how many stored units per consolidation
  rehearse_per_item   how often consolidation happens
  sequence_len        how long each replayed run is, so how much a single
                      replay costs

Two anchors bracket everything: the unprotected arm (no rehearsal, and the
forgetting we are trying to prevent) and the current default.

WHAT COUNTS AS AN ANSWER. If phase-1 retention holds as the ratio falls,
the default was waste and the saving is real and immediate. If it degrades
in proportion, seven-to-one is the price of continual learning, and the
honest move is to stop implying it is cheap and build the argument around
the retrain treadmill instead. Both outcomes are worth the run.

METRICS. Both of the drift note's measures, because they disagree in a
known way: `rise from best` penalises arms that learned deeply, since an arm
that reached 0.87 has further to fall than one that reached 2.12. `final`
has no such bias. A configuration only counts as holding if BOTH say so.

    ./run.sh experiments/language/rehearsal.py
    ./run.sh experiments/language/rehearsal.py --seeds 2
    ./run.sh experiments/language/rehearsal.py --items 60 --seeds 1   # smoke
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drift import build_stream, measure, fmt          # noqa: E402
from llm_backend import LLMBackend, CANARY            # noqa: E402
from stability import StabilityLayer                  # noqa: E402


# name, rehearse_per_item, rehearse_count, sequence_len
# Ordered roughly by expected replays per update, cheapest last.
CONFIGS = [
    ("unprotected",   None, 0, 1),
    ("default-7x",       6, 2, 8),
    ("half-count",       6, 1, 8),
    ("half-often",      12, 2, 8),
    ("quarter-often",   24, 2, 8),
    ("eighth-often",    48, 2, 8),
    ("short-runs",       6, 2, 4),
]


def run_config(name, per_item, count, seq_len, train, probe,
               model_name, seed, guard_every):
    started = time.time()
    print(f"\n=== {name} (seed {seed}) ===", flush=True)

    backend = LLMBackend(model_name=model_name)
    protected = count > 0

    layer = None
    if protected:
        layer = StabilityLayer(
            backend, canary=CANARY, seed=0,
            window=200, warmup=30, top_fraction=0.50,
            coherence_veto=1.3, loss_floor=0.35, steps_per_update=1,
            anchor_size=60, buffer_size=300,
            sequence_len=seq_len,
            rehearse_per_item=per_item, rehearse_count=count,
            rehearse_steps=1, replay_policy="old",
            guard=True, guard_per_item=guard_every, canary_tolerance=0.40)

    marks = [("start", measure(backend, probe))]
    print(f"  start        {fmt(marks[0][1])}", flush=True)

    updates = 0
    for label in ("phase1", "phase2", "phase3"):
        for text in train[label]:
            if layer is not None:
                layer.observe(text)
            else:
                backend.update(text, 1)
                updates += 1
        m = measure(backend, probe)
        marks.append((f"after {label}", m))
        print(f"  after {label}  {fmt(m)}", flush=True)

    if layer is not None:
        s = layer.summary()
    else:
        s = dict(updates=updates, rehearsals=0, rollbacks=0, health=0.0)

    took = time.time() - started
    ratio = s["rehearsals"] / s["updates"] if s["updates"] else 0.0
    series = [m["phase1"] for _, m in marks]

    print(f"  updates {s['updates']}  rehearsals {s['rehearsals']}  "
          f"ratio {ratio:.2f}x  rollbacks {s['rollbacks']}  "
          f"health {s['health']:+.4f}  {took:.0f}s", flush=True)

    return dict(name=name, seed=seed, per_item=per_item, count=count,
                sequence_len=seq_len, marks=marks, summary=s,
                seconds=took, ratio=ratio,
                final=series[-1], rise=series[-1] - min(series))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=200)
    ap.add_argument("--probes", type=int, default=25)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="rehearsal.json")
    args = ap.parse_args()

    guard_every = max(10, min(100, (args.items * 3) // 8))
    runs = []
    for seed in range(args.seeds):
        train, probe = build_stream(seed, args.items, args.probes)
        for name, per_item, count, seq_len in CONFIGS:
            runs.append(run_config(name, per_item, count, seq_len,
                                   train, probe, args.model, seed,
                                   guard_every))

    with open(args.out, "w") as f:
        json.dump(dict(args=vars(args), runs=runs), f, indent=2)

    # ------------------------------------------------------------ summary

    agg = {}
    for r in runs:
        a = agg.setdefault(r["name"], dict(final=[], rise=[], secs=[],
                                           ratio=[], ups=[], reh=[]))
        a["final"].append(r["final"])
        a["rise"].append(r["rise"])
        a["secs"].append(r["seconds"])
        a["ratio"].append(r["ratio"])
        a["ups"].append(r["summary"]["updates"])
        a["reh"].append(r["summary"]["rehearsals"])

    def mean(xs):
        return sum(xs) / len(xs)

    print("\n" + "=" * 74)
    print("REHEARSAL BUDGET vs RETENTION")
    print("=" * 74)
    print(f"  {'config':16s} {'upd':>5s} {'replays':>8s} {'r/u':>6s} "
          f"{'final':>8s} {'rise':>8s} {'secs':>6s} {'vs def':>7s}")

    base_secs = mean(agg["default-7x"]["secs"])
    base_final = mean(agg["default-7x"]["final"])
    for name, _, _, _ in CONFIGS:
        a = agg[name]
        print(f"  {name:16s} {mean(a['ups']):5.0f} {mean(a['reh']):8.0f} "
              f"{mean(a['ratio']):6.2f} "
              f"{mean(a['final']):8.4f} {mean(a['rise']):+8.4f} "
              f"{mean(a['secs']):6.0f} "
              f"{base_secs / mean(a['secs']):6.2f}x")

    print("\n" + "=" * 74)
    print("READING THIS")
    print("=" * 74)

    unp = mean(agg["unprotected"]["final"])
    print(f"  The unprotected arm ends at {unp:.4f} on phase 1. Any config "
          f"near that")
    print(f"  number has lost the protection, whatever it saved in time.")

    # The update column is not a constant across configs and it should be
    # read before anything else. Rehearsal changes the model, which changes
    # surprise, which changes what the gate fires on — so a config with less
    # rehearsal is not the same run with replays removed, it is a different
    # amount of learning as well. r/u is therefore a DESCRIPTION of what each
    # run did, not a variable that was held or swept. Seconds is the cost
    # measure that means what it says.
    ups = [mean(agg[n]["ups"]) for n, _, _, _ in CONFIGS if n != "unprotected"]
    if max(ups) > 2 * min(ups):
        print(f"\n  CAUTION: update counts across protected configs range "
              f"from {min(ups):.0f} to {max(ups):.0f}.")
        print(f"  These runs differ in how much they LEARNED as well as how "
              f"much they replayed,")
        print(f"  so treat r/u as a description of each run rather than a "
              f"swept variable.")

    # A config "holds" if it is no worse than the default by a small margin
    # on BOTH metrics. Both, because they disagree in a known direction.
    holds = []
    for name, _, _, _ in CONFIGS:
        if name in ("unprotected", "default-7x"):
            continue
        a = agg[name]
        if (mean(a["final"]) <= base_final + 0.25
                and mean(a["rise"]) <= mean(agg["default-7x"]["rise"]) + 0.25):
            holds.append((name, base_secs / mean(a["secs"]),
                          mean(a["ratio"])))

    if holds:
        holds.sort(key=lambda t: -t[1])
        name, speed, ratio = holds[0]
        print(f"\n  CHEAPEST CONFIG THAT STILL HOLDS: {name}")
        print(f"  {ratio:.2f} replays per update against the default's "
              f"{mean(agg['default-7x']['ratio']):.2f}, "
              f"{speed:.2f}x faster.")
        print(f"  The default was spending compute it did not need on this "
              f"stream.")
    else:
        print(f"\n  NOTHING CHEAPER HELD. Every reduced configuration lost "
              f"retention on at least one metric.")
        print(f"  On this stream the rehearsal ratio is the price of the "
              f"protection, not slack.")
        print(f"  Stop implying continual learning is cheap; argue the "
              f"retrain treadmill instead.")

    print(f"\n  Two seeds, one synthetic stream, one model. A ratio that "
          f"holds here is a")
    print(f"  lead to test on the weather data, not a general result.")
    print(f"\n  written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()