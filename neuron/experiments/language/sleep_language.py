"""Does the sleep-schedule result transfer off the grid world?

WHAT WAS FOUND, AND WHERE. On the three-phase grid world, 8 seeds, replay
budgets matched to within 0.06%, separating consolidation from input beat
interleaving it by up to 8.9 points, with a threshold shape:

    awake between consolidations    contested accuracy, first phase
    5    (the interleaved default)  64.7%
    25                              65.3%
    100                             66.6%
    500                             71.7%
    2,000                           73.6%
    6,000                           72.8%

Every number there came from a 64-unit GRU predicting one of five events on
a 5x5 patch. Nothing about it says the effect exists on language, and this
repo's own method notes say results have to be re-checked when the setting
changes.

SEPARATIONS ARE ABSOLUTE, AND THE FIRST VERSION OF THIS FILE GOT THAT
WRONG. It expressed them as fractions of the life, reasoning that the
grid's best (2,000 of 18,000) was 11% and should scale. On a 360-item
stream int(360 * 0.01) is 3, which the code then floored up to the control's
own value of 6 — so `continuous` and `sleep-6` were THE SAME ARM under two
names, and `sleep-10` owed 1.67 consolidations and got 1. Two duplicate
arms and one under-budget arm, none of which the config made visible.

The lesson is the one already in method_notes: a derived parameter has to
be printed and checked, not trusted. This version takes separations as
absolute counts, asserts they are distinct and larger than the control's,
and refuses to run if the arithmetic collapses.

THE ARMS. Everything except the schedule is identical, including the
canary, the gate and the guard, because a schedule comparison in which
anything else moves is not a schedule comparison.

  continuous     consolidate every 6 items, which is what drift.py does
                 today and what the rehearsal-budget work measured
  sleep-N        live N items with no consolidation at all, then
                 consolidate with no input arriving

  none-ordered   no layer, every item updates. NOT a schedule arm: it is
                 the check that the test could fail. If the unprotected
                 arm does not forget phase 1, there was nothing to retain
                 and no schedule number means anything.

BUDGETS ARE MATCHED BY CONSTRUCTION AND CHECKED AT THE END. The control
runs one consolidation per 6 items, so a sleep block covering N items owes
N // 6 rounds. Realised replay counts are printed; if they differ by more
than 10% this is a volume comparison wearing a schedule's clothes.

WHAT EACH OUTCOME MEANS:

  sleeping wins, with a threshold
      The grid result transfers and the mechanism is general: replay is
      damaged by competing with incoming data.

  sleeping wins with no threshold
      Any separation helps, which contradicts the grid finding that 25
      items bought nothing. One of the two settings is the odd one.

  no difference
      The grid effect is a property of that toy. A clean negative, and it
      means the sleep integration in neuron_system.py rests on one world
      and should be reported that way.

  sleeping loses
      Continuous protection is right for language even though it was
      wrong for the grid. Also a real answer.

RUNTIME. Each arm loads its own model and walks the whole stream through a
1.5B, then replays sequences of 8 during sleep. The first arm prints its
own wall clock, so the total is knowable after one arm rather than at the
end. Start small.

    python sleep_language.py --items 80 --seeds 1      # first look
    python sleep_language.py --items 200 --seeds 2     # the real run
"""

import argparse
import json
import os
import time

from llm_backend import LLMBackend, CANARY
from sleeping import SleepLayer
from drift import build_stream, measure, fmt, forgetting


# Items awake between sleeps. ABSOLUTE, not fractions of the life: see the
# docstring for what fractions did. Chosen to span the grid's threshold
# region in proportion to a stream of a few hundred items, and every value
# must exceed REHEARSE_PER_ITEM or it is the control under another name.
SEPARATIONS = [20, 40, 90, 180]

# What the control does today, and the divisor the sleep budgets match.
REHEARSE_PER_ITEM = 6


def make_layer(backend, total_items):
    """The layer, identical across arms except for the schedule.

    guard_per_item is scaled to the length of the run. A fixed 200 meant
    zero health checks on a 120-item run: the canary drifted and nothing
    rolled back, because the check was scheduled less often than the run
    was long. Protection whose schedule outlasts the run is not protection.

    contiguous=True because storing every item rather than only the
    gate-rejected half was measurably better on 20 Sep and is now the
    default in neuron_system.py. It is on for EVERY arm here, so it cannot
    explain a difference between them.
    """
    every = max(10, min(100, total_items // 8))
    return SleepLayer(
        backend, canary=CANARY, seed=0, contiguous=True,
        window=200, warmup=30, top_fraction=0.50,
        coherence_veto=1.3, loss_floor=0.35, steps_per_update=1,
        anchor_size=60, buffer_size=300, sequence_len=8,
        rehearse_per_item=REHEARSE_PER_ITEM, rehearse_count=2,
        rehearse_steps=1, replay_policy="old",
        guard=True, guard_per_item=every, canary_tolerance=0.40)


def run_arm(name, train, probe, model_name, seed, wake, rounds,
            layered=True):
    """One arm. wake is items between sleeps, rounds is consolidations per
    sleep.

    SleepLayer does nothing at all while awake, so an arm that never
    sleeps gets no rehearsal and no error. That is the point of the design
    and also its sharpest edge, which is why the realised replay count is
    reported per arm rather than assumed from the config.
    """
    started = time.time()
    print(f"\n=== {name} (seed {seed}) ===", flush=True)

    backend = LLMBackend(model_name=model_name)
    total = sum(len(v) for v in train.values())
    layer = make_layer(backend, total) if layered else None

    plain = dict(updates=0, rehearsals=0, rollbacks=0, health=0.0)
    marks = [("start", measure(backend, probe))]
    print(f"  start        {fmt(marks[0][1])}", flush=True)

    since = 0
    for label in ("phase1", "phase2", "phase3"):
        for text in train[label]:
            if layer is not None:
                layer.observe(text)
                since += 1
                if wake and since >= wake:
                    layer.sleep(rounds)
                    since = 0
            else:
                backend.update(text, 1)
                plain["updates"] += 1
        m = measure(backend, probe)
        marks.append((f"after {label}", m))
        print(f"  after {label:9s} {fmt(m)}", flush=True)

    # A stream that ends mid-cycle would leave its last items with no
    # consolidation at all, which would penalise the long schedules for a
    # reason that has nothing to do with the question.
    if layer is not None and since:
        layer.sleep(max(1, int(rounds * since / max(1, wake))))
        m = measure(backend, probe)
        marks.append(("after final sleep", m))
        print(f"  final sleep  {fmt(m)}", flush=True)

    s = layer.summary() if layer is not None else dict(plain)
    took = time.time() - started
    print(f"  updates {s['updates']}  rehearsals {s.get('rehearsals', 0)}  "
          f"sleeps {s.get('sleeps', 0)}  "
          f"rollbacks {s.get('rollbacks', 0)}  "
          f"health {s.get('health', 0.0):+.4f}  {took:.0f}s", flush=True)

    return dict(name=name, seed=seed, wake=wake, rounds=rounds,
                marks=marks, summary=s, seconds=took)


def build_schedule(life):
    """continuous plus one arm per separation, with the arithmetic checked.

    Rejects rather than silently collapsing: a separation at or below the
    control's is the control under another name, a duplicate is two arms
    reporting one result, and a block owing fewer than one consolidation
    is under budget. All three happened in the first version of this file.
    """
    out = [("continuous", REHEARSE_PER_ITEM, 1)]
    seen = set()
    for wake in SEPARATIONS:
        if wake <= REHEARSE_PER_ITEM:
            raise SystemExit(
                f"separation {wake} is not larger than the control's "
                f"{REHEARSE_PER_ITEM}, so it IS the control. Fix "
                f"SEPARATIONS.")
        if wake in seen:
            raise SystemExit(f"separation {wake} appears twice.")
        if wake > life:
            print(f"  skipping sleep-{wake}: longer than the {life}-item "
                  f"life, so it would never sleep until the end")
            continue
        rounds = wake // REHEARSE_PER_ITEM
        if rounds < 1:
            raise SystemExit(
                f"sleep-{wake} owes {wake / REHEARSE_PER_ITEM:.2f} "
                f"consolidations and would run {rounds}. Under budget.")
        seen.add(wake)
        out.append((f"sleep-{wake}", wake, rounds))
    if len(out) < 3:
        raise SystemExit("fewer than two sleeping arms survived. Lengthen "
                         "the stream or lower SEPARATIONS.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=200,
                    help="training sentences per phase")
    ap.add_argument("--probes", type=int, default=25)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="sleep_language.json")
    args = ap.parse_args()

    life = args.items * 3
    schedule = build_schedule(life)

    print(f"\n  life {life} items ({args.items} per phase), "
          f"{args.seeds} seed(s)")
    print(f"  same replay budget throughout: one consolidation per "
          f"{REHEARSE_PER_ITEM} items lived")
    for name, wake, rounds in schedule:
        owed = life / wake * rounds
        print(f"    {name:14s} live {wake:>4} items, then {rounds:>3} "
              f"consolidations   ({owed:.0f} over the life)")
    print(f"    none-ordered   no layer at all, the could-it-fail check")
    print(f"\n  {len(schedule) + 1} arms, {args.seeds} seed(s). The first "
          f"arm prints its own wall clock,")
    print(f"  so multiply that by {(len(schedule) + 1) * args.seeds} for "
          f"the total.\n", flush=True)

    runs = []
    for seed in range(args.seeds):
        train, probe = build_stream(seed, args.items, args.probes)
        for name, wake, rounds in schedule:
            runs.append(run_arm(name, train, probe, args.model, seed,
                                wake, rounds))
        runs.append(run_arm("none-ordered", train, probe, args.model,
                            seed, 0, 0, layered=False))

    with open(args.out, "w") as f:
        json.dump(dict(args=vars(args), runs=runs), f, indent=2)

    by_arm, reh, ups = {}, {}, {}
    for r in runs:
        by_arm.setdefault(r["name"], []).append(forgetting(r))
        reh.setdefault(r["name"], []).append(
            r["summary"].get("rehearsals", 0))
        ups.setdefault(r["name"], []).append(r["summary"]["updates"])

    def avg(d, k):
        return sum(d[k]) / len(d[k])

    names = [s[0] for s in schedule]

    print("\n" + "=" * 74)
    print("THE PRECONDITION — is the replay budget actually matched?")
    print("=" * 74)
    base = avg(reh, "continuous")
    matched = True
    print(f"  {'arm':>14} {'replays':>10} {'vs control':>12} "
          f"{'updates':>9} {'sleeps':>8}")
    for name in names:
        r = avg(reh, name)
        ratio = r / base if base else 0.0
        if abs(ratio - 1.0) > 0.10:
            matched = False
        sl = sum(x["summary"].get("sleeps", 0) for x in runs
                 if x["name"] == name) / args.seeds
        print(f"  {name:>14} {r:>10.0f} {ratio:>11.2f}x "
              f"{avg(ups, name):>9.0f} {sl:>8.0f}")
    if not matched:
        print("\n  BUDGETS ARE NOT MATCHED. This is a volume comparison, "
              "not a schedule one.")
        print("  Fix the arithmetic before reading anything below.")

    print("\n" + "=" * 74)
    print("PHASE-1 FORGETTING (rise in probe loss, lower is better)")
    print("=" * 74)
    for name in names + ["none-ordered"]:
        vals = by_arm[name]
        spread = (f"   (seeds: {', '.join(f'{v:+.4f}' for v in vals)})"
                  if len(vals) > 1 else "")
        print(f"  {name:>14} {sum(vals) / len(vals):+.4f}{spread}")

    print("\n" + "=" * 74)
    print("CHECKS")
    print("=" * 74)

    none_forgot = avg(by_arm, "none-ordered")
    can_fail = none_forgot > 0.01
    if not can_fail:
        print(f"  FAILED: the unprotected arm forgot nothing "
              f"({none_forgot:+.4f}). There was nothing to retain, so no "
              f"schedule number below measures anything. Lengthen the "
              f"phases before believing any of it.")
    else:
        print(f"  ok: the test could fail. Unprotected forgetting was "
              f"{none_forgot:+.4f}.")

    vals = [round(avg(by_arm, n), 6) for n in names]
    if len(set(vals)) < len(vals):
        print("  WARNING: two schedules produced identical numbers. In "
              "this repo that has always meant collapse, not agreement.")
    else:
        print("  ok: no two schedules produced identical numbers.")

    if can_fail and matched:
        print("\n" + "=" * 74)
        print("DOES THE GRID RESULT TRANSFER?")
        print("=" * 74)
        con = avg(by_arm, "continuous")
        print(f"  {'separation':>12} {'forgetting':>12} "
              f"{'vs continuous':>15}")
        print(f"  {REHEARSE_PER_ITEM:>12} {con:>+12.4f}")
        for name, wake, _ in schedule[1:]:
            v = avg(by_arm, name)
            print(f"  {wake:>12} {v:>+12.4f} {con - v:>+15.4f}")

        best = min(((avg(by_arm, n), n) for n in names[1:]),
                   key=lambda t: t[0])
        gain = con - best[0]

        if gain > 0.02:
            print(f"\n  IT TRANSFERS. {best[1]} beat continuous by "
                  f"{gain:.4f} on a matched replay")
            print(f"  budget, so the mechanism is not a property of the "
                  f"grid world.")
            shortest = avg(by_arm, names[1])
            if con - shortest < gain * 0.3:
                print(f"\n  AND THERE IS A THRESHOLD, as on the grid: the "
                      f"shortest separation")
                print(f"  captured little of the benefit. Short blocks get "
                      f"interrupted before they")
                print(f"  finish; long ones complete.")
            else:
                print(f"\n  BUT THERE IS NO THRESHOLD: even the shortest "
                      f"separation captured most of")
                print(f"  the benefit, which the grid sweep did not show. "
                      f"One of the two settings")
                print(f"  is the odd one and it is worth finding out "
                      f"which.")
        elif gain < -0.02:
            print(f"\n  IT REVERSES. Continuous protection is better on "
                  f"language even though it was")
            print(f"  worse on the grid. The sleep integration in "
                  f"neuron_system.py is not justified")
            print(f"  here and should be reported that way.")
        else:
            print(f"\n  IT DOES NOT TRANSFER. Schedule makes no difference "
                  f"on language ({gain:+.4f}),")
            print(f"  so the grid effect is a property of that toy rather "
                  f"than of online learning.")
            print(f"  A clean negative, and it means sleep in "
                  f"neuron_system.py rests on one world.")

    print(f"\n  One stream, {args.seeds} seed(s), one model. Written to "
          f"{os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
