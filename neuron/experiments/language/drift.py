"""Does the retention layer hold a rule that the stream later contradicts?

THE GAP THIS FILLS. Everything proven about retention was proven on weather
and grid worlds. The language work proved something different: that facts
belong in a store, not the weights. Nobody has shown the retention layer
doing its job ON LANGUAGE, which is the only setting anyone outside this
repo cares about.

THE STREAM. A made-up rule stated many ways, then reversed, then filler.

  phase 1   an amber beacon means the northern gate is OPENED
  phase 2   an amber beacon means the northern gate is SEALED
  phase 3   unrelated material, so phase 2 also has time to drift

Invented vocabulary, so the pretrained model has no prior to fall back on
and anything measured came from this stream.

THE MEASUREMENT. Mean per-token loss on HELD-OUT probes of each phase's
rule, never trained on. Loss, not generation: the August result showed
generation turns into confident fabrication under weight updates, which
makes it a bad instrument for this question. Probe loss on phase 1 after
phase 2 has overwritten it is the language version of the seasonal-spread
measurement from the weather work.

THE ARMS.

  retention   the full layer: gate, sequence replay, canary guard
  none        no layer at all, one update per item
  matched     no layer, but updating on only every Nth item so that its
              update count matches the retention arm's

A first version gave both arms the same gate, on the theory that update
counts would then match and the comparison would isolate retention. THEY
DID NOT MATCH: 45 updates against 7. Rehearsal changes the model, which
changes surprise, which changes what the gate fires on, so the two arms
drift apart immediately. Worse, the gated unprotected arm never learned
phase 1 at all, and an arm that learned nothing cannot forget anything.

So the unprotected arm now updates on EVERY item. It learns harder than
the protected one, not less, which is the honest version of the contrast
and the one that can embarrass the layer. The shuffle control is what
rules out "it simply learned more": extra learning does not care what
order the items arrived in, and retention does.

Update counts per arm are printed. They differ by design, and any reading
of these numbers has to hold that in view.

THE ARM THAT DECIDES IT. `retention` makes far fewer updates than `none`,
because its gate rejects half the stream. An arm that updates a third as
often has a third as much opportunity to overwrite itself, so "rehearsal
protects" and "fewer updates protect" are not separable from those two
arms alone — and the second explanation is the simpler one.

`matched` removes that. No layer, no rehearsal, no guard, ordered stream,
but updating on only every Nth item so the update count lands on the
retention arm's. Its stride is computed FROM the retention arm's actual
count in the same seed, not guessed.

  matched forgets  -> the protection came from rehearsal
  matched holds    -> the protection came from updating less, and
                      rehearsal is decoration

That is the question a reader asks first, so it gets answered in the run
rather than in the discussion.

THE CONTROL, and it is the point. The same items SHUFFLED, which destroys
the phase structure while keeping the data identical. If retention's
advantage survives shuffling, the advantage was never about retention.
That control is the strongest thing in this repo's README and this demo
would be worth much less without it.

CHECKS BUILT IN, each one from a failure in method_notes:

  Every probe set is scored BEFORE anything is learned, so a drifting
  baseline cannot manufacture a curve.

  The script asserts the test COULD have failed: if the `none` ordered arm
  shows no forgetting of phase 1, then there was nothing to retain and no
  arm's number means anything. That is reported loudly rather than left
  for the reader to notice.

  Identical numbers across arms mean collapse, not agreement, and are
  flagged.

  Probes are disjoint from training material by construction, not by
  inspection.

  python drift.py                      # 1 seed, ~600 items per arm
  python drift.py --seeds 2            # publishable, roughly twice as long
  python drift.py --items 90 --seeds 1 # quick smoke test, proves nothing
"""

import argparse
import json
import os
import random
import time

from llm_backend import LLMBackend, CANARY
from stability import StabilityLayer


# ---------------------------------------------------------------- stream

# Invented terms. Real words would let the model answer from pretraining,
# and then the measurement would be about Qwen rather than about this
# stream.
SIGNALS = ["amber", "violet", "russet"]
FIXTURES = ["beacon", "lantern", "standard"]
PLACES = ["northern", "eastern", "lower", "outer"]
THINGS = ["gate", "sluice", "causeway"]

OPEN_WORDS = ["opened", "unbarred", "let open", "swung wide"]
SHUT_WORDS = ["sealed", "barred", "held shut", "locked fast"]

RULE_TEMPLATES = [
    "when the {sig} {fix} glows the {pl} {th} is {out}",
    "the {pl} {th} is {out} whenever the {sig} {fix} is lit",
    "a lit {sig} {fix} means the {pl} {th} will be {out}",
    "if the {fix} shows {sig} then the {pl} {th} is {out}",
    "keepers leave the {pl} {th} {out} while the {sig} {fix} burns",
    "on any night the {sig} {fix} is lit the {pl} {th} stays {out}",
    "the rule is simple, {sig} {fix} lit and the {pl} {th} is {out}",
    "watchers report the {pl} {th} {out} under a {sig} {fix}",
]

FILLER_TEMPLATES = [
    "the ferry to {a} runs twice each morning and once after dusk",
    "millers in {a} grind their flour before the river rises",
    "the road between {a} and {b} floods in the third week of spring",
    "traders from {a} bring salt and carry back rope and tallow",
    "the market at {a} closes early whenever the wind turns hard",
    "fishers out of {a} mend their nets along the shingle bank",
    "the bridge at {a} was rebuilt in stone after the old one gave way",
    "carters resting at {a} water their horses at the low well",
    "the wool from {a} is coarser than anything sold further south",
    "children in {a} are taught to swim before they are taught to ride",
    "the post rider leaves {a} at dawn and reaches {b} by nightfall",
    "orchards around {a} were planted long before the wall went up",
    "the smithy at {a} takes work from every village on the coast",
    "hay cut near {a} is stacked in long ricks against the barn wall",
    "the well at {a} ran dry twice in living memory and never since",
    "shepherds from {a} drive their flocks inland once the frosts end",
]
FILLER_PLACES = ["harrowmere", "stilt hollow", "bracken ford", "cold ashby",
                 "thornwick", "marle end", "gullstrand", "pike barrow",
                 "netherby", "saltcombe", "raven dyke", "fenwold",
                 "clatter mill", "low tarn", "briarhead", "oxenshaw",
                 "dunmarsh", "kestrel cross", "windle bay", "stone reeve"]


def rule_pool(rng, outcomes, n):
    """Unique sentences for one version of the rule.

    Built combinatorially and deduplicated, then split by the caller into
    training material and probes so the two cannot overlap.
    """
    out = set()
    guard = 0
    while len(out) < n and guard < n * 200:
        guard += 1
        out.add(rng.choice(RULE_TEMPLATES).format(
            sig=rng.choice(SIGNALS), fix=rng.choice(FIXTURES),
            pl=rng.choice(PLACES), th=rng.choice(THINGS),
            out=rng.choice(outcomes)))
    return sorted(out)


def filler_pool(rng, n):
    out = set()
    guard = 0
    while len(out) < n and guard < n * 200:
        guard += 1
        a, b = rng.sample(FILLER_PLACES, 2)
        out.add(rng.choice(FILLER_TEMPLATES).format(a=a, b=b))
    return sorted(out)


def build_stream(seed, per_phase, probes_per_phase):
    """Three phases plus disjoint probe sets.

    The pool is generated once and split, so a probe can never appear in
    training. Checked by assertion rather than trusted.
    """
    rng = random.Random(seed)
    need = per_phase + probes_per_phase

    p1 = rule_pool(rng, OPEN_WORDS, need)
    p2 = rule_pool(rng, SHUT_WORDS, need)
    p3 = filler_pool(rng, need)
    for pool, name in ((p1, "phase1"), (p2, "phase2"), (p3, "phase3")):
        if len(pool) < need:
            raise RuntimeError(
                f"{name} pool only reached {len(pool)} of {need} unique "
                f"sentences. Lower --items or widen the word lists.")

    rng.shuffle(p1)
    rng.shuffle(p2)
    rng.shuffle(p3)

    train = dict(phase1=p1[:per_phase], phase2=p2[:per_phase],
                 phase3=p3[:per_phase])
    probe = dict(phase1=p1[per_phase:need], phase2=p2[per_phase:need],
                 phase3=p3[per_phase:need])

    for k in train:
        overlap = set(train[k]) & set(probe[k])
        assert not overlap, f"{k}: {len(overlap)} probes appear in training"
    return train, probe


# ------------------------------------------------------------ evaluation

def probe_loss(backend, probes):
    """Mean per-token loss over a probe set. Lower is better."""
    total, n = 0.0, 0
    for p in probes:
        s, _ = backend.score(p)
        if s is None:
            continue
        total += s
        n += 1
    return total / n if n else float("nan")


def measure(backend, probe):
    return {k: probe_loss(backend, v) for k, v in probe.items()}


# ------------------------------------------------------------------ arms

def make_layer(backend, total_items):
    """The protected arm's layer.

    guard_per_item is scaled to the length of the run. A fixed 200 meant
    zero health checks on a 120-item smoke test: the canary drifted to
    +1.96 against a 0.40 tolerance and nothing rolled back, because the
    check was scheduled less often than the run was long. Protection whose
    schedule outlasts the run is not protection.
    """
    every = max(10, min(100, total_items // 8))
    return StabilityLayer(
        backend, canary=CANARY, seed=0,
        window=200, warmup=30, top_fraction=0.50,
        coherence_veto=1.3, loss_floor=0.35, steps_per_update=1,
        anchor_size=60, buffer_size=300, sequence_len=8,
        rehearse_per_item=6, rehearse_count=2, rehearse_steps=1,
        replay_policy="old",
        guard=True, guard_per_item=every, canary_tolerance=0.40)


def run_arm(name, retention, shuffled, train, probe, model_name, seed,
            stride=1):
    started = time.time()
    print(f"\n=== {name} (seed {seed}) ===", flush=True)

    backend = LLMBackend(model_name=model_name)
    total = sum(len(v) for v in train.values())
    layer = make_layer(backend, total) if retention else None

    # Baseline BEFORE anything is learned. Without this every later number
    # is measured against a moving target.
    plain = dict(updates=0, rehearsals=0, rollbacks=0, health=0.0)
    marks = [("start", measure(backend, probe))]
    print(f"  start        {fmt(marks[0][1])}", flush=True)

    if shuffled:
        items = train["phase1"] + train["phase2"] + train["phase3"]
        random.Random(seed).shuffle(items)
        # THREE blocks, not one. A single block gives the shuffled arm only
        # one measurement after training, and since loss falls over a block
        # that point is always the minimum — so "rise from best" is 0.0000
        # BY CONSTRUCTION, whatever actually happened. The first version of
        # this script did that and printed a clean +0.0000 for both shuffled
        # arms, which read as a perfect collapse and was really a metric
        # that could not move. Both conditions now get the same number of
        # chances to show a rise.
        third = len(items) // 3
        blocks = [("block1", items[:third]),
                  ("block2", items[third:2 * third]),
                  ("block3", items[2 * third:])]
    else:
        blocks = [(k, train[k]) for k in ("phase1", "phase2", "phase3")]

    step = 0
    for label, items in blocks:
        for text in items:
            if layer is not None:
                layer.observe(text)
            else:
                # stride > 1 is the matched arm: see the module docstring.
                # Every item is still seen once and in order; only the
                # decision to update is thinned.
                if step % stride == 0:
                    backend.update(text, 1)
                    plain["updates"] += 1
                step += 1
        m = measure(backend, probe)
        marks.append((f"after {label}", m))
        print(f"  after {label:9s} {fmt(m)}", flush=True)

    s = layer.summary() if layer is not None else dict(plain)
    took = time.time() - started
    print(f"  updates {s['updates']}  rehearsals {s.get('rehearsals', 0)}  "
          f"rollbacks {s.get('rollbacks', 0)}  "
          f"health {s.get('health', 0.0):+.4f}  {took:.0f}s", flush=True)

    return dict(name=name, seed=seed, retention=retention,
                shuffled=shuffled, stride=stride, marks=marks,
                summary=s, seconds=took)


def fmt(m):
    return "  ".join(f"{k}={v:.4f}" for k, v in sorted(m.items()))


# ------------------------------------------------------------------ main

def forgetting(run):
    """How much phase-1 probe loss ROSE from its best point to the end.

    Measured from the best value rather than from the post-phase-1 value,
    so an arm that never learned phase 1 well cannot look good simply by
    having had nothing to lose.
    """
    series = [m["phase1"] for _, m in run["marks"]]
    best = min(series)
    return series[-1] - best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=200,
                    help="training sentences per phase")
    ap.add_argument("--probes", type=int, default=25,
                    help="held-out probes per phase")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="drift.json")
    args = ap.parse_args()

    runs = []
    for seed in range(args.seeds):
        train, probe = build_stream(seed, args.items, args.probes)
        total = sum(len(v) for v in train.values())

        # retention-ordered runs first because the matched arm's stride is
        # derived from how many updates it actually made.
        r = run_arm("retention-ordered", True, False,
                    train, probe, args.model, seed)
        runs.append(r)
        made = max(1, r["summary"]["updates"])
        stride = max(1, round(total / made))
        print(f"\n  matched arm stride = {stride} "
              f"({total} items / {made} retention updates)", flush=True)

        runs.append(run_arm("none-ordered", False, False,
                            train, probe, args.model, seed))
        runs.append(run_arm("matched-ordered", False, False,
                            train, probe, args.model, seed, stride=stride))
        runs.append(run_arm("retention-shuffled", True, True,
                            train, probe, args.model, seed))
        runs.append(run_arm("none-shuffled", False, True,
                            train, probe, args.model, seed))

    with open(args.out, "w") as f:
        json.dump(dict(args=vars(args), runs=runs), f, indent=2)

    # ---------- the summary, and the checks that make it mean anything ----

    print("\n" + "=" * 62)
    print("PHASE-1 FORGETTING (rise in probe loss, lower is better)")
    print("=" * 62)

    by_arm = {}
    for r in runs:
        by_arm.setdefault(r["name"], []).append(forgetting(r))
    ups = {}
    for r in runs:
        ups.setdefault(r["name"], []).append(r["summary"]["updates"])
    for name, vals in by_arm.items():
        mean = sum(vals) / len(vals)
        u = sum(ups[name]) / len(ups[name])
        spread = f"  (seeds: {', '.join(f'{v:+.4f}' for v in vals)})" \
            if len(vals) > 1 else ""
        print(f"  {name:22s} {mean:+.4f}   {u:.0f} updates{spread}")

    ordered_gap = (sum(by_arm["none-ordered"]) / args.seeds
                   - sum(by_arm["retention-ordered"]) / args.seeds)
    shuffled_gap = (sum(by_arm["none-shuffled"]) / args.seeds
                    - sum(by_arm["retention-shuffled"]) / args.seeds)

    matched_forgot = sum(by_arm["matched-ordered"]) / args.seeds
    retention_forgot = sum(by_arm["retention-ordered"]) / args.seeds
    print(f"\n  matched vs retention, ordered    "
          f"{matched_forgot - retention_forgot:+.4f}  "
          f"(positive = rehearsal is doing real work)")
    print(f"  retention's advantage, ordered   {ordered_gap:+.4f}")
    print(f"  retention's advantage, shuffled  {shuffled_gap:+.4f}")
    print(f"  collapse when order is destroyed {ordered_gap - shuffled_gap:+.4f}")

    print("\n" + "=" * 62)
    print("CHECKS")
    print("=" * 62)

    none_forgot = sum(by_arm["none-ordered"]) / args.seeds
    if none_forgot <= 0.01:
        print(f"  FAILED: the unprotected ordered arm forgot nothing "
              f"({none_forgot:+.4f}). There was nothing to retain, so no "
              f"number above measures retention. Lengthen the phases or "
              f"raise the learning rate before believing any of this.")
    else:
        print(f"  ok: the test could fail. Unprotected ordered forgetting "
              f"was {none_forgot:+.4f}.")

    vals = [round(sum(v) / len(v), 6) for v in by_arm.values()]
    if len(set(vals)) < len(vals):
        print("  WARNING: two arms produced identical numbers. In this "
              "repo that has always meant collapse, not agreement. Check "
              "the arms actually differ before reading anything into them.")
    else:
        print("  ok: no two arms produced identical numbers.")

    # Every arm must have had somewhere for the metric to move. An arm whose
    # phase-1 loss fell monotonically had no opportunity to show a rise, and
    # its 0.0000 says nothing about retention.
    flat = [r["name"] for r in runs
            if [m["phase1"] for _, m in r["marks"]][-1]
            == min(m["phase1"] for _, m in r["marks"])]
    if flat:
        print(f"  NOTE: phase-1 loss ended at its minimum in "
              f"{', '.join(sorted(set(flat)))}. Those arms had no chance to "
              f"show forgetting, so their 0.0000 is the metric being unable "
              f"to move rather than retention succeeding.")
    else:
        print("  ok: every arm had a non-final minimum, so every arm could "
              "have shown forgetting.")

    if matched_forgot - retention_forgot <= 0.01:
        print(f"  THE CONFOUND WINS: the matched arm forgot about as little "
              f"as retention ({matched_forgot:+.4f} against "
              f"{retention_forgot:+.4f}) while doing no rehearsal at all. "
              f"On this stream the protection comes from UPDATING LESS, not "
              f"from replay. Say that plainly rather than crediting the "
              f"retention layer.")
    else:
        print(f"  ok: the matched arm forgot {matched_forgot:+.4f} against "
              f"retention's {retention_forgot:+.4f} at the same update "
              f"count, so rehearsal is doing work that thinning updates "
              f"does not.")

    if ordered_gap <= 0:
        print(f"  NEGATIVE RESULT: retention did not help on the ordered "
              f"stream ({ordered_gap:+.4f}). That is a real answer about "
              f"language, not a bug to tune away.")
    elif shuffled_gap >= ordered_gap * 0.5:
        print(f"  WARNING: retention helped nearly as much on the SHUFFLED "
              f"stream ({shuffled_gap:+.4f} against {ordered_gap:+.4f}). "
              f"Whatever it is doing is not retention.")
    else:
        print(f"  ok: the advantage collapses when order is destroyed, "
              f"which is what retention predicts.")

    print(f"\n  written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()