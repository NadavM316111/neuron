"""Does an early constraint survive a long task, and does retention help?

THE GAP THIS AIMS AT. Frontier agents succeed on nearly 100% of tasks that
take a skilled human under about four minutes and on under 10% of tasks
taking more than about four hours. The length of task completable at 50%
reliability has doubled every seven months since 2019, recently every four.
The growth is driven primarily by reliability and the ability to adapt to
mistakes rather than by raw capability. So the binding constraint on AI
doing real work is how long it stays coherent, not how smart it is.

The specific failure is the interesting part. Studies of long-context web
agents report success rates of 40-50% on short-horizon versions of a task
falling below 10% once the same task is embedded in a longer interaction
history, EVEN WHEN the relevant information remains technically present in
the context window. The information is there and performance collapses
anyway. That rules out running out of room and rules out losing the notes.
Something degrades about what the system holds onto as a task lengthens.

WHY THIS IS NOT drift.py. That experiment established retention on a stream
that CONTRADICTS itself: phase 2 reversed phase 1's rule. Here the
intervening material does not contradict anything. It is simply unrelated
work, and lots of it. That is what a long task actually looks like — an
early instruction, then hundreds of steps of other business, then the
instruction still has to hold.

And this project's own results predict it might not help. The retention
layer helped on non-stationary streams and cost a little on stationary ones:
same weather data shuffled gave -1.56, -1.12, -2.68 across three climates,
and Fashion-MNIST was inside the noise. Unrelated filler is closer to
stationary than to contradictory. If retention only works against
contradiction, it will not touch this, and that is worth knowing before the
claim gets made.

TWO METRICS, because they disagree in a known direction. `degradation` is
end minus post-setup, and it PENALISES DEEP LEARNING: an arm that acquired
the constraint to 0.69 has further to fall than one that reached 1.32, so an
arm that learned less can look more stable. That bias is real here — the
plain arm makes several times more updates during setup and acquires the
constraint harder. `final` is the loss at the end and has no such bias. A
result only counts if both say the same thing.

THE DESIGN. Task length is the swept variable, mirroring the way the field
measures this.

  setup     N items establishing a rule, stated many ways
  filler    M items of unrelated material, no contradiction, M swept
  test      held-out probes on the rule

Two arms: with the stability layer and without. The question is not whether
loss rises with M — it will — but whether retention FLATTENS that curve.

WHAT A RESULT WOULD AND WOULD NOT MEAN. This is one synthetic task on one
1.5B model with no tools and no real agent loop. It is a model of the
phenomenon, not the phenomenon. If retention flattens the degradation curve
here, that is a reason to try it under a real agent. If it does not, the
memory-layer hypothesis for long-horizon failure is weaker than it looks.

THE FIRST RUN WAS INVALID and the failure is worth recording, because the
number it printed looked spectacular. With setup at 40 items the retention
arm made TWO updates and fired a rollback, so its post-setup loss was 6.4662
— the untrained value. It had never learned the constraint. The plain arm
learned it properly to 1.2509 and then lost it during the filler, ending at
5.9306.

Degradation measured as "end minus post-setup" therefore read +4.63 for the
arm that learned and forgot, and -4.63 for the arm that never learned and
picked the rule up incidentally during the filler. The script reported
retention flattening the curve by 9.26 points. It was comparing an arm that
had something to lose against an arm that had nothing.

Two causes, both in the harness. The gate's warmup is 30 items, so a
40-item setup was almost entirely warmup and produced no updates. And
guard_per_item scaled to total//8, which at 40 items meant a health check
every 10 items against a canary baseline taken from an untrained model, so
ordinary learning read as damage and triggered a rollback.

Fixed three ways. The setup phase must clear warmup by a margin, and the
script refuses to run otherwise. The guard checks on a fixed schedule rather
than one derived from a tiny total. And a LEARNABILITY GATE: any arm whose
post-setup loss is not well below its starting loss is reported as having
never learned the constraint, and no degradation number is read from it.

    python longhorizon.py                    # ~45 minutes
    python longhorizon.py --setup 120 --fillers 0 300 --seeds 1   # ~20 min
"""

import argparse
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.expanduser(
    "~/neuron/neuron/experiments/language"))
sys.path.insert(0, os.path.expanduser("~/neuron/neuron/core"))

import torch                                          # noqa: E402

from drift import build_stream, probe_loss            # noqa: E402
from llm_backend import LLMBackend, CANARY            # noqa: E402
from stability import StabilityLayer                  # noqa: E402


def free(*objs):
    for o in objs:
        del o
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


GUARD_EVERY = 200          # fixed, not derived from the run length


def make_layer(backend, total, guard=True):
    """The same configuration the runtime ships, so a result here transfers.

    rehearse_per_item is 24 rather than 6, from the 7 Sep sweep: a quarter
    of the replay work matched the old setting on final loss and beat it on
    retention.
    """
    # A fixed schedule. Deriving it from `total` meant a short run checked
    # health every 10 items against a baseline taken before any learning,
    # so ordinary progress read as damage and rolled the model back.
    return StabilityLayer(
        backend, canary=CANARY, seed=0,
        window=200, warmup=30, top_fraction=0.50,
        coherence_veto=1.3, loss_floor=0.35, steps_per_update=1,
        anchor_size=60, buffer_size=300, sequence_len=8,
        rehearse_per_item=24, rehearse_count=2, rehearse_steps=1,
        replay_policy="old", canary_from_stream=8,
        guard=guard, guard_per_item=GUARD_EVERY, canary_tolerance=0.40,
        # Tell the guard how long the run is. Without this it rolled back
        # near the end of short tasks and could not rebuild: at 250 filler
        # items, runs with two rollbacks ended at 6.28 and 6.43 while runs
        # with one ended at 2.01 and 2.28, split four out of four.
        expect_items=total, min_runway=200)


def run(retention, setup, filler, probes, model, seed, guard=True):
    """One arm at one task length."""
    backend = LLMBackend(model_name=model)
    total = len(setup) + len(filler)
    layer = make_layer(backend, total, guard) if retention else None

    before = probe_loss(backend, probes)

    # The constraint is established first, exactly as an instruction at the
    # start of a long task would be.
    for text in setup:
        if layer is not None:
            layer.observe(text)
        else:
            backend.update(text, 1)
    after_setup = probe_loss(backend, probes)

    # Then the rest of the task: unrelated work, none of it contradicting
    # the constraint, and a lot of it.
    for text in filler:
        if layer is not None:
            layer.observe(text)
        else:
            backend.update(text, 1)
    after_filler = probe_loss(backend, probes)

    s = (layer.summary() if layer is not None
         else dict(updates=total, rehearsals=0, rollbacks=0, health=0.0))
    free(backend, layer)
    return dict(retention=retention, filler=len(filler), seed=seed,
                before=before, after_setup=after_setup,
                after_filler=after_filler,
                degradation=after_filler - after_setup,
                updates=s["updates"], rehearsals=s["rehearsals"],
                rollbacks=s["rollbacks"], health=s["health"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", type=int, default=160,
                    help="items establishing the constraint")
    ap.add_argument("--fillers", type=int, nargs="+",
                    default=[0, 300, 900],
                    help="task lengths to sweep, in filler items")
    ap.add_argument("--probes", type=int, default=25)
    ap.add_argument("--no-guard", action="store_true")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="longhorizon.json")
    args = ap.parse_args()

    # The gate spends its first `warmup` items unable to judge surprise, so
    # a setup phase near that length produces almost no updates and the arm
    # never learns the constraint. 120 gives it 90 usable items.
    WARMUP = 30
    if args.setup < WARMUP * 4:
        raise SystemExit(
            f"  --setup {args.setup} is too short: the gate's warmup is "
            f"{WARMUP} items, so the retention arm would spend most of "
            f"the setup phase unable to learn and would never acquire the "
            f"constraint. Use at least {WARMUP * 4}.")

    need = args.setup + max(args.fillers) + args.probes
    print(f"  constraint from {args.setup} items, then "
          f"{args.fillers} items of unrelated work")
    print(f"  measuring whether the constraint survives\n", flush=True)

    runs = []
    for seed in range(args.seeds):
        # phase1 is the rule; phase3 is unrelated filler that contradicts
        # nothing. build_stream holds out probes from the same pool.
        train, probe = build_stream(seed, need, args.probes)
        setup = train["phase1"][:args.setup]
        filler_pool = train["phase3"]
        probes = probe["phase1"]

        for m in args.fillers:
            if m > len(filler_pool):
                print(f"  skipping filler {m}: pool holds "
                      f"{len(filler_pool)}")
                continue
            for retention in (True, False):
                t = time.time()
                r = run(retention, setup, filler_pool[:m], probes,
                        args.model, seed, not args.no_guard)
                runs.append(r)
                tag = "retention" if retention else "plain    "
                print(f"  filler {m:5d}  {tag}  "
                      f"setup {r['after_setup']:.4f} -> "
                      f"end {r['after_filler']:.4f}  "
                      f"({r['degradation']:+.4f})  "
                      f"{r['updates']} upd  {r['rollbacks']} rb  "
                      f"{time.time() - t:.0f}s", flush=True)

    with open(args.out, "w") as f:
        json.dump(dict(args=vars(args), runs=runs), f, indent=2)

    def pick(retention, m, field="degradation"):
        vals = [r[field] for r in runs
                if r["retention"] == retention and r["filler"] == m]
        return sum(vals) / len(vals) if vals else None

    lengths = sorted({r["filler"] for r in runs})

    print("\n" + "=" * 72)
    print("DEGRADATION OF THE EARLY CONSTRAINT AS THE TASK LENGTHENS")
    print("=" * 72)
    print(f"  {'':13} {'---- degradation ----':>23}  "
          f"{'---- final loss ----':>22}")
    print(f"  {'filler items':>13} {'plain':>10} {'retention':>11}  "
          f"{'plain':>10} {'retention':>11}")
    for m in lengths:
        p, r = pick(False, m), pick(True, m)
        pf = pick(False, m, "after_filler")
        rf = pick(True, m, "after_filler")
        print(f"  {m:>13} {p:>+10.4f} {r:>+11.4f}  "
              f"{pf:>10.4f} {rf:>11.4f}")

    # Did the two arms acquire the constraint to the same depth? If not,
    # degradation is biased toward whichever learned it less well.
    for m in lengths:
        ps = pick(False, m, "after_setup")
        rs = pick(True, m, "after_setup")
        if abs(ps - rs) > 0.3:
            print(f"\n  NOTE at filler {m}: the arms acquired the "
                  f"constraint to different depths")
            print(f"  (plain {ps:.4f}, retention {rs:.4f}). Degradation "
                  f"favours whichever learned")
            print(f"  it less well, so read the final-loss columns for "
                  f"that row.")

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)

    longest = lengths[-1]
    p_long, r_long = pick(False, longest), pick(True, longest)

    # DID EACH ARM LEARN THE CONSTRAINT AT ALL? An arm whose post-setup loss
    # is barely below where it started never acquired it, so "how much did
    # it degrade" is not a question about that arm. The first version of
    # this script read a +9.26 advantage off exactly that situation: the
    # retention arm had made two updates and had nothing to lose.
    def acquired(retention):
        rs = [r for r in runs if r["retention"] == retention]
        gains = [r["before"] - r["after_setup"] for r in rs]
        return sum(gains) / len(gains) if gains else 0.0

    g_plain, g_ret = acquired(False), acquired(True)
    print(f"  constraint acquired during setup: plain {g_plain:+.4f}, "
          f"retention {g_ret:+.4f}")

    MIN_GAIN = 1.0
    for name, gain in (("plain", g_plain), ("retention", g_ret)):
        if gain < MIN_GAIN:
            print(f"\n  THE {name.upper()} ARM NEVER LEARNED THE "
                  f"CONSTRAINT. Its loss fell only {gain:.4f}")
            print(f"  during setup, so it had nothing to lose and its "
                  f"degradation number is not")
            print(f"  interpretable. Nothing below is readable. Lengthen "
                  f"--setup, or check whether")
            print(f"  the guard is rolling back ordinary learning.")
            return

    rb = sum(r["rollbacks"] for r in runs if r["retention"])
    if rb:
        print(f"  note: {rb} rollback(s) fired in the retention arms, so "
              f"some learning was")
        print(f"  discarded during the run. Read the acquisition numbers "
              f"above before the")
        print(f"  degradation ones.")

    if p_long < 0.02:
        print(f"  NO DEGRADATION TO PREVENT. The unprotected arm's "
              f"constraint did not decay")
        print(f"  even after {longest} items of unrelated work "
              f"({p_long:+.4f}). Either the filler is")
        print(f"  too short or it is too different from the rule to "
              f"interfere. Nothing below")
        print(f"  is readable; lengthen the task before believing any of "
              f"it.")
        return

    print(f"  after {longest} items of unrelated work, the constraint "
          f"decayed {p_long:+.4f}")
    print(f"  without retention and {r_long:+.4f} with it")

    pf_long = pick(False, longest, "after_filler")
    rf_long = pick(True, longest, "after_filler")
    print(f"  final loss at {longest}: plain {pf_long:.4f}, "
          f"retention {rf_long:.4f} ({pf_long - rf_long:+.4f})")

    # DOES THE DIRECTION HOLD ACROSS LENGTHS? Reading only the longest row
    # is the same mistake the size sweep made with an endpoint-only trend
    # test. A first version of this verdict reported a clean win from the
    # 800 row while retention was WORSE at 250, because one seed's
    # retention run ended at 6.7384 against the other's 2.4005 for the
    # identical configuration.
    directions = []
    for m in lengths:
        if m == 0:
            continue                       # no filler, nothing to degrade
        pf, rf = pick(False, m, "after_filler"), pick(True, m, "after_filler")
        directions.append((m, pf - rf))
    helped = [m for m, d in directions if d > 0.05]
    hurt = [m for m, d in directions if d < -0.05]

    # And how variable is each arm? An average that hides a blow-up is not
    # a result. Spread is reported per arm at the longest length.
    def spread(retention, m):
        vals = [r["after_filler"] for r in runs
                if r["retention"] == retention and r["filler"] == m]
        return (max(vals) - min(vals)) if len(vals) > 1 else 0.0

    print(f"  direction by length: " + ", ".join(
        f"{m}:{d:+.2f}" for m, d in directions))
    sp_p, sp_r = spread(False, longest), spread(True, longest)
    print(f"  seed spread at {longest}: plain {sp_p:.4f}, "
          f"retention {sp_r:.4f}")

    if hurt and helped:
        print(f"\n  THE DIRECTION FLIPS. Retention helps at "
              f"{', '.join(map(str, helped))} and hurts at")
        print(f"  {', '.join(map(str, hurt))}. A win at the longest length "
              f"alone is not a result when")
        print(f"  the sign reverses underneath it. More seeds at the "
              f"lengths that disagree are")
        print(f"  what would settle this; the averages are currently "
              f"hiding at least one run")
        print(f"  that went badly wrong.")
        max_spread = max(spread(True, m) for m, _ in directions)
        if max_spread > 1.0:
            print(f"\n  Retention's runs vary by up to {max_spread:.4f} "
                  f"between seeds on identical")
            print(f"  configurations, so its averages are not stable "
                  f"enough to read a small effect")
            print(f"  from. {sum(r['rollbacks'] for r in runs)} rollback(s) "
                  f"fired in total, which is the likeliest source.")
        return

    # Both metrics must agree before anything is claimed.
    agree = ((p_long - r_long > 0.05) == (pf_long - rf_long > 0.05))
    if not agree:
        print(f"\n  THE TWO METRICS DISAGREE. Degradation says "
              f"{p_long - r_long:+.4f} and final loss says")
        print(f"  {pf_long - rf_long:+.4f}. That usually means the arms "
              f"acquired the constraint to")
        print(f"  different depths, so one metric is measuring how much "
              f"there was to lose rather")
        print(f"  than how well it was kept. Claim neither.")
        return

    if p_long - r_long > 0.05:
        print(f"\n  RETENTION FLATTENS THE CURVE at every length tested, "
              f"by {p_long - r_long:.4f}")
        print(f"  on degradation and")
        print(f"  {pf_long - rf_long:.4f} on final loss, which agree. "
              f"An early constraint")
        print(f"  survives a long task better with the layer than without, "
              f"on material that does")
        print(f"  not contradict it. That is a different claim from the "
              f"earlier drift result and")
        print(f"  a reason to try this under a real agent, where the "
              f"documented failure is")
        print(f"  exactly this: performance collapsing as a task lengthens "
              f"while the relevant")
        print(f"  information is still present.")
        print(f"\n  It is one synthetic task on one 1.5B with no tools. A "
              f"model of the problem,")
        print(f"  not the problem.")
    elif p_long - r_long < -0.05:
        print(f"\n  RETENTION MAKES IT WORSE by {r_long - p_long:.4f}. The "
              f"layer helps against")
        print(f"  contradiction and hurts here, which matches this "
              f"project's own scope finding:")
        print(f"  it cost a little on stationary streams (-1.56, -1.12, "
              f"-2.68 on shuffled")
        print(f"  weather). Unrelated filler behaves like a stationary "
              f"stream. The memory-layer")
        print(f"  hypothesis for long-horizon failure does not survive "
              f"this.")
    else:
        print(f"\n  NO DIFFERENCE ({p_long - r_long:+.4f}). The constraint "
              f"decays and retention does")
        print(f"  not change it. Whatever causes degradation over an "
              f"unrelated-work stream, the")
        print(f"  gate and replay do not address it. That is a real answer "
              f"and it argues against")
        print(f"  pointing this layer at long-horizon agents without "
              f"something else changing.")

    # Is degradation actually a function of length, or just of any filler?
    if len(lengths) >= 3:
        mid = lengths[len(lengths) // 2]
        p_mid = pick(False, mid)
        if p_long > p_mid + 0.05:
            print(f"\n  Degradation grows with length: {p_mid:+.4f} at "
                  f"{mid} items against {p_long:+.4f}")
            print(f"  at {longest}. So this is about task LENGTH and not "
                  f"merely about interruption.")
        else:
            print(f"\n  CAUTION: degradation does not grow with length "
                  f"({p_mid:+.4f} at {mid} items,")
            print(f"  {p_long:+.4f} at {longest}). Most of the damage "
                  f"happens early, so this is about")
            print(f"  being interrupted at all rather than about how long "
                  f"the task is. That is a")
            print(f"  weaker analogue of the agent failure it was built to "
                  f"model.")

    print(f"\n  {args.seeds} seed(s), synthetic, no tool use.")
    print(f"\n  written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
