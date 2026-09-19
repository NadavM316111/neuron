"""The 7B retention result, with enough seeds to believe it.

The single-seed run found the weather result reproduced at 7 billion
parameters: online showed a textbook recency gradient (oldest phase -0.247,
newest -2.634) while guarded inverted it (oldest -1.734, newest -0.705),
retaining seven times more early material. Guarded also degraded the model's
general language ability three times less (+0.300 vs +0.911).

One seed. This runs three, and reports the per-seed spread alongside the
mean, because a mean of three tells you nothing if the three disagree.

Also added: a SHUFFLED control, the thing that settled the mechanism question
on weather. Same sentences, same count, random order — so no phase is ever
"early" and there is nothing to retain. If guarded's advantage collapses
when shuffled, the effect is retention here too, not just better learning.
That control is what turned the weather result from a correlation into a
cause, and it is cheap to repeat.

Arms per seed:
  online            ordered stream, no protection
  guarded           ordered stream, sequence replay
  online-shuffled   shuffled stream, no protection
  guarded-shuffled  shuffled stream, sequence replay

Frozen is dropped — it measured zero change in the single-seed run, as
expected, and costs a model load each time.
"""

import json
import random
import time

import torch

from llm_backend import LLMBackend
from stability import StabilityLayer
from sevenb import (PHASES, PROBES, CANARY, ORDER, MODEL, LR,
                    SEQ_LEN, SEQ_COUNT, REPEATS, measure, controls)


SEEDS = [0, 1, 2]


def build_stream(shuffled, seed):
    """Three phases in order, each repeated within its phase.

    shuffled=True randomises the whole stream, so no phase is early or late
    and there is nothing for rehearsal to protect.
    """
    stream = []
    for phase in ORDER:
        for _ in range(REPEATS):
            stream.extend(PHASES[phase])
    if shuffled:
        random.Random(seed).shuffle(stream)
    return stream


def run(guarded, shuffled, seed):
    torch.manual_seed(seed)
    t0 = time.time()
    backend = LLMBackend(model_name=MODEL, focus_alpha=0.0)
    for g in backend.optimizer.param_groups:
        g["lr"] = LR

    before = measure(backend)
    ctrl_before = controls(backend)

    stream = build_stream(shuffled, seed)
    layer = None
    if guarded:
        layer = StabilityLayer(
            backend, canary=CANARY, seed=seed,
            window=100, warmup=20, top_fraction=0.50,
            coherence_veto=1.3, loss_floor=0.35, steps_per_update=1,
            rehearse_per_item=4, rehearse_count=SEQ_COUNT,
            rehearse_steps=1, anchor_size=40, buffer_size=200,
            sequence_len=SEQ_LEN, replay_policy="uniform",
            guard=True, guard_per_item=40, canary_tolerance=0.40)

    for sent in stream:
        if layer is not None:
            layer.observe(sent)
        else:
            backend.update(sent, 1)

    after = measure(backend)
    ctrl_after = controls(backend)
    stats = layer.summary() if layer else {}

    del backend, layer
    torch.cuda.empty_cache()
    return dict(
        deltas={p: after[p] - before[p] for p in ORDER},
        ctrl_delta=ctrl_after - ctrl_before,
        rollbacks=stats.get("rollbacks", 0),
        updates=stats.get("updates", len(stream)),
        minutes=(time.time() - t0) / 60)


ARMS = [
    ("online", False, False),
    ("guarded", True, False),
    ("online-shuffled", False, True),
    ("guarded-shuffled", True, True),
]


if __name__ == "__main__":
    print(f"7B retention, {len(SEEDS)} seeds, with a shuffled control")
    print(f"order: {' -> '.join(ORDER)}, "
          f"'{ORDER[0]}' is the early material\n")

    results = {}
    for name, guarded, shuffled in ARMS:
        runs = []
        for seed in SEEDS:
            r = run(guarded, shuffled, seed)
            runs.append(r)
            print(f"  {name:>18} seed {seed}: " +
                  "  ".join(f"{p} {r['deltas'][p]:+.3f}" for p in ORDER) +
                  f"  ctrl {r['ctrl_delta']:+.3f}  "
                  f"rb {r['rollbacks']}  {r['minutes']:.1f}m", flush=True)
        results[name] = runs
        print(flush=True)

    def mean(xs):
        return sum(xs) / len(xs)

    early = ORDER[0]
    late = ORDER[-1]

    print("=" * 84)
    print("SURPRISE CHANGE PER PHASE, mean of 3 seeds "
          "(negative = more familiar)")
    print(f"{'arm':>18} " + "  ".join(f"{p:>12}" for p in ORDER) +
          f" {'controls':>10}")
    print("-" * 84)
    for name, _, _ in ARMS:
        runs = results[name]
        cells = "  ".join(
            f"{mean([r['deltas'][p] for r in runs]):>+12.3f}" for p in ORDER)
        print(f"{name:>18} {cells} "
              f"{mean([r['ctrl_delta'] for r in runs]):>+10.3f}")
    print("=" * 84)

    print(f"\nPER-SEED SPREAD on the early phase ('{early}'):")
    for name, _, _ in ARMS:
        vals = [r["deltas"][early] for r in results[name]]
        print(f"  {name:>18}: " + "  ".join(f"{v:+.3f}" for v in vals) +
              f"   mean {mean(vals):+.3f}  spread {max(vals) - min(vals):.3f}")

    on = [r["deltas"][early] for r in results["online"]]
    gu = [r["deltas"][early] for r in results["guarded"]]
    on_s = [r["deltas"][early] for r in results["online-shuffled"]]
    gu_s = [r["deltas"][early] for r in results["guarded-shuffled"]]
    wins = sum(1 for a, b in zip(gu, on) if a < b)

    ordered_adv = mean(on) - mean(gu)
    shuffled_adv = mean(on_s) - mean(gu_s)

    print(f"\nGUARDED ADVANTAGE ON EARLY MATERIAL")
    print(f"  ordered stream : {ordered_adv:+.3f}  "
          f"(guarded wins {wins}/{len(SEEDS)} seeds)")
    print(f"  shuffled stream: {shuffled_adv:+.3f}")
    print(f"  collapse       : {ordered_adv - shuffled_adv:+.3f}")

    print(f"\nDAMAGE TO GENERAL LANGUAGE (controls, lower is better):")
    for name, _, _ in ARMS:
        vals = [r["ctrl_delta"] for r in results[name]]
        print(f"  {name:>18}: {mean(vals):+.3f}  "
              f"(spread {max(vals) - min(vals):.3f})")

    print("""
Two claims are being tested.

  DOES IT REPRODUCE? guarded should beat online on the early phase in all
      three seeds, with a spread smaller than the effect. If the seeds
      disagree, the single-seed result was noise and should be withdrawn.

  IS IT RETENTION? The shuffled arms have no early material, because
      shuffling means nothing is ever old. If guarded's advantage COLLAPSES
      when shuffled, the mechanism is retention — exactly as the weather
      control established. If the advantage survives shuffling, guarded is
      simply a better learner and the retention story is wrong at this
      scale.

The controls column matters independently. The single-seed run had online
degrading general language three times more than guarded. If that holds
across seeds it is a safety result in its own right, separate from
retention.
""")
    with open("sevenb3.json", "w") as f:
        json.dump({k: [dict(deltas=r["deltas"], ctrl=r["ctrl_delta"],
                            rollbacks=r["rollbacks"]) for r in v]
                   for k, v in results.items()}, f, indent=2)
    print("wrote sevenb3.json")