"""Does a lower learning rate trade fabrication for honest refusal?

Two runs today, one variable different:
  lr 1e-3 on real text  -> 4 fabrications out of 4, zero refusals
  lr 1e-4 on invented facts -> 3 honest refusals, 1 wrong, 1 known

That is a complete flip of the failure mode from a single parameter. If it
holds, it is a knob you turn in the product rather than a diagnostic.

Four rates, everything else identical. Stopping at 12 rounds because the
curve run showed consolidation peaks around round 3 and the plateau is
flat from 11 to 29, so 12 captures the peak without the long tail.

  known      facts answerable from weights with the store switched off
  wrong      confident errors. This is the number that matters.
  refused    honest admissions of ignorance. Not a failure.

  a rate with high known and low wrong  -> the setting to use
  wrong rises with known everywhere     -> a hard trade-off, and knowing
      that is itself the finding
"""

import json
import time
import random
import torch
from llm_backend import LLMBackend
from consolidate_curve import (
    FACTS, INTERFERENCE, CONTROLS, REFUSALS, has, verdict, exam,
    controls_score, MODEL,
)


SEED = 0
ROUNDS = 12
REPLAYS_PER_ROUND = 3
RATES = [1e-3, 3e-4, 1e-4, 3e-5]


def run(lr):
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    backend = LLMBackend(model_name=MODEL, focus_alpha=0.0)
    for g in backend.optimizer.param_groups:
        g["lr"] = lr

    t0 = time.time()
    rows = exam(backend, FACTS)
    history = [dict(round=0,
                    known=sum(1 for r in rows if r["v"] == "KNOWN"),
                    wrong=sum(1 for r in rows if r["v"] in ("WRONG", "MUDDLED")),
                    refused=sum(1 for r in rows if r["v"] == "REFUSED"),
                    controls=controls_score(backend))]

    print(f"  {'round':>6} {'known':>6} {'wrong':>6} {'refused':>8} {'ctrl':>5}")
    h = history[0]
    print(f"  {0:>6} {h['known']:>6} {h['wrong']:>6} {h['refused']:>8} "
          f"{h['controls']:>5}")

    for rnd in range(1, ROUNDS + 1):
        batch = []
        for f in FACTS:
            batch += [f["text"]] * REPLAYS_PER_ROUND
        batch += list(INTERFERENCE) * REPLAYS_PER_ROUND
        rng.shuffle(batch)
        for text in batch:
            backend.update(text, 1)

        rows = exam(backend, FACTS)
        rec = dict(round=rnd,
                   known=sum(1 for r in rows if r["v"] == "KNOWN"),
                   wrong=sum(1 for r in rows if r["v"] in ("WRONG", "MUDDLED")),
                   refused=sum(1 for r in rows if r["v"] == "REFUSED"),
                   controls=controls_score(backend))
        history.append(rec)
        print(f"  {rnd:>6} {rec['known']:>6} {rec['wrong']:>6} "
              f"{rec['refused']:>8} {rec['controls']:>5}")

    final_rows = [dict(a=r["a"], v=r["v"]) for r in rows]
    mins = (time.time() - t0) / 60

    del backend
    torch.cuda.empty_cache()
    return history, final_rows, mins


if __name__ == "__main__":
    print(f"{len(RATES)} learning rates, {ROUNDS} rounds x "
          f"{REPLAYS_PER_ROUND} replays, {len(FACTS)} facts\n")

    results = {}
    for lr in RATES:
        print("#" * 60)
        print(f"# lr {lr}")
        print("#" * 60)
        history, final_rows, mins = run(lr)

        peak_known = max(h["known"] for h in history)
        peak_round = next(h["round"] for h in history
                          if h["known"] == peak_known)
        # worst confident-error count seen at any point, not just the end
        worst_wrong = max(h["wrong"] for h in history)

        print(f"\n  final answers, weights only:")
        for r in final_rows:
            print(f"    [{r['v']:>8}]  {r['a'][:62]}")
        print(f"\n  peak known {peak_known}/{len(FACTS)} at round "
              f"{peak_round}   worst wrong {worst_wrong}   {mins:.1f} min\n")

        results[str(lr)] = dict(history=history, final=final_rows,
                                peak_known=peak_known,
                                peak_round=peak_round,
                                worst_wrong=worst_wrong,
                                end_known=history[-1]["known"],
                                end_wrong=history[-1]["wrong"],
                                end_refused=history[-1]["refused"],
                                end_controls=history[-1]["controls"],
                                minutes=mins)

    print("=" * 78)
    print(f"{'lr':>8} {'peak':>6} {'@rnd':>6} {'end known':>10} "
          f"{'end wrong':>10} {'worst wrong':>12} {'refused':>8} {'ctrl':>5}")
    print("-" * 78)
    for lr in RATES:
        r = results[str(lr)]
        print(f"{lr:>8} {r['peak_known']:>6} {r['peak_round']:>6} "
              f"{r['end_known']:>10} {r['end_wrong']:>10} "
              f"{r['worst_wrong']:>12} {r['end_refused']:>8} "
              f"{r['end_controls']}/5")
    print("=" * 78)
    print(f"""
Out of {len(FACTS)} facts.

  "wrong" is the number that decides this. A confident error is worse than
  a refusal, by the build plan's own standard: refusal turning into
  confident-wrong is damage, not progress.

  one rate with decent known and near-zero wrong -> use it
  wrong tracks known at every rate               -> a hard trade-off, and
      the honest conclusion is that you cannot get facts into weights
      without buying fabrication along with them
  every rate near zero known                     -> the single fact that
      consolidated in the curve run was luck, not a mechanism

Watch "worst wrong" as well as the end value. A rate that fabricates
mid-run and recovers is still unsafe in a live system, because a real user
would have been lied to during that window.
""")
    with open("lr_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote lr_sweep.json")