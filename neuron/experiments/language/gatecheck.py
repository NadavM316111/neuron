"""Where should the absolute bar go?

The gate spends 22 of 35 updates on filler. If facts score meaningfully
higher than filler on surprise, an absolute bar fixes that for free.
If the distributions overlap, the bar cannot work and the gate needs a
different signal.

No training. Two minutes.
"""

import torch
from llm_backend import LLMBackend
from qa7b import FACTS, FILLER, CANARY, MODEL


if __name__ == "__main__":
    backend = LLMBackend(model_name=MODEL)
    print(f"{backend.device}  {backend.trainable_count:,} trainable\n")

    fact_scores = []
    filler_scores = []

    print("FACTS")
    for f in FACTS:
        s, eq = backend.score(f["text"])
        fact_scores.append(s)
        print(f"  s={s:6.3f}  eq={eq:5.3f}  {f['text'][:62]}")

    print("\nFILLER")
    for t in FILLER:
        s, eq = backend.score(t)
        filler_scores.append(s)
        print(f"  s={s:6.3f}  eq={eq:5.3f}  {t[:62]}")

    print("\nCANARY")
    for t in CANARY:
        s, _ = backend.score(t)
        print(f"  s={s:6.3f}  {t[:62]}")

    fact_scores.sort()
    filler_scores.sort()

    def q(xs, p):
        return xs[min(len(xs) - 1, int(len(xs) * p))]

    print("\n" + "=" * 72)
    print(f"{'':>10} {'min':>7} {'25%':>7} {'median':>7} {'75%':>7} {'max':>7} {'mean':>7}")
    for name, xs in [("facts", fact_scores), ("filler", filler_scores)]:
        print(f"{name:>10} {xs[0]:>7.3f} {q(xs,.25):>7.3f} {q(xs,.5):>7.3f} "
              f"{q(xs,.75):>7.3f} {xs[-1]:>7.3f} "
              f"{sum(xs)/len(xs):>7.3f}")
    print("=" * 72)

    # Sweep candidate bars and report what each would keep.
    print(f"\n{'bar':>7} {'facts kept':>12} {'filler kept':>12} {'purity':>8}")
    print("-" * 45)
    lo = min(fact_scores + filler_scores)
    hi = max(fact_scores + filler_scores)
    for i in range(12):
        bar = lo + (hi - lo) * i / 11
        fk = sum(1 for s in fact_scores if s >= bar)
        gk = sum(1 for s in filler_scores if s >= bar)
        purity = fk / (fk + gk) if (fk + gk) else 0.0
        print(f"{bar:>7.3f} {fk:>5}/{len(fact_scores):<6} "
              f"{gk:>5}/{len(filler_scores):<6} {purity:>7.0%}")
    print("-" * 45)
    print("""
Read the purity column. Currently the gate runs at roughly 37% purity
(13 fact updates out of 35 total).

  a bar reaching 70%+ purity while keeping most facts -> set min_surprise
     there and rerun the QA test
  distributions overlap, no bar helps  -> surprise cannot separate facts
     from filler here, and the gate needs a different signal entirely
""")