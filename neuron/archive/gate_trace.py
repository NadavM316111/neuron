"""Why did the gate fire zero times in 10,000 steps?

In grow.py the guarded arm reported gate 0 with 4,000 rehearsals. Every bit
of learning came from replay. With warmup 50, top_fraction 0.50 and
loss_floor 0.02, roughly half of all items should have passed early on.

This instruments every decision instead of guessing. Two hundred items,
printing the reason for each and the numbers behind it.
"""

import random
import torch
from collections import Counter
from world import GridWorld, ACTIONS, EVENTS
from stability import StabilityLayer
from grow import WorldBackend, make_stream


if __name__ == "__main__":
    torch.manual_seed(0)
    train = make_stream(0, 400)
    backend = WorldBackend(0)

    print("raw surprise on the first 20 items, before any learning:")
    for item in train[:20]:
        s, _ = backend.score(item)
        print(f"  {item[2]:>9}  surprise {s:.4f}")

    print("\nnow through the stability layer, same settings as grow.py:\n")
    backend = WorldBackend(0)
    layer = StabilityLayer(
        backend, canary=train[:40], seed=0,
        warmup=50, top_fraction=0.50, coherence_veto=1e9,
        loss_floor=0.02, steps_per_update=1,
        rehearse_per_item=5, rehearse_count=2, rehearse_steps=1,
        anchor_size=100, buffer_size=500,
        guard=True, guard_per_item=500, canary_tolerance=0.5)

    reasons = Counter()
    print(f"{'i':>4} {'event':>9} {'reason':>13} {'surprise':>9} "
          f"{'thresh':>8} {'relative':>9}")
    print("-" * 60)
    for i, item in enumerate(train):
        r = layer.observe(item)
        reasons[r["reason"]] += 1
        if i < 80 or r["reason"] == "novel":
            s = r.get("surprise")
            t = r.get("threshold")
            rel = r.get("relative")
            print(f"{i:>4} {item[2]:>9} {r['reason']:>13} "
                  f"{s if s is None else f'{s:8.4f}'} "
                  f"{t if t is None else f'{t:7.4f}'} "
                  f"{rel if rel is None else f'{rel:8.4f}'}")

    print("\ndecision counts over 400 items:")
    for k, v in reasons.most_common():
        print(f"  {k:>13}: {v:>4}")

    s = layer.summary()
    print(f"\ngate fires {s['updates']}  rehearsals {s['rehearsals']}  "
          f"floored {s['floored']}  unremarkable {s['unremarkable']}  "
          f"warmed {s['warmed']}")
    print("""
Read the reason column.

  mostly "already_known"  -> rehearsal drives loss under the 0.02 floor
      faster than new items arrive, so everything looks learned. The floor
      is the culprit and it is too high for this loss scale.
  mostly "unremarkable"   -> items are below the rolling median. If
      rehearsal has already flattened the loss distribution, almost
      everything sits at the same tiny value and the median is meaningless.
  mostly "warmup"         -> the warmup never ends, which would be a bug.

Either way the mechanism is the same: rehearsal is learning everything
before the gate gets a chance to choose. The gate is not broken, it is
being pre-empted.
""")