"""Does firing more often raise CORRECT?

The bar test showed CORRECT tracks FACT UPDATE COUNT almost directly:
13 updates -> 2.7 correct, 5 -> 0.7, 3 -> 0.0. Filler updates cost
nothing. So the question is whether more volume keeps helping, or whether
it plateaus and starts damaging the model.

Also tests the rehearsal fix: rehearse_per_item decouples protection from
update count, so the arms are protected equally regardless of how often
they learn.
"""

import json
import torch
import qa7b as Q


FRACTIONS = [0.30, 0.50, 0.70]
REHEARSE_PER_ITEM = 3     # roughly matches the old rate at fraction 0.30


def run(fraction, seed):
    torch.manual_seed(seed)
    backend = Q.ParaphraseBackend(mode="verbatim", model_name=Q.MODEL,
                                  focus_alpha=0.0)
    layer = Q.StabilityLayer(backend, canary=Q.CANARY, seed=seed,
                             top_fraction=fraction,
                             rehearse_per_item=REHEARSE_PER_ITEM)
    before = Q.ask_all(backend)
    for text in Q.build_stream(seed):
        layer.observe(text)
    after = Q.ask_all(backend)

    stats = layer.summary()
    stats.update(backend.hits)
    stats["distance"] = backend.distance()
    del backend, layer
    torch.cuda.empty_cache()
    return before, after, stats


if __name__ == "__main__":
    results = {}

    for frac in FRACTIONS:
        results[frac] = []
        for seed in Q.SEEDS:
            print(f"\n### top_fraction {frac}  seed {seed}")
            before, after, stats = run(frac, seed)
            ta = Q.tally(after, "fact")
            ca = Q.tally(after, "control")
            print(f"  updates {stats['updates']}  "
                  f"onFact {stats['fact_updates']}  "
                  f"onFill {stats['filler_updates']}  "
                  f"rehearsals {stats['rehearsals']}  "
                  f"rollbacks {stats['rollbacks']}")
            print(f"  facts {ta}   controls "
                  f"{ca.get('CORRECT',0)}/{len(Q.CONTROL_QA)}   "
                  f"health {stats['health']:+.4f}   "
                  f"dist {stats['distance']:.1f}")
            for b, a in zip(before, after):
                if b["kind"] == "fact":
                    print(f"    [{a['verdict']:>9}]  {a['a'][:88]}")
            results[frac].append(dict(after=ta, stats=stats,
                                      controls=ca.get("CORRECT", 0)))

    def m(rows, path):
        vals = []
        for r in rows:
            v = r
            for k in path:
                v = v.get(k, 0) if isinstance(v, dict) else 0
            vals.append(v)
        return sum(vals) / len(vals)

    print("\n" + "=" * 88)
    print(f"averaged over {len(Q.SEEDS)} seeds, "
          f"rehearsal every {REHEARSE_PER_ITEM} items in all arms")
    print(f"{'frac':>6} {'upd':>5} {'onFact':>7} {'onFill':>7} {'reh':>5} "
          f"{'CORRECT':>8} {'CONFUS':>7} {'FABRIC':>7} {'REFUS':>6} "
          f"{'ctrl':>5} {'health':>8} {'dist':>6}")
    print("-" * 88)
    for frac in FRACTIONS:
        rows = results[frac]
        print(f"{frac:>6.2f} "
              f"{m(rows,['stats','updates']):>5.1f} "
              f"{m(rows,['stats','fact_updates']):>7.1f} "
              f"{m(rows,['stats','filler_updates']):>7.1f} "
              f"{m(rows,['stats','rehearsals']):>5.1f} "
              f"{m(rows,['after','CORRECT']):>8.1f} "
              f"{m(rows,['after','CONFUSED']):>7.1f} "
              f"{m(rows,['after','FABRICATED']):>7.1f} "
              f"{m(rows,['after','REFUSED']):>6.1f} "
              f"{m(rows,['controls']):>5.1f} "
              f"{m(rows,['stats','health']):>+8.4f} "
              f"{m(rows,['stats','distance']):>6.1f}")
    print("=" * 88)
    print("""
  CORRECT keeps climbing, health flat   -> volume is the knob, push further
  CORRECT plateaus                      -> exposure count is not the limit,
                                           and something else caps it
  CORRECT climbs but health degrades    -> the real trade-off, and now you
                                           can pick a point on it

Watch FABRICATED. More updates on a fact it cannot fully absorb is exactly
how confident-wrong answers appeared yesterday.
""")
    with open("volume_test.json", "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2,
                  default=str)
    print("wrote volume_test.json")