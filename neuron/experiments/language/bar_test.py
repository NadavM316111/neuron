"""Does an absolute surprise bar improve fact learning?

gatecheck showed facts at mean 4.517 and filler at 2.668, with a bar at
3.5 keeping 5/6 facts and only 3/28 filler. The gate currently runs at
~37% purity. This tests whether raising purity raises CORRECT.

Verbatim only, since paraphrase lost decisively.
"""

import json
import torch
import qa7b as Q


BARS = [0.0, 3.5, 3.9]


def run(bar, seed):
    torch.manual_seed(seed)
    backend = Q.ParaphraseBackend(mode="verbatim", model_name=Q.MODEL,
                                  focus_alpha=0.0)
    layer = Q.StabilityLayer(backend, canary=Q.CANARY, seed=seed,
                             min_surprise=bar)
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

    for bar in BARS:
        results[bar] = []
        for seed in Q.SEEDS:
            print(f"\n### bar {bar}  seed {seed}")
            before, after, stats = run(bar, seed)
            ta = Q.tally(after, "fact")
            ca = Q.tally(after, "control")
            total = stats["fact_updates"] + stats["filler_updates"]
            purity = stats["fact_updates"] / total if total else 0.0
            print(f"  updates {stats['updates']}  "
                  f"onFact {stats['fact_updates']}  "
                  f"onFill {stats['filler_updates']}  "
                  f"purity {purity:.0%}  "
                  f"below_bar {stats.get('below_bar', 0)}")
            print(f"  facts after {ta}   controls "
                  f"{ca.get('CORRECT',0)}/{len(Q.CONTROL_QA)}   "
                  f"health {stats['health']:+.4f}")
            for b, a in zip(before, after):
                if b["kind"] == "fact":
                    print(f"    [{a['verdict']:>9}]  {a['a'][:88]}")
            results[bar].append(dict(after=ta, stats=stats,
                                     controls=ca.get("CORRECT", 0),
                                     purity=purity))

    def m(rows, path):
        vals = []
        for r in rows:
            v = r
            for k in path:
                v = v.get(k, 0) if isinstance(v, dict) else 0
            vals.append(v)
        return sum(vals) / len(vals)

    print("\n" + "=" * 80)
    print(f"averaged over {len(Q.SEEDS)} seeds")
    print(f"{'bar':>6} {'upd':>5} {'onFact':>7} {'onFill':>7} {'purity':>7} "
          f"{'CORRECT':>8} {'FABRIC':>7} {'REFUS':>6} {'ctrl':>5} {'health':>8}")
    print("-" * 80)
    for bar in BARS:
        rows = results[bar]
        print(f"{bar:>6.1f} "
              f"{m(rows,['stats','updates']):>5.1f} "
              f"{m(rows,['stats','fact_updates']):>7.1f} "
              f"{m(rows,['stats','filler_updates']):>7.1f} "
              f"{m(rows,['purity']):>6.0%} "
              f"{m(rows,['after','CORRECT']):>8.1f} "
              f"{m(rows,['after','FABRICATED']):>7.1f} "
              f"{m(rows,['after','REFUSED']):>6.1f} "
              f"{m(rows,['controls']):>5.1f} "
              f"{m(rows,['stats','health']):>+8.4f}")
    print("=" * 80)
    print("""
bar 0.0 is the current behaviour, CORRECT 2.7 at ~37% purity.

  CORRECT rises with purity     -> the bar is a free win, make it default
  CORRECT flat despite purity   -> fact updates were not the bottleneck,
                                   and the limit is elsewhere
  CORRECT falls at 3.9          -> dropping 2 of 6 facts costs more than
                                   the cleaner budget gains
""")
    with open("bar_test.json", "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2,
                  default=str)
    print("wrote bar_test.json")