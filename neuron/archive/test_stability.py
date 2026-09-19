"""Proves the stability layer works against a backend that is not a model."""

from stability import StabilityLayer
from fake_backend import FakeBackend

CANARY = [f"canary-{i}" for i in range(5)]
ORDINARY = [f"ordinary-{i}" for i in range(30)]
NOVEL = [f"NOVEL-fact-{i}" for i in range(8)]
JUNK = [f"JUNK-garbage-{i}" for i in range(8)]


def build_stream(seed):
    import random
    rng = random.Random(seed)
    stream = [rng.choice(ORDINARY) for _ in range(15)]   # calibration
    body = NOVEL + JUNK + [rng.choice(ORDINARY) for _ in range(40)]
    rng.shuffle(body)
    return stream + body


def run(damage=0.0, guard=True, label=""):
    backend = FakeBackend(damage_per_update=damage)
    layer = StabilityLayer(backend, canary=CANARY, guard=guard, verbose=False)

    reasons = {}
    novel_hits = junk_hits = 0
    for item in build_stream(0):
        r = layer.observe(item)
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        if r["learned"]:
            if item in NOVEL:
                novel_hits += 1
            elif item in JUNK:
                junk_hits += 1

    print(f"\n--- {label} ---")
    print(" ", layer)
    print("  reasons:", reasons)
    print(f"  novel learned {novel_hits}/{len(NOVEL)}   "
          f"junk learned {junk_hits}/{len(JUNK)}")
    print(f"  health {layer.health():+.4f}   "
          f"anchor {len(layer.anchor)}  buffer {len(layer.buffer)}")
    return layer, novel_hits, junk_hits


if __name__ == "__main__":
    print("Stability layer, no machine learning anywhere in this test.")

    layer, novel, junk = run(damage=0.0, label="healthy backend")
    assert layer.stats["updates"] > 0, "gate never fired"
    assert layer.stats["vetoed"] > 0, "veto never fired"
    assert layer.stats["rehearsals"] > 0, "rehearsal never ran"
    assert junk == 0, f"veto leaked, {junk} junk items learned"
    assert novel > 0, "no novel items learned"
    print("  OK  gate fires, veto blocks junk, rehearsal runs")

    layer, _, _ = run(damage=0.30, label="degrading backend, guard ON")
    assert layer.stats["rollbacks"] > 0, "guard never fired on a degrading model"
    print("  OK  guard detected degradation and rolled back")

    layer_off, _, _ = run(damage=0.30, guard=False,
                          label="degrading backend, guard OFF")
    assert layer_off.stats["rollbacks"] == 0
    assert layer_off.health() > layer.health(), \
        "guard did not improve final health"
    print("  OK  guard-off ends worse than guard-on, as on 16 Aug")

    print("\nAll checks passed. stability.py has no dependency on "
          "transformers, tokenizers, or text.")