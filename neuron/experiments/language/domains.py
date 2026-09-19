"""Does retention work in the WEIGHTS, on real text?

Today's runs took the system out of the toy for FACTS: 40,000 real Wikipedia
sentences, embedding retrieval, honesty holding at every scale. But they did
not test retention, because everything went into the store and a store does
not forget. The weights sat idle.

This tests the weights, on real text, at length. There is NO STORE here at
all.

WHAT IS LEARNED: a domain — the vocabulary and phrasing of a subject area.
Not a fact, which could be looked up, but a distribution over language. That
lives in weights by construction; you cannot retrieve a writing style.

WHAT IS MEASURED: cross-entropy loss on held-out sentences from each domain.
Lower means the model finds that kind of text more familiar. Nothing is
generated and nothing is graded, which removes the class of error that spoiled
three experiments this week — every probe-based measure needed a generator
and a scorer, and both were wrong at least once.

THE SHAPE: read domain A, then B, then C, each once, in order. Then ask what
happened to A. An unprotected model should find A less familiar after
spending two domains away from it. A protected one should not.

  loss on A rises for online, holds for guarded -> retention works in the
      weights on real text. That is the claim that has only ever been shown
      on weather and on invented sentences.
  both rise -> retention does not transfer to real text, whatever it did on
      weather.
  neither rises -> the domains are too similar for forgetting to occur, and
      the test cannot answer the question.

A LEARNABILITY CHECK RUNS FIRST. If training on domain A does not lower the
loss on A, nothing is being learned and the retention question is unaskable.
Two experiments this week were void for want of exactly this check.
"""

import json
import os
import re
import time

import torch

from llm_backend import LLMBackend
from stability import StabilityLayer


MODEL = "Qwen/Qwen2.5-7B-Instruct"
LR = 1e-4

HELD_OUT = 60
CHECKPOINT_EVERY = 500
CACHE = "domain_corpus.json"

PHASE_SENTENCES = 900
SCAN_ARTICLES = 12000

# Crude keyword classification. The domains only need to differ enough in
# vocabulary that learning one does not teach the others.
DOMAINS = {
    "biology": ["species", "genus", "cells", "organism", "protein",
                "bacteria", "evolution", "habitat", "enzyme", "genome"],
    "law": ["court", "statute", "plaintiff", "constitutional", "judicial",
            "legislation", "defendant", "tribunal", "jurisdiction",
            "appellate"],
    "music": ["album", "guitar", "orchestra", "composer", "melody",
              "recorded", "symphony", "vocals", "tempo", "concerto"],
}


def classify(text):
    """Which domain does this sentence belong to, if any?

    Requires a clear winner: a sentence matching two domains equally is
    ambiguous and is dropped rather than assigned.
    """
    low = text.lower()
    scores = {d: sum(1 for k in words if k in low)
              for d, words in DOMAINS.items()}
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return None
    others = [v for d, v in scores.items() if d != best]
    if scores[best] <= max(others):
        return None
    return best


def fetch_domains():
    """Real sentences, sorted into domains by vocabulary."""
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            data = json.load(f)
        if isinstance(data, dict) and all(d in data for d in DOMAINS):
            return data

    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.en",
                      split="train", streaming=True)

    out = {d: [] for d in DOMAINS}
    seen = 0
    need = PHASE_SENTENCES + HELD_OUT
    for article in ds:
        seen += 1
        if seen > SCAN_ARTICLES:
            break
        title = article["title"]
        if title.startswith("List of") or "(disambiguation)" in title:
            continue
        for s in re.split(r"(?<=[.!?])\s+", article["text"][:6000]):
            s = s.strip()
            if not (60 < len(s) < 250) or s.startswith("="):
                continue
            d = classify(s)
            if d and len(out[d]) < need:
                out[d].append(s)
        if all(len(v) >= need for v in out.values()):
            break
        if seen % 500 == 0:
            print("  " + "  ".join(f"{d} {len(v)}"
                                   for d, v in out.items()), flush=True)

    with open(CACHE, "w") as f:
        json.dump(out, f)
    return out


def mean_loss(backend, sentences):
    """Average cross-entropy on held-out text. Lower is more familiar.

    This is the whole measurement. Nothing is generated, nothing is graded.
    """
    total = 0.0
    for s in sentences:
        v, _ = backend.score(s)
        total += v
    return total / len(sentences)


def run(arm, phases, held):
    """phases is an ordered list of (domain, sentences)."""
    t0 = time.time()
    backend = LLMBackend(model_name=MODEL, focus_alpha=0.0)
    for g in backend.optimizer.param_groups:
        g["lr"] = LR

    layer = None
    if arm == "guarded":
        first = phases[0][1]
        layer = StabilityLayer(
            backend, canary=first[:40], seed=0,
            window=200, warmup=30, top_fraction=0.50,
            coherence_veto=1.3, loss_floor=0.35, steps_per_update=1,
            rehearse_per_item=6, rehearse_count=2, rehearse_steps=1,
            anchor_size=60, buffer_size=300, sequence_len=8,
            guard=True, guard_per_item=400, canary_tolerance=0.40)

    curve = [dict(step=0, phase="start",
                  losses={d: mean_loss(backend, held[d]) for d in held},
                  mins=0.0)]
    print(f"    start: " + "  ".join(
        f"{d} {curve[0]['losses'][d]:.3f}" for d in held), flush=True)

    step = 0
    for domain, sentences in phases:
        for s in sentences:
            if arm != "frozen":
                if layer is not None:
                    layer.observe(s)
                else:
                    backend.update(s, 1)
            step += 1

            if step % CHECKPOINT_EVERY == 0:
                losses = {d: mean_loss(backend, held[d]) for d in held}
                st = layer.summary() if layer else {}
                curve.append(dict(step=step, phase=domain, losses=losses,
                                  rollbacks=st.get("rollbacks", 0),
                                  mins=(time.time() - t0) / 60))
                print(f"    {step:>5} in {domain:>8}: " + "  ".join(
                    f"{d} {losses[d]:.3f}" for d in held) +
                    f"   rb {st.get('rollbacks', 0)}  "
                    f"{(time.time() - t0) / 60:5.1f}m", flush=True)

    del backend, layer
    torch.cuda.empty_cache()
    return curve


if __name__ == "__main__":
    print("Does retention work in the WEIGHTS, on real text?\n")
    print("No store. Loss on held-out text is the only measurement.\n")

    data = fetch_domains()
    for d, v in data.items():
        print(f"  {d}: {len(v)} sentences")
    if any(len(v) < PHASE_SENTENCES + HELD_OUT for v in data.values()):
        print("\n  ABORT: not enough sentences in every domain.")
        raise SystemExit

    order = list(DOMAINS)
    held = {d: data[d][:HELD_OUT] for d in order}
    phases = [(d, data[d][HELD_OUT:HELD_OUT + PHASE_SENTENCES])
              for d in order]
    print(f"\norder: {' -> '.join(order)}, "
          f"{PHASE_SENTENCES} sentences each")
    print(f"{HELD_OUT} held-out sentences per domain, never trained on\n")

    # Nothing else is meaningful if the model does not learn a domain at
    # all. This check is what two void experiments this week lacked.
    print("learnability check: does one phase lower its own loss?",
          flush=True)
    probe = LLMBackend(model_name=MODEL, focus_alpha=0.0)
    for g in probe.optimizer.param_groups:
        g["lr"] = LR
    before = mean_loss(probe, held[order[0]])
    for s in phases[0][1][:600]:
        probe.update(s, 1)
    after = mean_loss(probe, held[order[0]])
    del probe
    torch.cuda.empty_cache()
    print(f"  {order[0]} loss {before:.3f} -> {after:.3f} "
          f"({after - before:+.3f})")
    if after >= before - 0.02:
        print("  ABORT: training on a domain did not make that domain more\n"
              "  familiar, so nothing is being learned and retention cannot\n"
              "  be measured.")
        raise SystemExit
    print("  good: the model learns a domain, so forgetting can be "
          "measured\n")

    results = {}
    for arm in ["frozen", "online", "guarded"]:
        print(f"--- {arm} ---", flush=True)
        results[arm] = run(arm, phases, held)
        print(flush=True)

    first = order[0]
    print("=" * 84)
    print(f"LOSS ON HELD-OUT '{first.upper()}' TEXT — the retention curve")
    print("(it is read first, then abandoned for two other domains)")
    print(f"{'step':>7} {'phase':>10} " + "  ".join(
        f"{a:>10}" for a in ["frozen", "online", "guarded"]))
    print("-" * 84)
    n = min(len(results[a]) for a in results)
    for i in range(n):
        row = results["frozen"][i]
        print(f"{row['step']:>7} {row['phase']:>10} " + "  ".join(
            f"{results[a][i]['losses'][first]:>10.3f}"
            for a in ["frozen", "online", "guarded"]))
    print("=" * 84)

    print(f"\nALL DOMAINS AT THE END:")
    for a in ["frozen", "online", "guarded"]:
        last = results[a][-1]["losses"]
        print(f"  {a:>8}: " + "  ".join(
            f"{d} {last[d]:.3f}" for d in order))

    def change(arm, domain):
        return (results[arm][-1]["losses"][domain]
                - results[arm][0]["losses"][domain])

    print(f"\nCHANGE IN LOSS FROM START TO END (negative is better):")
    for a in ["frozen", "online", "guarded"]:
        print(f"  {a:>8}: " + "  ".join(
            f"{d} {change(a, d):+.3f}" for d in order))

    on_first = change("online", first)
    gu_first = change("guarded", first)

    print(f"\nTHE EARLY DOMAIN ('{first}'), abandoned two phases ago:")
    print(f"  online  {on_first:+.3f}")
    print(f"  guarded {gu_first:+.3f}")
    print(f"  guarded advantage: {on_first - gu_first:+.3f}")

    print()
    if on_first > 0.02 and gu_first < on_first - 0.02:
        print("  RETENTION WORKS IN THE WEIGHTS ON REAL TEXT. The")
        print("  unprotected model found the early domain LESS familiar")
        print("  after moving away from it; the protected one held on.")
        print("  That claim has previously only been shown on weather and")
        print("  on invented sentences.")
    elif on_first <= 0.02:
        print("  NO FORGETTING OCCURRED. The unprotected model did not lose")
        print("  the early domain, so there was nothing to protect. The")
        print("  domains are probably too similar — real Wikipedia prose")
        print("  shares most of its distribution regardless of topic.")
    else:
        print("  RETENTION DID NOT HELP. Both arms lost the early domain")
        print("  by a similar amount, so the mechanism does not transfer")
        print("  to real text however well it worked on weather.")

    print("""
The frozen column is the reference: it never learns, so its loss should
barely move. Any drift there is measurement noise and sets the scale for
what counts as a real change in the other two.

Watch the other domains too. A protected model that holds the early domain
by refusing to learn the later ones has not retained anything — it has just
stopped learning. Retention means holding the old WHILE acquiring the new.
""")
    with open("domains.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote domains.json")