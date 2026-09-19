"""Does the retention result hold on a real language model?

The strongest result in this project: on real weather, across five climates,
sequence replay stopped a model forgetting material the stream had moved
past. Unprotected arms lost 26 points on the season furthest from the end of
training; the protected arm lost nothing. Proved causally — shuffle the
stream so there is nothing to retain, and the advantage vanishes.

Then the mechanisms held across a 36x network size increase, guarded beating
online at every width.

This is the same question at 7 billion parameters, on a language model
someone might actually use.

THE STREAM: three topic phases, each a block of sentences about a distinct
domain, seen once in order and never revisited. Early material is the first
phase, which the model has not seen for two thirds of the run.

  phase 1  a fictional coastal town's civic history
  phase 2  an invented card game's rules
  phase 3  a made-up botanical survey

All invented, so the model cannot already know any of it, and the three
domains share no vocabulary — so learning phase 3 has every opportunity to
overwrite phase 1.

WHAT IS MEASURED: surprise on held-out sentences from EACH phase, at the end.
Not question answering. The Aug 16-18 work established that gradient updates
on statements produce string memorisation rather than answerable knowledge,
and that surprise reduction is not a knowledge metric. So this measures
exactly what it can honestly claim to measure — whether the model retains
familiarity with early material, or loses it.

Arms:
  frozen     no learning. The floor, and the fabrication control.
  online     learns from every sentence. No protection.
  guarded    the same through stability.py with sequence replay.

If guarded holds phase 1 while online loses it, the retention result scales
to a real model. That is the claim worth having.
"""

import json
import time

import torch

from llm_backend import LLMBackend
from stability import StabilityLayer


MODEL = "Qwen/Qwen2.5-7B-Instruct"
SEEDS = [0]
LR = 1e-4              # the Aug 20 sweep: 1e-4 keeps the model honest
SEQ_LEN = 8
SEQ_COUNT = 2
REPEATS = 4            # how many times each phase's stream is presented


PHASES = {
    "harbour": [
        "The town of Marrenport was founded on a shingle bar at the mouth "
        "of the Feltwater estuary.",
        "Its first stone pier was completed in the spring of eighteen "
        "twelve and paid for by the herring trade.",
        "Marrenport's council met in the upper room of the customs house "
        "until the new hall opened.",
        "The Feltwater silted badly after the eighteen forties and larger "
        "vessels stopped calling.",
        "A lifeboat station was established at Marrenport following the "
        "loss of the collier Ardent.",
        "The town's population peaked at just over four thousand before "
        "the fishing declined.",
        "Marrenport's harbour master kept a daily log of every vessel that "
        "entered the estuary.",
        "The shingle bar shifted eastward each winter and had to be "
        "dredged every second year.",
        "Wooden warehouses lined the eastern quay until a fire destroyed "
        "most of them.",
        "The council levied a small toll on every barrel landed at the "
        "stone pier.",
        "Marrenport's fishing fleet numbered sixty boats at its largest.",
        "The customs house still stands at the landward end of the pier.",
    ],
    "cardgame": [
        "In the game of Vessling, each player begins with a hand of seven "
        "tiles drawn from the common pool.",
        "A Vessling tile shows a rank from one to nine and one of four "
        "suits called stems.",
        "Play proceeds clockwise and a turn consists of either laying a "
        "tile or drawing from the pool.",
        "A run in Vessling is three or more tiles of the same stem in "
        "ascending rank.",
        "Laying a completed run allows the player an immediate second turn.",
        "The pool is exhausted when fewer than four tiles remain face down.",
        "A player holding no tiles at the end of a turn wins the round "
        "outright.",
        "Vessling scoring counts the ranks left in the losing players' "
        "hands.",
        "A tile of rank nine may substitute for any rank within a run.",
        "Two players may not lay tiles of the same stem in consecutive "
        "turns.",
        "The dealer in Vessling rotates leftward after every completed "
        "round.",
        "A round is void if the pool empties before any player lays a run.",
    ],
    "botany": [
        "The sedge Carex ollmanii grows only on the limestone pavements of "
        "the Kerrow uplands.",
        "Its flowering spikes appear in late June and are a dull "
        "purple-brown.",
        "Carex ollmanii was first described by the surveyor Alice Ollman in "
        "her upland catalogue.",
        "The sedge tolerates thin alkaline soils and fails entirely on "
        "acid ground.",
        "Grazing pressure from upland sheep has reduced the Kerrow "
        "population considerably.",
        "Seed of Carex ollmanii remains viable for roughly three seasons in "
        "dry storage.",
        "The plant spreads by short rhizomes rather than by seed in most "
        "years.",
        "Ollman recorded fourteen separate colonies during her original "
        "survey.",
        "The sedge is most often found in the shelter of limestone "
        "grykes.",
        "Its leaves are narrower than those of the commoner upland sedges.",
        "Carex ollmanii has never been recorded outside the Kerrow "
        "uplands.",
        "The species flowers poorly in seasons following a dry spring.",
    ],
}

# Held-out sentences from the same domains, never trained on. Retention is
# measured as surprise on these.
PROBES = {
    "harbour": [
        "The stone pier at Marrenport was built with money from the "
        "herring trade.",
        "Vessels stopped calling at Marrenport once the Feltwater silted "
        "up.",
        "Marrenport's council once met above the customs house.",
    ],
    "cardgame": [
        "A Vessling player who empties their hand wins the round.",
        "Runs in Vessling are built from tiles sharing a stem.",
        "The Vessling pool is spent when few tiles remain face down.",
    ],
    "botany": [
        "Carex ollmanii is confined to the limestone of the Kerrow "
        "uplands.",
        "Alice Ollman catalogued the upland sedges including Carex "
        "ollmanii.",
        "The sedge fails on acid ground and needs thin alkaline soil.",
    ],
}

CANARY = [
    "Water freezes into ice when the temperature drops below zero.",
    "She placed the letter on the table and walked toward the window.",
    "Most birds have feathers and many of them are able to fly.",
    "He counted the coins twice before putting them back in the drawer.",
    "Bread is usually made from flour, water, yeast and salt.",
]

ORDER = ["harbour", "cardgame", "botany"]


def build_stream():
    """Three phases, in order, each repeated a few times within its phase.

    Repetition happens INSIDE a phase, so the phase is genuinely finished
    before the next begins. That is what makes early material old.
    """
    stream = []
    for phase in ORDER:
        for _ in range(REPEATS):
            stream.extend(PHASES[phase])
    return stream


def measure(backend):
    """Mean surprise on the held-out probes for each phase. Lower is more
    familiar. This is a FAMILIARITY measure, not a knowledge measure — the
    Aug 16-18 work established that surprise reduction does not imply the
    model can answer questions about the material."""
    out = {}
    for phase, sents in PROBES.items():
        total = 0.0
        for s in sents:
            v, _ = backend.score(s)
            total += v
        out[phase] = total / len(sents)
    return out


def controls(backend):
    total = 0.0
    for s in CANARY:
        v, _ = backend.score(s)
        total += v
    return total / len(CANARY)


def run(mode, seed):
    torch.manual_seed(seed)
    t0 = time.time()
    backend = LLMBackend(model_name=MODEL, focus_alpha=0.0)
    for g in backend.optimizer.param_groups:
        g["lr"] = LR

    before = measure(backend)
    ctrl_before = controls(backend)

    stats = {}
    if mode != "frozen":
        stream = build_stream()
        layer = None
        if mode == "guarded":
            layer = StabilityLayer(
                backend, canary=CANARY, seed=seed,
                window=100, warmup=20, top_fraction=0.50,
                coherence_veto=1.3, loss_floor=0.35, steps_per_update=1,
                rehearse_per_item=4, rehearse_count=SEQ_COUNT,
                rehearse_steps=1, anchor_size=40, buffer_size=200,
                sequence_len=SEQ_LEN, replay_policy="uniform",
                guard=True, guard_per_item=40, canary_tolerance=0.40)

        for i, sent in enumerate(stream):
            if layer is not None:
                layer.observe(sent)
            else:
                backend.update(sent, 1)
            if (i + 1) % 36 == 0:
                print(f"    {mode}: {i + 1}/{len(stream)}", flush=True)

        stats = layer.summary() if layer else {}

    after = measure(backend)
    ctrl_after = controls(backend)

    del backend
    torch.cuda.empty_cache()
    return dict(before=before, after=after,
                ctrl_before=ctrl_before, ctrl_after=ctrl_after,
                stats=stats, minutes=(time.time() - t0) / 60)


ARMS = ["frozen", "online", "guarded"]


if __name__ == "__main__":
    stream = build_stream()
    print(f"7B retention test: {len(ORDER)} phases x {REPEATS} repeats "
          f"= {len(stream)} sentences")
    print(f"order: {' -> '.join(ORDER)}")
    print(f"'{ORDER[0]}' is the early material, unseen for the last "
          f"{100 * (len(ORDER) - 1) // len(ORDER)}% of the stream\n")

    results = {}
    for mode in ARMS:
        print(f"--- {mode} ---", flush=True)
        r = run(mode, SEEDS[0])
        results[mode] = r
        deltas = {p: r["after"][p] - r["before"][p] for p in ORDER}
        print(f"  surprise change per phase: " +
              "  ".join(f"{p} {deltas[p]:+.3f}" for p in ORDER))
        print(f"  controls {r['ctrl_before']:.3f} -> "
              f"{r['ctrl_after']:.3f}  ({r['minutes']:.1f} min)")
        if r["stats"]:
            s = r["stats"]
            print(f"  gate {s['updates']}  rehearsals {s['rehearsals']}  "
                  f"buffer {s['buffer']}  rollbacks {s['rollbacks']}  "
                  f"health {s['health']:+.4f}")
        print(flush=True)

    print("=" * 74)
    print("SURPRISE ON HELD-OUT PROBES, lower is more familiar")
    print(f"{'arm':>9} " + "  ".join(f"{p:>12}" for p in ORDER) +
          f" {'controls':>10}")
    print("-" * 74)
    for mode in ARMS:
        r = results[mode]
        cells = "  ".join(f"{r['after'][p]:>12.3f}" for p in ORDER)
        print(f"{mode:>9} {cells} {r['ctrl_after']:>10.3f}")
    print("=" * 74)

    print("\nCHANGE FROM BEFORE, negative means it became more familiar")
    print(f"{'arm':>9} " + "  ".join(f"{p:>12}" for p in ORDER))
    print("-" * 74)
    for mode in ARMS:
        r = results[mode]
        cells = "  ".join(
            f"{r['after'][p] - r['before'][p]:>+12.3f}" for p in ORDER)
        print(f"{mode:>9} {cells}")
    print("=" * 74)

    early = ORDER[0]
    on = results["online"]["after"][early] - results["online"]["before"][early]
    gu = results["guarded"]["after"][early] - results["guarded"]["before"][early]
    late = ORDER[-1]
    on_l = results["online"]["after"][late] - results["online"]["before"][late]
    gu_l = results["guarded"]["after"][late] - results["guarded"]["before"][late]

    print(f"\nEARLY MATERIAL ('{early}', unseen for two thirds of the run):")
    print(f"  online  {on:+.3f}")
    print(f"  guarded {gu:+.3f}")
    print(f"  guarded advantage: {on - gu:+.3f}")
    print(f"\nMOST RECENT MATERIAL ('{late}'):")
    print(f"  online  {on_l:+.3f}")
    print(f"  guarded {gu_l:+.3f}")

    print("""
The claim being tested is retention, not knowledge. Surprise falling means
the model finds the material more familiar; the Aug 16-18 work established
that this does NOT mean it can answer questions about it, so nothing more
should be read into it.

  guarded holds the early phase while online loses it -> the retention
      result scales from a 100,000-parameter network to 7 billion. That is
      the strongest claim available from this project.

  both hold it -> at this stream length there is nothing to forget, and a
      longer or more interfering stream is needed.

  both lose it -> forgetting at 7B is not what the small-scale work
      measured, and the mechanism does not transfer.

Watch the controls column. If it rises, the model is degrading as a language
model, which is the failure the canary guard exists to catch.
""")
    with open("sevenb.json", "w") as f:
        json.dump({k: dict(before=v["before"], after=v["after"],
                           ctrl_before=v["ctrl_before"],
                           ctrl_after=v["ctrl_after"],
                           stats={kk: vv for kk, vv in v["stats"].items()})
                   for k, v in results.items()}, f, indent=2, default=str)
    print("wrote sevenb.json")