"""Does any REAL stream contradict itself enough for retention to matter?

The retention mechanism helps on contradictory streams — grid worlds with
opposing rules, weather across seasons — and does nothing, or slight harm,
on ordinary real text. Wikipedia domains changed subject without changing
distribution, and the layer paid its cost for nothing.

That leaves an open and important question: is any REALISTIC stream
contradictory enough to need this? If not, the mechanism has no home outside
constructed worlds and a large part of the system should be cut.

THIS TESTS IT ON REAL DATA, with nothing constructed. Weather from climates
whose rules genuinely oppose each other:

  Reykjavik   cold, wet, maritime. Falling pressure means rain and a small
              temperature change.
  Phoenix     hot, dry, continental. The same readings mean something
              different, and the daily swing dwarfs the synoptic one.
  Singapore   equatorial. Temperature barely moves; humidity carries the
              signal instead.

A model that learns to predict temperature in Reykjavik has learned
relationships that are WRONG in Phoenix. Not merely different — actively
misleading, the way "the door opens" contradicts "the door never opens".
That is what the grid worlds constructed by hand and what Wikipedia domains
lacked.

Three phases, one climate each, in order. Then measure what happened to the
first.

  online loses Reykjavik, guarded holds it -> retention has a home on real
      data, and the constructed worlds were not the only place it works.
  neither loses it -> even opposing climates are not contradictory enough,
      and the honest conclusion is that this mechanism belongs to
      constructed problems. That would be a strong reason to simplify the
      system.
  both lose it equally -> the mechanism does not transfer to real
      continuous data at all, which would contradict the weather result
      and need explaining.

A CONTRADICTION CHECK RUNS FIRST, before any training: a model trained on
one climate is tested on another. If it transfers well, the climates do not
contradict and the test cannot answer the question. Two experiments this
week were void for want of a check like this.
"""

import json
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from sensor import fetch, build_stream, CLASSES, VARIABLES
from sensorrun2 import add_lags
from stability import StabilityLayer


CLIMATES = ["reykjavik", "phoenix", "singapore"]
HIDDEN = 128
LR = 3e-4
SEEDS = [0, 1]
EPISODE = 720
LAG = 3
N_FEAT = len(VARIABLES) * (LAG + 1)
PHASE_STEPS = 12000
TEST_STEPS = 4000
SEQ_LEN = 10
SEQ_COUNT = 2
CHECKPOINT = 3000


def encode(feats):
    return torch.tensor(feats, dtype=torch.float32).unsqueeze(0)


class GRU(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.enc = nn.Linear(N_FEAT, HIDDEN)
        self.cell = nn.GRUCell(HIDDEN, HIDDEN)
        self.head = nn.Linear(HIDDEN, len(CLASSES))

    def forward(self, x, h=None):
        e = F.relu(self.enc(x))
        h = self.cell(e, h)
        return self.head(h), h


class Backend:
    def __init__(self, seed=0):
        self.net = GRU(seed)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.h = None

    def begin_sequence(self):
        self.h = None

    def _xy(self, item):
        return encode(item[0]), torch.tensor([CLASSES.index(item[1])])

    def score(self, item):
        self.net.eval()
        with torch.no_grad():
            x, y = self._xy(item)
            logits, _ = self.net(x, self.h)
            return float(F.cross_entropy(logits, y).item()), None

    def update(self, item, steps):
        self.net.train()
        last = None
        for _ in range(steps):
            x, y = self._xy(item)
            logits, h = self.net(x, self.h)
            loss = F.cross_entropy(logits, y)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.h = h.detach()
            last = float(loss.item())
        return last

    def reset_state(self):
        self.h = None

    def snapshot(self):
        return {k: v.detach().clone()
                for k, v in self.net.state_dict().items()}

    def restore(self, state):
        self.net.load_state_dict(state)

    def on_rollback(self):
        for g in self.opt.param_groups:
            g["lr"] = max(1e-6, g["lr"] * 0.5)


def load(climate):
    """Real hourly readings. Cached by the fetch helper."""
    return add_lags(build_stream(fetch(climate)))


def train_plain(net, data):
    """Ordinary training, for the contradiction check."""
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    h = None
    net.train()
    for i, item in enumerate(data):
        if i % EPISODE == 0:
            h = None
        logits, h = net(encode(item[0]), h)
        loss = F.cross_entropy(
            logits, torch.tensor([CLASSES.index(item[1])]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        h = h.detach()
    return net


def evaluate(net, data):
    net.eval()
    h = None
    hit = seen = 0
    with torch.no_grad():
        for i, item in enumerate(data):
            if i % EPISODE == 0:
                h = None
            logits, h = net(encode(item[0]), h)
            hit += int(logits.argmax(1).item()) == CLASSES.index(item[1])
            seen += 1
    return 100.0 * hit / seen


def run(arm, phases, tests, seed):
    b = Backend(seed)
    layer = None
    if arm == "guarded":
        layer = StabilityLayer(
            b, canary=phases[0][1][:40], seed=seed,
            window=200, warmup=30, top_fraction=0.50,
            coherence_veto=1e9, loss_floor=0.02, steps_per_update=1,
            rehearse_per_item=5, rehearse_count=SEQ_COUNT,
            rehearse_steps=1, anchor_size=100, buffer_size=500,
            sequence_len=SEQ_LEN, replay_policy="uniform",
            guard=True, guard_per_item=2000, canary_tolerance=0.5)

    curve = [dict(step=0, phase="start",
                  scores={c: evaluate(b.net, tests[c]) for c in CLIMATES})]
    step = 0
    t0 = time.time()

    for climate, data in phases:
        for item in data:
            if step % EPISODE == 0:
                b.reset_state()
            if arm == "frozen":
                pass
            elif layer is not None:
                layer.observe(item)
            else:
                b.update(item, 1)
            step += 1

            if step % CHECKPOINT == 0:
                b.reset_state()
                scores = {c: evaluate(b.net, tests[c]) for c in CLIMATES}
                st = layer.summary() if layer else {}
                curve.append(dict(step=step, phase=climate, scores=scores,
                                  rollbacks=st.get("rollbacks", 0)))
                print(f"    {step:>6} in {climate:>10}: " + "  ".join(
                    f"{c[:4]} {scores[c]:5.1f}" for c in CLIMATES) +
                    f"   rb {st.get('rollbacks', 0)}  "
                    f"{(time.time() - t0) / 60:5.1f}m", flush=True)

    del b, layer
    return curve


if __name__ == "__main__":
    print("Does any REAL stream contradict itself enough for retention?\n")
    print(f"climates: {' -> '.join(CLIMATES)}, {PHASE_STEPS} steps each\n")

    data = {}
    for c in CLIMATES:
        s = load(c)
        data[c] = s
        print(f"  {c}: {len(s)} hourly readings")

    tests = {c: data[c][-TEST_STEPS:] for c in CLIMATES}
    phases = [(c, data[c][:PHASE_STEPS]) for c in CLIMATES]

    # Do these climates actually contradict? A model trained on one and
    # tested on another should do BADLY if they do. If it transfers well,
    # there is no contradiction and the test cannot answer the question.
    print("\ncontradiction check: train on one climate, test on another")
    net = train_plain(GRU(0), data[CLIMATES[0]][:8000])
    own = evaluate(net, tests[CLIMATES[0]])
    others = {c: evaluate(net, tests[c]) for c in CLIMATES[1:]}
    del net
    print(f"  trained on {CLIMATES[0]}: own {own:.2f}%, " +
          ", ".join(f"{c} {v:.2f}%" for c, v in others.items()))
    drop = own - min(others.values())
    print(f"  largest drop when transferred: {drop:.2f} points")
    if drop < 5:
        print("\n  ABORT: the climates transfer well, so they do not\n"
              "  contradict and retention has nothing to protect here.\n"
              "  That is itself an answer: real weather across climates is\n"
              "  not a contradictory stream.")
        raise SystemExit
    print("  good: the climates genuinely disagree\n")

    def mean(xs):
        return sum(xs) / len(xs)

    results = {}
    for arm in ["frozen", "online", "guarded"]:
        print(f"--- {arm} ---", flush=True)
        curves = [run(arm, phases, tests, s) for s in SEEDS]
        results[arm] = curves
        print(flush=True)

    first = CLIMATES[0]
    print("=" * 80)
    print(f"ACCURACY ON '{first.upper()}' — learned first, then abandoned")
    print(f"{'step':>7} {'phase':>11} " + "  ".join(
        f"{a:>9}" for a in ["frozen", "online", "guarded"]))
    print("-" * 80)
    n = min(len(results[a][0]) for a in results)
    for i in range(n):
        row = results["frozen"][0][i]
        print(f"{row['step']:>7} {row['phase']:>11} " + "  ".join(
            f"{mean([c[i]['scores'][first] for c in results[a]]):>8.2f}%"
            for a in ["frozen", "online", "guarded"]))
    print("=" * 80)

    print(f"\nALL CLIMATES AT THE END:")
    for a in ["frozen", "online", "guarded"]:
        last = {c: mean([cu[-1]["scores"][c] for cu in results[a]])
                for c in CLIMATES}
        print(f"  {a:>8}: " + "  ".join(
            f"{c} {last[c]:5.2f}%" for c in CLIMATES))

    def change(arm, climate):
        return (mean([c[-1]["scores"][climate] for c in results[arm]])
                - mean([c[0]["scores"][climate] for c in results[arm]]))

    print(f"\nCHANGE FROM START TO END:")
    for a in ["online", "guarded"]:
        print(f"  {a:>8}: " + "  ".join(
            f"{c} {change(a, c):+6.2f}" for c in CLIMATES))

    on = change("online", first)
    gu = change("guarded", first)
    print(f"\nTHE EARLY CLIMATE ('{first}'), two phases ago:")
    print(f"  online  {on:+.2f}")
    print(f"  guarded {gu:+.2f}")
    print(f"  guarded advantage: {gu - on:+.2f}")

    print()
    if on < -3 and gu > on + 3:
        print("  RETENTION HAS A HOME ON REAL DATA. The unprotected model")
        print("  lost the early climate; the protected one held it. Real")
        print("  streams can contradict themselves enough for this to")
        print("  matter, and the mechanism is not confined to constructed")
        print("  worlds.")
    elif on >= -3:
        print("  NO FORGETTING. Even opposing climates did not make the")
        print("  model lose the first one. Combined with the Wikipedia")
        print("  result, that is a strong case that this mechanism belongs")
        print("  to constructed problems and a real system may not need it.")
    else:
        print("  RETENTION DID NOT HELP. The early climate was lost by both")
        print("  arms, which contradicts the earlier within-climate weather")
        print("  result and needs explaining before either is trusted.")

    print("""
Watch the other climates too. A protected model that holds the first climate
by failing to learn the later ones has not retained anything — it has
stopped learning. The all-climates table catches that.

This is the question the Wikipedia result forced: the mechanism works where
streams contradict, and real text does not contradict. If real weather
across opposing climates does not either, the honest conclusion is that a
deployed system may not need this machinery, and the project should say so.
""")
    with open("contradict.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote contradict.json")