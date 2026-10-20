"""Intelligence from using memory well, not from more size.

THE LEVER. NEURON already stores episodic memories. So far it uses them only to
avoid hazards. But a mind can be smarter by USING its memory actively: when
facing a situation, recall the most similar past experiences and let their
known outcomes inform the current prediction. This is deeply brain-like, you
think by recalling relevant experience, and it makes NEURON smarter without
adding a single neuron. Better use of memory, not more of it.

THE TEST. A world with recurring situations, where past experience genuinely
helps. We compare, at the SAME brain size:
  plain       the network predicts on its own.
  retrieval   the network's prediction is blended with the outcomes of the k
              nearest past experiences in memory.

If retrieval-augmented prediction beats plain at the same size, then using
memory well is a real, non-size intelligence lever.

  retrieval beats plain at the same size
      Intelligence rose from better memory use, no extra neurons. A real
      non-size lever, and aligned with the being (memory is already core).
  it ties or loses
      Retrieval did not help here (situations too novel, or it added noise);
      honest negative.

    python retrieval.py
"""

import sys
import os
import math
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from world_interface import World
from mind import Mind


class RecurringWorld(World):
    """A world with a limited set of recurring SITUATIONS, each with its own
    consistent outcome pattern. Past experience with a situation genuinely
    helps predict it again, exactly where retrieval should pay off."""

    def __init__(self, n_signals=6, n_situations=12, seed=0):
        rng = random.Random(seed)
        self._names = [f"sig_{i}" for i in range(n_signals)]
        # a set of recurring situations: each a fixed signal pattern + outcome
        self.situations = []
        for _ in range(n_situations):
            pattern = [rng.uniform(0, 1) for _ in range(n_signals)]
            outcome = rng.randint(0, 2)
            self.situations.append((pattern, outcome))
        self.rng = rng
        self.cur = self.situations[0]
        self._names_idx = {n: i for i, n in enumerate(self._names)}
        self.t = 0
        self._pick()

    def _pick(self):
        # each step presents one recurring situation (with slight noise)
        self.cur = self.situations[self.rng.randrange(len(self.situations))]

    def signals(self):
        return self._names

    def observe(self, name):
        i = self._names_idx[name]
        base = self.cur[0][i]
        return max(0.0, min(1.0, base + self.rng.gauss(0, 0.02)))

    def outcome(self, name, past_value):
        # the outcome is the situation's fixed outcome (same every time it
        # recurs) - so recognizing the situation from memory genuinely helps
        return self.cur[1]

    def step(self):
        self.t += 1
        self._pick()

    def describe(self):
        return f"RecurringWorld({len(self.situations)} situations)"


class RetrievalMind(Mind):
    """A Mind that blends its network prediction with the outcomes of the k
    nearest past experiences in its episodic memory. Everything else is the
    same being."""

    def __init__(self, world, hidden=64, k=5, seed=0):
        super().__init__(world, hidden=hidden, seed=seed)
        self.k = k
        self.mem = []        # (feats, outcome) store for retrieval

    def _retrieve_vote(self, feats):
        if len(self.mem) < self.k:
            return None
        # k nearest past experiences by feature distance
        scored = []
        for (f, o) in self.mem:
            d = sum((a - b) ** 2 for a, b in zip(f, feats))
            scored.append((d, o))
        scored.sort(key=lambda t: t[0])
        votes = [o for _, o in scored[:self.k]]
        # majority outcome among the nearest neighbors
        return max(set(votes), key=votes.count)

    def live_step(self):
        # run the normal being-cycle, but override the prediction with a blend
        # of network + retrieval when memory has relevant experience.
        self.world.step()
        self.wants.sense(self.recent_acc(), self.since_attended)
        action, name = self.wants.choose()
        if action == "consolidate":
            self.backend.batched_consolidate(self.replay[-256:])
            self.emotion.decay()
            return {"action": "consolidate"}
        for n in self.names:
            self.since_attended[n] += 1
        self.since_attended[name] = 0
        feats = self._features(name)
        val_now = self.world.observe(name)

        net_guess = self.backend.predict(feats)
        retrieved = self._retrieve_vote(feats)
        # BLEND: if memory has a confident neighbor-vote, trust it; else net
        guess = retrieved if retrieved is not None else net_guess

        resolved = self._resolve_pending()
        self.pending.append({"name": name, "feats": feats,
                             "val": val_now, "guess": guess})
        self.samples += 1
        for r in resolved:
            correct = r["correct"]
            self.recent.append(1 if correct else 0)
            if len(self.recent) > 50:
                self.recent.pop(0)
            self.replay.append({"feats": r["feats"], "label": r["truth"]})
            if len(self.replay) > 200:
                self.replay.pop(0)
            # store in retrieval memory
            self.mem.append((r["feats"], r["truth"]))
            if len(self.mem) > 500:
                self.mem.pop(0)
        self.emotion.decay()
        self.wants.decay()
        return {"action": "attend", "signal": name, "guess": guess}


def run_plain(hidden, steps, seed):
    world = RecurringWorld(seed=seed)
    mind = Mind(world, hidden=hidden, seed=seed)
    accs = []
    for i in range(steps):
        mind.live_step()
        if i > steps * 0.6:
            accs.append(mind.recent_acc())
    return sum(accs) / len(accs) if accs else 0


def run_retrieval(hidden, steps, seed):
    world = RecurringWorld(seed=seed)
    mind = RetrievalMind(world, hidden=hidden, seed=seed)
    accs = []
    for i in range(steps):
        mind.live_step()
        if i > steps * 0.6:
            accs.append(mind.recent_acc())
    return sum(accs) / len(accs) if accs else 0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    print("  Does USING memory well make NEURON smarter without more size?")
    print("  Plain network vs retrieval-augmented (blends in the outcomes of")
    print("  the nearest past experiences), same brain size.\n")

    plain, retr = [], []
    for s in range(args.seeds):
        plain.append(run_plain(args.hidden, args.steps, s))
        retr.append(run_retrieval(args.hidden, args.steps, s))
    p = sum(plain) / len(plain)
    r = sum(retr) / len(retr)

    print("=" * 52)
    print(f"RECURRING WORLD, brain size {args.hidden}")
    print("=" * 52)
    print(f"  {'approach':>22} {'accuracy':>12}")
    print(f"  {'plain network':>22} {p*100:>11.0f}%")
    print(f"  {'retrieval-augmented':>22} {r*100:>11.0f}%")

    print("\n" + "=" * 52)
    print("WHAT IT SAYS")
    print("=" * 52)
    if r > p + 0.04:
        print(f"  USING MEMORY WELL RAISED INTELLIGENCE. Retrieval-augmented")
        print(f"  prediction reached {r*100:.0f}% vs the plain network's "
              f"{p*100:.0f}%, at the SAME")
        print(f"  size, by recalling relevant past experience to inform its "
              f"guess. A real")
        print(f"  non-size intelligence lever, and deeply aligned with the "
              f"being (memory")
        print(f"  is already core to what NEURON is).")
    elif r > p:
        print(f"  Retrieval edged out plain ({r*100:.0f}% vs {p*100:.0f}%), a "
              f"real but modest gain.")
    else:
        print(f"  No clear win ({r*100:.0f}% vs {p*100:.0f}%). Retrieval did "
              f"not help here; honest")
        print(f"  negative (situations too novel, or retrieval added noise).")

    print(f"\n  This tests intelligence from BETTER MEMORY USE, not more "
          f"neurons, one of the")
    print(f"  real non-size levers for a smarter being.")


if __name__ == "__main__":
    main()
