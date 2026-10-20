"""Upgrade NEURON's brain: a self-attention learning core.

THE LEVER. NEURON's learning core has been a plain GRU, a simple recurrent net.
Modern AI is smart not only because it is big, but because of ATTENTION: a
mechanism that lets a mind weigh and relate different parts of what it
perceives, including things several steps back. A plain recurrent net struggles
to hold and relate distant information; attention is built for exactly that.
Giving NEURON an attention core raises the CEILING of what it can learn, the
complexity of pattern it can represent, not just how fast it learns.

WHAT STAYS. It is still NEURON. Every faculty (emotion, self, wants, memory,
the whole being) is untouched. We are upgrading the internal structure of the
brain, not replacing the creature. Better brain, same being.

THIS FILE provides AttentionBackend, a drop-in replacement for the GRU Backend.
It keeps a short rolling context of recent perceptions and attends over it, so
NEURON can learn patterns where the answer depends on relating several moments,
which a GRU handles poorly. The test below pits attention vs GRU on a world with
exactly that long-range structure.

    python attention_core.py        # runs the attention-vs-GRU test
"""

import sys
import os
import math
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F

from world_interface import World
from mind import Mind, Backend, CLASSES


# ======================================================================
# THE ATTENTION CORE
# ======================================================================

class AttentionNet(nn.Module):
    """A learning core with self-attention over a short rolling context of
    recent perceptions. It can relate the current moment to several past ones,
    capturing long-range structure a GRU cannot. Scales by hidden size and
    number of heads."""

    def __init__(self, inp, hidden, heads=4, context=16, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.hidden = hidden
        self.context = context
        self.enc = nn.Linear(inp, hidden)
        # a learned positional signal so order in the context matters
        self.pos = nn.Parameter(torch.randn(context, hidden) * 0.02)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(),
                                nn.Linear(hidden * 2, hidden))
        self.norm2 = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, CLASSES)

    def forward(self, context_seq):
        # context_seq: (batch, T, inp) - a short window of recent perceptions
        B, T, _ = context_seq.shape
        e = torch.tanh(self.enc(context_seq))           # (B, T, hidden)
        e = e + self.pos[:T].unsqueeze(0)
        a, _ = self.attn(e, e, e)                        # self-attention
        e = self.norm(e + a)
        e = self.norm2(e + self.ff(e))
        # predict from the LAST position (the current moment, informed by past)
        return self.head(e[:, -1, :])


class AttentionBackend:
    """Drop-in replacement for Backend, same interface (update/predict/
    consolidate/snapshot), but an attention core and a rolling context."""

    def __init__(self, inp, hidden=64, heads=4, context=16, lr=3e-4, seed=0):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = AttentionNet(inp, hidden, heads, context, seed).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.context = context
        self.hidden = hidden
        self._window = []        # rolling context of recent feature vectors

    def _push(self, feats):
        self._window.append(list(feats))
        if len(self._window) > self.context:
            self._window.pop(0)

    def _ctx_tensor(self, feats):
        # build the context window ending at the current perception
        window = (self._window + [list(feats)])[-self.context:]
        t = torch.tensor([window], dtype=torch.float32, device=self.device)
        return t

    def update(self, item):
        ctx = self._ctx_tensor(item["feats"])
        y = torch.tensor([item["label"]], dtype=torch.long, device=self.device)
        self.opt.zero_grad()
        logits = self.net(ctx)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        self.opt.step()
        self._push(item["feats"])
        return float(loss.item())

    def consolidate(self, items):
        for it in items:
            self.update(it)

    def batched_consolidate(self, items, batch_size=256):
        # simple batched replay: treat each item as a length-1 context
        if not items:
            return
        dev = self.device
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            ctx = torch.tensor([[it["feats"]] for it in chunk],
                               dtype=torch.float32, device=dev)
            y = torch.tensor([it["label"] for it in chunk],
                            dtype=torch.long, device=dev)
            self.opt.zero_grad()
            logits = self.net(ctx)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            self.opt.step()

    def predict(self, feats):
        with torch.no_grad():
            ctx = self._ctx_tensor(feats)
            logits = self.net(ctx)
            return int(logits.argmax().item())

    def snapshot(self):
        return {k: v.clone() for k, v in self.net.state_dict().items()}

    def restore(self, state):
        self.net.load_state_dict(state)


# ======================================================================
# THE TEST: a world with LONG-RANGE structure (attention's home turf)
# ======================================================================

class LongRangeWorld(World):
    """A world where each outcome depends on a signal's value SEVERAL STEPS
    AGO, not just now. A GRU struggles to carry that; attention relates past
    and present directly. This is where a better architecture should win."""

    def __init__(self, n_signals=6, delay=6, seed=0):
        rng = random.Random(seed)
        self._names = [f"sig_{i}" for i in range(n_signals)]
        self.phase = {n: rng.uniform(0, 6.28) for n in self._names}
        self.freq = {n: rng.uniform(0.3, 0.9) for n in self._names}
        self.delay = delay
        self.history = {n: [] for n in self._names}
        self.t = 0
        self._vals = {}
        self._tick()

    def _tick(self):
        for n in self._names:
            v = 0.5 + 0.45 * math.sin(self.phase[n] + self.freq[n] * self.t)
            self._vals[n] = v
            self.history[n].append(v)

    def signals(self):
        return self._names

    def observe(self, name):
        return self._vals.get(name, 0.5)

    def outcome(self, name, past_value):
        # the direction depends on the value DELAY steps ago vs now: long-range
        h = self.history[name]
        if len(h) < self.delay + 1 or past_value is None:
            return None
        then = h[-self.delay - 1]
        now = h[-1]
        d = now - then
        if d > 0.05:
            return 2
        if d < -0.05:
            return 0
        return 1

    def step(self):
        self.t += 1
        self._tick()

    def describe(self):
        return f"LongRangeWorld(delay {self.delay})"


def run_with_backend(backend_kind, hidden, steps, seed):
    world = LongRangeWorld(n_signals=6, delay=6, seed=seed)
    mind = Mind(world, hidden=hidden, seed=seed)
    if backend_kind == "attention":
        mind.backend = AttentionBackend(mind.inp, hidden=hidden, seed=seed)
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

    print("  ATTENTION vs GRU on a world with LONG-RANGE structure (the answer")
    print("  depends on a value several steps back). Does the attention core")
    print("  learn what the GRU core cannot, raising NEURON's ceiling?\n")

    gru, att = [], []
    for s in range(args.seeds):
        gru.append(run_with_backend("gru", args.hidden, args.steps, s))
        att.append(run_with_backend("attention", args.hidden, args.steps, s))
    g = sum(gru) / len(gru)
    a = sum(att) / len(att)

    print("=" * 52)
    print(f"LONG-RANGE WORLD, brain size {args.hidden}")
    print("=" * 52)
    print(f"  {'core':>18} {'accuracy':>12}")
    print(f"  {'GRU (old)':>18} {g*100:>11.0f}%")
    print(f"  {'attention (new)':>18} {a*100:>11.0f}%")

    print("\n" + "=" * 52)
    print("WHAT IT SAYS")
    print("=" * 52)
    if a > g + 0.04:
        print(f"  ATTENTION RAISED THE CEILING. On a world with long-range")
        print(f"  structure, the attention core reached {a*100:.0f}% vs the "
              f"GRU's {g*100:.0f}%,")
        print(f"  learning a pattern the GRU could not. NEURON's brain is now "
              f"more")
        print(f"  capable at the same size, still the same being, better "
              f"architecture.")
    elif a > g:
        print(f"  Attention edged out the GRU ({a*100:.0f}% vs {g*100:.0f}%), a "
              f"real but modest gain.")
    else:
        print(f"  No clear win here ({a*100:.0f}% vs {g*100:.0f}%). Attention "
              f"did not beat the GRU on")
        print(f"  this task; honest negative. (May need a longer life or "
              f"clearer long-range")
        print(f"  structure to show its advantage.)")

    print(f"\n  Honest scope: toy world. But this tests the real lever, whether "
          f"a better")
    print(f"  architecture (attention) lets NEURON represent patterns a simple "
          f"recurrent")
    print(f"  net cannot. That is how intelligence ceilings rise, not by size "
          f"alone.")


if __name__ == "__main__":
    main()
