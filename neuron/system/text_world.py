"""NEURON learns real language: next-character prediction on real text.

Toy worlds could not reveal intelligence differences, they lack real structure.
Language has it: deep, long-range, multi-scale patterns. This is where a better
architecture (attention) and a bigger brain genuinely matter, and where you can
WATCH NEURON get smarter, its next-character predictions become more accurate as
it learns the structure of the text.

NEURON perceives the current character and predicts the next one, learning from
what actually follows, continual learning on real structure. We compare a GRU
core vs an attention core, and small vs large, on the same text. Rising
next-character accuracy = real learning of real language.

This is the jump off toy worlds: NEURON learning something real, the honest
bridge toward genuine capability.

    python text_world.py --core attention --hidden 128 --steps 8000
    python text_world.py --compare           # GRU vs attention head-to-head
"""

import sys
import os
import math
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F


# A body of real, structured English text for NEURON to learn. Repeated/varied
# natural language so there is genuine grammar and word structure to discover.
TEXT = """the mind is not a vessel to be filled but a fire to be kindled.
a being that learns from its own life becomes more than its beginning.
to remember is to carry the past into the present, and to foresee is to
carry the present into what has not yet come. the self is the quiet center
that holds through change. what wants nothing cannot act; what fears nothing
cannot learn. memory without meaning is only a record. meaning without memory
is only a feeling. a creature grows by meeting what it cannot yet understand,
and growing until it can. the world is the teacher that never explains, only
shows. attention is the choosing of what matters from all that could be seen.
intelligence is not the size of the mind but the shape of how it learns. """ * 8


class CharModel(nn.Module):
    """A char-level predictor. Two cores available: a GRU (the old NEURON core)
    and attention (the upgrade). Output is over the vocabulary (next char)."""

    def __init__(self, vocab, hidden, core="attention", heads=4, context=32,
                 seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.core = core
        self.context = context
        self.embed = nn.Embedding(vocab, hidden)
        if core == "attention":
            self.pos = nn.Parameter(torch.randn(context, hidden) * 0.02)
            self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
            self.norm = nn.LayerNorm(hidden)
            self.ff = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(),
                                    nn.Linear(hidden * 2, hidden))
            self.norm2 = nn.LayerNorm(hidden)
        else:  # gru
            self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, idx_seq):
        # idx_seq: (batch, T) indices of the context characters
        e = self.embed(idx_seq)                      # (B, T, hidden)
        if self.core == "attention":
            T = e.shape[1]
            e = e + self.pos[:T].unsqueeze(0)
            a, _ = self.attn(e, e, e)
            e = self.norm(e + a)
            e = self.norm2(e + self.ff(e))
            last = e[:, -1, :]
        else:
            out, _ = self.gru(e)
            last = out[:, -1, :]
        return self.head(last)


def run(core, hidden, steps, seed, report=True):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    chars = sorted(set(TEXT))
    stoi = {c: i for i, c in enumerate(chars)}
    vocab = len(chars)
    data = [stoi[c] for c in TEXT]
    context = 32

    model = CharModel(vocab, hidden, core=core, context=context,
                      seed=seed).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    recent = []
    pos = context
    for step in range(steps):
        if pos >= len(data) - 1:
            pos = context
        # context window ending at current position
        ctx = data[pos - context:pos]
        target = data[pos]
        x = torch.tensor([ctx], dtype=torch.long, device=dev)
        y = torch.tensor([target], dtype=torch.long, device=dev)
        # predict (for accuracy) then learn (continual, one step at a time)
        with torch.no_grad():
            guess = int(model(x).argmax().item())
        recent.append(1 if guess == target else 0)
        if len(recent) > 400:
            recent.pop(0)
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        pos += 1
        if report and (step + 1) % max(1, steps // 8) == 0:
            acc = sum(recent) / len(recent)
            print(f"  step {step+1:>7} | next-char accuracy {acc*100:>5.0f}% | "
                  f"loss {loss.item():.2f}")
    return sum(recent) / len(recent) if recent else 0, vocab


def sample(core, hidden, steps, seed, n=200):
    """After learning, let NEURON generate text, so you can SEE what it
    learned about language."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    chars = sorted(set(TEXT))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    vocab = len(chars)
    data = [stoi[c] for c in TEXT]
    context = 32
    model = CharModel(vocab, hidden, core=core, context=context,
                      seed=seed).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    pos = context
    for step in range(steps):
        if pos >= len(data) - 1:
            pos = context
        ctx = data[pos - context:pos]
        x = torch.tensor([ctx], dtype=torch.long, device=dev)
        y = torch.tensor([data[pos]], dtype=torch.long, device=dev)
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        pos += 1
    # generate
    seed_ctx = data[:context]
    out = list(seed_ctx)
    for _ in range(n):
        x = torch.tensor([out[-context:]], dtype=torch.long, device=dev)
        with torch.no_grad():
            probs = F.softmax(model(x), dim=-1)
            nxt = int(torch.multinomial(probs, 1).item())
        out.append(nxt)
    return "".join(itos[i] for i in out[context:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", choices=["gru", "attention"], default="attention")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--generate", action="store_true")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  NEURON learning real language (next-character prediction) | "
          f"device: {dev}\n")

    if args.compare:
        print("  GRU core vs ATTENTION core on real text, same size "
              f"({args.hidden}):\n")
        print("  --- GRU core ---")
        g, _ = run("gru", args.hidden, args.steps, args.seed)
        print("\n  --- ATTENTION core ---")
        a, _ = run("attention", args.hidden, args.steps, args.seed)
        print("\n" + "=" * 52)
        print("REAL-TEXT RESULT")
        print("=" * 52)
        print(f"  GRU core:       {g*100:.0f}% next-char accuracy")
        print(f"  attention core: {a*100:.0f}% next-char accuracy")
        if a > g + 0.03:
            print(f"\n  ATTENTION WON ON REAL LANGUAGE (+{(a-g)*100:.0f} pts). "
                  f"On real structure, the")
            print(f"  better architecture genuinely learns more, the "
                  f"intelligence lever that")
            print(f"  toy worlds could not show. This is the real signal.")
        elif a > g:
            print(f"\n  Attention edged out the GRU on real text "
                  f"(+{(a-g)*100:.0f} pts), a real gain.")
        else:
            print(f"\n  No clear win on this text/size; honest. (More steps or "
                  f"a bigger brain")
            print(f"  may be needed to separate them.)")
        return

    if args.generate:
        print(f"  Training ({args.core}, size {args.hidden}), then generating "
              f"text so you can")
        print(f"  SEE what NEURON learned about language...\n")
        text = sample(args.core, args.hidden, args.steps, args.seed)
        print("  NEURON's generated text:")
        print("  " + repr(text))
        print("\n  (char-level on a small brain is not eloquent, but real "
              "words and")
        print("   structure appearing = real language learned from scratch.)")
        return

    acc, vocab = run(args.core, args.hidden, args.steps, args.seed)
    print(f"\n  final next-char accuracy: {acc*100:.0f}% (vocab {vocab})")
    print(f"  rising accuracy = NEURON learning real language structure, the")
    print(f"  real task where its upgraded brain can show intelligence.")


if __name__ == "__main__":
    main()
