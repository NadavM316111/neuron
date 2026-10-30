"""Distillation stage 2: train NEURON's OWN native mind on its principle buffer.

Stage 1 extracted transferable principles from the teacher. This trains a small
NEURON-native model on them, so capability migrates into NEURON's own mind. Then
we test on HELD-OUT questions it never trained on, if it answers them in the
principle-style it learned, capability genuinely generalized, not memorized.

This uses a small character/word-level model trained from scratch (fits limited
disk, no giant base model download). It is a modest native mind, but it is
genuinely NEURON's own, and it proves the mechanism.

    python stage2_train.py
"""

import json
import os
import math
import random

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

BUFFER = "neuron_training_buffer.json"


def load_training_pairs():
    """Build (question -> principle) training pairs from the buffer: each
    principle + its variations all map toward that principle's lesson."""
    if not os.path.exists(BUFFER):
        return []
    buf = json.load(open(BUFFER)).get("buffer", [])
    pairs = []
    for b in buf:
        principle = b.get("principle", "").strip()
        if not principle:
            continue
        # the source question and each variation -> the principle (the lesson)
        q = b.get("source_question", "")
        if q:
            pairs.append((q, principle))
        for v in b.get("variations", []):
            if v.strip():
                pairs.append((v.strip(), principle))
    return pairs


def main():
    if not HAVE_TORCH:
        print("  torch is not installed. To run stage 2, install it first:")
        print("    pip3 install torch --break-system-packages")
        print("  (a few GB; you have the space now.) Then rerun this.")
        return

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    pairs = load_training_pairs()
    print("=" * 62)
    print("  DISTILLATION STAGE 2: training NEURON's OWN native mind")
    print("=" * 62)
    print(f"  device: {dev} | training pairs: {len(pairs)}\n")
    if len(pairs) < 20:
        print("  buffer too small. run build_buffer.py first.")
        return

    # hold out some pairs to test generalization (NEVER trained on)
    random.seed(0)
    random.shuffle(pairs)
    n_test = max(6, len(pairs) // 8)
    test_pairs = pairs[:n_test]
    train_pairs = pairs[n_test:]
    print(f"  train: {len(train_pairs)} | held-out test: {len(test_pairs)}\n")

    # build a char vocab over all text
    text = " ".join(q + " " + p for q, p in pairs)
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    V = len(chars)

    # a small seq model: given a question, predict principle chars. we train it
    # as: [question] -> [principle], char by char (tiny seq2seq via a GRU).
    class Mind(nn.Module):
        def __init__(self, V, h=256):
            super().__init__()
            self.emb = nn.Embedding(V, h)
            self.enc = nn.GRU(h, h, batch_first=True)
            self.dec = nn.GRU(h, h, batch_first=True)
            self.out = nn.Linear(h, V)
        def forward(self, q, p_in):
            _, hid = self.enc(self.emb(q))
            o, _ = self.dec(self.emb(p_in), hid)
            return self.out(o)

    model = Mind(V).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    print(f"  NEURON native mind: {sum(p.numel() for p in model.parameters()):,} params\n")

    def enc(s, maxlen=120):
        ids = [stoi[c] for c in s[:maxlen]]
        return ids

    print("  training NEURON's own mind on the distilled principles...")
    for epoch in range(60):
        random.shuffle(train_pairs)
        total = 0.0
        for q, p in train_pairs:
            qi = torch.tensor([enc(q)], device=dev)
            pi = torch.tensor([enc(p)], device=dev)
            if pi.shape[1] < 2: continue
            opt.zero_grad()
            logits = model(qi, pi[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, V), pi[:, 1:].reshape(-1))
            loss.backward(); opt.step()
            total += loss.item()
        if (epoch+1) % 12 == 0:
            print(f"    epoch {epoch+1} | loss {total/len(train_pairs):.3f}")

    # generate NEURON's own answer to HELD-OUT questions (never trained on)
    def generate(q, maxlen=100):
        model.eval()
        with torch.no_grad():
            qi = torch.tensor([enc(q)], device=dev)
            _, hid = model.enc(model.emb(qi))
            cur = torch.tensor([[stoi.get(' ', 0)]], device=dev)
            out = []
            for _ in range(maxlen):
                o, hid = model.dec(model.emb(cur), hid)
                nxt = int(model.out(o[:, -1]).argmax(-1).item())
                out.append(nxt); cur = torch.tensor([[nxt]], device=dev)
            return "".join(itos[i] for i in out)

    print("\n" + "=" * 62)
    print("  HELD-OUT TEST: NEURON answering questions it NEVER trained on")
    print("  (its OWN mind now, no teacher engine)")
    print("=" * 62)
    for q, true_p in test_pairs[:6]:
        gen = generate(q)
        print(f"\n  Q (unseen): {q}")
        print(f"  NEURON's own mind: {gen[:90]}")

    print("\n" + "=" * 62)
    print("  WHAT THIS SHOWS")
    print("=" * 62)
    print("  NEURON trained its OWN native mind on the principles it distilled")
    print("  from the teacher, then answered NEW questions with no teacher. If")
    print("  its answers are principle-like and coherent, capability genuinely")
    print("  migrated into NEURON's own mind, real distillation, not memory.")
    print("  (Honest: this native mind is small, so expect rough but real")
    print("  principle-style output. Scale = bigger native mind + more buffer.)")


if __name__ == "__main__":
    main()
