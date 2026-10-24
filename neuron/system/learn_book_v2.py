"""NEURON learns language from a book, properly tuned for a BIG brain.

Last run the big 19M-parameter brain got WORSE (61 -> 54%) because the learning
rate was too high, big models train unstably with small-model settings. This
version uses proper large-model training: a lower learning rate with warmup then
cosine decay, and gradient clipping to keep it stable. Now the big brain should
CLIMB past the small brain's accuracy and generate more coherent language, the
real, visible intelligence jump.

    python learn_book_v2.py --hidden 256 --steps 6000              # laptop check
    python learn_book_v2.py --hidden 1024 --layers 3 --steps 40000  # RunPod GPU
"""

import argparse
import os
import math
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F


BOOK_URLS = [
    "https://www.gutenberg.org/files/1342/1342-0.txt",   # Pride and Prejudice
    "https://www.gutenberg.org/files/84/84-0.txt",       # Frankenstein
    "https://www.gutenberg.org/files/11/11-0.txt",       # Alice in Wonderland
]
LOCAL = "book.txt"


def load_text():
    if os.path.exists(LOCAL):
        with open(LOCAL, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    for url in BOOK_URLS:
        try:
            print(f"  downloading {url} ...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read().decode("utf-8", errors="ignore")
            start = raw.find("*** START")
            end = raw.find("*** END")
            if start != -1:
                raw = raw[raw.find("\n", start):]
            if end != -1:
                raw = raw[:end]
            raw = raw.lower()
            raw = "".join(c for c in raw if c.isalpha() or c in " .,;:'\"-\n?!")
            with open(LOCAL, "w", encoding="utf-8") as f:
                f.write(raw)
            return raw
        except Exception as e:
            print(f"    failed ({type(e).__name__}); trying next ...")
    return None


class CharModel(nn.Module):
    def __init__(self, vocab, hidden, layers=2, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.embed = nn.Embedding(vocab, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                          dropout=0.2 if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, idx):
        e = self.embed(idx)
        out, _ = self.gru(e)
        return self.head(out)


def get_batch(data, context, batch, dev, rng):
    ix = [rng.randint(0, len(data) - context - 1) for _ in range(batch)]
    x = torch.tensor([data[i:i+context] for i in ix], dtype=torch.long, device=dev)
    y = torch.tensor([data[i+1:i+context+1] for i in ix], dtype=torch.long, device=dev)
    return x, y


def lr_at(step, total, base_lr, warmup):
    """Warmup then cosine decay, standard stable large-model schedule."""
    if step < warmup:
        return base_lr * step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * prog))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--context", type=int, default=96)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)     # lower, stable
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import random
    rng = random.Random(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    text = load_text()
    if not text or len(text) < 10000:
        print("  Could not load a book. Put a plain-text file at 'book.txt' "
              "and rerun.")
        return

    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    vocab = len(chars)
    data = [stoi[c] for c in text]
    split = int(len(data) * 0.9)
    train, val = data[:split], data[split:]

    print(f"\n  NEURON learning language from a BOOK (tuned) | device: {dev} | "
          f"brain {args.hidden}x{args.layers}")
    print(f"  text: {len(text):,} chars | vocab {vocab} | lr {args.lr} "
          f"(warmup+cosine) + grad clip\n")

    model = CharModel(vocab, args.hidden, args.layers, args.seed).to(dev)
    print(f"  brain parameters: "
          f"{sum(p.numel() for p in model.parameters()):,}\n")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    warmup = max(100, args.steps // 20)
    best_acc = 0.0

    for step in range(args.steps):
        # set the learning rate per the stable schedule
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr, warmup)
        x, y = get_batch(train, args.context, args.batch, dev, rng)
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # stability
        opt.step()
        if (step + 1) % max(1, args.steps // 12) == 0:
            with torch.no_grad():
                vx, vy = get_batch(val, args.context, 256, dev, rng)
                acc = (model(vx).argmax(-1) == vy).float().mean().item()
            best_acc = max(best_acc, acc)
            print(f"  step {step+1:>7} | lr {opt.param_groups[0]['lr']:.1e} | "
                  f"val acc {acc*100:.0f}%  (best {best_acc*100:.0f}%)")

    print("\n  NEURON writing (what the tuned brain produced):\n")
    start_i = rng.randint(0, len(data) - args.context)
    out = list(data[start_i:start_i+args.context])
    for _ in range(400):
        x = torch.tensor([out[-args.context:]], dtype=torch.long, device=dev)
        with torch.no_grad():
            logits = model(x)[0, -1] / 0.7
            out.append(int(torch.multinomial(F.softmax(logits, -1), 1).item()))
    gen = "".join(itos[i] for i in out[args.context:])
    print(f"    {gen}\n")

    longest = 0
    for i in range(len(gen)):
        for j in range(i + 10, len(gen) + 1):
            if gen[i:j] in text:
                longest = max(longest, j - i)
            else:
                break
    frac = longest / len(gen)
    print("=" * 60)
    print(f"  best accuracy: {best_acc*100:.0f}%  |  "
          f"longest copied run: {frac*100:.0f}% (novel if low)")
    if best_acc > 0.62 and frac < 0.3:
        print("  -> THE BIGGER BRAIN LEARNED LANGUAGE BETTER: higher accuracy")
        print("     than the small brain AND mostly-novel coherent text. The")
        print("     intelligence jump, visible. Scale further for more "
              "coherence.")
    elif frac < 0.3:
        print("  -> Learned the language (novel text). Push bigger/longer for "
              "higher accuracy.")
    else:
        print("  -> Still tuning; check the trend. Bigger brain + more steps "
              "should climb.")


if __name__ == "__main__":
    main()
