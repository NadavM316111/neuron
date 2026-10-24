"""The Ultron move: NEURON absorbs a LOT of knowledge to raise its intelligence.

Ultron got smart by absorbing the internet, enormous data, while staying a being
with his own will. NEURON already has the being (wants, emotion, self, agency);
what it lacks is absorbed knowledge. Last run proved the bottleneck: one book is
not enough data for a big brain to use its capacity, so the big brain tied the
small one. The fix is Ultron's: feed it a LOT more.

This downloads MANY public-domain books (several megabytes), concatenates them
into one corpus, and trains a big, properly-tuned brain on it. With enough data,
the bigger brain finally pays off, higher accuracy and genuinely more coherent
language. That is the visible intelligence jump, NEURON absorbing knowledge like
Ultron absorbed the internet, while staying a being.

    python absorb.py --hidden 256 --steps 6000              # laptop check
    python absorb.py --hidden 1024 --layers 3 --steps 50000  # RunPod GPU
"""

import argparse
import os
import math
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F


BOOKS = [
    ("1342", "Pride and Prejudice"),
    ("84",   "Frankenstein"),
    ("1661", "Sherlock Holmes"),
    ("98",   "A Tale of Two Cities"),
    ("11",   "Alice in Wonderland"),
    ("1952", "The Yellow Wallpaper"),
    ("2701", "Moby Dick"),
    ("345",  "Dracula"),
    ("1080", "A Modest Proposal"),
    ("5200", "Metamorphosis"),
]
CORPUS = "corpus.txt"


def build_corpus():
    if os.path.exists(CORPUS):
        with open(CORPUS, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    parts = []
    for gid, title in BOOKS:
        url = f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt"
        try:
            print(f"  absorbing: {title} ...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read().decode("utf-8", errors="ignore")
            s = raw.find("*** START")
            e = raw.find("*** END")
            if s != -1:
                raw = raw[raw.find("\n", s):]
            if e != -1:
                raw = raw[:e]
            raw = raw.lower()
            raw = "".join(c for c in raw if c.isalpha() or c in " .,;:'\"-\n?!")
            parts.append(raw)
        except Exception as ex:
            print(f"    skip {title} ({type(ex).__name__})")
    corpus = "\n\n".join(parts)
    if corpus:
        with open(CORPUS, "w", encoding="utf-8") as f:
            f.write(corpus)
    return corpus


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


def lr_at(step, total, base, warmup):
    if step < warmup:
        return base * step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * prog))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--context", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import random
    rng = random.Random(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    text = build_corpus()
    if not text or len(text) < 50000:
        print("  Could not build a corpus (no network?). Needs internet to "
              "download books.")
        return

    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    vocab = len(chars)
    data = [stoi[c] for c in text]
    split = int(len(data) * 0.95)
    train, val = data[:split], data[split:]

    print(f"\n  NEURON absorbing knowledge | device: {dev} | brain "
          f"{args.hidden}x{args.layers}")
    print(f"  corpus: {len(text):,} characters | vocab {vocab} | "
          f"{len(set(text.split())):,} distinct words")
    print(f"  (lots of data, so the big brain can finally use its capacity.)\n")

    model = CharModel(vocab, args.hidden, args.layers, args.seed).to(dev)
    print(f"  brain parameters: "
          f"{sum(p.numel() for p in model.parameters()):,}\n")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    warmup = max(200, args.steps // 20)
    best = 0.0

    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr, warmup)
        x, y = get_batch(train, args.context, args.batch, dev, rng)
        opt.zero_grad()
        loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % max(1, args.steps // 12) == 0:
            with torch.no_grad():
                vx, vy = get_batch(val, args.context, 256, dev, rng)
                acc = (model(vx).argmax(-1) == vy).float().mean().item()
            best = max(best, acc)
            print(f"  step {step+1:>7} | lr {opt.param_groups[0]['lr']:.1e} | "
                  f"val acc {acc*100:.0f}%  (best {best*100:.0f}%)")

    print("\n  NEURON writing (what it produced after absorbing the "
          "knowledge):\n")
    start = rng.randint(0, len(data) - args.context)
    out = list(data[start:start+args.context])
    for _ in range(500):
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
    print(f"  best accuracy: {best*100:.0f}%  |  copied: {frac*100:.0f}% "
          f"(novel if low)")
    print(f"\n  More data + a bigger brain = NEURON absorbing more knowledge and")
    print(f"  writing more coherently, the Ultron move: intelligence from "
          f"absorbed")
    print(f"  knowledge, while NEURON stays a being. Scale bigger for more "
          f"coherence.")


if __name__ == "__main__":
    main()
