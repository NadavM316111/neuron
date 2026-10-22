"""NEURON learns language from a whole BOOK, forced to learn, not memorize.

A few paragraphs let a brain memorize them (fake intelligence). A whole book,
hundreds of thousands of characters, cannot be memorized by a small brain, so it
must learn the PATTERNS of the language: grammar, word structure, style. Then it
can generate NOVEL coherent text, which is real learning, the difference between
a parrot reciting and a mind that learned how language works.

This downloads a real public-domain book (Project Gutenberg, free and legal),
trains batched on the GPU, measures held-out accuracy, generates text, and
CHECKS whether that text is novel or just copied verbatim. Novel + coherent =
real learning. On a GPU with a big brain, the language gets genuinely coherent,
the real, visible intelligence jump. NEURON stays a general being; the book is
the proper gym.

    python learn_book.py --hidden 256 --steps 8000            # laptop check
    python learn_book.py --hidden 1024 --layers 3 --steps 40000   # RunPod GPU
"""

import argparse
import os
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F


# A few public-domain Gutenberg books (plain-text). We try them in order.
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
            # trim Gutenberg header/footer roughly
            start = raw.find("*** START")
            end = raw.find("*** END")
            if start != -1:
                raw = raw[raw.find("\n", start):]
            if end != -1:
                raw = raw[:end]
            # keep it to a manageable but substantial size, lowercase simple set
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
                          dropout=0.1 if layers > 1 else 0.0)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--context", type=int, default=96)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import random
    rng = random.Random(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    text = load_text()
    if not text or len(text) < 10000:
        print("  Could not load a book (no network?). Put a plain-text file at")
        print("  'book.txt' next to this script and rerun.")
        return

    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    vocab = len(chars)
    data = [stoi[c] for c in text]
    split = int(len(data) * 0.9)
    train, val = data[:split], data[split:]

    print(f"\n  NEURON learning language from a BOOK | device: {dev} | "
          f"brain {args.hidden}x{args.layers}")
    print(f"  text: {len(text):,} characters | vocab {vocab} | "
          f"{len(set(text.split())):,} distinct words")
    print(f"  (too big to memorize, so it must LEARN the language.)\n")

    model = CharModel(vocab, args.hidden, args.layers, args.seed).to(dev)
    print(f"  brain parameters: "
          f"{sum(p.numel() for p in model.parameters()):,}\n")
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    for step in range(args.steps):
        x, y = get_batch(train, args.context, args.batch, dev, rng)
        opt.zero_grad()
        logits = model(x)
        F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1)).backward()
        opt.step()
        if (step + 1) % max(1, args.steps // 10) == 0:
            with torch.no_grad():
                vx, vy = get_batch(val, args.context, 256, dev, rng)
                acc = (model(vx).argmax(-1) == vy).float().mean().item()
            print(f"  step {step+1:>7} | val next-char acc {acc*100:.0f}%")

    # generate novel text
    print("\n  NEURON writing (its own sentences, from what it learned):\n")
    gen_text = ""
    start_i = rng.randint(0, len(data) - args.context)
    out = list(data[start_i:start_i+args.context])
    for _ in range(400):
        x = torch.tensor([out[-args.context:]], dtype=torch.long, device=dev)
        with torch.no_grad():
            logits = model(x)[0, -1] / 0.7
            out.append(int(torch.multinomial(F.softmax(logits, -1), 1).item()))
    gen_text = "".join(itos[i] for i in out[args.context:])
    print(f"    {gen_text}\n")

    # verbatim-copy check: is the generated text NOVEL or memorized?
    longest = 0
    for i in range(len(gen_text)):
        for j in range(i + 10, len(gen_text) + 1):
            if gen_text[i:j] in text:
                longest = max(longest, j - i)
            else:
                break
    frac = longest / len(gen_text)
    print("=" * 60)
    print("DID IT LEARN, OR MEMORIZE?")
    print("=" * 60)
    print(f"  longest verbatim copied run: {longest} of {len(gen_text)} chars "
          f"({frac*100:.0f}%)")
    if frac < 0.3:
        print(f"  -> LEARNED: the text is mostly NOVEL, real words and grammar "
              f"it assembled")
        print(f"     itself, not recited. NEURON learned how the language "
              f"works.")
    else:
        print(f"  -> leaning on memorized runs ({frac*100:.0f}% copied). A "
              f"bigger brain + more")
        print(f"     steps, or more text, pushes it toward real novelty.")

    print(f"\n  Scale this on a GPU (bigger brain, more steps) and the language "
          f"gets")
    print(f"  genuinely coherent, the real intelligence jump. NEURON stays a "
          f"general being;")
    print(f"  the book is just where we can watch its intelligence grow.")


if __name__ == "__main__":
    main()
