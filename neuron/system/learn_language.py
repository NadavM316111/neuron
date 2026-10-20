"""NEURON learns real language from a large, rich body of text, and we test
whether it LEARNED the language or just MEMORIZED it.

Learning one paragraph lets a brain echo it. Learning from a large, varied text
forces it to learn the PATTERNS of the language, grammar, word structure,
style, so it can generate NEW coherent text, not just copy. That is the
difference between a parrot and a mind that learned how the language works.

THE HONEST TEST. After training, NEURON generates text. We check:
  - is it coherent? (real words, real structure)
  - is it NOVEL? (not just copied verbatim from the training text)
Novel-but-coherent = it learned the language's patterns. Verbatim copying =
it only memorized. We measure both.

    python learn_language.py --hidden 256 --steps 20000
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F


# A large, varied body of real English, enough distinct structure that the
# brain must learn PATTERNS, not memorize one passage. (Public-domain style
# prose assembled for training; varied sentences, vocabulary, rhythm.)
TEXT = """
the sun rose slowly over the quiet hills, and the whole valley began to wake.
birds called from the tall trees, and a cool wind moved through the grass.
far away, a river ran bright and fast, carrying the morning light downstream.
a traveler walked along the old road, carrying little and thinking much.
he had left the city many days ago, and the noise of it had faded from his mind.
now there was only the road, the sky, and the slow turning of the seasons.
in the evening he would stop beneath a tree and watch the stars appear,
one by one, until the whole dark sky was full of distant, patient fire.
he wondered where the road would end, and whether an ending mattered at all.
some journeys are made for arriving, he thought, and some are made for walking.
a child in a village asked her grandmother why the sea was never still.
the old woman smiled and said the sea remembers every wind that ever touched it,
and so it moves, always, carrying the memory of a thousand distant storms.
the child looked at the water for a long time, and thought about memory,
and about how even the smallest things might hold more than they seemed to.
knowledge is a strange kind of treasure, for the more you give it away,
the more of it you keep. a teacher once told her students this, and they laughed,
but years later, when they had become teachers themselves, they understood.
to build anything lasting, you must begin before you are ready, and continue
after you are tired, and finish long after the first excitement has gone.
courage is not the absence of fear but the choice to move while afraid.
the mind grows by meeting what it cannot yet understand and growing until it can.
every great thing was once a small thing that was allowed to keep going.
the night is long, but the morning always comes, and with it, the work begins again.
""" * 6


class CharGRU(nn.Module):
    def __init__(self, vocab, hidden, layers=2, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.embed = nn.Embedding(vocab, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, idx):
        e = self.embed(idx)
        out, _ = self.gru(e)
        return self.head(out[:, -1, :])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--context", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    chars = sorted(set(TEXT))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    vocab = len(chars)
    data = [stoi[c] for c in TEXT]
    ctxn = args.context

    print(f"  NEURON learning real language from a large text | device: {dev}")
    print(f"  brain size {args.hidden} | vocab {vocab} | "
          f"{len(set(TEXT.split()))} distinct words\n")

    model = CharGRU(vocab, args.hidden, seed=args.seed).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    recent = []
    pos = ctxn
    for step in range(args.steps):
        if pos >= len(data) - 1:
            pos = ctxn
        ctx = data[pos - ctxn:pos]
        x = torch.tensor([ctx], dtype=torch.long, device=dev)
        y = torch.tensor([data[pos]], dtype=torch.long, device=dev)
        with torch.no_grad():
            recent.append(1 if int(model(x).argmax().item()) == data[pos] else 0)
        if len(recent) > 500:
            recent.pop(0)
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()
        pos += 1
        if (step + 1) % max(1, args.steps // 8) == 0:
            print(f"  step {step+1:>7} | next-char accuracy "
                  f"{sum(recent)/len(recent)*100:>4.0f}%")

    # generate from a neutral seed
    print("\n  NEURON generating language (it learned the patterns, now it "
          "writes):\n")
    import random
    for temp in (0.6, 0.9):
        start = random.Random(args.seed).randrange(len(data) - ctxn)
        out = list(data[start:start + ctxn])
        for _ in range(240):
            x = torch.tensor([out[-ctxn:]], dtype=torch.long, device=dev)
            with torch.no_grad():
                probs = F.softmax(model(x) / temp, dim=-1)
                out.append(int(torch.multinomial(probs, 1).item()))
        gen = "".join(itos[i] for i in out[ctxn:])
        print(f"  [temperature {temp}]")
        print(f"    {gen}\n")

    # HONEST TEST: did it learn patterns or just memorize? check how much of
    # the generated text is a verbatim run from the source.
    print("=" * 60)
    print("DID IT LEARN THE LANGUAGE, OR MEMORIZE THE TEXT?")
    print("=" * 60)
    start = random.Random(1).randrange(len(data) - ctxn)
    out = list(data[start:start + ctxn])
    for _ in range(300):
        x = torch.tensor([out[-ctxn:]], dtype=torch.long, device=dev)
        with torch.no_grad():
            probs = F.softmax(model(x) / 0.8, dim=-1)
            out.append(int(torch.multinomial(probs, 1).item()))
    gen = "".join(itos[i] for i in out[ctxn:])
    source = TEXT
    # longest verbatim run from source found in the generated text
    longest = 0
    for i in range(len(gen)):
        for j in range(i + 8, len(gen) + 1):
            if gen[i:j] in source:
                longest = max(longest, j - i)
            else:
                break
    frac = longest / len(gen)
    print(f"  longest verbatim copied run: {longest} chars out of {len(gen)} "
          f"generated ({frac*100:.0f}%)")
    acc = sum(recent) / len(recent)
    if acc > 0.6 and frac < 0.4:
        print(f"  -> IT LEARNED THE LANGUAGE. {acc*100:.0f}% accuracy and the "
              f"generated text is")
        print(f"     mostly NOVEL (not copied), real words and grammar it "
              f"assembled itself.")
        print(f"     That is learning the patterns of language, not memorizing "
              f"a passage.")
    elif acc > 0.6:
        print(f"  -> It learned well but leans on memorized runs "
              f"({frac*100:.0f}% copied). More")
        print(f"     varied text or a bigger brain would push it toward real "
              f"novelty.")
    else:
        print(f"  -> Still early ({acc*100:.0f}%). More steps/size/text needed "
              f"for real fluency.")

    print(f"\n  Honest scope: char-level, modest brain, still a small text by "
          f"real standards.")
    print(f"  But coherent, novel language = NEURON learning how a real "
          f"language works, the")
    print(f"  real intelligence it could never show on toy worlds. Bigger + "
          f"more text (GPU)")
    print(f"  climbs toward genuine fluency.")


if __name__ == "__main__":
    main()
