"""Can language structure grow from nothing, online, in a single pass?

THE CLAIM THIS TESTS. Every language result in this project sits on a
pretrained Qwen. The "no training run" half of the vision has only ever been
demonstrated on a 9x9 grid, a 25x25 grid, and weather. Nobody here has asked
whether the same premise reaches TEXT: random weights, one pass, in order,
never revisited, no epochs, no shuffling.

WHAT WOULD COUNT. Character-level next-character prediction on held-out text,
against n-gram baselines that are deliberately given every advantage:

  uniform     log(V). The floor. Beating this means nothing was learned.
  unigram     letter frequencies, fit on ALL the training text.
  bigram      one character of context, fit on ALL the training text.
  trigram     two characters of context, fit on ALL the training text.

The baselines see the whole corpus and may count it as often as they like.
The network sees each sentence ONCE, in order, and never again. If it wins
under that handicap, the win is real. Beating bigram means it learned context
beyond a single character. Beating trigram would be strong.

THE ARMS, added after the first run. Beating n-grams shows the network
learned something. It does not show what the SINGLE PASS costs, and that is
the claim the vision rests on. So the same architecture is also trained the
conventional way on the same data:

  online        one pass, in order, never revisited. The claim.
  shuffled      one pass, order destroyed. Matched update count, so this
                isolates ORDER and nothing else.
  multipass     several epochs, shuffled. Conventional training. Uses more
                compute by design; the gap to `online` is what the single
                pass costs.

If online lands close to multipass, the premise is nearly free. If it lands
far off, the honest claim shrinks to "it learns, at a price".

WHAT THIS DOES NOT SHOW, stated plainly so the result is not oversold. It
does not show that a useful language model can be built without pretraining.
It is a character-level GRU with a few hundred thousand parameters on one
domain. It is the first rung, not the ladder. A positive result says the
premise survives contact with text at tiny scale; it says nothing about
whether it survives scale.

CHECKS BUILT IN, each from a failure logged in this repo's method notes:

  The alphabet is FIXED IN ADVANCE, not fit to the corpus. A vocabulary
  derived from the data would leak information the online learner is not
  supposed to have.

  Held-out text is drawn from the END of the file and never trained on.

  The untrained network is evaluated BEFORE any learning. If that reading is
  not close to log(V), the measurement is wrong before the experiment starts.

  If the network never beats unigram, the script says the test failed rather
  than reporting a curve that looks like progress.

  python scratch.py --train 15000 --seeds 2
  python scratch.py --train 2000 --eval 200 --seeds 1 --epochs 2   # smoke
"""

import argparse
import json
import random
import math
import os
import time
from collections import Counter, defaultdict

import torch
import torch.nn as nn


# Fixed in advance. Deriving this from the corpus would give the online
# learner information about text it has not seen yet.
ALPHABET = " abcdefghijklmnopqrstuvwxyz0123456789.,'-"
UNK = len(ALPHABET)
V = len(ALPHABET) + 1
INDEX = {c: i for i, c in enumerate(ALPHABET)}


def encode(text):
    return [INDEX.get(c, UNK) for c in text.lower()]


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class CharGRU(nn.Module):
    def __init__(self, hidden=256, embed=32, layers=1):
        super().__init__()
        self.emb = nn.Embedding(V, embed)
        self.gru = nn.GRU(embed, hidden, num_layers=layers, batch_first=True)
        self.out = nn.Linear(hidden, V)

    def forward(self, x, h=None):
        e = self.emb(x)
        y, h = self.gru(e, h)
        return self.out(y), h


# ------------------------------------------------------------- baselines

def ngram_loss(train_text, eval_texts, n, k=0.1):
    """Mean per-character cross entropy in nats, add-k smoothed.

    Fit on the ENTIRE training corpus, which the online learner never gets
    to do. The baselines are meant to be hard to beat, not fair.
    """
    counts = defaultdict(Counter)
    ctx_totals = Counter()
    ids = encode(train_text)
    for i in range(len(ids)):
        ctx = tuple(ids[max(0, i - n + 1):i])
        if len(ctx) < n - 1:
            ctx = (UNK,) * (n - 1 - len(ctx)) + ctx
        counts[ctx][ids[i]] += 1
        ctx_totals[ctx] += 1

    total, chars = 0.0, 0
    for text in eval_texts:
        e = encode(text)
        for i in range(len(e)):
            ctx = tuple(e[max(0, i - n + 1):i])
            if len(ctx) < n - 1:
                ctx = (UNK,) * (n - 1 - len(ctx)) + ctx
            c = counts[ctx][e[i]] + k
            z = ctx_totals[ctx] + k * V
            total -= math.log(c / z)
            chars += 1
    return total / chars if chars else float("nan")


# ------------------------------------------------------------ evaluation

@torch.no_grad()
def held_out_loss(model, texts, device):
    """Mean per-character cross entropy in nats on text never trained on."""
    model.eval()
    total, chars = 0.0, 0
    lossf = nn.CrossEntropyLoss(reduction="sum")
    for t in texts:
        ids = encode(t)
        if len(ids) < 2:
            continue
        x = torch.tensor([ids[:-1]], device=device)
        y = torch.tensor([ids[1:]], device=device)
        logits, _ = model(x)
        total += lossf(logits.reshape(-1, V), y.reshape(-1)).item()
        chars += y.numel()
    return total / chars if chars else float("nan")


def train_arm(mode, train_lines, eval_lines, args, seed, epochs):
    """One arm. Returns its curve and final held-out loss.

    Every arm starts from the SAME seed, so they begin at identical random
    weights and any difference is the training regime rather than the
    initialisation.
    """
    torch.manual_seed(seed)
    device = pick_device()
    model = CharGRU(args.hidden, args.embed).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss()

    before = held_out_loss(model, eval_lines, device)
    if abs(before - math.log(V)) > 0.35:
        print(f"    WARNING: untrained reads {before:.4f}, uniform floor is "
              f"{math.log(V):.4f}. The measurement is wrong before the arm "
              f"has started.")

    items = list(train_lines)
    if mode in ("shuffled", "multipass"):
        # A separate RNG so shuffling does not disturb weight init above.
        random.Random(seed + 1000).shuffle(items)

    started = time.time()
    curve = [(0, before)]
    updates = 0
    for ep in range(epochs):
        if mode == "multipass" and ep > 0:
            random.Random(seed + 1000 + ep).shuffle(items)
        for line in items:
            ids = encode(line)
            if len(ids) < 2:
                continue
            model.train()
            x = torch.tensor([ids[:-1]], device=device)
            y = torch.tensor([ids[1:]], device=device)
            logits, _ = model(x)
            loss = lossf(logits.reshape(-1, V), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            updates += 1

            if updates % args.every == 0:
                h = held_out_loss(model, eval_lines, device)
                curve.append((updates, h))
                print(f"    {updates:6d}  {h:.4f} nats  "
                      f"({h / math.log(2):.3f} bits/char)", flush=True)

    final = held_out_loss(model, eval_lines, device)
    if curve[-1][0] != updates:
        curve.append((updates, final))
    return dict(mode=mode, seed=seed, epochs=epochs, updates=updates,
                untrained=before, curve=curve, final=final,
                seconds=time.time() - started)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/wiki.txt")
    ap.add_argument("--train", type=int, default=15000)
    ap.add_argument("--eval", type=int, default=800)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--embed", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--epochs", type=int, default=3,
                    help="epochs for the multipass arm")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--every", type=int, default=2000)
    ap.add_argument("--out", default="scratch.json")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        raise SystemExit(f"no data at {os.path.abspath(args.data)} — run "
                         f"fetch_wiki.py first")

    lines = [l.strip() for l in open(args.data) if len(l.strip()) > 40]
    if len(lines) < args.train + args.eval:
        raise SystemExit(f"only {len(lines)} usable lines, need "
                         f"{args.train + args.eval}")

    train_lines = lines[:args.train]
    eval_lines = lines[-args.eval:]          # held out from the END

    uniform = math.log(V)
    print(f"  vocab {V}, {len(train_lines)} training sentences, "
          f"{len(eval_lines)} held out")
    print(f"  uniform floor  {uniform:.4f} nats  "
          f"({uniform / math.log(2):.3f} bits/char)")

    print("\n  baselines, fitted on the FULL training text ...", flush=True)
    train_text = " ".join(train_lines)
    base = {}
    for name, n in (("unigram", 1), ("bigram", 2), ("trigram", 3)):
        base[name] = ngram_loss(train_text, eval_lines, n)
        print(f"    {name:8s} {base[name]:.4f} nats  "
              f"({base[name] / math.log(2):.3f} bits/char)", flush=True)

    arms = [("online", 1), ("shuffled", 1), ("multipass", args.epochs)]
    runs = []
    for seed in range(args.seeds):
        for mode, epochs in arms:
            print(f"\n=== {mode} (seed {seed}, {epochs} epoch"
                  f"{'s' if epochs > 1 else ''}) ===", flush=True)
            runs.append(train_arm(mode, train_lines, eval_lines,
                                  args, seed, epochs))

    with open(args.out, "w") as f:
        json.dump(dict(args=vars(args), uniform=uniform, baselines=base,
                       runs=runs), f, indent=2)

    # ------------------------------------------------------------ summary

    agg = {}
    for r in runs:
        a = agg.setdefault(r["mode"], dict(final=[], secs=[], ups=[]))
        a["final"].append(r["final"])
        a["secs"].append(r["seconds"])
        a["ups"].append(r["updates"])

    def mean(xs):
        return sum(xs) / len(xs)

    print("\n" + "=" * 70)
    print("RESULT   mean per-character cross entropy on held-out text")
    print("=" * 70)
    print(f"  {'uniform floor':24s} {uniform:7.4f} nats  "
          f"{uniform / math.log(2):6.3f} bits/char")
    for n in ("unigram", "bigram", "trigram"):
        print(f"  {n + ' (full corpus)':24s} {base[n]:7.4f} nats  "
              f"{base[n] / math.log(2):6.3f} bits/char")
    for mode, _ in arms:
        a = agg[mode]
        spread = ""
        if len(a["final"]) > 1:
            spread = f"   seeds {', '.join(f'{v:.4f}' for v in a['final'])}"
        print(f"  {mode:24s} {mean(a['final']):7.4f} nats  "
              f"{mean(a['final']) / math.log(2):6.3f} bits/char  "
              f"{mean(a['ups']):6.0f} upd{spread}")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    on = mean(agg["online"]["final"])
    sh = mean(agg["shuffled"]["final"])
    mp = mean(agg["multipass"]["final"])

    if on >= uniform - 0.05:
        print("  FAILED. The online arm did not get meaningfully below the")
        print("  uniform floor, so nothing was learned.")
        return
    if on >= base["unigram"]:
        print("  FAILED. The online arm did not beat letter frequencies.")
        return

    beaten = [n for n in ("unigram", "bigram", "trigram") if on < base[n]]
    print(f"  Online beat: {', '.join(beaten) if beaten else 'nothing'}.")

    print(f"\n  cost of learning IN ORDER   {on - sh:+.4f} nats "
          f"(online vs shuffled, same updates)")
    print(f"  cost of the SINGLE PASS     {on - mp:+.4f} nats "
          f"(online vs {args.epochs} shuffled epochs)")

    if on - mp <= 0.05:
        print("\n  The single pass costs essentially nothing against")
        print("  conventional multi-epoch training on this task. That is the")
        print("  strongest form this result can take at this scale.")
    elif on - mp <= 0.25:
        print("\n  The single pass costs a little against conventional")
        print("  training. Report the gap; do not round it to zero.")
    else:
        print(f"\n  The single pass costs {on - mp:.4f} nats against")
        print("  conventional training. It learns, at a real price, and the")
        print("  claim should say so.")

    if abs(on - sh) > 0.1:
        print(f"\n  Order matters here: {on - sh:+.4f} nats against the same")
        print("  data shuffled. Worth knowing which direction and why.")

    print("\n  A character-level GRU on one domain. The first rung of the")
    print("  no-pretraining claim, not the claim.")
    print(f"\n  written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()