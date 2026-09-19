"""Does layer skipping cure greedy information collapse?

The accuracy gap between local learning and backpropagation has a known
diagnosis. Greedy local training COLLAPSES TASK-RELEVANT INFORMATION at
early layers: each module optimises its own loss, becomes discriminative
too early, and discards information later modules would have used. The
documented symptom is that early modules learn MORE discriminative features
than end-to-end training while the final accuracy is WORSE, and deeper
modules stop improving or actively degrade.

InfoPro fixes this by adding a term that preserves information about the
input. It works and it costs — a reconstruction network on top of every
module.

THE HYPOTHESIS HERE, which follows from yesterday's speed result rather than
from the literature: information collapse is greed in DEPTH. Early layers
over-optimise for themselves at the expense of later ones. And skipping
updates ALREADY limits how much a layer optimises. So a layer that skips
most of its updates cannot collapse information as fast, and skipping may
cure the accuracy gap as a side effect of being a speed mechanism.

If true it is a better answer than InfoPro, because it costs less rather
than more.

MEASURED, and the second is the point:

  final accuracy        does the gap to backprop close
  early-layer accuracy  the collapse signature. Greedy local learning makes
                        layer 1 MORE accurate than backprop's layer 1 while
                        the final layer is worse. If skipping works, early
                        layers should become LESS discriminative and the
                        final layer better.

That second measurement is what makes this a test of the mechanism rather
than of the outcome. A skipping arm that closes the gap WITHOUT changing
the early-layer signature would mean something else is happening.

Arms:
  bp             end-to-end backpropagation. The reference.
  local-full     greedy local losses, every layer every step. Should show
                 the collapse: better early, worse final.
  skip-early     skipping applied only to the first half of the layers,
                 where collapse happens.
  skip-all-50    skipping everywhere at the 50th percentile.
  skip-all-70    skipping everywhere at the 70th percentile.

CIFAR-10, which is where this gap is documented. A small convolutional
stack and one epoch, because the question is whether the SIGNATURE changes,
not whether the number is state of the art.
"""

import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


DEPTH = 6
WIDTH = 64
LR = 1e-3
BATCH = 128
EPOCHS = 1
SEEDS = [0, 1]
HISTORY = 100
DATA = "/Users/nadavminkowitz/neuron/data"
SUBSET = 20000          # training images, to keep this under ten minutes
TEST_SUBSET = 4000


class Net(nn.Module):
    """Six convolutional blocks, each with its own auxiliary classifier so
    a local loss can be computed. Backpropagation happens inside a block
    and, under the local rule, never between blocks."""

    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        chans = [3] + [WIDTH] * DEPTH
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(chans[i], chans[i + 1], 3, padding=1),
                nn.BatchNorm2d(chans[i + 1]),
                nn.ReLU(),
                nn.MaxPool2d(2) if i < 3 else nn.Identity())
            for i in range(DEPTH)])
        self.heads = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                          nn.Linear(WIDTH, 10))
            for _ in range(DEPTH)])

    def forward(self, x):
        h = x
        for block in self.blocks:
            h = block(h)
        return self.heads[-1](h)

    def all_heads(self, x):
        """Every layer's own prediction, for the collapse measurement."""
        outs = []
        h = x
        for block, head in zip(self.blocks, self.heads):
            h = block(h)
            outs.append(head(h))
        return outs


class Percentiles:
    """Per-layer loss history. Layers sit at different loss scales, so
    'unusually high' has to be judged within a layer."""

    def __init__(self, depth, window=HISTORY):
        self.hist = [[] for _ in range(depth)]
        self.window = window

    def push(self, i, v):
        h = self.hist[i]
        h.append(v)
        if len(h) > self.window:
            h.pop(0)

    def above(self, i, v, pct):
        h = self.hist[i]
        if len(h) < 20:
            return True
        k = int(len(h) * pct / 100.0)
        return v >= sorted(h)[min(k, len(h) - 1)]


def loaders():
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49, 0.48, 0.45), (0.25, 0.24, 0.26))])
    train = datasets.CIFAR10(DATA, train=True, download=False, transform=tf)
    test = datasets.CIFAR10(DATA, train=False, download=False, transform=tf)
    train = torch.utils.data.Subset(train, range(SUBSET))
    test = torch.utils.data.Subset(test, range(TEST_SUBSET))
    return (torch.utils.data.DataLoader(train, batch_size=BATCH,
                                        shuffle=True),
            torch.utils.data.DataLoader(test, batch_size=256))


def train_bp(net, loader):
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    t0 = time.perf_counter()
    net.train()
    for _ in range(EPOCHS):
        for x, y in loader:
            loss = F.cross_entropy(net(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return time.perf_counter() - t0, DEPTH * len(loader) * EPOCHS


def train_local(net, loader, skip_pct=None, early_only=False):
    """Greedy local training, optionally with skipping.

    Each block is optimised on its own head's loss with the input detached,
    so no gradient crosses a block boundary. That isolation is what causes
    information collapse, and it is also what makes skipping possible.
    """
    opts = [torch.optim.Adam(
        list(net.blocks[i].parameters()) + list(net.heads[i].parameters()),
        lr=LR) for i in range(DEPTH)]
    pct = Percentiles(DEPTH)
    updates = 0
    t0 = time.perf_counter()
    net.train()

    for _ in range(EPOCHS):
        for x, y in loader:
            h = x
            for i in range(DEPTH):
                h_in = h.detach()
                rep = net.blocks[i](h_in)
                loss = F.cross_entropy(net.heads[i](rep), y)
                v = float(loss.item())

                do = True
                if skip_pct is not None:
                    # early_only applies skipping just to the first half,
                    # which is where collapse is documented to happen
                    applies = (not early_only) or i < DEPTH // 2
                    if applies:
                        do = pct.above(i, v, skip_pct)
                pct.push(i, v)

                if do:
                    opts[i].zero_grad()
                    loss.backward()
                    opts[i].step()
                    updates += 1

                h = rep
    return time.perf_counter() - t0, updates


def evaluate(net, loader):
    """Accuracy at every depth. The final one is the model's answer; the
    early ones are the collapse signature."""
    net.eval()
    hits = [0] * DEPTH
    n = 0
    with torch.no_grad():
        for x, y in loader:
            outs = net.all_heads(x)
            n += y.numel()
            for i, o in enumerate(outs):
                hits[i] += int((o.argmax(1) == y).sum())
    return [100.0 * h / n for h in hits]


ARMS = [
    ("bp", None, False),
    ("local-full", None, False),
    ("skip-early", 60, True),
    ("skip-all-50", 50, False),
    ("skip-all-70", 70, False),
]


if __name__ == "__main__":
    print("Does layer skipping cure greedy information collapse?\n")
    print(f"CIFAR-10, {SUBSET} images, {DEPTH} blocks of {WIDTH}, "
          f"{EPOCHS} epoch, {len(SEEDS)} seeds\n")
    print("the collapse signature: greedy local learning makes EARLY "
          "layers more\naccurate than backprop while the FINAL layer is "
          "worse\n")

    def mean(xs):
        return sum(xs) / len(xs)

    results = {}
    for name, pct, early in ARMS:
        per_depth, secs, upds = [], [], []
        for seed in SEEDS:
            train_loader, test_loader = loaders()
            net = Net(seed)
            if name == "bp":
                t, u = train_bp(net, train_loader)
            else:
                t, u = train_local(net, train_loader, pct, early)
            per_depth.append(evaluate(net, test_loader))
            secs.append(t)
            upds.append(u)
            del net

        acc = [mean([p[i] for p in per_depth]) for i in range(DEPTH)]
        total = DEPTH * len(train_loader) * EPOCHS
        results[name] = dict(acc=acc, final=acc[-1], early=acc[0],
                             secs=mean(secs),
                             frac=100.0 * mean(upds) / total)
        print(f"  {name:>13}: final {acc[-1]:5.2f}%  layer1 {acc[0]:5.2f}%  "
              f"{mean(secs):5.1f}s  updates {results[name]['frac']:5.1f}%",
              flush=True)

    print("\n" + "=" * 84)
    print("ACCURACY AT EVERY DEPTH")
    print(f"{'arm':>13} " + "  ".join(f"L{i + 1:<5}" for i in range(DEPTH)))
    print("-" * 84)
    for name, _, _ in ARMS:
        print(f"{name:>13} " + "  ".join(
            f"{a:5.1f}%" for a in results[name]["acc"]))
    print("=" * 84)

    bp = results["bp"]
    full = results["local-full"]
    gap = bp["final"] - full["final"]

    print(f"\nTHE COLLAPSE SIGNATURE:")
    print(f"  backprop    layer1 {bp['early']:5.2f}%  ->  final "
          f"{bp['final']:5.2f}%")
    print(f"  local-full  layer1 {full['early']:5.2f}%  ->  final "
          f"{full['final']:5.2f}%")
    print(f"  local is {full['early'] - bp['early']:+.2f} at layer 1 and "
          f"{full['final'] - bp['final']:+.2f} at the end")
    if full["early"] > bp["early"] and full["final"] < bp["final"]:
        print("  -> collapse confirmed: better early, worse final")
    elif gap < 1:
        print("  -> NO GAP TO CLOSE at this size. The experiment cannot "
              "test the\n     hypothesis; a deeper network or more "
              "training is needed.")
    else:
        print("  -> a gap exists but the early-layer signature is absent, "
              "so the\n     cause may not be collapse here")

    print(f"\nDOES SKIPPING CLOSE THE GAP? (backprop final "
          f"{bp['final']:.2f}%)")
    for name, pct, early in ARMS:
        if pct is None:
            continue
        r = results[name]
        closed = (r["final"] - full["final"]) / gap * 100 if gap > 0.5 else 0
        print(f"  {name:>13}: final {r['final']:5.2f}% "
              f"({r['final'] - full['final']:+5.2f} vs local-full, "
              f"{closed:+4.0f}% of the gap)  layer1 "
              f"{r['early'] - full['early']:+5.2f}  "
              f"{full['secs'] / r['secs']:.2f}x faster")

    print("""
Two things have to happen together for the hypothesis to hold.

  THE GAP CLOSES. A skipping arm reaches backprop's final accuracy, or
      meaningfully closer to it than greedy local learning does.

  THE EARLY-LAYER SIGNATURE CHANGES. Layer 1 becomes LESS discriminative
      under skipping. That is what would show the mechanism is reduced
      greed rather than something incidental — a skipping arm that closes
      the gap while layer 1 stays equally discriminative is doing something
      else, and the explanation would be wrong even if the number is right.

If the gap closes AND early layers become less discriminative, skipping is a
cure for information collapse that costs less rather than more, which is
what InfoPro's reconstruction networks cost. That would be worth writing up.

If there is no gap at this scale, the honest conclusion is that this network
is too small to show the phenomenon, and the test needs more depth or more
training than a laptop affords.
""")
    with open("greedy.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote greedy.json")