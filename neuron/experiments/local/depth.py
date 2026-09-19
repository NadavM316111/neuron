"""Does early-layer skipping help MORE as networks get deeper?

Yesterday, on six blocks: skipping the first half of layers closed 43% of
the gap between greedy local learning and backpropagation, while skipping
EVERY layer closed none of it. That differential is the evidence — if the
mechanism were merely "fewer updates help", skipping everywhere would work
too. It only helps where information collapse is documented to happen.

THE PREDICTION THIS TESTS. The literature says greedy collapse gets WORSE
with depth: early modules become discriminative too soon, discard
information later modules need, and "the error due to greediness grows as
the problem gets more complex and requires more layers to cooperate". More
layers means more opportunity to collapse.

If skipping works by limiting how much early layers over-optimise, then a
DEEPER network should have MORE collapse to prevent and skipping should
close a LARGER share of the gap.

That is a real prediction with a direction. It can fail: the benefit could
shrink with depth, or vanish, or the whole thing could turn out to be a
six-layer coincidence.

Depths 4, 8 and 12, each with backprop, greedy local, and early skipping.
The number to watch is the SHARE of the gap closed at each depth — the
absolute gap will grow with depth on its own, so the raw points are not
comparable across depths and the fraction is.

Reduced from yesterday's settings to keep this to an afternoon: fewer
images, three epochs, one seed. The question is a trend across depth, not
a precise number at any one depth.
"""

import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


DEPTHS = [4, 8, 12]
WIDTH = 48
LR = 1e-3
BATCH = 128
EPOCHS = 3
SEED = 0
HISTORY = 100
SKIP_PCT = 60
DATA = "/Users/nadavminkowitz/neuron/data"
SUBSET = 12000
TEST_SUBSET = 3000


class Net(nn.Module):
    """A stack of conv blocks, each with its own auxiliary classifier so a
    local loss can be computed. Pooling only in the first few blocks, so
    deeper networks do not shrink the feature map to nothing."""

    def __init__(self, depth, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.depth = depth
        chans = [3] + [WIDTH] * depth
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(chans[i], chans[i + 1], 3, padding=1),
                nn.BatchNorm2d(chans[i + 1]),
                nn.ReLU(),
                nn.MaxPool2d(2) if i < 3 else nn.Identity())
            for i in range(depth)])
        self.heads = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                          nn.Linear(WIDTH, 10))
            for _ in range(depth)])

    def forward(self, x):
        h = x
        for block in self.blocks:
            h = block(h)
        return self.heads[-1](h)


class Percentiles:
    """Per-layer loss history. Layers sit at different loss scales, so
    'unusually high' must be judged within a layer."""

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
    tr = datasets.CIFAR10(DATA, train=True, download=False, transform=tf)
    te = datasets.CIFAR10(DATA, train=False, download=False, transform=tf)
    tr = torch.utils.data.Subset(tr, range(SUBSET))
    te = torch.utils.data.Subset(te, range(TEST_SUBSET))
    return (torch.utils.data.DataLoader(tr, batch_size=BATCH, shuffle=True),
            torch.utils.data.DataLoader(te, batch_size=256))


def evaluate(net, loader):
    net.eval()
    hits = n = 0
    with torch.no_grad():
        for x, y in loader:
            hits += int((net(x).argmax(1) == y).sum())
            n += y.numel()
    return 100.0 * hits / n


def run(arm, depth, train_loader, test_loader):
    """arm is 'bp', 'local', or 'skip-early'."""
    net = Net(depth, SEED)
    if arm == "bp":
        opts = [torch.optim.Adam(net.parameters(), lr=LR)]
    else:
        opts = [torch.optim.Adam(
            list(net.blocks[i].parameters())
            + list(net.heads[i].parameters()), lr=LR)
            for i in range(depth)]

    pct = Percentiles(depth)
    updates = 0
    t0 = time.perf_counter()

    for _ in range(EPOCHS):
        net.train()
        for x, y in train_loader:
            if arm == "bp":
                loss = F.cross_entropy(net(x), y)
                opts[0].zero_grad()
                loss.backward()
                opts[0].step()
                updates += depth
            else:
                h = x
                for i in range(depth):
                    rep = net.blocks[i](h.detach())
                    loss = F.cross_entropy(net.heads[i](rep), y)
                    v = float(loss.item())

                    do = True
                    if arm == "skip-early" and i < depth // 2:
                        # skipping applies only to the first half, which is
                        # where information collapse is documented to occur
                        do = pct.above(i, v, SKIP_PCT)
                    pct.push(i, v)

                    if do:
                        opts[i].zero_grad()
                        loss.backward()
                        opts[i].step()
                        updates += 1
                    h = rep

    secs = time.perf_counter() - t0
    acc = evaluate(net, test_loader)
    del net, opts
    return acc, secs, updates


if __name__ == "__main__":
    print("Does early-layer skipping help MORE as networks get deeper?\n")
    print(f"CIFAR-10, {SUBSET} images, {EPOCHS} epochs, depths {DEPTHS}, "
          f"width {WIDTH}\n")
    print("Greedy collapse is documented to WORSEN with depth. If skipping "
          "works by\nlimiting early over-optimisation, deeper networks "
          "should benefit MORE.\n")

    train_loader, test_loader = loaders()
    results = {}

    for depth in DEPTHS:
        print(f"--- depth {depth} ---", flush=True)
        row = {}
        for arm in ["bp", "local", "skip-early"]:
            acc, secs, upd = run(arm, depth, train_loader, test_loader)
            row[arm] = dict(acc=acc, secs=secs, updates=upd)
            print(f"  {arm:>11}: {acc:5.2f}%  {secs:6.1f}s", flush=True)
        gap = row["bp"]["acc"] - row["local"]["acc"]
        recovered = row["skip-early"]["acc"] - row["local"]["acc"]
        row["gap"] = gap
        row["recovered"] = recovered
        row["share"] = 100.0 * recovered / gap if gap > 0.5 else float("nan")
        results[depth] = row
        print(f"  gap {gap:.2f}, recovered {recovered:+.2f} "
              f"({row['share']:.0f}% of it)\n", flush=True)

    print("=" * 76)
    print(f"{'depth':>7} {'backprop':>10} {'local':>9} {'skip-early':>12} "
          f"{'gap':>7} {'closed':>9}")
    print("-" * 76)
    for depth in DEPTHS:
        r = results[depth]
        print(f"{depth:>7} {r['bp']['acc']:>9.2f}% {r['local']['acc']:>8.2f}% "
              f"{r['skip-early']['acc']:>11.2f}% {r['gap']:>6.2f} "
              f"{r['share']:>8.0f}%")
    print("=" * 76)

    shares = [results[d]["share"] for d in DEPTHS]
    gaps = [results[d]["gap"] for d in DEPTHS]

    print(f"\nDOES THE GAP GROW WITH DEPTH? "
          f"{gaps[0]:.1f} -> {gaps[-1]:.1f} points")
    print(f"DOES SKIPPING CLOSE MORE OF IT? "
          f"{shares[0]:.0f}% -> {shares[-1]:.0f}%")

    print("\nSPEED, skip-early against backprop:")
    for depth in DEPTHS:
        r = results[depth]
        print(f"  depth {depth:>2}: {r['bp']['secs'] / r['skip-early']['secs']:.2f}x")

    print()
    if shares[-1] > shares[0] + 10:
        print("  THE PREDICTION HELD. Skipping closes a larger share of the")
        print("  gap in deeper networks, which is what the collapse")
        print("  explanation predicts and what a coincidence would not.")
    elif shares[-1] < shares[0] - 10:
        print("  THE PREDICTION FAILED IN THE OPPOSITE DIRECTION. Skipping")
        print("  helps LESS with depth, so whatever it is doing, it is not")
        print("  preventing a problem that grows with depth.")
    else:
        print("  FLAT. Skipping closes roughly the same share at every")
        print("  depth. The effect is real but not depth-dependent, so the")
        print("  collapse explanation is not supported by this test even")
        print("  though the earlier differential supported it.")

    print("""
The share of the gap closed is the number that matters, not the raw points.
The absolute gap grows with depth on its own, so points recovered are not
comparable across depths.

  SHARE RISES WITH DEPTH -> the mechanism is preventing something that
      worsens with depth, which is exactly what greedy information
      collapse does. That would make yesterday's result a property of the
      method rather than of six-block networks, and it would be worth
      writing up.

  SHARE FLAT OR FALLING -> skipping does something real but not what was
      claimed. The honest write-up would report the effect without the
      collapse explanation.

Three epochs and one seed. This is a direction test, not a measurement.
""")
    with open("depth.json", "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print("wrote depth.json")