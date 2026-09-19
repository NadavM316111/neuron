"""Does the gap close with proper training, and does skipping get there
faster?

The 18-point gap measured this evening was on ONE epoch of 20,000 images.
That is not the field's fundamental gap — the literature reports that at
CIFAR scale, "there is almost no cost of introducing local greedy losses",
with local methods matching backprop's test error on MNIST and CIFAR
(Scaling Forward Gradient with Local Losses). The gap that genuinely
persists is on high-resolution ImageNet, where local-SSL trails 42-44%
against 48-55%, and the stated cause is that "the error due to greediness
grows as the problem gets more complex and requires more layers to
cooperate".

So the honest question for THIS project is not whether the field's gap can
be closed. It is whether the gap exists at all in the regime this project
operates in — small networks, single device, modest data — once training is
adequate.

TWO THINGS MEASURED, at every epoch rather than only at the end:

  DOES THE GAP CLOSE? If local-full converges toward backprop as epochs
      accumulate, the 18 points were an undertraining artifact and there is
      nothing to fix here. That would be a real answer, and it would mean
      the contribution from this thread is the SPEED lever rather than an
      accuracy fix.

  DOES SKIPPING GET THERE FASTER IN WALL CLOCK? This is the claim worth
      having. Accuracy per SECOND, not per epoch. Skipping does fewer
      updates per epoch, so it should lose per-epoch and win per-second.

The per-epoch curves matter more than the final numbers. A method that ends
in the same place but arrives sooner is the useful one for a system that
learns continuously on someone's laptop.

Arms:
  bp             end-to-end backpropagation
  local-full     greedy local losses, every layer every step
  skip-early     skipping on the first half of layers, where greedy
                 collapse happens. Closed 43% of the one-epoch gap.
  skip-all-50    skipping everywhere, which closed none of it
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
EPOCHS = 8               # the whole point: enough to converge
SEEDS = [0]              # one seed, because eight epochs x four arms
HISTORY = 100
DATA = "/Users/nadavminkowitz/neuron/data"
SUBSET = 20000
TEST_SUBSET = 4000


class Net(nn.Module):
    """Six convolutional blocks, each with its own auxiliary classifier."""

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


class Percentiles:
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


def evaluate(net, loader):
    net.eval()
    hits = n = 0
    with torch.no_grad():
        for x, y in loader:
            hits += int((net(x).argmax(1) == y).sum())
            n += y.numel()
    return 100.0 * hits / n


def run(name, skip_pct, early_only, seed):
    """Train for EPOCHS, recording accuracy and elapsed time after each.

    The curve is the result here, not the endpoint. A method that reaches
    the same accuracy sooner in wall clock is the useful one for a system
    meant to learn continuously on a laptop.
    """
    train_loader, test_loader = loaders()
    net = Net(seed)

    if name == "bp":
        opts = [torch.optim.Adam(net.parameters(), lr=LR)]
    else:
        opts = [torch.optim.Adam(
            list(net.blocks[i].parameters())
            + list(net.heads[i].parameters()), lr=LR)
            for i in range(DEPTH)]

    pct = Percentiles(DEPTH)
    curve = []
    elapsed = 0.0
    updates = 0

    for ep in range(EPOCHS):
        t0 = time.perf_counter()
        net.train()
        for x, y in train_loader:
            if name == "bp":
                loss = F.cross_entropy(net(x), y)
                opts[0].zero_grad()
                loss.backward()
                opts[0].step()
                updates += DEPTH
            else:
                h = x
                for i in range(DEPTH):
                    h_in = h.detach()
                    rep = net.blocks[i](h_in)
                    loss = F.cross_entropy(net.heads[i](rep), y)
                    v = float(loss.item())

                    do = True
                    if skip_pct is not None:
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
        elapsed += time.perf_counter() - t0
        acc = evaluate(net, test_loader)
        curve.append(dict(epoch=ep + 1, acc=acc, secs=elapsed))
        print(f"    {name:>12} epoch {ep + 1}: {acc:5.2f}%  "
              f"{elapsed:6.1f}s", flush=True)

    del net, opts
    return curve, updates


ARMS = [
    ("bp", None, False),
    ("local-full", None, False),
    ("skip-early", 60, True),
    ("skip-all-50", 50, False),
]


if __name__ == "__main__":
    print("Does the gap close with proper training?\n")
    print(f"CIFAR-10, {SUBSET} images, {EPOCHS} epochs, {DEPTH} blocks\n")
    print("The 18-point gap measured earlier was on ONE epoch. The "
          "literature says\nlocal greedy losses cost almost nothing at "
          "CIFAR scale with real training.\n")

    results = {}
    for name, pct, early in ARMS:
        print(f"  --- {name} ---", flush=True)
        curve, upd = run(name, pct, early, SEEDS[0])
        results[name] = dict(curve=curve, updates=upd)
        print()

    print("=" * 80)
    print("ACCURACY BY EPOCH")
    print(f"{'arm':>13} " + "  ".join(f"e{i + 1:<5}" for i in range(EPOCHS)))
    print("-" * 80)
    for name, _, _ in ARMS:
        print(f"{name:>13} " + "  ".join(
            f"{c['acc']:5.1f}" for c in results[name]["curve"]))
    print("=" * 80)

    bp_final = results["bp"]["curve"][-1]["acc"]
    bp_secs = results["bp"]["curve"][-1]["secs"]

    print(f"\nFINAL, after {EPOCHS} epochs:")
    for name, _, _ in ARMS:
        c = results[name]["curve"][-1]
        print(f"  {name:>13}: {c['acc']:5.2f}%  "
              f"({c['acc'] - bp_final:+5.2f} vs backprop)  "
              f"{c['secs']:6.1f}s  ({bp_secs / c['secs']:.2f}x)")

    # the gap at one epoch against the gap at the end
    e1 = {n: results[n]["curve"][0]["acc"] for n, _, _ in ARMS}
    eN = {n: results[n]["curve"][-1]["acc"] for n, _, _ in ARMS}
    print(f"\nDID THE GAP CLOSE WITH TRAINING?")
    print(f"  after 1 epoch : local-full is "
          f"{e1['local-full'] - e1['bp']:+.2f} vs backprop")
    print(f"  after {EPOCHS} epochs: local-full is "
          f"{eN['local-full'] - eN['bp']:+.2f} vs backprop")
    closed = (e1['bp'] - e1['local-full']) - (eN['bp'] - eN['local-full'])
    print(f"  the gap narrowed by {closed:+.2f} points")

    print(f"\nACCURACY PER SECOND — the number that matters for a system "
          f"that\nlearns continuously rather than in a training run:")
    for name, _, _ in ARMS:
        c = results[name]["curve"][-1]
        print(f"  {name:>13}: {c['acc'] / c['secs']:.3f} %/s")

    # when does each arm first reach backprop's halfway accuracy
    target = bp_final * 0.9
    print(f"\nSECONDS TO REACH {target:.1f}% "
          f"(90% of backprop's final):")
    for name, _, _ in ARMS:
        hit = next((c for c in results[name]["curve"]
                    if c["acc"] >= target), None)
        if hit:
            print(f"  {name:>13}: {hit['secs']:6.1f}s "
                  f"(epoch {hit['epoch']})")
        else:
            print(f"  {name:>13}: never reached")

    print("""
Two questions, and the second is the one worth publishing.

  DOES THE GAP CLOSE? If local-full converges toward backprop over eight
      epochs, then the 18 points measured earlier were undertraining and
      there is no accuracy problem to solve in this regime. That is a real
      answer even though it is not the exciting one: it would mean the
      contribution from this thread is the SPEED lever, not an accuracy
      fix.

  DOES SKIPPING ARRIVE SOONER? Accuracy per second, and time to reach a
      fixed target. Skipping does fewer updates per epoch, so it should
      lose per-epoch and win per-second. For a system learning
      continuously on a laptop, per-second is the only measure that
      matters — nobody counts epochs on a stream that never ends.

If local-full converges to backprop AND skipping reaches the same place
sooner in wall clock, the honest claim is: in the single-device regime,
local learning matches backpropagation and can be made faster than it. That
is narrower than "the gap is closed" and it is true.
""")
    with open("converge.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote converge.json")