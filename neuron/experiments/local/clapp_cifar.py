"""Does CLAPP reproduce on CIFAR-10?

The paper (arXiv 2601.21683, Table 2) reports for CIFAR-10:
    CLAPP++ (no 2D spatial dependence)   73.21
    CLAPP++ (with spatial dependence)    80.51
    BP-CLAPP++ (same loss, end to end)   80.49

This implements CLAPP WITHOUT spatial dependence, so the target is roughly
73, not 80. Landing near 73 means the local rule is implemented correctly
and the spatial part is worth adding next. Landing near 40 means something
is wrong and there is no point going further.

Also trains two controls on the same architecture and epoch budget:
    bp      the same network trained end to end with the same loss
    random  frozen random weights, to prove the features are learned

Protocol follows the paper: self-supervised pretraining with SimCLR-style
augmentations, then freeze, then a linear probe on the concatenated
representations from every layer.

Runs on Apple MPS. Reduced scale from the paper (they used 300 epochs on
4 A100s) so it fits on a laptop, which means the absolute number will be
lower. Read the RATIO between arms, not the absolute value.
"""

import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from clapp import CLAPPNet


EPOCHS = 15            # paper uses 300. this is a feasibility check.
BATCH = 128
LR = 2e-4              # paper: Adam at 2e-4
PROBE_EPOCHS = 10
PROBE_LR = 1e-3
SEEDS = [0]

# Smaller than the paper's network so it fits comfortably on a laptop.
CHANNELS = (64, 128, 128, 256)
POOLS = (False, True, True, True)


def device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def augment():
    """SimCLR-style augmentations, as the paper specifies."""
    return transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.3, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


class TwoViews:
    """Returns two differently augmented copies of the same image."""

    def __init__(self, tf):
        self.tf = tf

    def __call__(self, x):
        return self.tf(x), self.tf(x)


def plain():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def load(train, two_views):
    tf = TwoViews(augment()) if two_views else plain()
    return datasets.CIFAR10("./data", train=train, download=True, transform=tf)


def pretrain(mode, seed, dev):
    """mode: 'clapp' (local), 'bp' (end to end), 'random' (untrained)."""
    torch.manual_seed(seed)
    net = CLAPPNet(channels=CHANNELS, pools=POOLS, in_ch=3).to(dev)

    if mode == "random":
        return net, 0.0

    ds = load(train=True, two_views=True)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True,
                        num_workers=2)
    opt = torch.optim.Adam(net.parameters(), lr=LR)

    t0 = time.time()
    for ep in range(EPOCHS):
        total, n = 0.0, 0
        for (v1, v2), _ in loader:
            v1, v2 = v1.to(dev), v2.to(dev)
            # Negatives: shuffle the batch so each sample is paired with a
            # different image, as the paper does within a minibatch.
            perm = torch.randperm(v1.size(0), device=dev)
            v_neg = v1[perm]

            if mode == "clapp":
                losses = net.losses(v1, v_neg, v2)
                opt.zero_grad()
                # Each layer's loss backwards only into that layer, because
                # encode() detached the input at every boundary.
                sum(losses).backward()
                opt.step()
                total += float(sum(l.item() for l in losses))
            else:  # bp: same loss, but only at the TOP layer, end to end
                h_pos = net.encode(v1, detach=False)
                h_neg = net.encode(v_neg, detach=False)
                with torch.no_grad():
                    h_ref = net.encode(v2, detach=False)
                top = net.layers[-1]
                c = top.represent(h_ref[-1]).detach()
                loss = top.local_loss(h_pos[-1], h_neg[-1], c)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
            n += 1
        print(f"    epoch {ep + 1:>2}/{EPOCHS}  loss {total / n:.4f}  "
              f"({time.time() - t0:.0f}s)")
    return net, time.time() - t0


def probe(net, dev):
    """Freeze the network, train a linear classifier on its features."""
    net.eval()
    tr = DataLoader(load(True, False), batch_size=256, shuffle=True,
                    num_workers=2)
    te = DataLoader(load(False, False), batch_size=256, num_workers=2)

    clf = nn.Linear(net.feature_dim, 10).to(dev)
    opt = torch.optim.Adam(clf.parameters(), lr=PROBE_LR)

    for _ in range(PROBE_EPOCHS):
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            f = net.features(x)
            opt.zero_grad()
            F.cross_entropy(clf(f), y).backward()
            opt.step()

    correct = total = 0
    with torch.no_grad():
        for x, y in te:
            x, y = x.to(dev), y.to(dev)
            pred = clf(net.features(x)).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return 100.0 * correct / total


if __name__ == "__main__":
    dev = device()
    print(f"device {dev}   channels {CHANNELS}   {EPOCHS} epochs\n")
    results = {}

    for mode in ["random", "clapp", "bp"]:
        print(f"--- {mode} ---")
        accs = []
        for seed in SEEDS:
            net, secs = pretrain(mode, seed, dev)
            acc = probe(net, dev)
            accs.append(acc)
            print(f"  seed {seed}: linear probe {acc:.2f}%  "
                  f"(pretrain {secs / 60:.1f} min)")
            del net
            if dev == "mps":
                torch.mps.empty_cache()
        results[mode] = sum(accs) / len(accs)
        print()

    print("=" * 66)
    print(f"{'arm':>10} {'accuracy':>10}   what it means")
    print("-" * 66)
    print(f"{'random':>10} {results['random']:>9.2f}%   frozen random features, the floor")
    print(f"{'clapp':>10} {results['clapp']:>9.2f}%   LOCAL, no gradient crosses a layer")
    print(f"{'bp':>10} {results['bp']:>9.2f}%   same loss, end to end, the control")
    print("=" * 66)
    print("""
Paper reference for CIFAR-10 (Table 2, at full scale):
    CLAPP++ no spatial dependence  73.21
    CLAPP++ with it                80.51
    BP-CLAPP++                     80.49

This run is much smaller, so expect lower numbers everywhere. What matters:

  clapp clearly above random   -> the local rule is learning something
  clapp near bp                -> local matches end to end, which is the
                                  paper's central claim, reproduced
  clapp near random            -> the implementation is wrong, stop and fix
                                  it before going further
""")
    with open("clapp_cifar.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote clapp_cifar.json")