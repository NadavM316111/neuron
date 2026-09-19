"""Are the SAL paper's baselines undertrained?

The paper (arXiv 2601.21561) trains its BP baseline for 25 epochs with
plain SGD, batch 16, learning rate 0.0001, and converts colour datasets to
grayscale. Its reported baseline accuracies look far too low:

    CIFAR-10  30.81      MNIST  90.64      Digits  38.53     Semeion  35.16

SAL-16 is then reported as beating those:

    CIFAR-10  36.60      MNIST  94.71      Digits  71.63     Semeion  72.03

This script trains the SAME architecture (2 layers, 256 hidden, ReLU,
Kaiming init) three ways:

  paper     their exact recipe: SGD lr 1e-4, batch 16, 25 epochs, grayscale
  fair      Adam lr 1e-3, batch 128, 25 epochs, grayscale
  fair_rgb  Adam lr 1e-3, batch 128, 25 epochs, colour kept

If "fair" beats SAL-16, the paper's central comparison is void: SAL is not
beating backpropagation, it is beating an undertrained baseline.

Runs on CPU or Apple MPS. No GPU rental needed.
"""

import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


SEEDS = [0, 1, 2]
EPOCHS = 25
HIDDEN = 256

# From the paper's Table 2, for comparison.
PAPER = {
    "MNIST":    dict(baseline=90.64, sal16=94.71),
    "CIFAR10":  dict(baseline=30.81, sal16=36.60),
    "Digits":   dict(baseline=38.53, sal16=71.63),
    "Semeion":  dict(baseline=35.16, sal16=72.03),
}

RECIPES = {
    "paper":    dict(opt="sgd",  lr=1e-4, batch=16,  gray=True),
    "fair":     dict(opt="adam", lr=1e-3, batch=128, gray=True),
    "fair_rgb": dict(opt="adam", lr=1e-3, batch=128, gray=False),
}


def device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_mnist(gray=True):
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5,), (0.5,))])
    tr = datasets.MNIST("./data", train=True, download=True, transform=tf)
    te = datasets.MNIST("./data", train=False, download=True, transform=tf)
    return to_tensors(tr), to_tensors(te), 28 * 28, 10


def load_cifar(gray=True):
    ops = [transforms.ToTensor()]
    if gray:
        ops.insert(0, transforms.Grayscale(num_output_channels=1))
        ops.append(transforms.Normalize((0.5,), (0.5,)))
        dim = 32 * 32
    else:
        ops.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        dim = 32 * 32 * 3
    tf = transforms.Compose(ops)
    tr = datasets.CIFAR10("./data", train=True, download=True, transform=tf)
    te = datasets.CIFAR10("./data", train=False, download=True, transform=tf)
    return to_tensors(tr), to_tensors(te), dim, 10


def to_tensors(ds):
    loader = DataLoader(ds, batch_size=len(ds))
    x, y = next(iter(loader))
    return x.reshape(x.shape[0], -1), y


def load_digits_sklearn():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    d = load_digits()
    x = (d.data - d.data.mean()) / (d.data.std() + 1e-8)
    xtr, xte, ytr, yte = train_test_split(x, d.target, test_size=0.2,
                                          random_state=0, stratify=d.target)
    f = lambda a: torch.tensor(a, dtype=torch.float32)
    g = lambda a: torch.tensor(a, dtype=torch.long)
    return (f(xtr), g(ytr)), (f(xte), g(yte)), 64, 10


def load_semeion():
    """Semeion: 1593 handwritten digits, 16x16 binary. Fetched from UCI."""
    import urllib.request, os
    path = "./data/semeion.data"
    os.makedirs("./data", exist_ok=True)
    if not os.path.exists(path):
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
               "semeion/semeion.data")
        urllib.request.urlretrieve(url, path)
    raw = np.loadtxt(path)
    x = raw[:, :256]
    y = raw[:, 256:].argmax(axis=1)
    x = (x - 0.5) / 0.5
    from sklearn.model_selection import train_test_split
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2,
                                          random_state=0, stratify=y)
    f = lambda a: torch.tensor(a, dtype=torch.float32)
    g = lambda a: torch.tensor(a, dtype=torch.long)
    return (f(xtr), g(ytr)), (f(xte), g(yte)), 256, 10


LOADERS = {
    "MNIST":   lambda gray: load_mnist(gray),
    "CIFAR10": lambda gray: load_cifar(gray),
    "Digits":  lambda gray: load_digits_sklearn(),
    "Semeion": lambda gray: load_semeion(),
}


def make_model(d_in, n_classes):
    """Exactly the paper's architecture: 2 layers, 256 hidden, ReLU,
    Kaiming init."""
    m = nn.Sequential(
        nn.Linear(d_in, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, n_classes),
    )
    for layer in m:
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
    return m


def run(dataset, recipe_name, seed):
    cfg = RECIPES[recipe_name]
    torch.manual_seed(seed)
    np.random.seed(seed)

    (xtr, ytr), (xte, yte), d_in, n_classes = LOADERS[dataset](cfg["gray"])
    dev = device()
    model = make_model(d_in, n_classes).to(dev)

    if cfg["opt"] == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"])
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    loader = DataLoader(TensorDataset(xtr, ytr), batch_size=cfg["batch"],
                        shuffle=True)
    xte, yte = xte.to(dev), yte.to(dev)
    lossf = nn.CrossEntropyLoss()

    t0 = time.time()
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            lossf(model(xb), yb).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        acc = (model(xte).argmax(1) == yte).float().mean().item() * 100
    return acc, time.time() - t0


if __name__ == "__main__":
    print(f"device {device()}   architecture: {HIDDEN} hidden, 2 layers, "
          f"ReLU, Kaiming, {EPOCHS} epochs\n")

    results = {}
    for dataset in PAPER:
        results[dataset] = {}
        print(f"--- {dataset} ---")
        for recipe in RECIPES:
            if dataset in ("Digits", "Semeion") and recipe == "fair_rgb":
                continue                      # already single-channel
            accs, secs = [], []
            for seed in SEEDS:
                a, s = run(dataset, recipe, seed)
                accs.append(a)
                secs.append(s)
            mean = sum(accs) / len(accs)
            std = float(np.std(accs))
            results[dataset][recipe] = dict(mean=mean, std=std,
                                            secs=sum(secs) / len(secs))
            print(f"  {recipe:>9}  {mean:6.2f} +- {std:4.2f}   "
                  f"({sum(secs)/len(secs):5.1f}s per run)")
        print()

    print("=" * 76)
    print(f"{'dataset':>10} {'paper BP':>10} {'my paper':>10} "
          f"{'my fair':>10} {'SAL-16':>10} {'verdict':>16}")
    print("-" * 76)
    for dataset, ref in PAPER.items():
        mine_paper = results[dataset]["paper"]["mean"]
        mine_fair = max(results[dataset][r]["mean"]
                        for r in results[dataset] if r != "paper")
        beat = mine_fair > ref["sal16"]
        verdict = "BP wins" if beat else "SAL still ahead"
        print(f"{dataset:>10} {ref['baseline']:>10.2f} {mine_paper:>10.2f} "
              f"{mine_fair:>10.2f} {ref['sal16']:>10.2f} {verdict:>16}")
    print("=" * 76)
    print("""
"my paper" should roughly reproduce their baseline column. If it does, their
recipe is faithfully implemented here and the comparison below is fair.

"my fair" is the same architecture trained properly. If it beats SAL-16, the
paper's headline result is an artifact of an undertrained baseline, and
Stage 1 ends here rather than after a month of reimplementation.

If SAL-16 still wins on a properly trained baseline, the paper deserves the
full reproduction and Stage 1 proceeds as planned.
""")
    with open("baseline_check.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote baseline_check.json")