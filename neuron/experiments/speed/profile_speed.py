"""Where does the time actually go?

The field measured local learning at no wall-clock advantage. But CLAPP's
Table 3 measured "time per batched iteration on an A100 GPU" at batch size
256 — a saturated datacenter chip, where arithmetic dominates and a local
rule's advantage disappears into kernel efficiency.

This project runs batch size ONE on a laptop. A 128-unit GRU on a single
sample uses a fraction of a percent of the chip. In that regime the cost
should be dominated by MACHINERY rather than arithmetic: building the
autograd graph, launching kernels, Python overhead, optimiser bookkeeping.

Supporting evidence from this project's own numbers: the overlapping-window
fix cost six times the compute. It does not do six times the arithmetic. It
rebuilds a twenty-step autograd graph on every step.

THE HYPOTHESIS: in single-sample online learning, autodiff machinery is the
bottleneck, not FLOPs. If so, a closed-form local rule that needs no autodiff
at all could be several times faster rather than the 33% the FLOP accounting
predicts, because it deletes the thing that actually costs.

This measures it. Five components timed separately:

  encode      building the input tensor from raw features
  forward     the forward pass with NO graph (torch.no_grad)
  graph       the extra cost of building the autograd graph, measured as
              forward-with-grad minus forward-without
  backward    the backward pass
  optimiser   the Adam step

Then the same at batch 256, to show the regime difference directly. If the
machinery share collapses at batch 256 and dominates at batch 1, that is the
explanation for why the field's measurement does not transfer.

  machinery (encode + graph + optimiser) is 50%+ at batch 1 -> the
      hypothesis holds and a closed-form rule has a real target
  arithmetic dominates even at batch 1 -> hypothesis wrong, and the honest
      conclusion is that the speed problem here is the same one the field
      already failed to solve
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F


N_FEAT = 16          # matches the weather task with LAG=3
N_CLASS = 3
HIDDEN = 128
LR = 3e-4
REPS = 3000          # timed iterations per component
WARMUP = 300


class GRU(nn.Module):
    """The same shape as the weather model."""

    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.enc = nn.Linear(N_FEAT, HIDDEN)
        self.cell = nn.GRUCell(HIDDEN, HIDDEN)
        self.head = nn.Linear(HIDDEN, N_CLASS)

    def forward(self, x, h=None):
        e = F.relu(self.enc(x))
        h = self.cell(e, h)
        return self.head(h), h


def timed(fn, reps=REPS, warmup=WARMUP):
    """Median-of-thirds timing, to blunt scheduler noise."""
    for _ in range(warmup):
        fn()
    chunks = []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(reps // 3):
            fn()
        chunks.append((time.perf_counter() - t0) / (reps // 3))
    chunks.sort()
    return chunks[1] * 1e6           # microseconds per call


def measure(batch):
    net = GRU()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    raw = [[0.1 * i for i in range(N_FEAT)] for _ in range(batch)]
    y = torch.zeros(batch, dtype=torch.long)
    h0 = torch.zeros(batch, HIDDEN)

    out = {}

    # 1. Building the input tensor from raw python floats. This is real
    #    work in an online loop, where every step encodes a fresh sample.
    def do_encode():
        return torch.tensor(raw, dtype=torch.float32)
    out["encode"] = timed(do_encode)

    x = torch.tensor(raw, dtype=torch.float32)

    # 2. Forward with no graph. This is the pure arithmetic.
    net.eval()
    def do_forward():
        with torch.no_grad():
            net(x, h0)
    out["forward"] = timed(do_forward)

    # 3. Forward WITH graph. The difference is the cost of recording
    #    operations for autodiff.
    net.train()
    def do_forward_grad():
        logits, h = net(x, h0)
        return logits
    out["forward_grad"] = timed(do_forward_grad)
    out["graph"] = max(0.0, out["forward_grad"] - out["forward"])

    # 4. Backward. Needs a fresh graph each time, so the forward cost is
    #    subtracted out.
    def do_full():
        logits, h = net(x, h0)
        loss = F.cross_entropy(logits, y)
        loss.backward()
    out["fwd_plus_bwd"] = timed(do_full)
    out["backward"] = max(0.0, out["fwd_plus_bwd"] - out["forward_grad"])

    # 5. The optimiser step, with gradients already present.
    logits, _ = net(x, h0)
    F.cross_entropy(logits, y).backward()
    def do_opt():
        opt.step()
    out["optimiser"] = timed(do_opt)

    def do_zero():
        opt.zero_grad()
    out["zero_grad"] = timed(do_zero)

    return out


PARTS = ["encode", "forward", "graph", "backward", "optimiser", "zero_grad"]
MACHINERY = ["encode", "graph", "optimiser", "zero_grad"]
ARITHMETIC = ["forward", "backward"]


if __name__ == "__main__":
    print(f"GRU {N_FEAT}->{HIDDEN}->{N_CLASS}, "
          f"{sum(p.numel() for p in GRU().parameters())} parameters")
    print(f"{REPS} timed iterations per component, microseconds each\n")

    results = {}
    for batch in [1, 256]:
        print(f"--- batch size {batch} ---", flush=True)
        r = measure(batch)
        results[batch] = r
        total = sum(r[p] for p in PARTS)
        for p in PARTS:
            share = 100.0 * r[p] / total
            bar = "#" * int(share / 2)
            print(f"  {p:>12}: {r[p]:>8.1f} us  ({share:5.1f}%) {bar}")
        mach = sum(r[p] for p in MACHINERY)
        arith = sum(r[p] for p in ARITHMETIC)
        print(f"  {'':>12}  {'-' * 30}")
        print(f"  {'machinery':>12}: {mach:>8.1f} us  "
              f"({100.0 * mach / total:5.1f}%)")
        print(f"  {'arithmetic':>12}: {arith:>8.1f} us  "
              f"({100.0 * arith / total:5.1f}%)")
        print(f"  {'total':>12}: {total:>8.1f} us"
              f"{'' if batch == 1 else f'  ({total / batch:.2f} us/sample)'}")
        print()

    print("=" * 66)
    print("MACHINERY SHARE, batch 1 against batch 256")
    print(f"{'':>14} {'batch 1':>10} {'batch 256':>12}")
    print("-" * 66)
    for p in PARTS:
        a = 100.0 * results[1][p] / sum(results[1][q] for q in PARTS)
        b = 100.0 * results[256][p] / sum(results[256][q] for q in PARTS)
        print(f"{p:>14} {a:>9.1f}% {b:>11.1f}%")
    print("-" * 66)
    m1 = 100.0 * sum(results[1][p] for p in MACHINERY) / \
        sum(results[1][q] for q in PARTS)
    m2 = 100.0 * sum(results[256][p] for p in MACHINERY) / \
        sum(results[256][q] for q in PARTS)
    print(f"{'machinery':>14} {m1:>9.1f}% {m2:>11.1f}%")
    print("=" * 66)

    per1 = sum(results[1][p] for p in PARTS)
    per256 = sum(results[256][p] for p in PARTS) / 256
    print(f"\ncost per sample: batch 1 {per1:.1f} us, "
          f"batch 256 {per256:.2f} us")
    print(f"batching is {per1 / per256:.0f}x cheaper per sample, which is "
          f"the\nregime difference the field's measurements were taken in")

    print(f"""
The machinery row is the answer.

  high at batch 1, low at batch 256 -> CONFIRMED. The field measured local
      learning where arithmetic dominates and found no advantage. In the
      single-sample online regime the cost is autodiff machinery instead,
      which a closed-form rule removes entirely rather than reducing by a
      third. That is a different and much larger target, and nobody has
      measured it because nobody trains at batch size 1 on a laptop.

  high in both -> still interesting, but the problem is Python and PyTorch
      overhead rather than anything about learning rules, and the fix is
      engineering (torch.compile, a hand-written update) rather than a new
      algorithm.

  low at batch 1 -> hypothesis wrong. Arithmetic dominates even here, the
      speed problem is the same one the field already failed to solve, and
      the honest move is to stop.
""")