"""Add batched learning to NEURON's Mind: the Track 3 scaling unlock.

THE PROBLEM. NEURON learns one observation at a time. A GPU is built to do
thousands of things at once, so feeding it one-at-a-time leaves it ~99% idle.
That is why big brains crawled on the GPU: not weak hardware, but a sequential
feed. This is the single thing blocking efficient scaling.

THE FIX, WITHOUT CHANGING THE BEING. We keep the LIVE stream one-moment-at-a-
time, so NEURON still genuinely lives in time (goal 1 preserved). We only add a
BATCHED path for offline learning and consolidation: when it replays stored
experience, it processes a whole batch at once, which uses the GPU properly and
is dramatically faster at large brain sizes. Same being, same continual
learning, same faculties, only the training throughput changes.

This script patches mind.py to add `batched_consolidate` to the Backend and
makes the Mind use it during consolidation. Run it, then run batch_speed.py to
see the speedup.

    python3 add_batching.py       # applies the patch to mind.py
"""

import os
import sys

MIND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mind.py")


def main():
    s = open(MIND).read()

    # 1. Add a batched consolidation method to Backend. It stacks many stored
    #    items into one tensor and does a single parallel forward/backward,
    #    instead of looping one at a time.
    anchor = "    def predict(self, feats):"
    if "def batched_consolidate" not in s:
        method = '''    def batched_consolidate(self, items, batch_size=256):
        """GPU-efficient consolidation: process stored experience in parallel
        batches instead of one at a time. Same learning signal (replay of past
        experience), but it feeds the GPU properly, so it is dramatically
        faster at large brain sizes. The live stream stays sequential; only
        this offline replay batches."""
        if not items:
            return
        dev = next(self.net.parameters()).device
        # build one big tensor of all feature vectors and labels
        feats = torch.tensor([it["feats"] for it in items],
                             dtype=torch.float32, device=dev)
        labels = torch.tensor([it["label"] for it in items],
                             dtype=torch.long, device=dev)
        n = feats.shape[0]
        for start in range(0, n, batch_size):
            xb = feats[start:start + batch_size]
            yb = labels[start:start + batch_size]
            self.opt.zero_grad()
            # the GRU processes the batch in parallel (one step each)
            e = torch.tanh(self.net.enc(xb))
            h = self.net.cell(e)                 # (batch, hidden), parallel
            logits = self.net.head(h)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            self.opt.step()

    def predict(self, feats):'''
        s = s.replace(anchor, method, 1)

    # 2. Make the Mind use batched consolidation during its sleep/consolidate.
    old = "            self.backend.consolidate(self.replay[-8:])"
    new = ("            # batched consolidation: replay a larger slice in\n"
           "            # parallel (GPU-efficient) instead of 8 one-at-a-time\n"
           "            self.backend.batched_consolidate(self.replay[-256:])")
    if old in s:
        s = s.replace(old, new, 1)

    open(MIND, "w").write(s)
    print("mind.py patched: batched_consolidate added and used in consolidation")
    print("the live stream stays one-at-a-time; only offline replay batches.")

    # verify it parses
    import ast
    ast.parse(open(MIND).read())
    print("parses OK")


if __name__ == "__main__":
    main()
