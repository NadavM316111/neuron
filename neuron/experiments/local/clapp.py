"""CLAPP: the local learning rule from arXiv 2601.21683, built from the paper.

Each layer has its OWN loss. No gradient crosses layer boundaries. That is
the whole point: no correction travels backward through the network, so
peak memory does not scale with depth.

From the paper, Section 3, using their notation:

  Each layer l produces activity z^l for a positive sample and z^l for a
  negative one. Each is scored against a reference c^l through a trainable
  matrix B^l:  score = z^l . B^l c^l

  CLAPP uses a type 2 loss with f(x) = max(0, 1 - x), a hinge:
      L^l = f(score_pos) + f(-score_neg)

  For plain CLAPP the reference is z'^l, the same layer's activity on a
  second augmented view of the same image (Table 1).
  For CLAPP++DFB the reference is z'^L, the TOP layer's activity, which
  requires a forward pass first but still no backward pass (Algorithm 2).

  CLAPP is normalization-free and B^l is trainable, which is why the paper
  focuses its theory on CLAPP rather than Forward-Forward.

Not implemented here: the 2D spatial dependence of B^l (their main
contribution, Section 3.3). Start without it, add it if the baseline
reproduces. Their ablation says spatial dependence is where most of the
gain comes from, so expect to land near their "CLAPP++ (no 2D spatial
dependence)" row: 73.21 on CIFAR-10, not 80.51.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLAPPLayer(nn.Module):
    """One convolutional layer plus its own local loss and predictor.

    The layer's parameters are updated ONLY by its own loss. Gradients are
    stopped at the input, which is what makes this local.
    """

    def __init__(self, in_ch, out_ch, kernel=3, pool=False, pred_dim=None):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2)
        self.pool = nn.MaxPool2d(2) if pool else None
        # B^l in the paper: trainable, maps the reference into the score space.
        d = pred_dim or out_ch
        self.B = nn.Linear(d, out_ch, bias=False)
        self.out_ch = out_ch

    def forward(self, x):
        h = F.relu(self.conv(x))
        if self.pool is not None:
            h = self.pool(h)
        return h

    def represent(self, h):
        """Global average pool to a vector, as in the paper's equation 6."""
        return h.mean(dim=(2, 3))

    def local_loss(self, h_pos, h_neg, c):
        """Type 2 hinge loss from Table 1: f(x) = max(0, 1 - x).

        score_pos should be high, score_neg should be low.
        """
        z_pos = self.represent(h_pos)
        z_neg = self.represent(h_neg)
        proj = self.B(c)                      # B^l c^l
        s_pos = (z_pos * proj).sum(dim=1)
        s_neg = (z_neg * proj).sum(dim=1)
        return (F.relu(1.0 - s_pos) + F.relu(1.0 + s_neg)).mean()


class CLAPPNet(nn.Module):
    """A stack of CLAPP layers. Each trains itself.

    dfb=False  reference is the same layer's activity on a second view
    dfb=True   reference is the TOP layer's activity (CLAPP++DFB). Needs a
               forward pass to compute the reference first, but still no
               backward pass through the stack.
    """

    def __init__(self, channels=(128, 256, 256, 512, 1024, 1024),
                 pools=(False, True, False, True, True, True),
                 in_ch=3, dfb=False):
        super().__init__()
        self.dfb = dfb
        layers = []
        prev = in_ch
        top = channels[-1]
        for ch, p in zip(channels, pools):
            pred_dim = top if dfb else ch
            layers.append(CLAPPLayer(prev, ch, pool=p, pred_dim=pred_dim))
            prev = ch
        self.layers = nn.ModuleList(layers)
        self.channels = channels

    def encode(self, x, detach=True):
        """Forward pass. detach=True cuts the gradient at every boundary,
        which is what makes each layer local."""
        outs = []
        h = x
        for layer in self.layers:
            h = layer(h.detach() if detach else h)
            outs.append(h)
        return outs

    def losses(self, x_pos, x_neg, x_ref):
        """One loss per layer. Returns a list, to be backwarded separately.

        x_pos and x_ref are two augmented views of the same image.
        x_neg is a different image.
        """
        h_pos = self.encode(x_pos)
        h_neg = self.encode(x_neg)
        with torch.no_grad():
            h_ref = self.encode(x_ref)

        out = []
        for i, layer in enumerate(self.layers):
            if self.dfb:
                c = self.layers[-1].represent(h_ref[-1])
            else:
                c = layer.represent(h_ref[i])
            out.append(layer.local_loss(h_pos[i], h_neg[i], c.detach()))
        return out

    def features(self, x):
        """Concatenated pooled representations from every layer, which is
        what the paper feeds to the linear probe."""
        with torch.no_grad():
            outs = self.encode(x)
        return torch.cat([l.represent(h) for l, h in zip(self.layers, outs)],
                         dim=1)

    @property
    def feature_dim(self):
        return sum(self.channels)


def peak_memory_note():
    return """
The memory claim in the paper (Table 3): peak VRAM does not scale with
depth, because each layer's gradient can be applied immediately after its
own forward step and the activations then discarded. Their measured figures
were 11.10 GB for backprop vs 4.60 GB for CLAPP++ on ImageNet.

This implementation uses detach() rather than that ideal loop, which is
mathematically equivalent but does NOT realise the memory saving. The paper
does the same thing (their Algorithm 3). Worth knowing before claiming any
memory result of your own.
"""