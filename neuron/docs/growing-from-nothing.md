# Growing a language model from nothing, and what the single pass costs

A character-level network with random initial weights, shown each sentence
exactly once, in order, never revisited, learns text structure that beats an
n-gram model fitted on the entire corpus. The single pass costs 0.15 nats
against conventional multi-epoch training, two thirds of which is the
ordering of the stream rather than the number of passes. Half of that is
recoverable, at a quarter of the optimiser steps, without giving up the
ordering.

Two seeds throughout. A 235,000-parameter GRU on a laptop.

---

## 1. It learns, from nothing, in one pass

Mean per-character cross entropy on held-out text. The n-gram baselines are
fitted on the whole training corpus and may count it as often as they like.
The network sees each sentence once, in order, and never again.

| | nats | bits/char |
|---|---|---|
| uniform floor | 3.7377 | 5.392 |
| untrained network | 3.7399 | 5.396 |
| unigram, full corpus | 3.0166 | 4.352 |
| bigram, full corpus | 2.4801 | 3.578 |
| trigram, full corpus | 2.1326 | 3.077 |
| **grown from nothing, one pass** | **1.7543** | **2.531** |

The untrained network sits within 0.002 nats of the uniform floor, which is
the check that the measurement was sound before the experiment started. The
network passed trigram within the first 1,000 sentences and the gap widened
from there rather than closing.

The alphabet is fixed in advance rather than derived from the corpus, so the
online learner is not given information about text it has not seen. Held-out
sentences are taken from the end of the file and never trained on.

---

## 2. What the single pass costs

The interesting claim is not that it learns. It is what the premise costs
against training the same network conventionally.

| | nats | updates |
|---|---|---|
| online, one pass, in order | 1.7543 | 15,000 |
| same data, shuffled once | 1.6515 | 15,000 |
| three shuffled epochs | 1.6012 | 45,000 |

**The order of the stream costs 0.103 nats. The extra two epochs buy only
0.050 on top.** So two thirds of the single-pass penalty is not about seeing
data once. It is about consecutive samples being correlated, and shuffling
once recovers most of it without a second look at any sentence.

That reframes the problem. The thing to attack is correlation between
neighbours, not the number of passes.

---

## 3. Half the penalty comes back, at a quarter of the steps

Two ways to decorrelate without revisiting anything. Accumulating the
gradient over N sentences before stepping keeps the premise fully intact:
every sentence is still seen once, in order, and only the applying is
batched. A shuffle buffer holds the last 64 sentences and draws the next
update from them, which relaxes the ordering and is reported separately for
that reason.

| arm | nats | optimiser steps | order gap recovered |
|---|---|---|---|
| online | 1.7543 | 15,000 | 0% |
| **accumulate 4** | **1.7041** | **3,750** | **49%** |
| accumulate 8 | 1.7149 | 1,875 | 39% |
| accumulate 16 | 1.7587 | 938 | -4% |
| shuffle buffer of 64 | 1.7304 | 15,000 | 24% |
| fully shuffled (ceiling) | 1.6526 | 15,000 | 100% |

Accumulating over four recovers half the ordering penalty while making four
times fewer optimiser steps. Since Adam's cost is per step and independent of
how much was accumulated into it, that is a saving on the dominant cost of
single-sample online learning, taken alongside an accuracy gain rather than
against one.

The arm that keeps the premise beat the arm that relaxes it. The shuffle
buffer recovered 24% and cost strict ordering; accumulation recovered 49% and
cost nothing.

---

## 4. The window has an optimum, and a previous conclusion was wrong

Accumulation at 16 is worse than not accumulating at all. There is a peak,
and it is small.

This matters beyond this experiment. An earlier result in this project tested
accumulation on a grid world with 21 outcome classes and found it cost 12.7
to 16.6 points, and recorded the explanation that smooth physical streams can
average away noise while discrete symbolic streams cannot, because each event
is a distinct lesson. That went into the repo as a scope condition on the
mechanism.

Every arm in that sweep used a window of eight. Sweeping it properly, with
network width held fixed:

| window | simple forks | conjunctive forks | vs window 1 | speedup |
|---|---|---|---|---|
| 1 | 68.1% | 58.0% | — | 1.00x |
| **2** | **68.1%** | **57.2%** | **+0.1 / -0.8** | **1.49x** |
| 4 | 67.1% | 51.2% | -1.0 / -6.8 | 2.01x |
| 8 | 55.3% | 41.4% | -12.7 / -16.6 | 2.43x |
| 16 | 34.8% | 27.5% | -33.2 / -30.6 | 2.70x |

The degradation is monotone, and window 8 sits three quarters of the way down
it. **Window 2 holds accuracy on the grid world at 1.49x.** The original
conclusion was a single point on a curve, read as a property of the data.

The corrected rule is about neither smoothness nor discreteness:

> The usable accumulation window is set by how much consecutive samples
> repeat one another. It was 8 on hourly weather, 4 on character-level text,
> and 2 on a grid world of independent events. Past that window, averaging
> destroys distinct lessons and accuracy falls monotonically.

One variable explains all three domains, where the previous rule explained
two and got the third wrong. Note what the corrected rule does not say:
accumulation is not free in general, and a window chosen without measuring
can be much worse than not accumulating at all.

---

## Limitations

**Scale.** A 235,000-parameter character-level GRU on one domain. Nothing
here says the mechanisms survive to a useful language model, and that gap is
many orders of magnitude, not a few more experiments.

**The online curve is noisy.** It moved 1.7884, 1.8280, 1.8067 across
consecutive checkpoints, so final values carry roughly 0.02 nats of noise.
The differences reported above are larger than that, but not by much in the
case of accumulate-8 against accumulate-4.

**Wikipedia has no meaningful order.** The ordering effect measured here is
local topical correlation within articles, not a designed sequence. A stream
with real structure over time could behave differently in either direction.

**Two seeds, one architecture, one optimiser.** AdamW at 2e-3 throughout. The
accumulation result depends on Adam's cost being per-step, so it would not
transfer unchanged to a cheaper optimiser.

**The n-gram baselines are add-k smoothed at k=0.1** and not tuned. A better
smoothing scheme would narrow the gap the network wins by, though not close
it.

---

## What this does not show

That a useful language model can be built without a training run. This is the
first rung of that claim, not the claim.

That continual learning is cheap. Cutting optimiser steps fourfold helps and
does not change the shape: single-sample online learning remains far more
expensive per sample than batched training, which is the honest position and
the one still to be attacked.

---

## Reproducing

```
python fetch_wiki.py 20000
./run.sh experiments/language/scratch.py --train 15000 --seeds 2 --data <abs path>/data/wiki.txt
./run.sh experiments/language/decorrelate.py --train 15000 --seeds 2 --data <abs path>/data/wiki.txt
./run.sh experiments/speed/windowsweep.py
```

Raw output in `results/scratch.json`, `results/decorrelate.json` and
`results/windowsweep.json`.

The README's line about speed in the single-sample regime, and the scope
condition recorded alongside the August accumulation result, both need
updating to the corrected rule in section 4.
