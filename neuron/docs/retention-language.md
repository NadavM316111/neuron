# Retention on a contradictory language stream

A model learning continuously from a stream that later contradicts itself
loses what it learned first. Replaying stored runs of earlier material
prevents almost all of that loss. This note reports the effect on language,
with the two obvious alternative explanations tested and ruled out.

Qwen2.5-1.5B with LoRA, five arms, two seeds, on a laptop.

---

## The result

Phase-1 probe loss after the stream has moved on. Lower is better in both
columns. `final` is loss at the end of the run; `rise` is how far it climbed
from the arm's own best point.

| arm | updates | final | rise from best |
|---|---|---|---|
| retention, ordered | 226 | **2.11** | **-0.01** |
| unprotected, ordered | 600 | 2.70 | +1.73 |
| matched updates, ordered | 250 | 3.42 | +2.54 |
| retention, shuffled | 197 | — | +0.15 |
| unprotected, shuffled | 600 | — | +0.00 |

Both metrics rank the arms the same way, and every arm agrees across both
seeds.

Two comparisons carry the claim.

**Retention against matched updates.** The protected arm makes a third as
many updates as the unprotected one, because its gate rejects roughly half
the stream. Fewer updates means fewer chances to overwrite, so "rehearsal
protects" and "updating less protects" are not separable from those two arms
alone, and the second explanation is simpler. The matched arm removes it: no
layer, no rehearsal, no guard, but updating on only every Nth item so its
count lands on the protected arm's. It forgot +2.54, more than doing nothing
at all. Thinning updates does not protect. Rehearsal does.

**Ordered against shuffled.** The unprotected arm on the ordered stream
forgets +1.73. The same arm, the same 600 updates, the same sentences, with
the order destroyed, forgets +0.00 across both seeds and three checkpoints
where it could have risen. Forgetting here is caused by the order of the
stream, not by its content or by the amount of learning.

---

## Method

**Stream.** An invented rule, stated many ways, then reversed, then filler.

- phase 1: an amber beacon means the northern gate is *opened* (200 items)
- phase 2: an amber beacon means the northern gate is *sealed* (200 items)
- phase 3: unrelated material, so phase 2 also has time to drift (200 items)

Invented vocabulary throughout, so the pretrained model has no prior to fall
back on and anything measured came from this stream. Sentences are generated
combinatorially from templates and word lists, deduplicated, then split into
training material and held-out probes, so a probe cannot appear in training.

**Measurement.** Mean per-token loss on 25 held-out probes per phase,
measured before any learning and after each phase. Loss rather than
generation: an earlier result in this project established that teaching facts
by gradient update converts honest refusals into confident fabrications,
which makes generation a poor instrument for this question.

**Arms.**

- `retention` — the full stability layer: surprise gate, sequence replay of
  stored runs, canary rollback guard
- `unprotected` — no layer, one update per item
- `matched` — no layer, updating every Nth item, N derived from the
  retention arm's actual update count in the same seed
- the same, on a shuffled stream, in three equal blocks

**Setup.** Qwen2.5-1.5B-Instruct, LoRA rank 16, alpha 32, attention-only
targets (`q_proj`, `v_proj`). AdamW at 1e-3 with 0.995 decay. One gradient
step per accepted item. Gate: rolling window 200, top fraction 0.50, warmup
30, loss floor 0.35. Rehearsal: every 6 items, 2 units of 8 consecutive
sentences each, replayed in order, oldest-first policy. Guard: canary of 5
ordinary English sentences, checked every 75 items, rollback at 0.40. MPS on
a MacBook Air. About 6 minutes per protected arm, 2 minutes per unprotected
one.

---

## What retention costs

It holds a worse version of the rule. Phase-1 loss bottoms at 2.86 and 2.02
for the protected arm against 1.26 and 0.68 unprotected. The protected model
never learns phase 1 as well, it simply does not lose what it learned. That
is a trade, not a free win, and it should be stated first rather than found
by a reader.

The path is also messier than the endpoint suggests. In seed 0 the protected
arm went 2.86, then 4.90 during phase 2, then back to 2.72. Five rollbacks
fired; the guard discarded learning and rehearsal rebuilt it. It ends in the
right place by a route the summary number hides.

---

## How much rehearsal it needs

Rehearsal dominates the cost. The protected arm makes about 6.3 replays per
update and spends most of its time there. That ratio was chosen early and had
never been measured against anything, so a sweep varied it three ways on the
same benchmark: how many units per consolidation, how often consolidation
fires, and how long each replayed run is.

| config | updates | replays | final | rise from best |
|---|---|---|---|---|
| unprotected | 600 | 0 | 2.33 | +1.69 |
| every 6 items, 2 units (default) | 249 | 1576 | 1.71 | +0.16 |
| every 6 items, 1 unit | 214 | 792 | 2.12 | +0.31 |
| every 12 items, 2 units | 226 | 792 | 1.83 | +0.20 |
| **every 24 items, 2 units** | 190 | **400** | **1.71** | **+0.01** |
| every 48 items, 2 units | 134 | 192 | 1.85 | +0.05 |
| every 6 items, runs of 4 | 222 | 796 | 2.19 | +0.26 |

**A quarter of the replay work matches the default exactly.** 400 replays
against 1576, identical final loss, better rise. An eighth still protects,
at a small cost. The default was spending roughly four times more than the
protection required on this stream.

Two secondary findings. At the same replay count, replaying fewer units more
often is worse than replaying the same units less often (2.12 against 1.83 at
792 replays either way). And halving the replayed run length from 8 to 4 was
worse than either (2.19), which is consistent with the earlier finding that
sequence replay matters because runs are contiguous.

The shipped default has been changed from every 6 items to every 24.

**Cost is measured in replays, not seconds.** Wall clock on the machine used
here was unusable: the same configuration took 373s in one run and 1035s in
another for identical work, and one arm took 2844s against 254s for the same
work on the other seed. That is thermal throttling on a laptop after a day of
runs, not a property of any configuration. Replay count is hardware
independent and is what the table reports.

---

## Limitations

**The rise metric penalises deep learning.** An arm that learns phase 1 to
0.87 has more room to fall than one that reaches only 2.12. Some of the
protected arm's low `rise` is that it never got as low to begin with. This is
why the table reports final loss as well: on that metric, which has no such
bias, the protected arm still wins on both seeds. Neither metric alone would
be enough.

**The matched arm forgot more than the unprotected one.** +2.54 against
+1.73. Partly the same effect as above, since it learned phase 1 deepest of
all three (0.87). The conclusion that survives is the narrow one: thinning
updates does not confer protection. Claiming it actively hurts would need a
design that separates depth of learning from rate of update, and this one
does not.

**Synthetic stream.** The rule is invented and stated in eight templates.
Real contradictory streams are messier and the phases are not clean.

**Two seeds, one model size, one adapter configuration.** Attention-only
LoRA at rank 16 on a 1.5B. Nothing here says what happens at 7B or with full
fine-tuning.

**The rehearsal sweep changes update counts too.** Configurations with less
replay also made fewer updates, from 249 down to 134, because rehearsal
changes the model and therefore changes what the surprise gate fires on. The
recommended setting is close to the default on this axis (190 against 249);
the most aggressive one is not, and its result should be read with that in
view.

**Loss, not task performance.** Probe loss is a proxy. It does not establish
that the model would answer a question about phase 1 correctly.

---

## Related result: where facts belong

A separate check the same day, on the same system: 13,681 Wikipedia sentences
in a retrieval store, 11 hand-written questions, three prompt conditions.

| condition | answered |
|---|---|
| with retrieved notes, strict prompt | 10/11 |
| with retrieved notes, open prompt | 10/11 |
| no notes (closed book) | 0/11 |

The closed-book arm did not decline; it fabricated. Jurchen for Okhotsk,
Neil Armstrong for Borman, *Il Trovatore* for *Europa riconosciuta*, $40
billion for $5 billion, and an invented person running the Aldine press. The
store is 2023 Wikipedia and the model was pretrained on it, so this control
existed to check whether retrieval contributed anything at all. It
contributes essentially all of it.

Caveat: 11 questions, written after seeing the source sentences. The gap is
wide enough to survive that, but independently written probes would be
stronger.

Together with the drift result, this supports a split rather than a single
mechanism. Facts go to a store, where they are retrieved verbatim and never
invented. Patterns go to the weights, where retention keeps them from being
overwritten by a stream that changes its mind.

---

## What this does not show

Learning without a training run. Every number here sits on a pretrained
model, and the local-learning half of this project remains open.

That continual learning is cheap. Cutting the replay budget fourfold helps
and does not change the shape of the cost: the protected arm still replays
about twice per update, forever, for as long as it runs. What the sweep
establishes is that the previous figure was inflated, not that the mechanism
is inexpensive.

---

## Reproducing

```
./run.sh experiments/language/drift.py --items 200 --probes 25 --seeds 2
./run.sh experiments/language/rehearsal.py --seeds 2
./run.sh experiments/language/probes.py score
```

Both scripts assert their own preconditions and report when a number could
not have moved. Raw output is in `results/drift.json`,
`results/rehearsal.json` and `results/probes_result.json`.
