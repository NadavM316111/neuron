# NEURON: continual learning that does not break

**Nadav Minkowitz, 1 September 2026**

*Replaces the 31 August version. Two things changed materially: the store
now works at forty thousand real sentences, and the retention mechanism was
found NOT to help on ordinary real text. The second contradicts what the
previous version claimed and is corrected here.*

---

## The result

**A working split between weights and store.** Facts go in a store and are
retrieved exactly. Skills and patterns go in the weights. Running together on
a 7-billion-parameter model over forty thousand sentences of real Wikipedia,
the system answers from what it read and refuses what it never saw.

**A retention mechanism with a precisely known scope.** It preserves material
a stream has moved past — established causally on real weather across five
climates and on a 7B model across three seeds — and it does nothing, or
slight harm, when the stream is not genuinely non-stationary.

**A merging mechanism** that lets instances inherit from each other,
compounding across generations, solving the cold start with a twelve-fold
head start.

Ten days. About thirty dollars of rented GPU and a laptop.

---

## The most replicated finding

**Gradient updates make a language model fabricate. A store does not.**

| run | scale | frozen fabricates | system fabricates |
|---|---|---|---|
| retrieval test | 6 facts | 5/5 | 0/5 |
| whole-system test | 5 facts | 3/3 | 0/3 |
| real text | 6,000 sentences | 4/4 | 0/4 |
| large corpus | 40,000 sentences | 5/5 | 0/5 |

Four runs, three orders of magnitude apart, same outcome every time. Asked
about things that do not exist, the frozen model invents; the system says it
does not know.

**The original negative result.** Teaching facts by gradient update converts
honest refusals into confident fabrications. Taught "nineteen string
quartets", answered "forty four". Taught "eleven thousand moths", answered
"over 10 million". Different arms invented *different* wrong answers to the
same question, which shows the model generates domain-shaped text rather than
recalling anything.

**And the gradient arm is indistinguishable from frozen** on every measure,
after eight repetitions of each fact. Weight updates install nothing
answerable.

---

## The store, at scale

**Word overlap fails.** With 58 sentences it retrieved perfectly. With 40,000
it found almost nothing — 8.3% on twelve probes, which is one correct answer.
Too many sentences share common words.

**Embeddings fix it.** Sentences encoded as vectors, nearest neighbour to the
question.

| | early | late |
|---|---|---|
| frozen, no store | 0.0% | 0.0% |
| store, word overlap | 6.7% | 13.3% |
| **store, embeddings** | **26.7%** | **40.0%** |

**Retrieval of the source sentence improved by 53 points.** That measure —
does the store return the right sentence at all — separates the retriever
from the model, and it is the clean number.

**The answer scores are a floor.** The probe generator picks the first number
in a sentence, but sentences often contain several. Asked what number is
given for Aristotle, expecting 348, the model correctly answered "384-322
BC" and was scored wrong.

**Frozen at 0.0% is the control working.** Probes were filtered against the
frozen model first, so anything the store got right came from the stream.

---

## Retention: what it does and does not do

### Where it works

A stream arrives one moment at a time and is never revisited. A **gate**
decides what is worth learning from. **Rehearsal** stores what the gate
rejects and replays it in order. A **canary guard** rolls back if competence
on fixed inputs degrades.

| setting | effect |
|---|---|
| small world, contradictory rule phases | large win |
| real weather, chronological | large win, 5 of 5 climates |
| 7B model, ordered invented phases | large win, 3 of 3 seeds |
| stationary grid world | no effect |
| noisy grid world | slightly negative |
| Fashion-MNIST | inside the noise |
| **real Wikipedia, three topic domains** | **harmful** |
| **the same weather, shuffled** | **slightly negative** |

### Proving it causally

**A first control failed.** Singapore was chosen as a no-seasons control. The
advantage appeared anyway, because Singapore's weather is seasonal even
though its temperature is not.

**The control that worked: shuffle the stream.** Same data, only the order
destroyed.

| | ordered | shuffled | collapse |
|---|---|---|---|
| Chicago | +11.51 | -1.56 | +13.07 |
| Phoenix | +17.38 | -2.68 | +20.06 |
| 7B model | +1.242 | -0.103 | +1.345 |

**The advantage does not shrink. It goes to zero.**

### Real weather

Four years of hourly readings, no clock and no calendar. Predict what
temperature does three hours ahead.

| arm | accuracy |
|---|---|
| memoryless | 48.9% |
| online, unprotected | 47.7% |
| **extrapolation baseline** | **53.6%** |
| **online with retention** | **59.2%** |
| offline, five shuffled passes | 59.3% |

**A single pass over a stream never revisited matched conventional
training.** Unprotected loses 26 points on the season nine months stale.
Protected loses nothing.

### Where it does not work, and this is new

**Real text, three topic domains.** Biology, then law, then music — 900
sentences each from Wikipedia, read once in order. Measured as loss on
held-out sentences, so nothing is generated and nothing is graded.

| change in loss, start to end | biology | law | music |
|---|---|---|---|
| unprotected | **-0.186** | -0.237 | -0.326 |
| with retention | **+0.240** | -0.085 | -0.150 |

**Unprotected learned all three and forgot none.** Biology stayed improved
after two full phases away.

**So there was nothing to protect against, and the layer hurt.** It ended
worse than frozen on the first domain and learned less on every domain.

**Why.** Wikipedia prose about law and about biology are both encyclopedic
English. Different topics, not contradictory rules. The stream changes
subject without changing distribution, so it is not non-stationary in the
way that matters.

**This confirms the scope condition rather than breaking it** — the layer was
already known to cost a little when a stream is stationary. But it narrows
where the mechanism is useful, and the previous version of this document
overstated it.

### And it has a horizon

Over 480,000 steps — twenty rotations through three contradictory rule sets
— the protected arm **reversed**: 43.41% to 39.64%, while unprotected went
44.16% to 46.76%.

**Best explanation:** the buffer fills with material from every rule set at
once, so replaying it means training on a mixture that contradicts itself.
The unprotected arm simply tracks the current phase and settles into a
compromise.

**A follow-up testing forgetting policies was inconclusive** — the reversal
did not reproduce at half the length, and the metric rewards a model that is
mediocre at everything, which is what unprotected learning converges to. The
horizon is real but its cause is not established.

---

## Scale

| width | parameters | unprotected | protected | difference |
|---|---|---|---|---|
| 64 | 49,239 | 43.1% | 48.7% | +5.6 |
| 256 | 491,799 | 54.6% | 59.4% | +4.8 |
| 512 | 1,770,007 | 53.7% | 57.3% | +3.6 |

On the hardest rules the advantage grows with size, reaching +14.3.
Protection costs about 4.5x and that multiplier does not shrink.

**At 7B, a second result appeared by accident.** In a shuffled control,
unprotected damage varied by 3.143 across seeds — one run diverged badly. The
protected arm varied by 0.183 with nothing to retain. **The layer retains,
and separately it prevents divergence.**

---

## Inheritance

The cold start is the business risk no technical progress removes: software
that arrives blank and is useless for three weeks gets deleted in one.

**Merging works.** Floor 54.91%, joint-training ceiling 72.16%, plain weight
averaging **78.69%** — it beats joint training. The midpoint of the line
between two networks beats both ends, because both descend from the same
initialisation, which holds automatically for a shipped product.

**The cold start, measured.** Sixteen instances, each with its own layout and
one of three contradictory rule sets:

| steps | from blank | from merge |
|---|---|---|
| 500 | 4.50% | **69.22%** |
| 6,000 | 78.27% | 84.87% |

**A blank instance needs 6,000 steps to reach what a seeded one has after
500.** The advantage shrinks as the newcomer learns — a head start, not a
constraint.

**The apparent paradox:** merging sixteen contradictory lives produces a poor
arbiter (29.89%) but an excellent starting point. Averaging preserves what
the lives agreed on and cancels what they disagreed on.

**It compounds.** Over six generations: merge quality 4.25% to 63.50%,
newcomer speed 36.89% to 75.40%, best individual 68.49% to 100%. A
never-merging control that lived identical lives went 36.89% to **9.35%** —
without inheritance the population degrades.

**Upgrades.** Merging needs a shared initialisation, so shipping version 2
would strand every user. Git Re-Basin recovers half: 32.97% naive, **56.93%**
after permuting units into alignment.

**What failed.** Distillation converges the student to the teacher's ceiling
(65.99% against 67.43%) while a merged student reaches 93.65% — merging gives
a starting point you can surpass, distillation gives a target you converge
to. REPAIR costs 12.7 points on recurrent networks. Cross-architecture
merging fails three ways.

---

## Speed

At batch size one on a laptop, **the optimiser is 67% of the cost** — 288
microseconds against a 16 microsecond forward pass. Adam's cost scales with
parameter count, not batch size, so it costs the same at batch 256.

**The field's measurements do not transfer.** Published comparisons measure
time per batched iteration on a datacenter GPU at batch 256, where arithmetic
dominates. Nobody measures at batch one on a laptop, where machinery
dominates: 69.7% against 37.6%.

**Accumulate over 8 and step once:** 2.33x faster and 6 points more accurate,
because a single sample's gradient is noisy. **Scope condition:** on discrete
symbolic data the same change costs 12 to 16 points, because averaging blurs
distinct lessons rather than cancelling noise.

**Layer skipping beats backpropagation on a single device.** Backprop must
traverse every layer to update any layer; a local rule can skip.

| | accuracy | vs backprop | updates |
|---|---|---|---|
| backprop | 99.04% | 1.00x | 100% |
| local, no skipping | 100.00% | 0.64x | 100% |
| **local, skip 70%** | **98.82%** | **1.63x** | **25%** |

Confirmed at depths 4, 8 and 12: 1.77x, 1.80x, 1.78x. **The local rule itself
is slower** — the entire win comes from skipping.

**A hypothesis tested and refuted.** Greedy local training collapses
information at early layers, so skipping might cure the accuracy gap. At six
blocks it closed 43%, and skipping everywhere closed none — a differential
supporting the mechanism. **The depth test killed it:** 23% at depth 4, 21%
at depth 8, **-70% at depth 12**. A fix for a problem that worsens with depth
should help more with depth. It helps less, then harms.

---

## Uncertainty

Noise does not hurt perception — it hurts memory. A misread cell that gets
stored is a persistent wrong belief.

**Seven attempts. Four recovered nothing:** prediction entropy as an input
(it measures task difficulty, not perceptual ambiguity), input ambiguity as a
feature, ambiguity gating the state update, particles diversified by random
jitter.

**The fifth worked.** Each particle samples its own *interpretation* of an
ambiguous reading, so ambiguous cells split the particles into genuine
competing hypotheses.

| | simple forks | conjunctive |
|---|---|---|
| single particle | 56.2% | 43.8% |
| jitter, 8 particles | 55.5% | 46.6% |
| **interpretation, 8** | **63.8%** | **50.5%** |
| oracle | 72.4% | 72.6% |

The decisive comparison — interpretation against jitter at identical particle
count — is +8.24.

**Two further attempts showed the mechanism is simpler than expected.**
Likelihood weighting and threshold resampling both worked as implemented and
both changed nothing, because the particles are already correctly-drawn
samples from the posterior. **Averaging uniformly is right; the filtering
machinery is decoration.**

---

## What is not established

**Frontier scale.** Untestable on any affordable budget. Holds to 7B.

**Learning a person.** Rung 5 failed five ways. Shell history was one day's
work. File timestamps recorded package managers. Browser history came six
points short. App logging collapsed to the majority class. **The diagnosis:
the logger recorded what happened but never why.** A logger capturing dwell
and idle time has been running since 29 August.

**Retention on real text.** The mechanism helps on contradictory streams and
not on ordinary prose. Whether any realistic stream is contradictory enough
to need it is open.

**The no-training-run half of the vision.** Local rules save memory rather
than time, and the accuracy gap persists — 11.5 points after eight epochs. A
field-wide open problem since 2019.

---

## Method notes

Eleven controls failed in ways that changed conclusions. These are the most
portable thing here.

**Check the test can fail before running it.** Interesting events at 0.4% of
a stream cannot be measured, because ignoring them minimises the loss.

**Check the task is learnable before comparing methods on it.** Two
experiments were void because no arm could learn the task and every arm
scored the majority-class rate.

**Hiding a variable is not enough to test memory.** If its value is
predictable from base rates, a memoryless model matches a remembering one.

**A moving baseline produces artifact curves.** Advantage swung from +2.7 to
+32.8 purely because the guess rate moved.

**A control that has not converged is not a control.**

**Score where the task disagrees with itself.** Whole-stream accuracy hid a
28.7-point gap behind a 3.1-point one.

**Never test a retention mechanism on a stationary stream.** Made this
mistake three times.

**Identical numbers across different architectures mean collapse**, not
agreement.

**A transformation that should preserve behaviour must be checked.** A
permutation implementation scored 11.89% — worse than doing nothing — and
looked like a clean negative result.

**A ceiling below its own floor means the experiment is void.**

**Sampling intervals must divide the period being measured**, or aliasing
dwarfs the signal.

**Prefer measures with nothing to generate and nothing to grade.** Three
probe-based experiments in one week had generator or scorer bugs. Loss on
held-out text has neither.

---

## Where this leaves the vision

Software that arrives blank, grows on your machine, becomes uniquely yours.

| piece | state |
|---|---|
| learning without the training run | 60% — open field-wide |
| continuous learning without breaking | 75% — narrower scope than believed |
| **learning facts** | **80% — by separation, at real scale** |
| growing from nothing | 75% — rung 5 collecting |
| cheap enough for a laptop | 50% |
| handling uncertainty | 45% |
| **inheritance** | **92%** |

**About 70% overall.**

**And a caveat that no component percentage captures.** Everything is a 9×9
grid, a 25×25 grid, four years of weather, and a handful of 7B runs lasting
hours. Against the vision as described — a system living alongside someone
for years — this is perhaps 18%. Scale dominates, and it is the term that has
barely moved.

**One finding from the last day is worth holding.** On ordinary real text,
plain continual learning worked: the model learned three domains in sequence
and forgot none of them, with no protection at all. That is the "grows from a
stream" claim, on real data, without any of this machinery.

Which raises a question the rest of this document does not answer: how much
of the machinery is needed, and for which streams.
