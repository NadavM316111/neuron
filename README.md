# NEURON

Continual learning that does not break.

A retention mechanism for models learning from a live stream, measured
across a grid world, real weather from five climates, and a 7-billion
parameter language model. The central claim is established causally: remove
the cause and the effect vanishes.

Six days of work, about twenty dollars of rented GPU, one laptop.

---

## The result in one table

Does the retention layer help? It depends on one thing, and only one.

| setting | effect |
|---|---|
| small world, contradictory rule phases | large win |
| real weather, chronological | large win, 5 of 5 climates |
| 7B language model, ordered phases | large win, 3 of 3 seeds |
| stationary grid world | no effect |
| noisy grid world | slightly negative |
| Fashion-MNIST images | inside the noise |
| **the same weather data, shuffled** | **slightly negative** |

**It helps exactly when the stream is non-stationary, and costs a little
when it is not.**

That last row is the proof. Same data, same amount, only the order
destroyed. The advantage does not shrink — it goes to zero.

| shuffle control | ordered | shuffled | collapse |
|---|---|---|---|
| Chicago | +11.51 | -1.56 | +13.07 |
| Singapore | +5.21 | -1.12 | +6.33 |
| Phoenix | +17.38 | -2.68 | +20.06 |
| 7B language model | +1.242 | -0.103 | +1.345 |

---

## Running it

Everything is plain PyTorch on CPU. No GPU needed except for the 7B
experiments, which expect a rented machine.

```bash
pip install torch numpy
./run.sh experiments/uncertainty/pf_check.py     # 10 seconds, checks setup
./run.sh experiments/inheritance/merge.py        # 20 minutes
./run.sh experiments/grid/grow.py                # 15 minutes
```
`reproduce.py` is the whole argument in a few minutes: the layer helps on an
ordered stream and stops helping when the order is destroyed. Everything
else in the repo supports or qualifies that.
`run.sh` puts `core/` and `worlds/` on the path and runs from `results/`,
so outputs land beside the existing ones. Experiments use flat imports
(`from world import ...`), which is why the runner exists rather than a
package.

Some experiments download data on first run and cache it. The weather
experiments fetch from Open-Meteo's free archive; no key needed.

---

## Layout

```
core/          the library
  stability.py     the retention layer: gate, veto, rehearsal, guard,
                   sequence replay, replay policies
  llm_backend.py   transformer + LoRA, for the 7B experiments
  fake_backend.py  a dict-based backend proving the layer needs no ML

worlds/        environments
  world.py         9x9 grid, three contradictory rule sets
  bigworld.py      25x25, 21 outcomes, 6 hidden variables
  noisy.py         the same seen through imperfect senses
  sensor.py        real hourly weather

experiments/
  grid/            growing from nothing, rungs 1-5
  scale/           100k to 7B parameters, and the credit-assignment bug
  speed/           where the time actually goes
  uncertainty/     seven attempts at holding a belief loosely
  inheritance/     merging, generations, cold start
  language/        the 2026 fact-injection work
  local/           reproducing CLAPP, falsifying SAL

results/       every .json and .log this produced
docs/          the write-up and the competence curve
archive/       superseded scripts, kept for provenance
```

---

## What is established

**Retention through rehearsal works, and works because it is retention.**
Proven by removing the cause in two independent settings.

**It scales.** Holds across a 36-fold network size sweep and at 7 billion
parameters, 3 of 3 seeds. Protection costs about 4.5x and that multiplier
does not shrink with size.

**It prevents divergence, separately.** In a shuffled control the
unprotected arm's damage varied by 3.143 across seeds — one run blew up.
The protected arm varied by 0.183 with nothing to retain. Two jobs, not one.

**A network can grow from nothing.** Random weights, single-pass stream,
never revisited. It beats the standard baseline for real weather across
five climates and matches conventional multi-pass training.

**Inheritance works and compounds.** A newcomer seeded from a 16-way merge
needs 500 steps to reach what a blank one takes 6,000 to reach. Across six
generations, merged populations improve while a never-merging control
degrades from 36.89% to 9.35%.

**Speed in the single-sample regime is solved.** 2.4 to 4.3x, by not
stepping the optimiser every sample.

---

## What is not

**Frontier scale.** Untestable on any affordable budget.

**Learning a person.** Rung 5 failed five ways. Shell history was one day's
work. File timestamps turned out to record package managers, not a person.
Browser history came six points short. Deliberate app logging collapsed to
the majority class, and reformulating it did not beat "you will still be
where you are." The honest diagnosis is that collectable personal data does
not contain the causes of behaviour.

**The no-training-run half of the vision.** Local learning rules reproduce
and match backpropagation on small problems, but save memory rather than
time. That is a field-wide open problem.

**Uncertainty about stored beliefs.** Seven attempts. The fifth worked and
recovers about half the gap: when a reading is ambiguous, sample several
interpretations of it and carry them forward as separate hypotheses. The
weighting machinery on top of that turned out to be decoration.

**Merging across an architecture change.** Three approaches failed.
Same-architecture upgrades carry about half their learning; architecture
changes do not carry usefully.

---

## What is disproved

**Teaching a language model facts from a real text stream by gradient
updates converts honest refusals into confident fabrications.** Every probe
tested, both arms. Before: four honest "I don't know". After: seven
fabrications and one correct fact.

Different arms invented *different* wrong answers to the same question,
which shows the model generates domain-shaped text rather than recalling
anything.

For factual content, retrieval remains the working approach. Continual
learning by weight update looks right for skills, patterns and structure,
and wrong for facts.

---

## Method notes

Eight controls failed in ways that changed conclusions. The patterns are
consistent enough to be rules, and they are the most portable thing here.

**Check the test can fail before running it.** A world where the interesting
events are 0.4% of the stream cannot measure whether they are learned,
because ignoring them minimises the loss.

**Hiding a variable is not enough to test memory.** If its value is
predictable from base rates, a memoryless model matches a remembering one.

**A moving baseline produces artifact curves.** Advantage swung from +2.7 to
+32.8 across a sweep purely because the guess rate moved from 63% to 96%.

**A control that has not converged is not a control.** A fix appeared to
close 77% of a gap at half training length and 23% at full length.

**Score where the task disagrees with itself.** Whole-stream accuracy hid a
28.7-point gap behind a 3.1-point one.

**Never test a retention mechanism on a stationary stream.** Made this
mistake three times. If nothing drifts away there is nothing to retain.

**Identical numbers across different architectures mean collapse**, not
agreement.

**A transformation that should preserve behaviour must be checked.**
Permuting a network's units relabels them without changing the function. A
first implementation scored 11.89% — worse than doing nothing — and looked
like a clean negative result. The only reason it was caught is a check that
could only fail if the implementation were wrong.

---

## The bug worth knowing about

Every experiment detached the recurrent hidden state after every step, so
gradients reached back exactly one step. The network was told it predicted
wrongly and could only adjust the weights that processed *that moment*.
Nothing ever pushed it to have stored the relevant fact sixty steps earlier.

**Whatever memory it had was incidental.** The recurrent cell happens to
carry information forward and the one-step gradient happened to exploit it.
It was never trained to remember on purpose.

Fixing it closed 23% of the memory gap. A first version of the fix appeared
to close 77%, but was measured against an undertrained control.

---

## Valence, 20 Sep 2026

`core/valence.py`, `experiments/grid/valence_replay.py`

A persistent value per stored unit. Born from the adjusted surprise already
computed at intake, so it costs no extra forward pass. Updated toward the
model's current loss every time the unit is replayed. Ranks replay
selection and decides buffer eviction, so a unit that stays hard keeps
earning replay and one that has been absorbed decays and is dropped before
older material is.

Fifth arm against offline, uniform, old and surprising on the three-phase
grid world. Contested events in the earliest phase, 8 seeds:

| arm | keyed |
| --- | --- |
| offline | 68.0% |
| uniform | 56.3% |
| valence | 55.7% |
| surprising | 52.7% |
| old | 44.1% |

    valence minus surprising:   +2.9  (wins 7/8)
    valence minus uniform:      -0.6  (wins 5/8)
    valence minus offline:     -12.4  (wins 2/8)

**Persistent value beats recomputed value, by a little and consistently.**
`surprising` rescores candidates fresh every consolidation and has no
memory, so a unit that is hard once and easy later is two unrelated
observations. Accumulating across the unit's life is worth about 3 points.
It is not worth anything against random selection.

**The variance claim did not survive.** At 3 seeds valence spread 49 to 59
against uniform's 45 to 70 and looked far more consistent, which read as
the real finding. At 8 seeds both spread 37 to 70. A small-sample artifact
that looked exactly like a result.

**The first configuration was degenerate and the scores from it mean
nothing.** `valence_floor=0.05` clamped 543 of roughly 600 units to the
floor, so ranking was near-random across 90% of the pool. Fixed to 1e-4
with decay 0.9. The internals block the script prints exists to catch this,
and did.

**Replay selection is not the bottleneck.** Four policies now tested and
none closes the 12 to 24 point gap to offline on the first phase. The
remaining candidates are replay volume, what the gate stores in the first
place, and whether the gap is closeable at all. Offline sees every phase
interleaved in every epoch and never has to retain anything, so it may be
a ceiling rather than a target.

---

## Wanting, 20 Sep 2026

`experiments/grid/drive.py`, `experiments/grid/drive2.py`

Every policy in `acting.py` is instantaneous. `greedy` reads the current
predicted outcome, `curious` reads the current predicted entropy. Nothing
carries between steps, so the agent has no memory of having been wrong.

A drive is a persistent scalar per situation, built from the surprise the
agent actually experienced and decaying as outcomes become predictable. The
policy prefers what it has been wrong about rather than what it predicts it
is unsure about.

**The hypothesis: curiosity is blind exactly when it matters.** Entropy
measures how unsure the model *says* it is. A model that is confidently
wrong has low entropy, which is its state immediately after a rule
reversal. Accumulated actual error has no such blind spot.

Contested probes, 8 seeds, 4,000 steps per phase:

| arm | after p1 | p2 rule | fungus p2 | spread | repeat |
| --- | --- | --- | --- | --- | --- |
| passive | 87.8% | 84.4% | 959 | 0.83 | 46.3% |
| driven | 88.0% | 81.5% | 1499 | 0.59 | 41.5% |
| explore | 75.4% | 80.4% | 625 | 0.96 | 59.2% |
| curious | 64.1% | 73.2% | 595 | 1.01 | 73.6% |
| greedy | 54.7% | 55.2% | 141 | 2.07 | 89.8% |

    driven minus curious, phase-2 rule:  +8.2   (+3.6 at 3 seeds)
    driven minus curious, phase-1 rule: +23.9

**A drive is the first acting policy here that matches the passive random
walk.** 88.0% against 87.8% while choosing its own actions. Nothing in
`acting.py` came close to that.

**Steering weakly beat steering hard, and that inverts the obvious
reading.** `spread` is the drive's range across buckets, `repeat` the share
of consecutive steps landing on the same cell type. `driven` has the
LOWEST spread of any acting arm and the LOWEST repeat rate, below even the
random walk, and wins. `greedy` has the highest of both and learns nothing.
The relationship holds on all eight seeds. Whatever a drive is buying, it
is not decisiveness.

### The first attempt failed twice, and both failures were informative

**No gradient to act on.** One scalar per cell type is three buckets
updated across 8,000 steps, which is a running mean. Measured values ended
at 0.293 / 0.607 / 0.631, so the policy was choosing between near-equal
numbers. Raising the decay made it flatter still, which was the clue: the
problem was resolution, not timescale. Twelve buckets keyed on
(cell, action) gave it somewhere to develop structure.

**ORDER MATTERS MORE THAN COVERAGE, and no passive experiment in this repo
could have shown it.** Argmax over a drive makes the agent chase one
bucket until it decays, so experience arrives in runs. In `drive.py` the
driven arm ate MORE than any other arm — 1,487 fungus against the passive
walk's 944 — and learned LESS, 50.3% against 91.4%. Coverage was never its
constraint. Sampling from a softmax instead of taking the argmax kept the
preference, dropped the repeat rate to 41.5%, and took the same arm from
50.3% to 88.0% with its appetite essentially unchanged.

For a single-pass online learner, WHEN things arrive matters more than how
many of them do. A random walk interleaves by construction; any policy that
steers is by construction doing the opposite, and that is the cost a drive
has to pay for.

### What this is not

A persistent scalar that steers behaviour. Calling it a want is a
functional description and nothing more. One toy world, two phases, one
drive design, and the +8.2 sits on a metric where `curious` ranges from
51.0% to 93.0% across seeds.

`drive.py`'s own verdict line claimed the drive had collapsed onto one cell
type while the table above it showed the flattest row in the run. A script
that prints a diagnosis the data contradicts is worse than one that prints
nothing. `drive2.py` gates every conclusion on the spread measurement
instead.

---

## Valence, 20 Sep 2026

`core/valence.py`, `experiments/grid/valence_replay.py`

A persistent value per stored unit, born from the adjusted surprise already
computed at intake and updated toward current loss on every replay. Ranks
replay selection and decides buffer eviction, so a unit that stays hard
keeps earning replay and one that has been absorbed decays and is dropped
before older material is.

Fifth arm against offline, uniform, old and surprising. Contested events in
the earliest phase, 8 seeds:

| arm | keyed |
| --- | --- |
| offline | 68.0% |
| uniform | 56.3% |
| valence | 55.7% |
| surprising | 52.7% |
| old | 44.1% |

    valence minus surprising:   +2.9  (wins 7/8)
    valence minus uniform:      -0.6  (wins 5/8)

**Persistent value beats recomputed value, by a little and consistently.**
`surprising` rescores fresh every consolidation, so a unit that is hard
once and easy later is two unrelated observations. Accumulating across the
unit's life is worth about 3 points. It is worth nothing against random
selection.

**The variance claim did not survive.** At 3 seeds valence spread 49 to 59
against uniform's 45 to 70 and looked far more consistent, which read as
the real finding. At 8 seeds both spread 37 to 70.

**The first configuration was degenerate and its scores mean nothing.**
`valence_floor=0.05` clamped 543 of roughly 600 units to the floor, so
ranking was near-random across 90% of the pool. Fixed to 1e-4 with decay
0.9.

---

## Replay budget, 20 Sep 2026

`experiments/grid/replay_budget.py`

Four selection policies all land 12 to 24 points behind offline, and the
spread between them is smaller than the gap. So the untested variable was
HOW MUCH rather than WHICH. Consolidation frequency swept from every 40
items to every 1, uniform policy throughout, 8 seeds:

| every | replays | keyed | trap |
| --- | --- | --- | --- |
| 40 | 9,000 | 52.8% | 94.4% |
| 20 | 17,990 | 58.0% | 91.6% |
| 10 | 35,970 | 52.7% | 97.2% |
| 5 | 71,940 | 56.3% | 92.0% |
| 2 | 179,850 | 54.6% | 89.5% |
| 1 | 359,700 | 64.1% | 61.1% |
| offline | - | 68.0% | 55.8% |

**Within the online regime, replay volume buys nothing.** A 20x change from
every_40 to every_2 moves the first phase by less than the seed noise.

**At every_1 the system stops being an online learner.** Its profile across
the three phases (64.1 / 76.7 / 61.1) is almost exactly offline's
(68.0 / 77.0 / 55.8), including offline's weakness on the most recent
material, where every other setting scores 89 to 97%. It did not close the
gap by retaining better. It closed it by becoming the thing it was being
compared to, and inherited that thing's failure mode.

**So milestone 3 as stated is the wrong milestone.** Offline sees every
phase interleaved in every epoch and never has to retain anything. It is a
ceiling, not a target.

---

## Reading further

`docs/neuron.md` is the full write-up: every result, every failure, and the
reasoning behind both. `docs/neuron_curve.png` is competence over a
100,000-step life, with the phase changes marked.

`results/` holds the raw output of every experiment in this repo. Nothing in
the write-up is unsupported by a file there.
