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

## Stakes, 1 Oct 2026

`experiments/device/live3.py`

The system perceived a world nobody designed and took actions that left the
machine. Both real, and nothing it did mattered TO IT. It could be wrong
forever at no cost, and a run ended when the process was killed rather than
because the system failed at anything.

It now has credit. Every sample costs 0.10, every correct prediction pays
0.15, every consolidation round costs 0.05, so break-even is 67% accuracy.
At zero the life ends and the weights, the buffer and the drives go with
it. A new generation starts from random weights. There is no snapshot to
restore, because a failure you can roll back from is not a failure.

**The bar is set where the dumbest policy survives and a wrong one does
not.** The majority baseline on this task runs 75 to 90%, so a system that
learns nothing beyond "say PRESENT" lives. One that is actively wrong
starves. A world where the dumbest policy dies measures the harness; one
where nothing dies does not measure stakes.

**Sleep is now a bet it pays for, in real time.** It won on accuracy when
consolidation was free (20 Sep, +8.9 points), and the advantage vanished
when it cost energy (`lineage.py`, -4). This is the third setting and the
first where the cost is paid in a currency the system needs to survive.

This is also where the parked population work rejoins the project.
`pop.py` and `pop2.py` failed in a grid world that could not show whether
inheritance helps. Here lives are days long and the world is real, so
generations accumulate slowly and honestly.

**The honest limit: I chose the costs.** Stakes defined in a config file
are not stakes the world imposes, and a resource that exists only in this
repo is a long way from one anybody else recognises. The next rung needs a
transaction with a system that does not know this is an experiment.

---

## The body in the real world, 26 to 30 Sep 2026

`experiments/device/live2.py`, `experiments/device/reach.py`

Two systems ran continuously on a laptop for three days.

`live2` predicts whether the person will be at the keyboard in five
minutes, from idle time, the machine, the clock and Fort Lauderdale
weather. It consolidates when it believes nobody is there, which costs CPU
that it then observes.

`reach` chooses one public source to query per tick out of five, and three
allocation policies run side by side against the same world: round-robin,
random, and the drive from 20 Sep. It made 3,161 real requests over 79
hours.

### What held up

79 hours of continuous running across lid closes and sleeps, resuming from
disk every time. **Zero rollbacks** over the whole stretch with
`canary_from_stream` on, against 232 and accelerating in the 3,000,000-step
grid run without it. That fix had shipped in `stability.py` on 10 Sep and
nothing had ever used it.

### What the results showed

`live2`: ambiguous. Whenever presence was high and the baseline easy, edge
was exactly 0.0, meaning it was predicting PRESENT and nothing more. Edge
only went positive when presence dropped — +27.5 at 67% present, +12.5 at
74%. That is either a system whose learning only shows when the task is
hard, or a slight bias toward AWAY that pays off in mixed windows and costs
nothing in easy ones. The occasional -0.8 rows are mild evidence for the
second. Unresolved, and it needs a transition-specific metric rather than
overall accuracy.

`reach`: nothing. Round-robin and random produced identical accuracy to a
tenth of a point at every checkpoint across 1,060 ticks, before and after
the leak fix below. Most likely all three policies collapsed to predicting
SAME, which is right about 90% of the time at the chosen change threshold,
so their accuracies converge on the base rate regardless of what they
learned. Same failure as the original `live.py`: a prediction target with
almost no variance.

### Two bugs that cannot happen in a simulator

**The infrastructure destroyed its own sensor.** `caffeinate -i` prevents
idle sleep by continuously resetting HIDIdleTime, which is exactly the
signal the presence detector reads. A 21-hour overnight run reported 99%
presence through a night of sleep, trained entirely on a constant label,
and consolidated 2,065 times out of 2,320 samples because it believed
nobody was there. `caffeinate -s` prevents system sleep without touching
the idle timer. Verified: 45.08 seconds reported after a 45-second wait.

**The answer leaked into the features.** In `reach`, `world.ask()` updates
`last_val` and the feature vector reads `last_val`, so building features
after the query handed the model the very value it was being asked to
predict the direction of. Fixed by capturing features before the query.
This is a real-time hazard specifically: the observation and the query
happen at the same instant, so ordering within a tick is load-bearing in a
way it never is when replaying a file.

Neither bug has an analogue in a grid world. That is what leaving the
simulator costs, and finding them is the work.

---

## A body, 26 Sep 2026

`experiments/device/live.py`

Every experiment before this one ran in a world this project designed. Grid
worlds, invented vocabularies, synthetic contradictions: the rules were
written by hand, so every result was partly a result about whoever wrote
them. Even the weather run replayed a fixed file after the fact.

This one watches the machine it runs on. CPU, memory, battery, disk,
network, load, time of day, day of week, sampled every few seconds. It
predicts what CPU load does next, learns from being wrong, and decides when
to consolidate.

**The loop is actually closed.** Consolidating costs CPU. CPU is part of
what it observes. So its own choices change its own future input,
permanently, with no reset: in the grid worlds the agent acted but the
episode restarted, and here it is Tuesday afternoon exactly once.

**It persists.** Weights and counters go to disk every hundred samples, so
it survives being killed, a closed lid, or a restart, and resumes by
default.

First run, 1,736 samples at 0.5s intervals: the loop works end to end,
nothing leaked, and the guard fired zero times. It also learned nothing
beyond the majority class, which is correct for that interval: CPU rarely
moves more than the change threshold in half a second, so SAME dominates
and guessing SAME is optimal. At 10s the classes balance. Left running, the
thing it has to learn is a daily cycle, and a twenty-minute run contains
none of one.

Small and unglamorous, and it closes the last structural gap before a
device: a world nobody designed, running in real time, where its actions
change what it sees next.

---

## The guard degrades over a long life, 26 Sep 2026

`experiments/grid/longrun.py`

Everything in this repo is short. The longest run was 100,000 steps; most
finish in seconds. The central claim of the project is that a system
accumulates a life over time, and time had never been the variable.

3,000,000 steps, one continuous life, no resets, 16 minutes of wall clock.
A passive random walk rather than a policy, so slow layer failure could not
be confused with slow policy collapse.

**Almost everything survived.** Accuracy flat at 58 to 61% across the whole
life, replay buffer pinned at its cap of 500 with no leak, wall clock flat
at 33 seconds per 100,000-step block from start to finish. No drift, no
slowdown, no saturation.

**The guard did not.** Rollbacks over the life:

| step | rollbacks |
| --- | --- |
| 500,000 | 0 |
| 1,000,000 | 13 |
| 2,000,000 | 51 |
| 3,000,000 | 232 |

Not a constant rate. A curve bending upward. **In August the guard fired
zero times in a 100,000-step life and was written up as working. It needed
thirty times that length to show this.**

**The cause is in the health column, which oscillates with the world.** It
alternates between about -1.09 and about +0.3 in lockstep with the rule
reversals, and the positive side grows: +0.05 early, +0.47 by 2.6M. The
canary is built from one rule, so when the world flips, canary loss spikes
and the guard reads WORLD-CHANGE AS SELF-DAMAGE. Previous-rule accuracy
falls 3 points over the same span, consistent with rollbacks eating
retention.

`stability.py` has documented this mismatch since 10 Sep — "the canary
decides what the guard protects, and that is the whole problem" — and has
shipped `canary_from_stream` as the fix since then. **Nothing had ever used
it.** `live.py` is the first thing that does, and its guard has so far
fired zero times.

The general lesson is the one this run exists to make: a mechanism that has
never fired is not protection, it is an untested claim, and short runs
cannot tell the difference.

---

## Sleep, 20 Sep 2026

`core/sleeping.py`, `experiments/grid/sleep_schedule.py`

Everything in this repo consolidates every 5 items while the stream keeps
arriving. Living and consolidating are interleaved at the finest possible
grain. That setting was chosen early and never questioned, and no
experiment here has ever run any other shape.

An animal does the opposite: a long block of living with no consolidation,
then a block of consolidation with no input. Same work, different shape in
time.

Five schedules, **identical replay budgets** (71,960 to 72,000 replays,
1.00x across every arm), 8 seeds. Contested events in the earliest phase:

| schedule | awake between | keyed | open |
| --- | --- | --- | --- |
| continuous | 5 | 64.7% | 60.9% |
| micro | 25 | 65.3% | 62.7% |
| small | 100 | 66.6% | 63.9% |
| nap | 500 | 71.7% | 67.2% |
| sleep | 2,000 | 73.6% | 67.4% |
| hibernate | 6,000 | 72.8% | 62.7% |

**Sleeping wins by up to 8.9 points on the same replay budget.** The
system has been consolidating in the worst available shape since August.
Not too little replay. The right amount, spent wrong.

**There is a threshold, not a preference for separation.** Separating
consolidation from input by 25 items buys +0.6, which is nothing. The
effect turns on between 100 and 500 and is mostly there by 500. Short
blocks get interrupted before they achieve anything; long blocks complete.

**And there is an optimum.** `hibernate` at 6,000 comes in below `sleep` at
2,000, and drops sharply on the middle phase (62.7% against 67.4%). Two
forces trade off: longer blocks let consolidation finish, longer waking
periods let drift accumulate with nothing protecting earlier material.

The claim this supports is narrow and mechanical: the cost was
INTERFERENCE between replay and incoming data, not insufficient replay. An
offline phase is worth building into the system rather than interleaving.

---

## Dreaming, 20 Sep 2026

`core/dreaming.py`, `experiments/grid/dream_replay.py`

Sleep also recombines stored fragments into sequences that never occurred.
Outcomes cannot be invented — a dream that makes up what happens is
training on fiction — so every moment in a dream here is a real moment with
its real outcome, and only the ORDER is invented.

Three recombinations against veridical replay, 8 seeds, contested events in
the earliest phase:

    veridical    65.3%
    splice       64.2%   (-1.1)
    interleave   60.0%   (-5.3)
    scramble     63.2%   (-2.1)

**Recombination does not help.** Splice is inside noise, interleave is
worse.

**AND THE CONTROL FAILED, WHICH IS THE REAL FINDING.** `scramble` exists
to lose: August measured scrambled replay damaging shared knowledge, 96.2%
against 99.8%. Here scrambling a genuine contiguous trajectory costs 2.1
points on contested and 0.0 on shared. The model is not using sequence
order on this world at all.

So whatever the +15.5 sequence-replay result (21 Aug) was measuring, it was
not order. Most likely the fact that a sequence delivers 10 correlated
items per consolidation instead of 1. **That result needs re-checking and
should not be cited as evidence about ordering until it is.**

### The bug this found, which was worth more than the experiment

`StabilityLayer._remember` is called ONLY on branches where the gate
REJECTS an item. Anything it learns from goes to the weights and is never
stored. At `top_fraction=0.50` that is roughly half the stream.

**So a stored "sequence" of 10 was never 10 consecutive moments.** It was
10 non-adjacent rejected moments spanning about 20 steps, with gaps
wherever the gate fired. Every sequence result in this repo was measured on
units that are not trajectories.

`contiguous=True` stores every item into the run. The first dream run had
to be thrown away and rerun with it, and the fix alone moved the veridical
control from 60.7% to 65.3% on the first phase and 84.0% to 99.8% on the
last — larger than any effect the experiment was designed to measure.

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
