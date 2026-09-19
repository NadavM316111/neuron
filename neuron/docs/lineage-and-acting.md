# Lineage, cold start, and learning from what you did

Three results and one negative, all produced inside the runtime rather than
in experiment scripts. Merging two instances that lived contradictory lives
produces a model better than either. A newcomer seeded from that merge
starts useful and the advantage fades correctly. An agent that chooses its
own actions to get good outcomes destroys its own learning signal. And the
router, suspected of a failure, turned out not to be the cause.

Qwen2.5-1.5B with LoRA, on a laptop.

---

## 1. Merging works in the product

Two instances read contradictory streams of the same invented rule. Both
started from identical weights, confirmed: each read 6.3375 and 6.5631 on
the two held-out probe sets before learning anything.

| | opened | sealed | mean |
|---|---|---|---|
| A, lived "opened" | 0.8307 | 2.2262 | 1.5284 |
| B, lived "sealed" | 2.0994 | 0.6906 | 1.3950 |
| **merge of A and B** | **1.1920** | **1.0822** | **1.1371** |

Held-out cross entropy, lower is better.

The merge beats both parents on the mean, and it beats simple interpolation
on each rule separately: 1.19 on "opened" where the parents' midpoint is
1.47, and 1.08 on "sealed" where the midpoint is 1.46. Better than halfway
on both rules at once.

This reproduces the grid-world finding, where the interpolation midpoint beat
both ends, in language and through the runtime's own `export_lineage` and
`merge_lineage` rather than in a bespoke script.

**Why plain averaging is valid.** It only combines instances descending from
the same initialisation. The runtime pins the LoRA init seed for exactly
that reason, so copies of one install start from identical trainable
weights. A fingerprint check refuses mismatched merges, and that refusal was
verified to fire.

**A near miss worth recording.** The first attempt held both models in
memory at once, which exhausted a 1.5B float32 pair on this machine. The
second load did not raise. It produced a model whose output was uniform over
the vocabulary, reading ln(151936) = 11.9312 identically on every probe set.
Averaging a trained model with a uniform-noise model would have produced a
plausible-looking and meaningless merge. It was caught only because the same
number appeared twice to four decimal places. Instances are now trained
strictly one at a time, and a baseline-comparison check would catch it
directly.

---

## 2. A newcomer starts useful, and then outgrows it

A fresh instance lives a new draw of the "opened" rule — the same templates,
different sentences — so anything that transfers is the rule, not memorised
strings.

| sentences read | blank | seeded from merge | seeded from A alone |
|---|---|---|---|
| 0 | 6.4647 | 1.1502 | 0.8461 |
| 25 | 2.4839 | 1.1158 | 1.4355 |
| 50 | 1.6335 | 0.9542 | 0.8059 |
| 100 | 1.0863 | 1.3445 | 0.7156 |
| 200 | 0.7467 | 0.7090 | 0.6911 |
| 400 | 0.6010 | 0.5701 | 0.7406 |

**A blank instance needed 100 sentences to reach what the seeded one knew
before reading anything.** The advantage runs +1.37 at 25 sentences and
+0.03 by 400.

That fading is the correct shape and the thing to check for, not a
disappointment. An advantage that persisted would mean the newcomer was
constrained by its inheritance rather than accelerated by it. On the grid
world the same curve went from +64.7 points at 500 steps to +6.6 at 6,000.

**What this does not establish.** The third arm, seeded from the single
parent that already knew this rule, ended at 0.7406 against the merge's
0.5701. That reads as the merge beating the best parent, and it is not
solid: the single-parent curve went 0.8059, 0.7156, 0.6911, 0.7406, rising
at the end, which is noise rather than convergence. The merge was also worse
than blank at 100 sentences. At one seed these curves are bumpy and a
final-point comparison is fragile. The supported claim is that **merging is
not worse than copying your best instance**, which still matters, because it
means consolidating a population costs nothing.

**An unaddressed problem this exposed.** The merge-seeded instance began
knowing the opposite rule at 0.9301, inherited from B. It then lived an
"opened"-only life and that knowledge drifted to 1.7414. So it forgot
something it had inherited rather than learned. Inherited knowledge that the
newcomer's own life does not reinforce decays, and nothing in the retention
work addresses that case.

---

## 3. Acting to get a good outcome destroys the learning signal

The first test of learning from consequence rather than from a stream. A
grid world with berries and fungus, moving onto one eats it, and the rule
reverses between phases. Majority-class baseline 50.0%.

| policy | learned to | kept after reversal | fungus eaten, phase 2 |
|---|---|---|---|
| passive random walk | 88.3% | 63.7% | 982 |
| greedy (epsilon 0.1) | 56.2% | 55.8% | 184 |
| explore (epsilon 0.5) | 75.3% | 64.3% | 640 |
| curious (max uncertainty) | 68.3% | 59.7% | 494 |

**Greedy learned nothing**, 56.2% against a 50.0% baseline, on both seeds
and both runs. The mechanism is in the coverage column: it ate a fifth as
much as the passive arm. With half of all outcomes being "nothing" and the
rest split between good and bad, refusing to step on anything is the best
policy available to an agent that does not yet know the rule. Optimal
behaviour, and it switches the learning signal off completely.

So an acting system does not merely bias its data. It can starve itself.
Exploration is a precondition for consequence being a learning signal at
all, and it costs about 13 points against a random walk.

**Curiosity did not work.** Choosing the action whose outcome was most
uncertain reached 68.3% against plain randomness at 75.3%, on both seeds. It
beats value-seeking and loses to being random half the time, so on this
world the machinery does not earn its place. One observation worth a
follow-up: curiosity is the only policy whose sampling ROSE after the rule
reversed, from 363/454 to 805/638 items in one seed, while exploration's
fell. A reversal makes outcomes uncertain again, which is exactly what it
seeks. Two phases may be too short to reward that.

**Two design errors caught here, both mine.** The first version read
forgetting numbers off arms that had learned nothing, because it checked
that the passive arm could forget and never checked that the acting arm
could learn. The second credited curiosity for beating greedy when
exploration was beating curiosity on both seeds. Both were found by reading
the table rather than the verdict.

---

## 4. Retention on this grid world is not measurable at this scale

The passive arms give a retention comparison, and it moved between two runs
of the same code: advantage +6.2, then +0.2. Same seeds, only the arm list
differed.

So at two seeds this experiment cannot detect a retention effect. The honest
statement is not that the effect is small here, it is that it is inside the
noise. The language drift result had tight seed agreement and a
matched-compute confound arm, so it stands, but retention is clearly
domain-dependent and this is the weaker case.

---

## 5. Provenance, and the failure it caught immediately

Three capabilities added to the runtime: every stored sentence records its
source, every answer reports what it rests on, and a source can be removed.

On its first real run, asked what anarchism is skeptical of, `explain`
returned a correct answer and flagged that it shared **17% of its vocabulary
with the four retrieved notes**. None of them contained the answer. It came
from the model's pretraining while the prompt instructed it to use only the
notes: an ungrounded answer wearing a retrieval costume, which is worse than
a refusal because it reads as the system working.

The overlap test is word-level, not entailment, and is reported as a prompt
to check rather than a verdict.

**Deletion is exact on the store and honestly limited on the weights.**
`forget` removes every entry from a source verifiably, then states that
pattern influence on the weights is not removed and cannot be excised
surgically, and names the available remedy: restore a checkpoint and replay
the stream without that source. Machine unlearning is an open problem and a
runtime claiming otherwise would be overstating it.

---

## 6. The router is not the problem, and the test that said so was circular

The anarchism failure was initially attributed to the router: the sentence
that answered the question has no digit and no capitalised word past the
first, so the shipped rule should have sent it to the weights.

Measured on 20,000 real sentences:

| rule | to the store | to the weights |
|---|---|---|
| current (digit or 2+ propers past the first) | 69.6% | 30.4% |
| any capitalised word anywhere | 99.7% | 0.3% |

The 69.6/30.4 split is the first time the router has been measured rather
than assumed.

The coverage test intended to prove the bug found the current rule stored 11
of 11 answer-bearing sentences. **That test was circular** and the result is
worthless: the probe sentences were sampled FROM the store, so they were
guaranteed to have been stored. The router is not dropping informative
sentences on this evidence.

Which relocates the anarchism failure to **retrieval ranking**: the sentence
was likely stored and the embedding search preferred sentences containing the
word "anarchism" over the sentence answering the question. That is a harder
problem, and measuring it needs probes written against sentences chosen
before knowing what the store holds — the same trap that invalidated an
earlier retrieval measurement in this project.

The alternative rule is dead either way: 99.7% to the store is not a split.

---

## What is now true of the runtime

Installable, with a config file and a CLI. Reads a stream, routes, learns,
guards, stores, evicts, persists, answers, explains what an answer rests on,
and removes a source. Exports and merges lineage checkpoints and refuses
mismatched ones.

## What is not

Nothing has run longer than a few hours, and "runs indefinitely" is the
central claim.

Retrieval ranking is unmeasured and is the live suspect for ungrounded
answers.

The acting loop needs an exploration mechanism before consequence is usable,
and the one tried here lost to random.

Inherited knowledge decays when the newcomer's own life does not reinforce
it, and nothing addresses that.

Everything above is one seed or two, one model size, and toy or synthetic
streams.

---

## Reproducing

```
python lineage_test.py
python coldstart.py
./run.sh experiments/grid/acting.py
python route_eval.py
neuron explain "your question"
```

Raw output in `results/acting.json`, `coldstart.json` and the lineage
checkpoints under `~/.neuron/lineage`.
