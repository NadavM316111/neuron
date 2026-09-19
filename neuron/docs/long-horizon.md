# Long horizons, and the guard that caused what it was built to prevent

An instruction given at the start of a long task decays as unrelated work
piles up. The retention layer prevents about two nats of that decay, at two
task lengths, on three seeds. Finding it required turning off the rollback
guard, which turned out to be destroying the thing it exists to protect.

Qwen2.5-1.5B with LoRA, synthetic task, no tools.

---

## Why this task

The gap between an assistant and a colleague is measured, and it is not
about intelligence. Frontier agents succeed on nearly 100% of tasks that
take a skilled human under about four minutes and under 10% of tasks taking
more than about four hours. The length of task completable at 50%
reliability has doubled every seven months since 2019, recently every four,
and the growth is driven primarily by reliability and the ability to adapt
to mistakes rather than by raw capability.

The specific failure points at memory. Studies of long-context web agents
report success rates of 40 to 50% on short-horizon versions of a task
falling below 10% once the same task is embedded in a longer interaction
history, **even when the relevant information remains technically present in
the context window**. The information is there and performance collapses
anyway, which rules out running out of room and rules out losing the notes.

This is a model of that: an early constraint, then a lot of unrelated work,
then does the constraint still hold.

---

## The setup

- **setup**: 160 items establishing an invented rule, stated many ways
- **filler**: 250 or 800 items of unrelated material that contradicts
  nothing
- **test**: 25 held-out probes of the rule, mean per-token loss

Two arms, with the stability layer and without. Invented vocabulary
throughout, so nothing transfers from pretraining.

**This is not the earlier drift experiment.** That one reversed the rule in
phase 2 and established retention against CONTRADICTION. Here the
intervening material contradicts nothing, it is just volume, which is what a
long task actually looks like. And this project's own scope finding
predicted retention might not help: it cost a little on stationary streams,
-1.56, -1.12 and -2.68 across three shuffled weather climates, and
Fashion-MNIST was inside the noise.

---

## The result

Mean per-token loss on the rule's probes after the filler. Lower is better.
Three seeds.

| filler items | plain | retention | gap |
|---|---|---|---|
| 250 | 3.8569 | **1.8618** | +2.00 |
| 800 | 3.9764 | **1.9979** | +1.98 |

Degradation from the post-setup value agrees: +3.11 against +0.54 at 250,
and +3.23 against +0.87 at 800. Seed spread is 0.23 for retention against
0.59 for plain, so the protected arm is also the more stable one.

**The gap is the same at both lengths**, which is a cleaner shape than
expected. Degradation grows with length in the unprotected arm, and
retention holds it near flat.

**Two metrics, because they disagree in a known direction.** Degradation
penalises deep learning: the plain arm makes 410 or 960 updates against
retention's 91 to 247, acquires the constraint harder, and therefore has
further to fall. Final loss has no such bias. Both are reported and both
have to agree before anything is claimed.

---

## The guard was causing the failure

This result was invisible for three days because of the rollback guard.

With the guard on, the protected arm at 250 filler items ended at 6.4263,
6.7396, 6.2768 and 6.4937 across four seeds, far worse than the unprotected
arm's 3.5. Every one of those runs fired two rollbacks. With the guard off
and nothing else changed, the same arm ended at 1.4693, 1.6632 and 1.3998.

**The cause is a mismatch, not a bug.** The canary is generic English
sentences, so the guard measures general competence. The task's material is
invented vocabulary the canary knows nothing about. Learning the task
therefore raises canary loss, the guard reads that as damage, and rolls back
the very thing the run exists to learn. The guard was not too aggressive in
general. It was optimising the wrong quantity for this setting.

**The fix is scope, not sensitivity.** The canary now includes items the
system has actually seen, sampled across the run rather than from the start,
with the baseline recomputed whenever the canary changes. In testing against
a backend whose generic competence drifts as it learns, spurious rollbacks
went from five to zero while genuine collapse still triggered six.

With that in place and the guard back ON, all six retention runs fired
**zero rollbacks** and produced the table above.

An alternative, `collapse_only`, raises the threshold to four times
tolerance so the guard only fires on catastrophic damage. It reduced
spurious rollbacks from five to two in the same test. It treats the symptom
and is available but off by default.

**Both settings are off unless asked for**, because they change behaviour
the August results were measured under, including the run where the guard
recovered a model from total collapse.

---

## Two wrong diagnoses, recorded

**First wrong theory: rollbacks with no runway.** At 250 items four seeds
split perfectly, two ending near 2.1 and two near 6.3, and the rollback
count predicted which in four out of four: one rollback meant recovery, two
meant ruin. That looked like a late rollback discarding learning with no
remaining stream to rebuild from. A runway check was added so the guard
declines to roll back inside the last 200 items.

It made things worse. All four seeds then blew up. The damaging rollback was
the EARLY one landing just after acquisition, not a late one, and 250 filler
items was not enough to recover. The runway check remains in the code
because it is defensible on its own terms, and it was not the fix.

**Second wrong diagnosis, and this one was mechanical.** Three subsequent
runs with the canary fix produced results identical to the unfixed code, to
three decimal places. The experiment imports `stability.py` from
`~/neuron/neuron/core` rather than from the installed package, and that copy
predated the option. `**overrides` accepted `canary_from_stream=8`, stored it
in cfg, and nothing read it. About an hour of compute produced numbers that
were read as the fix having failed.

`StabilityLayer` now raises on unknown options and names the likely cause: a
second copy on `sys.path` older than the code passing the option. A setting
that silently does nothing is worse than a crash.

---

## What this is and is not

**Is.** Retention reduces long-horizon decay of an early constraint by about
two nats, at two task lengths, three seeds, both metrics agreeing, on
material that contradicts nothing. That is a different claim from the drift
result and it is aimed at a gap the field measures monthly.

**Is not.** One synthetic task. One model at 1.5B. No tool use, no agent
loop, no actions and no consequences. Probe loss rather than task
completion. And 800 filler items is minutes of equivalent agent work, not
the four hours where the documented failure bites.

Calling this "makes AI a colleague" would be several steps ahead of the
evidence. It is a mechanism that helps on a model of the problem.

---

## Reproducing

```
python3 longhorizon.py --setup 160 --fillers 250 800 --seeds 3
python3 longhorizon.py --setup 160 --fillers 250 --seeds 3 --no-guard
```

The second reproduces the guard failure, for anyone who doubts a component
built to protect the model was destroying the task.

Raw output in `longhorizon.json`.
