# Provenance: proving what shaped a model, and removing it

A model that learns continuously accumulates influence nobody recorded. This
note covers what was built to fix that, what it caught on its first run, and
an investigation where the obvious cause was wrong twice.

All of it is in the installed runtime rather than in experiment scripts.

---

## Why this and not something else

Two findings from the field pointed here. Real-time gradient updates to a
deployed model's base weights remain rare because the safety, evaluation and
rollback cost outweighs the benefit — the blocker on deployed learning is not
the learning, it is the governance of it. And on erasure: if personal data
sits in external retrieval layers rather than embedded in model weights,
deletion can be more direct and more auditable.

That second point describes this project's existing architecture. The
fact/skill split was built for a different reason — the whole-system result
at 7B showed weights learn patterns and retain zero facts while a store
holds every fact and learns no pattern — and it happens to be the split that
makes deletion tractable.

---

## 1. Attribution

Every stored sentence records its source. Every weight update is journalled
with its source. `neuron sources` shows both:

```
    stored  learned  source
         0       61  batch-a
         0       59  batch-b
    13,916        0  wiki
```

The journal records only items that CAUSED a weight update, not everything
seen: the gate rejects roughly half of what arrives and the store takes most
of the rest. On one measured read, 20,000 sentences produced 2,837 journal
entries. That is what makes keeping it affordable, and it is exactly the set
needed to replay.

---

## 2. Answers that say what they rest on, and when they don't

`neuron explain` returns the answer with the notes behind it and flags when
the two share little vocabulary.

**It caught a real failure on its first run.** Asked what anarchism is
skeptical of, the system answered correctly and reported **16% vocabulary
overlap** with the four retrieved notes. None of them contained the answer.
It came from the model's pretraining while the prompt instructed it to use
only the notes.

That is worse than a refusal, because it reads as retrieval working. Before
this existed, the only way to notice was to pass a flag and read the notes
by hand.

The overlap test is word-level, not entailment, and the output says so.

---

## 3. Which layer failed

An ungrounded answer alone is not actionable. The store lacking the material
and the ranker missing it need opposite fixes, and conflating them cost an
afternoon of chasing the wrong layer. So `explain` now names the cause:

| diagnosis | meaning | where to fix it |
|---|---|---|
| `grounded` | the answer uses the notes | nothing |
| `declined` | it said it did not know | nothing |
| `not_in_store` | no stored sentence shares the question's distinctive terms | what gets read and stored |
| `retrieval_missed` | the material is there and ranking did not surface it | the ranker |
| `retrieved_but_unused` | the right note came back and the answer ignored it | generation or the prompt |
| `question_too_generic` | too few distinctive terms to judge | nothing; the test says nothing here |

The check is a scan over the token sets the store already keeps, so it costs
no extra storage and is instant at tens of thousands of sentences. It would
need an inverted index at millions.

**Three bugs in this one feature, all found by testing cases rather than
reading the code.** Question words counted as distinctive, because "what"
appears in under 5% of Wikipedia sentences and passed a
document-frequency filter — so a stopword list was needed alongside it.
Terms the store had never seen were dropped before they could count, which
reported a question about something wholly unknown as "too generic to
judge", the opposite of what it means. And a single unseen term was enough
to declare absence, so "what is it made of" was reported as material the
store lacks. Both branches now require at least two pieces of evidence
before committing to a verdict.

---

## 4. Deletion: exact on the store, replay on the weights

`neuron forget <source>` removes every stored entry from that source, and
its absence is verifiable.

`neuron forget <source> --weights` does the other half. It restores the
latest weight checkpoint taken before the source first appeared in the
journal, then relearns every later entry except that source's. Measured on
real data with a 1.5B model:

```
  store: removed 0 entries, 1,198 remain
  weights: restoring and replaying ...
  restored to journal position 0, replayed 61 items, skipped 59
```

**What it is.** Retraining without the data, which is what an erasure
request asks for and what is normally called the gold standard for it.

**What it is not.** Undoing the updates. Gradient steps are not invertible
and machine unlearning is an open problem; no tool can excise the influence
surgically.

**What it is still not, and this is the caveat most tools would omit.** The
result is not identical to a world where the source never arrived. The
stability layer is stateful: its surprise window, rehearsal buffer and
canary baseline all saw the removed items during the original run, so the
sequence of decisions differs. The removed material is gone from what the
weights trained on. The path taken to get there is not the same path.

**Cost is set by checkpoint frequency**, and the run above shows why. With
checkpoints every 500 updates and only 120 journal entries, none existed
yet, so it fell back to the original weights and replayed everything.
Deleting something learned this morning is cheap; deleting something from
the first day is a full replay. The interval is currently a constructor
argument and should be exposed in the config file.

---

## 5. The retrieval investigation, where the cause was wrong twice

The anarchism failure was blamed on the router first: the sentence that
answers the question has no digit and no capitalised word past the first, so
the shipped rule should have sent it to the weights.

**Measured on 20,000 sentences**, the router sends 69.6% to the store and
30.4% to the weights — the first time it had been measured rather than
assumed. A rule counting any capitalised word anywhere sends 99.7% to the
store, which is not a split, so that alternative is dead.

**The coverage test intended to prove the bug was circular.** It found the
current rule stored 11 of 11 answer-bearing sentences, but those sentences
were sampled FROM the store, so they were guaranteed to have been stored.
The result is worthless and is recorded as worthless so that 11 of 11 is
never read as evidence.

Second hypothesis: the ranker. Dense retrieval matches topic, so asked what
anarchism is skeptical of it returns everything ABOUT anarchism, and the
sentence that answers is not closer than the rest. BM25 over the same store
ranks the answer-bearing sentence first on that exact query. Hybrid
retrieval was the expected fix.

| ranker | recall@1 | recall@4 |
|---|---|---|
| dense | 6/6 | 6/6 |
| lexical (BM25) | 6/6 | 6/6 |
| hybrid, any weight | 6/6 | 6/6 |

**Every ranker is perfect, so there is no ranking bug on this evidence.**

And this corrects an earlier number. `probes.py` had reported 6 of 11
retrieved. Five of the eleven probe sources are not in the current store at
all, and it was counting absence as retrieval failure. **Retrieval was never
6/11. It was 6/6 on what was present.**

**The actual cause.** A direct search found five stored sentences containing
"skeptical" and none of them was the one. The sentence is not in the store.
The most likely reason is upstream of the runtime entirely: the data fetcher
filters sentences to between 60 and 250 characters, and the article's
opening sentence exceeds that.

So the failure chain was: a length filter in a convenience script dropped
the sentence, the store never had it, retrieval could not return it, the
model answered from pretraining, and `explain` correctly flagged 16%
overlap. Three layers investigated, two wrong hypotheses, and the cause was
in the least interesting place.

---

## What this pillar is worth, honestly

**Real.** Source tracking through the store and the weights. Grounding
detection that caught a live failure. Layer diagnosis that reaches the right
conclusion on the case it was built for. Exact store deletion. Weight
deletion by replay, with its cost measured and its limits stated.

**Not.** The replay has never been verified to produce a model that behaves
as though it never saw the source; only that it trained without it. The
grounding check is word overlap. The layer diagnosis is a term-matching
heuristic and would need an inverted index past a few hundred thousand
sentences. And the whole feature has been exercised on one store and a
handful of questions.

---

## Reproducing

```
neuron read <file> --source batch-a
neuron sources
neuron explain "a question about something in that file"
neuron forget batch-a --weights -y
python3 retrieval_eval.py --probes <a probes.py file>
python3 route_eval.py
```
