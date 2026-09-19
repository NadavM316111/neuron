"""
NEURON stability layer.

Model-agnostic. Knows nothing about transformers, tokenizers, LoRA, images
or grid worlds. Decides what is worth learning from, what to rehearse, and
when the model has damaged itself badly enough to be rolled back.

Mechanisms, measured 16-21 Aug 2026:

  gate            surprise gate, top slice of a rolling window. Beat random
                  selection on language fact-learning (kept 1.80 vs 0.92).
                  On Fashion-MNIST it was actively HARMFUL (50.82% vs
                  54.45%). In the grid world it matched full learning on a
                  third less compute. Setting-dependent, not a general win.
  veto            coherence rejection. Junk hits 8.3 -> 2.0 at zero cost.
  anchor+rehearse replay of rejected-but-ordinary material. On Fashion-MNIST
                  this was THE mechanism, 70.2% of the recovery on its own.
  guard           canary rollback. Recovered a model from total collapse.
                  Protects general knowledge at high learning rates but does
                  NOT prevent fabrication about taught material. Fired zero
                  times across a 100,000-step grid-world life.
  loss floor      skip updates on already-learned material.

SEQUENCE REPLAY (sequence_len > 1), added 21 Aug. Storing runs of
consecutive moments and replaying them IN ORDER, rather than isolated
snapshots in random order. Worth +15.5 points at matched compute against
+4.0 for extra volume alone, and scrambled replay actively DAMAGED shared
knowledge (96.2% vs 99.8%). Essential for any model with hidden state.

REHEARSAL BUDGET, measured 7 Sep 2026 (experiments/language/rehearsal.py).
The rehearse_per_item=6, rehearse_count=2 setting was chosen early and never
tested. It produces roughly 6.3 replays per update, and on the language
drift benchmark that is about FOUR TIMES more than the protection needs:

  replays   final phase-1 loss   rise from best
  1576      1.7058              +0.1561      (rehearse_per_item=6)
   400      1.7069              +0.0140      (rehearse_per_item=24)
   192      1.8481              +0.0521      (rehearse_per_item=48)
     0      2.3293              +1.6851      (no rehearsal at all)

Two seeds. A quarter of the replay work matches the default exactly on final
loss and beats it on rise. An eighth still protects, at a small cost.

Halving rehearse_count instead of the frequency was WORSE at the same replay
count (2.1214 against 1.8337 for 792 replays either way), so replaying fewer
units more often is worse than replaying the same units less often. Halving
sequence_len to 4 was worse still (2.1858).

SCOPE: one synthetic contradictory stream, one 1.5B model, attention-only
LoRA. The weather and grid-world configurations were tuned separately and
this has NOT been re-checked against them.

REPLAY POLICY (replay_policy), added 21 Aug. Which stored units to replay:
  "uniform"    sample at random. The original.
  "old"        favour older material. Recent material is already being
               learned from the live stream, so replaying it is redundant;
               what needs protecting is what is falling out of reach.
  "surprising" favour units the model currently predicts badly. The surprise
               gate applied to replay rather than intake, targeting exactly
               the material that has been overwritten.

Scheduling: rehearse_per_item and guard_per_item fire off ITEMS SEEN. The
legacy rehearse_every and canary_every fire off UPDATES MADE, which ties
protection to learning rate and caused two failures.

Two experimental options, both OFF by default:
  min_surprise    absolute bar. TESTED AND REJECTED 17 Aug.
  drift_correct   gate on (surprise minus current canary mean).

Rejected, do not add back:
  gradient clipping   drift +0.61 -> +1.00, PROBE +0.56 -> -0.06.
  repetition/echo     as an importance signal. Drove PROBE negative.

THE GUARD REQUIRES A CANARY. Fixed 7 Sep 2026. With canary=[] every guard
path is silently dead: _canary_score returns 0.0, health() is permanently
0.0, and the rollback test asks whether 0.0 exceeds 0.0 + tolerance, which
it never does. The layer runs, reports healthy, and has no protection at
all. That is exactly how neuron_system.py was configured, so the guard has
never fired in the persistent system despite being listed as working.
guard=True with an empty canary is now a hard error rather than a quiet
no-op. Pass guard=False if you genuinely want it off.

AGES, fixed 7 Sep 2026. Age used to live in a dict keyed by id(unit), which
grew forever and could collide after garbage collection reused an address.
Stored units now carry their own age as (age, unit) tuples inside the
anchor list and buffer deque, so eviction disposes of the age with them.

A backend must implement:
    score(item)      -> (surprise: float, coherence: float | None)
    update(item, n)  -> float | None
    snapshot()       -> Any
    restore(state)   -> None
    on_rollback()    -> None   optional
    begin_sequence() -> None   optional, for sequence replay
    update_sequence(items, n) -> int | None   optional, for sequence replay
"""

from collections import deque
import random


DEFAULTS = dict(
    # gate
    window=40,
    top_fraction=0.30,
    warmup=12,
    min_surprise=0.0,
    drift_correct=False,
    canary_refresh=5,
    # veto
    coherence_veto=1.3,
    # floor
    loss_floor=0.35,
    # rehearsal
    rehearse_every=2,
    rehearse_per_item=None,
    rehearse_count=2,
    rehearse_steps=1,
    anchor_size=40,
    buffer_size=200,
    sequence_len=1,
    replay_policy="uniform",   # uniform | old | surprising
    replay_pool=8,             # candidates considered per pick when ranking
    # updates
    steps_per_update=3,
    # guard
    guard=True,
    canary_every=10,
    guard_per_item=None,
    canary_tolerance=0.40,
    # Rollback runway. expect_items is how many items the run will see, if
    # the caller knows; min_runway is how many must remain for a rollback
    # to be worth taking. See _check_health for the measurement behind it.
    expect_items=None,
    min_runway=200,
    # Canary scope. See _check_health: a canary of generic English makes the
    # guard protect general competence, which on a task with unfamiliar
    # material means rolling back the task itself.
    canary_from_stream=0,      # how many seen items to fold into the canary
    collapse_only=False,       # fire only on catastrophic damage
    collapse_multiple=4.0,     # what "catastrophic" means, x tolerance
)

POLICIES = ("uniform", "old", "surprising")


class StabilityLayer:
    def __init__(self, backend, canary=None, seed=0, verbose=False,
                 **overrides):
        # UNKNOWN KEYS ARE AN ERROR, not a stored no-op. Found 10 Sep the
        # expensive way: an experiment passed canary_from_stream=8 to a
        # copy of this file that predated the option, **overrides accepted
        # it, cfg stored it, and nothing read it. Three runs and about an
        # hour of compute produced results identical to the unfixed code,
        # and the conclusion drawn from them was that the fix had failed.
        # A setting that silently does nothing is worse than a crash.
        unknown = set(overrides) - set(DEFAULTS)
        if unknown:
            raise TypeError(
                f"StabilityLayer got unknown option(s): "
                f"{', '.join(sorted(unknown))}. Known options are: "
                f"{', '.join(sorted(DEFAULTS))}. If one of these is new, "
                f"the stability.py being imported is older than the code "
                f"passing it — check for a second copy on sys.path.")

        self.cfg = dict(DEFAULTS)
        self.cfg.update(overrides)

        # Found 20 Aug: history is a deque capped at `window`, so a warmup of
        # window or more can never be satisfied. The layer then sits in
        # warmup forever and the gate NEVER FIRES, silently. Fail loudly.
        if self.cfg["warmup"] >= self.cfg["window"]:
            raise ValueError(
                f"warmup ({self.cfg['warmup']}) must be less than window "
                f"({self.cfg['window']}). The history deque is capped at "
                f"window, so a larger warmup never ends and the gate never "
                f"fires."
            )
        if self.cfg["replay_policy"] not in POLICIES:
            raise ValueError(f"replay_policy must be one of {POLICIES}")

        # Found 7 Sep: guard=True with no canary is a silent no-op, not a
        # guard. Every health path short-circuits on an empty canary list.
        if self.cfg["guard"] and not canary:
            raise ValueError(
                "guard=True requires a non-empty canary. With canary=[] the "
                "health score is constant 0.0, health() always reports 0.0, "
                "and the rollback test can never fire. Pass a list of "
                "ordinary items the model should always handle, or set "
                "guard=False."
            )

        self.backend = backend
        self.canary = list(canary) if canary else []
        self.verbose = verbose
        self._rng = random.Random(seed)

        self.history = deque(maxlen=self.cfg["window"])
        # Both hold (age, unit) pairs. Age travels with the unit so eviction
        # disposes of it too.
        self.anchor = []
        self.buffer = deque(maxlen=self.cfg["buffer_size"])
        self._run = []

        self.stats = dict(seen=0, updates=0, vetoed=0, rehearsals=0,
                          warmed=0, floored=0, rollbacks=0,
                          unremarkable=0, below_bar=0, checks=0,
                          sequences=0, unscorable=0,
                          rollbacks_suppressed=0)

        self._base_canary = list(self.canary)
        self._stream_canary = []
        self.canary_baseline = self._canary_score()
        self._canary_now = self.canary_baseline
        self._checkpoint = self.backend.snapshot()

    # ---------- health ----------

    def _canary_score(self):
        """Mean surprise on a fixed set of ordinary items. If this rises,
        the model is getting worse at things it was never taught."""
        if not self.canary:
            return 0.0
        total = 0.0
        n = 0
        for item in self.canary:
            s, _ = self.backend.score(item)
            if s is None:
                continue
            total += s
            n += 1
        return total / n if n else 0.0

    def health(self):
        """Positive means ordinary competence has degraded."""
        return self._canary_score() - self.canary_baseline

    def _refresh_baseline(self):
        self._canary_now = self._canary_score()

    def _adjust(self, surprise):
        """Drift correction. Subtract the model's current general level of
        confusion, so a globally degraded model does not read as a stream
        full of novelty."""
        if not self.cfg["drift_correct"] or not self.canary:
            return surprise
        return surprise - self._canary_now

    def _check_health(self):
        """Roll back if ordinary competence has degraded past tolerance.

        THE CANARY DECIDES WHAT THE GUARD PROTECTS, AND THAT IS THE WHOLE
        PROBLEM. Measured 10 Sep on the long-horizon task: with the guard
        ON, the protected arm ended at 6.43, 6.74, 6.28 and 6.49 on four
        seeds. With the guard OFF and nothing else changed, the same arm
        ended at 1.47, 1.66 and 1.40, and beat the unprotected arm by 2.15
        nats. The guard was not helping slightly less than hoped. It was
        causing the failure it exists to prevent.

        The reason is a mismatch, not a bug. The canary is generic English,
        so the guard measures general competence. The task's material was
        invented vocabulary the canary knows nothing about, so LEARNING THE
        TASK RAISES CANARY LOSS, the guard reads that as damage, and rolls
        back the very thing the run exists to learn.

        Two settings follow from that, and both are about scope rather than
        aggression:

          canary_from_stream   add items the system has actually seen to the
                               canary, so "competence" includes the task and
                               not only generic English. This is the right
                               fix when the stream is the thing that matters.

          collapse_only        fire only on catastrophic damage, not on the
                               ordinary tolerance. A guard that exists to
                               recover a model from collapse — which is what
                               it was built for and did in August — should
                               not be arbitrating normal learning.

        Neither is on by default, because both change behaviour the August
        results were measured under. Turn one on deliberately.

        A ROLLBACK NEEDS RUNWAY. Found 8 Sep on a long-horizon experiment:
        a rollback that fires near the end of a stream discards learning the
        system has no remaining input to rebuild, and the result is worse
        than the damage it was preventing.

        The evidence was unambiguous. At a task length of 250 items, four
        seeds split perfectly in two: runs that fired ONE rollback ended at
        2.01 and 2.28, runs that fired TWO ended at 6.28 and 6.43. Nothing
        in between, and the rollback count predicted which cluster in four
        out of four. At 800 items, where there was runway left after the
        rollbacks, the same configuration ended at 2.57 to 3.19 on every
        seed and beat the unprotected arm by 1.38 nats.

        So the guard was not too aggressive in general. It was blind to how
        much stream remained. `expect_items` lets a caller say how long the
        run is, and within the last `min_runway` items the guard reports
        damage without discarding anything. Refusing to act is the right
        call there: the alternative is throwing away the task's actual
        content to protect general competence that nothing further will
        exercise anyway.

        With expect_items unset the behaviour is exactly as before, so a
        stream of unknown length is unaffected.
        """
        self.stats["checks"] += 1
        now = self._canary_score()
        self._canary_now = now
        tolerance = self.cfg["canary_tolerance"]
        if self.cfg["collapse_only"]:
            tolerance *= self.cfg["collapse_multiple"]

        if now > self.canary_baseline + tolerance:
            runway = self._runway()
            if runway is not None and runway < self.cfg["min_runway"]:
                # Damaged, and too late in the stream to recover from a
                # rollback. Record it and leave the weights alone.
                self.stats["rollbacks_suppressed"] += 1
                if self.verbose:
                    print(f"  DAMAGED but only {runway} items left, "
                          f"not rolling back")
                return False
            if self.verbose:
                print(f"  ROLLBACK  canary {self.canary_baseline:.3f} "
                      f"-> {now:.3f}")
            self.backend.restore(self._checkpoint)
            self.stats["rollbacks"] += 1
            if hasattr(self.backend, "on_rollback"):
                self.backend.on_rollback()
            self._canary_now = self._canary_score()
            return False
        self._checkpoint = self.backend.snapshot()
        return True

    def _runway(self):
        """Items expected to remain, or None if the caller did not say.

        None means an open-ended stream, which is the normal deployment
        case and where the guard should behave as it always has.
        """
        total = self.cfg["expect_items"]
        if not total:
            return None
        return max(0, total - self.stats["seen"])

    # ---------- memory ----------

    def _store(self, unit):
        """Earliest material is kept permanently, the rest rotates.

        A plain FIFO evicts the oldest anchor first, which is backwards. On
        16 Aug the buffer hit its cap at item 249/254 and the model
        collapsed at 275 in both seeds.
        """
        entry = (self.stats["seen"], unit)
        if len(self.anchor) < self.cfg["anchor_size"]:
            self.anchor.append(entry)
        else:
            self.buffer.append(entry)

    def _extend_canary(self, item):
        """Fold a seen item into the canary, so the guard's idea of
        "competence" includes the stream and not only generic English.

        Sampling is spread across the run rather than taken from the start:
        a canary built only from early items would protect the beginning of
        a stream and treat everything later as damage, which is the same
        mismatch in a different direction.

        The baseline is recomputed whenever the canary changes, because a
        baseline measured on a different set of items is not comparable.
        Items are added BEFORE the model learns them, so their loss at
        baseline is honest.
        """
        want = self.cfg["canary_from_stream"]
        if not want or len(self._stream_canary) >= want:
            return
        # Spread the picks over the first part of the run.
        every = max(1, self.cfg["window"] // 2)
        if self.stats["seen"] % every:
            return
        self._stream_canary.append(item)
        self.canary = list(self._base_canary) + list(self._stream_canary)
        self.canary_baseline = self._canary_score()
        self._canary_now = self.canary_baseline

    def _remember(self, item):
        """What gets stored: a single moment, or a run of them.

        Items enter here ONLY when the gate REJECTS them, so the gate and
        rehearsal are coupled by construction — you cannot run rehearsal
        with the gate wide open, because nothing would ever be stored.
        """
        n = self.cfg["sequence_len"]
        if n <= 1:
            self._store(item)
            return
        self._run.append(item)
        if len(self._run) >= n:
            unit = list(self._run)
            self._store(unit)
            self.stats["sequences"] += 1
            self._run = []

    def _first(self, unit):
        return unit[0] if isinstance(unit, list) else unit

    def _pick(self, pool, k):
        """Choose which stored units to replay.

        pool is a list of (age, unit) pairs; this returns bare units.

        uniform      at random
        old          favour what was stored longest ago. Recent material is
                     already being learned from the live stream, so
                     replaying it is redundant; what needs protecting is
                     what is falling out of reach.
        surprising   favour what the model currently predicts badly. The
                     surprise gate applied to replay rather than intake.
        """
        policy = self.cfg["replay_policy"]
        k = min(k, len(pool))
        if policy == "uniform" or k == len(pool):
            return [u for _, u in self._rng.sample(pool, k)]

        if policy == "old":
            ranked = sorted(pool, key=lambda e: e[0])
            return [u for _, u in ranked[:k]]

        # surprising: score a random subset, keep the worst. Scoring the
        # whole pool every consolidation would be far too slow.
        n = min(len(pool), max(k, self.cfg["replay_pool"]))
        cand = self._rng.sample(pool, n)
        scored = []
        for _, u in cand:
            s, _c = self.backend.score(self._first(u))
            scored.append((s if s is not None else 0.0, u))
        scored.sort(key=lambda t: -t[0])
        return [u for _, u in scored[:k]]

    def _replay(self, unit):
        """Replay one stored unit.

        A run is replayed IN ORDER, after telling the backend a new sequence
        is starting so it can reset its hidden state.

        Fixed 24 Aug: this used to call backend.update() once per item, and
        every backend detaches its hidden state after each update. So a
        ten-step trajectory was replayed as ten unconnected single steps and
        the gradient could never link step one to step ten — which is the
        entire reason for storing a SEQUENCE rather than snapshots. Every
        number sequence replay has ever produced was measured that way and
        understates it.

        Now the losses accumulate across the run and the backend applies
        them once, so credit flows through the whole trajectory. Backends
        that expose update_sequence() get the corrected path; the rest fall
        back to the old behaviour so nothing breaks.
        """
        if isinstance(unit, list):
            if hasattr(self.backend, "begin_sequence"):
                self.backend.begin_sequence()
            if hasattr(self.backend, "update_sequence"):
                n = self.backend.update_sequence(
                    unit, self.cfg["rehearse_steps"])
                return n if n is not None else 0
            done = 0
            for item in unit:
                if self.backend.update(
                        item, self.cfg["rehearse_steps"]) is not None:
                    done += 1
            return done
        if self.backend.update(unit, self.cfg["rehearse_steps"]) is not None:
            return 1
        return 0

    def _consolidate(self):
        pool = self.anchor + list(self.buffer)
        if not pool:
            return
        for unit in self._pick(pool, self.cfg["rehearse_count"]):
            self.stats["rehearsals"] += self._replay(unit)

    # ---------- the decision ----------

    def observe(self, item):
        """Offer a moment to the system. Usually nothing happens.

        Returns a dict describing what was decided and why. Callers that
        want honest counts must read learned/reason off the return value
        rather than assuming an update happened.
        """
        self.stats["seen"] += 1
        seen = self.stats["seen"]

        if self.cfg["canary_from_stream"]:
            self._extend_canary(item)

        # Item-scheduled protection. Independent of how often we learn.
        if self.cfg["rehearse_per_item"] and \
                seen % self.cfg["rehearse_per_item"] == 0:
            self._consolidate()

        if self.cfg["guard"] and self.canary and self.cfg["guard_per_item"] \
                and seen % self.cfg["guard_per_item"] == 0:
            self._check_health()

        surprise, coherence = self.backend.score(item)

        if surprise is None:
            self.stats["unscorable"] += 1
            return dict(learned=False, reason="unscorable")

        adjusted = self._adjust(surprise)
        self.history.append(adjusted)

        # Warmup. Cannot judge what is surprising before seeing anything.
        if len(self.history) < self.cfg["warmup"]:
            self.stats["warmed"] += 1
            self._remember(item)
            return dict(learned=False, reason="warmup", surprise=surprise)

        # Veto. Incoherent input is high-surprise and worth nothing.
        if coherence is not None and coherence > self.cfg["coherence_veto"]:
            self.stats["vetoed"] += 1
            return dict(learned=False, reason="incoherent",
                        surprise=surprise, coherence=coherence)

        recent = sorted(self.history)
        relative = recent[int(len(recent) * (1 - self.cfg["top_fraction"]))]
        threshold = max(relative, self.cfg["min_surprise"])

        if adjusted < threshold:
            if adjusted >= relative:
                self.stats["below_bar"] += 1
                reason = "below_bar"
            else:
                self.stats["unremarkable"] += 1
                reason = "unremarkable"
            self._remember(item)
            return dict(learned=False, reason=reason, surprise=surprise,
                        adjusted=adjusted, threshold=threshold,
                        relative=relative)

        # Already known. Adam amplifies near-zero gradients back to full step
        # size, so training on converged material is how divergence starts.
        if surprise < self.cfg["loss_floor"]:
            self.stats["floored"] += 1
            self._remember(item)
            return dict(learned=False, reason="already_known",
                        surprise=surprise)

        loss = self.backend.update(item, self.cfg["steps_per_update"])
        self.stats["updates"] += 1

        # Legacy update-scheduled protection, only when per-item is off.
        if self.cfg["rehearse_per_item"] is None and \
                self.stats["updates"] % self.cfg["rehearse_every"] == 0:
            self._consolidate()

        if (self.cfg["drift_correct"] and self.canary
                and self.stats["updates"] % self.cfg["canary_refresh"] == 0):
            self._refresh_baseline()

        if self.cfg["guard"] and self.canary \
                and self.cfg["guard_per_item"] is None \
                and self.stats["updates"] % self.cfg["canary_every"] == 0:
            self._check_health()

        if self.verbose:
            c = "  " if coherence is None else f" c={coherence:.3f}"
            print(f"  LEARN s={surprise:.3f}{c}  {str(item)[:48]}")

        return dict(learned=True, reason="novel", surprise=surprise,
                    adjusted=adjusted, coherence=coherence,
                    threshold=threshold, loss=loss)

    # ---------- introspection ----------

    def summary(self):
        s = dict(self.stats)
        s["health"] = self.health()
        s["guarded"] = bool(self.cfg["guard"] and self.canary)
        s["canary_size"] = len(self.canary)
        s["canary_from_stream"] = len(self._stream_canary)
        s["anchor"] = len(self.anchor)
        s["buffer"] = len(self.buffer)
        return s

    def __repr__(self):
        s = self.stats
        return (f"<StabilityLayer seen={s['seen']} updates={s['updates']} "
                f"vetoed={s['vetoed']} rehearsals={s['rehearsals']} "
                f"rollbacks={s['rollbacks']} checks={s['checks']}>")
