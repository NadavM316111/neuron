"""
NEURON stability layer.

Model-agnostic. Knows nothing about transformers, tokenizers, LoRA, or text.
Decides what is worth learning from, what to rehearse, and when the model has
damaged itself badly enough to be rolled back.

Mechanisms, all measured on 16-17 Aug 2026:

  gate            surprise gate, top slice of a rolling window. Beat random
                  selection at matched update count (kept 1.80 vs 0.92).
  veto            coherence rejection. Junk hits 8.3 -> 2.0, junk absorbed
                  +2.23 -> +0.13, at zero extra compute.
  anchor+rehearse replay of rejected-but-ordinary material. Drift +0.61 ->
                  +0.29 in all 3 seeds. The mechanism is DIRECTION not
                  distance: gated moved less far than always yet drifted 3x
                  more.
  guard           canary rollback. Observed recovering a model from total
                  collapse (probe -5.5 -> +0.39).
  loss floor      skip updates on already-learned material.

  rehearse_per_item   NEW 17 Aug. Rehearsal used to fire every N UPDATES, so
                  cutting updates also cut the protection. In the bar test,
                  arms with fewer updates had WORSE health (-0.26, -0.42 vs
                  +0.08) for exactly this reason. Setting rehearse_per_item
                  schedules consolidation off items SEEN instead, so
                  protection is independent of how often the system learns.

Two experimental options, both OFF by default:

  min_surprise    absolute bar. TESTED AND REJECTED for this purpose on
                  17 Aug: it ANDs with the relative threshold, so it cut
                  total updates 35 -> 7 -> 4 and CORRECT 2.7 -> 0.7 -> 0.0.
                  Purity rose to 83% and the model learned nothing. Filler
                  updates are not the enemy; fact-update VOLUME is the
                  bottleneck. Kept only so the result stays reproducible.

  drift_correct   gate on (surprise minus current canary mean). Fixes the
                  runaway where a damaged model has higher surprise on
                  everything and therefore fires on everything. On the fake
                  backend: updates 52 -> 25, health +62.4 -> +29.7.

Rejected, do not add back:
  gradient clipping   made drift worse (+0.61 -> +1.00) and destroyed
                      transfer (PROBE +0.56 -> -0.06).
  repetition/echo     as an importance signal. Drove PROBE negative.

A backend must implement:

    score(item)      -> (surprise: float, coherence: float | None)
    update(item, n)  -> float | None
    snapshot()       -> Any
    restore(state)   -> None
    on_rollback()    -> None       optional
"""

from collections import deque
import random


DEFAULTS = dict(
    # gate
    window=40,
    top_fraction=0.30,
    warmup=12,
    min_surprise=0.0,        # absolute bar. tested and rejected, leave at 0
    drift_correct=False,     # gate on surprise minus live canary baseline
    canary_refresh=5,
    # veto
    coherence_veto=1.3,
    # floor
    loss_floor=0.35,
    # rehearsal
    rehearse_every=2,        # legacy: every N UPDATES
    rehearse_per_item=None,  # preferred: every N ITEMS SEEN. None = legacy
    rehearse_count=2,
    rehearse_steps=1,
    anchor_size=40,
    buffer_size=200,
    # updates
    steps_per_update=3,
    # guard
    guard=True,
    canary_every=10,
    canary_tolerance=0.40,
)


class StabilityLayer:
    def __init__(self, backend, canary=None, seed=0, verbose=False, **overrides):
        self.cfg = dict(DEFAULTS)
        self.cfg.update(overrides)
        self.backend = backend
        self.canary = list(canary) if canary else []
        self.verbose = verbose
        self._rng = random.Random(seed)

        self.history = deque(maxlen=self.cfg["window"])
        self.anchor = []
        self.buffer = deque(maxlen=self.cfg["buffer_size"])

        self.stats = dict(seen=0, updates=0, vetoed=0, rehearsals=0,
                          warmed=0, floored=0, rollbacks=0,
                          unremarkable=0, below_bar=0)

        self.canary_baseline = self._canary_score()
        self._canary_now = self.canary_baseline
        self._checkpoint = self.backend.snapshot()

    # ---------- health ----------

    def _canary_score(self):
        """Mean surprise on a fixed set of ordinary items. If this rises, the
        model is getting worse at things it was never taught."""
        if not self.canary:
            return 0.0
        total = 0.0
        for item in self.canary:
            s, _ = self.backend.score(item)
            total += s
        return total / len(self.canary)

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
        now = self._canary_score()
        self._canary_now = now
        if now > self.canary_baseline + self.cfg["canary_tolerance"]:
            if self.verbose:
                print(f"  ROLLBACK  canary {self.canary_baseline:.3f} -> {now:.3f}")
            self.backend.restore(self._checkpoint)
            self.stats["rollbacks"] += 1
            if hasattr(self.backend, "on_rollback"):
                self.backend.on_rollback()
            self._canary_now = self._canary_score()
            return False
        self._checkpoint = self.backend.snapshot()
        return True

    # ---------- memory ----------

    def _remember(self, item):
        """Earliest coherent material is kept permanently, the rest rotates.

        A plain FIFO evicts the oldest anchor first, which is backwards. On
        16 Aug the buffer hit its cap at item 249/254 and the model collapsed
        at 275 in both seeds.
        """
        if len(self.anchor) < self.cfg["anchor_size"]:
            self.anchor.append(item)
        else:
            self.buffer.append(item)

    def _consolidate(self):
        pool = self.anchor + list(self.buffer)
        if not pool:
            return
        k = min(self.cfg["rehearse_count"], len(pool))
        for item in self._rng.sample(pool, k):
            if self.backend.update(item, self.cfg["rehearse_steps"]) is not None:
                self.stats["rehearsals"] += 1

    # ---------- the decision ----------

    def observe(self, item):
        """Offer a moment to the system. Usually nothing happens.

        Returns a dict describing what was decided and why.
        """
        self.stats["seen"] += 1

        # Item-scheduled rehearsal. Protection independent of learning rate.
        if self.cfg["rehearse_per_item"] and \
                self.stats["seen"] % self.cfg["rehearse_per_item"] == 0:
            self._consolidate()

        surprise, coherence = self.backend.score(item)

        if surprise is None:
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
            return dict(learned=False, reason="already_known", surprise=surprise)

        loss = self.backend.update(item, self.cfg["steps_per_update"])
        self.stats["updates"] += 1

        # Legacy update-scheduled rehearsal, only when per-item is off.
        if self.cfg["rehearse_per_item"] is None and \
                self.stats["updates"] % self.cfg["rehearse_every"] == 0:
            self._consolidate()

        if (self.cfg["drift_correct"] and self.canary
                and self.stats["updates"] % self.cfg["canary_refresh"] == 0):
            self._refresh_baseline()

        if self.cfg["guard"] and self.canary and \
                self.stats["updates"] % self.cfg["canary_every"] == 0:
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
        s["anchor"] = len(self.anchor)
        s["buffer"] = len(self.buffer)
        return s

    def __repr__(self):
        s = self.stats
        return (f"<StabilityLayer seen={s['seen']} updates={s['updates']} "
                f"vetoed={s['vetoed']} rehearsals={s['rehearsals']} "
                f"rollbacks={s['rollbacks']}>")