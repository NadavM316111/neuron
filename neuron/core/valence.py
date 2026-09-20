"""Persistent value per stored unit.

WHAT IS ALREADY HERE, so this does not re-run an answered question.

  The gate uses instantaneous surprise to decide INTAKE.
  replay_policy="surprising" uses instantaneous surprise to decide REPLAY.
  Store.hits counts retrievals and drives eviction in the store.

So surprise-as-importance has been tested at both ends, and the store half
already has a crude usefulness signal. What nothing on the weights side has
is a value that PERSISTS. A stored unit's only attribute is its age. The
surprise that caused it to be stored is measured once and discarded.

WHAT THIS ADDS. Every stored unit carries a scalar that:

  is born from the adjusted surprise that was already computed for it at
      intake, so it costs no extra forward pass;
  is updated every time the unit is replayed, from the model's CURRENT loss
      on it, as an exponential moving average;
  decays toward the floor as material is absorbed, so a unit that was hard
      once and is easy now stops consuming replay budget;
  ranks replay selection, so budget goes to what has stayed important;
  decides buffer eviction, so the least valuable unit is dropped rather
      than the oldest.

WHY THIS IS NOT "surprising" AGAIN. That policy rescores candidates fresh on
every consolidation and keeps the worst right now. It has no memory: a unit
that is hard at one moment and easy at the next is treated as two unrelated
observations, and the ranking cost is paid again every time. Value here
accumulates across the unit's whole life, so a unit that is CONSISTENTLY
hard outranks one that is momentarily hard. That is the difference being
tested, and it is the thing the architecture has never had.

HONEST SCOPE. This is a weighting on replay. It functions like a value
signal. It is not a feeling, a want, or a drive, and nothing here should be
described as one.

UNTESTED. Every number below is a guess and the experiment exists to
replace it. valence_decay=0.7 was chosen so a unit needs roughly three
replays to move most of the way to its current difficulty.
"""

from stability import StabilityLayer


class ValenceLayer(StabilityLayer):
    """StabilityLayer with a persistent value per stored unit.

    Entries are [age, unit, value] lists rather than (age, unit) tuples, so
    value can be updated in place. Everything that reads the pool is
    overridden here; the parent's own _pick and _consolidate are never
    called, so stability.py needs no changes at all.
    """

    def __init__(self, backend, canary=None, seed=0, verbose=False,
                 valence_decay=0.7, valence_floor=0.05, valence_pool=16,
                 **overrides):
        super().__init__(backend, canary=canary, seed=seed,
                         verbose=verbose, **overrides)

        # The parent set replay_policy to something; it is never read here
        # because _pick is overridden. Left alone rather than mutated so
        # summary() still reports what the caller asked for.
        self.valence_decay = valence_decay
        self.valence_floor = valence_floor
        self.valence_pool = valence_pool

        # Replaces the parent's list and deque. A plain list for the buffer
        # because a deque evicts the OLDEST on overflow, which is exactly
        # the policy being tested against.
        self.anchor = []
        self.buffer = []

        self.vstats = dict(stored=0, evicted_low=0, revalued=0, faded=0,
                           unscorable=0)

    # ---------- storage ----------

    def _birth_value(self):
        """The surprise already computed for the item being stored.

        observe() appends the adjusted surprise to history immediately after
        scoring, and every branch that calls _remember does so AFTER that
        append. So history[-1] is this item's own surprise and no extra
        forward pass is needed.

        The exception is the unscorable branch, which returns before the
        append and never stores. If history is somehow empty, fall back to
        1.0 so a fresh unit is not born at the floor and starved of replay
        before it has ever been evaluated.
        """
        if self.history:
            return float(self.history[-1])
        return 1.0

    def _store(self, unit):
        value = max(self._birth_value(), self.valence_floor)
        entry = [self.stats["seen"], unit, value]

        if len(self.anchor) < self.cfg["anchor_size"]:
            self.anchor.append(entry)
            self.vstats["stored"] += 1
            return

        self.buffer.append(entry)
        self.vstats["stored"] += 1

        if len(self.buffer) > self.cfg["buffer_size"]:
            worst = min(range(len(self.buffer)),
                        key=lambda i: self.buffer[i][2])
            self.buffer.pop(worst)
            self.vstats["evicted_low"] += 1

    # ---------- selection ----------

    def _pick_entries(self, pool, k):
        """Highest value wins, from a random subset.

        Ranking the whole pool would make the best units the only units
        replayed, which starves everything else and collapses the buffer to
        a handful of permanently hard cases. Sampling valence_pool
        candidates first keeps the budget pointed at what matters while
        still giving everything a path back in.
        """
        k = min(k, len(pool))
        if k <= 0:
            return []
        if k >= len(pool):
            return list(pool)
        n = min(len(pool), max(k, self.valence_pool))
        cand = self._rng.sample(pool, n)
        cand.sort(key=lambda e: -e[2])
        return cand[:k]

    def _pick(self, pool, k):
        """Parent-compatible signature, used by nothing here.

        Kept so an external caller that reaches for _pick gets units rather
        than entries and does not silently receive a different shape.
        """
        return [e[1] for e in self._pick_entries(pool, k)]

    # ---------- the update ----------

    def _revalue(self, entry):
        """Move value toward the model's current loss on the unit.

        This is the whole mechanism. A unit that stays hard keeps its value
        and keeps earning replay. A unit the model has absorbed decays
        toward the floor and stops competing for budget, and is the first
        thing dropped when the buffer overflows.

        Scored AFTER the replay, so the value reflects what the unit is
        worth going forward rather than what it was worth a moment ago.
        """
        s, _ = self.backend.score(self._first(entry[1]))
        if s is None:
            self.vstats["unscorable"] += 1
            return
        old = entry[2]
        blended = self.valence_decay * old + (1.0 - self.valence_decay) * s
        entry[2] = max(blended, self.valence_floor)
        self.vstats["revalued"] += 1
        if entry[2] < old:
            self.vstats["faded"] += 1

    def _consolidate(self):
        pool = self.anchor + self.buffer
        if not pool:
            return
        for entry in self._pick_entries(pool, self.cfg["rehearse_count"]):
            self.stats["rehearsals"] += self._replay(entry[1])
            self._revalue(entry)

    # ---------- introspection ----------

    def values(self):
        pool = self.anchor + self.buffer
        return sorted((e[2] for e in pool), reverse=True)

    def summary(self):
        s = super().summary()
        s.update(self.vstats)
        vals = self.values()
        if vals:
            s["valence_max"] = vals[0]
            s["valence_median"] = vals[len(vals) // 2]
            s["valence_min"] = vals[-1]
            s["valence_at_floor"] = sum(
                1 for v in vals if v <= self.valence_floor * 1.001)
        return s
