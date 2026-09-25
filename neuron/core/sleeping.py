"""Consolidation as a separate phase, rather than interleaved with living.

WHAT SLEEP IS, beyond replay. Biological sleep replays the day, prunes weak
connections, strengthens important ones, and recombines fragments. This repo
tested recombination (dreaming.py, 20 Sep) and it did not help. Two parts
are still untouched, and this file is about the one nobody has looked at:
the SHAPE of consolidation in time.

Right now this system consolidates every 5 items while the stream keeps
arriving. Living and consolidating are interleaved at the finest possible
grain. Nothing sleeps. An animal does the opposite: it lives for a long
block with no consolidation at all, then stops taking input entirely and
does a concentrated block of consolidation on what it already has.

That is a SCHEDULE difference, not a budget difference, and it has never
been tested here. The same total replay, spent in a different shape.

WHY IT MIGHT MATTER, and both directions are plausible:

  SLEEP WINS      During a sleep block the live stream is not competing
                  for the weights, so replay is not immediately overwritten
                  by whatever arrived next. Consolidation gets to finish.
                  The system also holds a full block of experience before
                  consolidating it, so what it replays is chosen from more
                  material.

  SLEEP LOSES     Going 1,000 items with no replay means 1,000 items of
                  drift with nothing protecting earlier material, and a
                  concentrated block afterwards may be too late to recover
                  what was lost during the waking period. The interleaved
                  schedule protects continuously, which is why it was
                  chosen in the first place.

There is no obvious answer, which is what makes it worth running.

HOW IT WORKS. The layer suppresses its own consolidation during waking,
so nothing is replayed no matter what the config says. sleep(n) lifts the
suppression, runs n consolidations back to back with no new input, and
re-suppresses. The caller decides when to sleep and for how long, so the
schedule lives in the experiment rather than in the layer.

Contiguous storage is inherited from DreamLayer and defaults ON here,
because it was measurably better on 20 Sep: storing every item rather than
only the gate-rejected half moved the first phase from 60.7% to 65.3% and
the last from 84.0% to 99.8%.

Dreaming is inherited too and defaults OFF, so this measures schedule
alone.
"""

from dreaming import DreamLayer


class SleepLayer(DreamLayer):
    """Consolidation happens only when the caller says to sleep.

    The parent's own scheduling (rehearse_per_item, rehearse_every) is
    neutralised rather than reconfigured, so a caller that forgets to
    change those settings still gets the intended behaviour instead of a
    silent mixture of both schedules.
    """

    def __init__(self, backend, canary=None, seed=0, verbose=False,
                 contiguous=True, **overrides):
        super().__init__(backend, canary=canary, seed=seed,
                         verbose=verbose, dream_fraction=0.0,
                         contiguous=contiguous, **overrides)
        self._asleep = False
        self.slstats = dict(sleeps=0, consolidations=0, awake_items=0,
                            replays_asleep=0)

    def _consolidate(self):
        """Suppressed while awake. This is the whole mechanism.

        The parent calls this on its own schedule and the call is simply
        dropped, so waking life never replays anything regardless of how
        the layer was configured.
        """
        if not self._asleep:
            return
        before = self.stats["rehearsals"]
        super()._consolidate()
        self.slstats["consolidations"] += 1
        self.slstats["replays_asleep"] += (
            self.stats["rehearsals"] - before)

    def observe(self, item):
        self.slstats["awake_items"] += 1
        return super().observe(item)

    def sleep(self, rounds):
        """Stop taking input and consolidate, rounds times.

        The backend's hidden state is reset at the start and restored to
        None at the end: a sleep block is not a continuation of the waking
        trajectory, and carrying state across the boundary would make the
        first replayed sequence look like it followed the last lived
        moment, which it did not.
        """
        if rounds <= 0:
            return 0
        if hasattr(self.backend, "reset_state"):
            self.backend.reset_state()

        self._asleep = True
        before = self.stats["rehearsals"]
        try:
            for _ in range(rounds):
                self._consolidate()
        finally:
            self._asleep = False

        if hasattr(self.backend, "reset_state"):
            self.backend.reset_state()
        self.slstats["sleeps"] += 1
        return self.stats["rehearsals"] - before

    def summary(self):
        s = super().summary()
        s.update(self.slstats)
        return s
