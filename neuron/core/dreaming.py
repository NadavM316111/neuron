"""Replay of experience that never happened.

WHAT SLEEP ACTUALLY DOES, and which part is missing here. Biological sleep
replays the day's experience, prunes weak connections, strengthens
important ones, and RECOMBINES stored fragments into sequences that never
occurred. This repo has replay. It has nothing that recombines.

Recombination is the interesting one because it is the only mechanism that
can produce experience the system never had. Everything else redistributes
attention over what did happen.

THE CONSTRAINT THAT SHAPES THE DESIGN. You cannot invent outcomes. A dream
that makes up what happens when the agent steps on a key is training on
fiction, and the model learns the fiction. So every individual moment in a
dream here is a REAL moment with its real outcome, and only the ORDER is
invented. Real fragments, impossible trajectory.

CONTIGUITY, and this is why the first run of this experiment failed.

StabilityLayer calls _remember ONLY on the branches where the gate REJECTS
an item: warmup, below_bar, unremarkable, already_known. Anything the gate
accepts goes to the weights and is never stored. At top_fraction=0.50 that
is roughly half of everything.

So a stored "sequence" of 10 was never 10 consecutive moments. It was 10
non-adjacent rejected moments spanning about 20 steps of the life, with
gaps wherever the gate fired. The stored runs were ALREADY partially
scrambled.

That is why the first dream run came back with scrambling doing no damage:
there was little original order left to destroy. It also means the +15.5
sequence-replay result from August was measured on units that are not
really trajectories, which is worth knowing on its own.

contiguous=True stores EVERY item into the run, learned or rejected, so a
unit is an actual contiguous slice of the life. The gate still decides what
the weights learn from; it no longer decides what a sequence is made of.

Ordering is preserved without touching stability.py: each observe() handles
exactly one item, so catching the learned ones immediately after the parent
returns puts them in the run at the right position.

THREE WAYS TO RECOMBINE, and the third exists to fail:

  splice       first half of one stored run, second half of another. Local
               order survives on both sides of the join.
  interleave   alternate moments from two runs. Almost no original order
               survives, though every moment is still real.
  scramble     shuffle within a single run. THE KNOWN-BAD CONTROL.
               Scrambled replay damaged shared knowledge on 21 Aug (96.2%
               against 99.8%). If scramble does not lose, the experiment
               is not measuring what it claims.

dream_fraction is how much of consolidation is dreaming rather than
remembering. At 0.0 this layer is StabilityLayer plus whatever contiguous
is set to, which is how the veridical control arm is built.
"""

from stability import StabilityLayer


MODES = ("splice", "interleave", "scramble")


class DreamLayer(StabilityLayer):
    """StabilityLayer that sometimes replays recombined sequences.

    Two behaviour changes, both optional and both off by default so
    nothing measured previously moves:

      contiguous       store every item into the run, not only the ones
                       the gate rejected.
      dream_fraction   replace some of consolidation with recombined units.

    A dream is built at replay time and never stored, so it cannot
    accumulate or crowd out real memories.
    """

    def __init__(self, backend, canary=None, seed=0, verbose=False,
                 dream_fraction=0.5, dream_mode="splice", contiguous=False,
                 **overrides):
        super().__init__(backend, canary=canary, seed=seed,
                         verbose=verbose, **overrides)
        if dream_mode not in MODES:
            raise ValueError(f"dream_mode must be one of {MODES}")
        if not 0.0 <= dream_fraction <= 1.0:
            raise ValueError("dream_fraction must be between 0 and 1")

        self.dream_fraction = dream_fraction
        self.dream_mode = dream_mode
        self.contiguous = contiguous
        self.dstats = dict(dreams=0, memories=0, too_short=0,
                           no_partner=0, stored_learned=0,
                           stored_rejected=0)

    # ---------- contiguity ----------

    def observe(self, item):
        """Parent decision, plus storage of the items it learned from.

        The parent stores on every rejection branch and returns without
        storing when it learns. Catching the learned case here appends it
        to the run immediately after, which is the correct position because
        one call handles one item.

        The gate still decides what the WEIGHTS learn from. It no longer
        decides what a SEQUENCE is made of, which it was doing by accident.
        """
        d = super().observe(item)
        if d.get("learned"):
            self.dstats["stored_learned"] += 1
            if self.contiguous:
                self._remember(item)
        elif d.get("reason") != "unscorable":
            self.dstats["stored_rejected"] += 1
        return d

    def contiguity(self):
        """Share of the life that made it into stored runs.

        Near 0.5 without contiguous, since the gate rejects about half at
        top_fraction=0.50. Near 1.0 with it. Read this before trusting any
        conclusion about order: if the runs are not contiguous, scrambling
        them cannot do much damage and the control cannot bite.
        """
        a = self.dstats["stored_learned"]
        r = self.dstats["stored_rejected"]
        total = a + r
        if not total:
            return 0.0
        return (total if self.contiguous else r) / total

    # ---------- building a dream ----------

    def _splice(self, a, b):
        """First half of one run, second half of another.

        Both halves keep their internal order, so most adjacent pairs are
        real pairs. Only the join is invented.
        """
        cut_a = max(1, len(a) // 2)
        cut_b = max(1, len(b) // 2)
        return list(a[:cut_a]) + list(b[cut_b:])

    def _interleave(self, a, b):
        """Alternate moments from two runs. Far more aggressive: almost no
        original ordering survives, though every moment is still real."""
        out = []
        for i in range(max(len(a), len(b))):
            if i < len(a):
                out.append(a[i])
            if i < len(b):
                out.append(b[i])
        return out[:max(len(a), len(b))]

    def _scramble(self, a, _b):
        """Shuffle within one run. The known-bad control."""
        out = list(a)
        self._rng.shuffle(out)
        return out

    def _dream(self, pool):
        """Build one recombined unit, or None if it cannot.

        Returns None rather than quietly falling back to a real memory, so
        the caller counts the failure. A dream layer that silently stopped
        dreaming would look exactly like one that was working.
        """
        runs = [u for _, u in pool if isinstance(u, list) and len(u) >= 2]
        if not runs:
            self.dstats["too_short"] += 1
            return None

        a = self._rng.choice(runs)
        if self.dream_mode == "scramble":
            return self._scramble(a, None)

        if len(runs) < 2:
            self.dstats["no_partner"] += 1
            return None
        b = self._rng.choice([r for r in runs if r is not a] or runs)

        if self.dream_mode == "splice":
            return self._splice(a, b)
        return self._interleave(a, b)

    # ---------- consolidation ----------

    def _consolidate(self):
        pool = self.anchor + list(self.buffer)
        if not pool:
            return
        for unit in self._pick(pool, self.cfg["rehearse_count"]):
            if self.dream_fraction and \
                    self._rng.random() < self.dream_fraction:
                dreamt = self._dream(pool)
                if dreamt is not None:
                    self.stats["rehearsals"] += self._replay(dreamt)
                    self.dstats["dreams"] += 1
                    continue
            self.stats["rehearsals"] += self._replay(unit)
            self.dstats["memories"] += 1

    # ---------- introspection ----------

    def summary(self):
        s = super().summary()
        s.update(self.dstats)
        total = self.dstats["dreams"] + self.dstats["memories"]
        s["dream_share"] = (self.dstats["dreams"] / total) if total else 0.0
        s["dream_mode"] = self.dream_mode
        s["contiguous"] = self.contiguous
        s["contiguity"] = self.contiguity()
        return s
