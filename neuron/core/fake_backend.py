"""A backend with no machine learning in it, to prove the stability layer is
genuinely model-agnostic.

The "model" is a dict from item to a number between 0 and 1 representing how
well it is known. score() returns the inverse. update() moves it toward 1.
Coherence is faked by a marker in the item text.

If the gate, veto, rehearsal and guard all behave correctly against this, the
stability layer has no hidden dependency on transformers.
"""

import random


class FakeBackend:
    def __init__(self, seed=0, damage_per_update=0.0):
        self.known = {}
        self.rng = random.Random(seed)
        # Simulates a model that degrades as it learns, so the guard can fire.
        self.damage_per_update = damage_per_update
        self.damage = 0.0

    def score(self, item):
        base = 1.0 - self.known.get(item, 0.0)
        surprise = base * 5.0 + self.damage
        coherence = 3.0 if "JUNK" in str(item) else 0.2
        return surprise, coherence

    def update(self, item, steps):
        for _ in range(steps):
            cur = self.known.get(item, 0.0)
            self.known[item] = min(1.0, cur + 0.25)
            self.damage += self.damage_per_update
        return (1.0 - self.known[item]) * 5.0

    def snapshot(self):
        return (dict(self.known), self.damage)

    def restore(self, state):
        self.known, self.damage = dict(state[0]), state[1]

    def on_rollback(self):
        self.damage_per_update *= 0.5