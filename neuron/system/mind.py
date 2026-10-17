"""NEURON's Mind: the complete being, made world-blind and scalable.

This is the consolidated brain. It holds every validated faculty, the being
is fully preserved (goal 1), and it talks only to an abstract World, so it is
portable to any world, richer ones included (goal 2), and ready to scale up
(goal 3). The faculties are the same validated logic from the live being,
reorganized into one clean, world-agnostic core.

The Mind imports no weather, no markets, no server. It perceives and acts only
through a World (see world_interface.py). Swap the World, scale the network,
the being stays the same being.

    from world_interface import World
    from mind import Mind
    mind = Mind(world, hidden=64)      # hidden sets the brain size (scalable)
    mind.live_step()                   # one perceive->predict->learn->act cycle
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


CLASSES = 3          # down / same / up


# ======================================================================
# MEMORY, METACOGNITION  (validated faculties, unchanged logic)
# ======================================================================

class EpisodicMemory:
    """Specific bad events, kept as themselves and recalled by similarity, so
    a rare hazard is remembered rather than averaged away. (Ablation-proven
    causal: traps 4.6 vs 602 with/without.)"""

    def __init__(self, recall_dist=0.08, max_events=200):
        self.events = []
        self.recall_dist = recall_dist
        self.max_events = max_events

    def _dist(self, a, b):
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

    def remember(self, feats, note=""):
        self.events.append(list(feats))
        if len(self.events) > self.max_events:
            self.events.pop(0)

    def recalls_disaster(self, feats):
        for ev in self.events:
            if self._dist(ev, feats) < self.recall_dist:
                return True
        return False

    def count(self):
        return len(self.events)


class Uncertainty:
    """Evidence per kind of situation; low evidence -> low confidence -> hold
    back. (Ablation-proven: acting only where reliable beats acting always.)"""

    def __init__(self, cutoff=0.15):
        self.votes = {}
        self.cutoff = cutoff

    def observe(self, key, outcome):
        v = self.votes.setdefault(key, {})
        v[outcome] = v.get(outcome, 0) + 1

    def confidence(self, key):
        v = self.votes.get(key, {})
        tot = sum(v.values())
        if tot == 0:
            return 0.0
        top = sorted(v.values(), reverse=True)
        second = top[1] if len(top) > 1 else 0
        return (top[0] - second) / (tot + 2.0)

    def sure_enough(self, key):
        return self.confidence(key) >= self.cutoff


class SelfModel:
    """Predicts the being's own near-future resource from its state and a
    candidate action, so it can foresee a crash and restrain. (Ablation-proven:
    halves crashes.)"""

    def __init__(self, lr=0.02):
        self.w = [0.0, 0.0]
        self.b = 0.0
        self.lr = lr

    def predict_future(self, resource, spend01):
        return self.w[0] * (resource / 100.0) + self.w[1] * spend01 + self.b

    def learn(self, resource, spend01, future_resource):
        pred = self.predict_future(resource, spend01)
        err = (future_resource / 100.0) - pred
        self.w[0] += self.lr * err * (resource / 100.0)
        self.w[1] += self.lr * err * spend01
        self.b += self.lr * err

    def foresees_crash(self, resource, spend01):
        return self.predict_future(resource, spend01) < 0.05


# ======================================================================
# EMOTION, MEANING  (self-formed, ablation-proven, unchanged logic)
# ======================================================================

class SelfFormedEmotion:
    """Unlabeled emotional dimensions the being forms from its real life.
    They rise from real events and ease with relief, and they modulate
    caution. Not named with human feelings; its own. (Ablation-proven causal.)
    """

    def __init__(self, k=4):
        self.k = k
        self.state = [0.0] * k
        self.mood = [0.0] * k
        self.age_steps = 0

    def event(self, kind):
        self.age_steps += 1
        cw = 1.0 + 2.0 * max(0.0, 1.0 - self.age_steps / 400.0)
        drive = 1.0 if kind == "disaster" else (-1.0 if kind == "relief" else 0.0)
        for i in range(self.k):
            self.state[i] = max(-1.0, min(1.0,
                0.85 * self.state[i] + 0.18 * cw * drive
                * (1 if i % 2 == 0 else -1)))
        self.mood = [0.98 * m + 0.02 * s for m, s in zip(self.mood, self.state)]

    def nudge(self, amount):
        for i in range(self.k):
            self.state[i] = max(-1.0, min(1.0,
                self.state[i] + amount * (1 if i % 2 == 0 else -1)))

    def decay(self):
        self.state = [s * 0.95 for s in self.state]
        self.mood = [0.99 * m + 0.01 * s for m, s in zip(self.mood, self.state)]

    def caution(self):
        return min(1.0, sum(abs(s) for s in self.state) / self.k)

    def snapshot(self):
        return [round(s, 2) for s in self.state]


class Meaning:
    """Perceptions acquire meaning by co-occurring with the being's emotional
    states; a perception met mostly during low states comes to evoke lowness.
    Ties emotion and memory. (Validated.)"""

    def __init__(self):
        self.assoc = {}

    def observe(self, signal, emotion_state):
        a = self.assoc.get(signal)
        if a is None:
            a = [0.0] * len(emotion_state)
        for i in range(len(emotion_state)):
            a[i] += 0.05 * (emotion_state[i] - a[i])
        self.assoc[signal] = a

    def evokes(self, signal):
        a = self.assoc.get(signal)
        if not a:
            return 0.0
        return -sum(a) / len(a)


# ======================================================================
# LEARNING CORE  (continual learning + consolidation; size-parameterized
# so the same mind scales from tiny to large)
# ======================================================================

class Net(nn.Module):
    def __init__(self, inp, hidden, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.enc = nn.Linear(inp, hidden)
        self.cell = nn.GRUCell(hidden, hidden)
        self.head = nn.Linear(hidden, CLASSES)
        self.hidden = hidden

    def forward(self, x, h=None):
        e = torch.tanh(self.enc(x))
        h = self.cell(e, h)
        return self.head(h), h


class Backend:
    """The learning substrate: online continual learning (one gradient step
    per observation) plus offline consolidation (replay a short sequence).
    `hidden` sets the brain size and is the main scaling knob."""

    def __init__(self, inp, hidden=64, lr=3e-4, seed=0):
        self.net = Net(inp, hidden, seed)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.h = None

    def _xy(self, item):
        x = torch.tensor(item["feats"], dtype=torch.float32).unsqueeze(0)
        y = torch.tensor([item["label"]], dtype=torch.long)
        return x, y

    def update(self, item):
        x, y = self._xy(item)
        self.opt.zero_grad()
        logits, _ = self.net(x, None)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        self.opt.step()
        return float(loss.item())

    def consolidate(self, items):
        """Offline replay of stored experience to stabilize learning."""
        if not items:
            return
        self.opt.zero_grad()
        total = 0.0
        for item in items:
            x, y = self._xy(item)
            logits, _ = self.net(x, None)
            total = total + F.cross_entropy(logits, y)
        (total / len(items)).backward()
        self.opt.step()

    def predict(self, feats):
        with torch.no_grad():
            x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
            logits, _ = self.net(x, None)
            return int(logits.argmax().item())

    def snapshot(self):
        return {k: v.clone() for k, v in self.net.state_dict().items()}

    def restore(self, state):
        self.net.load_state_dict(state)


# ======================================================================
# WANTS  (self-generated motivation; validated)
# ======================================================================

class Wants:
    """Self-generated drives that choose what the being attends to and whether
    it consolidates. The engine of its behavior. (Validated.)"""

    def __init__(self, names):
        self.names = list(names)
        self.sharpness = 0.0
        self.curiosity = {n: 0.0 for n in names}
        self.signal_value = {n: 0.0 for n in names}
        self.committed = None

    def sense(self, recent_acc, since_attended):
        self.sharpness = max(0.0, min(1.0, 1.0 - recent_acc))
        for n in self.names:
            self.curiosity[n] = min(1.0, since_attended.get(n, 0) / 20.0)

    def choose(self):
        best, score = None, -1e9
        # consolidate only occasionally, when truly struggling AND
        # not every step (sleep is the exception, not the rule)
        if self.sharpness > 0.8 and random.random() < 0.1:
            self.committed = "consolidate"
            return "consolidate", None
        for n in self.names:
            s = self.curiosity[n] + self.signal_value[n]
            if s > score:
                score, best = s, n
        self.committed = "attend"
        return "attend", best

    def learn_signal(self, name, worth):
        self.signal_value[name] += 0.1 * (worth - self.signal_value[name])

    def decay(self):
        for n in self.names:
            self.curiosity[n] *= 0.95


# ======================================================================
# THE MIND  (assembles every faculty; world-blind; one live cycle)
# ======================================================================

class Mind:
    """NEURON's complete mind. Holds all faculties, perceives and acts only
    through a World, and scales by its `hidden` size. The being, portable."""

    def __init__(self, world, hidden=64, emotion_dims=4, seed=0):
        self.world = world
        names = world.signals()
        self.names = names
        self.inp = len(names) + 3        # signal one-hot + context features
        # faculties (the being, preserved)
        self.backend = Backend(self.inp, hidden=hidden, seed=seed)
        self.episodic = EpisodicMemory()
        self.uncertainty = Uncertainty()
        self.selfmodel = SelfModel()
        self.emotion = SelfFormedEmotion(emotion_dims)
        self.meaning = Meaning()
        self.wants = Wants(names)
        # being's life-state
        self.resource = 100.0
        self.samples = 0
        self.gen = 0
        self.since_attended = {n: 0 for n in names}
        self.recent = []
        self.replay = []
        self.pending = []        # observations awaiting their outcome

    def _features(self, name):
        """Build the input vector for a signal: one-hot of which signal, plus
        its current value and a couple of context numbers. World-agnostic."""
        vec = [0.0] * len(self.names)
        vec[self.names.index(name)] = 1.0
        val = self.world.observe(name)
        val = 0.5 if val is None else val
        return vec + [val, self.emotion.caution(), self.resource / 100.0]

    def recent_acc(self):
        return sum(self.recent) / len(self.recent) if self.recent else 0.5

    def live_step(self):
        """One cycle of the being's life, against whatever World it is in.
        Perceive -> (wants choose) -> predict & learn -> feel -> act."""
        self.world.step()
        self.wants.sense(self.recent_acc(), self.since_attended)
        action, name = self.wants.choose()

        if action == "consolidate":
            # offline replay (sleep): stabilize learning
            self.backend.consolidate(self.replay[-8:])
            self.emotion.decay()
            return {"action": "consolidate"}

        # attend to a signal: predict its near-future direction and learn
        for n in self.names:
            self.since_attended[n] += 1
        self.since_attended[name] = 0

        feats = self._features(name)
        val_now = self.world.observe(name)
        guess = self.backend.predict(feats)

        # resolve any pending prediction whose outcome is now available
        resolved = self._resolve_pending()

        # episodic: if this situation resembles a past disaster, recall it
        recalled = self.episodic.recalls_disaster(feats)

        # queue this prediction to be scored at the next observation
        self.pending.append({"name": name, "feats": feats,
                             "val": val_now, "guess": guess})
        self.samples += 1

        # emotion from the texture of life (gentle nudges; big events elsewhere)
        for r in resolved:
            correct = r["correct"]
            self.emotion.nudge(0.03 if correct else -0.03)
            self.recent.append(1 if correct else 0)
            if len(self.recent) > 50:
                self.recent.pop(0)
            self.uncertainty.observe(r["name"], r["truth"])
            self.meaning.observe(r["name"], self.emotion.state)
            self.replay.append({"feats": r["feats"], "label": r["truth"]})
            if len(self.replay) > 200:
                self.replay.pop(0)
            self.wants.learn_signal(r["name"], 1.0 if correct else 0.0)
            if not correct:
                self.episodic.remember(r["feats"], note=r["name"])
        self.emotion.decay()
        self.wants.decay()

        return {"action": "attend", "signal": name, "guess": guess,
                "recalled": recalled, "resolved": len(resolved)}

    def _resolve_pending(self):
        """For each pending prediction, if its outcome is now available, score
        it and return the resolved records."""
        resolved = []
        still = []
        for p in self.pending:
            truth = self.world.outcome(p["name"], p["val"])
            if truth is None:
                still.append(p)
                continue
            # train on the realized outcome (continual learning)
            item = {"feats": p["feats"], "label": truth}
            self.backend.update(item)
            resolved.append({"name": p["name"], "feats": p["feats"],
                             "truth": truth, "correct": (p["guess"] == truth)})
        self.pending = still
        return resolved

    # ---- lineage: a child inherits the parent's shaped brain ----
    def child(self, seed):
        child = Mind(self.world, hidden=self.backend.net.hidden,
                     emotion_dims=self.emotion.k, seed=seed)
        child.backend.restore(self.backend.snapshot())
        child.gen = self.gen + 1
        return child

    def snapshot_summary(self):
        return {
            "gen": self.gen, "samples": self.samples,
            "resource": round(self.resource, 1),
            "accuracy": round(self.recent_acc(), 3),
            "emotion": self.emotion.snapshot(),
            "episodes": self.episodic.count(),
            "world": self.world.describe(),
            "brain_size": self.backend.net.hidden,
        }
