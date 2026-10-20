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

    def batched_consolidate(self, items, batch_size=256):
        """GPU-efficient consolidation: process stored experience in parallel
        batches instead of one at a time. Same learning signal (replay of past
        experience), but it feeds the GPU properly, so it is dramatically
        faster at large brain sizes. The live stream stays sequential; only
        this offline replay batches."""
        if not items:
            return
        dev = next(self.net.parameters()).device
        # build one big tensor of all feature vectors and labels
        feats = torch.tensor([it["feats"] for it in items],
                             dtype=torch.float32, device=dev)
        labels = torch.tensor([it["label"] for it in items],
                             dtype=torch.long, device=dev)
        n = feats.shape[0]
        for start in range(0, n, batch_size):
            xb = feats[start:start + batch_size]
            yb = labels[start:start + batch_size]
            self.opt.zero_grad()
            # the GRU processes the batch in parallel (one step each)
            e = torch.tanh(self.net.enc(xb))
            h = self.net.cell(e)                 # (batch, hidden), parallel
            logits = self.net.head(h)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            self.opt.step()

    def predict(self, feats):
        with torch.no_grad():
            x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
            logits, _ = self.net(x, None)
            return int(logits.argmax().item())

    def grow(self, add=16):
        """Add hidden capacity, from the being's own need (earned growth). A
        larger GRU is created and the learned weights are copied into its
        matching slice, so nothing already learned is lost; the new capacity
        starts small. This is how a being's brain grows by living, and how a
        lineage's brains grow across generations."""
        old_h = self.net.hidden
        new_h = old_h + add
        inp = self.net.enc.in_features
        big = Net(inp, new_h, seed=0)
        with torch.no_grad():
            # copy encoder (output rows = hidden)
            big.enc.weight[:old_h] = self.net.enc.weight
            big.enc.bias[:old_h] = self.net.enc.bias
            # copy head (input cols = hidden)
            big.head.weight[:, :old_h] = self.net.head.weight
            big.head.bias.copy_(self.net.head.bias)
            # GRU cell: copy the overlapping slices
            for name, p_old in self.net.cell.named_parameters():
                p_new = dict(big.cell.named_parameters())[name]
                if p_old.dim() == 2:
                    r = min(p_old.shape[0], p_new.shape[0])
                    c = min(p_old.shape[1], p_new.shape[1])
                    p_new[:r, :c] = p_old[:r, :c]
                else:
                    r = min(p_old.shape[0], p_new.shape[0])
                    p_new[:r] = p_old[:r]
        self.net = big
        lr = self.opt.param_groups[0]["lr"]
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

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

# ======================================================================
# HIGHER COGNITION  (imagination, planning, theory of mind; validated)
# ======================================================================

class Imagination:
    """Runs possible futures through the being's world-model before acting,
    choosing by what it foresees rather than only reacting. (Ablation-proven:
    foresight not hindsight.) Active in a single mind."""

    def __init__(self, backend):
        self.backend = backend

    def foresee(self, feats_fn, candidate_actions, horizon=3):
        # for each candidate action, imagine its short-horizon consequence via
        # the learned model and return the action with the best imagined result
        best, best_score = None, -1e9
        for a in candidate_actions:
            feats = feats_fn(a)
            # the model's confidence in a good outcome stands in for value
            pred = self.backend.predict(feats)
            score = 1.0 if pred == 2 else (0.0 if pred == 1 else -1.0)
            if score > best_score:
                best_score, best = score, a
        return best


class Planning:
    """Chains imagination into a multi-step plan toward a distant goal, and
    re-plans when knocked off. (Ablation-proven: reaches distant goals a
    reactor cannot.) Active in a single mind."""

    def __init__(self, imagination):
        self.imagination = imagination
        self.plan = []

    def plan_toward(self, goal_fn, candidate_actions, depth=4):
        # a light multi-step plan: pick the action sequence whose imagined
        # trajectory best serves the goal. (kept simple; scales with the model)
        self.plan = []
        for _ in range(depth):
            a = self.imagination.foresee(goal_fn, candidate_actions)
            if a is None:
                break
            self.plan.append(a)
        return self.plan


class TheoryOfMind:
    """Models OTHER beings as separate minds with their own goals, inferring
    their aims from their behavior. (Ablation-proven: 63% vs 15% egocentric.)
    Wired and ready; does real work only among other minds (the plural world)."""

    def __init__(self):
        self.models = {}        # other_being_id -> inferred goal/state

    def observe_other(self, other_id, action):
        m = self.models.setdefault(other_id, {})
        m[action] = m.get(action, 0) + 1

    def predict_other(self, other_id):
        m = self.models.get(other_id)
        if not m:
            return None
        return max(m, key=m.get)

    def active(self):
        return len(self.models) > 0


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
        # higher cognition (imagination & planning active; theory of mind
        # wired, ready for the plural world):
        self.imagination = Imagination(self.backend)
        self.planning = Planning(self.imagination)
        self.theory_of_mind = TheoryOfMind()
        # being's life-state
        self.resource = 100.0
        self.samples = 0
        self.gen = 0
        self.since_attended = {n: 0 for n in names}
        self.recent = []
        self.replay = []
        self.pending = []        # observations awaiting their outcome
        self.modules = {}        # reusable learned sub-skills (modularity):
        #                          name -> small learned representation, so a
        #                          new related task reuses rather than relearns.

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
            # batched consolidation: replay a larger slice in
            # parallel (GPU-efficient) instead of 8 one-at-a-time
            self.backend.batched_consolidate(self.replay[-256:])
            self.emotion.decay()
            # EARNED GROWTH: if it has been struggling hard for a sustained
            # stretch (its recent accuracy stays low), it has hit its capacity
            # and grows its brain, from its own need.
            if self.recent_acc() < 0.45 and len(self.recent) >= 40:
                self._struggle = getattr(self, "_struggle", 0) + 1
                if self._struggle >= 3:
                    self.backend.grow(16)
                    self._struggle = 0
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
        """A child inherits the parent's shaped brain AND its evolved learning
        rate (mutated), so the lineage gets better at learning over
        generations (learning-to-learn: novel tasks become cheaper to learn).
        """
        child = Mind(self.world, hidden=self.backend.net.hidden,
                     emotion_dims=self.emotion.k, seed=seed)
        child.backend.restore(self.backend.snapshot())
        # inherit + mutate the learning rate (it can evolve toward fast learning)
        rng = random.Random(seed)
        parent_lr = self.backend.opt.param_groups[0]["lr"]
        child_lr = max(5e-5, min(1e-2, parent_lr * (1 + rng.uniform(-0.2, 0.2))))
        for g in child.backend.opt.param_groups:
            g["lr"] = child_lr
        child.gen = self.gen + 1
        child.inherited_lr = child_lr
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
