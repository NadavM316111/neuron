"""Milestone 4: does it keep improving past 100,000 steps?

The plan: "It keeps improving past 100,000 steps. No plateau, no decay.
This is the one that supports the product claim."

That claim is "it never graduates" — an AI that keeps getting better the
longer you have it. If competence plateaus at 20,000 steps, or peaks and
decays, the product is a thing that gets good and then stops, which is a
different and much less interesting proposition.

A long life: ten phases of 10,000 steps, cycling through three contradictory
rule sets so the world keeps changing the way a real one would. Layout is
fixed so the ONLY thing that changes is what the world does.

Measured every 5,000 steps on held-out streams from EVERY rule set, scored
on contested events only, so you see:

  current   how well it handles the rules in force right now
  past      how well it still handles rules it has lived under before
  overall   the mean across all three

Two arms, because the question is whether the protection holds up over a
long life, not whether it exists:
  online     one pass, no protection. Expected to be pure recency forever.
  sequences  the full stability layer with sequence replay.

Runs unattended. Progress prints as it goes, so it can be watched with
tail -f if run detached.
"""

import json
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from world import GridWorld, ACTIONS, EVENTS, DELTA, LOCKED, KEY
from stability import StabilityLayer


RULES = ["keyed", "open", "trap"]
N_PHASES = 10
PHASE_STEPS = 10000            # 100,000 steps total
EPISODE = 200
HIDDEN = 64
LR = 3e-4
SEEDS = [0]                    # one seed: this is a shape question
N_CELLS = 5
TEST_STEPS = 1500
CHECK_EVERY = 5000
SEQ_LEN = 10
SEQ_COUNT = 2

# The life: which rules are in force during each phase.
LIFE = [RULES[i % len(RULES)] for i in range(N_PHASES)]


def obs_dim():
    """No has_key. The model must remember it."""
    return 9 * N_CELLS + len(ACTIONS)


def encode(obs, action):
    v = torch.zeros(obs_dim())
    for i, cell in enumerate(obs["patch"]):
        v[i * N_CELLS + cell] = 1.0
    v[9 * N_CELLS + ACTIONS.index(action)] = 1.0
    return v


class GRU(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.enc = nn.Linear(obs_dim(), HIDDEN)
        self.cell = nn.GRUCell(HIDDEN, HIDDEN)
        self.head = nn.Linear(HIDDEN, len(EVENTS))

    def forward(self, x, h=None):
        e = F.relu(self.enc(x))
        h = self.cell(e, h)
        return self.head(h), h


class Backend:
    def __init__(self, seed=0):
        self.net = GRU(seed)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.grad_steps = 0
        self.h = None
        self._saved = None

    def begin_sequence(self):
        """A replayed run is starting. Park the live state, start clean."""
        self._saved = self.h
        self.h = None

    def _xy(self, item):
        return encode(item[0], item[1]).unsqueeze(0), \
            torch.tensor([EVENTS.index(item[2])])

    def score(self, item):
        self.net.eval()
        with torch.no_grad():
            x, y = self._xy(item)
            logits, _ = self.net(x, self.h)
            return float(F.cross_entropy(logits, y).item()), None

    def update(self, item, steps):
        self.net.train()
        last = None
        for _ in range(steps):
            x, y = self._xy(item)
            logits, h = self.net(x, self.h)
            loss = F.cross_entropy(logits, y)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.h = h.detach()
            self.grad_steps += 1
            last = float(loss.item())
        return last

    def reset_state(self):
        self.h = None
        self._saved = None

    def snapshot(self):
        return {k: v.detach().clone()
                for k, v in self.net.state_dict().items()}

    def restore(self, state):
        self.net.load_state_dict(state)

    def on_rollback(self):
        for g in self.opt.param_groups:
            g["lr"] = max(1e-5, g["lr"] * 0.5)

    def contested_score(self, test, episode=EPISODE):
        """Accuracy on events where the rule sets disagree. Replayed in
        order so the hidden state means something."""
        self.net.eval()
        saved = self.h
        h = None
        hit = seen = 0
        with torch.no_grad():
            for i, item in enumerate(test):
                if i % episode == 0:
                    h = None
                x, y = self._xy(item)
                logits, h = self.net(x, h)
                if item[3]:
                    seen += 1
                    if int(logits.argmax(1).item()) == int(y.item()):
                        hit += 1
        self.h = saved
        return 100.0 * hit / seen if seen else None


def outcome_under(state, action, rules, grid):
    r, c, keys = state
    dr, dc = DELTA[action]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < len(grid) and 0 <= nc < len(grid[0])):
        return "blocked"
    cell = grid[nr][nc]
    if cell == LOCKED:
        if rules == "open":
            return "unlocked"
        if rules == "trap":
            return "blocked"
        return "unlocked" if keys > 0 else "blocked"
    if cell == KEY:
        if rules == "trap":
            return "blocked"
        return "moved" if keys >= 1 else "got_key"
    if cell == 1:
        return "blocked"
    return "moved"


def walk(layout_seed, walk_seed, rules, n, episode=EPISODE, mark=False):
    rng = random.Random(walk_seed)
    world = GridWorld(layout_seed, rules=rules)
    out = []
    for i in range(n):
        if i % episode == 0:
            world.reset()
        obs = world.observe()
        action = rng.choice(ACTIONS)
        state = (world.r, world.c, world.keys)
        _, event = world.step(action)
        if mark:
            others = [outcome_under(state, action, o, world.grid)
                      for o in RULES if o != rules]
            out.append((obs, action, event, any(o != event for o in others)))
        else:
            out.append((obs, action, event))
    return out


def run(mode, seed):
    """Live one long life, checking in every CHECK_EVERY steps."""
    torch.manual_seed(seed)
    b = Backend(seed)
    tests = {r: walk(seed, 9000 + i, r, TEST_STEPS, mark=True)
             for i, r in enumerate(RULES)}

    layer = None
    if mode == "sequences":
        warm = walk(seed, 7777, LIFE[0], 40)
        layer = StabilityLayer(
            b, canary=warm, seed=seed,
            window=200, warmup=30, top_fraction=0.50,
            coherence_veto=1e9, loss_floor=0.02, steps_per_update=1,
            rehearse_per_item=5, rehearse_count=SEQ_COUNT, rehearse_steps=1,
            anchor_size=100, buffer_size=500, sequence_len=SEQ_LEN,
            guard=True, guard_per_item=2000, canary_tolerance=0.5)

    history = []
    step = 0
    t0 = time.time()

    for phase_i, rules in enumerate(LIFE):
        stream = walk(seed, seed * 1000 + phase_i, rules, PHASE_STEPS)
        for i, item in enumerate(stream):
            if step % EPISODE == 0:
                b.reset_state()
            if layer is not None:
                layer.observe(item)
            else:
                b.update(item, 1)
            step += 1

            if step % CHECK_EVERY == 0:
                b.reset_state()
                scores = {r: b.contested_score(tests[r]) for r in RULES}
                lived = set(LIFE[:phase_i + 1])
                past = [scores[r] for r in lived if r != rules]
                rec = dict(step=step, phase=phase_i + 1, rules=rules,
                           current=scores[rules],
                           past=sum(past) / len(past) if past else None,
                           overall=sum(scores.values()) / len(scores),
                           scores=scores,
                           rollbacks=(layer.summary()["rollbacks"]
                                      if layer else 0))
                history.append(rec)
                p = f"{rec['past']:5.1f}%" if rec["past"] is not None else "   -  "
                print(f"  {mode:>9} step {step:>6}  phase {phase_i + 1:>2} "
                      f"({rules:>5})  current {rec['current']:5.1f}%  "
                      f"past {p}  overall {rec['overall']:5.1f}%  "
                      f"({time.time() - t0:.0f}s)")
                b.reset_state()

    out = dict(history=history, grad_steps=b.grad_steps,
               minutes=(time.time() - t0) / 60)
    del b, layer
    return out


if __name__ == "__main__":
    print(f"a long life: {N_PHASES} phases x {PHASE_STEPS} steps = "
          f"{N_PHASES * PHASE_STEPS} total")
    print(f"rules cycle: {' -> '.join(LIFE)}")
    print(f"checking every {CHECK_EVERY} steps on contested events from "
          f"every rule set\n")

    results = {}
    for mode in ["online", "sequences"]:
        print(f"--- {mode} ---")
        results[mode] = run(mode, SEEDS[0])
        print(f"  {results[mode]['minutes']:.1f} min, "
              f"{results[mode]['grad_steps']} gradient steps\n")

    print("=" * 76)
    print("OVERALL competence across all rule sets, over the life")
    print(f"{'step':>7} " + "  ".join(f"{m:>11}" for m in results))
    print("-" * 76)
    n = len(results["online"]["history"])
    for i in range(n):
        row = f"{results['online']['history'][i]['step']:>7} "
        for m in results:
            row += f"  {results[m]['history'][i]['overall']:>10.1f}%"
        print(row)
    print("=" * 76)

    for mode in results:
        h = results[mode]["history"]
        overalls = [r["overall"] for r in h]
        first_half = overalls[:len(overalls) // 2]
        second_half = overalls[len(overalls) // 2:]
        peak = max(overalls)
        peak_at = h[overalls.index(peak)]["step"]
        final = overalls[-1]
        print(f"\n{mode}:")
        print(f"  first half mean  {sum(first_half) / len(first_half):5.1f}%")
        print(f"  second half mean {sum(second_half) / len(second_half):5.1f}%")
        print(f"  peak {peak:.1f}% at step {peak_at}, final {final:.1f}%")
        print(f"  drop from peak   {peak - final:+5.1f}")
        print(f"  rollbacks        {h[-1]['rollbacks']}")

    print("""
MILESTONE 4 asks whether it keeps improving past 100,000 steps.

  second half clearly above first, final near peak -> still improving. The
      product claim holds: it does not graduate.
  second half flat -> it plateaus. Competence caps out and more time buys
      nothing, which changes the proposition considerably.
  final well below peak -> it peaks and decays. That is the worst answer and
      the most important to know, because it means long-term use degrades
      the thing.

Watch "past" in the per-step lines too. Overall can hold steady while the
system quietly cycles — good at whatever is current, losing everything else.
""")
    with open("longrun.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote longrun.json")