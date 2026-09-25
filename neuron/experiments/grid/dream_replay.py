"""Does replaying experience that never happened help?

FIRST ATTEMPT FAILED ITS OWN CONTROL, and the reason was a property of the
architecture nobody had noticed.

StabilityLayer stores a run only from items the gate REJECTS. At
top_fraction=0.50 that is roughly half the stream, so a stored "sequence"
of 10 was 10 non-adjacent moments spanning about 20 steps, with gaps
wherever the gate fired. The stored runs were already partially scrambled,
which is why scrambling them further did no measurable damage and why the
control could not bite.

That also means the +15.5 sequence-replay result from August (21 Aug) was
measured on units that are not really trajectories. Worth knowing
regardless of what happens here.

contiguous=True stores every item into the run, so a unit is an actual
contiguous slice of the life. Every online arm below uses it, including the
control, so the comparison is clean and the only difference between arms is
what happens at replay time.

CONTIGUITY IS PRINTED AND CHECKED. If it is not near 1.0 the fix did not
take and nothing downstream is interpretable.

FIVE ARMS:

  offline      shuffled batches, every phase in every epoch. A ceiling
               rather than a target: replay_budget.py showed nothing
               online reaches it without ceasing to be online.
  veridical    ordinary replay of real contiguous runs. The control.
  splice       half of consolidation replays first-half-of-A plus
               second-half-of-B. Local order survives on both sides.
  interleave   half alternates moments from two runs. Aggressive.
  scramble     half shuffles within one run. THE KNOWN-BAD CONTROL.

READ THE CONTROL FIRST. The script checks contiguity and then scramble
before printing any conclusion.
"""

import json
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from world import (GridWorld, ACTIONS, EVENTS, DELTA, LOCKED, KEY,
                   contradiction_check)
from dreaming import DreamLayer


PHASES = ["keyed", "open", "trap"]
PHASE_STEPS = 6000
EPISODE = 200
HIDDEN = 64
LR = 3e-4
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
N_CELLS = 5
OFFLINE_EPOCHS = 5
OFFLINE_BATCH = 32
TEST_STEPS = 2000
SEQ_LEN = 10
SEQ_COUNT = 2
DREAM_FRACTION = 0.5


def obs_dim():
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

    def update_sequence(self, items, steps):
        """Replay a run IN ORDER with the loss accumulated across it.

        This is what makes sequence replay mean anything: gradient flows
        through the whole trajectory rather than through N unconnected
        single steps. It is also what a dream depends on, since a dream is
        nothing but an altered trajectory. Without it, order is invisible
        to the optimiser and scrambling cannot possibly matter.
        """
        self.net.train()
        done = 0
        for _ in range(steps):
            h = None
            total = 0.0
            self.opt.zero_grad()
            for item in items:
                x, y = self._xy(item)
                logits, h = self.net(x, h)
                total = total + F.cross_entropy(logits, y)
                done += 1
            (total / max(1, len(items))).backward()
            self.opt.step()
            self.grad_steps += len(items)
        return done

    def update_batch(self, items):
        self.net.train()
        x = torch.stack([encode(i[0], i[1]) for i in items])
        y = torch.tensor([EVENTS.index(i[2]) for i in items])
        logits, _ = self.net(x, None)
        loss = F.cross_entropy(logits, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.grad_steps += len(items)
        return float(loss.item())

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

    def evaluate_split(self, test, episode=EPISODE):
        self.net.eval()
        h = None
        hit = {True: 0, False: 0}
        seen = {True: 0, False: 0}
        with torch.no_grad():
            for i, item in enumerate(test):
                if i % episode == 0:
                    h = None
                x, y = self._xy(item)
                logits, h = self.net(x, h)
                c = item[3]
                seen[c] += 1
                if int(logits.argmax(1).item()) == int(y.item()):
                    hit[c] += 1
        return (100.0 * hit[True] / seen[True] if seen[True] else None,
                100.0 * hit[False] / seen[False] if seen[False] else None)


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
                      for o in PHASES if o != rules]
            out.append((obs, action, event, any(o != event for o in others)))
        else:
            out.append((obs, action, event))
    return out


def build_life(seed):
    out = []
    for i, rules in enumerate(PHASES):
        out += walk(seed, seed * 100 + i, rules, PHASE_STEPS)
    return out


def build_tests(seed):
    return {rules: walk(seed, 9000 + seed * 10 + i, rules, TEST_STEPS,
                        mark=True)
            for i, rules in enumerate(PHASES)}


COMMON = dict(
    window=200, warmup=30, top_fraction=0.50,
    coherence_veto=1e9, loss_floor=0.02, steps_per_update=1,
    rehearse_per_item=5, rehearse_count=SEQ_COUNT, rehearse_steps=1,
    anchor_size=100, buffer_size=500, sequence_len=SEQ_LEN,
    replay_policy="uniform", replay_pool=8,
    guard=True, guard_per_item=500, canary_tolerance=0.5)

ARMS = ["offline", "veridical", "splice", "interleave", "scramble"]


def build_layer(mode, backend, life, seed):
    """Every online arm is a DreamLayer with contiguous runs.

    veridical is the same object with dreaming switched off, so the only
    difference between the control and the dream arms is what happens at
    replay time. Building the control from a different class would
    reintroduce exactly the confound this rewrite exists to remove.
    """
    return DreamLayer(
        backend, canary=life[:40], seed=seed,
        contiguous=True,
        dream_fraction=(0.0 if mode == "veridical" else DREAM_FRACTION),
        dream_mode=("splice" if mode == "veridical" else mode),
        **COMMON)


def run(mode, seed, life, tests):
    b = Backend(seed)
    t0 = time.time()
    stats = {}

    if mode == "offline":
        rng = random.Random(seed)
        for _ in range(OFFLINE_EPOCHS):
            order = list(life)
            rng.shuffle(order)
            for i in range(0, len(order) - OFFLINE_BATCH, OFFLINE_BATCH):
                b.update_batch(order[i:i + OFFLINE_BATCH])
    else:
        layer = build_layer(mode, b, life, seed)
        for i, item in enumerate(life):
            if i % EPISODE == 0:
                b.reset_state()
            layer.observe(item)
        stats = layer.summary()
        del layer

    b.reset_state()
    scores = {}
    for r in PHASES:
        con, sha = b.evaluate_split(tests[r])
        scores[r] = dict(contested=con, shared=sha)
    out = dict(scores=scores, grad_steps=b.grad_steps,
               rehearsals=stats.get("rehearsals", 0),
               dreams=stats.get("dreams", 0),
               dream_share=stats.get("dream_share", 0.0),
               contiguity=stats.get("contiguity", 0.0),
               sequences=stats.get("sequences", 0),
               minutes=(time.time() - t0) / 60)
    del b
    return out


if __name__ == "__main__":
    print("do the phases genuinely contradict each other?")
    worst = contradiction_check(SEEDS)
    if worst < 5:
        print(f"\nABORT: only {worst:.1f}% differ at worst.")
        raise SystemExit
    print(f"\n  good, {worst:.1f}% at worst\n")

    print(f"life: {' -> '.join(PHASES)}, {PHASE_STEPS} steps each")
    print("stored runs are now CONTIGUOUS: every item enters the run, not")
    print("only the ones the gate rejected. That is the fix the first run")
    print("needed, and the reason its control could not bite.\n")

    results = {a: {p: dict(contested=[], shared=[]) for p in PHASES}
               for a in ARMS}
    dreams = {a: [] for a in ARMS}
    contig = {a: [] for a in ARMS}
    seqs = {a: [] for a in ARMS}
    mins = {a: [] for a in ARMS}

    for seed in SEEDS:
        life = build_life(seed)
        tests = build_tests(seed)
        for arm in ARMS:
            r = run(arm, seed, life, tests)
            for p in PHASES:
                results[arm][p]["contested"].append(
                    r["scores"][p]["contested"])
                results[arm][p]["shared"].append(r["scores"][p]["shared"])
            dreams[arm].append(r["dreams"])
            contig[arm].append(r["contiguity"])
            seqs[arm].append(r["sequences"])
            mins[arm].append(r["minutes"])
        print(f"  seed {seed} done")

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    print("\n" + "=" * 92)
    print("THE PRECONDITION — are the stored runs actually contiguous?")
    print("=" * 92)
    print(f"  {'arm':>11} {'contiguity':>11} {'sequences':>10} "
          f"{'dreams':>8} {'share':>7}")
    print("  contiguity 1.00 means every item entered a run. 0.50 means "
          "the old behaviour.")
    print("-" * 92)
    for arm in ARMS:
        if arm == "offline":
            continue
        print(f"  {arm:>11} {mean(contig[arm]):>11.2f} "
              f"{mean(seqs[arm]):>10.0f} {mean(dreams[arm]):>8.0f} "
              f"{mean(dreams[arm]) and mean([d / max(1, s) for d, s in zip(dreams[arm], [max(1, x) for x in seqs[arm]])]) or 0:>7.2f}")

    print("\n" + "=" * 92)
    print("CONTESTED events — where remembering the phase matters")
    print(f"{'arm':>11} " + "  ".join(f"{p:>10}" for p in PHASES) +
          f" {'min':>6}")
    print(f"{'':>11} {'(1st)':>10}  {'':>10}  {'(last)':>10}")
    print("-" * 92)
    for arm in ARMS:
        cells = "  ".join(
            f"{mean(results[arm][p]['contested']):>9.1f}%" for p in PHASES)
        print(f"{arm:>11} {cells} {mean(mins[arm]):>6.1f}")
    print("=" * 92)

    print("\nSHARED events — identical under every rule set")
    print("  (this is where scrambled replay did its damage in August)")
    for arm in ARMS:
        cells = "  ".join(
            f"{mean(results[arm][p]['shared']):>9.1f}%" for p in PHASES)
        print(f"{arm:>11} {cells}")

    first = PHASES[0]
    ver = mean(results["veridical"][first]["contested"])
    scr = mean(results["scramble"][first]["contested"])
    spl = mean(results["splice"][first]["contested"])
    inter = mean(results["interleave"][first]["contested"])

    ver_sh = mean([mean(results["veridical"][p]["shared"]) for p in PHASES])
    scr_sh = mean([mean(results["scramble"][p]["shared"]) for p in PHASES])
    c = mean(contig["veridical"])

    print("\n" + "=" * 92)
    print("THE CONTROLS, READ FIRST")
    print("=" * 92)

    if c < 0.9:
        print(f"  CONTIGUITY FIX DID NOT TAKE: {c:.2f}, expected near "
              f"1.00. The runs are still")
        print(f"  gappy, scrambling them cannot do much damage, and "
              f"nothing below is readable.")
    else:
        print(f"  Contiguity {c:.2f}. Stored runs are real trajectories "
              f"now.")
        print(f"\n  scramble is the known-bad arm. August measured "
              f"scrambled replay damaging")
        print(f"  shared knowledge, 96.2% against 99.8%.")
        print(f"\n  shared:     veridical {ver_sh:.1f}%   scramble "
              f"{scr_sh:.1f}%   ({scr_sh - ver_sh:+.1f})")
        print(f"  contested:  veridical {ver:.1f}%   scramble "
              f"{scr:.1f}%   ({scr - ver:+.1f})")

    control_ok = c >= 0.9 and ((scr_sh < ver_sh - 0.3) or (scr < ver - 3))
    if c >= 0.9 and not control_ok:
        print(f"\n  CONTROL STILL FAILED. With contiguous runs, "
              f"scrambling them STILL does no")
        print(f"  damage. That is itself a finding: on this world the "
              f"model is not using")
        print(f"  sequence order at all, and the August +15.5 result came "
              f"from something else")
        print(f"  about sequence replay rather than from order. Worth "
              f"chasing separately.")
    elif control_ok:
        print(f"\n  Control holds.")

    print("\n" + "=" * 92)
    print(f"DOES DREAMING HELP? contested events, '{first}'")
    print("=" * 92)
    print(f"  veridical   {ver:>6.1f}%")
    print(f"  splice      {spl:>6.1f}%   ({spl - ver:+.1f})")
    print(f"  interleave  {inter:>6.1f}%   ({inter - ver:+.1f})")
    print(f"  scramble    {scr:>6.1f}%   ({scr - ver:+.1f})")

    if control_ok:
        print()
        if spl >= ver + 3:
            print(f"  RECOMBINATION GENERATES USEFUL EXPERIENCE. Splicing "
                  f"beat replaying what")
            print(f"  actually happened by {spl - ver:.1f} points. The "
                  f"model learns transferable")
            print(f"  structure from sequence order rather than memorising "
                  f"trajectories, and")
            print(f"  invented orders teach it more than real ones. That "
                  f"is a way to get more out")
            print(f"  of a fixed history, which is what a system with a "
                  f"long life needs.")
        elif spl <= ver - 3:
            print(f"  RECOMBINATION HURTS. Splicing lost "
                  f"{ver - spl:.1f} points while scramble lost")
            print(f"  {ver - scr:.1f}. Stored order carries information "
                  f"the model depends on, and")
            print(f"  inventing orders destroys it. Dreaming has to "
                  f"preserve structure rather")
            print(f"  than recombine freely.")
        else:
            print(f"  NO DIFFERENCE ({spl - ver:+.1f}). The model does not "
                  f"care whether a run actually")
            print(f"  happened, so recombination buys nothing here. It "
                  f"costs nothing either, and")
            print(f"  that is the precondition for generating experience "
                  f"at scale.")

        if inter < spl - 3 and spl > scr:
            print(f"\n  AND THE GRADIENT IS ORDERED: splice {spl:.1f} > "
                  f"interleave {inter:.1f} > scramble {scr:.1f}.")
            print(f"  How much a dream helps tracks how much original "
                  f"order it preserves. That is")
            print(f"  the mechanism visible directly rather than inferred "
                  f"from one comparison.")

    print(f"\n  One world, {len(SEEDS)} seeds, dream_fraction "
          f"{DREAM_FRACTION}. Every moment inside a dream is")
    print(f"  real with its real outcome; only the order is invented.")

    with open("dream_replay.json", "w") as f:
        json.dump(dict(results=results, dreams=dreams, contig=contig),
                  f, indent=2, default=str)
    print("\nwrote dream_replay.json")
