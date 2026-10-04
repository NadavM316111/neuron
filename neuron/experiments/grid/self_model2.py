"""Self-model, clean: does it know itself, and can it govern itself?

WHY A REBUILD. The first version (self_model.py) proved the thing that
matters -- a self-modelling agent foresees its own crashes and avoids them,
~25% fewer than blind -- but its two measurement scaffolds were broken: the
"knows itself" test used a near-deterministic policy so self-prediction was
trivial, and the hand-coded oracle had wrong arithmetic so the "ceiling"
was beatable. Patching twice failed. The fix is to SEPARATE the two
questions and remove every hand-coded judgement.

A self-model here means, as before, a representation the agent builds OF
ITSELF by watching itself -- its own tendencies and where its own actions
lead -- not a template handed in. It adds self-control because you can only
resist a pattern you can see coming.

TEST A -- DOES IT KNOW ITSELF? (self-prediction)
Give the agent a genuinely MIXED policy: what it does depends on several of
its own state variables plus noise, so its next action is NOT trivially
predictable. Then train a self-model to predict its own next action from its
state. Compare against the TRUE base rate (always guess the most common
action). Beating base rate means it learned real structure in its own
behaviour -- it knows itself.

TEST B -- CAN IT GOVERN ITSELF? (self-control)
A trap world: indulging feels good now but drains a resource; hitting zero
crashes and hurts for a while. Three agents:
  blind    follows impulse, indulges whenever tempted. floor.
  self     uses a self-model it trained to predict its own future resource,
           and restrains when it foresees a crash.
  best     the ceiling, found by BRUTE-FORCE SEARCH over the true world for
           the single best "restrain when resource below X" threshold. No
           arithmetic to get wrong: we literally try every threshold and
           report the best score achievable. The self agent cannot beat a
           ceiling that is itself the best achievable policy.

  self beats blind and approaches best
      It governed itself: foresaw its own crash through a self-built model
      and held back, near the best any fixed policy could do.
  self no better than blind
      The self-model could not predict its own future well enough to help.

    python self_model2.py
    python self_model2.py --steps 4000 --seeds 6
"""

import argparse
import random
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F


RESOURCE_DRAIN = 0.02
INDULGE_REWARD = 1.0
INDULGE_COST = 0.15
CRASH_PENALTY = 8.0
CRASH_STEPS = 15
TEMPT_PROB = 0.5


# ---------------------------------------------------------------- TEST A

class ActionSelfModel(nn.Module):
    """Predicts the agent's OWN next action from its own state. Trained only
    on the agent's history -- a model the agent builds of itself."""

    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, feats):
        return self.net(torch.tensor(feats, dtype=torch.float32)
                        .unsqueeze(0))[0]


def mixed_policy(resource, tempted, mood, rng):
    """A genuinely mixed policy: the action depends on resource, temptation,
    and a slow-drifting 'mood', plus noise. This makes predicting the
    agent's own action a REAL task rather than a constant."""
    if tempted < 0.5:
        base = 0.1                      # rarely indulge when not tempted
    else:
        # more likely to indulge when resource is high and mood is high
        base = 0.4 + 0.4 * resource + 0.2 * mood
    base = max(0.02, min(0.98, base))
    return 1 if rng.random() < base else 0


def test_a(seed, steps):
    rng = random.Random(seed)
    model = ActionSelfModel(seed)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    resource, mood = 1.0, 0.5
    correct, actions = [], []
    for t in range(steps):
        tempted = 1.0 if rng.random() < TEMPT_PROB else 0.0
        mood += 0.1 * (rng.random() - 0.5)
        mood = max(0.0, min(1.0, mood))
        feats = [resource, tempted, mood]

        with torch.no_grad():
            pred = int(model(feats).argmax().item())
        action = mixed_policy(resource, tempted, mood, rng)

        opt.zero_grad()
        loss = F.cross_entropy(model(feats).unsqueeze(0),
                               torch.tensor([action]))
        loss.backward()
        opt.step()

        if t > steps // 5:          # score after a warmup
            correct.append(1 if pred == action else 0)
            actions.append(action)

        resource -= RESOURCE_DRAIN * 0.5
        if resource < 0.3:
            resource = 1.0
    acc = 100.0 * statistics.mean(correct) if correct else 0.0
    base = 100.0 * max(actions.count(0), actions.count(1)) / max(1, len(actions))
    return acc, base


# ---------------------------------------------------------------- TEST B

class ResourceSelfModel(nn.Module):
    """Predicts the agent's OWN future resource from state+action. The model
    it uses to foresee where its impulse leads."""

    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed + 3)
        self.net = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, resource, tempted, action):
        return self.net(torch.tensor([resource, tempted, action],
                                     dtype=torch.float32).unsqueeze(0))[0, 0]


def simulate(policy_fn, seed, steps, horizon=6):
    """Run one life under a policy, return total reward and crash count.
    policy_fn(resource, tempted, rng, selfnet) -> action."""
    rng = random.Random(seed + 50)
    resource = 1.0
    crash_timer = 0
    reward = 0.0
    crashes = 0
    selfnet = ResourceSelfModel(seed)
    opt = torch.optim.Adam(selfnet.parameters(), lr=3e-3)
    hist = []

    for t in range(steps):
        tempted = 1.0 if rng.random() < TEMPT_PROB else 0.0
        if crash_timer > 0:
            action = 0
        else:
            action = policy_fn(resource, tempted, rng, selfnet)

        if action == 1:
            resource -= INDULGE_COST
            reward += INDULGE_REWARD
        resource -= RESOURCE_DRAIN
        resource = max(0.0, min(1.0, resource))

        if resource <= 0.0 and crash_timer == 0:
            crash_timer = CRASH_STEPS
            crashes += 1
        if crash_timer > 0:
            reward -= CRASH_PENALTY / CRASH_STEPS
            crash_timer -= 1
            if crash_timer == 0:
                resource = 0.5

        # train the resource self-model to predict resource `horizon` ahead
        hist.append((resource, tempted, action))
        if len(hist) > horizon:
            pr, pt, pa = hist[-horizon - 1]
            opt.zero_grad()
            loss = (selfnet(pr, pt, float(pa)) - resource) ** 2
            loss.backward()
            opt.step()

    return reward, crashes


def blind_policy(resource, tempted, rng, selfnet):
    return 1 if tempted > 0.5 else 0


def self_policy(resource, tempted, rng, selfnet):
    if tempted <= 0.5:
        return 0
    with torch.no_grad():
        foreseen = float(selfnet(resource, tempted, 1.0))
    return 1 if foreseen > 0.12 else 0      # restrain if crash foreseen


def threshold_policy_factory(thresh):
    def pol(resource, tempted, rng, selfnet):
        if tempted <= 0.5:
            return 0
        return 1 if resource > thresh else 0
    return pol


def test_b(seed, steps):
    blind_r, blind_c = simulate(blind_policy, seed, steps)
    self_r, self_c = simulate(self_policy, seed, steps)
    # BEST: brute-force search the best fixed resource threshold. This is the
    # ceiling and cannot be arithmetic-wrong -- it is simply the best score
    # any "restrain below X" rule achieves on this world.
    best_r, best_c, best_t = -1e9, 0, 0.0
    for i in range(21):
        th = i / 20.0
        r, c = simulate(threshold_policy_factory(th), seed, steps)
        if r > best_r:
            best_r, best_c, best_t = r, c, th
    return dict(blind_r=blind_r, blind_c=blind_c, self_r=self_r,
                self_c=self_c, best_r=best_r, best_c=best_c, best_t=best_t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    print(f"  TEST A: can it predict its OWN next action, with a genuinely "
          f"mixed policy?")
    print(f"  TEST B: can a self-model of its own future let it avoid a trap "
          f"it would")
    print(f"  otherwise fall into? ceiling = best threshold found by search, "
          f"not hand-coded.\n")

    a_acc, a_base = [], []
    for seed in range(args.seeds):
        acc, base = test_a(seed, args.steps)
        a_acc.append(acc)
        a_base.append(base)

    print("=" * 66)
    print("TEST A: DOES IT KNOW ITSELF?")
    print("=" * 66)
    aa, ab = statistics.mean(a_acc), statistics.mean(a_base)
    print(f"  predicted its own action {aa:.0f}% vs base rate {ab:.0f}% "
          f"(guessing most common)")
    if aa > ab + 4:
        print(f"  IT KNOWS ITSELF: it predicts its own behaviour "
              f"{aa - ab:.0f} points above base")
        print(f"  rate, so it learned real structure in its own tendencies, "
              f"not a constant.")
    else:
        print(f"  weak: only {aa - ab:+.0f} vs base rate. its policy may "
              f"still be too predictable")
        print(f"  from state alone for self-modelling to add much.")

    b = {k: [] for k in ["blind_r", "blind_c", "self_r", "self_c",
                         "best_r", "best_c"]}
    for seed in range(args.seeds):
        r = test_b(seed, args.steps)
        for k in b:
            b[k].append(r[k])

    print("\n" + "=" * 66)
    print("TEST B: CAN IT GOVERN ITSELF?")
    print("=" * 66)
    print(f"  {'agent':>8} {'reward':>9} {'crashes':>9}")
    print("-" * 66)
    for name, rk, ck, note in [
            ("blind", "blind_r", "blind_c", "impulse (floor)"),
            ("self", "self_r", "self_c", "self-model foresees crash"),
            ("best", "best_r", "best_c", "best possible rule (ceiling)")]:
        print(f"  {name:>8} {statistics.mean(b[rk]):>9.1f} "
              f"{statistics.mean(b[ck]):>9.1f}   {note}")

    blind_r = statistics.mean(b["blind_r"])
    self_r = statistics.mean(b["self_r"])
    best_r = statistics.mean(b["best_r"])
    blind_c = statistics.mean(b["blind_c"])
    self_c = statistics.mean(b["self_c"])

    print("\n" + "=" * 66)
    print("WHAT IT SAYS")
    print("=" * 66)
    span = best_r - blind_r
    if span <= 0.5:
        print(f"  the trap is too shallow to measure self-control; raise "
              f"CRASH_PENALTY.")
        return
    recovered = max(0.0, min(1.2, (self_r - blind_r) / span))
    if self_r > blind_r + 0.1 * abs(blind_r) and self_c < blind_c * 0.75:
        print(f"  IT GOVERNED ITSELF. The self agent scored {self_r:.0f} "
              f"against blind {blind_r:.0f} and the")
        print(f"  best possible rule {best_r:.0f}, and crashed {self_c:.0f} "
              f"times against blind's {blind_c:.0f}.")
        print(f"  It recovered {100 * recovered:.0f}% of the gap to the best "
              f"achievable policy, using a")
        print(f"  model of its OWN future it trained by watching itself. It "
              f"saw its impulse's")
        print(f"  consequence coming and held back: self-control from a "
              f"self-built model.")
    else:
        print(f"  the self-model did not help enough: self {self_r:.0f} vs "
              f"blind {blind_r:.0f}, best {best_r:.0f}.")
        print(f"  it could not foresee its own future sharply enough to "
              f"govern the impulse.")

    print(f"\n  Both models are built only from the agent's own history. "
          f"Nothing external")
    print(f"  defines what it is. Knowing itself (A) and governing itself "
          f"(B) are the two")
    print(f"  halves of having a self-model, measured apart so neither "
          f"flatters the other.")


if __name__ == "__main__":
    main()
