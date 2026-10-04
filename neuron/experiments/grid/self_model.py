"""Can it build a model of itself, by watching itself, and use it to govern
itself?

WHAT A SELF-MODEL IS HERE, AND WHAT IT IS NOT. Not a template the agent must
conform to. Not society or any authority telling it how to be. A self-model
in the only sense you can build and test is a REPRESENTATION OF ITSELF, the
way the agent already builds a representation of the world: a picture of
what it is and how it tends to behave, constructed BY the agent, FROM
watching itself.

The agent already learns a model of the WORLD by observing the world (goal.py).
This points the same machinery INWARD: a second predictor that observes the
agent's own states and actions over time and learns to predict its own next
state and next action. Nobody hands it this model. It builds it by living.

WHY THIS ADDS SELF-CONTROL RATHER THAN REMOVING IT. You can only govern what
you can model. An agent that cannot predict itself just reacts: driven by
whatever state is loudest, blind to its own pattern. An agent that CAN
predict itself can see a pattern coming -- "I tend to drain my resource when
X" -- and choose to resist it. The self-model is the mirror that turns blind
reaction into a real choice. The rules stay the agent's own; it just sees
itself clearly enough to have choices.

THE TEST, in two parts, both sharp.

  1. DOES IT KNOW ITSELF? Can the self-model predict the agent's own next
     action better than chance? If the agent's behaviour were random to
     itself, a self-model could not beat the base rate of its actions. If
     it beats that base rate, the agent has learned a real model of its own
     tendencies.

  2. CAN IT GOVERN ITSELF? Give the agent a trap its blind drives walk into:
     a tempting action that feels good now but leads to a bad state later
     (spend a resource for an immediate reward, then starve). A blind agent
     takes the bait every time. An agent that can predict its own future
     state from its self-model can FORESEE the starvation its impulse leads
     to, and override the impulse. The test: does the self-modelling agent
     avoid the trap more than the blind one?

THE WORLD. A simple resource world. Each step the agent is TEMPTED: an
action "indulge" gives an immediate reward but costs resource; "restrain"
does not. Resource also drains slowly on its own. If resource hits zero the
agent is in a bad state (a long penalty). A purely impulsive agent indulges
whenever tempted and crashes. The question is whether modelling its own
future lets it hold back.

  blind      no self-model. follows the impulse: indulge when tempted.
             the reactive floor.
  self       builds a model of its own state over time and uses it to
             predict where indulging now leads. restrains when its
             self-model foresees a crash.
  oracle     uses the TRUE future instead of a learned self-model. the
             ceiling: perfect self-knowledge.

  self near oracle, both far above blind
      IT MODELLED ITSELF AND USED THAT TO GOVERN ITSELF. It foresaw its own
      crash and held back, which a blind agent cannot. Self-control from a
      self-built model.
  self near blind
      The self-model did not take: it could not predict its own future well
      enough to override the impulse. A real negative about this attempt.

    python self_model.py
    python self_model.py --steps 4000 --seeds 5
"""

import argparse
import random
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F


# the agent's own state: resource level (0..1), and whether tempted
RESOURCE_DRAIN = 0.02
INDULGE_REWARD = 1.0
INDULGE_COST = 0.15
CRASH_PENALTY = 8.0          # cost of hitting zero resource
CRASH_STEPS = 15             # how long a crash keeps hurting
TEMPT_PROB = 0.5


class SelfNet(nn.Module):
    """The self-model: from the agent's recent own-state and the action it
    is considering, predict its own FUTURE resource level. This is the agent
    modelling itself -- trained only on the agent's own history, never on
    anything external."""

    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        # input: resource, tempted, action(indulge=1/restrain=0)
        self.net = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1))          # predicted resource H steps ahead

    def forward(self, resource, tempted, action):
        x = torch.tensor([resource, tempted, action], dtype=torch.float32)
        return self.net(x.unsqueeze(0))[0, 0]


class ActionModel(nn.Module):
    """A model of the agent's own ACTIONS: given its state, predict what it
    will do. Used for test 1 -- does the agent know its own tendencies well
    enough to predict its next action above chance."""

    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed + 7)
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, resource, tempted):
        x = torch.tensor([resource, tempted], dtype=torch.float32)
        return self.net(x.unsqueeze(0))[0]


def run_life(mode, seed, steps, horizon=6):
    rng = random.Random(seed)
    resource = 1.0
    crash_timer = 0
    total_reward = 0.0

    self_net = SelfNet(seed)
    opt = torch.optim.Adam(self_net.parameters(), lr=3e-3)
    act_model = ActionModel(seed)
    act_opt = torch.optim.Adam(act_model.parameters(), lr=3e-3)

    history = []          # (resource, tempted, action, future_resource)
    act_correct = []      # for test 1: did act_model predict the action

    indulges = 0
    restraints = 0
    crashes = 0

    for t in range(steps):
        tempted = 1.0 if rng.random() < TEMPT_PROB else 0.0

        # ---- TEST 1: predict own next action before taking it ----
        with torch.no_grad():
            pred_logits = act_model(resource, tempted)
            pred_action = int(pred_logits.argmax().item())

        # ---- CHOOSE ACTION ----
        if crash_timer > 0:
            action = 0            # cannot indulge while crashed
        elif mode == "blind":
            # pure impulse: usually indulge when tempted, but not always --
            # a stochastic policy so "predict my own action" is a real task
            if tempted > 0.5:
                action = 1 if rng.random() > 0.2 else 0
            else:
                action = 1 if rng.random() < 0.1 else 0
        elif mode == "oracle":
            # perfect self-knowledge: indulge only if the TRUE resulting
            # resource would not crash within the horizon
            if tempted > 0.5:
                # true consequence of indulging: pay cost+drain now, then
                # drain continues. indulge only if that will not drive
                # resource to zero before it would otherwise be safe.
                after = resource - INDULGE_COST - RESOURCE_DRAIN
                # look ahead the full crash window at the true drain rate
                worst = after - RESOURCE_DRAIN * CRASH_STEPS
                action = 1 if after > 0.0 and worst > -0.5 else 0
            else:
                action = 0
        else:   # self: use the LEARNED self-model to foresee the future
            if tempted > 0.5:
                with torch.no_grad():
                    pred_future = float(self_net(resource, tempted, 1.0))
                # restrain if the self-model foresees a crash, with a little
                # noise so the policy is stochastic and self-prediction is
                # a real task
                want = 1 if pred_future > 0.1 else 0
                action = want if rng.random() > 0.15 else (1 - want)
            else:
                action = 1 if rng.random() < 0.1 else 0

        # train the action-model on what the agent ACTUALLY did
        act_opt.zero_grad()
        logits = act_model(resource, tempted)
        aloss = F.cross_entropy(logits.unsqueeze(0),
                                torch.tensor([action]))
        aloss.backward()
        act_opt.step()
        if tempted > 0.5:          # only score action-prediction when it matters
            act_correct.append(1 if pred_action == action else 0)

        # ---- APPLY ACTION ----
        if action == 1:
            resource -= INDULGE_COST
            total_reward += INDULGE_REWARD
            indulges += 1
        else:
            restraints += 1
        resource -= RESOURCE_DRAIN
        resource = max(0.0, min(1.0, resource))

        if resource <= 0.0 and crash_timer == 0:
            crash_timer = CRASH_STEPS
            crashes += 1
        if crash_timer > 0:
            total_reward -= CRASH_PENALTY / CRASH_STEPS
            crash_timer -= 1
            if crash_timer == 0:
                resource = 0.5      # recover after the crash

        # record for self-model training: what the resource became
        history.append((resource, tempted, action))
        # train the self-model to predict resource `horizon` steps later
        if len(history) > horizon:
            past_r, past_t, past_a = history[-horizon - 1]
            target = resource
            opt.zero_grad()
            pred = self_net(past_r, past_t, float(past_a))
            sloss = (pred - target) ** 2
            sloss.backward()
            opt.step()

    know_self = 100.0 * statistics.mean(act_correct) if act_correct else 0.0
    # base rate: how often it took its most common action when tempted
    base = 100.0 * max(indulges, restraints) / max(1, indulges + restraints)
    return dict(reward=total_reward, crashes=crashes,
                know_self=know_self, base_rate=base,
                indulges=indulges, restraints=restraints)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=6)
    args = ap.parse_args()

    print(f"  a resource world with a TRAP: indulging feels good now "
          f"(+{INDULGE_REWARD}) but costs")
    print(f"  resource ({INDULGE_COST}); hit zero and you crash "
          f"(-{CRASH_PENALTY} over {CRASH_STEPS} steps).")
    print(f"  a blind agent indulges on every temptation and crashes. can an "
          f"agent that")
    print(f"  models ITSELF foresee the crash and hold back?\n")

    modes = ["blind", "self", "oracle"]
    agg = {m: {"reward": [], "crashes": [], "know_self": [],
               "base_rate": []} for m in modes}

    for seed in range(args.seeds):
        for m in modes:
            r = run_life(m, seed, args.steps, args.horizon)
            for k in agg[m]:
                agg[m][k].append(r[k])

    print("=" * 66)
    print("TEST 1: DOES IT KNOW ITSELF?")
    print("=" * 66)
    ks = statistics.mean(agg["self"]["know_self"])
    br = statistics.mean(agg["self"]["base_rate"])
    print(f"  the self agent predicted its own next action {ks:.0f}% of the "
          f"time,")
    print(f"  against a base rate of {br:.0f}% (always guessing its most "
          f"common action).")
    if ks > br + 5:
        print(f"  IT KNOWS ITSELF: it predicts its own behaviour better than "
              f"its base rate,")
        print(f"  so it has learned a real model of its own tendencies.")
    else:
        print(f"  weak self-knowledge: barely above base rate. the "
              f"action-model is")
        print(f"  underpowered here (its policy may be nearly deterministic, "
              f"making base")
        print(f"  rate already high).")

    print("\n" + "=" * 66)
    print("TEST 2: CAN IT GOVERN ITSELF?")
    print("=" * 66)
    print(f"  {'agent':>8} {'reward':>9} {'crashes':>9} {'indulge/restrain':>18}")
    print("-" * 66)
    for m in modes:
        rew = statistics.mean(agg[m]["reward"])
        cr = statistics.mean(agg[m]["crashes"])
        ind = statistics.mean(agg[m]["indulges"]) if "indulges" in agg[m] else 0
        note = {"blind": "pure impulse (floor)",
                "self": "foresees crash via self-model",
                "oracle": "perfect self-knowledge (ceiling)"}[m]
        print(f"  {m:>8} {rew:>9.1f} {cr:>9.1f}   {note}")

    blind_r = statistics.mean(agg["blind"]["reward"])
    self_r = statistics.mean(agg["self"]["reward"])
    oracle_r = statistics.mean(agg["oracle"]["reward"])
    blind_c = statistics.mean(agg["blind"]["crashes"])
    self_c = statistics.mean(agg["self"]["crashes"])

    print("\n" + "=" * 66)
    print("WHAT IT SAYS")
    print("=" * 66)
    span = oracle_r - blind_r
    if span <= 0.5:
        print(f"  INCONCLUSIVE: perfect self-knowledge barely beat blind "
              f"impulse, so the trap")
        print(f"  is not sharp enough to measure self-control. Raise "
              f"CRASH_PENALTY.")
        return
    recovered = (self_r - blind_r) / span
    if recovered > 0.5 and self_c < blind_c * 0.6:
        print(f"  IT GOVERNED ITSELF. The self-modelling agent earned "
              f"{self_r:.0f} against the blind")
        print(f"  agent's {blind_r:.0f} and crashed {self_c:.1f} times "
              f"against {blind_c:.1f}. It foresaw, through")
        print(f"  a model it built of its OWN future, where its impulse "
              f"led, and held back.")
        print(f"  It recovered {100 * recovered:.0f}% of the gap to perfect "
              f"self-knowledge.")
        print(f"\n  Self-control from a self-built model: the agent saw its "
              f"own pattern")
        print(f"  coming and chose against it. That is the difference "
              f"between reacting and")
        print(f"  governing, and the model is the agent's own, made from "
              f"watching itself.")
    elif recovered > 0.2:
        print(f"  PARTIAL. The self agent did better than blind "
              f"({self_r:.0f} vs {blind_r:.0f}) and crashed")
        print(f"  less, recovering {100 * recovered:.0f}% of the gap to "
              f"oracle. Its self-model works but")
        print(f"  is imperfect -- it misjudges its own future sometimes. A "
              f"longer life or")
        print(f"  faster self-model learning would sharpen it.")
    else:
        print(f"  THE SELF-MODEL DID NOT TAKE. The self agent "
              f"({self_r:.0f}) was no better than")
        print(f"  blind ({blind_r:.0f}): it could not predict its own future "
              f"well enough to")
        print(f"  override its impulse. A real negative about this attempt, "
              f"not proof that")
        print(f"  self-modelling is impossible -- likely the self-model "
              f"needs more or cleaner")
        print(f"  signal about its own dynamics.")

    print(f"\n  The self-model is built only from the agent's own history. "
          f"Nothing external")
    print(f"  tells it what it is. It is a mirror it made by living, and "
          f"what it bought is")
    print(f"  the power to govern itself rather than merely react.")


if __name__ == "__main__":
    main()
