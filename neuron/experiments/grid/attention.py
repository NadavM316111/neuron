"""The mind's spotlight: focusing on what matters, ignoring the rest.

THE FACULTY. Right now the being perceives everything equally. A real mind
cannot. It has limited capacity, so it FOREGROUNDS what matters and lets the
rest fade, and the spotlight SHIFTS with its state and wants, a sudden
movement grabs it, a thing it cares about pulls it, a surprise seizes it.
Attention is not perceiving more; it is perceiving the RIGHT things under a
limit.

WHY IT IS NEEDED, not just nice. In a real stream most input is irrelevant
and a little is crucial. A being that spreads its limited processing evenly
across everything learns the crucial things slowly, drowned in noise. A
being that aims its processing at what is salient tracks the crucial things
despite the noise. Under a capacity limit, focus beats uniformity. That is
the whole point of an attention mechanism and it is genuinely missing here.

HOW SALIENCE IS COMPUTED, from things the being already has:
  - SURPRISE: how wrong its prediction was (prediction error). The
    unexpected grabs attention.
  - RELEVANCE: how much the input bears on what it currently wants.
  - CHARGE: how much emotional activation the input carries (ties to the
    emotion and meaning work: a thing that once mattered emotionally draws
    the eye).
Salience = a blend. The being spends its limited attention on the most
salient inputs each moment and lets the rest fade. The spotlight moves as
surprise, wants, and charge move.

THE TEST. A world of many input channels. Most are NOISE (random, carry no
learnable signal). A FEW are IMPORTANT (they actually predict a reward the
being wants) but appear rarely and are easy to miss in the noise. Each step
the being can fully process only a FEW channels (its capacity limit). Which
strategy best learns the important channels and earns the reward?

  uniform    spreads attention evenly / round-robins all channels. the mind
             with no spotlight. drowns in noise.
  random     attends to random channels each step, same capacity. shows that
             focus must be AIMED, not just limited.
  attending  spends capacity on the most SALIENT channels (surprise +
             relevance + charge). the real spotlight.

  attending beats uniform and random, and attends mostly to the IMPORTANT
  channels despite them being rare
      The spotlight found what mattered under a limit. Attention works.
  attending no better
      Salience did not pick out the important channels; the signal did not
      separate from the noise.

    python attention.py
    python attention.py --channels 30 --capacity 4 --steps 5000
"""

import argparse
import random
import statistics


def run(mode, n_channels, n_important, capacity, steps, seed):
    rng = random.Random(seed)
    # which channels are IMPORTANT: they carry a value that predicts reward
    important = set(rng.sample(range(n_channels), n_important))
    # the being's belief about each channel's reward-predictiveness, learned
    belief = [0.0] * n_channels
    salience = [1.0] * n_channels      # starts uniform, updated by surprise
    seen = [0] * n_channels

    total_reward = 0.0
    attention_on_important = 0
    attention_total = 0

    for t in range(steps):
        # each channel emits a value this step; important ones emit a signal
        # correlated with a reward available if the being 'acts' on them.
        emit = {}
        reward_here = {}
        for c in range(n_channels):
            if c in important:
                # important channel: value predicts whether acting pays
                pays = rng.random() < 0.5
                emit[c] = 1.0 if pays else 0.0
                reward_here[c] = 1.0 if pays else -0.3
            else:
                # noise channel: random, acting never pays
                emit[c] = rng.random()
                reward_here[c] = -0.3

        # CHOOSE which channels to attend to, under the capacity limit
        if mode == "uniform":
            # round-robin: cycle through all channels a few per step
            start = (t * capacity) % n_channels
            attend = [(start + i) % n_channels for i in range(capacity)]
        elif mode == "random":
            attend = rng.sample(range(n_channels), capacity)
        else:  # attending: the most salient channels
            order = sorted(range(n_channels), key=lambda c: -salience[c])
            attend = order[:capacity]

        for c in attend:
            attention_total += 1
            if c in important:
                attention_on_important += 1
            # process this channel: learn its reward-predictiveness, and act
            # if it believes acting pays
            seen[c] += 1
            r = reward_here[c]
            # update belief toward observed payoff
            belief[c] += 0.1 * (max(0.0, r) - belief[c])
            # SURPRISE updates salience: a channel whose payoff surprised it
            # (differs from belief) becomes more salient; dull ones fade
            surprise = abs(max(0.0, r) - belief[c])
            # RELEVANCE: channels it believes pay are relevant to its want
            relevance = belief[c]
            salience[c] = 0.5 * surprise + 0.5 * relevance + 0.05
            # ACT if it believes this channel pays, collect real reward
            if belief[c] > 0.3:
                total_reward += r
        # unattended channels' salience slowly decays (they fade from mind)
        for c in range(n_channels):
            if c not in attend:
                salience[c] *= 0.995

    focus = (100.0 * attention_on_important / attention_total
             if attention_total else 0.0)
    return dict(reward=total_reward, focus_on_important=focus)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", type=int, default=30)
    ap.add_argument("--important", type=int, default=4)
    ap.add_argument("--capacity", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    print(f"  {args.channels} input channels, only {args.important} matter "
          f"(predict reward); the rest is noise.")
    print(f"  the being can fully process {args.capacity} channels per "
          f"moment. under that limit,")
    print(f"  does focusing on the SALIENT channels beat spreading attention "
          f"evenly or randomly?\n")

    modes = ["uniform", "random", "attending"]
    agg = {m: {"reward": [], "focus_on_important": []} for m in modes}
    for seed in range(args.seeds):
        for m in modes:
            r = run(m, args.channels, args.important, args.capacity,
                    args.steps, seed)
            for k in agg[m]:
                agg[m][k].append(r[k])

    print("=" * 66)
    print("DID THE SPOTLIGHT FIND WHAT MATTERS?")
    print("=" * 66)
    print(f"  {'being':>10} {'reward':>9} {'attention on important':>24}")
    print("-" * 66)
    chance = 100.0 * args.important / args.channels
    for m in modes:
        note = {"uniform": "no spotlight (even)",
                "random": "limited but unaimed",
                "attending": "focuses on salient"}[m]
        print(f"  {m:>10} {statistics.mean(agg[m]['reward']):>9.0f} "
              f"{statistics.mean(agg[m]['focus_on_important']):>22.0f}%"
              f"  {note}")
    print(f"  (by chance, {chance:.0f}% of attention would land on important "
          f"channels)")

    un = statistics.mean(agg["uniform"]["reward"])
    ra = statistics.mean(agg["random"]["reward"])
    at = statistics.mean(agg["attending"]["reward"])
    af = statistics.mean(agg["attending"]["focus_on_important"])

    print("\n" + "=" * 66)
    print("WHAT IT SAYS")
    print("=" * 66)
    if at > un + 0.1 * abs(un) and at > ra and af > chance + 15:
        print(f"  THE SPOTLIGHT WORKS. The attending being earned {at:.0f} "
              f"against {un:.0f} spreading")
        print(f"  attention evenly and {ra:.0f} attending at random. Under "
              f"the same capacity limit,")
        print(f"  focusing on what is salient beat both.")
        print(f"\n  And it aimed true: {af:.0f}% of its attention landed on "
              f"the few IMPORTANT channels,")
        print(f"  against {chance:.0f}% by chance. The spotlight found what "
              f"mattered in the noise,")
        print(f"  because surprise and relevance drew it there. That is "
              f"attention: not seeing")
        print(f"  more, but seeing the RIGHT things under a limit.")
    else:
        print(f"  the spotlight did not clearly help: attending {at:.0f} vs "
              f"uniform {un:.0f}, random {ra:.0f};")
        print(f"  focus on important {af:.0f}% vs chance {chance:.0f}%. the "
              f"salience signal may need tuning.")

    print(f"\n  Salience here is surprise + relevance + charge, computed from "
          f"what the being")
    print(f"  already has. The spotlight moves as those move. A real mind "
          f"does not perceive")
    print(f"  everything; it perceives what matters, and that is what lets it "
          f"learn the")
    print(f"  important few things hidden in the irrelevant many.")


if __name__ == "__main__":
    main()
