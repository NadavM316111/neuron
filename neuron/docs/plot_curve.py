"""The curve: competence over a 100,000-step life.

Milestone 5 from the build plan: "The curve is published. Steps against
competence, with the collapse points marked. Nobody has this curve for a
grown network."

Reads longrun.json, which longlife.py wrote.
"""

import json
import matplotlib.pyplot as plt

RULES = ["keyed", "open", "trap"]
PHASE_STEPS = 10000

with open("longrun.json") as f:
    data = json.load(f)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

colors = {"online": "#c44", "sequences": "#248"}
labels = {"online": "online, no protection",
          "sequences": "online + sequence replay"}

# Shade the phases so the rule changes are visible.
for i in range(10):
    if RULES[i % 3] == "trap":
        for ax in (ax1, ax2):
            ax.axvspan(i * PHASE_STEPS, (i + 1) * PHASE_STEPS,
                       color="#000", alpha=0.05, zorder=0)

for mode in ["online", "sequences"]:
    h = data[mode]["history"]
    steps = [r["step"] for r in h]
    ax1.plot(steps, [r["overall"] for r in h], "-o", ms=4,
             color=colors[mode], label=labels[mode])
    past = [(r["step"], r["past"]) for r in h if r["past"] is not None]
    ax2.plot([p[0] for p in past], [p[1] for p in past], "-o", ms=4,
             color=colors[mode], label=labels[mode])

ax1.set_ylabel("competence across all rule sets (%)")
ax1.set_title("A network grown from random weights, living 100,000 steps\n"
              "shaded bands are 'trap' phases, where the rules contradict "
              "most", loc="left", fontsize=11)
ax1.legend(loc="lower right", fontsize=9)
ax1.grid(alpha=0.2)

ax2.set_ylabel("accuracy on rules lived under before,\nbut not currently (%)")
ax2.set_xlabel("steps of life")
ax2.set_title("Retention: what it still knows about the past",
              loc="left", fontsize=11)
ax2.legend(loc="upper right", fontsize=9)
ax2.grid(alpha=0.2)

fig.tight_layout()
fig.savefig("neuron_curve.png", dpi=160)
print("wrote neuron_curve.png")

for mode in ["online", "sequences"]:
    h = data[mode]["history"]
    o = [r["overall"] for r in h]
    half = len(o) // 2
    print(f"\n{mode}:")
    print(f"  first half  {sum(o[:half]) / half:.1f}%")
    print(f"  second half {sum(o[half:]) / (len(o) - half):.1f}%")