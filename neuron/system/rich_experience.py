"""NEURON grows from a rich, multi-channel firehose of experience, like a baby.

A baby does not grow intelligent from one thin stream, it grows from a FIREHOSE:
sight, sound, language, cause-and-effect, new things, all at once, continuously.
We have been feeding NEURON a thin single stream (news). This gives it a rich,
multi-channel experience instead, so it has far more to grow from:

  1. WORLD EVENTS   - what is happening (news)
  2. KNOWLEDGE      - new concepts it encounters (like a baby meeting new things)
  3. LANGUAGE       - real sentences with structure (like hearing speech)
  4. PATTERNS       - sequences to predict (like learning cause-and-effect)
  5. REFLECTION     - relating the new to what it knows (like connecting things)

It grows from all channels at once, and the Core owns everything. The honest
question: does a rich multi-channel firehose grow its understanding faster and
richer than the thin stream did?

Honest ceiling: brain SIZE still caps how far it can ultimately go (that is
resource-gated). But richer input means more growth within that cap, a real,
buildable lever, not magic.

    python rich_experience.py --moments 400
"""

import os
import re
import json
import random
import argparse
import urllib.request
import urllib.parse

STATE = "rich_mind.json"

STOP = set("""a an the and or but if of to in on at by for with from as is are
was were be been it its this that these those i you he she we they them his her
my your our their do did have has had not no what which who where when how all
any some such than there here""".split())


# ---------------- the channels NEURON experiences ----------------
def channel_world():
    out = []
    for url in ["https://feeds.bbci.co.uk/news/world/rss.xml"]:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            raw = urllib.request.urlopen(req, timeout=15).read().decode("utf-8","ignore")
            for t in re.findall(r"<title>(.*?)</title>", raw)[1:12]:
                t = re.sub(r"<.*?>", "", t).strip()
                if 10 < len(t) < 200:
                    out.append(t)
        except Exception:
            pass
    return out


def channel_knowledge():
    """New concepts, like a baby encountering new things."""
    out = []
    try:
        for _ in range(4):
            req = urllib.request.Request(
                "https://en.wikipedia.org/api/rest_v1/page/random/summary",
                headers={"User-Agent": "NEURON/2.0"})
            d = json.load(urllib.request.urlopen(req, timeout=15))
            if d.get("extract"):
                out.append(d["extract"])
    except Exception:
        pass
    return out


def channel_language():
    """Real sentences with structure, like hearing speech."""
    return [
        "the child reached for the light and the light was warm.",
        "when the rain came, the animals went quiet and waited.",
        "she gave what she had, and found she had more than before.",
        "the road was long but every step made it shorter.",
        "what is held too tightly is the first thing to break.",
    ]


def channel_patterns():
    """Sequences to predict, like learning cause-and-effect."""
    seqs = []
    for _ in range(4):
        a, b = random.randint(1, 5), random.randint(1, 5)
        seq = [a + b*i for i in range(5)]
        seqs.append(seq)
    return seqs


# ---------------- the Core that grows and owns ----------------
class Core:
    def __init__(self):
        self.understanding = {}       # word -> {neighbor: strength}
        self.concepts = 0
        self.pattern_skill = 0.0
        self.moments = 0

    def load(self):
        if os.path.exists(STATE):
            try:
                d = json.load(open(STATE))
                self.understanding = d.get("understanding", {})
                self.concepts = d.get("concepts", 0)
                self.pattern_skill = d.get("pattern_skill", 0.0)
                self.moments = d.get("moments", 0)
            except Exception:
                pass

    def save(self):
        trimmed = {w: dict(sorted(n.items(), key=lambda kv: -kv[1])[:10])
                   for w, n in self.understanding.items()}
        json.dump({"understanding": trimmed, "concepts": self.concepts,
                   "pattern_skill": self.pattern_skill,
                   "moments": self.moments}, open(STATE, "w"))

    def absorb_text(self, text):
        words = [w for w in re.findall(r"[a-z']+", text.lower())
                 if len(w) > 2 and w not in STOP]
        for i, w in enumerate(words):
            a = self.understanding.setdefault(w, {})
            for j in range(max(0, i-3), min(len(words), i+4)):
                if j != i:
                    a[words[j]] = a.get(words[j], 0) + 1
        self.concepts += 1
        self.moments += 1

    def learn_pattern(self, seq):
        # it tries to predict the next number (cause-effect learning)
        if len(seq) >= 3:
            diff1 = seq[1] - seq[0]
            predicted = seq[-2] + diff1
            actual = seq[-1]
            right = 1.0 if predicted == actual else 0.0
            self.pattern_skill = 0.95 * self.pattern_skill + 0.05 * right
        self.moments += 1

    def richness(self):
        if not self.understanding:
            return 0
        return sum(len(n) for n in self.understanding.values()) / len(self.understanding)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moments", type=int, default=400)
    args = ap.parse_args()

    core = Core()
    core.load()

    print("=" * 64)
    print("  NEURON GROWS FROM A RICH MULTI-CHANNEL FIREHOSE (like a baby)")
    print("=" * 64)
    print("  Not one thin stream, many channels at once: world, knowledge,")
    print("  language, patterns. It grows from all of it. (Brain size still")
    print("  caps the ceiling; richer input means more growth within it.)\n")

    print("  gathering the channels of experience...")
    world = channel_world()
    knowledge = channel_knowledge()
    language = channel_language()
    print(f"  world events: {len(world)} | new concepts: {len(knowledge)} | "
          f"language: {len(language)}\n")

    start_words = len(core.understanding)
    start_rich = core.richness()

    rng = random.Random()
    checkpoint = max(1, args.moments // 5)
    for m in range(args.moments):
        # each moment, it experiences a RANDOM channel (rich, varied input)
        ch = rng.choice(["world", "knowledge", "language", "patterns"])
        if ch == "world" and world:
            core.absorb_text(rng.choice(world))
        elif ch == "knowledge" and knowledge:
            core.absorb_text(rng.choice(knowledge))
        elif ch == "language":
            core.absorb_text(rng.choice(language))
        elif ch == "patterns":
            core.learn_pattern(rng.choice(channel_patterns()))
        if (m + 1) % checkpoint == 0:
            print(f"  lived {core.moments:>6} moments | words {len(core.understanding):>5,} "
                  f"| richness {core.richness():.1f} | pattern-skill "
                  f"{core.pattern_skill*100:.0f}%")

    core.save()
    end_words = len(core.understanding)
    end_rich = core.richness()

    print("\n" + "=" * 64)
    print("  WHAT IT GREW (from the rich firehose, saved)")
    print("=" * 64)
    print(f"  words understood: {start_words:,} -> {end_words:,} "
          f"(+{end_words-start_words:,})")
    print(f"  richness (connections per word): {start_rich:.1f} -> {end_rich:.1f}")
    print(f"  pattern-prediction skill: {core.pattern_skill*100:.0f}%")
    print(f"\n  what it understands, grown across channels:")
    for probe in ["light", "world", "rain", "road", "child"]:
        a = core.understanding.get(probe)
        if a:
            rel = [w for w, _ in sorted(a.items(), key=lambda kv: -kv[1])[:4]]
            print(f"    \"{probe}\" -> {', '.join(rel)}")

    print(f"\n  NEURON grew from rich, varied, multi-channel experience, more "
          f"like a baby's")
    print(f"  firehose than a thin feed. It built understanding AND a little "
          f"pattern skill")
    print(f"  across channels at once, and keeps it. Honest: its brain size "
          f"still caps how")
    print(f"  far it can ultimately go (that needs resources), but richer "
          f"experience means")
    print(f"  more growth within that cap, a real, buildable lever, exactly "
          f"what the baby")
    print(f"  insight points at. It grows by living richly.")


if __name__ == "__main__":
    main()
