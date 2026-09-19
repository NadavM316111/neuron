import random
import torch
from collections import deque
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
device = "mps"
LR = 1e-3
STEPS_PER_LEARN = 3
SEEDS = [0, 1, 2]

tokenizer = AutoTokenizer.from_pretrained(MODEL)


def fresh_model():
    base = AutoModelForCausalLM.from_pretrained(MODEL).to(device).float()
    cfg = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    return get_peft_model(base, cfg)


def token_surprises(model, text, grad=False):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    ids = inputs["input_ids"]
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        logits = model(**inputs).logits
    preds = logits[:, :-1, :]
    targets = ids[:, 1:]
    return torch.nn.functional.cross_entropy(
        preds.reshape(-1, preds.size(-1)),
        targets.reshape(-1),
        reduction="none",
    )


def score(model, text):
    model.eval()
    return token_surprises(model, text).mean().item()


def learn(model, optimizer, text):
    model.train()
    for _ in range(STEPS_PER_LEARN):
        loss = token_surprises(model, text, grad=True).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class Gate:
    def __init__(self, window=40, top_fraction=0.30):
        self.history = deque(maxlen=window)
        self.top_fraction = top_fraction

    def observe(self, s):
        self.history.append(s)

    def check(self, s):
        self.history.append(s)
        recent = sorted(self.history)
        return s >= recent[int(len(recent) * (1 - self.top_fraction))]


CALIBRATION = [
    "The sky is blue on a clear day and the clouds are white.",
    "Dogs are common household pets that many families keep at home.",
    "Water boils at one hundred degrees Celsius at normal sea level.",
    "Cats sleep for many hours each day in warm sunny places.",
    "Bread is made from flour and water mixed together and baked.",
    "Winter is generally colder than summer in most northern places.",
    "Books contain printed pages that are bound together along one edge.",
    "Cars typically have four wheels and an engine under the hood.",
    "Rain falls from the clouds when the water droplets become heavy.",
    "Grass is green and grows in lawns and open fields everywhere.",
    "Shoes are worn on the feet to protect them while walking.",
    "Fire is hot and produces both light and warmth while burning.",
]

TARGETS_A = [
    "Nadav Minkowitz is building a runtime system that he calls NEURON.",
    "Clickflo is an application that organizes screenshots automatically for you.",
    "TROY Capital is a paper trading simulator built for learning finance.",
    "Mink Studios produces short films and various other video projects.",
    "Hephaestus is the agent that writes the code inside the system.",
]

PHASE_A = [
    TARGETS_A[0],
    "Windows are made of glass and let daylight into a room.",
    TARGETS_A[1],
    "Birds have feathers and most of them are able to fly.",
    TARGETS_A[2],
    "Snow is frozen water that falls during the cold winter weather.",
    TARGETS_A[3],
    "Coffee is a popular morning drink for people all around the world.",
    TARGETS_A[4],
    "Chairs are used for sitting and usually have four sturdy legs.",
]

PHASE_B = [
    "The harbor master at Kelso Point keeps a logbook of every arriving vessel.",
    "Dry stone walling requires no mortar and relies on careful stone placement.",
    "The Aldwych branch line closed to passengers in nineteen ninety four.",
    "Fermented shark is buried underground for weeks before it is eaten.",
    "Telephones let people speak to each other across very long distances.",
    "The clarinet player at the Bellhouse rehearses every Tuesday evening.",
    "Tortoiseshell cats are almost always female because of their chromosomes.",
    "The lighthouse on Skerry Rock has been automated since the nineteen eighties.",
    "Trees grow new green leaves during the warmer months of spring.",
    "Cochineal insects are crushed to produce a deep red natural dye.",
    "The mill at Fenwick Bottom ground flour until the river was diverted.",
    "Doors are usually made of wood and swing open on metal hinges.",
    "Sourdough starters can survive for decades if they are fed regularly.",
    "The bell ringers at Saint Alden practice a method called Grandsire Triples.",
    "Pewter is an alloy made mostly of tin with small amounts of copper.",
]

CONTROLS = [
    "Paris is the capital city of France and a major tourist destination.",
    "The Pacific Ocean is the largest ocean on the surface of Earth.",
    "Einstein developed the theory of relativity in the early twentieth century.",
    "Shakespeare wrote many famous plays including Hamlet and Romeo and Juliet.",
]


def avg(model, texts):
    return sum(score(model, t) for t in texts) / len(texts)


def run_arm(mode, seed, random_rate=None):
    """mode: frozen | always | gated | random
    random_rate: fires with this probability, matched to what the gate did."""
    torch.manual_seed(seed)
    rng = random.Random(seed)

    model = fresh_model()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )
    gate = Gate()
    for text in CALIBRATION:
        gate.observe(score(model, text))

    start_A = avg(model, TARGETS_A)
    start_C = avg(model, CONTROLS)

    hit_targets = 0

    def process(stream):
        nonlocal hit_targets
        n = 0
        for text in stream:
            s = score(model, text)
            if mode == "frozen":
                do = False
            elif mode == "always":
                do = True
            elif mode == "gated":
                do = gate.check(s)
            else:
                gate.observe(s)
                do = rng.random() < random_rate
            if do:
                learn(model, optimizer, text)
                n += 1
                if text in TARGETS_A:
                    hit_targets += 1
        return n

    updates_A = process(PHASE_A)
    mid_A = avg(model, TARGETS_A)
    updates_B = process(PHASE_B)
    end_A = avg(model, TARGETS_A)
    end_C = avg(model, CONTROLS)

    del model, optimizer
    if device == "mps":
        torch.mps.empty_cache()

    return dict(
        learned=start_A - mid_A,
        forgot=end_A - mid_A,
        kept=start_A - end_A,
        drift=end_C - start_C,
        updates=updates_A + updates_B,
        targets_hit=hit_targets,
    )


print("Four arms across 3 seeds. This will take a while.\n")

all_results = {"frozen": [], "always": [], "gated": [], "random": []}

for seed in SEEDS:
    print(f"seed {seed}")
    for mode in ["frozen", "always", "gated"]:
        r = run_arm(mode, seed)
        all_results[mode].append(r)
        print(f"  {mode:>7}: {r['updates']:>2} updates "
              f"({r['targets_hit']}/5 targets)  kept {r['kept']:+.4f}")

    # Random arm fires at whatever rate the gate just used, same seed.
    gate_updates = all_results["gated"][-1]["updates"]
    rate = gate_updates / (len(PHASE_A) + len(PHASE_B))
    r = run_arm("random", seed, random_rate=rate)
    all_results["random"].append(r)
    print(f"  {'random':>7}: {r['updates']:>2} updates "
          f"({r['targets_hit']}/5 targets)  kept {r['kept']:+.4f}   [rate {rate:.2f}]")


def mean(rows, key):
    return sum(r[key] for r in rows) / len(rows)


print("\n" + "=" * 84)
print(f"{'arm':>8} {'updates':>8} {'tgt hit':>8} {'learned':>10} "
      f"{'forgot':>10} {'kept':>10} {'drift':>10}")
print("-" * 84)
for name in ["frozen", "always", "gated", "random"]:
    rows = all_results[name]
    print(f"{name:>8} {mean(rows,'updates'):>8.1f} {mean(rows,'targets_hit'):>8.1f} "
          f"{mean(rows,'learned'):>+10.4f} {mean(rows,'forgot'):>+10.4f} "
          f"{mean(rows,'kept'):>+10.4f} {mean(rows,'drift'):>+10.4f}")
print("=" * 84)
print("""
Averages over 3 seeds.

THE COMPARISON THAT MATTERS: gated vs random.
Same number of updates. Only the choosing differs.

  gated KEPT clearly higher than random  ->  the gate is selecting well
  gated and random roughly equal         ->  the gate is an expensive coin flip

'tgt hit' shows how many of the 5 target facts each arm actually
trained on. If gated hits 5/5 and random hits 2/5, that IS the gate working.
""")