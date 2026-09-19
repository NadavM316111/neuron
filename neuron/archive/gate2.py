import torch
from collections import deque
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
device = "mps"

print("Loading...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL).to(device)
model.eval()
print("Loaded.\n")


def token_surprises(text):
    """Per-word surprise values for one piece of text."""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    ids = inputs["input_ids"]

    with torch.no_grad():
        logits = model(**inputs).logits

    # Line up each prediction with the word it was trying to predict.
    preds = logits[:, :-1, :]
    targets = ids[:, 1:]

    losses = torch.nn.functional.cross_entropy(
        preds.reshape(-1, preds.size(-1)),
        targets.reshape(-1),
        reduction="none",
    )
    return losses


def score_mean(text):
    return token_surprises(text).mean().item()


def score_max(text):
    return token_surprises(text).max().item()


def score_top3(text):
    """Average of the three most surprising words. Less jumpy than pure max."""
    s = token_surprises(text)
    k = min(3, s.numel())
    return s.topk(k).values.mean().item()


class Gate:
    def __init__(self, window=50, top_fraction=0.20, warmup=10):
        self.history = deque(maxlen=window)
        self.top_fraction = top_fraction
        self.warmup = warmup

    def check(self, score):
        self.history.append(score)
        if len(self.history) < self.warmup:
            return None, 0.0
        recent = sorted(self.history)
        cutoff = int(len(recent) * (1 - self.top_fraction))
        threshold = recent[cutoff]
        return score >= threshold, threshold


# Stream. True = genuinely novel, should fire. False = ordinary, should not.
# First 10 are calibration only, marked None, never judged.
STREAM = [
    ("The sky is blue on a clear day and the clouds are white.", None),
    ("Dogs are common household pets that many families keep.", None),
    ("Water boils at one hundred degrees Celsius at sea level.", None),
    ("Paris is the capital city of France and sits on the Seine.", None),
    ("Cats sleep for many hours each day in warm sunny places.", None),
    ("The Earth orbits the Sun once every three hundred sixty five days.", None),
    ("Bread is made from flour and water mixed together and baked.", None),
    ("Winter is generally colder than summer in most northern places.", None),
    ("Books contain printed pages that are bound along one edge.", None),
    ("Cars typically have four wheels and an engine under the hood.", None),

    ("Nadav Minkowitz is building a runtime system called NEURON.", True),
    ("The sun is bright and gives light to the whole planet.", False),
    ("Trees grow new green leaves during the spring months.", False),
    ("NEURON uses a surprise gate to decide when the model learns.", True),
    ("Coffee is a popular morning drink for people around the world.", False),
    ("Chairs are used for sitting and usually have four legs.", False),
    ("The gate fires only on the top twenty percent of incoming moments.", True),
    ("Rain falls from clouds when the water droplets get heavy.", False),
    ("Grass is green and grows in lawns and fields everywhere.", False),
    ("Telephones let people speak to each other across long distances.", False),
    ("Clickflo is an application that organizes screenshots automatically.", True),
    ("The ocean contains salt water and covers most of the planet.", False),
    ("Mink Studios produces short films and other video projects.", True),
    ("Birds have feathers and most of them are able to fly.", False),
    ("Shoes are worn on the feet to protect them while walking.", False),
    ("TROY Capital is a paper trading simulator for learning finance.", True),
    ("Fire is hot and produces both light and warmth when burning.", False),
    ("Windows are made of glass and let daylight into a room.", False),
    ("Hephaestus is the agent that writes code inside the system.", True),
    ("Snow is frozen water that falls during cold winter weather.", False),
]


def evaluate(name, scorer, top_fraction=0.20):
    gate = Gate(top_fraction=top_fraction)
    hits = misses = false_alarms = correct_skips = 0
    rows = []

    for text, label in STREAM:
        score = scorer(text)
        fired, threshold = gate.check(score)
        rows.append((score, threshold, fired, label, text))

        if label is None:
            continue
        if fired and label:
            hits += 1
        elif fired and not label:
            false_alarms += 1
        elif not fired and label:
            misses += 1
        else:
            correct_skips += 1

    print(f"\n=== {name} (top {top_fraction:.0%}) ===")
    print(f"{'score':>7} {'thresh':>7}  {'gate':>6} {'want':>6}   text")
    print("-" * 88)
    for score, threshold, fired, label, text in rows:
        if label is None:
            gate_str, want_str = "warmup", "-"
        else:
            gate_str = "LEARN" if fired else "."
            want_str = "LEARN" if label else "."
        flag = " <<<" if label is not None and fired != label else ""
        print(f"{score:7.3f} {threshold:7.3f}  {gate_str:>6} {want_str:>6}   {text[:44]}{flag}")

    total = hits + misses + false_alarms + correct_skips
    print("-" * 88)
    print(f"caught {hits}/{hits + misses} novel   "
          f"false alarms {false_alarms}/{false_alarms + correct_skips}   "
          f"accuracy {(hits + correct_skips) / total:.0%}")
    return hits, misses, false_alarms, correct_skips


evaluate("MEAN (original)", score_mean)
evaluate("MAX", score_max)
evaluate("TOP-3 MEAN", score_top3)