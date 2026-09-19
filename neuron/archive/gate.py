import torch
from collections import deque
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
device = "mps"

print("Loading...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL).to(device).float()
model.eval()
print("Loaded.\n")


def token_surprises(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    ids = inputs["input_ids"]
    with torch.no_grad():
        logits = model(**inputs).logits
    preds = logits[:, :-1, :]
    targets = ids[:, 1:]
    return torch.nn.functional.cross_entropy(
        preds.reshape(-1, preds.size(-1)),
        targets.reshape(-1),
        reduction="none",
    )


def score(text):
    """Mean per-token surprise. Won the gate3 comparison."""
    return token_surprises(text).mean().item()


class Gate:
    """Fires on the top slice of recent surprise. Relative, so it
    survives the model changing underneath it."""

    def __init__(self, window=60, top_fraction=0.30, warmup=15):
        self.history = deque(maxlen=window)
        self.top_fraction = top_fraction
        self.warmup = warmup
        self.seen = 0

    def check(self, s):
        self.history.append(s)
        self.seen += 1
        if self.seen <= self.warmup:
            return None, 0.0
        recent = sorted(self.history)
        threshold = recent[int(len(recent) * (1 - self.top_fraction))]
        return s >= threshold, threshold


# (text, should_fire, category)
# NP = novel + proper noun   NN = novel, plain words
# ON = ordinary + proper noun   OO = ordinary, plain words
STREAM = [
    ("The sky is blue on a clear day and the clouds are white.", None, "CAL"),
    ("Dogs are common household pets that many families keep at home.", None, "CAL"),
    ("Water boils at one hundred degrees Celsius at normal sea level.", None, "CAL"),
    ("Cats sleep for many hours each day in warm sunny places.", None, "CAL"),
    ("Bread is made from flour and water mixed together and baked.", None, "CAL"),
    ("Winter is generally colder than summer in most northern places.", None, "CAL"),
    ("Books contain printed pages that are bound together along one edge.", None, "CAL"),
    ("Cars typically have four wheels and an engine under the hood.", None, "CAL"),
    ("Rain falls from the clouds when the water droplets become heavy.", None, "CAL"),
    ("Grass is green and grows in lawns and open fields everywhere.", None, "CAL"),
    ("Shoes are worn on the feet to protect them while walking.", None, "CAL"),
    ("Fire is hot and produces both light and warmth while burning.", None, "CAL"),
    ("Windows are made of glass and let daylight into a room.", None, "CAL"),
    ("Birds have feathers and most of them are able to fly.", None, "CAL"),
    ("Snow is frozen water that falls during the cold winter weather.", None, "CAL"),

    ("Nadav Minkowitz is building a runtime system that he calls NEURON.", True, "NP"),
    ("Clickflo is an application that organizes screenshots automatically for you.", True, "NP"),
    ("Mink Studios produces short films and various other video projects.", True, "NP"),
    ("TROY Capital is a paper trading simulator built for learning finance.", True, "NP"),
    ("Hephaestus is the agent that writes the code inside the system.", True, "NP"),
    ("Sokr and Bookly were two earlier projects that never fully shipped.", True, "NP"),
    ("Static Rebellion is a band that plays original songs at benefit shows.", True, "NP"),

    ("My younger cousin refuses to eat any food that is green.", True, "NN"),
    ("The third drawer in the kitchen sticks unless you lift it slightly.", True, "NN"),
    ("He writes all of his best song lyrics while sitting in traffic.", True, "NN"),
    ("The upstairs bathroom light flickers only on rainy days for some reason.", True, "NN"),
    ("She keeps exactly nine identical black notebooks stacked beside her bed.", True, "NN"),
    ("The old bicycle in the garage has a bent frame and no seat.", True, "NN"),
    ("His father taught him to always fold the map along the old creases.", True, "NN"),

    ("Paris is the capital city of France and a major tourist destination.", False, "ON"),
    ("Amazon sells books and many other products to customers over the internet.", False, "ON"),
    ("Einstein developed the theory of relativity in the early twentieth century.", False, "ON"),
    ("The Pacific Ocean is the largest ocean on the surface of Earth.", False, "ON"),
    ("Shakespeare wrote many famous plays including Hamlet and Romeo and Juliet.", False, "ON"),
    ("Microsoft produces the Windows operating system used on many computers.", False, "ON"),
    ("The Nile is a long river that flows northward through northeastern Africa.", False, "ON"),

    ("Coffee is a popular morning drink for people all around the world.", False, "OO"),
    ("Chairs are used for sitting and usually have four sturdy legs.", False, "OO"),
    ("Telephones let people speak to each other across very long distances.", False, "OO"),
    ("The ocean contains salt water and covers most of the whole planet.", False, "OO"),
    ("Trees grow new green leaves during the warmer months of spring.", False, "OO"),
    ("The sun is bright and gives light to the entire planet below.", False, "OO"),
    ("Doors are usually made of wood and swing open on metal hinges.", False, "OO"),
]

gate = Gate()
rows = []
for text, label, cat in STREAM:
    s = score(text)
    fired, threshold = gate.check(s)
    rows.append((s, threshold, fired, label, cat, text))

print(f"{'score':>7} {'thresh':>7} {'cat':>4} {'gate':>6} {'want':>6}   text")
print("-" * 96)

stats = {}
for s, threshold, fired, label, cat, text in rows:
    if label is None:
        gate_str, want_str, flag = "warmup", "-", ""
    else:
        gate_str = "LEARN" if fired else "."
        want_str = "LEARN" if label else "."
        flag = " <<<" if fired != label else ""
        d = stats.setdefault(cat, {"right": 0, "wrong": 0})
        d["right" if fired == label else "wrong"] += 1
    print(f"{s:7.3f} {threshold:7.3f} {cat:>4} {gate_str:>6} {want_str:>6}   {text[:42]}{flag}")

print("-" * 96)
right = sum(d["right"] for d in stats.values())
total = sum(d["right"] + d["wrong"] for d in stats.values())
for cat in ["NP", "NN", "ON", "OO"]:
    if cat in stats:
        d = stats[cat]
        print(f"  {cat}: {d['right']}/{d['right'] + d['wrong']} correct")
print(f"  OVERALL: {right}/{total} = {right/total:.0%}")