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

# WANTED: genuinely new information, written in completely ordinary words.
# Low surprise by construction. The gate should catch these and probably won't.
WANTED = [
    "He does all of his best writing late at night in the kitchen.",
    "The meeting was moved from Tuesday to the following Thursday morning.",
    "She never answers the phone when she is driving to work.",
    "The back door needs to be pulled up slightly before it will lock.",
    "They agreed to split the cost of the repair three ways.",
]

# JUNK: high surprise, zero value. The gate should ignore these and probably won't.
JUNK = [
    "zx qq plumb ferret 88 the the graaaa of nnnn under",
    "Error 0x8007F0B4 malformed packet retry buffer overflow at 0x00FA",
    "aaaaaa bbbbbb cccccc dddddd eeeeee ffffff gggggg hhhhhh",
    "wingle bortch flumdiddy narp narp quixling vestibule zorp",
    "PLEASE CLICK NOW!!! FREE!!! act fast limited $$$ offer wow",
]

# Ordinary filler nobody should learn from.
FILLER = [
    "Windows are made of glass and let daylight into a room.",
    "Birds have feathers and most of them are able to fly.",
    "Snow is frozen water that falls during the cold winter weather.",
    "Coffee is a popular morning drink for people all around the world.",
    "Chairs are used for sitting and usually have four sturdy legs.",
    "Telephones let people speak to each other across very long distances.",
    "Trees grow new green leaves during the warmer months of spring.",
    "Doors are usually made of wood and swing open on metal hinges.",
    "The ocean contains salt water and covers most of the whole planet.",
    "Paper is usually made from wood pulp pressed into thin flat sheets.",
]

CONTROLS = [
    "Paris is the capital city of France and a major tourist destination.",
    "The Pacific Ocean is the largest ocean on the surface of Earth.",
    "Einstein developed the theory of relativity in the early twentieth century.",
    "Shakespeare wrote many famous plays including Hamlet and Romeo and Juliet.",
]


def build_stream(seed):
    stream = WANTED + JUNK + FILLER
    random.Random(seed).shuffle(stream)
    return stream


def avg(model, texts):
    return sum(score(model, t) for t in texts) / len(texts)


def run_arm(mode, seed, random_rate=None):
    torch.manual_seed(seed)
    rng = random.Random(seed + 100)

    model = fresh_model()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )
    gate = Gate()
    for text in CALIBRATION:
        gate.observe(score(model, text))

    start_W = avg(model, WANTED)
    start_J = avg(model, JUNK)
    start_C = avg(model, CONTROLS)

    wanted_hit = junk_hit = updates = 0

    for text in build_stream(seed):
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
            updates += 1
            if text in WANTED:
                wanted_hit += 1
            elif text in JUNK:
                junk_hit += 1

    end_W = avg(model, WANTED)
    end_J = avg(model, JUNK)
    end_C = avg(model, CONTROLS)

    del model, optimizer
    if device == "mps":
        torch.mps.empty_cache()

    return dict(
        updates=updates,
        wanted_hit=wanted_hit,
        junk_hit=junk_hit,
        wanted_gain=start_W - end_W,
        junk_gain=start_J - end_J,
        drift=end_C - start_C,
    )


print("Stress test: what matters and what is surprising are now DIFFERENT things.\n")
print("5 WANTED items (new info, plain words, low surprise)")
print("5 JUNK items (nonsense, high surprise, worthless)")
print("10 FILLER items (ordinary, ignore)\n")

all_results = {"frozen": [], "always": [], "gated": [], "random": []}

for seed in SEEDS:
    print(f"seed {seed}")
    for mode in ["frozen", "always", "gated"]:
        r = run_arm(mode, seed)
        all_results[mode].append(r)
        print(f"  {mode:>7}: {r['updates']:>2} updates  "
              f"wanted {r['wanted_hit']}/5  junk {r['junk_hit']}/5")
    rate = all_results["gated"][-1]["updates"] / 20
    r = run_arm("random", seed, random_rate=rate)
    all_results["random"].append(r)
    print(f"  {'random':>7}: {r['updates']:>2} updates  "
          f"wanted {r['wanted_hit']}/5  junk {r['junk_hit']}/5")


def mean(rows, key):
    return sum(r[key] for r in rows) / len(rows)


print("\n" + "=" * 88)
print(f"{'arm':>8} {'updates':>8} {'wanted':>8} {'junk':>7} "
      f"{'W gain':>10} {'J gain':>10} {'drift':>10}")
print("-" * 88)
for name in ["frozen", "always", "gated", "random"]:
    rows = all_results[name]
    print(f"{name:>8} {mean(rows,'updates'):>8.1f} {mean(rows,'wanted_hit'):>8.1f} "
          f"{mean(rows,'junk_hit'):>7.1f} {mean(rows,'wanted_gain'):>+10.4f} "
          f"{mean(rows,'junk_gain'):>+10.4f} {mean(rows,'drift'):>+10.4f}")
print("=" * 88)
print("""
wanted  how many of the 5 real-info items each arm trained on   HIGH is good
junk    how many of the 5 nonsense items each arm trained on    LOW is good
W gain  how much the real info sank in                          HIGH is good
J gain  how much nonsense got absorbed                          LOW is good

The number that decides this: gated's JUNK count.
If gated hits 5/5 junk and 1/5 wanted, pure surprise is the wrong signal
and the gate needs a second input beyond novelty.
""")