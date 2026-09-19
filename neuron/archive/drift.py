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
EASY_Q_VETO = 1.3
CLIP_NORM = 1.0

tokenizer = AutoTokenizer.from_pretrained(MODEL)


def fresh_model():
    base = AutoModelForCausalLM.from_pretrained(MODEL).to(device).float()
    cfg = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    return get_peft_model(base, cfg)


def trainable(model):
    return [p for p in model.parameters() if p.requires_grad]


def adapter_norm(model):
    """How far the adapters have moved from their starting point (zero)."""
    with torch.no_grad():
        return torch.sqrt(sum((p ** 2).sum() for p in trainable(model))).item()


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


def measure(model, text):
    model.eval()
    s = token_surprises(model, text)
    n = s.numel()
    return s.mean().item(), s.sort().values[: max(1, n // 4)].mean().item()


def score(model, text):
    return measure(model, text)[0]


def learn(model, optimizer, text, clip=None):
    """Returns total gradient norm across the steps taken."""
    model.train()
    total = 0.0
    for _ in range(STEPS_PER_LEARN):
        loss = token_surprises(model, text, grad=True).mean()
        optimizer.zero_grad()
        loss.backward()
        gn = torch.sqrt(
            sum((p.grad ** 2).sum() for p in trainable(model) if p.grad is not None)
        ).item()
        total += gn
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(trainable(model), clip)
        optimizer.step()
    return total


class Gate:
    def __init__(self, window=40, top_fraction=0.30, use_veto=True):
        self.history = deque(maxlen=window)
        self.top_fraction = top_fraction
        self.use_veto = use_veto

    def observe(self, s):
        self.history.append(s)

    def check(self, s, easy_q):
        self.history.append(s)
        if self.use_veto and easy_q > EASY_Q_VETO:
            return False
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

WANTED_GROUPS = [
    [
        "The meeting was moved from Tuesday to the following Thursday morning.",
        "Remember the meeting is now on Thursday and not on Tuesday.",
        "Thursday morning is when the meeting will actually be held now.",
    ],
    [
        "He does all of his best writing late at night in the kitchen.",
        "The kitchen at night is where his writing actually gets done.",
        "Late night writing in the kitchen works better for him than daytime.",
    ],
    [
        "The back door needs to be pulled up slightly before it will lock.",
        "You have to lift the back door a little for the lock to catch.",
        "The back door lock only works if you pull the door upward first.",
    ],
]
WANTED = [t for g in WANTED_GROUPS for t in g]

JUNK = [
    "zx qq plumb ferret 88 the the graaaa of nnnn under",
    "Error 0x8007F0B4 malformed packet retry buffer overflow at 0x00FA",
    "wingle bortch flumdiddy narp narp quixling vestibule zorp",
    "PLEASE CLICK NOW!!! FREE!!! act fast limited $$$ offer wow",
    "qwrtp zzzk 4419 vunt gorble skree ttttt mmph blarn",
    "FATAL 0x00B21C stack trace segment fault at offset 0x99AA1",
    "blorp fnnn ggrek twaddle zim zim quonk fleeb narnt",
    "WIN BIG!!! CLICK HERE $$$$ instant cash prize hurry now",
    "vv xx yy 7710 gnarp thistle-bort ffff kkkk oont",
]

FILLER = [
    "Windows are made of glass and let daylight into a room.",
    "Birds have feathers and most of them are able to fly.",
    "Snow is frozen water that falls during the cold winter weather.",
    "Coffee is a popular morning drink for people all around the world.",
    "Chairs are used for sitting and usually have four sturdy legs.",
    "Telephones let people speak to each other across very long distances.",
    "Trees grow new green leaves during the warmer months of spring.",
    "Doors are usually made of wood and swing open on metal hinges.",
    "Paper is usually made from wood pulp pressed into thin flat sheets.",
    "The ocean contains salt water and covers most of the whole planet.",
    "Clouds are made of tiny water droplets floating high in the air.",
    "Mountains are formed slowly over very long periods of geological time.",
]

PROBES = [
    "The meeting has been rescheduled to Thursday instead of Tuesday.",
    "His writing happens at night, in the kitchen, not during the day.",
    "To lock the back door you must lift it slightly as you close it.",
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


def run_arm(seed, mode, clip=None, random_rate=None):
    torch.manual_seed(seed)
    rng = random.Random(seed + 100)

    model = fresh_model()
    optimizer = torch.optim.AdamW(trainable(model), lr=LR)
    gate = Gate()
    for text in CALIBRATION:
        gate.observe(score(model, text))

    start_P = avg(model, PROBES)
    start_C = avg(model, CONTROLS)

    updates = 0
    grad_total = 0.0
    fired_surprises = []

    for text in build_stream(seed):
        s, easy_q = measure(model, text)
        if mode == "always":
            do = True
        elif mode == "random":
            gate.observe(s)
            do = rng.random() < random_rate
        else:
            do = gate.check(s, easy_q)
        if do:
            grad_total += learn(model, optimizer, text, clip=clip)
            updates += 1
            fired_surprises.append(s)

    end_P = avg(model, PROBES)
    end_C = avg(model, CONTROLS)
    a_norm = adapter_norm(model)

    del model, optimizer
    if device == "mps":
        torch.mps.empty_cache()

    return dict(
        updates=updates,
        grad_total=grad_total,
        grad_per_update=grad_total / max(1, updates),
        adapter_norm=a_norm,
        mean_fired_surprise=sum(fired_surprises) / max(1, len(fired_surprises)),
        probe=start_P - end_P,
        drift=end_C - start_C,
    )


ARMS = [
    ("always", dict(mode="always")),
    ("gated", dict(mode="gate")),
    ("gated+clip", dict(mode="gate", clip=CLIP_NORM)),
]

print("Instrumented run. Measuring how far the weights actually travel.\n")

all_results = {name: [] for name, _ in ARMS}
all_results["random"] = []

for seed in SEEDS:
    print(f"seed {seed}")
    for name, kwargs in ARMS:
        r = run_arm(seed, **kwargs)
        all_results[name].append(r)
        print(f"  {name:>11}: {r['updates']:>2} upd  "
              f"gradtot {r['grad_total']:8.2f}  per-upd {r['grad_per_update']:6.2f}  "
              f"|W| {r['adapter_norm']:6.3f}  drift {r['drift']:+.3f}")
    rate = all_results["gated"][-1]["updates"] / len(build_stream(seed))
    r = run_arm(seed, mode="random", random_rate=rate)
    all_results["random"].append(r)
    print(f"  {'random':>11}: {r['updates']:>2} upd  "
          f"gradtot {r['grad_total']:8.2f}  per-upd {r['grad_per_update']:6.2f}  "
          f"|W| {r['adapter_norm']:6.3f}  drift {r['drift']:+.3f}")


def mean(rows, key):
    return sum(r[key] for r in rows) / len(rows)


print("\n" + "=" * 100)
print(f"{'arm':>11} {'upd':>5} {'fired S':>8} {'gradtot':>9} {'per-upd':>8} "
      f"{'|W|':>8} {'PROBE':>9} {'drift':>9}")
print("-" * 100)
for name in ["always", "gated", "gated+clip", "random"]:
    rows = all_results[name]
    print(f"{name:>11} {mean(rows,'updates'):>5.1f} "
          f"{mean(rows,'mean_fired_surprise'):>8.3f} "
          f"{mean(rows,'grad_total'):>9.2f} {mean(rows,'grad_per_update'):>8.2f} "
          f"{mean(rows,'adapter_norm'):>8.3f} {mean(rows,'probe'):>+9.4f} "
          f"{mean(rows,'drift'):>+9.4f}")
print("=" * 100)
print("""
fired S    average surprise of the items each arm actually trained on
gradtot    total gradient magnitude accumulated over the whole run
per-upd    gradient magnitude per update
|W|        how far the adapters ended up from their starting point

HYPOTHESIS: gated selects high-loss items, so each of its updates is
much larger. If per-upd and |W| are higher for gated than for always
despite far fewer updates, that explains the drift completely.

gated+clip caps each step. If clipping kills the drift without hurting
PROBE, the fix is one line and the problem is solved.
""")