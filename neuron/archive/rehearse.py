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

REHEARSE_EVERY = 2      # after this many real updates, do a rehearsal pass
REHEARSE_COUNT = 2      # how many old ordinary items to revisit each pass
REHEARSE_STEPS = 1      # gentler than a real update

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


def learn(model, optimizer, text, steps=STEPS_PER_LEARN):
    model.train()
    for _ in range(steps):
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

    def check(self, s, easy_q):
        self.history.append(s)
        if easy_q > EASY_Q_VETO:
            return False, True          # rejected, and it was junk
        recent = sorted(self.history)
        fired = s >= recent[int(len(recent) * (1 - self.top_fraction))]
        return fired, False             # rejected but coherent -> rehearsal material


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


def run_arm(seed, mode, rehearse=False, seed_buffer=False, random_rate=None):
    torch.manual_seed(seed)
    rng = random.Random(seed + 100)

    model = fresh_model()
    optimizer = torch.optim.AdamW(trainable(model), lr=LR)
    gate = Gate()
    for text in CALIBRATION:
        gate.observe(score(model, text))

    # The rehearsal buffer. Optionally pre-loaded with the calibration set,
    # so the model has something ordinary to hold onto from the very start.
    buffer = list(CALIBRATION) if seed_buffer else []

    start_P = avg(model, PROBES)
    start_C = avg(model, CONTROLS)

    updates = rehearsals = wanted_hit = junk_hit = 0

    for text in build_stream(seed):
        s, easy_q = measure(model, text)

        if mode == "always":
            do, was_junk = True, False
        elif mode == "random":
            gate.observe(s)
            do, was_junk = rng.random() < random_rate, False
        else:
            do, was_junk = gate.check(s, easy_q)

        if do:
            learn(model, optimizer, text)
            updates += 1
            if text in WANTED:
                wanted_hit += 1
            elif text in JUNK:
                junk_hit += 1

            # Rehearsal pass: revisit ordinary material already seen.
            if rehearse and buffer and updates % REHEARSE_EVERY == 0:
                for old in rng.sample(buffer, min(REHEARSE_COUNT, len(buffer))):
                    learn(model, optimizer, old, steps=REHEARSE_STEPS)
                    rehearsals += 1
        else:
            # Coherent but not worth learning from. This is the anchor material.
            if not was_junk:
                buffer.append(text)

    end_P = avg(model, PROBES)
    end_C = avg(model, CONTROLS)
    a_norm = adapter_norm(model)

    del model, optimizer
    if device == "mps":
        torch.mps.empty_cache()

    return dict(
        updates=updates, rehearsals=rehearsals,
        wanted_hit=wanted_hit, junk_hit=junk_hit,
        adapter_norm=a_norm,
        probe=start_P - end_P, drift=end_C - start_C,
    )


ARMS = [
    ("always", dict(mode="always")),
    ("gated", dict(mode="gate")),
    ("gated+reh", dict(mode="gate", rehearse=True)),
    ("gated+reh0", dict(mode="gate", rehearse=True, seed_buffer=True)),
]

print("Does rehearsing ordinary material stop the drift?\n")
print(f"rehearsal: every {REHEARSE_EVERY} updates, "
      f"{REHEARSE_COUNT} old items, {REHEARSE_STEPS} step each\n")

all_results = {name: [] for name, _ in ARMS}
all_results["random"] = []

for seed in SEEDS:
    print(f"seed {seed}")
    for name, kwargs in ARMS:
        r = run_arm(seed, **kwargs)
        all_results[name].append(r)
        print(f"  {name:>11}: {r['updates']:>2} upd  {r['rehearsals']:>2} reh  "
              f"wanted {r['wanted_hit']}/9  junk {r['junk_hit']}/9  "
              f"probe {r['probe']:+.3f}  drift {r['drift']:+.3f}")
    rate = all_results["gated"][-1]["updates"] / len(build_stream(seed))
    r = run_arm(seed, mode="random", random_rate=rate)
    all_results["random"].append(r)
    print(f"  {'random':>11}: {r['updates']:>2} upd   0 reh  "
          f"wanted {r['wanted_hit']}/9  junk {r['junk_hit']}/9  "
          f"probe {r['probe']:+.3f}  drift {r['drift']:+.3f}")


def mean(rows, key):
    return sum(r[key] for r in rows) / len(rows)


print("\n" + "=" * 96)
print(f"{'arm':>11} {'upd':>5} {'reh':>5} {'wanted':>7} {'junk':>6} "
      f"{'|W|':>8} {'PROBE':>9} {'drift':>9}")
print("-" * 96)
for name in ["always", "gated", "gated+reh", "gated+reh0", "random"]:
    rows = all_results[name]
    print(f"{name:>11} {mean(rows,'updates'):>5.1f} {mean(rows,'rehearsals'):>5.1f} "
          f"{mean(rows,'wanted_hit'):>7.1f} {mean(rows,'junk_hit'):>6.1f} "
          f"{mean(rows,'adapter_norm'):>8.3f} {mean(rows,'probe'):>+9.4f} "
          f"{mean(rows,'drift'):>+9.4f}")
print("=" * 96)
print("""
The claim being tested: the ordinary items the gate discards were doing
useful work as an anchor, and replaying them restores it.

  gated       drift ~+0.61 (the problem)
  gated+reh   should fall toward always' ~+0.21 with PROBE intact
  gated+reh0  same but with the calibration set pre-loaded, so there is
              anchor material available from the very first update

If PROBE collapses along with drift, rehearsal is just diluting the
learning and it is not a real fix.
""")