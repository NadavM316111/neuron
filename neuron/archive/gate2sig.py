import re
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

EASY_Q_VETO = 1.3      # above this, treat as incoherent and refuse
ECHO_WEIGHT = 2.0      # how much repetition boosts a low-surprise item

tokenizer = AutoTokenizer.from_pretrained(MODEL)

STOPWORDS = set("""a an the and or but if of to in on at for with from by is are was
were be been being it its this that these those he she they them his her their you
your i my we our as not no do does did done have has had will would can could should
than then there here when what which who how all any some most more much many very
just also only over under up down out into about after before while during""".split())


def content_words(text):
    words = re.findall(r"[a-z]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


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


def measure(model, text):
    """One forward pass gives us both signals. No extra cost."""
    model.eval()
    s = token_surprises(model, text)
    n = s.numel()
    easy_q = s.sort().values[: max(1, n // 4)].mean().item()
    return s.mean().item(), easy_q


def score(model, text):
    return measure(model, text)[0]


def learn(model, optimizer, text):
    model.train()
    for _ in range(STEPS_PER_LEARN):
        loss = token_surprises(model, text, grad=True).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class TwoSignalGate:
    """Fires on novelty, vetoes incoherence, boosts what recurs."""

    def __init__(self, window=40, top_fraction=0.30,
                 easy_veto=EASY_Q_VETO, echo_weight=ECHO_WEIGHT,
                 use_veto=True, use_echo=True):
        self.history = deque(maxlen=window)
        self.top_fraction = top_fraction
        self.easy_veto = easy_veto
        self.echo_weight = echo_weight
        self.use_veto = use_veto
        self.use_echo = use_echo
        self.recent_words = deque(maxlen=25)

    def _echo(self, text):
        """Fraction of this item's content words seen recently."""
        words = content_words(text)
        if not words or not self.recent_words:
            return 0.0
        seen = set().union(*self.recent_words)
        return len(words & seen) / len(words)

    def _combined(self, surprise, text):
        if not self.use_echo:
            return surprise
        return surprise * (1.0 + self.echo_weight * self._echo(text))

    def observe(self, surprise, text):
        self.history.append(self._combined(surprise, text))
        self.recent_words.append(content_words(text))

    def check(self, surprise, easy_q, text):
        combined = self._combined(surprise, text)
        self.history.append(combined)
        self.recent_words.append(content_words(text))

        if self.use_veto and easy_q > self.easy_veto:
            return False, "VETO"

        recent = sorted(self.history)
        threshold = recent[int(len(recent) * (1 - self.top_fraction))]
        return combined >= threshold, "fire" if combined >= threshold else "."


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

# Each WANTED topic appears THREE times in different wording.
# This is the realistic part: things that matter come up again.
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

# Junk appears once each and never returns. Nothing echoes it.
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

# Held-out probes: the SAME facts, worded differently again.
# Tests whether the fact transferred, not whether the sentence memorized.
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


def run_arm(name, seed, mode, use_veto=True, use_echo=True, random_rate=None):
    torch.manual_seed(seed)
    rng = random.Random(seed + 100)

    model = fresh_model()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )
    gate = TwoSignalGate(use_veto=use_veto, use_echo=use_echo)
    for text in CALIBRATION:
        gate.observe(score(model, text), text)

    start_W = avg(model, WANTED)
    start_P = avg(model, PROBES)
    start_J = avg(model, JUNK)
    start_C = avg(model, CONTROLS)

    wanted_hit = junk_hit = updates = vetoed = 0

    for text in build_stream(seed):
        surprise, easy_q = measure(model, text)
        if mode == "always":
            do = True
        elif mode == "random":
            gate.observe(surprise, text)
            do = rng.random() < random_rate
        else:
            do, why = gate.check(surprise, easy_q, text)
            if why == "VETO":
                vetoed += 1
        if do:
            learn(model, optimizer, text)
            updates += 1
            if text in WANTED:
                wanted_hit += 1
            elif text in JUNK:
                junk_hit += 1

    end_W = avg(model, WANTED)
    end_P = avg(model, PROBES)
    end_J = avg(model, JUNK)
    end_C = avg(model, CONTROLS)

    del model, optimizer
    if device == "mps":
        torch.mps.empty_cache()

    return dict(
        updates=updates, vetoed=vetoed,
        wanted_hit=wanted_hit, junk_hit=junk_hit,
        W=start_W - end_W, P=start_P - end_P,
        J=start_J - end_J, drift=end_C - start_C,
    )


ARMS = [
    ("always", dict(mode="always")),
    ("surprise", dict(mode="gate", use_veto=False, use_echo=False)),
    ("+veto", dict(mode="gate", use_veto=True, use_echo=False)),
    ("+echo", dict(mode="gate", use_veto=False, use_echo=True)),
    ("both", dict(mode="gate", use_veto=True, use_echo=True)),
]

print(f"Stream: {len(WANTED)} wanted (3 facts x 3 wordings), "
      f"{len(JUNK)} junk, {len(FILLER)} filler\n")

all_results = {name: [] for name, _ in ARMS}
all_results["random"] = []

for seed in SEEDS:
    print(f"seed {seed}")
    for name, kwargs in ARMS:
        r = run_arm(name, seed, **kwargs)
        all_results[name].append(r)
        print(f"  {name:>9}: {r['updates']:>2} upd  "
              f"wanted {r['wanted_hit']:>2}/9  junk {r['junk_hit']}/9  "
              f"veto {r['vetoed']:>2}  probe {r['P']:+.3f}")
    rate = all_results["both"][-1]["updates"] / len(build_stream(seed))
    r = run_arm("random", seed, mode="random", random_rate=rate)
    all_results["random"].append(r)
    print(f"  {'random':>9}: {r['updates']:>2} upd  "
          f"wanted {r['wanted_hit']:>2}/9  junk {r['junk_hit']}/9  "
          f"veto  -  probe {r['P']:+.3f}")


def mean(rows, key):
    return sum(r[key] for r in rows) / len(rows)


print("\n" + "=" * 92)
print(f"{'arm':>9} {'upd':>5} {'wanted':>7} {'junk':>6} {'veto':>5} "
      f"{'W gain':>9} {'PROBE':>9} {'J gain':>9} {'drift':>9}")
print("-" * 92)
for name in ["always", "surprise", "+veto", "+echo", "both", "random"]:
    rows = all_results[name]
    print(f"{name:>9} {mean(rows,'updates'):>5.1f} {mean(rows,'wanted_hit'):>7.1f} "
          f"{mean(rows,'junk_hit'):>6.1f} {mean(rows,'vetoed'):>5.1f} "
          f"{mean(rows,'W'):>+9.4f} {mean(rows,'P'):>+9.4f} "
          f"{mean(rows,'J'):>+9.4f} {mean(rows,'drift'):>+9.4f}")
print("=" * 92)
print("""
PROBE is the honest number. Those sentences are the same three facts in
wording the model never trained on. Gains there mean the FACT transferred.
W gain can rise from pure memorization; PROBE cannot.

Read down the ablation:
  surprise -> +veto   should cut junk with no cost to wanted
  +veto    -> both    should raise wanted via repetition
  both vs random      the whole thing versus chance at equal update count
""")