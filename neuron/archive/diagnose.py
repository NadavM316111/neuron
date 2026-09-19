import torch
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


def stats(text):
    s = token_surprises(text)
    n = s.numel()
    sorted_s = s.sort().values
    q1 = sorted_s[: max(1, n // 4)].mean().item()   # the EASIEST quarter
    return {
        "n": n,
        "mean": s.mean().item(),
        "median": s.median().item(),
        "min": s.min().item(),
        "max": s.max().item(),
        "std": s.std().item() if n > 1 else 0.0,
        "easy_q": q1,
        "frac_hi": (s > 6.0).float().mean().item(),
    }


WANTED = [
    "He does all of his best writing late at night in the kitchen.",
    "The meeting was moved from Tuesday to the following Thursday morning.",
    "She never answers the phone when she is driving to work.",
    "The back door needs to be pulled up slightly before it will lock.",
    "They agreed to split the cost of the repair three ways.",
]

JUNK = [
    "zx qq plumb ferret 88 the the graaaa of nnnn under",
    "Error 0x8007F0B4 malformed packet retry buffer overflow at 0x00FA",
    "aaaaaa bbbbbb cccccc dddddd eeeeee ffffff gggggg hhhhhh",
    "wingle bortch flumdiddy narp narp quixling vestibule zorp",
    "PLEASE CLICK NOW!!! FREE!!! act fast limited $$$ offer wow",
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
]

# Novel AND unusual-looking, the case the gate already handles well.
NOVEL_ODD = [
    "Nadav Minkowitz is building a runtime system that he calls NEURON.",
    "The harbor master at Kelso Point keeps a logbook of every arriving vessel.",
    "Cochineal insects are crushed to produce a deep red natural dye.",
    "The bell ringers at Saint Alden practice a method called Grandsire Triples.",
]

GROUPS = [
    ("WANTED", WANTED),
    ("NOVEL_ODD", NOVEL_ODD),
    ("FILLER", FILLER),
    ("JUNK", JUNK),
]

KEYS = ["mean", "median", "min", "max", "std", "easy_q", "frac_hi"]

print(f"{'group':>10} {'n':>3}  " + "  ".join(f"{k:>7}" for k in KEYS) + "   text")
print("-" * 110)

collected = {}
for name, items in GROUPS:
    collected[name] = []
    for text in items:
        st = stats(text)
        collected[name].append(st)
        row = "  ".join(f"{st[k]:7.3f}" for k in KEYS)
        print(f"{name:>10} {st['n']:>3}  {row}   {text[:34]}")
    print()

print("=" * 110)
print("GROUP AVERAGES")
print(f"{'group':>10}  " + "  ".join(f"{k:>7}" for k in KEYS))
print("-" * 110)
avgs = {}
for name, _ in GROUPS:
    rows = collected[name]
    avgs[name] = {k: sum(r[k] for r in rows) / len(rows) for k in KEYS}
    print(f"{name:>10}  " + "  ".join(f"{avgs[name][k]:7.3f}" for k in KEYS))

print("\n" + "=" * 110)
print("SEPARATION: how far JUNK sits from the real-text groups on each statistic")
print("(bigger gap = better junk detector; sign shows direction)")
print("-" * 110)
real_names = ["WANTED", "NOVEL_ODD", "FILLER"]
for k in KEYS:
    junk_v = avgs["JUNK"][k]
    real_v = sum(avgs[n][k] for n in real_names) / len(real_names)
    spread = max(abs(avgs[n][k] - real_v) for n in real_names) + 1e-6
    ratio = (junk_v - real_v) / spread
    print(f"  {k:>8}: junk {junk_v:7.3f}  real {real_v:7.3f}  "
          f"gap {junk_v - real_v:+7.3f}  normalized {ratio:+6.2f}")

print("""
WHAT TO LOOK FOR

  mean     the signal you use now. Junk should be high, but so is NOVEL_ODD.
  easy_q   average surprise of the EASIEST quarter of tokens.
           Real sentences always contain a few free words. Junk should not.
           If junk's easy_q is far above everything else, you have a free
           junk filter that costs nothing extra to compute.
  frac_hi  fraction of tokens the model found genuinely hard.

The statistic with the largest normalized separation is your junk detector.
""")