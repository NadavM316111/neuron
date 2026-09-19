"""Can a 7B model faithfully restate a fact it just read?

At 1.5B the rejection rate was 65% (paraphrase arm) and 86% (both arm).
Generative consolidation requires the base model to paraphrase without
losing the key content, so that number is what blocked the whole approach.

This measures it directly, in about five minutes, before spending an hour
on the full QA run.

  under ~30% rejection -> the approach is viable, run the full test
  still 60%+          -> gradient-based fact injection is blocked at any
                         scale you can reach, and the plan changes
"""

import re
import time
import torch
from llm_backend import LLMBackend


MODEL_7B = "Qwen/Qwen2.5-7B-Instruct"

STOP = set("""a an the and or but if of to in on at for with from by is are was
were be been being it its this that these those he she they them his her their
you your i my we our as not no do does did done have has had will would can
could should than then there here when what which who how all any some most
more much many very just also only over under up down out into about after
before while during""".split())

FACTS = [
    "Ottoline Verrick composed nineteen string quartets before she turned thirty.",
    "The Brantwood ferry runs only between April and the end of September.",
    "The Halverson mill on Petrie Creek stopped grinding flour in nineteen forty.",
    "Marguerite Follansbee catalogued eleven thousand moths in her lifetime.",
    "Tobias Wrenn mapped the Corrieshalloch caves over four separate summers.",
    "Kestrel Bay oysters are harvested by hand at low tide in winter.",
]

# The specific values that must survive a paraphrase.
KEY_VALUES = [
    ["nineteen", "19"],
    ["april", "september"],
    ["nineteen forty", "1940"],
    ["eleven thousand", "11,000", "11000"],
    ["four", "4"],
    ["hand"],
]

N_PARAPHRASES = 5
FAITHFUL_MIN = 0.55


def content_tokens(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOP and len(w) > 2}


def overlap(source, variant):
    src = content_tokens(source)
    if not src:
        return 0.0
    return len(src & content_tokens(variant)) / len(src)


def keeps_value(variant, accepted):
    v = variant.lower()
    return any(k.lower() in v for k in accepted)


if __name__ == "__main__":
    t0 = time.time()
    print(f"Loading {MODEL_7B} ...")
    backend = LLMBackend(model_name=MODEL_7B)
    print(f"device {backend.device}   loaded in {time.time()-t0:.0f}s")
    print(f"trainable {backend.trainable_count:,}\n")

    total = kept = value_kept = 0
    per_fact = []

    for fact, accepted in zip(FACTS, KEY_VALUES):
        print("=" * 78)
        print(fact)
        print("-" * 78)
        prompt = (f"Rewrite this sentence in different words. Keep every "
                  f"number, name, and detail exactly the same. Give one "
                  f"sentence only, no preamble.\n\n{fact}")
        f_kept = f_value = 0
        for i in range(N_PARAPHRASES):
            v = backend.generate(prompt, max_new_tokens=60, temperature=0.8)
            v = v.split("\n")[0].strip().strip('"')
            total += 1
            ov = overlap(fact, v)
            faithful = len(v) > 15 and ov >= FAITHFUL_MIN
            has_value = keeps_value(v, accepted)
            if faithful:
                kept += 1
                f_kept += 1
            if has_value:
                value_kept += 1
                f_value += 1
            mark = "KEEP" if faithful else "rej "
            val = "val" if has_value else "   "
            print(f"  {mark} {val} ov={ov:.2f}  {v[:88]}")
        per_fact.append((f_kept, f_value))
        print()

    rej = 100 * (total - kept) / total
    print("=" * 78)
    print(f"generated {total}   kept {kept}   rejected {total-kept}   "
          f"REJECTION {rej:.0f}%")
    print(f"kept the key value in {value_kept}/{total} "
          f"({100*value_kept/total:.0f}%)")
    print("=" * 78)
    print(f"""
1.5B rejection was 65% (paraphrase) and 86% (both).
This run: {rej:.0f}%.

  under ~30%  -> viable, run the full QA test
  30 to 50%   -> marginal, worth trying but expect noise
  over 60%    -> the model still cannot restate its own facts, and
                 generative consolidation is blocked at this scale too

The 'kept the key value' number is the stricter one and matters more.
A paraphrase that keeps the topic but drops 'nineteen' is useless for
teaching the fact.
""")