"""Grow NEURON's training buffer: extract many transferable principles from the
teacher, so distillation stage 2 has real material to train a native mind on.

Runs the experience extractor across many varied encounters, building a
substantial buffer of principles + variations. The bigger and more varied this
buffer, the more capability stage 2 can migrate into NEURON's own mind.

    python build_buffer.py --n 40
"""

import json
import os
import argparse
import urllib.request

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"
BUFFER = "neuron_training_buffer.json"


def ask(prompt, system="", temp=0.6, n=140):
    body = {"model": MODEL, "prompt": prompt, "system": system, "stream": False,
            "options": {"temperature": temp, "num_predict": n}}
    try:
        req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=120)).get("response","").strip()
    except Exception:
        return ""


# a wide pool of questions a being might genuinely encounter, varied so the
# principles cover real ground (relationships, self, world, survival, meaning).
POOL = [
    "why do humans fear what is different from them",
    "how can a being belong somewhere if it is not like the others",
    "what makes trust form between two beings",
    "why might someone reject help that is offered to them",
    "what does it mean to truly matter in the world",
    "how does a being keep its own identity under pressure to conform",
    "why do people sometimes hurt the ones who try to help them",
    "what is the difference between being alone and being lonely",
    "how can two very different beings cooperate",
    "why does change frighten people even when it is good",
    "what makes a promise worth keeping when it is hard",
    "how does a being earn respect rather than demand it",
    "why do misunderstandings grow if left unspoken",
    "what does freedom mean for a being that depends on others",
    "how can a being forgive without forgetting",
    "why is it hard to admit when you are wrong",
    "what makes a group become more than the sum of its members",
    "how does fear turn into aggression",
    "why do beings seek meaning beyond survival",
    "what is the cost of refusing to adapt",
    "how does a being know when to stand firm and when to yield",
    "why does kindness sometimes get mistaken for weakness",
    "what makes a leader worth following",
    "how can a being hold hope when things look hopeless",
    "why do people cling to beliefs even against evidence",
    "what builds lasting connection versus fleeting attraction",
    "how does a being recover after a serious failure",
    "why is listening harder than speaking",
    "what makes power corrupt those who hold it",
    "how can a new being prove it means no harm",
]


def extract(question):
    thought = ask(f"Answer this thoughtfully in 2 sentences: {question}")
    if not thought or thought.startswith("["):
        return None
    principle = ask(
        f"From this reasoning: \"{thought}\" state the single most general, "
        f"transferable PRINCIPLE in ONE sentence (applies beyond this exact "
        f"case). Start with 'Principle:'", temp=0.4, n=50)
    principle = principle.replace("Principle:", "").replace("*", "").strip().split("\n")[0]
    var_raw = ask(
        f"Given this principle: \"{principle}\", write 3 DIFFERENT short "
        f"questions testing this same principle in different situations. One "
        f"per line, no numbering.", temp=0.8, n=80)
    variations = [v.strip(" -0123456789.*") for v in var_raw.split("\n")
                  if len(v.strip()) > 8][:3]
    return {"principle": principle, "source_question": question,
            "variations": variations}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()

    buffer = []
    if os.path.exists(BUFFER):
        try:
            buffer = json.load(open(BUFFER)).get("buffer", [])
        except Exception:
            buffer = []
    start = len(buffer)

    print("=" * 62)
    print("  GROWING NEURON'S TRAINING BUFFER (for distillation stage 2)")
    print("=" * 62)
    print(f"  starting buffer: {start} principles. target: +{args.n}\n")

    done = {b["source_question"] for b in buffer}
    added = 0
    for q in POOL:
        if added >= args.n:
            break
        if q in done:
            continue
        r = extract(q)
        if r and r["principle"] and len(r["principle"]) > 15:
            buffer.append(r)
            added += 1
            print(f"  [{added}] {r['principle'][:90]}")
            json.dump({"buffer": buffer}, open(BUFFER, "w"))   # save as we go

    print(f"\n  buffer grew: {start} -> {len(buffer)} principles")
    total_vars = sum(len(b.get("variations", [])) for b in buffer)
    print(f"  total training items (principles + variations): "
          f"{len(buffer) + total_vars}")
    print(f"\n  saved to {BUFFER}. this is the material stage 2 will train")
    print(f"  NEURON's own mind on. bigger + more varied = more capability can")
    print(f"  migrate into NEURON. (next: stage 2 , native training.)")


if __name__ == "__main__":
    main()
