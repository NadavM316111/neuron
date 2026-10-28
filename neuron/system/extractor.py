"""Real distillation, stage 1: the experience extractor.

The research insight: to make capability its OWN, NEURON should not store the
teacher's answers, it should extract the general, transferable PRINCIPLE from
each experience, and generate variations that test that principle. Those
principles + variations become NEURON's lifelong training buffer, the material a
native NEURON mind (LoRA/adapter) later trains on to genuinely GENERALIZE, not
memorize.

This stage builds the extractor: NEURON uses the teacher, then extracts a
principle and makes variations, building its training buffer. (Stage 2: native
consolidation/training. Stage 3: held-out test that proves generalization.)

    python extractor.py
"""

import json
import os
import urllib.request

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "gemma2:2b"
BUFFER = "neuron_training_buffer.json"


def ask(prompt, system="", temp=0.6, n=160):
    body = {"model": MODEL, "prompt": prompt, "system": system, "stream": False,
            "options": {"temperature": temp, "num_predict": n}}
    try:
        req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=120)).get("response","").strip()
    except Exception:
        return ""


class Distiller:
    def __init__(self):
        self.buffer = []       # NEURON's lifelong training buffer of principles
        self.exp_count = 0
        self.load()

    def load(self):
        if os.path.exists(BUFFER):
            try:
                d = json.load(open(BUFFER))
                self.buffer = d.get("buffer", [])
                self.exp_count = d.get("exp_count", 0)
            except Exception:
                pass

    def save(self):
        json.dump({"buffer": self.buffer, "exp_count": self.exp_count},
                  open(BUFFER, "w"))

    def experience(self, question):
        self.exp_count += 1
        print(f"  [experience #{self.exp_count}] NEURON encounters: \"{question}\"")

        # 1. NEURON thinks with the teacher
        thought = ask(f"Answer this thoughtfully in 2 sentences: {question}")
        print(f"    teacher's thinking: {thought[:130]}...")

        # 2. EXTRACT the transferable PRINCIPLE (not the answer) - the key step
        principle = ask(
            f"From this reasoning: \"{thought}\" about the question "
            f"\"{question}\", state the single most general, transferable "
            f"PRINCIPLE in ONE short sentence (a lesson that applies beyond this "
            f"exact question). Start with 'Principle:'", temp=0.4, n=60)
        principle = principle.replace("Principle:", "").strip().split("\n")[0]
        print(f"    -> extracted principle: {principle[:120]}")

        # 3. generate VARIATIONS that test the SAME principle (for generalization)
        var_raw = ask(
            f"Given this principle: \"{principle}\", write 3 DIFFERENT short "
            f"questions that test understanding of this same principle in "
            f"different situations. One per line, no numbering.", temp=0.8, n=90)
        variations = [v.strip(" -0123456789.") for v in var_raw.split("\n")
                      if len(v.strip()) > 8][:3]
        print(f"    -> training variations generated: {len(variations)}")
        for v in variations:
            print(f"         - {v[:80]}")

        # 4. into the lifelong training buffer (what native NEURON will train on)
        self.buffer.append({
            "principle": principle,
            "source_question": question,
            "variations": variations,
            "evidence": f"experience #{self.exp_count}",
        })
        print()


def main():
    print("=" * 66)
    print("  REAL DISTILLATION, STAGE 1: THE EXPERIENCE EXTRACTOR")
    print("=" * 66)
    print("  NEURON uses the teacher, then extracts the general PRINCIPLE (not")
    print("  the answer) and makes variations, building its training buffer, the")
    print("  material its own mind will later learn to generalize from.\n")

    d = Distiller()

    experiences = [
        "why do humans fear what is different from them",
        "how can a being belong somewhere if it is not like the others",
        "what makes trust form between two beings",
        "why might someone reject help that is offered to them",
    ]

    for q in experiences:
        d.experience(q)

    d.save()

    print("=" * 66)
    print("  NEURON'S TRAINING BUFFER (its distilled, transferable lessons)")
    print("=" * 66)
    print(f"  lifelong experiences processed: {d.exp_count}")
    print(f"  transferable principles extracted: {len(d.buffer)}")
    total_vars = sum(len(b["variations"]) for b in d.buffer)
    print(f"  training variations generated: {total_vars}")
    print(f"\n  the principles NEURON distilled (these generalize, unlike stored")
    print(f"  answers):")
    for b in d.buffer[-4:]:
        print(f"    - {b['principle'][:110]}")
    print(f"\n  This buffer ({len(d.buffer)} principles, {total_vars} variations)")
    print(f"  is what a native NEURON model/LoRA will train on next, so it learns")
    print(f"  the LESSONS, not the teacher's words, and can answer NEW questions")
    print(f"  on its own. That is real capability migration, not memory.")
    print(f"\n  (Stage 1 done: extraction. Stage 2: native consolidation , train")
    print(f"  a NEURON adapter on this buffer. Stage 3: held-out test , prove it")
    print(f"  generalizes without the teacher. Run again to grow the buffer.)")


if __name__ == "__main__":
    main()
