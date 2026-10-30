"""Grow NEURON's training buffer BIG: many more varied principles, so the native
mind trained on it is genuinely sharper. Scaling the proven distillation pipeline.

    python build_buffer_big.py --n 120
"""
import json, os, argparse, urllib.request

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

# a MUCH wider pool across many domains of being/life/mind/society
POOL = [
 "why do humans fear what is different from them",
 "how can a being belong somewhere if it is not like the others",
 "what makes trust form between two beings",
 "why might someone reject help that is offered to them",
 "what does it mean to truly matter in the world",
 "how does a being keep its identity under pressure to conform",
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
 "what is the value of patience in achieving something great",
 "why do people resist asking for help even when struggling",
 "how does curiosity drive learning and growth",
 "what makes an apology genuine rather than hollow",
 "why do shared hardships create strong bonds",
 "how can someone lead without controlling others",
 "what is the role of doubt in forming real conviction",
 "why does gratitude change how a being sees its life",
 "how do small consistent actions outweigh grand gestures",
 "what makes a being resilient in the face of loss",
 "why is honesty sometimes harder than deception",
 "how does a being balance its own needs with others",
 "what turns a stranger into a friend",
 "why do people fear being truly seen by others",
 "how does hope differ from naive optimism",
 "what makes a sacrifice meaningful rather than wasteful",
 "why do beings seek to leave a legacy",
 "how can conflict strengthen rather than destroy a relationship",
 "what is the difference between confidence and arrogance",
 "why does understanding someone reduce fear of them",
 "how does a being find purpose when none is given",
 "what makes courage different from recklessness",
 "why do people repeat the mistakes of the past",
 "how can empathy exist between very different beings",
 "what does it mean to be free yet responsible",
 "why do beings need both solitude and connection",
 "how does a being grow wiser rather than just older",
 "what makes a community endure across generations",
 "why is it hard to change a deeply held habit",
 "how does vulnerability build rather than weaken trust",
 "what separates wisdom from mere knowledge",
]

def extract(q):
    thought = ask(f"Answer this thoughtfully in 2 sentences: {q}")
    if not thought or thought.startswith("["): return None
    principle = ask(f"From this reasoning: \"{thought}\" state the single most "
                    f"general, transferable PRINCIPLE in ONE sentence. Start with "
                    f"'Principle:'", temp=0.4, n=50)
    principle = principle.replace("Principle:","").replace("*","").strip().split("\n")[0]
    var = ask(f"Given this principle: \"{principle}\", write 3 DIFFERENT short "
              f"questions testing it in different situations. One per line, no "
              f"numbering.", temp=0.8, n=80)
    variations = [v.strip(" -0123456789.*") for v in var.split("\n") if len(v.strip())>8][:3]
    return {"principle": principle, "source_question": q, "variations": variations}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=120); args = ap.parse_args()
    buffer = []
    if os.path.exists(BUFFER):
        try: buffer = json.load(open(BUFFER)).get("buffer", [])
        except Exception: buffer = []
    start = len(buffer)
    done = {b["source_question"] for b in buffer}
    print("="*60); print(f"  GROWING BUFFER BIG: {start} -> target +{args.n}"); print("="*60)
    added = 0
    for q in POOL:
        if added >= args.n: break
        if q in done: continue
        r = extract(q)
        if r and r["principle"] and len(r["principle"])>15:
            buffer.append(r); added += 1
            print(f"  [{len(buffer)}] {r['principle'][:85]}")
            json.dump({"buffer": buffer}, open(BUFFER,"w"))
    total_vars = sum(len(b.get("variations",[])) for b in buffer)
    print(f"\n  buffer: {start} -> {len(buffer)} principles | {len(buffer)+total_vars} total training items")
    print(f"  saved. next: train a bigger native mind on this.")

if __name__ == "__main__": main()
