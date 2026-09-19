"""The program. Feed it text, ask it questions, check on it.

Until now this project was forty experiment scripts. This is one thing that
runs, keeps its state on disk, and can be stopped and resumed.

    ./run.sh neuron_run.py read somefile.txt
    ./run.sh neuron_run.py ask "what did it say about X?"
    ./run.sh neuron_run.py status
    ./run.sh neuron_run.py wiki 2000
    ./run.sh neuron_run.py reset

State lives in a directory (neuron_state by default) and survives being
turned off, which nothing in this project did before.

By default it runs WITHOUT a language model, so the store and routing work
on a laptop with nothing installed. Pass --model to attach a real one.
"""

import argparse
import json
import os
import re
import sys

from neuron_system import NeuronSystem


def make_backend(model_name):
    if not model_name:
        return None
    from llm_backend import LLMBackend
    b = LLMBackend(model_name=model_name, focus_alpha=0.0)
    for g in b.optimizer.param_groups:
        g["lr"] = 1e-4
    return b


def sentences_from_file(path):
    with open(path) as f:
        text = f.read()
    for s in re.split(r"(?<=[.!?])\s+", text):
        s = s.strip()
        if s:
            yield s


def wiki_sentences(n, skip=0):
    """A stream to try it on, if you have nothing to hand.

    `skip` matters: the dataset always starts at the same place, so without
    it a second run replays the same sentences and the store correctly
    rejects every one as a duplicate. Skipping ahead makes repeated runs
    continue the stream rather than repeat it.
    """
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.en",
                      split="train", streaming=True)
    count = 0
    for article in ds:
        if article["title"].startswith("List of"):
            continue
        for s in re.split(r"(?<=[.!?])\s+", article["text"][:4000]):
            s = s.strip()
            if 60 < len(s) < 250 and not s.startswith("="):
                count += 1
                if count <= skip:
                    continue
                yield s
                if count >= n + skip:
                    return


def main():
    p = argparse.ArgumentParser(description="the NEURON system")
    p.add_argument("command",
                   choices=["read", "ask", "status", "wiki", "reset"])
    p.add_argument("argument", nargs="?", default=None)
    p.add_argument("--state", default="neuron_state")
    p.add_argument("--model", default=None,
                   help="e.g. Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--no-retention", action="store_true")
    p.add_argument("--no-embeddings", action="store_true")
    args = p.parse_args()

    if args.command == "reset":
        import shutil
        if os.path.isdir(args.state):
            shutil.rmtree(args.state)
            print(f"removed {args.state}")
        else:
            print("nothing to remove")
        return

    backend = make_backend(args.model)
    system = NeuronSystem(backend=backend, path=args.state,
                          use_embeddings=not args.no_embeddings,
                          retention=not args.no_retention)

    resumed = system.load()
    if resumed:
        print(f"resumed from {args.state}: "
              f"{len(system.store.sentences)} sentences stored")
    else:
        print(f"starting fresh in {args.state}")

    if args.command == "status":
        st = system.status()
        print(f"\n  stored sentences : {st['stored']}")
        print(f"  embeddings       : {st['embeddings']}")
        print(f"  retention layer  : {st['retention']}")
        print(f"  lifetime counts  : {st['stats']}")
        meta = os.path.join(args.state, "meta.json")
        if os.path.exists(meta):
            with open(meta) as f:
                m = json.load(f)
            print(f"  last saved       : {m.get('saved_at')}")
            print(f"  wiki position    : {m.get('wiki_read', 0)} sentences")
            if "layer" in m:
                print(f"  layer            : {m['layer']}")
        return

    if args.command == "ask":
        if not args.argument:
            print("  ask what?")
            return
        answer = system.ask(args.argument)
        print(f"\n  Q: {args.argument}")
        print(f"  A: {answer}")
        if system.refused(answer):
            print("  (refused — the notes did not contain it)")
        return

    if args.command == "read":
        if not args.argument or not os.path.exists(args.argument):
            print(f"  no such file: {args.argument}")
            return
        print(f"reading {args.argument} ...")
        stats = system.read(sentences_from_file(args.argument),
                            progress_every=500)

    elif args.command == "wiki":
        n = int(args.argument or 1000)
        # Continue the stream rather than replaying it. The position is
        # kept in the state directory so it survives restarts.
        pos_file = os.path.join(args.state, "wiki_position.json")
        skip = 0
        if os.path.exists(pos_file):
            with open(pos_file) as f:
                skip = json.load(f).get("read", 0)
        print(f"reading {n} sentences from Wikipedia, "
              f"skipping the first {skip} already seen ...")
        stats = system.read(wiki_sentences(n, skip=skip),
                            progress_every=500)
        os.makedirs(args.state, exist_ok=True)
        with open(pos_file, "w") as f:
            json.dump(dict(read=skip + n), f)

    system.store.reindex()
    system.save()
    print(f"\n  {stats}")
    print(f"  saved to {args.state}")
    print(f"  {len(system.store.sentences)} sentences stored in total")


if __name__ == "__main__":
    main()