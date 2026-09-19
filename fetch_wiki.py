"""Drain Wikipedia to a local file, once.

neuron_run.py's wiki command re-streams from the top on every run and skips
forward to where it left off. That looked cheap because the skip loop is
just an iterator, but reaching sentence N means DOWNLOADING enough parquet
to contain it. Unauthenticated that crawls, and resume time grows with
everything already read — which makes a system meant to run for months
slower to start every single session.

A remote dataset is a source you drain once, not a tape you rewind. This
writes sentences to a plain text file, one per line, and remembers how far
it got. Run it in the background while you work; after that every read is
local and instant.

    python fetch_wiki.py 50000                  # append 50k more sentences
    python fetch_wiki.py 50000 --out data/wiki.txt

    ./run.sh neuron_run.py read ../data/wiki.txt

Set HF_TOKEN first. Anonymous requests to the Hub are rate limited hard
enough to be the dominant cost here.
"""

import argparse
import json
import os
import re
import sys
import time


def sentences_from(article, min_len=60, max_len=250):
    """The same filter neuron_run.py applies, kept identical on purpose."""
    if article["title"].startswith("List of"):
        return
    for s in re.split(r"(?<=[.!?])\s+", article["text"][:4000]):
        s = s.strip()
        if min_len < len(s) < max_len and not s.startswith("="):
            yield s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int, help="how many sentences to add")
    ap.add_argument("--out", default="data/wiki.txt")
    ap.add_argument("--report", type=int, default=1000)
    args = ap.parse_args()

    if not os.environ.get("HF_TOKEN"):
        print("  no HF_TOKEN set. This will work but will be slow and may "
              "be rate limited. huggingface.co/settings/tokens")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pos_file = args.out + ".pos"

    # How many sentences the file already holds, and how far into the
    # dataset we got. Counted from the file rather than trusted from the
    # position file, so a truncated write cannot silently desync them.
    have = 0
    if os.path.exists(args.out):
        with open(args.out) as f:
            have = sum(1 for _ in f)
    skip = 0
    if os.path.exists(pos_file):
        with open(pos_file) as f:
            skip = json.load(f).get("drained", 0)
    if skip != have:
        print(f"  position file says {skip}, file holds {have}. "
              f"Trusting the file.")
        skip = have

    print(f"  {have} sentences already cached in {args.out}")
    print(f"  fetching {args.n} more, skipping the first {skip}")

    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.en",
                      split="train", streaming=True)

    started = time.time()
    count = 0      # sentences seen in this pass, including skipped ones
    written = 0
    try:
        with open(args.out, "a") as out:
            for article in ds:
                for s in sentences_from(article):
                    count += 1
                    if count <= skip:
                        continue
                    out.write(s.replace("\n", " ") + "\n")
                    written += 1
                    if written % args.report == 0:
                        out.flush()
                        rate = written / max(1e-9, time.time() - started)
                        print(f"  {written} written, {rate:.0f}/s",
                              flush=True)
                    if written >= args.n:
                        raise StopIteration
    except (StopIteration, KeyboardInterrupt):
        pass
    except Exception as e:
        # Partial progress is still progress. Record it and exit cleanly so
        # the next run picks up rather than starting over.
        print(f"  stopped early: {e}")

    with open(pos_file, "w") as f:
        json.dump(dict(drained=skip + written), f)

    took = time.time() - started
    print(f"  wrote {written} sentences in {took:.0f}s "
          f"({written / max(1e-9, took):.0f}/s)")
    print(f"  {skip + written} total in {args.out}")


if __name__ == "__main__":
    main()