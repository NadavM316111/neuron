"""Why does collapse cluster at item ~275? Two hypotheses:
  A. something in the stream at that position
  B. the rehearsal buffer (maxlen 200) evicts the anchor material
"""

import random
from collections import deque
from neuron import Neuron, DEFAULTS

import ablate  # reuse the exact stream builder

BUFFER_MAX = DEFAULTS["buffer_size"]

for seed in [0, 1]:
    stream = ablate.build_stream(seed)
    print(f"\n=== seed {seed} ===")

    facts = set()
    for a, b in ablate.EARLY_FACTS:
        facts.add(a); facts.add(b)
    for f in ablate.LATE_FACTS:
        facts.add(f)

    print("\nStream composition per 25-item block:")
    print(f"{'block':>10} {'facts':>6} {'junk':>6} {'filler':>7}")
    for start in range(0, len(stream), 25):
        block = stream[start:start + 25]
        nf = sum(1 for t in block if t in facts)
        nj = sum(1 for t in block if t not in facts and t not in ablate.FILLER_POOL)
        print(f"{start:>4}-{start+24:>4} {nf:>6} {nj:>6} {len(block)-nf-nj:>7}")

    # Simulate buffer occupancy without running the model.
    # Roughly: anything not a fact and not junk lands in the buffer.
    buf = deque(maxlen=BUFFER_MAX)
    first_evict = None
    anchor = set()
    for i, t in enumerate(stream, start=1):
        if i <= 12:
            anchor.add(t)
        if t in ablate.FILLER_POOL or i <= 12:
            if len(buf) == BUFFER_MAX and first_evict is None:
                first_evict = i
            buf.append(t)
    print(f"\nBuffer reaches its {BUFFER_MAX} cap at roughly item {first_evict}")
    print("(after that, the oldest anchor material starts getting evicted)")

print("""
READ:
  If facts/junk density spikes near 275 in both seeds -> hypothesis A.
  If the buffer hits its cap somewhere around 250-300 -> hypothesis B,
  and the fix is a buffer that keeps a permanent reserved slice of
  early anchor material instead of a plain FIFO.
""")