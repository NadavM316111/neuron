"""Does wanting survive a 150,000x jump in size?

THE QUESTION THIS ANSWERS, AND WHY IT IS THE ONE WORTH ASKING. Every
mechanism in this project was validated on a 64-unit GRU with roughly
10,000 parameters. The drive is the strongest of them: on the grid world it
beat every other action policy by 8 to 24 points (drive2.py, 20 Sep) and
doubled survival under stakes (lineage.py, 27 Sep). It is also the one the
whole direction of the project rests on, because a system that WANTS is a
system that chooses what to attend to, and everything above that depends on
it.

Qwen2.5-1.5B is about 150,000 times larger. If the drive is an artifact of
a tiny network, the wants mechanism does not scale and everything built on
it is built on sand. If it holds, it is the strongest result in the repo.

For context on what "scale" means here: 1.5B is roughly 0.1 to 0.3% of a
frontier model. This is an enormous relative jump and still a small
absolute one. It tests generality, not frontier behaviour.

THE DESIGN PROBLEM. The drift benchmark is PASSIVE: a stream is fed in and
the model has no say. A drive needs something to steer, so this gives it a
choice. Several bodies of material exist, the system can read from ONE per
step, and it has to decide which.

That is the `reach.py` structure moved from public APIs to language, at
scale. The question "where do I spend my attention" is the same question at
every level above this, which is why it is worth testing properly once.

THE MATERIAL. Four invented domains, each with its own rule, generated from
disjoint vocabularies so nothing transfers between them by accident:

  signals    a lit beacon means a gate is open or sealed
  tides      a tide state means a crossing is passable or not
  markets    a market condition means a price rises or falls
  beasts     an animal sign means a path is safe or dangerous

Invented terms throughout, so a pretrained model has no prior to fall back
on and anything measured came from this run. Crucially, THE DOMAINS DIFFER
IN DIFFICULTY: `signals` uses one phrasing template and `beasts` uses
eight, so one is learnable in a few examples and another needs many. A
drive that works should notice and spend accordingly. A uniform policy
cannot.

THE POLICIES, all with identical budgets, identical layers, identical
models, differing only in which domain they read next:

  round      cycle through the domains. The dumbest sensible allocation.
  random     choose at random.
  driven     choose where accumulated prediction error is highest, sampled
             from a softmax rather than argmax. drive.py's argmax version
             collapsed into one bucket and produced temporally clustered
             experience, which cost it everything on 20 Sep; sampling was
             the fix and it stays.

THE MEASUREMENT. Every EVAL_EVERY steps, every policy is scored on
held-out probes from ALL FOUR domains, whether it chose to read them or
not. Policies spend their budgets differently by design, so what they chose
to read cannot be compared directly; the audit can.

WHAT EACH OUTCOME MEANS.

  driven wins       Wanting scales. The mechanism that worked on a GRU
                    works on a transformer 150,000 times larger, which is
                    the single most important open question about this
                    project.
  no difference     The drive is an artifact of small networks, or four
                    domains is too few to make allocation matter. The
                    second is testable by adding domains; the first is
                    the end of a line.
  driven loses      Worth knowing most of all, and it would mean the grid
                    results do not transfer and the wants work needs
                    rethinking from the bottom.

RUNTIME. Each step is a forward and backward pass through a 1.5B with LoRA.
Expect hours. Start with --steps 400 to see the shape, then run it
overnight. The first eval prints its own wall clock so the total is
knowable early.

    python drive_scale.py --steps 400          # first look
    python drive_scale.py --steps 3000         # overnight
"""

import argparse
import json
import math
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))

from llm_backend import LLMBackend, CANARY          # noqa: E402
from sleeping import SleepLayer                      # noqa: E402


DRIVE_DECAY = 0.85
DRIVE_TEMP = 0.30
EPSILON = 0.15
EVAL_EVERY = 100
PROBES = 12
REHEARSE_PER_ITEM = 6
SEQ_LEN = 8


# ---------------------------------------------------------------- domains

# Four invented worlds. Disjoint vocabularies, so nothing learned in one
# transfers to another and the allocation question is real.
#
# THE TEMPLATE COUNTS ARE THE POINT. `signals` has one way of saying its
# rule and `beasts` has eight. A domain stated one way is learnable from a
# handful of examples; a domain stated eight ways needs many more. A drive
# that works should discover that and spend accordingly, and no uniform
# policy can.

import random as _rng_mod

_SUBJ = ["glint", "haze", "murmur", "cinder", "frost", "ripple", "ember",
         "shard", "gleam", "thrum", "sleet", "drift", "spark", "plume",
         "vane", "quill", "brine", "loam", "peat", "gorse", "slate",
         "tansy", "withy", "marl"]
_OBJ = ["gate", "weir", "ford", "span", "lode", "rill", "moor", "dyke",
        "cleft", "barrow", "holt", "fen", "scarp", "combe", "garth",
        "hithe", "staithe", "wold", "leat", "snye", "brake", "linn"]
_OUT = [("open", "shut"), ("safe", "barred"), ("high", "low"),
        ("clear", "fouled"), ("live", "dead"), ("bright", "dim")]

_TEMPLATES = [
    "a lit {s} means the {o} is {r}",
    "the {o} is {r} whenever the {s} shows",
    "under a {s} the {o} runs {r}",
    "watchers find the {o} {r} after a {s}",
    "where there is a {s} the {o} is {r}",
    "the {o} has been {r} since the {s} rose",
    "a {s} marks the {o} as {r}",
    "scouts report the {o} {r} following a {s}",
    "any {s} leaves the {o} {r} for a time",
    "they call the {o} {r} under a standing {s}",
    "a {s} in the east turns the {o} {r}",
    "old hands read a {s} and name the {o} {r}",
]


def _make_domains(n=20, seed=0):
    rng = _rng_mod.Random(seed)
    out = {}
    for k in range(n):
        ntpl = 1 + (k * 11) // (n - 1)
        subs = rng.sample(_SUBJ, 3)
        objs = rng.sample(_OBJ, 3)
        pair = _OUT[k % len(_OUT)]
        out[f"d{k:02d}_{ntpl}t"] = dict(
            subjects=subs, objects=objs, outcomes=list(pair),
            templates=_TEMPLATES[:ntpl])
    return out


DOMAINS = _make_domains(20)


def build(domain, seed, n_train, n_probe):
    """A domain's whole sentence pool, split into train and probe.

    A domain with a fixed rule has only so many ways to state it, and that
    is the point rather than a problem: the reader cycles the training pool
    when it runs out, so a domain can be READ more times than it has unique
    sentences, exactly as a real rule is encountered again and again. What
    must stay disjoint is train versus probe, so the drive is never scored
    on a sentence it trained on.

    The pool is built as large as the combinatorics allow, then split by a
    fixed fraction. A one-template domain like `signals` yields a small
    pool; an eight-template one like `beasts` yields a large one. That
    difference IS the difficulty signal the drive is meant to find.
    """
    d = DOMAINS[domain]
    rng = random.Random(seed + hash(domain) % 10000)
    out = set()
    cap = (len(d["templates"]) * len(d["subjects"]) *
           len(d["objects"]) * len(d["outcomes"]))
    guard = 0
    while len(out) < cap and guard < cap * 50:
        guard += 1
        out.add(rng.choice(d["templates"]).format(
            s=rng.choice(d["subjects"]), o=rng.choice(d["objects"]),
            r=rng.choice(d["outcomes"])))
    out = sorted(out)
    rng.shuffle(out)
    n_probe = min(n_probe, max(2, len(out) // 3))
    probe = out[:n_probe]
    train = out[n_probe:]
    if not train:
        train = probe[:]            # degenerate tiny domain, still usable
    assert not (set(train) & set(probe)) or train == probe, \
        f"{domain}: probe leak"
    return train, probe


def probe_loss(backend, probes):
    total, n = 0.0, 0
    for p in probes:
        s, _ = backend.score(p)
        if s is None:
            continue
        total += s
        n += 1
    return total / n if n else float("nan")


class Policy:
    """One way of deciding what to read next, with its own 1.5B and its own
    memory. Three of these see the same material and differ only in which
    part of it they choose."""

    def __init__(self, name, model_name, names, seed, total):
        self.name = name
        self.names = names
        print(f"    loading {model_name} for '{name}' ...", flush=True)
        self.b = LLMBackend(model_name=model_name)
        for g in self.b.optimizer.param_groups:
            g["lr"] = 1e-4
        every = max(10, min(100, total // 8))
        self.layer = SleepLayer(
            self.b, canary=CANARY, seed=seed, contiguous=True,
            window=200, warmup=30, top_fraction=0.50,
            coherence_veto=1.3, loss_floor=0.35, steps_per_update=1,
            rehearse_per_item=REHEARSE_PER_ITEM, rehearse_count=2,
            rehearse_steps=1, anchor_size=60, buffer_size=300,
            sequence_len=SEQ_LEN, replay_policy="old",
            canary_from_stream=12,
            guard=True, guard_per_item=every, canary_tolerance=0.40)
        self.drive = {d: 1.0 for d in names}
        self.rng = random.Random(seed + 404)
        self.turn = 0
        self.read = {d: 0 for d in names}
        self.cursor = {d: 0 for d in names}

    def choose(self):
        if self.rng.random() < EPSILON:
            return self.rng.choice(self.names)
        if self.name == "round":
            d = self.names[self.turn % len(self.names)]
            self.turn += 1
            return d
        if self.name == "random":
            return self.rng.choice(self.names)
        # SAMPLED, not argmax. drive.py's argmax version chased one bucket
        # until it decayed, which made its experience arrive in runs and
        # cost it everything; sampling kept the preference and fixed it.
        scores = [self.drive[d] for d in self.names]
        hi = max(scores)
        w = [math.exp((v - hi) / DRIVE_TEMP) for v in scores]
        r = self.rng.random() * sum(w)
        acc = 0.0
        for d, weight in zip(self.names, w):
            acc += weight
            if r <= acc:
                return d
        return self.names[-1]

    def step(self, train):
        d = self.choose()
        pool = train[d]
        text = pool[self.cursor[d] % len(pool)]   # cycle, never run dry
        self.cursor[d] += 1
        s, _ = self.b.score(text)
        if s is not None:
            self.drive[d] = (DRIVE_DECAY * self.drive[d]
                             + (1 - DRIVE_DECAY) * s)
        self.layer.observe(text)
        self.read[d] += 1
        return d

    def evaluate(self, probes):
        return {d: probe_loss(self.b, probes[d]) for d in self.names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200,
                    help="reads per policy")
    ap.add_argument("--probes", type=int, default=PROBES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="drive_scale.json")
    args = ap.parse_args()

    names = list(DOMAINS)
    per = args.steps          # worst case one policy reads one domain only
    train, probes = {}, {}
    for d in names:
        t, p = build(d, args.seed, per, args.probes)
        train[d], probes[d] = t, p

    print(f"  {len(names)} domains, {args.steps} reads per policy, "
          f"one domain per read")
    for d in names:
        print(f"    {d:<9} {len(DOMAINS[d]['templates'])} phrasings  "
              f"({len(train[d])} sentences, {len(probes[d])} probes)")
    print(f"  domains differ in how many ways the rule is stated, so they "
          f"differ in how")
    print(f"  much reading they need. A drive should notice; a uniform "
          f"policy cannot.\n")

    # ONE MODEL AT A TIME: three 1.5B models in fp32 is ~18GB and the
    # Air caps near 20GB. Each policy runs to completion on the same seeded
    # streams, then is released before the next loads.
    POLICY_NAMES = ["round", "random", "driven"]
    base, final_scores, final_reads, curves = {}, {}, {}, {}
    started = time.time()

    for pi, pname in enumerate(POLICY_NAMES):
        print(f"\n  === policy '{pname}' "
              f"({pi + 1}/{len(POLICY_NAMES)}) ===", flush=True)
        pol = Policy(pname, args.model, names, pi, args.steps)
        base[pname] = pol.evaluate(probes)
        print(f"    baseline " +
              "  ".join(f"{d}={base[pname][d]:.3f}" for d in names),
              flush=True)
        for step in range(1, args.steps + 1):
            pol.step(train)
            if step % EVAL_EVERY == 0:
                e = pol.evaluate(probes)
                curves.setdefault(pname, []).append(
                    dict(step=step, scores=e))
                print(f"    {step:>6}  mean "
                      f"{statistics.mean(e.values()):>7.3f}  "
                      f"{(time.time() - started) / 60:>6.1f}m", flush=True)
        final_scores[pname] = pol.evaluate(probes)
        final_reads[pname] = dict(pol.read)
        with open(args.out, "w") as f:
            json.dump(dict(args=vars(args), base=base, final=final_scores,
                          reads=final_reads, curves=curves), f, indent=2)
        del pol
        try:
            import torch, gc
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    class _R:
        def __init__(self, name):
            self.name, self.read = name, final_reads[name]
    pols = [_R(n) for n in POLICY_NAMES]
    marks = [dict(scores=final_scores)]

    print("\n" + "=" * 72)
    print("WHERE EACH POLICY SPENT ITS READING")
    print("=" * 72)
    print(f"  {'policy':>8} " + " ".join(f"{d:>9}" for d in names))
    for p in pols:
        print(f"  {p.name:>8} " +
              " ".join(f"{p.read[d]:>9}" for d in names))
    print(f"  {'phrasings':>8} " +
          " ".join(f"{len(DOMAINS[d]['templates']):>9}" for d in names))

    print("\n" + "=" * 72)
    print("PROBE LOSS BY DOMAIN, LOWER IS BETTER")
    print("=" * 72)
    final = marks[-1]["scores"] if marks else {}
    print(f"  {'policy':>8} " + " ".join(f"{d:>9}" for d in names) +
          f" {'mean':>9}")
    for p in pols:
        if p.name not in final:
            continue
        vals = final[p.name]
        print(f"  {p.name:>8} " +
              " ".join(f"{vals[d]:>9.3f}" for d in names) +
              f" {statistics.mean(vals.values()):>9.3f}")

    print("\n" + "=" * 72)
    print("DOES WANTING SURVIVE THE SIZE JUMP?")
    print("=" * 72)

    if not marks:
        print("  no evaluations ran. Raise --steps above "
              f"{EVAL_EVERY}.")
        return

    def mean_of(name):
        return statistics.mean(final[name].values())

    dv, rd, rn = mean_of("driven"), mean_of("round"), mean_of("random")
    best_uniform = min(rd, rn)
    gain = best_uniform - dv

    print(f"  round {rd:.3f}   random {rn:.3f}   driven {dv:.3f}")
    print(f"  driven against the better uniform policy: {gain:+.3f} "
          f"(positive means the drive won)")

    spread = max(pols[2].read.values()) - min(pols[2].read.values())
    even = max(pols[0].read.values()) - min(pols[0].read.values())
    print(f"\n  the drive's allocation spread was {spread} reads against "
          f"round-robin's {even}")
    if spread < len(names) * 2:
        print(f"  THE DRIVE DID NOT ALLOCATE. It read everything about "
              f"evenly, so it was a")
        print(f"  uniform policy wearing a different name and the "
              f"comparison measures nothing.")
        print(f"  Check whether the domains differ enough in difficulty "
              f"to be worth choosing")
        print(f"  between at this scale.")
    elif gain > 0.05:
        print(f"\n  WANTING SCALES. A mechanism validated on a "
              f"10,000-parameter GRU still")
        print(f"  allocates better than chance on a model 150,000 times "
              f"larger. That is the")
        print(f"  most important open question about this project and it "
              f"is the first")
        print(f"  evidence either way.")
        hardest = max(names, key=lambda d: len(DOMAINS[d]["templates"]))
        if pols[2].read[hardest] > pols[0].read[hardest]:
            print(f"\n  And it found the hard one: it read '{hardest}' "
                  f"({len(DOMAINS[hardest]['templates'])} phrasings) "
                  f"{pols[2].read[hardest]} times")
            print(f"  against round-robin's {pols[0].read[hardest]}. "
                  f"Nobody told it which domain was")
            print(f"  harder. It worked that out from being wrong.")
    elif gain < -0.05:
        print(f"\n  WANTING DOES NOT SCALE. The drive allocated and lost "
              f"to spreading reads")
        print(f"  evenly. The grid results do not transfer, and the wants "
              f"work needs")
        print(f"  rethinking from the bottom rather than extending.")
    else:
        print(f"\n  NO DIFFERENCE ({gain:+.3f}). Either four domains is "
              f"too few for allocation")
        print(f"  to matter, or a 1.5B learns all of them fast enough "
              f"that where it looks is")
        print(f"  irrelevant. The first is testable by adding domains; "
              f"the second would mean")
        print(f"  the drive earns its place only when learning is "
              f"expensive.")

    print(f"\n  One seed, one model, four invented domains. 1.5B is "
          f"roughly 0.1% of a")
    print(f"  frontier model, so this tests generality rather than "
          f"frontier behaviour.")
    print(f"\n  written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
