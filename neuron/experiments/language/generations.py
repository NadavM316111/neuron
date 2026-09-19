"""Does knowledge compound across generations, on language?

THE GRID-WORLD RESULT, Aug 27. Six generations of eight instances, each
generation seeded from the previous generation's merge, each living a new
life. Merge quality went 4.25% to 63.50%. A control population living the
IDENTICAL lives for the identical total steps, but never inheriting, went
36.89% to 9.35% — it did not merely stagnate, it degraded. Merging beat
never-merging by 66 points at the final generation.

WHY IT NEEDS DOING ON TEXT. Model merging is a mature field with tooling and
surveys, but it is mature for merging TASK SPECIALISTS ONCE. A 2026
systematic study of in-the-wild merging found that current techniques mostly
do not extract useful updates from heterogeneous, conflicting versions, which
is the same thing this project found when it merged contradictory lives into
a poor arbiter. Generations of CONTINUALLY LEARNING instances, merged
repeatedly, is the case nobody has settled, and every result here so far is
on a grid world.

THE SETUP. Each generation, N instances each read a DIFFERENT slice of the
stream, single pass, in order, never revisited. Their weights are averaged.
The merge is measured, and then seeds the next generation. Six generations.

WHY AVERAGING IS LEGITIMATE HERE. Merging plain weights requires every
instance to descend from the same initialisation. That held on the grid world
because the same seed was used, and it holds BY CONSTRUCTION in a shipped
product because every copy ships from the same initial weights. This script
enforces it: generation 0 instances are all built from one seed, and every
later generation is built from the previous merge. Verified rather than
assumed — a check confirms the instances are not identical afterwards, since
identical instances would make the merge meaningless.

THE CONTROL IS THE POINT. A second population lives the SAME slices in the
same order for the same total number of sentences and never merges. It is the
only thing that distinguishes "inheritance compounds knowledge" from "more
data helps". Without it this experiment says nothing.

SELECTION, added 8 Sep. The version above merges every instance, so the
lineage accumulates and does not improve: bad instances dilute good ones.
This adds differential survival — only the best fraction breeds — and runs
it against merging everyone on identical streams.

  all       every instance contributes equally. The original.
  weighted  every instance contributes, weighted by how well it did.

THE CONFOUND THE FIRST VERSION HAD. It compared merging all six instances
against merging only the best three, and lost by 0.0102. But averaging MORE
instances is itself beneficial — the merge beats its own best individual in
every run here, which is the whole midpoint-beats-ends effect — so that
comparison changed two things at once: who breeds, and how many. A result
attributed to selection could equally have been three parents being worse
than six.

So selection is now applied as WEIGHTS rather than as exclusion. All six
instances contribute, and an instance that did better contributes more.
Same parents, same breadth, only influence differs. Equal weights reproduce
plain averaging exactly, so the two arms differ in one thing.

The weighting is softmax over negative loss with a temperature. Low
temperature approaches picking the single best instance; high temperature
approaches equal weights. The default is set so the best instance gets
roughly twice the influence of the worst, which is a real difference without
being exclusion in disguise.

Both arms live the SAME slices in the same order, so the only difference is
who breeds. A third arm, the never-merging control, stays as the baseline
that separates "inheritance compounds" from "more data helps".

What could go wrong, and it is the reason to run it rather than assume:
selection reduces diversity, and diversity is what averaging was preserving.
A lineage bred from the best half may improve faster and inherit a narrower
world model, and the newcomer head start depends on the shared structure
that breadth provides.

  ./run.sh experiments/language/generations.py --data <abs path>
  ./run.sh experiments/language/generations.py --gens 3 --instances 4 \\
      --slice 1200 --data <abs path>            # quick version
"""

import argparse
import copy
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scratch import (CharGRU, V, encode, held_out_loss,   # noqa: E402
                     ngram_loss, pick_device)


def live(model, lines, args, device):
    """One instance, one slice. Single pass, in order, never revisited."""
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss()
    batch, pending = [], 0
    for line in lines:
        ids = encode(line)
        if len(ids) < 2:
            continue
        model.train()
        x = torch.tensor([ids[:-1]], device=device)
        y = torch.tensor([ids[1:]], device=device)
        logits, _ = model(x)
        batch.append(lossf(logits.reshape(-1, V), y.reshape(-1)))
        pending += 1
        if pending >= args.accum:
            opt.zero_grad()
            (sum(batch) / len(batch)).backward()
            opt.step()
            batch, pending = [], 0
    if batch:
        opt.zero_grad()
        (sum(batch) / len(batch)).backward()
        opt.step()
    return model


def merge(models, weights=None):
    """Weight averaging, optionally weighted per instance.

    Plain averaging because the grid-world sweep found Fisher weighting and
    rehearsal repair within 1.2 points of it, so the simplest wins.

    `weights` is what makes selection possible: pass a value per model and
    the average is weighted by it. Equal weights reproduce plain averaging
    exactly, so the selected and unselected arms differ in one thing only.
    """
    out = copy.deepcopy(models[0])
    states = [m.state_dict() for m in models]
    if weights is None:
        weights = [1.0] * len(states)
    total = sum(weights)
    merged = out.state_dict()
    for k in merged:
        merged[k] = sum(w * s[k].float()
                        for w, s in zip(weights, states)) / total
    out.load_state_dict(merged)
    return out


def fitness_weights(losses, temperature):
    """Influence per instance, from its held-out loss. Softmax over -loss.

    This is selection without exclusion. Every instance still breeds, so the
    breadth that averaging preserves is intact, and the only thing that
    changes is how much each one counts. That matters because the previous
    version dropped half the population and could not separate "selection
    helps" from "more parents help".

    temperature -> 0 approaches taking only the best instance.
    temperature -> infinity approaches equal weights, i.e. plain averaging.
    """
    import math
    lo = min(losses)
    raw = [math.exp(-(l - lo) / temperature) for l in losses]
    total = sum(raw)
    return [r / total for r in raw]


def select(models, losses, keep):
    """Keep the best `keep` instances by held-out loss. The rest do not breed.

    KEPT FOR REFERENCE AND NOT USED BY DEFAULT. Exclusion lost to equal
    averaging by 0.0102 on one seed, and that comparison was confounded by
    parent count. fitness_weights is the unconfounded version.

    WHY THIS IS THE INTERESTING ARM. Without it every instance contributes
    equally to the merge, including the ones that learned badly, so the
    lineage ACCUMULATES rather than IMPROVES. Averaging preserves what the
    lives agreed on and cancels what they disagreed on, which makes a good
    starting point and a poor improvement engine — the population experiment
    on the grid world scored 28-30% merging sixteen contradictory lives
    against a solo reference of 57.71%.

    Selection is differential survival, the oldest mechanism in
    neuroevolution, applied here to instances that learned online from
    their own streams. The question is whether it improves the lineage or
    collapses its diversity, because the same averaging that wastes
    excellence is also what preserves the shared structure a newcomer
    inherits.
    """
    order = sorted(range(len(models)), key=lambda i: losses[i])
    chosen = order[:max(1, keep)]
    return [models[i] for i in chosen], chosen


def divergence(models):
    """Mean pairwise L2 between instances, to confirm they actually differ.

    If the instances end a generation identical, averaging them is a no-op
    and every number downstream is meaningless. Checked rather than assumed.
    """
    flat = [torch.cat([p.detach().flatten() for p in m.parameters()])
            for m in models]
    total, n = 0.0, 0
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            total += torch.norm(flat[i] - flat[j]).item()
            n += 1
    return total / n if n else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/wiki.txt")
    ap.add_argument("--gens", type=int, default=6)
    ap.add_argument("--instances", type=int, default=8)
    ap.add_argument("--slice", type=int, default=2000,
                    help="sentences each instance reads per generation")
    ap.add_argument("--eval", type=int, default=800)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--embed", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--accum", type=int, default=4,
                    help="accumulation window; 4 won the earlier sweep")
    ap.add_argument("--temperature", type=float, default=0.02,
                    help="fitness weighting sharpness; lower is more "
                         "selective, higher approaches equal weights")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="generations.json")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        raise SystemExit(f"no data at {os.path.abspath(args.data)}")

    lines = [l.strip() for l in open(args.data) if len(l.strip()) > 40]
    need = args.gens * args.instances * args.slice + args.eval
    if len(lines) < need:
        raise SystemExit(f"only {len(lines)} usable lines, need {need}. "
                         f"Lower --gens, --instances or --slice, or fetch "
                         f"more with fetch_wiki.py.")

    eval_lines = lines[-args.eval:]
    pool = lines[:args.gens * args.instances * args.slice]
    device = pick_device()

    trigram = ngram_loss(" ".join(pool[:20000]), eval_lines, 3)
    print(f"  {args.gens} generations, {args.instances} instances, "
          f"{args.slice} sentences each")
    print(f"  {len(pool)} sentences consumed in total, {args.eval} held out")
    print(f"  uniform {math.log(V):.4f}, trigram {trigram:.4f} nats")
    print(f"  device {device}, accumulation {args.accum}\n", flush=True)

    def slice_for(gen, inst):
        i = (gen * args.instances + inst) * args.slice
        return pool[i:i + args.slice]

    # Both populations start from the SAME weights, so the only difference
    # between them is whether they inherit.
    torch.manual_seed(args.seed)
    origin = CharGRU(args.hidden, args.embed).to(device)
    merged = copy.deepcopy(origin)          # merges everyone
    merged_sel = copy.deepcopy(origin)      # merges all, weighted
    control = [copy.deepcopy(origin) for _ in range(args.instances)]
    keep = max(1, args.instances // 2)      # only for the legacy select()
    print(f"  weighted arm: all {args.instances} breed, weighted by loss "
          f"at temperature {args.temperature}\n", flush=True)

    rows = []
    started = time.time()
    for gen in range(args.gens):
        # --- merge-everyone population
        pop = [live(copy.deepcopy(merged), slice_for(gen, inst),
                    args, device)
               for inst in range(args.instances)]
        div = divergence(pop)
        merged = merge(pop)
        m_loss = held_out_loss(merged, eval_lines, device)
        best = min(held_out_loss(m, eval_lines, device) for m in pop)

        # --- weighted population: identical slices, ALL instances breed,
        #     weighted by how well each did. Same parents as `all`.
        pop_s = [live(copy.deepcopy(merged_sel), slice_for(gen, inst),
                      args, device)
                 for inst in range(args.instances)]
        losses_s = [held_out_loss(m, eval_lines, device) for m in pop_s]
        w = fitness_weights(losses_s, args.temperature)
        div_s = divergence(pop_s)
        merged_sel = merge(pop_s, weights=w)
        s_loss = held_out_loss(merged_sel, eval_lines, device)
        s_best = min(losses_s)
        spread = max(losses_s) - min(losses_s)

        # --- control: same slices, same order, no inheritance ever
        for inst in range(args.instances):
            control[inst] = live(control[inst], slice_for(gen, inst),
                                 args, device)
        c_best = min(held_out_loss(m, eval_lines, device) for m in control)
        c_merged = held_out_loss(merge(control), eval_lines, device)

        rows.append(dict(gen=gen + 1, merge=m_loss, best=best,
                         divergence=div, selected=s_loss,
                         selected_best=s_best, selected_div=div_s,
                         weights=w, spread=spread,
                         control_best=c_best, control_merge=c_merged))
        print(f"  gen {gen + 1}  all {m_loss:.4f}  weighted {s_loss:.4f}  "
              f"control {c_best:.4f}  |  best inst {best:.4f}  "
              f"spread {spread:.4f}  w {min(w):.2f}-{max(w):.2f}  "
              f"div {div:.1f}/{div_s:.1f}", flush=True)

        if div < 1e-6:
            print("  STOP: instances are identical, so the merge is a no-op "
                  "and nothing below this line means anything.")
            break

    with open(args.out, "w") as f:
        json.dump(dict(args=vars(args), trigram=trigram, rows=rows),
                  f, indent=2)

    # ------------------------------------------------------------ verdict

    print("\n" + "=" * 72)
    print("GENERATIONS   held-out cross entropy, lower is better")
    print("=" * 72)
    print(f"  {'gen':>4} {'merge all':>10} {'merge best':>11} "
          f"{'control':>9} {'diversity':>18}")
    for r in rows:
        print(f"  {r['gen']:>4} {r['merge']:>10.4f} {r['selected']:>11.4f} "
              f"{r['control_best']:>9.4f} "
              f"{r['divergence']:>8.1f} /{r['selected_div']:>8.1f}")

    first, last = rows[0], rows[-1]
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)

    m_d = last["merge"] - first["merge"]
    s_d = last["selected"] - first["selected"]
    c_d = last["control_best"] - first["control_best"]

    print(f"  merge everyone    {first['merge']:.4f} -> "
          f"{last['merge']:.4f}  ({m_d:+.4f})")
    print(f"  merge best half   {first['selected']:.4f} -> "
          f"{last['selected']:.4f}  ({s_d:+.4f})")
    print(f"  never merge       {first['control_best']:.4f} -> "
          f"{last['control_best']:.4f}  ({c_d:+.4f})")

    adv = last["control_best"] - last["merge"]
    if m_d >= -0.005:
        print(f"\n  KNOWLEDGE DID NOT COMPOUND at all. Neither merged arm "
              f"improved across")
        print(f"  generations, so nothing below distinguishes the arms and "
              f"the grid-world")
        print(f"  result did not transfer.")
        return
    if adv <= 0:
        print(f"\n  MERGING ADDS NOTHING. Both merged arms improved and so "
              f"did the control,")
        print(f"  which never inherited. On this stream more data explains "
              f"the gain.")
        return

    print(f"\n  Inheritance is real: merging everyone ended {adv:+.4f} "
          f"ahead of never merging.")

    gap = last["merge"] - last["selected"]

    # CAN WEIGHTING MATTER AT ALL? If the instances all did about equally
    # well, every weighting scheme collapses to equal weights and the arms
    # are the same experiment run twice. That has to be checked before the
    # gap is read as anything, because a difference of 0.01 between two
    # nearly identical procedures is noise by construction.
    ratios = [max(r["weights"]) / min(r["weights"]) for r in rows]
    spreads = [r["spread"] for r in rows]
    print(f"\n  instance spread per generation: "
          f"{', '.join(f'{v:.4f}' for v in spreads)}")
    print(f"  influence ratio best:worst:     "
          f"{', '.join(f'{v:.2f}x' for v in ratios)}")

    if max(ratios) < 1.5:
        print(f"\n  THE TEST CANNOT DETECT A WEIGHTING EFFECT. The best "
              f"instance never got more")
        print(f"  than {max(ratios):.2f}x the influence of the worst, "
              f"because the instances all did")
        print(f"  about equally well (spread {max(spreads):.4f}). Both arms "
              f"are effectively plain")
        print(f"  averaging, so the {gap:+.4f} between them is noise and "
              f"says nothing about")
        print(f"  selection either way.")
        print(f"\n  To test it properly the instances have to genuinely "
              f"differ. Give them")
        print(f"  different amounts of data, or different quality, so some "
              f"really do learn more.")
        print(f"  Lowering --temperature only sharpens weights that are "
              f"being applied to")
        print(f"  differences too small to mean anything.")
        print(f"\n  One seed, {args.instances} instances. "
              f"{time.time() - started:.0f}s.")
        return

    if gap > 0.01:
        print(f"\n  SELECTION HELPS. Weighting by performance ended "
              f"{gap:+.4f} ahead of equal")
        print(f"  weights, with the same parents and the same breadth. "
              f"The lineage IMPROVES")
        print(f"  rather than merely accumulating, which is the")
        print(f"  difference between a population that drifts and one that "
              f"gets better without")
        print(f"  anyone retraining it. That is the mechanism, and it is "
              f"differential survival,")
        print(f"  which is the oldest idea in neuroevolution applied to "
              f"instances that learned")
        print(f"  online from their own streams.")
    elif gap < -0.01:
        print(f"\n  SELECTION HURTS by {-gap:.4f}, with parent count held "
              f"equal this time. So it")
        print(f"  is not that three parents beat six: giving good "
              f"instances more influence is")
        print(f"  itself worse than counting everyone equally. The "
              f"diversity that averaging preserves is")
        print(f"  load-bearing. Averaging cancels contested beliefs and "
              f"keeps shared structure;")
        print(f"  selecting narrows that structure toward whatever the "
              f"winners happened to see.")
        print(f"  Worth knowing before building a lineage protocol on "
              f"selection.")
    else:
        print(f"\n  SELECTION MAKES NO DIFFERENCE ({gap:+.4f}). At "
              f"{args.instances} instances over")
        print(f"  {args.gens} generations there is not enough spread "
              f"between the best and worst")
        print(f"  for who breeds to matter. More instances, or streams that "
              f"differ more, would")
        print(f"  be the test.")

    # Diversity is the thing selection spends, so it has to be reported.
    d_all = last["divergence"]
    d_sel = last["selected_div"]
    if d_sel < d_all * 0.8:
        print(f"\n  DIVERSITY COST: the selected population ended at "
              f"{d_sel:.1f} pairwise distance")
        print(f"  against {d_all:.1f} for merging everyone. Selection is "
              f"spending variation, and")
        print(f"  variation is what a newcomer inherits as shared "
              f"structure. Any gain above")
        print(f"  has to be read against that.")
    else:
        print(f"\n  Diversity held: {d_sel:.1f} against {d_all:.1f}, so "
              f"selection did not collapse")
        print(f"  the population over {args.gens} generations.")

    if last["best"] < last["merge"] - 0.005:
        print(f"\n  No consensus trap: the best individual "
              f"({last['best']:.4f}) beat the merge.")
    else:
        print(f"\n  POSSIBLE CONSENSUS TRAP: no instance beat the merge at "
              f"the final generation.")

    print(f"\n  One seed, character level, {args.instances} instances, "
          f"keep {keep}. "
          f"{time.time() - started:.0f}s.")
    print(f"\n  written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()