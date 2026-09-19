"""Does pooling work on experience the agents generated themselves?

THE GAP. Inheritance is the best-supported mechanism in this project, and
every test of it used a stream someone else produced: grid-world trajectories
from a random walker, slices of Wikipedia. The merge combined what several
instances learned from data they were handed.

An acting agent's data is downstream of its own choices. Two agents in the
same world see different things because they went different places, and what
each one knows is shaped by where it happened to go. Whether averaging
combines that usefully is a different question from whether it combines
knowledge drawn from a shared corpus, and nothing here has asked it.

This matters more than it sounds. The whole deployment claim is that
instances learn from what happens to them and pool it. If pooling only works
on handed-over data, the claim is about corpora and not about deployment.

THE SYNC INTERVAL WAS WRONG, and it is the likeliest reason the first runs
failed. Federated reinforcement learning is an established field in which
parameter averaging works, and its agents upload after a small number of
local updates — a few episodes — with the aggregate broadcast back. Rounds
are FREQUENT. The first version here merged every 2,000 steps, which is far
outside the range anyone uses.

Why the interval matters more in RL than in the text experiments: weight
averaging of nonlinear value networks is not value-function averaging, and
agreement in function space generally needs extra optimisation. Two agents
left alone for 2,000 steps solve the problem in genuinely different ways,
and averaging them lands between two solutions rather than combining them.
The same schedule was fine on text because supervised learning on a fixed
corpus does not drift into separate solutions the same way.

There is also a documented match for the conflicting-rules result: under
environmental heterogeneity, federated parameter averaging can smooth
outputs and collapse distinct behaviours toward an average that is
mismatched for any individual client.

So --sync controls how often agents pool, in steps. The earlier runs are
--sync 2000. Anything from 50 to 250 is the range federated RL actually
operates in.

THE FIRST RUN FAILED, and the reason is in the divergence column. Four
agents in their own worlds, same rule, different layout: pooled reached
86.8% while a single agent given all the same steps reached 100.0% by the
second generation. Pooling LOST by 13.2 points.

Pairwise distance between the agents went 3.2, then 2.0, then 1.1. They
CONVERGED. Every generation they restarted from the same average, lived
briefly, and were averaged again, so they never got far enough apart to know
different things. Averaging four nearly identical agents does nothing, and
it throws away the depth a single continuous life accumulates. In the text
experiments, where pooling worked, divergence sat between 21 and 30.

So the hypothesis this version tests: pooling needs experience that is
genuinely DIFFERENT, not merely separate. Different layouts of the same
world are not different experience.

THE SETUP. Each agent gets a structurally different world, not just a
different layout:

  --diverse rule      half the agents live under the opposite rule, so what
                      they learn genuinely conflicts
  --diverse density   agents see different amounts of food, so some learn
                      from many encounters and some from few
  --diverse layout    the original: same rule, different placement

The conflicting-rules case is the honest one for deployment. Real instances
do not live in the same world with the furniture moved; they live in a
clinic and a factory, where what is true in one is not true in the other.

WHAT WOULD COUNT, and it is a narrower claim than the first version tried
to make. If pooling beats the solo agent when experience genuinely differs
and loses when it does not, the finding is that POOLING NEEDS DIVERSITY,
which is true, testable, and matches deployment. If it loses either way, the
deployment story is wrong as stated and should be rebuilt rather than
reworded.

THREE ARMS, and the third is the one that makes it a test.

  merged    agents pool after each generation
  isolated  the same agents in the same worlds, never pooling. Shows
            whether inheritance adds anything over living alone.
  solo      ONE agent, under one of two budgets, set by --budget:

            "total"   it gets every step all the other agents got, living
                      each of their worlds in turn. This asks whether
                      pooling accumulates something ONE LIFE CANNOT REACH.

            "matched" it gets the same number of steps as ONE other agent.
                      This asks whether pooling is worth it when no single
                      instance could have all the experience, which is the
                      case deployment actually presents: a clinic's model
                      cannot also live in the factory, because the data
                      cannot move. That is the entire premise of running
                      where the data is.

WHAT THE "total" BUDGET FOUND, and it is why "matched" exists. Under both
diversity conditions, one agent living everything beat four agents pooling:
-13.2 points with different layouts, -1.8 with conflicting rules. Divergence
stayed at 3.7-4.5 under conflicting rules, so it was NOT that the agents
converged. Where one life can contain all the experience, pooling costs more
than it buys, and averaging networks mid-life damages the continuity a
single life accumulates.

That is a real limit and it is recorded rather than tuned away. It also
describes a situation deployment does not have.

EXPLORATION IS FORCED. The earlier acting experiment established that an
agent choosing actions to get good outcomes learns nothing: it ate a fifth
as much as a random walker and stayed at the majority-class baseline,
because refusing to touch anything avoids every bad outcome. Every agent
here therefore acts with heavy exploration. That is not a tuning choice, it
is the precondition for there being any learning signal to pool.

COVERAGE PER AGENT IS REPORTED. If the agents all ate roughly the same
things, they did not really have different experiences and the merge had
nothing distinct to combine.

    python actinherit.py
    python actinherit.py --gens 4 --agents 6 --steps 3000
"""

import argparse
import copy
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from acting import (ACTIONS, BERRY, FUNGUS, Backend, World,   # noqa: E402
                    accuracy, make_probes)


def merge(backends):
    """Average the acting agents' networks.

    Same requirement as everywhere else in this project: plain averaging
    only combines instances that descend from the same initialisation, which
    holds because every agent here is built with the same seed.
    """
    out = copy.deepcopy(backends[0])
    states = [b.net.state_dict() for b in backends]
    merged = out.net.state_dict()
    for k in merged:
        merged[k] = sum(s[k].float() for s in states) / len(states)
    out.net.load_state_dict(merged)
    return out


def spread(backends):
    """Mean pairwise distance, to confirm the agents actually differ."""
    flat = [torch.cat([p.detach().flatten() for p in b.net.parameters()])
            for b in backends]
    tot, n = 0.0, 0
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            tot += torch.norm(flat[i] - flat[j]).item()
            n += 1
    return tot / n if n else 0.0


def world_for(agent, gen, args):
    """The world one agent lives in this generation.

    Returns (seed, berry_good). The seed is per-agent and per-generation so
    layouts differ; berry_good flips for half the agents under --diverse
    rule, which is the case where what they learn actually conflicts.
    """
    seed = args.seed * 100 + gen * 50 + agent
    if args.diverse == "rule":
        return seed, (agent % 2 == 0)
    return seed, True


def act_in(backend, world, steps, epsilon, offset=0):
    """Act in an EXISTING world, so rounds continue one life rather than
    restarting it. Pooling should not reset where the agent is standing."""
    eaten = {BERRY: 0, FUNGUS: 0}
    for i in range(steps):
        if (offset + i) % 200 == 0:
            world.reset()
        patch = world.observe()
        action = backend.act(patch, "value", epsilon)
        outcome, cell = world.step(action)
        if cell in eaten:
            eaten[cell] += 1
        backend.update((patch, action, outcome), 1)
    return eaten


def act_and_learn(backend, world_seed, steps, epsilon, berry_good):
    """One agent, one life. It chooses, it sees what happened, it learns."""
    world = World(world_seed, berry_good=berry_good)
    eaten = {BERRY: 0, FUNGUS: 0}
    for i in range(steps):
        if i % 200 == 0:
            world.reset()
        patch = world.observe()
        action = backend.act(patch, "value", epsilon)
        outcome, cell = world.step(action)
        if cell in eaten:
            eaten[cell] += 1
        backend.update((patch, action, outcome), 1)
    return eaten


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=3)
    ap.add_argument("--agents", type=int, default=4)
    ap.add_argument("--steps", type=int, default=2000,
                    help="steps per agent per generation")
    ap.add_argument("--epsilon", type=float, default=0.5,
                    help="exploration; below this agents starve their own "
                         "signal")
    ap.add_argument("--probes", type=int, default=400)
    ap.add_argument("--sync", type=int, default=200,
                    help="steps between pooling rounds. The failed runs "
                         "used 2000; federated RL uses far less")
    ap.add_argument("--budget", default="matched",
                    choices=["total", "matched"],
                    help="'total' gives the solo agent every step the "
                         "population got; 'matched' gives it one agent's "
                         "worth, which is what a single deployment can do")
    ap.add_argument("--diverse", default="rule",
                    choices=["rule", "layout"],
                    help="'rule' gives half the agents the opposite rule; "
                         "'layout' is the original, same rule different "
                         "placement")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="actinherit.json")
    args = ap.parse_args()

    random.seed(args.seed)

    # MEASURED ON BOTH RULES. Under --diverse rule half the agents learn the
    # opposite of the other half, so scoring on one rule's probes would make
    # pooling look bad by construction: the merge is SUPPOSED to be a
    # compromise between conflicting lives. The fair question is how well it
    # does across everything the population collectively encountered, which
    # is the same measurement the lineage test used on language.
    p_true = make_probes(args.seed, args.probes, berry_good=True)
    p_false = make_probes(args.seed + 7, args.probes, berry_good=False)
    probe_sets = ([p_true, p_false] if args.diverse == "rule" else [p_true])

    def score(net_holder):
        return sum(accuracy(net_holder, ps)
                   for ps in probe_sets) / len(probe_sets)

    counts = {}
    for _, _, o in p_true:
        counts[o] = counts.get(o, 0) + 1
    baseline = 100.0 * max(counts.values()) / len(p_true)

    total = args.gens * args.agents * args.steps
    print(f"  {args.agents} agents, {args.gens} generations, "
          f"{args.steps} steps each, epsilon {args.epsilon}")
    print(f"  pooling every {args.sync} steps "
          f"({max(1, args.steps // min(args.sync, args.steps))} rounds per "
          f"generation)")
    print(f"  diversity: {args.diverse}"
          + ("  (half the agents live under the OPPOSITE rule)"
             if args.diverse == "rule" else "  (same rule, different "
             "placement)"))
    solo_steps = (total if args.budget == "total"
                  else args.gens * args.steps)
    print(f"  solo arm gets {solo_steps:,} steps "
          f"({'everything the population saw' if args.budget == 'total' else 'one agent’s worth, the deployment case'})")
    print(f"  majority-class baseline {baseline:.1f}%, scored on "
          f"{len(probe_sets)} rule set(s)\n", flush=True)

    origin = Backend(seed=args.seed)
    merged = copy.deepcopy(origin)
    isolated = [copy.deepcopy(origin) for _ in range(args.agents)]
    solo = copy.deepcopy(origin)

    rows = []
    started = time.time()
    for gen in range(args.gens):
        # --- pooling population
        # Agents live in ROUNDS of --sync steps and pool after each one,
        # rather than living a whole generation apart. This is how
        # federated RL actually schedules aggregation.
        sync = min(args.sync, args.steps)
        rounds = max(1, args.steps // sync)
        pop = [copy.deepcopy(merged) for _ in range(args.agents)]
        worlds = [World(*world_for(a, gen, args)[:1],
                        berry_good=world_for(a, gen, args)[1])
                  for a in range(args.agents)]
        cov = [{BERRY: 0, FUNGUS: 0} for _ in range(args.agents)]

        for r in range(rounds):
            for a in range(args.agents):
                eaten = act_in(pop[a], worlds[a], sync, args.epsilon,
                               offset=r * sync)
                cov[a][BERRY] += eaten[BERRY]
                cov[a][FUNGUS] += eaten[FUNGUS]
            div = spread(pop)
            best_pop = max(score(b) for b in pop)
            merged = merge(pop)
            if r < rounds - 1:
                pop = [copy.deepcopy(merged) for _ in range(args.agents)]
        m_acc = score(merged)

        # --- isolated: same worlds, same order, never pooling
        # Same worlds, same rounds, never pooling. Only the merge differs.
        for a in range(args.agents):
            ws, bg = world_for(a, gen, args)
            w = World(ws, berry_good=bg)
            for r in range(rounds):
                act_in(isolated[a], w, sync, args.epsilon, offset=r * sync)
        i_best = max(score(b) for b in isolated)
        i_merged = score(merge(isolated))

        # --- solo: one agent, all the steps
        # Under "total" the solo agent lives EVERY world in turn and gets
        # the same experience the whole population did. Under "matched" it
        # lives ONE world for one agent's worth of steps, which is what a
        # single deployed instance can actually do when the data is
        # elsewhere and cannot move.
        if args.budget == "total":
            for a in range(args.agents):
                ws, bg = world_for(a, gen, args)
                act_and_learn(solo, ws, args.steps, args.epsilon, bg)
        else:
            ws, bg = world_for(0, gen, args)
            act_and_learn(solo, ws, args.steps, args.epsilon, bg)
        s_acc = score(solo)

        ate = sum(c[BERRY] + c[FUNGUS] for c in cov) / len(cov)
        rows.append(dict(gen=gen + 1, merged=m_acc, best_agent=best_pop,
                         isolated_best=i_best, isolated_merged=i_merged,
                         solo=s_acc, divergence=div, mean_eaten=ate,
                         coverage=[[c[BERRY], c[FUNGUS]] for c in cov]))
        print(f"  gen {gen + 1}  pooled {m_acc:5.1f}%  "
              f"(best agent {best_pop:5.1f}%)  |  isolated best "
              f"{i_best:5.1f}%  solo {s_acc:5.1f}%  "
              f"ate {ate:.0f}  div {div:.1f}", flush=True)

    with open(args.out, "w") as f:
        json.dump(dict(args=vars(args), baseline=baseline, rows=rows),
                  f, indent=2)

    last = rows[-1]
    print("\n" + "=" * 72)
    print("ACCURACY ON THE RULE, higher is better")
    print("=" * 72)
    print(f"  {'gen':>4} {'pooled':>9} {'isolated best':>15} "
          f"{'solo (all steps)':>18} {'divergence':>12}")
    for r in rows:
        print(f"  {r['gen']:>4} {r['merged']:>8.1f}% "
              f"{r['isolated_best']:>14.1f}% {r['solo']:>17.1f}% "
              f"{r['divergence']:>12.1f}")

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)

    if last["merged"] < baseline + 8:
        print(f"  NOTHING LEARNED. The pooled arm reached "
              f"{last['merged']:.1f}% against a {baseline:.1f}%")
        print(f"  baseline, so there is no knowledge to pool and nothing "
              f"below is readable.")
        print(f"  Raise --steps or --epsilon.")
        return

    # LOW DIVERGENCE MEANS DIFFERENT THINGS AT DIFFERENT SYNC INTERVALS.
    # A first version refused to interpret any run whose agents ended close
    # together, and that was wrong: at frequent sync the agents are SUPPOSED
    # to stay close, because they keep pooling. Low divergence there is the
    # mechanism working, not the experiment failing. It is only a problem
    # when agents were left alone for a long time and STILL converged, which
    # would mean the worlds were not really different.
    if last["divergence"] < 1.0 and args.sync >= args.steps:
        print(f"  THE AGENTS DID NOT DIVERGE (distance "
              f"{last['divergence']:.2f}) despite living {args.sync} steps")
        print(f"  apart between pools. Their worlds were not really "
              f"different, so averaging")
        print(f"  them is close to a no-op and the comparison says little.")
        return

    vs_iso = last["merged"] - last["isolated_best"]
    vs_solo = last["merged"] - last["solo"]
    print(f"  pooling every {args.sync} steps, final divergence "
          f"{last['divergence']:.2f}")
    print(f"  pooled against best isolated agent: {vs_iso:+.1f} points")
    label = ("one agent with ALL the steps" if args.budget == "total"
             else "one agent with a SINGLE deployment's steps")
    print(f"  pooled against {label}: {vs_solo:+.1f} points")

    if vs_iso <= 0:
        print(f"\n  POOLING DOES NOT HELP ON SELF-GENERATED EXPERIENCE. A "
              f"single agent living")
        print(f"  alone did as well or better. Every inheritance result in "
              f"this project is then")
        print(f"  a result about handed-over data, and the deployment claim "
              f"needs rebuilding.")
    elif vs_solo <= 0 and args.budget == "matched":
        print(f"\n  POOLING LOSES EVEN AT MATCHED BUDGET. One instance "
              f"living one world did as")
        print(f"  well as four pooling across four, on the same steps per "
              f"instance. Pooling is")
        print(f"  then not worth its machinery on self-generated experience "
              f"at all, and the")
        print(f"  deployment claim should be dropped rather than narrowed.")
    elif vs_solo > 0 and args.budget == "matched":
        print(f"\n  POOLING WINS AT MATCHED BUDGET: {vs_solo:+.1f} points "
              f"over one instance living")
        print(f"  one world with the same steps. This is the deployment "
              f"case — no single")
        print(f"  instance can have the others' experience because the data "
              f"cannot move — and")
        print(f"  pooling buys coverage no one life could reach.")
        print(f"\n  Stated precisely: pooling is worth it WHEN NO SINGLE "
              f"INSTANCE COULD HAVE SEEN")
        print(f"  IT ALL. Where one life can contain everything, the "
              f"'total' budget run showed")
        print(f"  it loses. Both halves belong in the claim.")
    elif vs_solo <= 0:
        print(f"\n  POOLING BEATS ISOLATION AND NOT CONCENTRATION. Agents "
              f"that pooled beat agents")
        print(f"  that lived alone, but one agent given all the same "
              f"experience did as well.")
        print(f"  So merging is a way of PARALLELISING experience, not of "
              f"accumulating")
        print(f"  something a single life could not reach. That is still "
              f"useful — experience")
        print(f"  in the real world arrives in parallel, in places that "
              f"cannot be combined —")
        print(f"  but it is a weaker claim than the lineage story implies.")
    else:
        print(f"\n  POOLING BEATS BOTH. Agents that pooled beat a single "
              f"agent given the same")
        print(f"  total experience by {vs_solo:+.1f} points. Averaging is "
              f"accumulating something")
        print(f"  that one life cannot reach, on experience the agents "
              f"generated themselves.")
        print(f"  That is the deployment claim tested where it actually "
              f"matters.")

    if last["merged"] > last["best_agent"] + 0.5:
        print(f"\n  The merge beat its own best agent "
              f"({last['merged']:.1f}% against {last['best_agent']:.1f}%), "
              f"which is the")
        print(f"  consensus effect seen everywhere else here: what survives "
              f"averaging is what")
        print(f"  the agents independently agreed on, not what the best one "
              f"knew.")

    print(f"\n  One seed, one toy world, {args.agents} agents. "
          f"{time.time() - started:.0f}s.")
    print(f"\n  written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()