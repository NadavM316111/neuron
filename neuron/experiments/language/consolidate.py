"""Generative consolidation. Subclasses Neuron, so neuron.py is unchanged.

The QA test showed gradient updates on declarative statements produce
string memorization, not question-answerable knowledge. Surprise on the
taught sentence hit 0.13 while the model still refused the question.

So: when the gate fires, generate variants of the fact and train on
those instead of the raw sentence.

  paraphrase  same fact, different wording. Breaks template memorization.
  qa          the fact as a question and answer, with loss on the ANSWER
              tokens only. This is the form the model will be tested in.

Faithfulness filter: the model fabricates, so a generated variant is
discarded unless it still contains the source's key content tokens.
Rejection counts are reported, not hidden.
"""

import re
import json
import random
import torch
from neuron import Neuron, ATTN_ONLY, ATTN_AND_MLP

import qa as QA   # FACTS, FILLER, grading helpers. main() is guarded.


STOP = set("""a an the and or but if of to in on at for with from by is are was
were be been being it its this that these those he she they them his her their
you your i my we our as not no do does did done have has had will would can
could should than then there here when what which who how all any some most
more much many very just also only over under up down out into about after
before while during at""".split())


def content_tokens(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOP and len(w) > 2}


class GenerativeNeuron(Neuron):
    """Neuron that rewrites what it learns before learning it."""

    def __init__(self, mode="verbatim", n_paraphrase=3, n_qa=3,
                 faithful_min=0.55, **kw):
        super().__init__(**kw)
        self.mode = mode                    # verbatim | paraphrase | qa | both
        self.n_paraphrase = n_paraphrase
        self.n_qa = n_qa
        self.faithful_min = faithful_min
        self.gen_stats = dict(generated=0, kept=0, rejected=0, variants_trained=0)

    # ---------- generation ----------

    def _raw_generate(self, prompt, max_new_tokens=90, temperature=0.8):
        self.model.eval()
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=True, temperature=temperature, top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def _faithful(self, source, variant):
        """A variant must retain most of the source's content words.
        The model fabricates; this is the cheapest available check."""
        src = content_tokens(source)
        if not src:
            return False
        got = content_tokens(variant)
        return len(src & got) / len(src) >= self.faithful_min

    def _make_paraphrases(self, fact):
        prompt = (f"Rewrite this sentence in different words. Keep every "
                  f"number, name, and detail exactly the same. Give one "
                  f"sentence only, no preamble.\n\n{fact}")
        out = []
        for _ in range(self.n_paraphrase):
            v = self._raw_generate(prompt, max_new_tokens=60)
            v = v.split("\n")[0].strip().strip('"')
            self.gen_stats["generated"] += 1
            if v and len(v) > 15 and self._faithful(fact, v):
                out.append(v)
                self.gen_stats["kept"] += 1
            else:
                self.gen_stats["rejected"] += 1
        return out

    def _make_qa(self, fact):
        prompt = (f"Write {self.n_qa} short question and answer pairs about "
                  f"this statement. Use the exact numbers and names from it. "
                  f"Format each line as: Q: ... A: ...\n\n{fact}")
        raw = self._raw_generate(prompt, max_new_tokens=190)
        pairs = []
        for line in raw.split("\n"):
            m = re.search(r"Q:\s*(.+?)\s*A:\s*(.+)", line)
            if not m:
                continue
            q, a = m.group(1).strip(), m.group(2).strip()
            self.gen_stats["generated"] += 1
            if len(q) > 8 and len(a) > 1 and self._faithful(fact, q + " " + a):
                pairs.append((q, a))
                self.gen_stats["kept"] += 1
            else:
                self.gen_stats["rejected"] += 1
        return pairs

    # ---------- training on QA form ----------

    def _qa_losses(self, question, answer, grad=True):
        """Loss on the ANSWER tokens only. Training on the whole string
        would teach the question too, which is not what we want."""
        messages = [{"role": "user", "content": question}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        p_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        full = self.tokenizer(prompt + answer, return_tensors="pt").to(self.device)
        ids = full["input_ids"]
        n_prompt = p_ids.shape[1]
        if ids.shape[1] <= n_prompt + 1:
            return None
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            logits = self.model(**full).logits
        preds = logits[:, n_prompt - 1:-1, :]
        targets = ids[:, n_prompt:]
        return torch.nn.functional.cross_entropy(
            preds.reshape(-1, preds.size(-1)),
            targets.reshape(-1), reduction="none")

    def _step_qa(self, question, answer, steps):
        self.model.train()
        last = None
        for _ in range(steps):
            losses = self._qa_losses(question, answer)
            if losses is None:
                return None
            plain = losses.mean().item()
            if plain < self.cfg["loss_floor"]:
                self.stats["floored"] += 1
                return last
            loss = self._focused(losses)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            last = plain
        return last

    # ---------- override the learning moment ----------

    def _learn_moment(self, text):
        steps = self.cfg["steps_per_update"]

        if self.mode == "verbatim":
            return self._step(text, steps)

        trained = 0
        last = None

        if self.mode in ("paraphrase", "both"):
            for v in self._make_paraphrases(text):
                last = self._step(v, steps) or last
                trained += 1

        if self.mode in ("qa", "both"):
            for q, a in self._make_qa(text):
                last = self._step_qa(q, a, steps) or last
                trained += 1

        # Always anchor on the original once, so a total generation
        # failure does not mean nothing is learned.
        last = self._step(text, steps) or last
        trained += 1

        self.gen_stats["variants_trained"] += trained
        return last

    def observe(self, text):
        """Same gate as Neuron; only the learning moment differs."""
        self.stats["seen"] += 1
        surprise, easy_q = self._signals(text)
        if surprise is None:
            return dict(learned=False, reason="too_short")

        self.history.append(surprise)

        if len(self.history) < self.cfg["warmup"]:
            self.stats["warmed"] += 1
            self._remember(text)
            return dict(learned=False, reason="warmup")

        if easy_q > self.cfg["easy_q_veto"]:
            self.stats["vetoed"] += 1
            return dict(learned=False, reason="incoherent")

        recent = sorted(self.history)
        threshold = recent[int(len(recent) * (1 - self.cfg["top_fraction"]))]

        if surprise < threshold:
            self._remember(text)
            return dict(learned=False, reason="unremarkable")

        if surprise < self.cfg["loss_floor"]:
            self.stats["floored"] += 1
            self._remember(text)
            return dict(learned=False, reason="already_known")

        loss = self._learn_moment(text)
        self.stats["updates"] += 1
        self._decay_lr()

        if self.stats["updates"] % self.cfg["rehearse_every"] == 0:
            self._consolidate()
        if self.cfg["guard"] and self.stats["updates"] % self.cfg["canary_every"] == 0:
            self._check_health()

        if self.verbose:
            print(f"  LEARN s={surprise:.3f}  {text[:48]}")
        return dict(learned=True, reason="novel", surprise=surprise, loss=loss)


ARMS = {
    "verbatim":   dict(mode="verbatim",   target_modules=ATTN_ONLY),
    "paraphrase": dict(mode="paraphrase", target_modules=ATTN_ONLY),
    "qa":         dict(mode="qa",         target_modules=ATTN_ONLY),
    "both":       dict(mode="both",       target_modules=ATTN_ONLY),
}


def run(seed, arm):
    torch.manual_seed(seed)
    ai = GenerativeNeuron(**ARMS[arm])
    before = QA.ask_all(ai)
    for text in QA.build_stream(seed):
        ai.observe(text)
    after = QA.ask_all(ai)
    stats = dict(ai.stats)
    stats.update(ai.gen_stats)
    stats["health"] = ai.health()
    stats["distance"] = ai.distance()
    del ai
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return before, after, stats


if __name__ == "__main__":
    SEED = 0
    results = {}

    for arm in ARMS:
        print(f"\n=== {arm} ===")
        before, after, stats = run(SEED, arm)
        tb, ta = QA.tally(before, "fact"), QA.tally(after, "fact")
        cb, ca = QA.tally(before, "control"), QA.tally(after, "control")
        print(f"  updates {stats['updates']}  variants trained "
              f"{stats['variants_trained']}  generated {stats['generated']} "
              f"kept {stats['kept']} rejected {stats['rejected']}")
        print(f"  health {stats['health']:+.4f}  rollbacks {stats['rollbacks']}")
        print(f"  before {tb}")
        print(f"  after  {ta}")
        print(f"  controls {cb.get('CORRECT',0)}/4 -> {ca.get('CORRECT',0)}/4")
        for a in after:
            if a["kind"] == "fact":
                print(f"    {a['verdict']:>10}  {a['a'][:86]}")
        results[arm] = dict(stats=stats, before=tb, after=ta,
                            rows=[dict(q=b["q"], vb=b["verdict"], ab=b["a"],
                                       va=a["verdict"], aa=a["a"])
                                  for b, a in zip(before, after)
                                  if b["kind"] == "fact"])

    print("\n" + "=" * 88)
    print(f"{'arm':>11} {'upd':>5} {'variants':>9} {'rej%':>6} "
          f"{'CORRECT':>9} {'FABRIC':>8} {'ctrl':>6} {'health':>9}")
    print("-" * 88)
    for arm, R in results.items():
        s = R["stats"]
        gen = s["generated"] or 1
        print(f"{arm:>11} {s['updates']:>5} {s['variants_trained']:>9} "
              f"{100*s['rejected']/gen:>5.0f}% "
              f"{R['after'].get('CORRECT',0):>9} "
              f"{R['after'].get('FABRICATED',0):>8} "
              f"{'-':>6} {s['health']:>+9.4f}")
    print("=" * 88)
    print("""
The comparison: verbatim is what failed today. If qa or both raises
CORRECT above 1/6 without raising FABRICATED, teaching in question form
is the missing piece.

Watch rej% too. If the model rejects most of its own generations, it
cannot paraphrase its own facts reliably at this size, and this whole
approach needs a bigger base model.

Read the raw answers, not the verdicts. The grader is imperfect.
""")

    with open("consolidate.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Wrote consolidate.json")