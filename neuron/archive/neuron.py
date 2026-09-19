"""
NEURON: a runtime that lets a language model keep learning while it runs.

Aug 16 2026 build, revised after the QA test showed the system was
converting honest refusals into confident fabrications.

Two suspected causes of that, both now configurable:

  target_modules   was attention only (q_proj, v_proj). Attention decides
                   what to attend to; factual content lives mostly in the
                   MLP layers. Editing attention teaches the frame, not
                   the fact. Default now includes the MLP projections.

  focus_alpha      loss was averaged uniformly over tokens, so the one
                   token carrying the value ("nineteen") got 1/13 of the
                   gradient while easy structural words dominated. Now
                   tokens are weighted by their own surprise, so gradient
                   concentrates on the content. focus_alpha=0 restores
                   the old uniform behaviour.

Other findings baked in:
  - mean per-token surprise beats max for gating
  - easy_q coherence veto rejects junk for free from the same forward pass
  - rehearsal is not optional; without it drift triples
  - gradient clipping actively hurts, do not add it back
  - the canary guard is the load-bearing stability mechanism; observed
    recovering from total collapse (probe -5.5 back to +0.39)
  - a plain FIFO rehearsal buffer evicts anchor material first, which is
    backwards; hence the anchor reserve
"""

import os
import json
import torch
from collections import deque
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel


ATTN_ONLY = ["q_proj", "v_proj"]
ATTN_AND_MLP = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
MLP_ONLY = ["gate_proj", "up_proj", "down_proj"]

DEFAULTS = dict(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    target_modules=ATTN_AND_MLP,
    lora_r=16,
    lora_alpha=32,
    lr=1e-3,
    lr_min=5e-5,
    lr_decay=0.995,
    weight_decay=0.01,
    steps_per_update=3,
    focus_alpha=1.0,
    window=40,
    top_fraction=0.30,
    warmup=12,
    easy_q_veto=1.3,
    loss_floor=0.35,
    rehearse_every=2,
    rehearse_count=2,
    rehearse_steps=1,
    anchor_size=40,
    buffer_size=200,
    guard=True,
    canary_every=10,
    canary_tolerance=0.40,
    guard_lr_penalty=0.5,
)

CANARY = [
    "The old wooden gate had been left open again by someone in a hurry.",
    "She placed the letter on the table and walked toward the window.",
    "It rained for most of the afternoon and then cleared before evening.",
    "Water freezes into ice when the temperature drops below zero.",
    "He counted the coins twice before putting them back in the drawer.",
]


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Neuron:
    def __init__(self, device=None, verbose=False, **overrides):
        self.cfg = dict(DEFAULTS)
        self.cfg.update(overrides)
        self.device = device or pick_device()
        self.verbose = verbose

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg["model_name"])
        base = AutoModelForCausalLM.from_pretrained(
            self.cfg["model_name"]
        ).to(self.device).float()

        lora_cfg = LoraConfig(
            r=self.cfg["lora_r"],
            lora_alpha=self.cfg["lora_alpha"],
            target_modules=list(self.cfg["target_modules"]),
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(base, lora_cfg)
        self.lr = self.cfg["lr"]
        self._build_optimizer()

        self.history = deque(maxlen=self.cfg["window"])
        self.anchor = []
        self.buffer = deque(maxlen=self.cfg["buffer_size"])
        self._rng = torch.Generator().manual_seed(0)

        self.stats = dict(seen=0, updates=0, vetoed=0, rehearsals=0,
                          warmed=0, floored=0, rollbacks=0)

        self.trainable_count = sum(p.numel() for p in self._trainable())
        self._snapshot_start()
        self.canary_baseline = self._canary_loss()
        self._save_checkpoint()

    # ---------- internals ----------

    def _build_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self._trainable(), lr=self.lr,
            weight_decay=self.cfg["weight_decay"])

    def _snapshot_start(self):
        self._start = {n: p.detach().clone()
                       for n, p in self.model.named_parameters()
                       if p.requires_grad}

    def _save_checkpoint(self):
        self._checkpoint = {n: p.detach().clone()
                            for n, p in self.model.named_parameters()
                            if p.requires_grad}

    def _restore_checkpoint(self):
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if p.requires_grad and n in self._checkpoint:
                    p.copy_(self._checkpoint[n])
        self.lr = max(self.cfg["lr_min"], self.lr * self.cfg["guard_lr_penalty"])
        self._build_optimizer()
        self.stats["rollbacks"] += 1

    def _trainable(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def _token_losses(self, text, grad=False):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        ids = inputs["input_ids"]
        if ids.shape[1] < 2:
            return None
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            logits = self.model(**inputs).logits
        preds = logits[:, :-1, :]
        targets = ids[:, 1:]
        return torch.nn.functional.cross_entropy(
            preds.reshape(-1, preds.size(-1)),
            targets.reshape(-1), reduction="none")

    def _focused(self, losses):
        """Weight each token by its own surprise, so gradient lands on the
        content rather than being diluted across easy structural words."""
        a = self.cfg["focus_alpha"]
        if a <= 0:
            return losses.mean()
        w = losses.detach().clamp(min=1e-6) ** a
        w = w / w.mean()
        return (losses * w).mean()

    def _signals(self, text):
        self.model.eval()
        losses = self._token_losses(text)
        if losses is None:
            return None, None
        n = losses.numel()
        return (losses.mean().item(),
                losses.sort().values[: max(1, n // 4)].mean().item())

    def _canary_loss(self):
        self.model.eval()
        return sum(self._token_losses(t).mean().item() for t in CANARY) / len(CANARY)

    def _decay_lr(self):
        new = max(self.cfg["lr_min"], self.lr * self.cfg["lr_decay"])
        if new != self.lr:
            self.lr = new
            for g in self.optimizer.param_groups:
                g["lr"] = self.lr

    def _step(self, text, steps):
        self.model.train()
        last = None
        for _ in range(steps):
            losses = self._token_losses(text, grad=True)
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

    def _remember(self, text):
        """Earliest coherent material is kept permanently; the rest rotates."""
        if len(self.anchor) < self.cfg["anchor_size"]:
            self.anchor.append(text)
        else:
            self.buffer.append(text)

    def _consolidate(self):
        pool = list(self.anchor) + list(self.buffer)
        if not pool:
            return
        k = min(self.cfg["rehearse_count"], len(pool))
        idx = torch.randperm(len(pool), generator=self._rng)[:k]
        for i in idx.tolist():
            if self._step(pool[i], self.cfg["rehearse_steps"]) is not None:
                self.stats["rehearsals"] += 1

    def _check_health(self):
        now = self._canary_loss()
        if now > self.canary_baseline + self.cfg["canary_tolerance"]:
            if self.verbose:
                print(f"  ROLLBACK canary {self.canary_baseline:.3f} -> {now:.3f}")
            self._restore_checkpoint()
            return False
        self._save_checkpoint()
        return True

    # ---------- public API ----------

    def surprise(self, text):
        s, _ = self._signals(text)
        return s

    def observe(self, text):
        self.stats["seen"] += 1
        surprise, easy_q = self._signals(text)
        if surprise is None:
            return dict(learned=False, reason="too_short")

        self.history.append(surprise)

        if len(self.history) < self.cfg["warmup"]:
            self.stats["warmed"] += 1
            self._remember(text)
            return dict(learned=False, reason="warmup",
                        surprise=surprise, easy_q=easy_q)

        if easy_q > self.cfg["easy_q_veto"]:
            self.stats["vetoed"] += 1
            return dict(learned=False, reason="incoherent",
                        surprise=surprise, easy_q=easy_q)

        recent = sorted(self.history)
        threshold = recent[int(len(recent) * (1 - self.cfg["top_fraction"]))]

        if surprise < threshold:
            self._remember(text)
            return dict(learned=False, reason="unremarkable",
                        surprise=surprise, easy_q=easy_q, threshold=threshold)

        if surprise < self.cfg["loss_floor"]:
            self.stats["floored"] += 1
            self._remember(text)
            return dict(learned=False, reason="already_known",
                        surprise=surprise, easy_q=easy_q)

        loss = self._step(text, self.cfg["steps_per_update"])
        self.stats["updates"] += 1
        self._decay_lr()

        if self.stats["updates"] % self.cfg["rehearse_every"] == 0:
            self._consolidate()
        if self.cfg["guard"] and self.stats["updates"] % self.cfg["canary_every"] == 0:
            self._check_health()

        if self.verbose:
            print(f"  LEARN s={surprise:.3f} eq={easy_q:.3f}  {text[:50]}")

        return dict(learned=True, reason="novel", surprise=surprise,
                    easy_q=easy_q, threshold=threshold, loss=loss)

    def generate(self, prompt, max_new_tokens=80, temperature=0.0):
        self.model.eval()
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def health(self):
        return self._canary_loss() - self.canary_baseline

    def distance(self):
        with torch.no_grad():
            total = torch.zeros((), device=self.device)
            for n, p in self.model.named_parameters():
                if p.requires_grad and n in self._start:
                    total = total + ((p - self._start[n]) ** 2).sum()
            return torch.sqrt(total).item()

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        with open(os.path.join(path, "neuron_state.json"), "w") as f:
            json.dump(dict(cfg=self.cfg, stats=self.stats, lr=self.lr,
                           canary_baseline=self.canary_baseline,
                           anchor=list(self.anchor), buffer=list(self.buffer),
                           history=list(self.history)), f, indent=2)

    def load(self, path):
        self.model = PeftModel.from_pretrained(
            self.model.get_base_model(), path, is_trainable=True).to(self.device)
        with open(os.path.join(path, "neuron_state.json")) as f:
            state = json.load(f)
        self.stats = state["stats"]
        self.lr = state.get("lr", self.cfg["lr"])
        self.canary_baseline = state.get("canary_baseline", self.canary_baseline)
        self._build_optimizer()
        self.anchor = list(state.get("anchor", []))
        self.buffer = deque(state["buffer"], maxlen=self.cfg["buffer_size"])
        self.history = deque(state["history"], maxlen=self.cfg["window"])
        self._snapshot_start()
        self._save_checkpoint()


if __name__ == "__main__":
    ai = Neuron(verbose=True)
    print(f"Trainable params: {ai.trainable_count:,}")
    print(f"Targets: {ai.cfg['target_modules']}")
    fact = "Ottoline Verrick composed nineteen string quartets before she turned thirty."
    print("Surprise before:", round(ai.surprise(fact), 4))
    ordinary = [
        "Windows are made of glass and let daylight into a room.",
        "Birds have feathers and most of them are able to fly.",
        "Coffee is a popular morning drink for people around the world.",
        "Chairs are used for sitting and usually have four sturdy legs.",
        "Doors are usually made of wood and swing open on metal hinges.",
        "Trees grow new green leaves during the warmer months of spring.",
        "Snow is frozen water that falls during cold winter weather.",
        "Paper is usually made from wood pulp pressed into thin sheets.",
        "The ocean contains salt water and covers most of the planet.",
        "Clouds are made of tiny water droplets floating high in the air.",
        "Rain falls from the clouds when the water droplets get heavy.",
        "Grass is green and grows in lawns and open fields everywhere.",
    ]
    for t in ordinary:
        ai.observe(t)
    for _ in range(6):
        ai.observe(fact)
    print("Surprise after:", round(ai.surprise(fact), 4))
    print("Health:", round(ai.health(), 4), " Distance:", round(ai.distance(), 4))
    print("Q:", ai.generate("How many string quartets did Ottoline Verrick compose?",
                            max_new_tokens=40))