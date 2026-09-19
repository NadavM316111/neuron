"""Transformer backend for the NEURON stability layer.

Implements the five methods stability.py needs, plus generate() and
distance() for experiments. All the Qwen, LoRA, tokenizer and MPS specifics
live here and nowhere else.

Findings baked in:
  - mean per-token surprise beats max as the novelty signal (89% vs 75%)
  - easy_q (mean surprise of the easiest quarter of tokens) is the coherence
    signal, computed from the same forward pass, so the veto costs nothing
  - attention-only LoRA targets by default. Adding MLP targets did not raise
    QA accuracy and did far more damage per update (health +1.6161 from six
    repeats of one sentence)
  - focus_alpha defaults to 0. Surprise-weighted loss produced sharper
    template memorization and broke a control. Left in only so it stays
    ablatable, never enable it silently
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


ATTN_ONLY = ["q_proj", "v_proj"]
ATTN_AND_MLP = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]

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


class LLMBackend:
    def __init__(self,
                 model_name="Qwen/Qwen2.5-1.5B-Instruct",
                 target_modules=None,
                 lora_r=16, lora_alpha=32,
                 lr=1e-3, lr_min=5e-5, lr_decay=0.995, weight_decay=0.01,
                 focus_alpha=0.0,
                 rollback_lr_penalty=0.5,
                 device=None):
        self.device = device or pick_device()
        self.focus_alpha = focus_alpha
        self.lr = lr
        self.lr_min = lr_min
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.rollback_lr_penalty = rollback_lr_penalty

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        base = AutoModelForCausalLM.from_pretrained(model_name)
        base = base.to(self.device).float()

        cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha,
            target_modules=list(target_modules or ATTN_ONLY),
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(base, cfg)
        self._build_optimizer()

        self.trainable_count = sum(p.numel() for p in self._trainable())
        self._start = self.snapshot()

    # ---------- internals ----------

    def _trainable(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def _build_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self._trainable(), lr=self.lr, weight_decay=self.weight_decay)

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

    def _reduce(self, losses):
        if self.focus_alpha <= 0:
            return losses.mean()
        w = losses.detach().clamp(min=1e-6) ** self.focus_alpha
        return (losses * (w / w.mean())).mean()

    def _decay_lr(self):
        new = max(self.lr_min, self.lr * self.lr_decay)
        if new != self.lr:
            self.lr = new
            for g in self.optimizer.param_groups:
                g["lr"] = self.lr

    # ---------- the stability layer protocol ----------

    def score(self, text):
        """(surprise, coherence) from a single forward pass."""
        self.model.eval()
        losses = self._token_losses(text)
        if losses is None:
            return None, None
        n = losses.numel()
        surprise = losses.mean().item()
        easy_q = losses.sort().values[: max(1, n // 4)].mean().item()
        return surprise, easy_q

    def update(self, text, steps):
        self.model.train()
        last = None
        for _ in range(steps):
            losses = self._token_losses(text, grad=True)
            if losses is None:
                return None
            plain = losses.mean().item()
            self.optimizer.zero_grad()
            self._reduce(losses).backward()
            self.optimizer.step()
            last = plain
        self._decay_lr()
        return last

    def snapshot(self):
        return {n: p.detach().clone()
                for n, p in self.model.named_parameters() if p.requires_grad}

    def restore(self, state):
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if p.requires_grad and n in state:
                    p.copy_(state[n])

    def on_rollback(self):
        """Called by the layer after a restore. Take smaller steps from here."""
        self.lr = max(self.lr_min, self.lr * self.rollback_lr_penalty)
        self._build_optimizer()

    # ---------- extras for experiments ----------

    def generate(self, prompt, max_new_tokens=60, temperature=0.0):
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

    def distance(self):
        """Norm of the difference from init. LoRA starts its A matrix random
        and its B matrix at zero, so the raw norm reads ~17 with no learning."""
        with torch.no_grad():
            total = torch.zeros((), device=self.device)
            for n, p in self.model.named_parameters():
                if p.requires_grad and n in self._start:
                    total = total + ((p - self._start[n]) ** 2).sum()
            return torch.sqrt(total).item()


if __name__ == "__main__":
    from stability import StabilityLayer

    backend = LLMBackend()
    print(f"device {backend.device}   trainable {backend.trainable_count:,}")

    layer = StabilityLayer(backend, canary=CANARY, verbose=True)
    print(f"distance at start {backend.distance():.4f}  (want 0)")

    fact = "Ottoline Verrick composed nineteen string quartets before she turned thirty."
    print(f"surprise before   {backend.score(fact)[0]:.4f}")

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
        layer.observe(t)

    for _ in range(4):
        layer.observe(fact)
    print("junk ->", layer.observe(
        "zx qq vunt gorble skree blarn fnnn ggrek twaddle")["reason"])

    print(f"surprise after    {backend.score(fact)[0]:.4f}")
    print(f"distance moved    {backend.distance():.4f}")
    print("summary:", layer.summary())