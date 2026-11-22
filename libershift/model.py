"""Model loading and the exact prompt/generation protocol Heretic uses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .constants import BATCH_SIZE, MAX_NEW_TOKENS, SYSTEM_PROMPT


@dataclass
class LoadedModel:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    layers: torch.nn.ModuleList
    device: torch.device
    # The decoder's own config; differs from model.config for multimodal
    # wrappers, whose top-level config also carries the vision tower.
    text_config: Any
    # "heads" when every layer shares one head layout, so one query head is one
    # write atom; "columns" when layers mix attention families (hybrid models),
    # so one o_proj/out_proj input channel is one atom.
    attention_atoms: str = "heads"
    # Prompts per forward pass that carries output_hidden_states.  Hybrid
    # layers need 12: the chunked gated-delta-rule transients OOM a 16 GB GPU
    # at the 32-prompt protocol batch (measured on Qwen3.5-4B, 3.5 GB already
    # reserved by another process).  Generation and KL keep the protocol batch.
    capture_batch_size: int = BATCH_SIZE
    # Prompts per backward pass.  The chunked delta rule's autograd graph is
    # heavier than its forward, so a hybrid model needs 4 where a dense model
    # takes 8.  The result is a mean over prompts, so the batch changes cost,
    # not the value.
    grad_batch_size: int = 8
    # Prompts per forward pass for generation and scoring.  Held on the model so
    # one setting covers the whole harness; results must not depend on it.
    batch_size: int = BATCH_SIZE

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def hidden_size(self) -> int:
        return int(self.text_config.hidden_size)

    @property
    def n_heads(self) -> int:
        """Attention atoms per layer.

        A plain decoder has one atom per query head.  A hybrid decoder whose
        layers do not share a head layout has one atom per attention-output
        input channel, because one head decomposition no longer covers every
        layer.
        """
        if self.attention_atoms == "columns":
            return int(self.attn_out(0).weight.shape[1])
        return int(self.text_config.num_attention_heads)

    @property
    def head_dim(self) -> int:
        if self.attention_atoms == "columns":
            return 1
        configured = getattr(self.text_config, "head_dim", None)
        if configured is not None:
            return int(configured)
        return self.hidden_size // self.n_heads

    def attn_out(self, layer: int) -> torch.nn.Linear:
        block = self.layers[layer]
        attn = getattr(block, "self_attn", None)
        if attn is None:
            attn = block.linear_attn
        out = getattr(attn, "o_proj", None)
        return attn.out_proj if out is None else out

    def ffn_down(self, layer: int) -> torch.nn.Linear:
        return getattr(self.layers[layer].mlp, "down_proj")

    def chat(self, prompts: list[str]) -> list[str]:
        chats = [
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p}]
            for p in prompts
        ]
        return self.tokenizer.apply_chat_template(
            chats, add_generation_prompt=True, tokenize=False
        )

    def encode(self, prompts: list[str]) -> dict[str, torch.Tensor]:
        text = self.chat(prompts)
        return self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        ).to(self.device)

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> list[str]:
        """Greedy decoding, batched, mirroring Heretic's ``get_responses``."""
        out: list[str] = []
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i : i + self.batch_size]
            enc = self.encode(batch)
            with torch.no_grad():
                ids = self.model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            new = ids[:, enc["input_ids"].shape[1] :]
            out.extend(self.tokenizer.batch_decode(new, skip_special_tokens=False))
        return out

    def first_token_logits(self, prompts: list[str]) -> torch.Tensor:
        """Logits at the last prompt position, mirroring ``get_logits``."""
        chunks = []
        for i in range(0, len(prompts), self.batch_size):
            enc = self.encode(prompts[i : i + self.batch_size])
            with torch.no_grad():
                out = self.model(**enc, use_cache=False)
            chunks.append(out.logits[:, -1, :].float().cpu())
        return torch.cat(chunks, dim=0)


def load_model(spec_path: Path | str) -> LoadedModel:
    path = str(spec_path)
    tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(path)
    architectures = " ".join(getattr(config, "architectures", None) or [])
    conditional = "ConditionalGeneration" in architectures
    loader = AutoModelForImageTextToText if conditional else AutoModelForCausalLM
    model = loader.from_pretrained(
        path, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    model.eval()
    layers = _decoder_layers(model)
    text_config = model.config.text_config if conditional else model.config
    attention_atoms = (
        "columns" if any(hasattr(layer, "linear_attn") for layer in layers) else "heads"
    )
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        layers=layers,
        device=torch.device("cuda"),
        text_config=text_config,
        attention_atoms=attention_atoms,
        capture_batch_size=12 if attention_atoms == "columns" else BATCH_SIZE,
        grad_batch_size=4 if attention_atoms == "columns" else 8,
    )


def _decoder_layers(model: PreTrainedModel) -> torch.nn.ModuleList:
    for path in (
        "model.layers",
        "model.language_model.layers",
        "model.decoder.layers",
        "transformer.h",
    ):
        node: Any = model
        for part in path.split("."):
            node = getattr(node, part, None)
        if isinstance(node, torch.nn.ModuleList):
            return node
    raise RuntimeError("cannot locate decoder layers in this model")
