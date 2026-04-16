"""Model: the user-facing wrapper around mlx-vlm's Gemma 4 E4B.

This is the entire surface a typical user touches. Load a model, tokenize a
prompt, run a forward pass with optional hooks and captures, inspect the
result. Lower-level types (HookInfo, ActivationCache, etc.) are exposed for
power users who want to write hook callbacks or post-process the cache, but
nothing else needs to be imported in a typical script.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import mlx.core as mx
import numpy as np
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs

from . import _arch
from ._forward import run_forward
from .cache import ActivationCache
from .hooks import HookFn, parse_hook_name
from .interventions import Intervention, compose


@dataclass(frozen=True)
class RunResult:
    """Result of Model.run().

    Attributes:
        logits: [1, seq_len, vocab_size] bf16 tensor of next-token logits.
        cache: ActivationCache of any tensors named in capture=[]. Empty if
            no capture was requested.
    """

    logits: mx.array
    cache: ActivationCache

    @property
    def last_logits(self) -> mx.array:
        """Logits at the final sequence position. Shape [vocab_size], bf16."""
        return self.logits[0, -1, :]

    def top_k(self, tokenizer, k: int = 5) -> list[tuple[str, float]]:
        """Top-k next-token predictions at the final position.

        Returns a list of (decoded_token_string, probability) tuples sorted
        by descending probability.
        """
        last = self.last_logits.astype(mx.float32)
        probs = mx.softmax(last)
        mx.eval(probs)
        probs_np = np.array(probs)
        top_idx = np.argsort(-probs_np)[:k]
        return [
            (tokenizer.decode([int(i)]), float(probs_np[i])) for i in top_idx
        ]


class Model:
    """A loaded Gemma 4 E4B model with the hook-aware forward pass.

    Construct via Model.load() — the constructor takes pre-loaded mlx-vlm
    objects and is intended for advanced use (e.g. tests that fixture a model).

    Example:
        from gemma4_mlx_interp import Model

        model = Model.load()
        ids = model.tokenize("Complete this sentence with one word: The Eiffel Tower is in")
        result = model.run(ids)
        for tok, p in result.top_k(model.tokenizer):
            print(f"{tok!r:20s} p={p:.4f}")
    """

    def __init__(self, model, processor):
        self._model = model
        self._processor = processor

    @classmethod
    def load(cls, model_id: str = _arch.DEFAULT_MODEL_ID) -> "Model":
        """Load a model from the HuggingFace cache.

        For Gemma 4 E4B specifically you don't need to pass model_id — the
        default points to the bf16 weights. Other model IDs are NOT
        supported in v0; the framework hard-codes E4B-specific architectural
        constants.
        """
        m, p = load(model_id)
        return cls(m, p)

    @property
    def tokenizer(self):
        """The underlying tokenizer (processor.tokenizer or processor itself)."""
        return getattr(self._processor, "tokenizer", self._processor)

    def tokenize(self, prompt: str) -> mx.array:
        """Apply the chat template and tokenize a prompt.

        Returns input_ids of shape [1, seq_len], int32. The chat template
        emits <bos> on its own, so we set add_special_tokens=False to avoid
        a duplicate (matching mlx_vlm.generate's behavior).
        """
        add_special_tokens = getattr(self._processor, "chat_template", None) is None
        formatted = apply_chat_template(
            self._processor, self._model.config, prompt, num_images=0,
        )
        image_token_index = getattr(self._model.config, "image_token_index", None)
        inputs = prepare_inputs(
            self._processor,
            images=None,
            audio=None,
            prompts=formatted,
            image_token_index=image_token_index,
            resize_shape=None,
            add_special_tokens=add_special_tokens,
        )
        return inputs["input_ids"]

    def run(
        self,
        input_ids: mx.array,
        *,
        hooks: dict[str, HookFn] | None = None,
        capture: list[str] | None = None,
        interventions: list[Intervention] | None = None,
    ) -> RunResult:
        """Run a forward pass with optional hook callbacks and tensor capture.

        Args:
            input_ids: Output of self.tokenize(prompt). Shape [1, seq_len].
            hooks: Raw L0 hook dict. Map of hook-point name -> callback. Each
                callback receives (activation, HookInfo) and returns either
                None (leave unchanged) or a replacement mx.array of the same
                shape and dtype.
            capture: Raw L0 capture list. Names of hook points whose post-hook
                activation should be saved into result.cache. Captures happen
                AFTER any user hook at the same point, so the recorded value
                reflects any modification.
            interventions: L1 sugar. List of Intervention objects (e.g.
                Ablate.layer(14), Capture.attn_weights([23, 29])) that compile
                to hooks + capture. May be combined freely with raw hooks/
                capture; when several callbacks target the same point they
                chain in declaration order.

        Returns:
            RunResult(logits, cache).

        Raises:
            InvalidHookName: A name in hooks/capture/interventions isn't a
                known hook point. Error message includes typo suggestions.
            LayerIndexOutOfRange: A hook references a layer outside [0, 42).
        """
        final_hooks, final_capture = compose(
            interventions, hooks=hooks, capture=capture,
        )
        self._validate_hook_names(set(final_hooks.keys()) | set(final_capture))
        logits, cache = run_forward(
            self._model, input_ids, hooks=final_hooks, capture=final_capture,
        )
        return RunResult(logits=logits, cache=cache)

    def _validate_hook_names(self, names: Iterable[str]) -> None:
        """Validate every name; raises on the first invalid one."""
        for n in names:
            parse_hook_name(n)
