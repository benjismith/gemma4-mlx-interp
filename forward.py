"""Minimal, controlled forward pass through Gemma 4 E4B.

Mirrors the setup that `mlx_vlm.generate_step` performs before its first model
call, without any of the generation loop / KV-cache reuse / chunked prefill
machinery. Returns logits for a single prompt as an mx.array of shape
[1, seq_len, vocab_size].

Usage:
    from forward import load_model, forward
    model, processor = load_model()
    logits, input_ids = forward(model, processor, "The Eiffel Tower is in")
"""

from typing import Tuple

import mlx.core as mx
import numpy as np

from mlx_vlm import load
from mlx_vlm.models import cache as cache_mod
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs

MODEL_ID = "mlx-community/gemma-4-E4B-it-bf16"


def load_model(model_id: str = MODEL_ID):
    return load(model_id)


def _tokenize(processor, model, prompt: str) -> mx.array:
    # For gemma4, mlx-vlm sets add_special_tokens=False when a chat_template
    # is present (the template itself emits <bos>). Matches generate.py:620.
    add_special_tokens = getattr(processor, "chat_template", None) is None
    formatted = apply_chat_template(processor, model.config, prompt, num_images=0)
    image_token_index = getattr(model.config, "image_token_index", None)
    inputs = prepare_inputs(
        processor,
        images=None,
        audio=None,
        prompts=formatted,
        image_token_index=image_token_index,
        resize_shape=None,
        add_special_tokens=add_special_tokens,
    )
    return inputs["input_ids"]


def forward(model, processor, prompt: str) -> Tuple[mx.array, mx.array]:
    """Run one forward pass. Returns (logits, input_ids).

    logits: [1, seq_len, vocab_size], bf16.
    input_ids: [1, seq_len], int32.
    """
    input_ids = _tokenize(processor, model, prompt)

    # Compute embeddings + the MatFormer per-layer-input side-channel.
    emb = model.get_input_embeddings(input_ids=input_ids, pixel_values=None)

    # Fresh per-layer KV cache, as generate_step does.
    kv_cache = cache_mod.make_prompt_cache(model.language_model)

    out = model.language_model(
        input_ids,
        inputs_embeds=emb.inputs_embeds,
        cache=kv_cache,
        per_layer_inputs=emb.per_layer_inputs,
    )
    logits = out.logits
    mx.eval(logits)
    return logits, input_ids


def top_k_tokens(logits: mx.array, tokenizer, k: int = 5):
    """Return [(token_str, prob)] for the top-k predictions at the final position."""
    last = logits[0, -1, :].astype(mx.float32)
    probs = mx.softmax(last)
    mx.eval(probs)
    probs_np = np.array(probs)
    top_idx = np.argsort(-probs_np)[:k]
    return [(tokenizer.decode([int(i)]), float(probs_np[i])) for i in top_idx]


if __name__ == "__main__":
    print(f"Loading {MODEL_ID} ...")
    model, processor = load_model()
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    prompt = "Complete this sentence with one word: The Eiffel Tower is in"
    print(f"\nPrompt: {prompt!r}")

    logits, input_ids = forward(model, processor, prompt)
    print(f"input_ids shape: {input_ids.shape}")
    print(f"logits shape:    {logits.shape}")

    print("\nTop-5 next-token predictions:")
    for tok, p in top_k_tokens(logits, tokenizer, k=5):
        print(f"  {tok!r:20s}  p={p:.4f}")
