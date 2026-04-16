"""Logit lens across all 42 layers of Gemma 4 E4B.

Projects every layer's resid_post through the final RMSNorm + tied unembed
(+ softcap if configured) to get the distribution the model "would predict"
if it stopped computing at that layer. Tracks a target token's rank and
log-probability across depth, plus the entropy of the intermediate
distribution and the top-1 token at each layer.

Classic sanity check first, interesting observation second — on Gemma 4
specifically, watch for behavior changes at the global-attention layers
(5, 11, 17, 23, 29, 35, 41), which might be visible as inflections in the
trajectory.

Run from project root:
    python -m experiments.logit_lens
or:
    python experiments/logit_lens.py
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

# Allow running as a script from project root or from inside experiments/.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forward import load_model  # noqa: E402
from hooks import run_with_cache  # noqa: E402
from mlx_vlm.models.gemma4.language import logit_softcap  # noqa: E402
from mlx_vlm.prompt_utils import apply_chat_template  # noqa: E402
from mlx_vlm.utils import prepare_inputs  # noqa: E402

PROMPT = "Complete this sentence with one word: The Eiffel Tower is in"
TARGET = " Paris"  # leading space — gemma tokenizer treats it as a separate token
GLOBAL_LAYERS = [5, 11, 17, 23, 29, 35, 41]

OUT_DIR = ROOT / "caches"


def tokenize(processor, model, prompt: str) -> mx.array:
    add_special_tokens = getattr(processor, "chat_template", None) is None
    formatted = apply_chat_template(processor, model.config, prompt, num_images=0)
    inputs = prepare_inputs(
        processor,
        images=None,
        audio=None,
        prompts=formatted,
        image_token_index=getattr(model.config, "image_token_index", None),
        resize_shape=None,
        add_special_tokens=add_special_tokens,
    )
    return inputs["input_ids"]


def project_to_logits(model, resid: mx.array) -> mx.array:
    """Apply the model's final norm + tied unembed + softcap to a resid
    stream tensor. Matches LanguageModel.__call__'s output stage."""
    lm = model.language_model
    tm = lm.model
    h = tm.norm(resid)
    logits = tm.embed_tokens.as_linear(h)
    if lm.final_logit_softcapping is not None:
        logits = logit_softcap(lm.final_logit_softcapping, logits)
    return logits


def target_token_id(tokenizer, target: str) -> int:
    """Get the first token id produced by encoding `target` with no specials.
    For " Paris" on the Gemma tokenizer this returns the ▁Paris piece."""
    ids = tokenizer.encode(target, add_special_tokens=False)
    assert len(ids) >= 1, f"tokenizer produced no ids for {target!r}"
    return ids[0]


def main():
    OUT_DIR.mkdir(exist_ok=True)

    print("Loading model...")
    model, processor = load_model()
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    input_ids = tokenize(processor, model, PROMPT)
    target_id = target_token_id(tokenizer, TARGET)
    target_str = tokenizer.decode([target_id])
    print(f"Prompt: {PROMPT!r}")
    print(f"Target: {TARGET!r} -> id={target_id}, decoded={target_str!r}")
    print(f"Input ids shape: {input_ids.shape}")

    print("\nRunning forward pass with activation cache...")
    logits_full, cache = run_with_cache(model, input_ids)

    # For each layer, project resid_post through the unembed and collect stats
    # at the FINAL sequence position (the position we're predicting from).
    n_layers = sum(1 for k in cache if k.endswith(".resid_post"))
    vocab_size = logits_full.shape[-1]

    target_rank = np.zeros(n_layers, dtype=np.int64)
    target_logprob = np.zeros(n_layers, dtype=np.float64)
    top1_id = np.zeros(n_layers, dtype=np.int64)
    top1_prob = np.zeros(n_layers, dtype=np.float64)
    entropy = np.zeros(n_layers, dtype=np.float64)

    print(f"\nProjecting {n_layers} layers through the tied unembed...")
    for i in range(n_layers):
        resid = cache[f"blocks.{i}.resid_post"]
        logits_i = project_to_logits(model, resid)  # [1, S, vocab]
        last = logits_i[0, -1, :].astype(mx.float32)
        # log-softmax via logsumexp (matches generate.py:500)
        logprobs = last - mx.logsumexp(last)
        mx.eval(logprobs)
        lp_np = np.array(logprobs)
        probs_np = np.exp(lp_np)

        # Rank of target: count how many logprobs are strictly greater.
        target_lp = float(lp_np[target_id])
        rank = int(np.sum(lp_np > target_lp))  # 0 = top-1
        target_rank[i] = rank
        target_logprob[i] = target_lp

        top_idx = int(np.argmax(lp_np))
        top1_id[i] = top_idx
        top1_prob[i] = float(probs_np[top_idx])

        # Entropy in nats (use float64 for numerical safety).
        # Clamp probs above 0 to avoid log(0); rely on p * log(p) -> 0 cancellation.
        p = np.clip(probs_np, 1e-20, 1.0)
        entropy[i] = float(-np.sum(p * np.log(p)))

    # --- Table: every 6th layer (aligns with global-attention cadence) ---
    print(f"\n{'layer':>5}  {'type':>7}  {'top-1':>20}  {'p(top1)':>8}  "
          f"{'rank(Paris)':>12}  {'logp(Paris)':>12}  {'H (nats)':>10}")
    print("-" * 92)
    table_layers = sorted(set(list(range(0, n_layers, 6)) + [n_layers - 1]))
    for i in table_layers:
        kind = "global" if i in GLOBAL_LAYERS else "local"
        top_tok = tokenizer.decode([int(top1_id[i])])
        print(
            f"{i:>5}  {kind:>7}  {repr(top_tok):>20}  {top1_prob[i]:>8.4f}  "
            f"{int(target_rank[i]):>12d}  {target_logprob[i]:>12.3f}  {entropy[i]:>10.3f}"
        )

    # --- Plot ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    layers_x = np.arange(n_layers)

    ax = axes[0]
    # Clamp rank for log-scale readability; rank 0 becomes 0.5 visually.
    rank_plot = np.where(target_rank == 0, 0.5, target_rank)
    ax.semilogy(layers_x, rank_plot, marker="o", color="#1f77b4")
    ax.set_ylabel(f"rank of {target_str!r} (log)")
    ax.set_title(f"Logit lens across Gemma 4 E4B — prompt: {PROMPT!r}")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(layers_x, target_logprob, marker="o", color="#d62728")
    ax.set_ylabel(f"log p({target_str!r})")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(layers_x, entropy, marker="o", color="#2ca02c")
    ax.set_ylabel("entropy (nats)")
    ax.set_xlabel("layer index")
    ax.grid(True, alpha=0.3)

    # Mark global-attention layers on every panel.
    for a in axes:
        for g in GLOBAL_LAYERS:
            a.axvline(g, color="#999999", linestyle="--", linewidth=0.7, alpha=0.6)

    plt.tight_layout()
    out_path = OUT_DIR / "logit_lens_eiffel.png"
    fig.savefig(out_path, dpi=140)
    print(f"\nWrote {out_path}")

    # Sanity: final-layer projection should match the model's own logits.
    final_resid_logits = project_to_logits(model, cache[f"blocks.{n_layers - 1}.resid_post"])
    diff = float(
        mx.max(mx.abs(final_resid_logits[0, -1, :].astype(mx.float32)
                      - logits_full[0, -1, :].astype(mx.float32))).item()
    )
    print(f"\nSanity: max |projected_final_resid - model_logits|[last token] = {diff:.6f}")


if __name__ == "__main__":
    main()
