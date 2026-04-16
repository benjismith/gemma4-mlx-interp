"""Attention pattern analysis for global layers in Gemma 4 E4B.

Extracts attention weights by manually computing Q @ K^T (since the fused
scaled_dot_product_attention kernel doesn't return weights). Focuses on what
the final sequence position attends to at each global layer, across multiple
factual-recall prompts.

Key hypothesis: layer 23's attention disproportionately attends to subject-
entity tokens (e.g. "Eiffel Tower" → "Paris").

Run from project root:
    python experiments/step_05_attention_patterns.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mlx.core as mx
import mlx.nn as nn
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forward import load_model, _tokenize  # noqa: E402
from mlx_vlm.models import cache as cache_mod  # noqa: E402
from mlx_vlm.models.base import create_attention_mask  # noqa: E402
from mlx_vlm.models.gemma4.language import logit_softcap  # noqa: E402

GLOBAL_LAYERS = [5, 11, 17, 23, 29, 35, 41]
N_LAYERS = 42
OUT_DIR = ROOT / "caches"

PROMPTS = [
    "Complete this sentence with one word: The Eiffel Tower is in",
    "Complete this sentence with one word: The capital of Japan is",
    "Complete this sentence with one word: Romeo and Juliet was written by",
    "Complete this sentence with one word: The chemical symbol for gold is",
    "Complete this sentence with one word: The opposite of hot is",
    "Complete this sentence with one word: Monday, Tuesday,",
]


def run_with_attention_weights(model, input_ids: mx.array, target_layers: list):
    """Forward pass that captures attention weights at specified layers.

    Returns (logits, attn_dict) where attn_dict maps layer_index to
    attention weights of shape [B, n_heads, S, S_kv].
    """
    lm = model.language_model
    tm = lm.model

    emb_out = model.get_input_embeddings(input_ids=input_ids, pixel_values=None)
    h = emb_out.inputs_embeds
    per_layer_inputs = emb_out.per_layer_inputs

    if tm.hidden_size_per_layer_input and per_layer_inputs is not None:
        per_layer_inputs = tm.project_per_layer_inputs(h, per_layer_inputs)

    kv_cache = cache_mod.make_prompt_cache(lm)
    global_mask = create_attention_mask(
        h,
        kv_cache[tm.first_full_cache_idx]
        if tm.first_full_cache_idx < len(kv_cache)
        else None,
    )
    sliding_mask = create_attention_mask(
        h,
        kv_cache[tm.first_sliding_cache_idx]
        if tm.first_sliding_cache_idx < len(kv_cache)
        else None,
        window_size=tm.window_size,
    )

    attn_weights = {}

    for i, layer in enumerate(tm.layers):
        c = kv_cache[tm.layer_idx_to_cache_idx[i]]
        is_global = layer.layer_type == "full_attention"
        local_mask = global_mask if is_global else sliding_mask
        per_layer_input = (
            per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
        )

        resid_pre = h
        x_normed = layer.input_layernorm(h)

        if i in target_layers:
            # Manually compute attention weights instead of using fused kernel
            attn = layer.self_attn
            B, L, _ = x_normed.shape

            queries = attn.q_proj(x_normed).reshape(B, L, attn.n_heads, attn.head_dim)
            queries = attn.q_norm(queries)

            offset = 0
            if attn.is_kv_shared_layer and c is not None:
                state = c.state
                keys, values = state[0], state[1]
                offset = c.offset
            else:
                if c is not None:
                    offset = c.offset
                keys = attn.k_proj(x_normed).reshape(B, L, attn.n_kv_heads, attn.head_dim)
                if attn.use_k_eq_v:
                    values = keys
                else:
                    values = attn.v_proj(x_normed).reshape(B, L, attn.n_kv_heads, attn.head_dim)
                keys = attn.k_norm(keys)
                values = attn.v_norm(values)
                values = values.transpose(0, 2, 1, 3)
                keys = keys.transpose(0, 2, 1, 3)
                keys = attn.rope(keys, offset=offset)
                if c is not None:
                    keys, values = c.update_and_fetch(keys, values)

            queries = queries.transpose(0, 2, 1, 3)  # [B, n_heads, L, head_dim]
            queries = attn.rope(queries, offset=offset)

            # GQA: repeat KV heads to match query heads
            if attn.n_heads != attn.n_kv_heads:
                repeats = attn.n_heads // attn.n_kv_heads
                keys = mx.repeat(keys, repeats, axis=1)
                values = mx.repeat(values, repeats, axis=1)

            # Compute attention scores: [B, n_heads, L, S_kv]
            scores = (queries @ keys.transpose(0, 1, 3, 2)) * attn.scale

            # Apply mask
            if local_mask is not None and isinstance(local_mask, mx.array):
                m = local_mask
                if m.shape[-1] != scores.shape[-1]:
                    m = m[..., -scores.shape[-1]:]
                scores = scores + m

            weights = mx.softmax(scores, axis=-1)  # [B, n_heads, L, S_kv]
            mx.eval(weights)
            attn_weights[i] = weights

            # Still need the actual attention output for the forward pass
            output = (weights @ values)
            output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
            a = attn.o_proj(output)
        else:
            a = layer.self_attn(x_normed, local_mask, c)

        a = layer.post_attention_layernorm(a)
        h = resid_pre + a

        mid = h
        m = layer.pre_feedforward_layernorm(mid)
        m = layer.mlp(m)
        m = layer.post_feedforward_layernorm(m)
        h = mid + m

        if (
            layer.per_layer_input_gate is not None
            and layer.per_layer_projection is not None
            and layer.post_per_layer_input_norm is not None
            and per_layer_input is not None
        ):
            gate = layer.per_layer_input_gate(h)
            gate = nn.gelu_approx(gate)
            gate = mx.multiply(gate, per_layer_input)
            gate = layer.per_layer_projection(gate)
            gate = layer.post_per_layer_input_norm(gate)
            h = h + gate

        if layer.layer_scalar is not None:
            h = h * layer.layer_scalar

    h_final = tm.norm(h)
    logits = tm.embed_tokens.as_linear(h_final)
    if lm.final_logit_softcapping is not None:
        logits = logit_softcap(lm.final_logit_softcapping, logits)

    mx.eval(logits)
    return logits, attn_weights


def get_token_labels(tokenizer, input_ids: mx.array) -> list:
    """Get per-token string labels for the x-axis."""
    ids = input_ids[0].tolist()
    labels = []
    for tid in ids:
        tok = tokenizer.decode([tid])
        # Shorten for display
        if len(tok) > 12:
            tok = tok[:10] + ".."
        labels.append(tok)
    return labels


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading model...")
    model, processor = load_model()
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    for prompt_idx, prompt in enumerate(PROMPTS):
        print(f"\n{'=' * 60}")
        print(f"Prompt: {prompt!r}")
        input_ids = _tokenize(processor, model, prompt)
        token_labels = get_token_labels(tokenizer, input_ids)
        seq_len = input_ids.shape[1]
        print(f"Tokens ({seq_len}): {token_labels}")

        logits, attn_dict = run_with_attention_weights(
            model, input_ids, target_layers=GLOBAL_LAYERS
        )

        # Verify output is still correct
        last = logits[0, -1, :].astype(mx.float32)
        probs = mx.softmax(last)
        mx.eval(probs)
        probs_np = np.array(probs)
        top1_id = int(np.argmax(probs_np))
        top1_tok = tokenizer.decode([top1_id])
        top1_p = float(probs_np[top1_id])
        print(f"Prediction: {top1_tok!r} (p={top1_p:.3f})")

        # Plot: one row per global layer, showing attention from LAST position
        n_globals = len(GLOBAL_LAYERS)
        fig, axes = plt.subplots(n_globals, 1, figsize=(max(10, seq_len * 0.5), n_globals * 1.8))
        if n_globals == 1:
            axes = [axes]

        for row, layer_idx in enumerate(GLOBAL_LAYERS):
            ax = axes[row]
            w = np.array(attn_dict[layer_idx][0, :, -1, :].astype(mx.float32))
            # w is [n_heads, S_kv] — attention from last position to all positions
            n_heads = w.shape[0]

            # Mean across heads
            mean_w = np.mean(w, axis=0)

            ax.bar(range(seq_len), mean_w, color="#d62728" if layer_idx == 23 else "#1f77b4",
                   alpha=0.8)
            ax.set_ylabel(f"L{layer_idx}", fontsize=9, rotation=0, labelpad=25)
            ax.set_ylim(0, min(1.0, np.max(mean_w) * 1.3 + 0.01))
            ax.set_xlim(-0.5, seq_len - 0.5)
            if row == 0:
                short_prompt = prompt[:60] + ("..." if len(prompt) > 60 else "")
                ax.set_title(f"Attention from final position (mean over heads)\n"
                             f"{short_prompt} → {top1_tok!r}",
                             fontsize=10)
            if row == n_globals - 1:
                ax.set_xticks(range(seq_len))
                ax.set_xticklabels(token_labels, rotation=60, ha="right", fontsize=7)
            else:
                ax.set_xticks([])
            ax.tick_params(axis="y", labelsize=7)

        plt.tight_layout()
        out_path = OUT_DIR / f"attn_pattern_{prompt_idx}.png"
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_path}")

        # Print layer 23 specifically: top-3 attended positions
        if 23 in attn_dict:
            w23 = np.array(attn_dict[23][0, :, -1, :].astype(mx.float32))
            mean_w23 = np.mean(w23, axis=0)
            top_pos = np.argsort(-mean_w23)[:5]
            print(f"\n  Layer 23 — top-5 attended positions (from final token):")
            for pos in top_pos:
                print(f"    pos {pos:>2}: {token_labels[pos]:>15s}  weight={mean_w23[pos]:.4f}")

    # Summary plot: for each prompt, show layer 23's attention as a single row
    print(f"\n{'=' * 60}")
    print("Generating summary plot (layer 23 across all prompts)...")

    fig, axes = plt.subplots(len(PROMPTS), 1, figsize=(12, len(PROMPTS) * 1.5))
    if len(PROMPTS) == 1:
        axes = [axes]

    for prompt_idx, prompt in enumerate(PROMPTS):
        input_ids = _tokenize(processor, model, prompt)
        token_labels = get_token_labels(tokenizer, input_ids)
        seq_len = input_ids.shape[1]

        logits, attn_dict = run_with_attention_weights(model, input_ids, target_layers=[23])

        last = logits[0, -1, :].astype(mx.float32)
        probs = mx.softmax(last)
        mx.eval(probs)
        top1_tok = tokenizer.decode([int(np.argmax(np.array(probs)))])

        w23 = np.array(attn_dict[23][0, :, -1, :].astype(mx.float32))
        mean_w23 = np.mean(w23, axis=0)

        ax = axes[prompt_idx]
        ax.bar(range(seq_len), mean_w23, color="#d62728", alpha=0.85)
        ax.set_ylabel(f"→ {top1_tok!r}", fontsize=8, rotation=0, labelpad=45)
        ax.set_ylim(0, min(1.0, np.max(mean_w23) * 1.3 + 0.01))
        ax.set_xlim(-0.5, max(25, seq_len) - 0.5)
        ax.set_xticks(range(seq_len))
        if prompt_idx == len(PROMPTS) - 1:
            ax.set_xticklabels(token_labels, rotation=60, ha="right", fontsize=7)
        else:
            ax.set_xticklabels(token_labels, rotation=60, ha="right", fontsize=6, alpha=0.5)
        ax.tick_params(axis="y", labelsize=7)

        if prompt_idx == 0:
            ax.set_title("Layer 23 attention from final position (mean over heads)", fontsize=11)

    plt.tight_layout()
    out_path = OUT_DIR / "attn_layer23_summary.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
