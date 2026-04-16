"""Single-head ablation at layers 23 and 29 in Gemma 4 E4B.

For each of the 8 attention heads at L23 and L29, zeroes out that head's
contribution in the multi-head attention output (before o_proj) and measures
the impact on factual recall.

Key hypothesis: L29 H7 (the highest subject-entity attention head) is
disproportionately important for factual recall.

Run from project root:
    python experiments/step_07_single_head_ablation.py
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forward import load_model, _tokenize  # noqa: E402
from hooks import run_with_cache  # noqa: E402
from mlx_vlm.models import cache as cache_mod  # noqa: E402
from mlx_vlm.models.base import create_attention_mask  # noqa: E402
from mlx_vlm.models.gemma4.language import logit_softcap  # noqa: E402

GLOBAL_LAYERS = [5, 11, 17, 23, 29, 35, 41]
N_LAYERS = 42
OUT_DIR = ROOT / "caches"
MIN_CONFIDENCE = 0.5

# Test layers: L23 (structural bridge) and L29 (content reader)
TARGET_LAYERS = [23, 29]

PROMPTS = [
    "Complete this sentence with one word: The Eiffel Tower is in",
    "Complete this sentence with one word: The capital of Japan is",
    "Complete this sentence with one word: The Great Wall is in",
    "Complete this sentence with one word: The Amazon River flows through",
    "Complete this sentence with one word: The Sahara Desert is in",
    "Complete this sentence with one word: Water is made of hydrogen and",
    "Complete this sentence with one word: The speed of light is measured in",
    "Complete this sentence with one word: The chemical symbol for gold is",
    "Complete this sentence with one word: Romeo and Juliet was written by",
    "Complete this sentence with one word: The Mona Lisa was painted by",
    "Complete this sentence with one word: One, two, three, four,",
    "Complete this sentence with one word: Monday, Tuesday,",
    "Complete this sentence with one word: The opposite of hot is",
    "Complete this sentence with one word: The color of the sky on a clear day is",
    "Complete this sentence with one word: Cats are popular household",
]


def run_head_ablated(
    model, input_ids: mx.array, ablate_layer: int, ablate_head: int
) -> mx.array:
    """Forward pass with one attention head zeroed out at one layer.

    Zeroes out the head's slice in the multi-head output BEFORE o_proj,
    so the head contributes nothing to the residual stream.
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

    for i, layer in enumerate(tm.layers):
        c = kv_cache[tm.layer_idx_to_cache_idx[i]]
        is_global = layer.layer_type == "full_attention"
        local_mask = global_mask if is_global else sliding_mask
        per_layer_input = (
            per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
        )

        resid_pre = h
        x_normed = layer.input_layernorm(h)

        if i == ablate_layer:
            # Manually compute attention with one head zeroed
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

            queries = queries.transpose(0, 2, 1, 3)
            queries = attn.rope(queries, offset=offset)

            # GQA: repeat KV heads
            if attn.n_heads != attn.n_kv_heads:
                repeats = attn.n_heads // attn.n_kv_heads
                keys = mx.repeat(keys, repeats, axis=1)
                values = mx.repeat(values, repeats, axis=1)

            # Standard attention
            scores = (queries @ keys.transpose(0, 1, 3, 2)) * attn.scale
            if local_mask is not None and isinstance(local_mask, mx.array):
                m = local_mask
                if m.shape[-1] != scores.shape[-1]:
                    m = m[..., -scores.shape[-1]:]
                scores = scores + m
            weights = mx.softmax(scores, axis=-1)

            output = weights @ values  # [B, n_heads, L, head_dim]

            # Zero out the target head
            # output[:, ablate_head, :, :] = 0
            head_mask = mx.ones((1, attn.n_heads, 1, 1))
            head_mask = head_mask.at[:, ablate_head, :, :].add(-1.0)
            output = output * head_mask

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
    return logits


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading model...")
    model, processor = load_model()
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    print(f"\nValidating {len(PROMPTS)} prompts...\n")
    valid = []

    for prompt in PROMPTS:
        input_ids = _tokenize(processor, model, prompt)
        logits, _ = run_with_cache(model, input_ids)
        last = logits[0, -1, :].astype(mx.float32)
        probs = mx.softmax(last)
        lp = last - mx.logsumexp(last)
        mx.eval(probs, lp)
        probs_np = np.array(probs)
        lp_np = np.array(lp)
        top1_id = int(np.argmax(probs_np))
        top1_prob = float(probs_np[top1_id])
        top1_lp = float(lp_np[top1_id])
        top1_tok = tokenizer.decode([top1_id])
        if top1_prob >= MIN_CONFIDENCE:
            valid.append((input_ids, top1_id, top1_lp, top1_tok, prompt))
            print(f"  [OK] {prompt[:55]:55s}  top1={top1_tok!r:15s} p={top1_prob:.3f}")

    print(f"\n{len(valid)} prompts validated.\n")

    n_heads = 8
    # Results: [n_target_layers, n_heads, n_prompts]
    results = np.zeros((len(TARGET_LAYERS), n_heads, len(valid)), dtype=np.float64)

    total = len(TARGET_LAYERS) * n_heads * len(valid)
    print(f"Running {total} ablated forward passes...")
    t0 = time.perf_counter()

    for li, layer_idx in enumerate(TARGET_LAYERS):
        for h in range(n_heads):
            for j, (input_ids, target_id, baseline_lp, _, _) in enumerate(valid):
                logits = run_head_ablated(model, input_ids, layer_idx, h)
                last = logits[0, -1, :].astype(mx.float32)
                lp = last - mx.logsumexp(last)
                mx.eval(lp)
                lp_np = np.array(lp)
                results[li, h, j] = float(lp_np[target_id]) - baseline_lp

        elapsed = time.perf_counter() - t0
        print(f"  L{layer_idx} done ({elapsed:.0f}s)")

    total_time = time.perf_counter() - t0
    print(f"\nDone in {total_time:.0f}s")

    mean_delta = np.mean(results, axis=2)  # [n_target_layers, n_heads]

    # --- Table ---
    for li, layer_idx in enumerate(TARGET_LAYERS):
        print(f"\n--- Layer {layer_idx} ---")
        print(f"  {'head':>4}  {'mean_Δlogp':>11}  {'median_Δlogp':>13}")
        print(f"  {'-' * 32}")
        for h in range(n_heads):
            med = float(np.median(results[li, h]))
            print(f"    H{h}   {mean_delta[li, h]:>+11.4f}  {med:>+13.4f}")

    # --- Comparison ---
    print(f"\n{'=' * 50}")
    print("Comparison: most damaging head per layer")
    print(f"{'=' * 50}")
    for li, layer_idx in enumerate(TARGET_LAYERS):
        worst_head = int(np.argmin(mean_delta[li]))
        print(f"  L{layer_idx}: H{worst_head} (mean Δlogp = {mean_delta[li, worst_head]:+.4f})")

    # Check if L29 H7 specifically stands out
    l29_idx = TARGET_LAYERS.index(29)
    h7_delta = mean_delta[l29_idx, 7]
    h7_rank = int(np.sum(mean_delta[l29_idx] < h7_delta))  # 0 = most damaging
    print(f"\n  L29 H7 specifically: mean Δlogp = {h7_delta:+.4f}, "
          f"rank {h7_rank + 1}/8 at L29")

    # Per-prompt detail for the most interesting heads
    print(f"\n--- Per-prompt detail for L29 H7 ---")
    for j, (_, target_id, _, target_tok, prompt) in enumerate(valid):
        delta = results[l29_idx, 7, j]
        # Also check what the top-1 becomes
        logits = run_head_ablated(model, valid[j][0], 29, 7)
        last = logits[0, -1, :].astype(mx.float32)
        probs = mx.softmax(last)
        mx.eval(probs)
        probs_np = np.array(probs)
        new_top1 = tokenizer.decode([int(np.argmax(probs_np))])
        still = "YES" if int(np.argmax(probs_np)) == target_id else "no"
        print(f"  {prompt[:50]:50s}  Δ={delta:>+7.3f}  still_top1={still:>3s}  "
              f"(now: {new_top1!r})")

    # --- Plot ---
    fig, axes = plt.subplots(1, len(TARGET_LAYERS), figsize=(12, 5), sharey=True)

    for li, layer_idx in enumerate(TARGET_LAYERS):
        ax = axes[li]
        heads = np.arange(n_heads)
        colors = ["#e74c3c" if (layer_idx == 29 and h == 7) else "#3498db"
                  for h in range(n_heads)]
        ax.bar(heads, mean_delta[li], color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xlabel("head index")
        ax.set_title(f"Layer {layer_idx}")
        ax.set_xticks(heads)
        ax.set_xticklabels([f"H{h}" for h in range(n_heads)])
        ax.axhline(0, color="black", linewidth=0.5)
        ax.grid(True, alpha=0.3, axis="y")

    axes[0].set_ylabel("mean Δ log p(target)")
    fig.suptitle("Single-head ablation impact — L23 vs L29\n(red = L29 H7, the content-attention head)",
                 fontsize=11)

    plt.tight_layout()
    out_path = OUT_DIR / "single_head_ablation.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
