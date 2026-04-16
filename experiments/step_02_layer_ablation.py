"""Per-layer zero-ablation experiment on Gemma 4 E4B.

For each of the 42 layers, skips that layer's contribution to the residual
stream and measures the resulting loss (negative log-prob of the model's own
top-1 prediction). Layers 0-23 own KV cache entries that may be shared with
later layers, so we still run their attention to populate the cache — we
just don't add the output to the residual stream.

Runs on the same 15-prompt battery as the logit-lens batch experiment.
42 layers × 15 prompts = 630 forward passes (~2 minutes on M2 Pro).

Run from project root:
    python experiments/step_02_layer_ablation.py
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


def run_ablated_forward(model, input_ids: mx.array, ablate_layer: int) -> mx.array:
    """Forward pass with one layer's residual contribution zeroed out.

    Returns logits [1, seq_len, vocab_size].

    For layers 0–23 (which own KV cache entries), we still run attention to
    populate the cache — downstream KV-shared layers (24+) depend on it.
    For layers 24–41 (KV-shared, read-only), we skip entirely.
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

        if i == ablate_layer:
            # Populate KV cache for non-shared layers (0-23) so downstream
            # KV-shared layers still work. Shared layers (24+) only read
            # from the cache and don't need to run attention.
            if not layer.self_attn.is_kv_shared_layer:
                a = layer.input_layernorm(h)
                _ = layer.self_attn(a, local_mask, c)
            # Skip residual contribution: h stays unchanged.
            continue

        # --- Normal layer execution (mirrors hooks.py) ---
        resid_pre = h
        a = layer.input_layernorm(h)
        a = layer.self_attn(a, local_mask, c)
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

    # Collect prompts with confident predictions.
    print(f"\nValidating {len(PROMPTS)} prompts...\n")
    valid = []  # (input_ids, target_id, baseline_logprob)

    for prompt in PROMPTS:
        input_ids = _tokenize(processor, model, prompt)
        logits, _ = run_with_cache(model, input_ids)

        last = logits[0, -1, :].astype(mx.float32)
        lp = last - mx.logsumexp(last)
        probs = mx.softmax(last)
        mx.eval(lp, probs)
        probs_np = np.array(probs)
        lp_np = np.array(lp)

        top1_id = int(np.argmax(probs_np))
        top1_prob = float(probs_np[top1_id])
        top1_lp = float(lp_np[top1_id])
        top1_tok = tokenizer.decode([top1_id])

        status = "OK" if top1_prob >= MIN_CONFIDENCE else "SKIP"
        print(f"  [{status}] {prompt[:55]:55s}  top1={top1_tok!r:15s} p={top1_prob:.3f}")

        if top1_prob >= MIN_CONFIDENCE:
            valid.append((input_ids, top1_id, top1_lp))

    print(f"\n{len(valid)} / {len(PROMPTS)} prompts validated.\n")

    # For each layer, ablate and measure loss delta across all prompts.
    # loss_delta[i, j] = ablated_logprob[i,j] - baseline_logprob[j]
    # (negative = layer was helpful; more negative = more important)
    loss_delta = np.zeros((N_LAYERS, len(valid)), dtype=np.float64)

    print(f"Running {N_LAYERS} × {len(valid)} = {N_LAYERS * len(valid)} ablated forward passes...")
    t0 = time.perf_counter()

    for i in range(N_LAYERS):
        for j, (input_ids, target_id, baseline_lp) in enumerate(valid):
            logits = run_ablated_forward(model, input_ids, ablate_layer=i)
            last = logits[0, -1, :].astype(mx.float32)
            lp = last - mx.logsumexp(last)
            mx.eval(lp)
            lp_np = np.array(lp)
            ablated_lp = float(lp_np[target_id])
            loss_delta[i, j] = ablated_lp - baseline_lp

        elapsed = time.perf_counter() - t0
        eta = elapsed / (i + 1) * (N_LAYERS - i - 1)
        if (i + 1) % 6 == 0 or i == N_LAYERS - 1:
            mean_delta = float(np.mean(loss_delta[i]))
            print(f"  layer {i:>2}: mean Δlogp = {mean_delta:>+8.3f}  "
                  f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]")

    total_time = time.perf_counter() - t0
    print(f"\nDone in {total_time:.0f}s ({total_time / (N_LAYERS * len(valid)) * 1000:.0f}ms per pass)")

    # Aggregate: mean loss delta per layer.
    mean_delta = np.mean(loss_delta, axis=1)

    # --- Table ---
    print(f"\n{'layer':>5}  {'type':>7}  {'mean_Δlogp':>11}  {'median_Δlogp':>13}")
    print("-" * 45)
    for i in range(N_LAYERS):
        kind = "GLOBAL" if i in GLOBAL_LAYERS else "local"
        med = float(np.median(loss_delta[i]))
        print(f"{i:>5}  {kind:>7}  {mean_delta[i]:>+11.3f}  {med:>+13.3f}")

    # Summary stats: global vs local
    global_deltas = mean_delta[GLOBAL_LAYERS]
    local_mask = np.ones(N_LAYERS, dtype=bool)
    local_mask[GLOBAL_LAYERS] = False
    local_deltas = mean_delta[local_mask]

    print(f"\n--- Summary ---")
    print(f"  Global layers (n=7):  mean Δlogp = {np.mean(global_deltas):>+.3f}, "
          f"median = {np.median(global_deltas):>+.3f}")
    print(f"  Local layers  (n=35): mean Δlogp = {np.mean(local_deltas):>+.3f}, "
          f"median = {np.median(local_deltas):>+.3f}")

    # Top 5 most damaging ablations
    most_damaging = np.argsort(mean_delta)[:5]
    print(f"\n  5 most damaging layers to ablate:")
    for idx in most_damaging:
        kind = "GLOBAL" if idx in GLOBAL_LAYERS else "local"
        print(f"    layer {idx:>2} ({kind:>6}): mean Δlogp = {mean_delta[idx]:>+.3f}")

    n_global_in_top5 = sum(1 for idx in most_damaging if idx in GLOBAL_LAYERS)
    print(f"\n  {n_global_in_top5} / 5 most damaging are global-attention layers")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#d62728" if i in GLOBAL_LAYERS else "#1f77b4" for i in range(N_LAYERS)]
    bars = ax.bar(range(N_LAYERS), mean_delta, color=colors, edgecolor="white", linewidth=0.3)

    ax.set_xlabel("layer index")
    ax.set_ylabel("mean Δ log p(target)  ← more damaging")
    ax.set_title("Layer ablation impact — Gemma 4 E4B (15 prompts)")
    ax.set_xticks(range(0, N_LAYERS, 3))
    ax.grid(True, alpha=0.3, axis="y")

    # Legend
    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(color="#d62728", label="global attention"),
                 Patch(color="#1f77b4", label="local (sliding window)")],
        loc="lower left",
    )

    plt.tight_layout()
    out_path = OUT_DIR / "layer_ablation.png"
    fig.savefig(out_path, dpi=140)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
