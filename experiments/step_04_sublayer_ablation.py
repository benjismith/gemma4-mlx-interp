"""Sub-layer ablation: attention vs MLP across all 42 layers of Gemma 4 E4B.

For each layer, independently ablates:
  (a) the attention branch (zero out attn contribution, keep MLP + gate)
  (b) the MLP branch (zero out MLP contribution, keep attention + gate)

Still populates KV caches when ablating attention on non-shared layers,
so downstream KV-shared layers aren't starved.

Run from project root:
    python experiments/step_04_sublayer_ablation.py
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


def run_sublayer_ablated(
    model,
    input_ids: mx.array,
    ablate_attn_layer: int = -1,
    ablate_mlp_layer: int = -1,
) -> mx.array:
    """Forward pass with one layer's attention or MLP branch zeroed.

    ablate_attn_layer: skip attention contribution for this layer
    ablate_mlp_layer: skip MLP contribution for this layer
    Set to -1 (default) to leave that branch intact.
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

        # --- Attention branch ---
        resid_pre = h
        if i == ablate_attn_layer:
            # Populate KV cache for non-shared layers, but don't add to residual
            if not layer.self_attn.is_kv_shared_layer:
                a = layer.input_layernorm(h)
                _ = layer.self_attn(a, local_mask, c)
            # h stays as resid_pre
        else:
            a = layer.input_layernorm(h)
            a = layer.self_attn(a, local_mask, c)
            a = layer.post_attention_layernorm(a)
            h = resid_pre + a

        # --- MLP branch ---
        mid = h
        if i == ablate_mlp_layer:
            pass  # skip MLP entirely
        else:
            m = layer.pre_feedforward_layernorm(mid)
            m = layer.mlp(m)
            m = layer.post_feedforward_layernorm(m)
            h = mid + m

        # --- Per-layer gate (always runs) ---
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

        status = "OK" if top1_prob >= MIN_CONFIDENCE else "SKIP"
        print(f"  [{status}] {prompt[:55]:55s}  top1={top1_tok!r:15s} p={top1_prob:.3f}")

        if top1_prob >= MIN_CONFIDENCE:
            valid.append((input_ids, top1_id, top1_lp))

    print(f"\n{len(valid)} / {len(PROMPTS)} prompts validated.\n")

    # Run ablations: attention and MLP separately for all 42 layers.
    attn_delta = np.zeros((N_LAYERS, len(valid)), dtype=np.float64)
    mlp_delta = np.zeros((N_LAYERS, len(valid)), dtype=np.float64)

    total_passes = N_LAYERS * len(valid) * 2
    print(f"Running {total_passes} ablated forward passes...")
    t0 = time.perf_counter()

    for i in range(N_LAYERS):
        for j, (input_ids, target_id, baseline_lp) in enumerate(valid):
            # Attention ablation
            logits = run_sublayer_ablated(model, input_ids, ablate_attn_layer=i)
            last = logits[0, -1, :].astype(mx.float32)
            lp = last - mx.logsumexp(last)
            mx.eval(lp)
            lp_np = np.array(lp)
            attn_delta[i, j] = float(lp_np[target_id]) - baseline_lp

            # MLP ablation
            logits = run_sublayer_ablated(model, input_ids, ablate_mlp_layer=i)
            last = logits[0, -1, :].astype(mx.float32)
            lp = last - mx.logsumexp(last)
            mx.eval(lp)
            lp_np = np.array(lp)
            mlp_delta[i, j] = float(lp_np[target_id]) - baseline_lp

        elapsed = time.perf_counter() - t0
        eta = elapsed / (i + 1) * (N_LAYERS - i - 1)
        if (i + 1) % 6 == 0 or i == N_LAYERS - 1:
            print(f"  layer {i:>2}: attn Δ={np.mean(attn_delta[i]):>+7.3f}  "
                  f"mlp Δ={np.mean(mlp_delta[i]):>+7.3f}  "
                  f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]")

    total_time = time.perf_counter() - t0
    print(f"\nDone in {total_time:.0f}s ({total_time / total_passes * 1000:.0f}ms per pass)")

    mean_attn = np.mean(attn_delta, axis=1)
    mean_mlp = np.mean(mlp_delta, axis=1)

    # --- Table ---
    print(f"\n{'layer':>5}  {'type':>7}  {'attn_Δlogp':>11}  {'mlp_Δlogp':>10}  {'dominant':>9}")
    print("-" * 55)
    for i in range(N_LAYERS):
        kind = "GLOBAL" if i in GLOBAL_LAYERS else "local"
        dominant = "attn" if abs(mean_attn[i]) > abs(mean_mlp[i]) else "MLP"
        if abs(mean_attn[i]) < 0.01 and abs(mean_mlp[i]) < 0.01:
            dominant = "-"
        print(f"{i:>5}  {kind:>7}  {mean_attn[i]:>+11.3f}  {mean_mlp[i]:>+10.3f}  {dominant:>9}")

    # Summary: global vs local, attn vs MLP
    print(f"\n--- Summary ---")
    for layer_type, indices in [("Global", GLOBAL_LAYERS),
                                 ("Local", [i for i in range(N_LAYERS) if i not in GLOBAL_LAYERS])]:
        a = mean_attn[indices]
        m = mean_mlp[indices]
        print(f"  {layer_type:6s} (n={len(indices):>2}): "
              f"attn mean Δ={np.mean(a):>+.3f}, mlp mean Δ={np.mean(m):>+.3f}")

    # Critical middle (10-24)
    mid = list(range(10, 25))
    print(f"  Middle (10-24, n={len(mid)}): "
          f"attn mean Δ={np.mean(mean_attn[mid]):>+.3f}, "
          f"mlp mean Δ={np.mean(mean_mlp[mid]):>+.3f}")

    # --- Plot ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    layers_x = np.arange(N_LAYERS)
    width = 0.35

    # Panel 1: side-by-side bars
    ax = axes[0]
    ax.bar(layers_x - width / 2, mean_attn, width, label="attention ablated",
           color="#e74c3c", edgecolor="white", linewidth=0.3)
    ax.bar(layers_x + width / 2, mean_mlp, width, label="MLP ablated",
           color="#3498db", edgecolor="white", linewidth=0.3)
    ax.set_ylabel("mean Δ log p(target)")
    ax.set_title("Sub-layer ablation — Gemma 4 E4B (15 prompts)")
    ax.legend(loc="lower left")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    for g in GLOBAL_LAYERS:
        ax.axvline(g, color="#999999", linestyle="--", linewidth=0.7, alpha=0.4)

    # Panel 2: difference (attn - MLP), showing which branch dominates
    diff = mean_attn - mean_mlp
    colors_diff = ["#e74c3c" if d < 0 else "#3498db" for d in diff]
    ax = axes[1]
    ax.bar(layers_x, diff, color=colors_diff, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("layer index")
    ax.set_ylabel("attn Δ − MLP Δ\n← attn more critical | MLP more critical →")
    ax.set_title("Which branch matters more per layer? (negative = attention dominates)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(range(0, N_LAYERS, 3))
    ax.grid(True, alpha=0.3, axis="y")
    for g in GLOBAL_LAYERS:
        ax.axvline(g, color="#999999", linestyle="--", linewidth=0.7, alpha=0.4)

    plt.tight_layout()
    out_path = OUT_DIR / "sublayer_ablation.png"
    fig.savefig(out_path, dpi=140)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
