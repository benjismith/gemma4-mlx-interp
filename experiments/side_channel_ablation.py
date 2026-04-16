"""Ablate the MatFormer per-layer-input side-channel on Gemma 4 E4B.

Zeros out the per_layer_input_gate contribution in every layer and measures
how much the model's predictions degrade. If degradation is small, the
side-channel is closer to vestigial; if large, it's load-bearing and worth
investigating further.

Also tests per-layer granularity: ablate the side-channel in one layer at
a time to see if any specific layers depend on it more than others.

Run from project root:
    python experiments/side_channel_ablation.py
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


def run_side_channel_ablated(
    model, input_ids: mx.array, ablate_layers: set = None
) -> mx.array:
    """Forward pass with per-layer-input gate zeroed out.

    If ablate_layers is None, ablates ALL layers. Otherwise, only the
    specified layer indices.
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

        should_ablate_gate = (ablate_layers is None) or (i in ablate_layers)

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

        # Per-layer gate: run normally or skip entirely
        per_layer_input = (
            per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
        )
        if (
            not should_ablate_gate
            and layer.per_layer_input_gate is not None
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
            valid.append((input_ids, top1_id, top1_lp, top1_tok, prompt))

    print(f"\n{len(valid)} / {len(PROMPTS)} prompts validated.\n")

    # --- Test 1: Ablate ALL layers' side-channel ---
    print("=" * 60)
    print("Test 1: Ablate side-channel in ALL layers")
    print("=" * 60)

    full_ablation_deltas = []
    for input_ids, target_id, baseline_lp, target_tok, prompt in valid:
        logits = run_side_channel_ablated(model, input_ids, ablate_layers=None)
        last = logits[0, -1, :].astype(mx.float32)
        lp = last - mx.logsumexp(last)
        mx.eval(lp)
        lp_np = np.array(lp)
        ablated_lp = float(lp_np[target_id])
        delta = ablated_lp - baseline_lp

        top1_id_abl = int(np.argmax(lp_np))
        top1_tok_abl = tokenizer.decode([top1_id_abl])
        same = "YES" if top1_id_abl == target_id else "NO"
        print(f"  {prompt[:50]:50s}  target={target_tok!r:12s}  "
              f"Δlogp={delta:>+7.3f}  still_top1={same}  "
              f"(now: {top1_tok_abl!r})")
        full_ablation_deltas.append(delta)

    mean_full = np.mean(full_ablation_deltas)
    n_still_top1 = sum(1 for d in full_ablation_deltas if d > -0.5)
    print(f"\n  Mean Δlogp (all-layer ablation): {mean_full:+.3f}")
    print(f"  Prompts where target remains top-1: {n_still_top1} / {len(valid)}")

    # --- Test 2: Ablate one layer's side-channel at a time ---
    print(f"\n{'=' * 60}")
    print("Test 2: Ablate side-channel in ONE layer at a time")
    print("=" * 60)

    per_layer_delta = np.zeros((N_LAYERS, len(valid)), dtype=np.float64)

    t0 = time.perf_counter()
    for i in range(N_LAYERS):
        for j, (input_ids, target_id, baseline_lp, _, _) in enumerate(valid):
            logits = run_side_channel_ablated(model, input_ids, ablate_layers={i})
            last = logits[0, -1, :].astype(mx.float32)
            lp = last - mx.logsumexp(last)
            mx.eval(lp)
            lp_np = np.array(lp)
            per_layer_delta[i, j] = float(lp_np[target_id]) - baseline_lp

        if (i + 1) % 7 == 0:
            elapsed = time.perf_counter() - t0
            eta = elapsed / (i + 1) * (N_LAYERS - i - 1)
            print(f"  layer {i:>2} done  [{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]")

    total_time = time.perf_counter() - t0
    print(f"\nDone in {total_time:.0f}s")

    mean_per_layer = np.mean(per_layer_delta, axis=1)

    print(f"\n{'layer':>5}  {'type':>7}  {'mean_Δlogp':>11}")
    print("-" * 30)
    for i in range(N_LAYERS):
        kind = "GLOBAL" if i in GLOBAL_LAYERS else "local"
        if abs(mean_per_layer[i]) > 0.01:
            print(f"{i:>5}  {kind:>7}  {mean_per_layer[i]:>+11.4f}")

    most_affected = np.argsort(mean_per_layer)[:5]
    print(f"\n  5 layers most affected by single-layer gate ablation:")
    for idx in most_affected:
        kind = "GLOBAL" if idx in GLOBAL_LAYERS else "local"
        print(f"    layer {idx:>2} ({kind:>6}): mean Δlogp = {mean_per_layer[idx]:>+.4f}")

    # --- Plot ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))

    # Top: full ablation comparison
    ax = axes[0]
    ax.bar(range(len(valid)), full_ablation_deltas, color="#d62728")
    ax.set_ylabel("Δ log p(target)")
    ax.set_title("Side-channel ablation (ALL layers zeroed) — per prompt")
    ax.set_xticks(range(len(valid)))
    ax.set_xticklabels([v[3] for v in valid], rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    # Bottom: per-layer single ablation
    ax = axes[1]
    colors = ["#d62728" if i in GLOBAL_LAYERS else "#1f77b4" for i in range(N_LAYERS)]
    ax.bar(range(N_LAYERS), mean_per_layer, color=colors, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("layer index")
    ax.set_ylabel("mean Δ log p(target)")
    ax.set_title("Side-channel ablation (ONE layer at a time)")
    ax.set_xticks(range(0, N_LAYERS, 3))
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(color="#d62728", label="global attention"),
                 Patch(color="#1f77b4", label="local (sliding window)")],
        loc="lower left",
    )

    plt.tight_layout()
    out_path = OUT_DIR / "side_channel_ablation.png"
    fig.savefig(out_path, dpi=140)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
