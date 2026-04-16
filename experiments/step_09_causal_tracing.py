"""Causal tracing (activation patching) on Gemma 4 E4B.

For each paired prompt:
  1. Run CLEAN prompt, cache residual stream at every (layer, position).
  2. Run CORRUPT prompt, but at one chosen (layer, position), swap in the
     clean residual and let the rest of the forward pass proceed.
  3. Measure recovery of the CLEAN answer's probability.

Positions × layers where patching restores the clean answer reveal the
causal path for factual information, even when that information isn't
decodable through the unembed (per finding 08).

Run from project root:
    python experiments/step_09_causal_tracing.py
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

PAIRS = [
    ("Complete this sentence with one word: The Eiffel Tower is in",
     "Complete this sentence with one word: The Great Wall is in"),
    ("Complete this sentence with one word: The capital of Japan is",
     "Complete this sentence with one word: The capital of France is"),
    ("Complete this sentence with one word: Romeo and Juliet was written by",
     "Complete this sentence with one word: Pride and Prejudice was written by"),
]


def forward_with_patch(
    model,
    input_ids: mx.array,
    clean_resid_post: dict = None,
    patch_layer: int = -1,
    patch_position: int = -1,
) -> mx.array:
    """Forward pass where at (patch_layer, patch_position), after the layer
    completes, we overwrite the resid_post at that single position with the
    clean value before continuing to the next layer.

    If clean_resid_post is None or patch_layer=-1, runs unpatched.
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

        # Standard layer computation
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

        # PATCH: overwrite resid_post at (patch_layer, patch_position)
        # with the clean value before continuing
        if i == patch_layer and clean_resid_post is not None:
            clean_val = clean_resid_post[patch_layer]  # [1, seq_len, d_model]
            # Build a mask: 1 at patch_position, 0 elsewhere
            seq_len = h.shape[1]
            mask = mx.zeros((1, seq_len, 1), dtype=h.dtype)
            mask = mask.at[:, patch_position, :].add(1.0)
            h = h * (1 - mask) + clean_val * mask

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

    for pair_idx, (clean_prompt, corrupt_prompt) in enumerate(PAIRS):
        print(f"\n{'=' * 70}")
        print(f"Pair {pair_idx}: CLEAN   = {clean_prompt!r}")
        print(f"            CORRUPT = {corrupt_prompt!r}")

        clean_ids = _tokenize(processor, model, clean_prompt)
        corrupt_ids = _tokenize(processor, model, corrupt_prompt)
        seq_len = clean_ids.shape[1]
        assert clean_ids.shape[1] == corrupt_ids.shape[1], "Prompt lengths must match"

        token_labels_clean = [tokenizer.decode([t]) for t in clean_ids[0].tolist()]
        token_labels_corrupt = [tokenizer.decode([t]) for t in corrupt_ids[0].tolist()]

        # Identify the diff positions
        diff_pos = [i for i in range(seq_len)
                    if clean_ids[0, i].item() != corrupt_ids[0, i].item()]
        print(f"Diff positions: {diff_pos} "
              f"({[token_labels_clean[p] for p in diff_pos]} vs "
              f"{[token_labels_corrupt[p] for p in diff_pos]})")

        # Get clean and corrupt predictions
        clean_logits, clean_cache = run_with_cache(model, clean_ids)
        corrupt_logits, _ = run_with_cache(model, corrupt_ids)

        clean_last = clean_logits[0, -1, :].astype(mx.float32)
        corrupt_last = corrupt_logits[0, -1, :].astype(mx.float32)
        clean_probs = mx.softmax(clean_last)
        corrupt_probs = mx.softmax(corrupt_last)
        mx.eval(clean_probs, corrupt_probs)

        clean_top1_id = int(np.argmax(np.array(clean_probs)))
        corrupt_top1_id = int(np.argmax(np.array(corrupt_probs)))
        clean_answer = tokenizer.decode([clean_top1_id])
        corrupt_answer = tokenizer.decode([corrupt_top1_id])

        print(f"Clean answer:   {clean_answer!r} (p={float(clean_probs[clean_top1_id]):.3f})")
        print(f"Corrupt answer: {corrupt_answer!r} (p={float(corrupt_probs[corrupt_top1_id]):.3f})")

        # Baseline corrupt-run probability of the CLEAN answer
        baseline_clean_prob_in_corrupt = float(corrupt_probs[clean_top1_id])
        baseline_clean_prob = float(clean_probs[clean_top1_id])
        print(f"p(clean answer) in clean run:   {baseline_clean_prob:.4f}")
        print(f"p(clean answer) in corrupt run: {baseline_clean_prob_in_corrupt:.4f}")

        # Extract clean resid_post dict keyed by layer index
        clean_resid_by_layer = {
            i: clean_cache[f"blocks.{i}.resid_post"] for i in range(N_LAYERS)
        }

        # For each (layer, position), patch and measure p(clean_answer)
        patch_results = np.zeros((N_LAYERS, seq_len), dtype=np.float64)
        print(f"\nRunning {N_LAYERS * seq_len} patched forward passes...")
        t0 = time.perf_counter()

        for L in range(N_LAYERS):
            for P in range(seq_len):
                logits = forward_with_patch(
                    model, corrupt_ids,
                    clean_resid_post=clean_resid_by_layer,
                    patch_layer=L, patch_position=P,
                )
                last = logits[0, -1, :].astype(mx.float32)
                probs = mx.softmax(last)
                mx.eval(probs)
                probs_np = np.array(probs)
                patch_results[L, P] = float(probs_np[clean_top1_id])

            if (L + 1) % 7 == 0 or L == N_LAYERS - 1:
                elapsed = time.perf_counter() - t0
                eta = elapsed / (L + 1) * (N_LAYERS - L - 1)
                print(f"  layer {L:>2} done  [{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]")

        total_time = time.perf_counter() - t0
        print(f"Done in {total_time:.0f}s")

        # Recovery: p(clean) under patch MINUS p(clean) under unpatched corrupt
        recovery = patch_results - baseline_clean_prob_in_corrupt

        # Print top patches
        flat = recovery.flatten()
        top_indices = np.argsort(-flat)[:10]
        print(f"\nTop-10 patches that most recover p({clean_answer!r}):")
        print(f"  {'layer':>5}  {'pos':>4}  {'token':>12}  {'recovery':>9}  {'p(clean)':>9}")
        for idx in top_indices:
            L, P = idx // seq_len, idx % seq_len
            print(f"  {L:>5}  {P:>4}  {token_labels_corrupt[P]:>12s}  "
                  f"{recovery[L, P]:>+8.3f}   {patch_results[L, P]:>7.3f}")

        # Heatmap
        fig, axes = plt.subplots(1, 2, figsize=(max(12, seq_len * 0.6), 8))

        ax = axes[0]
        im = ax.imshow(patch_results, aspect="auto", cmap="Greens",
                       vmin=0, vmax=max(0.01, np.max(patch_results)),
                       origin="lower", interpolation="nearest")
        ax.set_xlabel("patched position")
        ax.set_ylabel("patched layer")
        ax.set_title(f"p({clean_answer!r}) after patching (corrupt → clean)\n"
                     f"baseline: {baseline_clean_prob_in_corrupt:.3f} in corrupt, "
                     f"{baseline_clean_prob:.3f} in clean")
        ax.set_xticks(range(seq_len))
        ax.set_xticklabels(token_labels_corrupt, rotation=70, ha="right", fontsize=6)
        plt.colorbar(im, ax=ax, shrink=0.6)

        for p in diff_pos:
            ax.axvline(p, color="red", linewidth=1.5, alpha=0.7, linestyle="--")
        for g in GLOBAL_LAYERS:
            ax.axhline(g, color="gray", linewidth=0.5, alpha=0.5, linestyle=":")

        ax = axes[1]
        im = ax.imshow(recovery, aspect="auto", cmap="RdBu_r",
                       vmin=-np.max(np.abs(recovery)) * 0.5,
                       vmax=np.max(np.abs(recovery)) * 0.5,
                       origin="lower", interpolation="nearest")
        ax.set_xlabel("patched position")
        ax.set_title(f"recovery: Δ p({clean_answer!r}) vs unpatched corrupt run")
        ax.set_xticks(range(seq_len))
        ax.set_xticklabels(token_labels_corrupt, rotation=70, ha="right", fontsize=6)
        plt.colorbar(im, ax=ax, shrink=0.6)

        for p in diff_pos:
            ax.axvline(p, color="red", linewidth=1.5, alpha=0.7, linestyle="--")
        for g in GLOBAL_LAYERS:
            ax.axhline(g, color="gray", linewidth=0.5, alpha=0.5, linestyle=":")

        fig.suptitle(f"Causal tracing: clean = {clean_prompt[:40]}..  corrupt = {corrupt_prompt[:40]}..",
                     fontsize=10)
        plt.tight_layout()
        out_path = OUT_DIR / f"causal_trace_{pair_idx}.png"
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
