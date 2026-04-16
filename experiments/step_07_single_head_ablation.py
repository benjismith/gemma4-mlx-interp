"""Single-head ablation at layers 23 and 29 in Gemma 4 E4B.

For each of the 8 attention heads at L23 (the structural-bridge layer) and
L29 (the highest-subject-attention layer per finding 06), zeroes out that
head's contribution and measures the impact on factual recall.

Key finding (per docs/findings/step_07_single_head_ablation.md): L29 H7,
the head with the highest subject-entity attention, has near-zero causal
impact when ablated. No single head is a bottleneck — heads function as a
redundant ensemble.

Run from project root:
    python experiments/step_07_single_head_ablation.py
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gemma4_mlx_interp import Ablate, FACTUAL_15, Model  # noqa: E402

OUT_DIR = ROOT / "caches"
TARGET_LAYERS = [23, 29]
N_HEADS = 8


def _last_logp(logits: mx.array) -> np.ndarray:
    last = logits[0, -1, :].astype(mx.float32)
    lp = last - mx.logsumexp(last)
    mx.eval(lp)
    return np.array(lp)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading model...")
    model = Model.load()

    print(f"\nValidating prompts...\n")
    valid = FACTUAL_15.validate(model)
    print()
    if len(valid) == 0:
        print("No prompts validated; aborting.")
        return

    # results[layer_idx, head, prompt] = Δ log p(target)
    results = np.zeros((len(TARGET_LAYERS), N_HEADS, len(valid)), dtype=np.float64)

    total = len(TARGET_LAYERS) * N_HEADS * len(valid)
    print(f"Running {total} ablated forward passes...")
    t0 = time.perf_counter()

    for li, layer_idx in enumerate(TARGET_LAYERS):
        for h in range(N_HEADS):
            ablation = Ablate.head(layer_idx, head=h)
            for j, vp in enumerate(valid):
                result = model.run(vp.input_ids, interventions=[ablation])
                lp_np = _last_logp(result.logits)
                results[li, h, j] = float(lp_np[vp.target_id]) - vp.baseline_lp

        elapsed = time.perf_counter() - t0
        print(f"  L{layer_idx} done ({elapsed:.0f}s)")

    total_time = time.perf_counter() - t0
    print(f"\nDone in {total_time:.0f}s")

    mean_delta = np.mean(results, axis=2)  # [n_target_layers, n_heads]

    # ---- Per-layer head leaderboard ----
    for li, layer_idx in enumerate(TARGET_LAYERS):
        print(f"\n--- Layer {layer_idx} ---")
        print(f"  {'head':>4}  {'mean_Δlogp':>11}  {'median_Δlogp':>13}")
        print(f"  {'-' * 32}")
        for h in range(N_HEADS):
            med = float(np.median(results[li, h]))
            print(f"    H{h}   {mean_delta[li, h]:>+11.4f}  {med:>+13.4f}")

    # ---- Comparison ----
    print(f"\n{'=' * 50}")
    print("Comparison: most damaging head per layer")
    print(f"{'=' * 50}")
    for li, layer_idx in enumerate(TARGET_LAYERS):
        worst_head = int(np.argmin(mean_delta[li]))
        print(f"  L{layer_idx}: H{worst_head} (mean Δlogp = {mean_delta[li, worst_head]:+.4f})")

    # ---- Specifically report L29 H7 (the candidate 'content head') ----
    l29_idx = TARGET_LAYERS.index(29)
    h7_delta = mean_delta[l29_idx, 7]
    h7_rank = int(np.sum(mean_delta[l29_idx] < h7_delta))  # 0 = most damaging
    print(f"\n  L29 H7 specifically: mean Δlogp = {h7_delta:+.4f}, "
          f"rank {h7_rank + 1}/{N_HEADS} at L29")

    # Per-prompt detail for L29 H7 — does the model still produce the right answer?
    print(f"\n--- Per-prompt detail for L29 H7 ---")
    h7_ablation = Ablate.head(29, head=7)
    for j, vp in enumerate(valid):
        delta = results[l29_idx, 7, j]
        result = model.run(vp.input_ids, interventions=[h7_ablation])
        last = result.last_logits.astype(mx.float32)
        probs = mx.softmax(last)
        mx.eval(probs)
        new_top1_id = int(np.argmax(np.array(probs)))
        new_top1 = model.tokenizer.decode([new_top1_id])
        still = "YES" if new_top1_id == vp.target_id else "no"
        print(f"  {vp.prompt.text[:50]:50s}  Δ={delta:>+7.3f}  still_top1={still:>3s}  "
              f"(now: {new_top1!r})")

    # ---- Plot ----
    fig, axes = plt.subplots(1, len(TARGET_LAYERS), figsize=(12, 5), sharey=True)
    for li, layer_idx in enumerate(TARGET_LAYERS):
        ax = axes[li]
        heads = np.arange(N_HEADS)
        colors = ["#e74c3c" if (layer_idx == 29 and h == 7) else "#3498db"
                  for h in range(N_HEADS)]
        ax.bar(heads, mean_delta[li], color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xlabel("head index")
        ax.set_title(f"Layer {layer_idx}")
        ax.set_xticks(heads)
        ax.set_xticklabels([f"H{h}" for h in range(N_HEADS)])
        ax.axhline(0, color="black", linewidth=0.5)
        ax.grid(True, alpha=0.3, axis="y")
    axes[0].set_ylabel("mean Δ log p(target)")
    fig.suptitle("Single-head ablation impact — L23 vs L29\n"
                 "(red = L29 H7, the candidate content-attention head)",
                 fontsize=11)

    plt.tight_layout()
    out_path = OUT_DIR / "single_head_ablation.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
