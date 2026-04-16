"""Ablate the MatFormer per-layer-input side-channel on Gemma 4 E4B.

Two tests:
  1. Full ablation: zero the side-channel gate at ALL 42 layers; measure
     Δ log p(top-1) per prompt. Catastrophic per finding 03.
  2. Per-layer ablation: zero only one layer's gate at a time; identify
     which specific layers depend most on the side-channel.

Run from project root:
    python experiments/step_03_side_channel_ablation.py
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

from gemma4_mlx_interp import (  # noqa: E402
    Ablate, FACTUAL_15, GLOBAL_LAYERS, Model, N_LAYERS, bar_by_layer,
)

OUT_DIR = ROOT / "caches"


def _last_logp_and_top1(model, logits: mx.array) -> tuple[np.ndarray, int, str]:
    last = logits[0, -1, :].astype(mx.float32)
    lp = last - mx.logsumexp(last)
    probs = mx.softmax(last)
    mx.eval(lp, probs)
    lp_np = np.array(lp)
    top1_id = int(np.argmax(np.array(probs)))
    top1_tok = model.tokenizer.decode([top1_id])
    return lp_np, top1_id, top1_tok


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading model...")
    model = Model.load()

    print(f"\nValidating prompts...\n")
    valid = FACTUAL_15.validate(model)
    print()

    # ---- Test 1: Full ablation across all layers ----
    print("=" * 60)
    print("Test 1: Ablate side-channel in ALL layers")
    print("=" * 60)
    full = Ablate.side_channel()  # default: all layers

    full_deltas = []
    for vp in valid:
        result = model.run(vp.input_ids, interventions=[full])
        lp_np, top1_id, top1_tok = _last_logp_and_top1(model, result.logits)
        ablated_lp = float(lp_np[vp.target_id])
        delta = ablated_lp - vp.baseline_lp
        same = "YES" if top1_id == vp.target_id else "NO"
        print(f"  {vp.prompt.text[:50]:50s}  target={vp.target_token!r:12s}  "
              f"Δlogp={delta:>+7.3f}  still_top1={same}  (now: {top1_tok!r})")
        full_deltas.append(delta)

    mean_full = float(np.mean(full_deltas))
    n_still = sum(1 for d in full_deltas if d > -0.5)
    print(f"\n  Mean Δlogp (all-layer ablation): {mean_full:+.3f}")
    print(f"  Prompts where target remains top-1: {n_still} / {len(valid)}")

    # ---- Test 2: One layer at a time ----
    print(f"\n{'=' * 60}")
    print("Test 2: Ablate side-channel in ONE layer at a time")
    print("=" * 60)

    per_layer_delta = np.zeros((N_LAYERS, len(valid)), dtype=np.float64)
    t0 = time.perf_counter()
    for i in range(N_LAYERS):
        ablation = Ablate.side_channel(layers=[i])
        for j, vp in enumerate(valid):
            result = model.run(vp.input_ids, interventions=[ablation])
            lp_np, _, _ = _last_logp_and_top1(model, result.logits)
            per_layer_delta[i, j] = float(lp_np[vp.target_id]) - vp.baseline_lp

        if (i + 1) % 7 == 0:
            elapsed = time.perf_counter() - t0
            eta = elapsed / (i + 1) * (N_LAYERS - i - 1)
            print(f"  layer {i:>2} done  [{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]")

    print(f"\nDone in {time.perf_counter() - t0:.0f}s")

    mean_per_layer = np.mean(per_layer_delta, axis=1)

    # Print only layers with meaningful effect
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

    # ---- Plot: 2-panel ----
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))

    # Top: full ablation per prompt
    ax = axes[0]
    ax.bar(range(len(valid)), full_deltas, color="#d62728")
    ax.set_ylabel("Δ log p(target)")
    ax.set_title("Side-channel ablation (ALL layers zeroed) — per prompt")
    ax.set_xticks(range(len(valid)))
    ax.set_xticklabels([vp.target_token for vp in valid],
                       rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    # Bottom: per-layer single ablation (uses bar_by_layer)
    bar_by_layer(
        mean_per_layer, ax=axes[1],
        ylabel="mean Δ log p(target)",
        title="Side-channel ablation (ONE layer at a time)",
    )

    plt.tight_layout()
    out_path = OUT_DIR / "side_channel_ablation.png"
    fig.savefig(out_path, dpi=140)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
