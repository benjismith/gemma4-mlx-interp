"""Weight-level analysis of every attention head in Gemma 4 E4B.

For each of 42 layers x 8 Q-heads = 336 heads:
  - READ tokens: top-10 tokens by ||W_Q[h] @ E_unit[t]||^2
  - KEY tokens:  top-10 tokens by ||W_K[kv_group(h)] @ E_unit[t]||^2
  - OV circuit:  top-5 singular components of W_O[h-slice] @ W_V[kv_group(h)],
                 each with top-10 output tokens and top-10 input tokens

Uses unit-normalized token embeddings because the model's pre-attention
RMSNorm rescales each residual to roughly unit RMS before Q/K/V
projection. Raw embeddings produce rankings dominated by rare-token
magnitude artifacts.

All results dumped to caches/head_weight_analysis.json for downstream
browsing; a summary table is printed to stdout. A grid visualization
(42 x 8 heatmap, cells colored by rank-0 OV singular value) is saved
to caches/head_weight_analysis.png.

Expected runtime: ~10 minutes for all 336 heads on M2 Pro.

Run from project root:
    python experiments/step_26_head_weight_analysis.py
"""

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gemma4_mlx_interp import (  # noqa: E402
    GLOBAL_LAYERS, Model, N_LAYERS,
    get_head_spec, head_key_tokens, head_read_tokens, ov_circuit, qk_circuit,
)
from gemma4_mlx_interp.head_weights import _unit_normalized_embed  # noqa: E402

OUT_DIR = ROOT / "caches"
N_HEADS = 8
OV_COMPONENTS = 5
TOP_K_TOKENS = 10


def analyze_head(model, layer: int, head: int, embed) -> dict:
    spec = get_head_spec(model, layer, head)
    reads = head_read_tokens(model, layer, head, k=TOP_K_TOKENS, embed=embed)
    keys = head_key_tokens(model, layer, head, k=TOP_K_TOKENS, embed=embed)
    ov = ov_circuit(model, layer, head, k_tokens=TOP_K_TOKENS,
                    n_components=OV_COMPONENTS, embed=embed)
    return {
        "layer": layer,
        "head": head,
        "kv_group": spec.kv_group,
        "head_dim": spec.head_dim,
        "is_global": spec.is_global,
        "is_kv_shared": spec.is_kv_shared,
        "read_tokens": [[t, s] for t, s in reads],
        "key_tokens": [[t, s] for t, s in keys],
        "ov_components": [
            {
                "rank": c.rank,
                "strength": c.strength,
                "writes": [[t, s] for t, s in c.left_tokens],
                "when_sees": [[t, s] for t, s in c.right_tokens],
            }
            for c in ov.components
        ],
    }


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading model...")
    model = Model.load()
    print("Computing unit-normalized embedding matrix...")
    t0 = time.perf_counter()
    E_unit = _unit_normalized_embed(model)
    print(f"  [{time.perf_counter() - t0:.1f}s]  shape={E_unit.shape}")

    # ---- Sweep all heads ----
    print(f"\nAnalyzing {N_LAYERS} x {N_HEADS} = {N_LAYERS * N_HEADS} heads...")
    t_start = time.perf_counter()
    all_heads = []
    for L in range(N_LAYERS):
        for h in range(N_HEADS):
            result = analyze_head(model, L, h, E_unit)
            all_heads.append(result)
        elapsed = time.perf_counter() - t_start
        eta = elapsed / (L + 1) * (N_LAYERS - L - 1)
        kind = "GLOBAL" if L in GLOBAL_LAYERS else "local"
        shared = " (kv-shared)" if all_heads[-1]["is_kv_shared"] else ""
        print(f"  L{L:>2} {kind}{shared}  head_dim={all_heads[-1]['head_dim']:>3}  "
              f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]")

    total = time.perf_counter() - t_start
    print(f"\nDone in {total:.0f}s ({total / len(all_heads) * 1000:.0f}ms/head)")

    # ---- Save JSON ----
    json_path = OUT_DIR / "head_weight_analysis.json"
    with open(json_path, "w") as f:
        json.dump({
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "ov_components": OV_COMPONENTS,
            "top_k_tokens": TOP_K_TOKENS,
            "heads": all_heads,
        }, f, indent=1)
    print(f"Wrote {json_path}")

    # ---- Print summary table ----
    print(f"\n{'=' * 100}")
    print(f"Per-head summary: rank-0 OV component (top-3 writes)")
    print(f"{'=' * 100}")
    print(f"{'L':>3}  {'h':>2}  {'type':>6}  {'sigma_0':>8}  {'writes (top 3)'}")
    print("-" * 100)
    for h_info in all_heads:
        L = h_info["layer"]
        hd = h_info["head"]
        kind = "GLB" if h_info["is_global"] else "loc"
        ov0 = h_info["ov_components"][0]
        top_writes = "  ".join(f"{t!r}" for t, _ in ov0["writes"][:3])
        print(f"  {L:>3}  {hd:>2}  {kind:>6}  {ov0['strength']:>8.3f}  {top_writes}")

    # ---- Visualization: heatmap of rank-0 OV singular values ----
    sigma_0 = np.zeros((N_LAYERS, N_HEADS), dtype=np.float32)
    for h_info in all_heads:
        sigma_0[h_info["layer"], h_info["head"]] = h_info["ov_components"][0]["strength"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 9))

    # Panel 1: rank-0 OV singular value heatmap
    ax = axes[0]
    im = ax.imshow(sigma_0, aspect="auto", cmap="viridis",
                   interpolation="nearest")
    ax.set_xlabel("Q-head index")
    ax.set_ylabel("layer")
    ax.set_xticks(range(N_HEADS))
    ax.set_yticks(range(0, N_LAYERS, 3))
    ax.set_title("OV-circuit rank-0 singular value\n"
                 f"(larger = stronger single-direction copy)", fontsize=10)
    # Mark global layers with a left-edge band
    for g in GLOBAL_LAYERS:
        ax.axhline(g, color="red", linewidth=0.4, alpha=0.35, xmin=-0.02, xmax=0.02)
    plt.colorbar(im, ax=ax)

    # Panel 2: Top-3 rank-0 OV sigma leaderboard (annotated)
    ax = axes[1]
    flat = [(h["layer"], h["head"], h["ov_components"][0]["strength"],
             h["ov_components"][0]["writes"][0][0], h["is_global"])
            for h in all_heads]
    flat.sort(key=lambda x: -x[2])
    top20 = flat[:20]
    ys = np.arange(len(top20))
    values = [x[2] for x in top20]
    colors = ["#d62728" if x[4] else "#1f77b4" for x in top20]
    ax.barh(ys, values, color=colors)
    ax.set_yticks(ys)
    ax.set_yticklabels(
        [f"L{L:>2} h{h}  {tok!r}" for L, h, _, tok, _ in top20],
        fontsize=8,
    )
    ax.invert_yaxis()
    ax.set_xlabel("rank-0 OV singular value")
    ax.set_title("Top 20 heads by OV rank-0 strength\n"
                 "(red=global, blue=local; token = top write)",
                 fontsize=10)
    ax.grid(True, alpha=0.3, axis="x")

    fig.suptitle(
        "Static weight-level per-head analysis — Gemma 4 E4B",
        fontsize=12,
    )
    plt.tight_layout()
    out_path = OUT_DIR / "head_weight_analysis.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")

    # ---- Highlight interesting heads ----
    print(f"\n{'=' * 70}")
    print(f"Top 10 heads by OV rank-0 singular value")
    print(f"{'=' * 70}")
    for L, h, sigma, tok, is_global in flat[:10]:
        kind = "GLOBAL" if is_global else "local "
        print(f"  L{L:>2} h{h}  {kind}  sigma_0={sigma:.3f}  top write={tok!r}")


if __name__ == "__main__":
    main()
