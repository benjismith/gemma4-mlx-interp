"""Step 32 — Direct Logit Attribution demo: Paris vs. Berlin.

Canonical circuit-analysis question: on the prompt "the capital of France is",
which components (layers, branches, heads) contribute most to the model
preferring "Paris" over "Berlin"?

Uses the new attribution primitives in mechbench_core:

    accumulated_resid   — per-layer residual stack
    decompose_resid     — per-branch (attn/mlp/gate) contributions
    head_results        — per-head residual writes at one layer
    logit_attrs         — project any residual stack through the tied unembed

Writes two plots and a leaderboard to the shell:

    caches/dla_paris_berlin_by_layer.png  — (Paris - Berlin) at each layer's
                                            resid_post, plus per-branch bars
                                            at the top-contributing layers.
    caches/dla_paris_berlin_per_head.png  — [n_layers x n_heads] heatmap of
                                            per-head contribution to the
                                            Paris - Berlin logit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mechbench_core import (  # noqa: E402
    Capture,
    GLOBAL_LAYERS,
    Model,
    N_LAYERS,
    accumulated_resid,
    bar_by_layer,
    decompose_resid,
    head_heatmap,
    head_results,
    logit_attrs,
)


PROMPT = "Complete this sentence with one word: The capital of France is"
TARGETS = (" Paris", " Berlin")


def main() -> None:
    print("Loading model...")
    model = Model.load()
    n_heads = model.arch.n_heads

    ids = model.tokenize(PROMPT)
    seq_len = int(ids.shape[1])
    print(f"Prompt: {PROMPT!r}   seq_len={seq_len}")

    paris_id = int(model.tokenizer.encode(TARGETS[0], add_special_tokens=False)[0])
    berlin_id = int(model.tokenizer.encode(TARGETS[1], add_special_tokens=False)[0])
    print(f"Token ids: Paris={paris_id}, Berlin={berlin_id}")

    print("Running forward pass with attribution captures (~30s)...")
    interventions = [
        Capture.residual(range(N_LAYERS), point="post"),
        Capture.attn_out(range(N_LAYERS)),
        Capture.mlp_out(range(N_LAYERS)),
        Capture.gate_out(range(N_LAYERS)),
        Capture.per_head_out(range(N_LAYERS)),
    ]
    result = model.run(ids, interventions=interventions)
    cache = result.cache

    # Layer-level: how does (Paris - Berlin) evolve through the residual stream?
    resid_stack = accumulated_resid(cache)
    layer_attrs = logit_attrs(model, resid_stack, [paris_id, berlin_id])
    layer_diff = layer_attrs[:, 0] - layer_attrs[:, 1]
    print(f"\nLayer-level Paris - Berlin (at final position):")
    for i in range(N_LAYERS):
        marker = " G" if i in GLOBAL_LAYERS else "  "
        print(f"  L{i:02d}{marker}   Paris={layer_attrs[i, 0]:+7.3f}  "
              f"Berlin={layer_attrs[i, 1]:+7.3f}  diff={layer_diff[i]:+7.3f}")

    # Branch decomposition: at each layer, how do attn / mlp / gate contribute
    # (additively, since this is a raw-residual projection)?
    parts = decompose_resid(cache)
    branch_diff = {}
    for k, v in parts.items():
        attrs = logit_attrs(model, v, [paris_id, berlin_id])
        branch_diff[k] = attrs[:, 0] - attrs[:, 1]

    # Per-head contributions: [n_layers, n_heads] grid.
    head_diff = np.zeros((N_LAYERS, n_heads), dtype=np.float32)
    for layer in range(N_LAYERS):
        heads = head_results(model, cache, layer=layer)
        attrs = logit_attrs(model, heads, [paris_id, berlin_id])
        head_diff[layer] = attrs[:, 0] - attrs[:, 1]

    # Leaderboard: top-10 individual (layer, head) writers to the France->Paris direction.
    flat = [(l, h, head_diff[l, h]) for l in range(N_LAYERS) for h in range(n_heads)]
    flat.sort(key=lambda x: x[2], reverse=True)
    print("\nTop-10 heads by Paris - Berlin contribution:")
    for rank, (l, h, v) in enumerate(flat[:10], start=1):
        marker = " (global)" if l in GLOBAL_LAYERS else ""
        print(f"  #{rank:2d}   L{l:02d} H{h}   Δlogit = {v:+7.3f}{marker}")

    # Plot 1: layer-level bars with branch decomposition overlay.
    caches_dir = Path(__file__).resolve().parent.parent / "caches"
    caches_dir.mkdir(exist_ok=True)
    out1 = caches_dir / "dla_paris_berlin_by_layer.png"

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    bar_by_layer(
        layer_diff,
        ax=ax_top,
        title="Cumulative DLA: (Paris - Berlin) logit at each layer's resid_post",
        ylabel="Δ logit (raw, no norm fold)",
        show_global_lines=True,
    )

    x = np.arange(N_LAYERS)
    width = 0.28
    for i, (k, color) in enumerate(
        [("attn", "#d62728"), ("mlp", "#1f77b4"), ("gate", "#2ca02c")]
    ):
        ax_bot.bar(
            x + (i - 1) * width,
            branch_diff[k][:, ],  # [N_LAYERS]
            width=width,
            color=color,
            label=k,
        )
    ax_bot.axhline(0, color="black", linewidth=0.5)
    ax_bot.set_xlabel("layer")
    ax_bot.set_ylabel("Δ logit (per-branch marginal)")
    ax_bot.set_title("Per-branch DLA: attn / mlp / gate contributions at each layer")
    ax_bot.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out1, dpi=120)
    plt.close(fig)
    print(f"\nWrote {out1}")

    # Plot 2: per-head heatmap.
    out2 = caches_dir / "dla_paris_berlin_per_head.png"
    ax = head_heatmap(
        head_diff,
        title="Per-head DLA: (Paris - Berlin) contribution at each (layer, head)",
        colorbar_label="Δ logit",
    )
    ax.figure.tight_layout()
    ax.figure.savefig(out2, dpi=120)
    plt.close(ax.figure)
    print(f"Wrote {out2}")


if __name__ == "__main__":
    main()
