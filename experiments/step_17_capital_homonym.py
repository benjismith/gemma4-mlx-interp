"""Homonym sense disambiguation for the word 'capital'.

Take 32 prompts using 'capital' in 4 distinct senses (city / finance /
uppercase / punishment, 8 each). For each prompt, extract the residual
at the 'capital' token's position at every layer, and ask:

  - At what layer do the senses start to cluster geometrically?
  - At layer 0 they should be ~identical (just the embedding for 'capital').
    At what depth does context-driven sense disambiguation kick in?
  - How clean is the separation at the deepest layers?
  - Do the per-sense centroids (mean-subtracted) decode to sense-relevant
    tokens through the unembed?

Run from project root:
    python experiments/step_17_capital_homonym.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gemma4_mlx_interp import (  # noqa: E402
    GLOBAL_LAYERS, Model, N_LAYERS,
    centroid_decode, cluster_purity, cosine_matrix, fact_vectors_at,
    intra_inter_separation, iterate_clusters, nearest_neighbor_purity,
    silhouette_cosine,
)
from experiments.prompts import HOMONYM_CAPITAL_ALL  # noqa: E402

OUT_DIR = ROOT / "caches"
PCA_LAYERS = [0, 10, 20, 30, 41]


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading model...")
    model = Model.load()

    print(f"\nValidating {len(HOMONYM_CAPITAL_ALL)} prompts (no filtering)...\n")
    valid = HOMONYM_CAPITAL_ALL.validate(
        model, verbose=False, min_confidence=0.0, require_target_match=False,
    )
    n = len(valid)
    print(f"  {n} of {len(HOMONYM_CAPITAL_ALL)} validated.\n")

    labels = valid.labels
    senses = valid.categories
    print(f"Senses present: {senses}")
    print(f"Counts: {dict((s, int(np.sum(labels == s))) for s in senses)}")

    # ---- Extract at every layer at the 'capital' position ----
    print(f"\nExtracting residuals at the 'capital' position across all "
          f"{N_LAYERS} layers ({n} forward passes total)...")
    vecs_by_layer = fact_vectors_at(
        model, valid, layers=range(N_LAYERS), position="subject",
    )
    print(f"  done. Each layer: array of shape {vecs_by_layer[0].shape}")

    # ---- Per-layer separation metrics ----
    print(f"\n{'=' * 70}")
    print(f"Per-layer 4-sense separation metrics (k=4, chance purity = 0.25)")
    print(f"{'=' * 70}")
    print(f"{'layer':>6}  {'intra':>8}  {'inter':>8}  {'sep':>8}  "
          f"{'NN':>6}  {'purity':>8}  {'sil':>8}")
    print("-" * 65)
    sil_per_layer = np.zeros(N_LAYERS)
    sep_per_layer = np.zeros(N_LAYERS)
    nn_per_layer = np.zeros(N_LAYERS)
    purity_per_layer = np.zeros(N_LAYERS)

    for L in range(N_LAYERS):
        vecs = vecs_by_layer[L]
        intra, inter, sep = intra_inter_separation(vecs, labels)
        nn_rate, _ = nearest_neighbor_purity(vecs, labels)
        sil = silhouette_cosine(vecs, labels)
        km = KMeans(n_clusters=4, n_init=10, random_state=42).fit(vecs)
        purity = cluster_purity(labels.tolist(), km.labels_.tolist())
        sil_per_layer[L] = sil
        sep_per_layer[L] = sep
        nn_per_layer[L] = nn_rate
        purity_per_layer[L] = purity
        # Print every 3 layers + globals
        if L % 3 == 0 or L in GLOBAL_LAYERS or L == N_LAYERS - 1:
            tag = "GLOBAL" if L in GLOBAL_LAYERS else ""
            print(f"  L{L:>3} {tag:>6s}  {intra:+.4f}  {inter:+.4f}  "
                  f"{sep:+.4f}  {nn_rate:.3f}  {purity:.3f}    {sil:+.4f}")

    # ---- Find the depth profile inflection point ----
    # The first layer where silhouette > 0.1 is the rough sense-emergence depth
    emerge_layer = next((L for L in range(N_LAYERS) if sil_per_layer[L] > 0.1), None)
    peak_layer = int(np.argmax(sil_per_layer))
    print(f"\nFirst layer where silhouette > 0.1: "
          f"{emerge_layer if emerge_layer is not None else 'never'}")
    print(f"Peak silhouette layer: {peak_layer} (silhouette={sil_per_layer[peak_layer]:+.4f}, "
          f"NN={nn_per_layer[peak_layer]:.3f}, purity={purity_per_layer[peak_layer]:.3f})")

    # ---- Plot: depth profile of separation metrics ----
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(N_LAYERS)
    ax.plot(x, sil_per_layer, marker="o", label="silhouette (cosine)",
            color="#1f77b4", linewidth=2)
    ax.plot(x, sep_per_layer, marker="s", label="intra-inter separation",
            color="#2ca02c", linewidth=2)
    ax.plot(x, nn_per_layer, marker="^", label="NN same-sense rate (right axis)",
            color="#d62728", linewidth=2)
    ax.set_xlabel("layer")
    ax.set_ylabel("metric value")
    ax.axhline(0.0, color="gray", linewidth=0.5)
    ax.axhline(0.25, color="gray", linestyle=":", linewidth=0.5, label="chance NN/purity")
    ax.set_title("Sense-disambiguation depth profile for 'capital' "
                 "(4 senses, 8 prompts each, residual at the 'capital' position)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    for g in GLOBAL_LAYERS:
        ax.axvline(g, color="#999999", linestyle="--", linewidth=0.7, alpha=0.5)
    plt.tight_layout()
    out1 = OUT_DIR / "homonym_capital_depth_profile.png"
    fig.savefig(out1, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out1}")

    # ---- Plot: PCA at selected layers ----
    sense_colors = {
        "sense_city": "#1f77b4",
        "sense_finance": "#2ca02c",
        "sense_uppercase": "#ff7f0e",
        "sense_punishment": "#d62728",
    }
    sense_short = {
        "sense_city": "city",
        "sense_finance": "finance",
        "sense_uppercase": "uppercase",
        "sense_punishment": "punishment",
    }
    fig, axes = plt.subplots(1, len(PCA_LAYERS), figsize=(20, 5))
    for ax, L in zip(axes, PCA_LAYERS):
        vecs = vecs_by_layer[L]
        proj = PCA(n_components=2, random_state=42).fit_transform(vecs)
        for sense, _, mask in iterate_clusters(proj, labels):
            ax.scatter(proj[mask, 0], proj[mask, 1],
                       c=sense_colors[sense],
                       label=sense_short[sense],
                       s=70, alpha=0.85, edgecolors="black", linewidths=0.4)
        ax.set_title(f"Layer {L}\nsil={sil_per_layer[L]:+.3f}, "
                     f"NN={nn_per_layer[L]:.2f}", fontsize=10)
        ax.set_xlabel("PC1")
        if ax is axes[0]:
            ax.set_ylabel("PC2")
            ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Homonym disambiguation for 'capital' across depth (PCA at selected layers)",
                 fontsize=12)
    plt.tight_layout()
    out2 = OUT_DIR / "homonym_capital_pca_grid.png"
    fig.savefig(out2, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out2}")

    # ---- Centroid decoding at multiple layers ----
    # Peak silhouette is in the engine room (vocab-opaque per finding 08), so
    # those decodings are likely gibberish. Try the canonical readout depth (30)
    # and the very-late layers too, where representations should be more
    # decodable through the unembed.
    decode_layers = sorted(set([peak_layer, 25, 30, 35, 41]))
    for L in decode_layers:
        print(f"\n{'=' * 70}")
        print(f"Centroid decoding at layer {L} (mean-subtracted)  "
              f"[silhouette={sil_per_layer[L]:+.3f}, NN={nn_per_layer[L]:.3f}]")
        print(f"{'=' * 70}\n")
        vecs_L = vecs_by_layer[L]
        overall_mean = vecs_L.mean(axis=0)
        for sense, sense_vecs, _ in iterate_clusters(vecs_L, labels):
            top = centroid_decode(
                model, sense_vecs, k=8, mean_subtract=True,
                overall_mean=overall_mean,
            )
            print(f"  [{sense_short[sense]:>10s}] " +
                  "  ".join(f"{t!r}({p:.3f})" for t, p in top))


if __name__ == "__main__":
    main()
