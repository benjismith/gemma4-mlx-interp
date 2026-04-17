"""Experiment 1 from docs/proposals/factorization-experiments.md.

2x2 design varying operation-type and 'capital'-word-presence:
  A1 = capital-lookup, 'capital' present in prompt
  A2 = capital-lookup, 'capital' absent (paraphrased)
  B1 = letter-counting, 'capital' present (operand contains it)
  B2 = letter-counting, 'capital' absent

Test which dimension the mid-layer subject-position activations cluster
along:
  - Operation-type (A vs B): factorization claim survives
  - Word-presence (1 vs 2):  the claim was a surface-token confound
  - Four distinct clusters:  mixed; both axes contribute

Run from project root:
    python experiments/step_15_operation_disambiguation.py
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
    Model, cluster_purity, cosine_matrix, fact_vectors_at,
    intra_inter_separation, iterate_clusters,
    nearest_neighbor_purity, silhouette_cosine,
)
from experiments.prompts import DISAMBIG_ALL  # noqa: E402

OUT_DIR = ROOT / "caches"
LAYER = 30


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading model...")
    model = Model.load()

    print(f"\nValidating {len(DISAMBIG_ALL)} prompts (no confidence floor, no target match)...\n")
    valid = DISAMBIG_ALL.validate(
        model, verbose=True, min_confidence=0.0, require_target_match=False,
    )
    n = len(valid)
    print(f"\n{n} of {len(DISAMBIG_ALL)} validated.\n")

    # 4-way category labels (the 'cell' label)
    cell_labels = valid.labels

    # Two binary candidate labelings
    op_labels = np.array(
        ["lookup" if c.startswith("A") else "counting" for c in cell_labels]
    )
    word_labels = np.array(
        ["capital_present" if c.endswith("_present") else "capital_absent"
         for c in cell_labels]
    )

    print(f"Extracting fact vectors at layer {LAYER} (subject position)...")
    vecs = fact_vectors_at(model, valid, layers=[LAYER])[LAYER]
    print(f"  vector matrix: {vecs.shape}")

    # ---- Statistics under each hypothesis ----
    def _report(name: str, labels: np.ndarray, k: int):
        print(f"\n  -- Grouping: {name}  (k={k}) --")
        intra, inter, sep = intra_inter_separation(vecs, labels)
        nn_rate, _ = nearest_neighbor_purity(vecs, labels)
        sil = silhouette_cosine(vecs, labels)
        unique = list(dict.fromkeys(labels.tolist()))
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(vecs)
        purity = cluster_purity(labels.tolist(), km.labels_.tolist())
        print(f"     intra-group cos:      {intra:+.4f}")
        print(f"     inter-group cos:      {inter:+.4f}")
        print(f"     separation:           {sep:+.4f}")
        print(f"     NN same-group rate:   {nn_rate:.3f}")
        print(f"     k-means purity:       {purity:.3f}  (chance = {1/k:.3f})")
        print(f"     silhouette (cosine):  {sil:+.4f}")
        return {
            "name": name, "intra": intra, "inter": inter, "sep": sep,
            "nn_rate": nn_rate, "purity": purity, "sil": sil,
        }

    print(f"\n{'=' * 60}")
    print(f"Cluster quality under each candidate grouping (layer {LAYER})")
    print(f"{'=' * 60}")
    op_stats = _report("operation-type (A vs B)", op_labels, k=2)
    word_stats = _report("word-presence (1 vs 2)", word_labels, k=2)
    cell_stats = _report("4-cell (A1/A2/B1/B2)", cell_labels, k=4)

    print(f"\n{'=' * 60}")
    print(f"Verdict")
    print(f"{'=' * 60}")
    op_sep = op_stats["sep"]
    word_sep = word_stats["sep"]
    print(f"  separation under operation-type hypothesis: {op_sep:+.4f}")
    print(f"  separation under word-presence hypothesis:  {word_sep:+.4f}")
    print(f"  ratio (op / word):                          {op_sep / max(word_sep, 1e-6):.2f}x")
    print()
    if op_sep > word_sep * 2:
        print(f"  -> OPERATION-TYPE wins decisively. Factorization claim survives.")
    elif word_sep > op_sep * 2:
        print(f"  -> WORD-PRESENCE wins decisively. The original cluster signal was a surface-token confound.")
    elif abs(op_sep - word_sep) < 0.01:
        print(f"  -> Both hypotheses fit similarly; check the 4-cell purity for the mixed-result reading.")
    else:
        s = "operation-type" if op_sep > word_sep else "word-presence"
        print(f"  -> {s} fits modestly better, but the result is not clean.")

    # NN within-cell hit rate (sanity check that the 4 cells are at all distinguishable)
    nn_cell, _ = nearest_neighbor_purity(vecs, cell_labels)
    print(f"\n  4-cell NN same-cell hit rate: {nn_cell:.3f}  (chance = 0.226)")

    # ---- PCA: 2-panel scatter colored by both labelings ----
    pca = PCA(n_components=2, random_state=42).fit(vecs)
    proj = pca.transform(vecs)
    var = pca.explained_variance_ratio_.sum()
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Panel 1: colored by operation-type
    ax = axes[0]
    op_colors = {"lookup": "#1f77b4", "counting": "#d62728"}
    for op, _, mask in iterate_clusters(proj, op_labels):
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   c=op_colors[op], label=f"{op} (n={int(mask.sum())})",
                   s=80, alpha=0.85, edgecolors="black", linewidths=0.5)
    ax.set_title(f"Colored by OPERATION-TYPE\n"
                 f"separation = {op_stats['sep']:+.4f}, "
                 f"sil = {op_stats['sil']:+.4f}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 2: colored by word-presence
    ax = axes[1]
    word_colors = {"capital_present": "#2ca02c", "capital_absent": "#9467bd"}
    for wp, _, mask in iterate_clusters(proj, word_labels):
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   c=word_colors[wp], label=f"{wp} (n={int(mask.sum())})",
                   s=80, alpha=0.85, edgecolors="black", linewidths=0.5)
    ax.set_title(f"Colored by 'capital'-WORD-PRESENCE\n"
                 f"separation = {word_stats['sep']:+.4f}, "
                 f"sil = {word_stats['sil']:+.4f}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Operation-word disambiguation at layer {LAYER} (PCA var {var:.1%})",
        fontsize=12,
    )
    plt.tight_layout()
    out_path = OUT_DIR / "operation_disambiguation.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")

    # ---- Also: the cell-level 4-color PCA, for diagnostic completeness ----
    fig, ax = plt.subplots(figsize=(10, 8))
    cell_colors = {
        "A1_lookup_capital_present": "#1f77b4",
        "A2_lookup_capital_absent":  "#aec7e8",
        "B1_counting_capital_present": "#d62728",
        "B2_counting_capital_absent":  "#ff9896",
    }
    cell_short = {
        "A1_lookup_capital_present": "A1: lookup + capital",
        "A2_lookup_capital_absent":  "A2: lookup, no capital",
        "B1_counting_capital_present": "B1: counting + capital",
        "B2_counting_capital_absent":  "B2: counting, no capital",
    }
    for c, _, mask in iterate_clusters(proj, cell_labels):
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   c=cell_colors[c], label=cell_short[c],
                   s=100, alpha=0.85, edgecolors="black", linewidths=0.5)
    ax.set_title(f"Colored by all 4 cells (layer {LAYER})\n"
                 f"4-cell NN hit rate: {nn_cell:.3f}, "
                 f"4-cell purity: {cell_stats['purity']:.3f}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path2 = OUT_DIR / "operation_disambiguation_4cell.png"
    fig.savefig(out_path2, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path2}")


if __name__ == "__main__":
    main()
