"""Big sweep: 12 categories x 8 prompts with statistical baselines.

Scales the geometric analysis from finding 10 (5 categories / 40 prompts)
to 12 categories / 96 prompts. Adds a random-subset baseline: compute
centroids of random 8-prompt subsets and project — if random centroids
also decode to multilingual concept words the effect isn't signal.

Per finding 12: cluster quality scales perfectly (NN purity 1.000,
k-means purity 1.000), every category centroid decodes to its
multilingual relational frame, random-subset centroids mostly decode to
noise.

Run from project root:
    python experiments/step_12_big_sweep.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gemma4_mlx_interp import (  # noqa: E402
    BIG_SWEEP_96, Model,
    centroid_decode, cluster_purity, cosine_matrix, fact_vectors_at,
    intra_inter_separation, nearest_neighbor_purity,
    pca_scatter, silhouette_cosine, similarity_heatmap,
)

OUT_DIR = ROOT / "caches"
PROJECT_LAYER = 30
EXTRACT_LAYERS = [15, 30]


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading model...")
    model = Model.load()

    print(f"\nValidating {len(BIG_SWEEP_96)} prompts...\n")
    valid = BIG_SWEEP_96.validate(model, verbose=False, min_confidence=0.0)
    n = len(valid)
    print(f"  validated: {n} of {len(BIG_SWEEP_96)}")
    labels = np.array([vp.prompt.category for vp in valid])

    present_cats = list(dict.fromkeys(labels.tolist()))
    for cat in present_cats:
        count = int(np.sum(labels == cat))
        print(f"  {cat:>11s}: {count} prompts")

    print(f"\nExtracting fact vectors at layers {EXTRACT_LAYERS}...")
    vecs_by_layer = fact_vectors_at(model, valid, EXTRACT_LAYERS)
    vecs = vecs_by_layer[PROJECT_LAYER]

    # ---- Cluster quality ----
    print(f"\n{'=' * 70}")
    print(f"Cluster quality at layer {PROJECT_LAYER}")
    print(f"{'=' * 70}")

    k = len(present_cats)
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(vecs)
    purity = cluster_purity(labels.tolist(), km.labels_.tolist())
    sil = silhouette_cosine(vecs, labels)
    nn_rate, _ = nearest_neighbor_purity(vecs, labels)
    intra, inter, sep = intra_inter_separation(vecs, labels)

    print(f"  n = {n}, k = {k}")
    print(f"  k-means purity: {purity:.3f} (chance = {1 / k:.3f})")
    print(f"  silhouette (cosine, ground-truth labels): {sil:+.4f}")
    print(f"  NN same-category hit rate: {nn_rate:.3f}")
    print(f"  intra-cat cos: {intra:+.4f}")
    print(f"  inter-cat cos: {inter:+.4f}")
    print(f"  separation:    {sep:+.4f}")

    # ---- PCA ----
    fig, ax = plt.subplots(figsize=(11, 9))
    pca_scatter(
        vecs, labels, ax=ax,
        title=f"Fact-vector geometry at layer {PROJECT_LAYER} — "
              f"{n} prompts, {k} categories",
    )
    plt.tight_layout()
    out_path = OUT_DIR / "big_sweep_pca.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")

    # ---- Centroid decoding per category ----
    print(f"\n{'=' * 70}")
    print(f"Category centroid decoding at layer {PROJECT_LAYER} "
          f"(mean-subtracted, top-8)")
    print(f"{'=' * 70}\n")

    overall_mean = vecs.mean(axis=0)
    centroid_top_tokens: dict[str, list[str]] = {}
    for cat in present_cats:
        mask = labels == cat
        top = centroid_decode(model, vecs[mask], k=8, mean_subtract=True,
                              overall_mean=overall_mean)
        centroid_top_tokens[cat] = [t for t, _ in top]
        print(f"  [{cat:>11s}] " + "  ".join(f"{t!r}({p:.3f})" for t, p in top))

    # ---- Random-subset baseline ----
    print(f"\n{'=' * 70}")
    print("Random-subset baseline (centroids of random 8-prompt subsets)")
    print(f"{'=' * 70}\n")
    rng = np.random.default_rng(42)
    n_random = 10
    random_top_tokens: list[list[str]] = []
    for r in range(n_random):
        idx = rng.choice(n, size=8, replace=False)
        top = centroid_decode(model, vecs[idx], k=8, mean_subtract=True,
                              overall_mean=overall_mean)
        random_top_tokens.append([t for t, _ in top])
        print(f"  [random #{r + 1:>2d}]  " +
              "  ".join(f"{t!r}({p:.3f})" for t, p in top))

    # ---- Token-set uniqueness ----
    print(f"\n{'=' * 70}")
    print("Token-set uniqueness analysis")
    print(f"{'=' * 70}\n")

    cat_overlaps = []
    cat_names = list(centroid_top_tokens.keys())
    for i in range(len(cat_names)):
        for j in range(i + 1, len(cat_names)):
            a = set(centroid_top_tokens[cat_names[i]])
            b = set(centroid_top_tokens[cat_names[j]])
            jac = len(a & b) / len(a | b) if a | b else 0.0
            cat_overlaps.append(jac)

    rand_overlaps = []
    for i in range(len(random_top_tokens)):
        for j in range(i + 1, len(random_top_tokens)):
            a = set(random_top_tokens[i])
            b = set(random_top_tokens[j])
            jac = len(a & b) / len(a | b) if a | b else 0.0
            rand_overlaps.append(jac)

    print(f"  Category x category Jaccard overlap: "
          f"mean {np.mean(cat_overlaps):.3f}, max {np.max(cat_overlaps):.3f}")
    print(f"  Random x random Jaccard overlap:     "
          f"mean {np.mean(rand_overlaps):.3f}, max {np.max(rand_overlaps):.3f}")

    # ---- Similarity heatmap ----
    fig, ax = plt.subplots(figsize=(12, 10))
    similarity_heatmap(
        vecs, labels, ax=ax,
        title=f"Cosine similarity at layer {PROJECT_LAYER} "
              f"(n={n}, grouped by category)",
    )
    plt.tight_layout()
    out_path = OUT_DIR / "big_sweep_similarity_heatmap.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
