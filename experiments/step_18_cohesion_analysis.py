"""Cohesion vs centroid decoding sharpness.

Cohesion of a cluster = mean cosine similarity between each member vector
and the cluster centroid. A scalar in [-1, 1]; higher = tighter cluster.

Question: does within-cluster cohesion predict how cleanly the cluster's
centroid decodes through the model's tied unembed?

Test corpus:
  - BIG_SWEEP_96: 12 categorical clusters (capital, element, author,
    landmark, opposite, past_tense, plural, french, profession,
    animal_home, color_mix, math), 8 prompts each.
  - HOMONYM_CAPITAL_ALL: 4 sense clusters (city, finance, uppercase,
    punishment), 8 prompts each.

Total: 16 clusters at layer 30 and again at layer 41 (the canonical
decoding depths from findings 12 and 17 respectively).

For each cluster, compute:
  - cohesion (the new metric)
  - centroid top-1 decoded token + its probability (mean-subtracted via
    the cluster's own corpus overall mean)
  - centroid top-1 entropy of decoded distribution

Plot cohesion vs top-1 decoded probability, scatter colored by corpus.
Compute Pearson correlation. Print per-cluster table.

Run from project root:
    python experiments/step_18_cohesion_analysis.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gemma4_mlx_interp import (  # noqa: E402
    Model, cohesion, fact_vectors_at, iterate_clusters, vocab_concentration,
)
from experiments.prompts import BIG_SWEEP_96, HOMONYM_CAPITAL_ALL  # noqa: E402

OUT_DIR = ROOT / "caches"
LAYERS = [30, 41]


def analyze_corpus(model, name: str, prompt_set, layers: list[int]) -> list[dict]:
    """Validate the corpus, extract subject-position vectors at each layer,
    return one record per (cluster, layer) tuple."""
    print(f"\nValidating {name} ({len(prompt_set)} prompts)...")
    valid = prompt_set.validate(
        model, verbose=False, min_confidence=0.0, require_target_match=False,
    )
    n = len(valid)
    print(f"  {n} of {len(prompt_set)} validated.")

    print(f"  Extracting residuals at layers {layers}...")
    vecs_by_layer = fact_vectors_at(model, valid, layers=layers)
    labels = valid.labels

    records = []
    for L in layers:
        vecs = vecs_by_layer[L]
        overall_mean = vecs.mean(axis=0)
        for cat, cluster_vecs, mask in iterate_clusters(vecs, labels):
            coh = cohesion(cluster_vecs)
            # Mean-subtracted centroid (corpus-wide mean)
            centroid_sub = (cluster_vecs.mean(axis=0) - overall_mean).astype(np.float32)
            probs = model.decoded_distribution(centroid_sub)
            conc = vocab_concentration(probs, k=5)
            top1_id = int(np.argmax(probs))
            top1_tok = model.tokenizer.decode([top1_id])
            records.append({
                "corpus": name,
                "layer": L,
                "cluster": cat,
                "n": int(mask.sum()),
                "cohesion": coh,
                "top1_tok": top1_tok,
                "top1_p": conc.top1,
                "top5_mass": conc.top_k_mass,
                "entropy_bits": conc.entropy_bits,
                "effective_size": conc.effective_vocab_size,
            })
    return records


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading model...")
    model = Model.load()

    records = []
    records += analyze_corpus(model, "BIG_SWEEP_96", BIG_SWEEP_96, LAYERS)
    records += analyze_corpus(model, "HOMONYM_CAPITAL", HOMONYM_CAPITAL_ALL, LAYERS)

    # ---- Per-layer printout ----
    for L in LAYERS:
        print(f"\n{'=' * 80}")
        print(f"Layer {L} -- per-cluster cohesion + centroid decoding sharpness")
        print(f"{'=' * 80}")
        print(f"{'corpus':>15s}  {'cluster':>15s}  {'n':>2}  "
              f"{'cohesion':>9s}  {'top-1 tok':>16s}  "
              f"{'top1_p':>7s}  {'top5_mass':>9s}  {'ent (bits)':>10s}")
        print("-" * 100)
        for r in records:
            if r["layer"] != L:
                continue
            print(
                f"{r['corpus']:>15s}  {r['cluster']:>15s}  {r['n']:>2}  "
                f"{r['cohesion']:>9.4f}  {r['top1_tok']!r:>16s}  "
                f"{r['top1_p']:>7.4f}  {r['top5_mass']:>9.4f}  {r['entropy_bits']:>10.3f}"
            )

    # ---- Correlation analysis ----
    print(f"\n{'=' * 80}")
    print(f"Correlation: cohesion vs top-1 decoded probability")
    print(f"{'=' * 80}")
    for L in LAYERS:
        L_recs = [r for r in records if r["layer"] == L]
        x = np.array([r["cohesion"] for r in L_recs])
        y = np.array([r["top1_p"] for r in L_recs])
        ent = np.array([r["entropy_bits"] for r in L_recs])
        # Pearson correlations
        rho_top1 = float(np.corrcoef(x, y)[0, 1])
        rho_ent = float(np.corrcoef(x, ent)[0, 1])
        print(f"\n  Layer {L} (n={len(L_recs)}):")
        print(f"    cohesion -> top-1 probability: r = {rho_top1:+.3f}")
        print(f"    cohesion -> entropy (bits):    r = {rho_ent:+.3f}  "
              f"(negative expected if higher cohesion -> sharper decoding)")
        # Ranges
        print(f"    cohesion range: [{x.min():.4f}, {x.max():.4f}]")
        print(f"    top-1 prob range: [{y.min():.4f}, {y.max():.4f}]")

    # ---- Scatter plot ----
    fig, axes = plt.subplots(1, len(LAYERS), figsize=(7 * len(LAYERS), 6),
                              sharey=False)
    if len(LAYERS) == 1:
        axes = [axes]
    corpus_color = {"BIG_SWEEP_96": "#1f77b4", "HOMONYM_CAPITAL": "#d62728"}

    for ax, L in zip(axes, LAYERS):
        L_recs = [r for r in records if r["layer"] == L]
        for r in L_recs:
            color = corpus_color[r["corpus"]]
            ax.scatter(r["cohesion"], r["top1_p"], c=color, s=140,
                       alpha=0.85, edgecolors="black", linewidths=0.5)
            ax.annotate(
                f"{r['cluster']}\n{r['top1_tok']!r}",
                xy=(r["cohesion"], r["top1_p"]),
                xytext=(8, 5), textcoords="offset points",
                fontsize=7, alpha=0.85,
            )
        x = np.array([r["cohesion"] for r in L_recs])
        y = np.array([r["top1_p"] for r in L_recs])
        rho = float(np.corrcoef(x, y)[0, 1])
        ax.set_xlabel("cohesion (mean cos to centroid)")
        ax.set_ylabel("top-1 decoded probability of (mean-sub) centroid")
        ax.set_title(f"Layer {L}\nPearson r = {rho:+.3f}  (n = {len(L_recs)})",
                     fontsize=11)
        ax.grid(True, alpha=0.3)
        # Linear fit for visual reference
        if len(x) >= 2:
            coef = np.polyfit(x, y, 1)
            xs_line = np.array([x.min(), x.max()])
            ax.plot(xs_line, coef[0] * xs_line + coef[1],
                    color="black", linestyle="--", linewidth=1, alpha=0.5)

    # Legend in the leftmost panel
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=c, markeredgecolor="black",
               markersize=10, label=name)
        for name, c in corpus_color.items()
    ]
    axes[0].legend(handles=legend_handles, loc="upper left", fontsize=9)

    fig.suptitle(
        "Cohesion vs centroid decoding sharpness across 16 clusters\n"
        "(12 BIG_SWEEP categories + 4 HOMONYM_CAPITAL senses)",
        fontsize=12,
    )
    plt.tight_layout()
    out_path = OUT_DIR / "cohesion_vs_decoding.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
