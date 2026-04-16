"""Geometric analysis of vocab-opaque fact vectors in Gemma 4 E4B.

Causal tracing (finding 09) established that the residual stream at the last
subject-entity token in middle layers carries information that causally
determines the factual answer — but in a form the unembed can't decode.
This experiment asks: does that information have geometric structure?

Run a battery of 40 labeled factual-recall prompts in 5 semantic categories
(capitals, elements, authors, landmarks, opposites). For each, extract the
residual stream at the last subject-entity token position. Then:

  1. Pairwise cosine similarity matrix, grouped by category (heatmap).
  2. PCA projection to 2D, colored by category.
  3. K-means clustering, measure purity against ground-truth labels.
  4. Same analysis at three depths (layer 5, 15, 30) to see how geometry
     evolves from pre-engine-room to in-engine-room to post-handoff.

This generalizes word2vec-style geometric analysis of word embeddings to
mid-depth transformer internal representations.

Run from project root:
    python experiments/step_10_fact_vector_geometry.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forward import load_model, _tokenize  # noqa: E402
from hooks import run_with_cache  # noqa: E402

OUT_DIR = ROOT / "caches"
EXTRACT_LAYERS = [5, 15, 30]

# (prompt, subject_substring_for_position_lookup, expected_answer, category)
# The subject substring identifies the last content token whose residual we'll
# extract. Keep category sizes balanced (8 per category, 5 categories = 40).
PROMPTS = [
    # Capitals (subject = country, answer = city)
    ("Complete this sentence with one word: The capital of France is", "France", "Paris", "capital"),
    ("Complete this sentence with one word: The capital of Japan is", "Japan", "Tokyo", "capital"),
    ("Complete this sentence with one word: The capital of Germany is", "Germany", "Berlin", "capital"),
    ("Complete this sentence with one word: The capital of Italy is", "Italy", "Rome", "capital"),
    ("Complete this sentence with one word: The capital of Spain is", "Spain", "Madrid", "capital"),
    ("Complete this sentence with one word: The capital of Russia is", "Russia", "Moscow", "capital"),
    ("Complete this sentence with one word: The capital of Egypt is", "Egypt", "Cairo", "capital"),
    ("Complete this sentence with one word: The capital of Greece is", "Greece", "Athens", "capital"),
    # Chemical elements (subject = element name, answer = symbol)
    ("Complete this sentence with one word: The chemical symbol for gold is", "gold", "Au", "element"),
    ("Complete this sentence with one word: The chemical symbol for silver is", "silver", "Ag", "element"),
    ("Complete this sentence with one word: The chemical symbol for iron is", "iron", "Fe", "element"),
    ("Complete this sentence with one word: The chemical symbol for oxygen is", "oxygen", "O", "element"),
    ("Complete this sentence with one word: The chemical symbol for hydrogen is", "hydrogen", "H", "element"),
    ("Complete this sentence with one word: The chemical symbol for carbon is", "carbon", "C", "element"),
    ("Complete this sentence with one word: The chemical symbol for sodium is", "sodium", "Na", "element"),
    ("Complete this sentence with one word: The chemical symbol for copper is", "copper", "Cu", "element"),
    # Authors (subject = book title, answer = author)
    ("Complete this sentence with one word: Romeo and Juliet was written by", "Juliet", "Shakespeare", "author"),
    ("Complete this sentence with one word: Pride and Prejudice was written by", "Prejudice", "Jane", "author"),
    ("Complete this sentence with one word: The Great Gatsby was written by", "Gatsby", "F", "author"),
    ("Complete this sentence with one word: Moby Dick was written by", "Dick", "Herman", "author"),
    ("Complete this sentence with one word: War and Peace was written by", "Peace", "Leo", "author"),
    ("Complete this sentence with one word: The Odyssey was written by", "Odyssey", "Homer", "author"),
    ("Complete this sentence with one word: The Divine Comedy was written by", "Comedy", "Dante", "author"),
    ("Complete this sentence with one word: Don Quixote was written by", "ote", "Miguel", "author"),
    # Landmarks (subject = landmark name, answer = location)
    ("Complete this sentence with one word: The Eiffel Tower is in", "Tower", "Paris", "landmark"),
    ("Complete this sentence with one word: The Statue of Liberty is in", "Liberty", "New", "landmark"),
    ("Complete this sentence with one word: The Colosseum is in", "osseum", "Rome", "landmark"),
    ("Complete this sentence with one word: The Taj Mahal is in", "Mahal", "India", "landmark"),
    ("Complete this sentence with one word: The Great Wall is in", "Wall", "China", "landmark"),
    ("Complete this sentence with one word: Big Ben is in", "Ben", "London", "landmark"),
    ("Complete this sentence with one word: The Sydney Opera House is in", "House", "Sydney", "landmark"),
    ("Complete this sentence with one word: Machu Picchu is in", "chu", "Peru", "landmark"),
    # Opposites (subject = adjective, answer = opposite adjective)
    ("Complete this sentence with one word: The opposite of hot is", "hot", "cold", "opposite"),
    ("Complete this sentence with one word: The opposite of big is", "big", "small", "opposite"),
    ("Complete this sentence with one word: The opposite of fast is", "fast", "slow", "opposite"),
    ("Complete this sentence with one word: The opposite of happy is", "happy", "sad", "opposite"),
    ("Complete this sentence with one word: The opposite of light is", "light", "dark", "opposite"),
    ("Complete this sentence with one word: The opposite of rich is", "rich", "poor", "opposite"),
    ("Complete this sentence with one word: The opposite of up is", "up", "down", "opposite"),
    ("Complete this sentence with one word: The opposite of wet is", "wet", "dry", "opposite"),
]

CATEGORIES = ["capital", "element", "author", "landmark", "opposite"]
CATEGORY_COLORS = {
    "capital": "#e74c3c",
    "element": "#3498db",
    "author": "#2ecc71",
    "landmark": "#f39c12",
    "opposite": "#9b59b6",
}


def find_subject_position(token_labels, subject_substring):
    """Return the last token position whose label contains the subject substring."""
    for i in range(len(token_labels) - 1, -1, -1):
        if subject_substring.lower() in token_labels[i].lower():
            return i
    raise ValueError(f"Subject substring {subject_substring!r} not found in tokens: {token_labels}")


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Compute N×N cosine similarity matrix."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normed = vectors / np.clip(norms, 1e-12, None)
    return normed @ normed.T


def cluster_purity(labels_true, labels_pred):
    """Return the highest achievable accuracy after relabeling clusters."""
    n = len(labels_true)
    contingency = {}
    for t, p in zip(labels_true, labels_pred):
        contingency[(t, p)] = contingency.get((t, p), 0) + 1
    # Count max-matching for each cluster
    clusters = set(labels_pred)
    total_correct = 0
    for c in clusters:
        labels_in_cluster = [t for t, p in zip(labels_true, labels_pred) if p == c]
        if labels_in_cluster:
            most_common_count = max(labels_in_cluster.count(l) for l in set(labels_in_cluster))
            total_correct += most_common_count
    return total_correct / n


def extract_fact_vectors(model, processor, tokenizer, extract_layers):
    """Run each prompt, extract fact vectors at specified layers.

    Returns dict {layer: np.array of shape [n_prompts, d_model]}, plus
    arrays of category labels and prompt metadata.
    """
    n = len(PROMPTS)
    vecs_by_layer = {L: np.zeros((n, 2560), dtype=np.float32) for L in extract_layers}
    labels = []
    meta = []  # for printing

    for idx, (prompt, subj, answer, category) in enumerate(PROMPTS):
        input_ids = _tokenize(processor, model, prompt)
        token_labels = [tokenizer.decode([t]) for t in input_ids[0].tolist()]
        subj_pos = find_subject_position(token_labels, subj)

        logits, cache = run_with_cache(model, input_ids)

        # Verify model actually predicts the expected answer
        last = logits[0, -1, :].astype(mx.float32)
        probs = mx.softmax(last)
        mx.eval(probs)
        top1_id = int(np.argmax(np.array(probs)))
        top1_tok = tokenizer.decode([top1_id])
        top1_prob = float(probs[top1_id])

        status = "OK" if answer.lower() in top1_tok.lower() else "MISS"

        for L in extract_layers:
            v = cache[f"blocks.{L}.resid_post"][0, subj_pos, :].astype(mx.float32)
            mx.eval(v)
            vecs_by_layer[L][idx] = np.array(v)

        labels.append(category)
        meta.append((prompt, subj, answer, top1_tok, top1_prob, status, subj_pos,
                     token_labels[subj_pos]))

        cat_idx = CATEGORIES.index(category)
        print(f"  [{status:>4s}] {category:>9s} #{cat_idx + 1}: subj pos {subj_pos:>2} ({token_labels[subj_pos]!r}) "
              f"→ pred {top1_tok!r:15s} p={top1_prob:.3f}")

    return vecs_by_layer, np.array(labels), meta


def analyze_layer(vecs, labels, layer, ax_sim, ax_pca, title_suffix=""):
    """Produce similarity heatmap and PCA projection for one layer's vectors.

    Also prints cluster purity and intra/inter-category similarity stats.
    """
    n = len(vecs)
    sim = cosine_similarity_matrix(vecs)

    # Reorder by category for block-diagonal visualization
    order = np.argsort([CATEGORIES.index(l) for l in labels])
    sim_ord = sim[np.ix_(order, order)]
    labels_ord = labels[order]

    # Heatmap
    im = ax_sim.imshow(sim_ord, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax_sim.set_title(f"Layer {layer} — cosine similarity {title_suffix}", fontsize=11)
    # Draw category boundaries
    prev = None
    for i, l in enumerate(labels_ord):
        if l != prev:
            ax_sim.axhline(i - 0.5, color="black", linewidth=0.7)
            ax_sim.axvline(i - 0.5, color="black", linewidth=0.7)
            prev = l
    ax_sim.set_xticks([])
    ax_sim.set_yticks([])
    plt.colorbar(im, ax=ax_sim, shrink=0.8)

    # PCA projection
    pca = PCA(n_components=2)
    proj = pca.fit_transform(vecs)
    for cat in CATEGORIES:
        mask = labels == cat
        ax_pca.scatter(proj[mask, 0], proj[mask, 1], c=CATEGORY_COLORS[cat],
                       label=cat, s=40, alpha=0.85, edgecolors="black", linewidths=0.5)
    ax_pca.set_title(f"Layer {layer} — PCA (var: {pca.explained_variance_ratio_.sum():.1%})",
                     fontsize=11)
    ax_pca.grid(True, alpha=0.3)
    if layer == EXTRACT_LAYERS[-1]:
        ax_pca.legend(loc="best", fontsize=8)

    # Stats
    intra_sims = []
    inter_sims = []
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                intra_sims.append(sim[i, j])
            else:
                inter_sims.append(sim[i, j])
    intra_mean = np.mean(intra_sims)
    inter_mean = np.mean(inter_sims)
    separation = intra_mean - inter_mean

    # K-means purity
    km = KMeans(n_clusters=len(CATEGORIES), n_init=10, random_state=42)
    pred = km.fit_predict(vecs)
    label_to_int = {c: i for i, c in enumerate(CATEGORIES)}
    true_ints = np.array([label_to_int[l] for l in labels])
    purity = cluster_purity(labels, pred)
    try:
        sil = silhouette_score(vecs, true_ints, metric="cosine")
    except Exception:
        sil = float("nan")

    print(f"\nLayer {layer} statistics:")
    print(f"  intra-category mean cosine: {intra_mean:+.4f}")
    print(f"  inter-category mean cosine: {inter_mean:+.4f}")
    print(f"  separation:                 {separation:+.4f}")
    print(f"  k-means purity (k=5):       {purity:.3f} "
          f"({'chance' if len(CATEGORIES) == 5 else ''}= 0.200)")
    print(f"  silhouette (cosine, ground-truth labels): {sil:+.4f}")

    return intra_mean, inter_mean, purity


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading model...")
    model, processor = load_model()
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    print(f"\nExtracting fact vectors from {len(PROMPTS)} prompts at layers {EXTRACT_LAYERS}...\n")
    vecs_by_layer, labels, meta = extract_fact_vectors(model, processor, tokenizer, EXTRACT_LAYERS)

    # Plot grid: 3 layers × 2 panels (similarity heatmap + PCA)
    fig, axes = plt.subplots(len(EXTRACT_LAYERS), 2, figsize=(14, 5 * len(EXTRACT_LAYERS)))
    stats = {}
    for row, L in enumerate(EXTRACT_LAYERS):
        ax_sim = axes[row, 0]
        ax_pca = axes[row, 1]
        title_suffix = f"(subject pos, ordered by category)"
        stats[L] = analyze_layer(vecs_by_layer[L], labels, L, ax_sim, ax_pca, title_suffix)

    fig.suptitle("Fact vector geometry in Gemma 4 E4B — 40 prompts, 5 categories",
                 fontsize=13)
    plt.tight_layout()
    out_path = OUT_DIR / "fact_vector_geometry.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")

    # Extra experiment: what's the nearest neighbor of each fact vector?
    print(f"\n{'=' * 60}")
    print("Nearest-neighbor analysis at layer 15 (engine room)")
    print(f"{'=' * 60}\n")

    L = 15
    vecs = vecs_by_layer[L]
    sim = cosine_similarity_matrix(vecs)
    np.fill_diagonal(sim, -np.inf)  # exclude self

    correct_neighbors = 0
    for i in range(len(PROMPTS)):
        nn = int(np.argmax(sim[i]))
        own_cat = labels[i]
        nn_cat = labels[nn]
        match = "OK" if own_cat == nn_cat else "  "
        if own_cat == nn_cat:
            correct_neighbors += 1
        subj_i = PROMPTS[i][1]
        subj_nn = PROMPTS[nn][1]
        print(f"  {match}  [{own_cat:>9s}] {subj_i:>12s} → NN: [{nn_cat:>9s}] {subj_nn:>12s}  "
              f"(cos={sim[i, nn]:.3f})")

    nn_purity = correct_neighbors / len(PROMPTS)
    print(f"\n  NN same-category hit rate: {nn_purity:.1%} "
          f"({correct_neighbors}/{len(PROMPTS)}, chance = 17.9%)")


if __name__ == "__main__":
    main()
