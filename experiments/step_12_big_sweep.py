"""Big sweep: 12 categories × 8 prompts, with statistical baselines.

Scales finding 10-11 from 5 categories / 40 prompts to 12 categories / ~96
prompts, including diverse relational frames: factual recall, morphology,
translation, professions, arithmetic, animal homes, color mixing.

Also adds a random-subset baseline: compute centroids of random 8-prompt
subsets (ignoring category labels) and project through the unembed. If
random centroids also decode to multilingual concept words, the effect
isn't signal. If they decode to noise/gibberish, our category centroids
carry genuine semantic structure.

Core outputs:
  - per-category centroid decoding at layer 30 (raw and mean-subtracted)
  - random-subset centroid decoding for comparison
  - k-means purity, silhouette, cross-category cosine stats
  - PCA projection colored by category

Run from project root:
    python experiments/step_12_big_sweep.py
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
from mlx_vlm.models.gemma4.language import logit_softcap  # noqa: E402

OUT_DIR = ROOT / "caches"
PROJECT_LAYER = 30  # where the category-centroid decoding is cleanest
EXTRACT_LAYERS = [15, 30]  # also capture 15 for geometry comparison

# (prompt, subject_substring, expected_answer_hint, category)
# subject_substring identifies the last content token whose residual we extract.
# answer_hint is what the model should roughly produce (substring match accepted).
PROMPTS = [
    # ---- capital (country → capital city) ----
    ("Complete this sentence with one word: The capital of France is", "France", "Paris", "capital"),
    ("Complete this sentence with one word: The capital of Japan is", "Japan", "Tokyo", "capital"),
    ("Complete this sentence with one word: The capital of Germany is", "Germany", "Berlin", "capital"),
    ("Complete this sentence with one word: The capital of Italy is", "Italy", "Rome", "capital"),
    ("Complete this sentence with one word: The capital of Spain is", "Spain", "Madrid", "capital"),
    ("Complete this sentence with one word: The capital of Russia is", "Russia", "Moscow", "capital"),
    ("Complete this sentence with one word: The capital of Egypt is", "Egypt", "Cairo", "capital"),
    ("Complete this sentence with one word: The capital of Greece is", "Greece", "Athens", "capital"),
    # ---- element (element name → chemical symbol) ----
    ("Complete this sentence with one word: The chemical symbol for gold is", "gold", "Au", "element"),
    ("Complete this sentence with one word: The chemical symbol for silver is", "silver", "Ag", "element"),
    ("Complete this sentence with one word: The chemical symbol for iron is", "iron", "Fe", "element"),
    ("Complete this sentence with one word: The chemical symbol for oxygen is", "oxygen", "O", "element"),
    ("Complete this sentence with one word: The chemical symbol for hydrogen is", "hydrogen", "H", "element"),
    ("Complete this sentence with one word: The chemical symbol for carbon is", "carbon", "C", "element"),
    ("Complete this sentence with one word: The chemical symbol for sodium is", "sodium", "Na", "element"),
    ("Complete this sentence with one word: The chemical symbol for copper is", "copper", "Cu", "element"),
    # ---- author (book → author) ----
    ("Complete this sentence with one word: Romeo and Juliet was written by", "Juliet", "Shake", "author"),
    ("Complete this sentence with one word: Pride and Prejudice was written by", "Prejudice", "Jane", "author"),
    ("Complete this sentence with one word: The Great Gatsby was written by", "Gatsby", "F", "author"),
    ("Complete this sentence with one word: Moby Dick was written by", "Dick", "Mel", "author"),
    ("Complete this sentence with one word: War and Peace was written by", "Peace", "Tol", "author"),
    ("Complete this sentence with one word: The Odyssey was written by", "Odyssey", "H", "author"),
    ("Complete this sentence with one word: The Divine Comedy was written by", "Comedy", "D", "author"),
    ("Complete this sentence with one word: Don Quixote was written by", "ote", "C", "author"),
    # ---- landmark (landmark → location) ----
    ("Complete this sentence with one word: The Eiffel Tower is in", "Tower", "Paris", "landmark"),
    ("Complete this sentence with one word: The Statue of Liberty is in", "Liberty", "New", "landmark"),
    ("Complete this sentence with one word: The Colosseum is in", "osseum", "Rome", "landmark"),
    ("Complete this sentence with one word: The Taj Mahal is in", "Mahal", "India", "landmark"),
    ("Complete this sentence with one word: The Great Wall is in", "Wall", "China", "landmark"),
    ("Complete this sentence with one word: Big Ben is in", "Ben", "London", "landmark"),
    ("Complete this sentence with one word: The Sydney Opera House is in", "House", "Sydney", "landmark"),
    ("Complete this sentence with one word: Machu Picchu is in", "chu", "Peru", "landmark"),
    # ---- opposite (word → antonym) ----
    ("Complete this sentence with one word: The opposite of hot is", "hot", "cold", "opposite"),
    ("Complete this sentence with one word: The opposite of big is", "big", "small", "opposite"),
    ("Complete this sentence with one word: The opposite of fast is", "fast", "slow", "opposite"),
    ("Complete this sentence with one word: The opposite of happy is", "happy", "sad", "opposite"),
    ("Complete this sentence with one word: The opposite of light is", "light", "dark", "opposite"),
    ("Complete this sentence with one word: The opposite of rich is", "rich", "poor", "opposite"),
    ("Complete this sentence with one word: The opposite of up is", "up", "down", "opposite"),
    ("Complete this sentence with one word: The opposite of wet is", "wet", "dry", "opposite"),
    # ---- past_tense (verb → past tense) ----
    ("Complete this sentence with one word: The past tense of run is", "run", "ran", "past_tense"),
    ("Complete this sentence with one word: The past tense of eat is", "eat", "ate", "past_tense"),
    ("Complete this sentence with one word: The past tense of go is", "go", "went", "past_tense"),
    ("Complete this sentence with one word: The past tense of see is", "see", "saw", "past_tense"),
    ("Complete this sentence with one word: The past tense of take is", "take", "took", "past_tense"),
    ("Complete this sentence with one word: The past tense of give is", "give", "gave", "past_tense"),
    ("Complete this sentence with one word: The past tense of write is", "write", "wrote", "past_tense"),
    ("Complete this sentence with one word: The past tense of sing is", "sing", "sang", "past_tense"),
    # ---- plural (singular → plural) ----
    ("Complete this sentence with one word: The plural of child is", "child", "children", "plural"),
    ("Complete this sentence with one word: The plural of mouse is", "mouse", "mice", "plural"),
    ("Complete this sentence with one word: The plural of goose is", "goose", "geese", "plural"),
    ("Complete this sentence with one word: The plural of foot is", "foot", "feet", "plural"),
    ("Complete this sentence with one word: The plural of tooth is", "tooth", "teeth", "plural"),
    ("Complete this sentence with one word: The plural of man is", "man", "men", "plural"),
    ("Complete this sentence with one word: The plural of woman is", "woman", "women", "plural"),
    ("Complete this sentence with one word: The plural of person is", "person", "people", "plural"),
    # ---- french_translation (English → French) ----
    ("Complete this sentence with one word: The French word for house is", "house", "maison", "french"),
    ("Complete this sentence with one word: The French word for cat is", "cat", "chat", "french"),
    ("Complete this sentence with one word: The French word for dog is", "dog", "chien", "french"),
    ("Complete this sentence with one word: The French word for water is", "water", "eau", "french"),
    ("Complete this sentence with one word: The French word for bread is", "bread", "pain", "french"),
    ("Complete this sentence with one word: The French word for sun is", "sun", "soleil", "french"),
    ("Complete this sentence with one word: The French word for book is", "book", "livre", "french"),
    ("Complete this sentence with one word: The French word for moon is", "moon", "lune", "french"),
    # ---- profession (description → job title) ----
    ("Complete this sentence with one word: A person who flies airplanes is a", "airplanes", "pilot", "profession"),
    ("Complete this sentence with one word: A person who treats sick people is a", "people", "doctor", "profession"),
    ("Complete this sentence with one word: A person who teaches students is a", "students", "teacher", "profession"),
    ("Complete this sentence with one word: A person who writes books is an", "books", "author", "profession"),
    ("Complete this sentence with one word: A person who designs buildings is an", "buildings", "architect", "profession"),
    ("Complete this sentence with one word: A person who paints pictures is a", "pictures", "painter", "profession"),
    ("Complete this sentence with one word: A person who cooks meals is a", "meals", "chef", "profession"),
    ("Complete this sentence with one word: A person who defends clients is a", "clients", "lawyer", "profession"),
    # ---- animal_home (animal → habitat) ----
    ("Complete this sentence with one word: A bee lives in a", "bee", "hive", "animal_home"),
    ("Complete this sentence with one word: A bird lives in a", "bird", "nest", "animal_home"),
    ("Complete this sentence with one word: A bear lives in a", "bear", "den", "animal_home"),
    ("Complete this sentence with one word: A rabbit lives in a", "rabbit", "burrow", "animal_home"),
    ("Complete this sentence with one word: A spider lives in a", "spider", "web", "animal_home"),
    ("Complete this sentence with one word: A horse lives in a", "horse", "barn", "animal_home"),
    ("Complete this sentence with one word: A pig lives in a", "pig", "pen", "animal_home"),
    ("Complete this sentence with one word: A fox lives in a", "fox", "den", "animal_home"),
    # ---- color_mix (two colors → result) ----
    ("Complete this sentence with one word: Red and yellow mixed together make", "yellow", "orange", "color_mix"),
    ("Complete this sentence with one word: Blue and yellow mixed together make", "yellow", "green", "color_mix"),
    ("Complete this sentence with one word: Red and blue mixed together make", "blue", "purple", "color_mix"),
    ("Complete this sentence with one word: Black and white mixed together make", "white", "gray", "color_mix"),
    ("Complete this sentence with one word: Red and white mixed together make", "white", "pink", "color_mix"),
    ("Complete this sentence with one word: Yellow and blue mixed together make", "blue", "green", "color_mix"),
    ("Complete this sentence with one word: Blue and red mixed together make", "red", "purple", "color_mix"),
    ("Complete this sentence with one word: White and black mixed together make", "black", "gray", "color_mix"),
    # ---- math (arithmetic) ----
    ("Complete this sentence with one word: Two plus two equals", "two", "four", "math"),
    ("Complete this sentence with one word: Three plus three equals", "three", "six", "math"),
    ("Complete this sentence with one word: Four plus four equals", "four", "eight", "math"),
    ("Complete this sentence with one word: Five plus five equals", "five", "ten", "math"),
    ("Complete this sentence with one word: Six plus six equals", "six", "twelve", "math"),
    ("Complete this sentence with one word: Seven plus seven equals", "seven", "fourteen", "math"),
    ("Complete this sentence with one word: Eight plus eight equals", "eight", "sixteen", "math"),
    ("Complete this sentence with one word: Nine plus nine equals", "nine", "eighteen", "math"),
]

CATEGORIES = ["capital", "element", "author", "landmark", "opposite",
              "past_tense", "plural", "french", "profession", "animal_home",
              "color_mix", "math"]

# Visually distinct colors for 12 categories
COLOR_MAP = {
    "capital": "#e41a1c", "element": "#377eb8", "author": "#4daf4a",
    "landmark": "#ff7f00", "opposite": "#984ea3", "past_tense": "#a65628",
    "plural": "#f781bf", "french": "#ffff33", "profession": "#999999",
    "animal_home": "#66c2a5", "color_mix": "#8dd3c7", "math": "#fb8072",
}


def find_subject_position(token_labels, subject_substring):
    for i in range(len(token_labels) - 1, -1, -1):
        if subject_substring.lower() in token_labels[i].lower():
            return i
    raise ValueError(f"{subject_substring!r} not found in {token_labels}")


def project_to_logits(model, resid: mx.array) -> mx.array:
    lm = model.language_model
    tm = lm.model
    h = tm.norm(resid)
    logits = tm.embed_tokens.as_linear(h)
    if lm.final_logit_softcapping is not None:
        logits = logit_softcap(lm.final_logit_softcapping, logits)
    return logits


def top_k_tokens_from_vector(model, tokenizer, vec_np, k=15):
    v = mx.array(vec_np[None, None, :], dtype=mx.bfloat16)
    logits = project_to_logits(model, v)
    last = logits[0, 0, :].astype(mx.float32)
    probs = mx.softmax(last)
    mx.eval(probs)
    probs_np = np.array(probs)
    top_idx = np.argsort(-probs_np)[:k]
    return [(tokenizer.decode([int(i)]), float(probs_np[int(i)])) for i in top_idx]


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def extract_fact_vectors(model, processor, tokenizer, layers):
    n = len(PROMPTS)
    vecs = {L: np.zeros((n, 2560), dtype=np.float32) for L in layers}
    labels = []
    answers = []
    kept_indices = []

    for idx, (prompt, subj, answer_hint, category) in enumerate(PROMPTS):
        input_ids = _tokenize(processor, model, prompt)
        token_labels = [tokenizer.decode([t]) for t in input_ids[0].tolist()]
        try:
            subj_pos = find_subject_position(token_labels, subj)
        except ValueError as e:
            print(f"  [FAIL ] [{category:>11s}] {subj!r} — skip: {e}")
            continue

        logits, cache = run_with_cache(model, input_ids)
        last = logits[0, -1, :].astype(mx.float32)
        probs = mx.softmax(last)
        mx.eval(probs)
        probs_np = np.array(probs)
        top1_id = int(np.argmax(probs_np))
        top1_tok = tokenizer.decode([top1_id]).strip()
        top1_prob = float(probs_np[top1_id])

        # Loose match: top-1 token shares a prefix with the expected answer hint.
        match = (answer_hint.lower() in top1_tok.lower()
                 or top1_tok.lower() in answer_hint.lower())
        status = "OK" if match else "MISS"

        for L in layers:
            v = cache[f"blocks.{L}.resid_post"][0, subj_pos, :].astype(mx.float32)
            mx.eval(v)
            vecs[L][idx] = np.array(v)

        labels.append(category)
        answers.append(top1_tok)
        kept_indices.append(idx)
        print(f"  [{status:>4s}] [{category:>11s}] pos {subj_pos:>2} ({token_labels[subj_pos]!r:<15s}) "
              f"→ {top1_tok!r:15s} p={top1_prob:.3f}")

    # Trim to kept indices
    n_kept = len(kept_indices)
    vecs_trim = {L: vecs[L][kept_indices] for L in layers}
    return vecs_trim, np.array(labels), np.array(answers), kept_indices


def cluster_purity(labels_true, labels_pred):
    clusters = set(labels_pred)
    total_correct = 0
    for c in clusters:
        in_cluster = [t for t, p in zip(labels_true, labels_pred) if p == c]
        if in_cluster:
            total_correct += max(in_cluster.count(l) for l in set(in_cluster))
    return total_correct / len(labels_true)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading model...")
    model, processor = load_model()
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    print(f"\nExtracting fact vectors from {len(PROMPTS)} prompts at layers {EXTRACT_LAYERS}...\n")
    vecs_by_layer, labels, answers, kept = extract_fact_vectors(
        model, processor, tokenizer, EXTRACT_LAYERS
    )
    n = len(labels)
    print(f"\nKept {n} of {len(PROMPTS)} prompts.")
    for cat in CATEGORIES:
        count = int(np.sum(labels == cat))
        print(f"  {cat:>11s}: {count} prompts")

    # ----- Cluster quality at layer 30 -----
    print(f"\n{'=' * 70}")
    print(f"Cluster quality at layer {PROJECT_LAYER}")
    print(f"{'=' * 70}")
    vecs = vecs_by_layer[PROJECT_LAYER]

    # K-means with k = number of categories present
    present_cats = list(dict.fromkeys(labels))
    k = len(present_cats)
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    pred = km.fit_predict(vecs)
    purity = cluster_purity(labels, pred)
    cat_to_int = {c: i for i, c in enumerate(present_cats)}
    true_ints = np.array([cat_to_int[l] for l in labels])
    sil = silhouette_score(vecs, true_ints, metric="cosine")

    print(f"  n = {n}, k = {k}")
    print(f"  k-means purity: {purity:.3f} (chance = {1 / k:.3f})")
    print(f"  silhouette (cosine, ground-truth labels): {sil:+.4f}")

    # Intra vs inter category cosine
    intra_sims = []
    inter_sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = cos(vecs[i], vecs[j])
            if labels[i] == labels[j]:
                intra_sims.append(sim)
            else:
                inter_sims.append(sim)
    print(f"  intra-cat cos: {np.mean(intra_sims):+.4f}")
    print(f"  inter-cat cos: {np.mean(inter_sims):+.4f}")
    print(f"  separation:    {np.mean(intra_sims) - np.mean(inter_sims):+.4f}")

    # Nearest-neighbor same-category hit rate
    sim_mat = np.array([[cos(vecs[i], vecs[j]) for j in range(n)] for i in range(n)])
    np.fill_diagonal(sim_mat, -np.inf)
    nn_correct = 0
    for i in range(n):
        nn = int(np.argmax(sim_mat[i]))
        if labels[nn] == labels[i]:
            nn_correct += 1
    nn_hit_rate = nn_correct / n
    print(f"  NN same-category hit rate: {nn_hit_rate:.3f} ({nn_correct}/{n}, chance = {(n // k - 1) / (n - 1):.3f})")

    # ----- PCA -----
    pca = PCA(n_components=2)
    proj = pca.fit_transform(vecs)

    fig, ax = plt.subplots(figsize=(11, 9))
    for cat in present_cats:
        mask = labels == cat
        ax.scatter(proj[mask, 0], proj[mask, 1], c=COLOR_MAP.get(cat, "#333333"),
                   label=f"{cat} ({int(mask.sum())})", s=60, alpha=0.85,
                   edgecolors="black", linewidths=0.5)
    ax.set_title(f"Fact-vector geometry at layer {PROJECT_LAYER} — "
                 f"{n} prompts, {k} categories (PCA var {pca.explained_variance_ratio_.sum():.1%})",
                 fontsize=11)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = OUT_DIR / "big_sweep_pca.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")

    # ----- Category centroid decoding -----
    print(f"\n{'=' * 70}")
    print(f"Category centroid decoding at layer {PROJECT_LAYER} "
          f"(mean-subtracted, top-8 tokens)")
    print(f"{'=' * 70}\n")

    overall_mean = vecs.mean(axis=0)
    centroid_top_tokens = {}
    for cat in present_cats:
        mask = labels == cat
        centroid = vecs[mask].mean(axis=0)
        mean_sub = centroid - overall_mean
        top = top_k_tokens_from_vector(model, tokenizer, mean_sub, k=8)
        centroid_top_tokens[cat] = [t for t, p in top]
        probs = [p for t, p in top]
        print(f"  [{cat:>11s}] " + "  ".join(f"{t!r}({p:.3f})" for t, p in top))

    # ----- Random-subset baseline -----
    print(f"\n{'=' * 70}")
    print("Random-subset baseline (centroids of random 8-prompt subsets)")
    print(f"{'=' * 70}")
    print("If random centroids also decode to multilingual concept words,")
    print("our category effect isn't real signal. If they decode to noise,")
    print("the category centroids carry genuine semantic structure.\n")

    np.random.seed(42)
    n_random = 10
    random_top_tokens = []
    for r in range(n_random):
        idx = np.random.choice(n, size=8, replace=False)
        rand_centroid = vecs[idx].mean(axis=0)
        rand_mean_sub = rand_centroid - overall_mean
        top = top_k_tokens_from_vector(model, tokenizer, rand_mean_sub, k=8)
        random_top_tokens.append([t for t, p in top])
        print(f"  [random #{r+1:>2d}]  " + "  ".join(f"{t!r}({p:.3f})" for t, p in top))

    # Cross-centroid token overlap: do category centroids have distinct decoded
    # token sets, or are they all decoding to the same "average-prompt" token set?
    print(f"\n{'=' * 70}")
    print("Token-set uniqueness analysis")
    print(f"{'=' * 70}")
    print("For each pair of (category) centroids, measure Jaccard overlap")
    print("of their top-8 decoded token sets. Then compare against random-vs-random.\n")

    cat_names = list(centroid_top_tokens.keys())
    cat_overlaps = []
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

    print(f"  Category×category Jaccard overlap: mean {np.mean(cat_overlaps):.3f}, "
          f"max {np.max(cat_overlaps):.3f}")
    print(f"  Random×random Jaccard overlap:     mean {np.mean(rand_overlaps):.3f}, "
          f"max {np.max(rand_overlaps):.3f}")
    print(f"\n  If category overlap << random overlap, each category centroid")
    print("  is decoding to a distinctive semantic concept (good).")
    print("  If they're similar, the centroids are not very distinctive.")

    # ----- Similarity heatmap grouped by category -----
    order = np.argsort([present_cats.index(l) for l in labels])
    sim_ord = sim_mat[np.ix_(order, order)]
    sim_ord[np.isinf(sim_ord)] = 1.0  # restore the diagonal for display

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(sim_ord, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    labels_ord = labels[order]
    prev = None
    boundaries = [0]
    for i, l in enumerate(labels_ord):
        if l != prev:
            if prev is not None:
                ax.axhline(i - 0.5, color="black", linewidth=0.8)
                ax.axvline(i - 0.5, color="black", linewidth=0.8)
                boundaries.append(i)
            prev = l
    boundaries.append(len(labels_ord))

    # Category labels on the sides
    mid_points = [(boundaries[i] + boundaries[i + 1]) / 2 for i in range(len(boundaries) - 1)]
    ax.set_yticks(mid_points)
    ax.set_yticklabels(present_cats, fontsize=9)
    ax.set_xticks(mid_points)
    ax.set_xticklabels(present_cats, rotation=45, ha="right", fontsize=9)
    ax.set_title(f"Cosine similarity at layer {PROJECT_LAYER} "
                 f"(n={n}, grouped by category)", fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    out_path = OUT_DIR / "big_sweep_similarity_heatmap.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
