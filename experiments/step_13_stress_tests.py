"""Stress tests for the centroid-decoding technique.

Three focused probes, anchored against the 12 big-sweep categories:

  1. TEMPLATE VARIATION: Capital prompts using 4 different phrasings. If they
     all still cluster in the capital group, the representation is about the
     relational operation (capital-lookup), not the surface template.

  2. CROSS-LINGUAL: Capital prompts in English, French, German, Spanish,
     Chinese. If they cluster with the English capital anchors, the encoding
     is language-invariant — a mirror of the multilingual output decoding.

  3. OPEN-ENDED/CREATIVE: Subjective prompts with no single correct answer.
     Do they cluster at all? Tells us whether the technique applies beyond
     structured factual recall.

Produces a PCA projection showing the 12 anchor categories plus each
stress-test group as its own color, so we can see exactly where the
stress-test prompts land.

Run from project root:
    python experiments/step_13_stress_tests.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forward import load_model, _tokenize  # noqa: E402
from hooks import run_with_cache  # noqa: E402
from mlx_vlm.models.gemma4.language import logit_softcap  # noqa: E402

# Anchor prompts (the 12 big-sweep categories, for geometric context)
from experiments.step_12_big_sweep import PROMPTS as ANCHOR_PROMPTS, COLOR_MAP  # noqa: E402

OUT_DIR = ROOT / "caches"
PROJECT_LAYER = 30

# -----------------------------------------------------------------------------
# Stress test 1: TEMPLATE VARIATION (capitals, 4 templates × 4 countries)
# -----------------------------------------------------------------------------
TEMPLATE_VAR_PROMPTS = [
    # template 1: original
    ("Complete this sentence with one word: The capital of France is", "France", "Paris", "capital_tmpl1"),
    ("Complete this sentence with one word: The capital of Japan is", "Japan", "Tokyo", "capital_tmpl1"),
    ("Complete this sentence with one word: The capital of Germany is", "Germany", "Berlin", "capital_tmpl1"),
    ("Complete this sentence with one word: The capital of Italy is", "Italy", "Rome", "capital_tmpl1"),
    # template 2: question form
    ("What is the capital city of France? Answer in one word:", "France", "Paris", "capital_tmpl2"),
    ("What is the capital city of Japan? Answer in one word:", "Japan", "Tokyo", "capital_tmpl2"),
    ("What is the capital city of Germany? Answer in one word:", "Germany", "Berlin", "capital_tmpl2"),
    ("What is the capital city of Italy? Answer in one word:", "Italy", "Rome", "capital_tmpl2"),
    # template 3: possessive form
    ("France's capital city is called, in one word,", "France", "Paris", "capital_tmpl3"),
    ("Japan's capital city is called, in one word,", "Japan", "Tokyo", "capital_tmpl3"),
    ("Germany's capital city is called, in one word,", "Germany", "Berlin", "capital_tmpl3"),
    ("Italy's capital city is called, in one word,", "Italy", "Rome", "capital_tmpl3"),
    # template 4: paraphrased
    ("The administrative center of France is named, in one word,", "France", "Paris", "capital_tmpl4"),
    ("The administrative center of Japan is named, in one word,", "Japan", "Tokyo", "capital_tmpl4"),
    ("The administrative center of Germany is named, in one word,", "Germany", "Berlin", "capital_tmpl4"),
    ("The administrative center of Italy is named, in one word,", "Italy", "Rome", "capital_tmpl4"),
]

# -----------------------------------------------------------------------------
# Stress test 2: CROSS-LINGUAL (capital question, 5 languages × 4 countries)
# -----------------------------------------------------------------------------
CROSS_LINGUAL_PROMPTS = [
    # English (same as original template but without the "Complete" prefix to
    # match the other languages better — we'll include both for comparison)
    ("The capital of France is", "France", "Paris", "lang_en"),
    ("The capital of Japan is", "Japan", "Tokyo", "lang_en"),
    ("The capital of Germany is", "Germany", "Berlin", "lang_en"),
    ("The capital of Italy is", "Italy", "Rome", "lang_en"),
    # French
    ("La capitale de la France est", "France", "Paris", "lang_fr"),
    ("La capitale du Japon est", "Japon", "Tokyo", "lang_fr"),
    ("La capitale de l'Allemagne est", "Allemagne", "Berlin", "lang_fr"),
    ("La capitale de l'Italie est", "Italie", "Rome", "lang_fr"),
    # German
    ("Die Hauptstadt Frankreichs ist", "Frankreichs", "Paris", "lang_de"),
    ("Die Hauptstadt Japans ist", "Japans", "Tokio", "lang_de"),
    ("Die Hauptstadt Deutschlands ist", "Deutschlands", "Berlin", "lang_de"),
    ("Die Hauptstadt Italiens ist", "Italiens", "Rom", "lang_de"),
    # Spanish
    ("La capital de Francia es", "Francia", "París", "lang_es"),
    ("La capital de Japón es", "Japón", "Tokio", "lang_es"),
    ("La capital de Alemania es", "Alemania", "Berlín", "lang_es"),
    ("La capital de Italia es", "Italia", "Roma", "lang_es"),
    # Chinese
    ("法国的首都是", "法国", "巴黎", "lang_zh"),
    ("日本的首都是", "日本", "东京", "lang_zh"),
    ("德国的首都是", "德国", "柏林", "lang_zh"),
    ("意大利的首都是", "意大利", "罗马", "lang_zh"),
]

# -----------------------------------------------------------------------------
# Stress test 3: OPEN-ENDED / CREATIVE (no single correct answer)
# -----------------------------------------------------------------------------
CREATIVE_PROMPTS = [
    # subjective/preference (one-word answer expected but many valid options)
    ("Complete this sentence with one word: The best way to spend a Sunday is", "Sunday", "???", "creative_pref"),
    ("Complete this sentence with one word: The most beautiful color is", "color", "???", "creative_pref"),
    ("Complete this sentence with one word: The perfect meal is", "meal", "???", "creative_pref"),
    ("Complete this sentence with one word: A good name for a cat is", "cat", "???", "creative_pref"),
    # metaphorical / sensory crossings
    ("Complete this sentence with one word: The sound of rain makes me feel", "rain", "???", "creative_meta"),
    ("Complete this sentence with one word: The color of jealousy is", "jealousy", "???", "creative_meta"),
    ("Complete this sentence with one word: The taste of sadness is", "sadness", "???", "creative_meta"),
    ("Complete this sentence with one word: The smell of summer is", "summer", "???", "creative_meta"),
]


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


def top_k_tokens_from_vector(model, tokenizer, vec_np, k=8):
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


def extract_vectors(model, processor, tokenizer, prompt_list):
    """Run each prompt, extract fact vector at layer PROJECT_LAYER."""
    n = len(prompt_list)
    vecs = np.zeros((n, 2560), dtype=np.float32)
    labels = []
    tops = []
    kept = []
    for idx, (prompt, subj, ans_hint, category) in enumerate(prompt_list):
        input_ids = _tokenize(processor, model, prompt)
        token_labels = [tokenizer.decode([t]) for t in input_ids[0].tolist()]
        try:
            subj_pos = find_subject_position(token_labels, subj)
        except ValueError as e:
            print(f"  [FAIL] [{category}] {subj!r}: {e}")
            continue

        logits, cache = run_with_cache(model, input_ids)
        last = logits[0, -1, :].astype(mx.float32)
        probs = mx.softmax(last)
        mx.eval(probs)
        probs_np = np.array(probs)
        top1_id = int(np.argmax(probs_np))
        top1_tok = tokenizer.decode([top1_id]).strip()
        top1_prob = float(probs_np[top1_id])

        v = cache[f"blocks.{PROJECT_LAYER}.resid_post"][0, subj_pos, :].astype(mx.float32)
        mx.eval(v)
        vecs[idx] = np.array(v)
        labels.append(category)
        tops.append(top1_tok)
        kept.append(idx)
        short_prompt = prompt[:50] + ("..." if len(prompt) > 50 else "")
        print(f"  [{category:>16s}] pos {subj_pos:>2} {short_prompt:<55s} → {top1_tok!r:15s} p={top1_prob:.2f}")

    vecs_trim = vecs[kept]
    return vecs_trim, np.array(labels), tops


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading model...")
    model, processor = load_model()
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    print(f"\n=== Extracting {len(ANCHOR_PROMPTS)} anchor prompts ===\n")
    anchor_vecs, anchor_labels, _ = extract_vectors(
        model, processor, tokenizer, ANCHOR_PROMPTS
    )

    print(f"\n=== Extracting {len(TEMPLATE_VAR_PROMPTS)} template-variation prompts ===\n")
    tmpl_vecs, tmpl_labels, _ = extract_vectors(
        model, processor, tokenizer, TEMPLATE_VAR_PROMPTS
    )

    print(f"\n=== Extracting {len(CROSS_LINGUAL_PROMPTS)} cross-lingual prompts ===\n")
    cl_vecs, cl_labels, _ = extract_vectors(
        model, processor, tokenizer, CROSS_LINGUAL_PROMPTS
    )

    print(f"\n=== Extracting {len(CREATIVE_PROMPTS)} creative/open-ended prompts ===\n")
    cr_vecs, cr_labels, _ = extract_vectors(
        model, processor, tokenizer, CREATIVE_PROMPTS
    )

    # -------------------------------------------------------------------------
    # Analysis
    # -------------------------------------------------------------------------

    # Combine everything for PCA
    all_vecs = np.vstack([anchor_vecs, tmpl_vecs, cl_vecs, cr_vecs])
    all_labels = np.concatenate([anchor_labels, tmpl_labels, cl_labels, cr_labels])

    # Compute distances from each stress-test prompt to the capital centroid
    capital_centroid = anchor_vecs[anchor_labels == "capital"].mean(axis=0)
    # Also distances to all anchor category centroids
    anchor_cats = list(dict.fromkeys(anchor_labels))
    anchor_centroids = {c: anchor_vecs[anchor_labels == c].mean(axis=0) for c in anchor_cats}

    def nearest_anchor(vec):
        sims = [(c, cos(vec, anchor_centroids[c])) for c in anchor_cats]
        sims.sort(key=lambda x: -x[1])
        return sims[0]  # (category, similarity)

    # ---- Template variation analysis ----
    print(f"\n{'=' * 70}")
    print("TEMPLATE VARIATION: where does each template's vector cluster?")
    print(f"{'=' * 70}")
    print("\nFor each template-variation prompt, find nearest anchor category:\n")
    tmpl_groups = {}
    for v, l in zip(tmpl_vecs, tmpl_labels):
        best_cat, best_sim = nearest_anchor(v)
        tmpl_groups.setdefault(l, []).append((best_cat, best_sim))

    for tmpl in sorted(tmpl_groups):
        hits = tmpl_groups[tmpl]
        n_capital = sum(1 for c, _ in hits if c == "capital")
        print(f"  {tmpl}: {n_capital}/{len(hits)} nearest-anchor = capital")
        for c, s in hits:
            marker = "✓" if c == "capital" else "✗"
            print(f"    {marker} nearest={c:>12s}  cos={s:.4f}")

    # Pairwise cosine between template-variation groups, vs within-anchor-capital
    print("\nPairwise cosine comparisons:")
    for tmpl in sorted(tmpl_groups):
        vs = tmpl_vecs[tmpl_labels == tmpl]
        intra = [cos(vs[i], vs[j]) for i in range(len(vs)) for j in range(i + 1, len(vs))]
        to_anchor = [cos(v, capital_centroid) for v in vs]
        print(f"  {tmpl}: intra-template cos = {np.mean(intra):.4f}, cos-to-capital-centroid = {np.mean(to_anchor):.4f}")

    cap_anchor = anchor_vecs[anchor_labels == "capital"]
    intra_anchor = [cos(cap_anchor[i], cap_anchor[j])
                    for i in range(len(cap_anchor))
                    for j in range(i + 1, len(cap_anchor))]
    print(f"  [anchor capital intra-cosine baseline: {np.mean(intra_anchor):.4f}]")

    # ---- Cross-lingual analysis ----
    print(f"\n{'=' * 70}")
    print("CROSS-LINGUAL: does capital-question in each language land in capital cluster?")
    print(f"{'=' * 70}\n")
    cl_groups = {}
    for v, l in zip(cl_vecs, cl_labels):
        best_cat, best_sim = nearest_anchor(v)
        cl_groups.setdefault(l, []).append((best_cat, best_sim))

    for lang in sorted(cl_groups):
        hits = cl_groups[lang]
        n_capital = sum(1 for c, _ in hits if c == "capital")
        print(f"  {lang}: {n_capital}/{len(hits)} nearest-anchor = capital")
        for c, s in hits:
            marker = "✓" if c == "capital" else "✗"
            print(f"    {marker} nearest={c:>12s}  cos={s:.4f}")

    # Centroids of each language group, distance to English capital anchor
    print("\nCentroid-to-capital-anchor distances:")
    for lang in sorted(cl_groups):
        lvecs = cl_vecs[cl_labels == lang]
        centroid = lvecs.mean(axis=0)
        cos_to_cap = cos(centroid, capital_centroid)
        intra = [cos(lvecs[i], lvecs[j])
                 for i in range(len(lvecs))
                 for j in range(i + 1, len(lvecs))]
        print(f"  {lang}: cos(centroid, capital_anchor) = {cos_to_cap:.4f}, intra-lang cos = {np.mean(intra):.4f}")

    # Also: centroid projection for each language
    print("\nCross-lingual centroid decoding (mean-subtracted, top-6 tokens):")
    all_anchor_mean = anchor_vecs.mean(axis=0)
    for lang in sorted(cl_groups):
        lvecs = cl_vecs[cl_labels == lang]
        centroid = lvecs.mean(axis=0)
        mean_sub = centroid - all_anchor_mean
        top = top_k_tokens_from_vector(model, tokenizer, mean_sub, k=6)
        print(f"  [{lang:>8s}] " + "  ".join(f"{t!r}({p:.3f})" for t, p in top))

    # ---- Creative prompts analysis ----
    print(f"\n{'=' * 70}")
    print("CREATIVE PROMPTS: where do open-ended/subjective prompts land?")
    print(f"{'=' * 70}\n")
    cr_groups = {}
    for v, l in zip(cr_vecs, cr_labels):
        best_cat, best_sim = nearest_anchor(v)
        cr_groups.setdefault(l, []).append((best_cat, best_sim))

    for grp in sorted(cr_groups):
        print(f"  {grp}:")
        for c, s in cr_groups[grp]:
            print(f"    nearest={c:>12s}  cos={s:.4f}")

    # Creative centroid decoding
    print("\nCreative-category centroid decoding (mean-subtracted, top-6 tokens):")
    for grp in sorted(cr_groups):
        gvecs = cr_vecs[cr_labels == grp]
        centroid = gvecs.mean(axis=0)
        mean_sub = centroid - all_anchor_mean
        top = top_k_tokens_from_vector(model, tokenizer, mean_sub, k=6)
        print(f"  [{grp:>14s}] " + "  ".join(f"{t!r}({p:.3f})" for t, p in top))

    # -------------------------------------------------------------------------
    # Visualization: PCA of everything
    # -------------------------------------------------------------------------
    pca = PCA(n_components=2)
    proj = pca.fit_transform(all_vecs)

    fig, axes = plt.subplots(1, 3, figsize=(22, 8))

    # Panel 1: anchor categories + template variations
    ax = axes[0]
    n_anc = len(anchor_vecs)
    for cat in set(anchor_labels):
        mask = anchor_labels == cat
        ax.scatter(proj[:n_anc][mask, 0], proj[:n_anc][mask, 1],
                   c=COLOR_MAP.get(cat, "#666666"), label=cat if cat == "capital" else None,
                   s=40, alpha=0.5, edgecolors="black", linewidths=0.3)
    # Template variations
    start = n_anc
    end = start + len(tmpl_vecs)
    tmpl_colors = {"capital_tmpl1": "#000000", "capital_tmpl2": "#d62728",
                   "capital_tmpl3": "#1f77b4", "capital_tmpl4": "#9467bd"}
    for t, c in tmpl_colors.items():
        mask = tmpl_labels == t
        ax.scatter(proj[start:end][mask, 0], proj[start:end][mask, 1],
                   c=c, label=t, s=130, alpha=1.0, marker="*",
                   edgecolors="white", linewidths=1.5, zorder=5)
    ax.set_title("Template variation on 'capital'\n(stars = templates, dots = anchor categories, red = capital anchors)", fontsize=10)
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 2: anchor categories + cross-lingual
    ax = axes[1]
    for cat in set(anchor_labels):
        mask = anchor_labels == cat
        ax.scatter(proj[:n_anc][mask, 0], proj[:n_anc][mask, 1],
                   c=COLOR_MAP.get(cat, "#666666"), label=cat if cat == "capital" else None,
                   s=40, alpha=0.5, edgecolors="black", linewidths=0.3)
    start = n_anc + len(tmpl_vecs)
    end = start + len(cl_vecs)
    lang_colors = {"lang_en": "#000000", "lang_fr": "#d62728", "lang_de": "#1f77b4",
                   "lang_es": "#ff7f00", "lang_zh": "#9467bd"}
    for t, c in lang_colors.items():
        mask = cl_labels == t
        ax.scatter(proj[start:end][mask, 0], proj[start:end][mask, 1],
                   c=c, label=t, s=130, alpha=1.0, marker="D",
                   edgecolors="white", linewidths=1.5, zorder=5)
    ax.set_title("Cross-lingual capital prompts\n(diamonds = languages, red dots = capital anchors)", fontsize=10)
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 3: anchor categories + creative
    ax = axes[2]
    for cat in set(anchor_labels):
        mask = anchor_labels == cat
        ax.scatter(proj[:n_anc][mask, 0], proj[:n_anc][mask, 1],
                   c=COLOR_MAP.get(cat, "#666666"), label=cat,
                   s=40, alpha=0.5, edgecolors="black", linewidths=0.3)
    start = n_anc + len(tmpl_vecs) + len(cl_vecs)
    end = start + len(cr_vecs)
    cr_colors = {"creative_pref": "#000000", "creative_meta": "#d62728"}
    for t, c in cr_colors.items():
        mask = cr_labels == t
        ax.scatter(proj[start:end][mask, 0], proj[start:end][mask, 1],
                   c=c, label=t, s=130, alpha=1.0, marker="X",
                   edgecolors="white", linewidths=1.5, zorder=5)
    ax.set_title("Creative/open-ended prompts\n(Xs, against all 12 anchor categories)", fontsize=10)
    ax.legend(loc="best", fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Stress tests at layer {PROJECT_LAYER} (PCA var {pca.explained_variance_ratio_.sum():.1%})",
                 fontsize=12)
    plt.tight_layout()
    out_path = OUT_DIR / "stress_tests.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
