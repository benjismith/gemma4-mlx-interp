"""Centroid unembed projection and vector arithmetic on fact vectors.

Follow-up to finding 10, which established clean category structure in the
vocab-opaque fact vectors. Three experiments:

  1. CENTROID UNEMBED PROJECTION: Average the 8 fact vectors in each category
     to get a 'prototype'. Project the prototype through the model's final
     norm + tied unembed. Report top-K tokens. Individual vectors are vocab-
     opaque; averaging cancels per-instance noise and should amplify the
     category's shared relational frame.

  2. WITHIN-CATEGORY ARITHMETIC: For capital prompts, compute pairwise diff
     vectors v(France) - v(Japan), etc. Measure whether there's a consistent
     'between-facts' direction, or whether diffs are near-orthogonal (fact-
     specific, no linear structure).

  3. CROSS-CATEGORY SAME-ANSWER: We have natural pairs like
     (Eiffel Tower → Paris, landmark) and (France → Paris, capital) that
     differ in category but share the answer. Does category-centroid
     subtraction bring same-answer prompts closer? Tests whether the
     category-plus-answer decomposition is linear.

Run from project root:
    python experiments/step_11_centroid_and_arithmetic.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forward import load_model, _tokenize  # noqa: E402
from hooks import run_with_cache  # noqa: E402
from mlx_vlm.models.gemma4.language import logit_softcap  # noqa: E402
from experiments.step_10_fact_vector_geometry import (  # noqa: E402
    PROMPTS, CATEGORIES, find_subject_position
)

OUT_DIR = ROOT / "caches"
# Project centroids at multiple depths to see how decodability evolves
PROJECT_LAYERS = [15, 25, 30, 35, 41]


def project_to_logits(model, resid: mx.array) -> mx.array:
    """Apply final norm + tied unembed + softcap (same path the model's output uses)."""
    lm = model.language_model
    tm = lm.model
    h = tm.norm(resid)
    logits = tm.embed_tokens.as_linear(h)
    if lm.final_logit_softcapping is not None:
        logits = logit_softcap(lm.final_logit_softcapping, logits)
    return logits


def extract_vectors_at_layers(model, processor, tokenizer, layers):
    """Extract resid_post at subject positions, at multiple layers.

    Returns {layer: np.ndarray[n_prompts, d_model]}, plus labels and answers.
    """
    n = len(PROMPTS)
    vecs = {L: np.zeros((n, 2560), dtype=np.float32) for L in layers}
    labels = []
    answers = []

    for idx, (prompt, subj, answer, category) in enumerate(PROMPTS):
        input_ids = _tokenize(processor, model, prompt)
        token_labels = [tokenizer.decode([t]) for t in input_ids[0].tolist()]
        subj_pos = find_subject_position(token_labels, subj)

        _, cache = run_with_cache(model, input_ids)
        for L in layers:
            v = cache[f"blocks.{L}.resid_post"][0, subj_pos, :].astype(mx.float32)
            mx.eval(v)
            vecs[L][idx] = np.array(v)

        labels.append(category)
        answers.append(answer)

    return vecs, np.array(labels), np.array(answers)


def top_k_tokens_from_vector(model, tokenizer, vec_np, k=15):
    """Project a numpy vector through the model's output head and return top-k tokens."""
    # bf16 is what the rest of the network uses; cast appropriately.
    v = mx.array(vec_np[None, None, :], dtype=mx.bfloat16)  # [1, 1, d_model]
    logits = project_to_logits(model, v)
    last = logits[0, 0, :].astype(mx.float32)
    probs = mx.softmax(last)
    mx.eval(probs)
    probs_np = np.array(probs)
    top_idx = np.argsort(-probs_np)[:k]
    return [(tokenizer.decode([int(i)]), float(probs_np[int(i)])) for i in top_idx]


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def experiment_centroid_projection(model, tokenizer, vecs_by_layer, labels):
    """For each category, compute centroid and project through unembed. Also
    try mean-subtracted centroid (remove the overall mean first)."""
    print(f"\n{'=' * 70}")
    print("Experiment 1: centroid projection through unembed")
    print(f"{'=' * 70}")

    for L in PROJECT_LAYERS:
        vecs = vecs_by_layer[L]
        overall_mean = vecs.mean(axis=0)

        print(f"\n--- Layer {L} ---")
        for cat in CATEGORIES:
            mask = labels == cat
            centroid = vecs[mask].mean(axis=0)
            mean_subtracted = centroid - overall_mean

            raw_top = top_k_tokens_from_vector(model, tokenizer, centroid, k=10)
            ms_top = top_k_tokens_from_vector(model, tokenizer, mean_subtracted, k=10)

            print(f"\n  [{cat:>9s}] centroid top-10:")
            print("    raw:      " + "  ".join(f"{t!r}({p:.3f})" for t, p in raw_top[:6]))
            print("    mean-sub: " + "  ".join(f"{t!r}({p:.3f})" for t, p in ms_top[:6]))


def experiment_within_category_arithmetic(vecs_by_layer, labels, answers):
    """For each category, compute pairwise diff vectors and measure their
    internal consistency via pairwise cosine between diffs."""
    print(f"\n{'=' * 70}")
    print("Experiment 2: within-category diff vector consistency")
    print(f"{'=' * 70}")
    print("\nFor each category, compute all pairwise diff vectors v(A) - v(B).")
    print("Then measure mean cosine between diff vectors.")
    print("If there's a consistent 'between-facts' direction, this cosine is high.")
    print("If diffs are orthogonal (fact-specific), cosine is near zero.\n")

    for L in PROJECT_LAYERS:
        vecs = vecs_by_layer[L]
        print(f"--- Layer {L} ---")
        for cat in CATEGORIES:
            indices = np.where(labels == cat)[0]
            diffs = []
            # All ordered pairs A != B (excluding symmetric duplicates)
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    diffs.append(vecs[indices[i]] - vecs[indices[j]])
            diffs = np.array(diffs)
            # Mean pairwise cosine of diff vectors (excluding diagonal and self-negative pairs)
            n_d = len(diffs)
            sims = []
            for i in range(n_d):
                for j in range(i + 1, n_d):
                    sims.append(cos(diffs[i], diffs[j]))
            mean_sim = np.mean(sims) if sims else float("nan")
            # Also: how big are the diffs vs the centroid distance?
            centroid = vecs[indices].mean(axis=0)
            centroid_norms = [np.linalg.norm(centroid - vecs[i]) for i in indices]
            diff_norms = [np.linalg.norm(d) for d in diffs]
            print(f"  [{cat:>9s}] n_diffs={n_d:>3d}  mean pairwise cos(diffs): {mean_sim:+.4f}  "
                  f"|centroid-to-item|≈{np.mean(centroid_norms):.2f}  "
                  f"|diff|≈{np.mean(diff_norms):.2f}")


def experiment_cross_category_analogy(vecs_by_layer, labels, answers):
    """Test whether same-answer prompts from different categories align better
    after category-centroid subtraction."""
    print(f"\n{'=' * 70}")
    print("Experiment 3: cross-category same-answer alignment")
    print(f"{'=' * 70}")
    print("\nFor prompt pairs that share an answer across categories (e.g., both")
    print("Eiffel Tower and France predict 'Paris'), measure cosine similarity")
    print("before and after subtracting the category centroid from each.\n")

    # Identify cross-category same-answer pairs
    same_answer_pairs = []
    for i in range(len(PROMPTS)):
        for j in range(i + 1, len(PROMPTS)):
            # Normalize answer for matching (case-insensitive substring)
            ai, aj = answers[i].lower(), answers[j].lower()
            if ai == aj and labels[i] != labels[j]:
                same_answer_pairs.append((i, j))

    if not same_answer_pairs:
        print("No cross-category same-answer pairs found in the prompt set.")
        # Fallback: try pairs that share a CITY answer (Paris, Rome)
        city_answers = ["Paris", "Rome"]
        for city in city_answers:
            indices = [i for i, a in enumerate(answers) if a == city]
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    a, b = indices[i], indices[j]
                    if labels[a] != labels[b]:
                        same_answer_pairs.append((a, b))
        if not same_answer_pairs:
            print("  (Nothing even in the fallback)")
            return

    print(f"Found {len(same_answer_pairs)} cross-category same-answer pair(s):")
    for i, j in same_answer_pairs:
        p_i = PROMPTS[i]
        p_j = PROMPTS[j]
        print(f"  {p_i[1]!r}({labels[i]}) + {p_j[1]!r}({labels[j]}) → both say {answers[i]!r}")

    # Also construct control pairs: same category, DIFFERENT answers
    # (should NOT align after centroid subtraction)
    control_pairs = []
    # Different category, different answer — baseline
    np.random.seed(42)
    for _ in range(10):
        i = np.random.randint(len(PROMPTS))
        j = np.random.randint(len(PROMPTS))
        while j == i or labels[i] == labels[j] or answers[i] == answers[j]:
            j = np.random.randint(len(PROMPTS))
        control_pairs.append((i, j))

    print()
    for L in PROJECT_LAYERS:
        vecs = vecs_by_layer[L]
        # Category centroids
        centroids = {c: vecs[labels == c].mean(axis=0) for c in CATEGORIES}

        print(f"--- Layer {L} ---")

        # Same-answer pairs
        raw_sims = []
        sub_sims = []
        for i, j in same_answer_pairs:
            raw_sims.append(cos(vecs[i], vecs[j]))
            vi_sub = vecs[i] - centroids[labels[i]]
            vj_sub = vecs[j] - centroids[labels[j]]
            sub_sims.append(cos(vi_sub, vj_sub))
        print(f"  same-answer cross-category ({len(same_answer_pairs)} pairs):")
        print(f"    cos before centroid-sub: {np.mean(raw_sims):+.4f}")
        print(f"    cos after  centroid-sub: {np.mean(sub_sims):+.4f}  "
              f"(change: {np.mean(sub_sims) - np.mean(raw_sims):+.4f})")

        # Control pairs
        raw_sims_c = []
        sub_sims_c = []
        for i, j in control_pairs:
            raw_sims_c.append(cos(vecs[i], vecs[j]))
            vi_sub = vecs[i] - centroids[labels[i]]
            vj_sub = vecs[j] - centroids[labels[j]]
            sub_sims_c.append(cos(vi_sub, vj_sub))
        print(f"  different-answer cross-category ({len(control_pairs)} control pairs):")
        print(f"    cos before centroid-sub: {np.mean(raw_sims_c):+.4f}")
        print(f"    cos after  centroid-sub: {np.mean(sub_sims_c):+.4f}  "
              f"(change: {np.mean(sub_sims_c) - np.mean(raw_sims_c):+.4f})")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading model...")
    model, processor = load_model()
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    print(f"\nExtracting {len(PROMPTS)} fact vectors at layers {PROJECT_LAYERS}...\n")
    vecs_by_layer, labels, answers = extract_vectors_at_layers(
        model, processor, tokenizer, PROJECT_LAYERS
    )
    print(f"(Extracted {len(PROMPTS)} vectors of dim 2560 at {len(PROJECT_LAYERS)} layers)")

    experiment_centroid_projection(model, tokenizer, vecs_by_layer, labels)
    experiment_within_category_arithmetic(vecs_by_layer, labels, answers)
    experiment_cross_category_analogy(vecs_by_layer, labels, answers)


if __name__ == "__main__":
    main()
