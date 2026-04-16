"""Smoke test for L2 — prompts, lens, geometry.

Five integration checks against the model:

  1. PromptSet.validate on FACTUAL_15: should keep most/all 15 prompts (the
     prompts were chosen because Gemma 4 E4B answers them confidently).
  2. logit_lens_final on the Eiffel Tower prompt: rank should crash from
     ~100k+ in early layers to 0 by layer 41 (the classic phase transition
     from finding 01).
  3. fact_vectors_at on a small sample of BIG_SWEEP_96: shapes correct,
     extracted vectors have meaningful norms.
  4. Centroid decoding of the BIG_SWEEP capital category at layer 30:
     top tokens should include 'capital' or its multilingual equivalents
     (per finding 12).
  5. Geometry stats on the full BIG_SWEEP at layer 30: nearest-neighbor
     same-category hit rate should be ~1.0; k-means-style cluster purity
     should be ~1.0 (per finding 12).

Run from project root with the venv active:
    python -m gemma4_mlx_interp._smoke_l2
"""

from __future__ import annotations

import sys
import time

import numpy as np
from sklearn.cluster import KMeans

from . import (
    BIG_SWEEP_96,
    FACTUAL_15,
    Capture,
    Model,
    centroid_decode,
    cluster_purity,
    cosine_matrix,
    fact_vectors_at,
    intra_inter_separation,
    logit_lens_final,
    nearest_neighbor_purity,
    silhouette_cosine,
)


def main() -> int:
    print("Loading model...")
    t0 = time.perf_counter()
    model = Model.load()
    print(f"Loaded in {time.perf_counter() - t0:.1f}s.\n")

    all_pass = True

    # --- 1. PromptSet.validate ---
    print("=" * 60)
    print("1. PromptSet.validate on FACTUAL_15")
    print("=" * 60)
    valid_15 = FACTUAL_15.validate(model)
    n_kept = len(valid_15)
    n_skipped = len(valid_15.skipped)
    ok_validate = n_kept >= 12  # we should keep most; some may miss strict target match
    print(f"  Result: {n_kept} kept / {n_skipped} skipped")
    if n_skipped:
        print("  Skipped:")
        for p in valid_15.skipped:
            print(f"    - {p.text[:55]:55s}  (target was {p.target!r})")
    print(f"  [{'OK' if ok_validate else 'FAIL'}] kept >= 12")
    all_pass &= ok_validate

    # --- 2. logit_lens_final on Eiffel Tower ---
    print()
    print("=" * 60)
    print("2. logit_lens_final on the Eiffel Tower prompt")
    print("=" * 60)
    eiffel = next(vp for vp in valid_15 if "Eiffel" in vp.prompt.text)
    result = model.run(
        eiffel.input_ids,
        interventions=[Capture.residual(layers=range(42), point="post")],
    )
    ranks, logprobs = logit_lens_final(model, result.cache, eiffel.target_id)
    print(f"  Target: {eiffel.target_token!r}  (id={eiffel.target_id})")
    print(f"  Rank trajectory:")
    for i in [0, 6, 12, 18, 24, 27, 30, 33, 36, 41]:
        print(f"    layer {i:>2}: rank {ranks[i]:>7d}, log p {logprobs[i]:>+7.3f}")
    early_rank = int(ranks[12])
    final_rank = int(ranks[41])
    ok_lens = early_rank > 1000 and final_rank == 0
    print(f"  [{'OK' if ok_lens else 'FAIL'}] early rank ({early_rank}) > 1000 "
          f"and final rank ({final_rank}) == 0")
    all_pass &= ok_lens

    # --- 3. fact_vectors_at on first 16 prompts of BIG_SWEEP_96 ---
    print()
    print("=" * 60)
    print("3. fact_vectors_at: shape and norm sanity")
    print("=" * 60)
    sample_prompts = type(BIG_SWEEP_96)(
        prompts=BIG_SWEEP_96.prompts[:16], name="BIG_SWEEP_sample",
    )
    print("  (validating 16 sample prompts...)")
    sample_valid = sample_prompts.validate(model, verbose=False)
    print(f"  validated: {len(sample_valid)} of 16")
    vecs_dict = fact_vectors_at(model, sample_valid, layers=[15, 30])
    ok_shapes = (
        vecs_dict[15].shape == (len(sample_valid), 2560)
        and vecs_dict[30].shape == (len(sample_valid), 2560)
    )
    norms = np.linalg.norm(vecs_dict[30], axis=1)
    ok_norms = bool(norms.min() > 1.0 and norms.max() < 1e6)
    print(f"  vecs[15].shape={vecs_dict[15].shape}, vecs[30].shape={vecs_dict[30].shape}")
    print(f"  vecs[30] norms: min={norms.min():.2f}, max={norms.max():.2f}, mean={norms.mean():.2f}")
    print(f"  [{'OK' if ok_shapes and ok_norms else 'FAIL'}] shapes correct and norms sane")
    all_pass &= ok_shapes and ok_norms

    # --- 4. Centroid decoding of BIG_SWEEP_96.capital at layer 30 ---
    print()
    print("=" * 60)
    print("4. Full BIG_SWEEP_96 + centroid decoding + clustering")
    print("=" * 60)
    print(f"  (validating all {len(BIG_SWEEP_96)} prompts...)")
    big_valid = BIG_SWEEP_96.validate(model, verbose=False)
    print(f"  validated: {len(big_valid)} of {len(BIG_SWEEP_96)}")

    print(f"  (extracting fact vectors at layer 30 for {len(big_valid)} prompts...)")
    big_vecs = fact_vectors_at(model, big_valid, layers=[30])[30]
    big_labels = np.array([vp.prompt.category for vp in big_valid])
    overall_mean = big_vecs.mean(axis=0)

    # Capital centroid decoding
    cap_mask = big_labels == "capital"
    cap_top = centroid_decode(
        model, big_vecs[cap_mask], k=8, mean_subtract=True, overall_mean=overall_mean,
    )
    cap_tokens = [t.lower() for t, _ in cap_top]
    cap_concept_present = any(
        any(w in t for w in ["capital", "city", "राजधानी", "首都", "capitale", "столи", "tadt"])
        for t in cap_tokens
    )
    print(f"  capital centroid top-8: {[(t, round(p, 3)) for t, p in cap_top]}")
    print(f"  [{'OK' if cap_concept_present else 'FAIL'}] decodes to a 'capital'-flavored concept")
    all_pass &= cap_concept_present

    # --- 5. Clustering: NN purity, k-means purity, silhouette ---
    print()
    print("=" * 60)
    print("5. BIG_SWEEP_96 clustering at layer 30")
    print("=" * 60)
    nn_rate, _ = nearest_neighbor_purity(big_vecs, big_labels)
    intra, inter, sep = intra_inter_separation(big_vecs, big_labels)
    cats = list(dict.fromkeys(big_labels.tolist()))
    km = KMeans(n_clusters=len(cats), n_init=10, random_state=42).fit(big_vecs)
    purity = cluster_purity(big_labels.tolist(), km.labels_.tolist())
    sil = silhouette_cosine(big_vecs, big_labels)
    print(f"  NN same-category hit rate: {nn_rate:.3f}  (expect ~1.0)")
    print(f"  k-means purity (k={len(cats)}):       {purity:.3f}  (expect ~1.0)")
    print(f"  silhouette (cosine):       {sil:+.4f}")
    print(f"  intra-cat cos: {intra:+.4f}, inter-cat cos: {inter:+.4f}, sep: {sep:+.4f}")
    ok_clustering = nn_rate >= 0.95 and purity >= 0.95 and sil > 0.3
    print(f"  [{'OK' if ok_clustering else 'FAIL'}] NN >= 0.95, purity >= 0.95, silhouette > 0.3")
    all_pass &= ok_clustering

    print()
    if not all_pass:
        print("L2 SMOKE TEST FAILED.")
        return 1
    print("L2 smoke test passed.")
    print("Prompts, lens, fact vectors, centroid decoding, geometry stats all working.")
    print("L2 is ready; remaining migrations may begin.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
