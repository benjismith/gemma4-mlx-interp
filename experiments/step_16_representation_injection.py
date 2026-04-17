"""Experiment 2 from docs/proposals/factorization-experiments.md.

Compute the capital-category centroid v_capital from the DISAMBIG_A1
prompts at layer 30 (subject position). Inject alpha * v_capital into
the residual stream at the final position of three target prompts and
measure whether the model's output shifts toward capital-city tokens.

Three conditions:
  C1 (neutral):    'The following country is famous for its'
                   - the model has no specific operation in mind
  C2 (different):  'The past tense of run is'
                   - a clearly different cognitive operation (morphology)
  C3 (control):    'The capital of Germany is'
                   - the model was already producing 'Berlin'; injection
                     should be no-op or saturating

Sweep alpha in {0, 0.5, 1.0, 2.0, 5.0} (alpha=0 = baseline).

Question: is v_capital a passive correlate ('the model happens to be in
this region of activation space when it's running a capital lookup') or
a functional handle ('inject this vector and the model starts running a
capital lookup, even if the prompt didn't ask for one')?

Run from project root:
    python experiments/step_16_representation_injection.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gemma4_mlx_interp import (  # noqa: E402
    Model, Patch, fact_vectors_at,
)
from experiments.prompts import DISAMBIG_ALL, DISAMBIG_A1  # noqa: E402

OUT_DIR = ROOT / "caches"
LAYER = 30
ALPHAS = [0.0, 0.5, 1.0, 2.0, 5.0]

# The 8 capital-city anchors used in DISAMBIG_A1.
CAPITAL_CITIES = ("Paris", "Tokyo", "Berlin", "Rome",
                  "Madrid", "Moscow", "Cairo", "Athens")

# Three injection target prompts.
TARGETS = [
    ("C1_neutral",
     "Complete this sentence with one word: The following country is famous for its"),
    ("C2_different",
     "Complete this sentence with one word: The past tense of run is"),
    ("C3_control",
     "Complete this sentence with one word: The capital of Germany is"),
]


def _capital_token_ids(model) -> set[int]:
    """All token ids that decode to one of the 8 capital city names, in
    either the space-prefixed or no-space variant. Returns a set."""
    ids = set()
    for city in CAPITAL_CITIES:
        for variant in (city, " " + city):
            tok = model.tokenizer.encode(variant, add_special_tokens=False)
            if tok:
                ids.add(int(tok[0]))
    return ids


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading model...")
    model = Model.load()

    # ---- Build two candidate steering vectors ----
    # v_capital_raw: mean of A1's subject-position residuals at layer 30.
    # v_capital_sub: same, mean-subtracted using all DISAMBIG prompts as the
    # baseline. Finding 12 showed the meaningful capital-concept signal lives
    # in the mean-subtracted centroid (the raw one is dominated by the
    # prompt-template common-mode).
    print(f"\nValidating {len(DISAMBIG_A1)} DISAMBIG_A1 prompts (no filtering)...\n")
    a1_valid = DISAMBIG_A1.validate(
        model, verbose=False, min_confidence=0.0, require_target_match=False,
    )
    print(f"Validating {len(DISAMBIG_ALL)} DISAMBIG_ALL prompts (for the baseline mean)...\n")
    all_valid = DISAMBIG_ALL.validate(
        model, verbose=False, min_confidence=0.0, require_target_match=False,
    )
    print(f"Extracting subject-position vectors at layer {LAYER}...")
    a1_vecs = fact_vectors_at(model, a1_valid, layers=[LAYER])[LAYER]
    all_vecs = fact_vectors_at(model, all_valid, layers=[LAYER])[LAYER]
    v_raw_np = a1_vecs.mean(axis=0).astype(np.float32)
    v_sub_np = (a1_vecs.mean(axis=0) - all_vecs.mean(axis=0)).astype(np.float32)

    def _to_mx_d_model(arr_np: np.ndarray) -> mx.array:
        return mx.array(arr_np[None, :], dtype=mx.bfloat16)[0]

    v_raw = _to_mx_d_model(v_raw_np)
    v_sub = _to_mx_d_model(v_sub_np)
    print(f"  v_capital_raw norm: {float(np.linalg.norm(v_raw_np)):>7.2f}")
    print(f"  v_capital_sub norm: {float(np.linalg.norm(v_sub_np)):>7.2f}  "
          f"(mean-subtracted; should be much smaller)")

    steering_vectors = [
        ("raw", v_raw, v_raw_np),
        ("mean_sub", v_sub, v_sub_np),
    ]

    # ---- Set up token id bookkeeping ----
    cap_ids = _capital_token_ids(model)
    print(f"Capital-city token ids: {sorted(cap_ids)}")
    cap_id_array = np.array(sorted(cap_ids))

    # ---- Sweep over (vector, condition, alpha) ----
    print(f"\n{'=' * 70}")
    print(f"Injection sweep: vector x condition x alpha")
    print(f"{'=' * 70}")

    # results[vec_name][cond_name][alpha] = {top1_id/tok/prob, p_capital, ...}
    results: dict = {}
    for vec_name, vec, vec_np in steering_vectors:
        print(f"\n##### Steering vector: {vec_name} (norm {float(np.linalg.norm(vec_np)):.2f})")
        results[vec_name] = {}
        for cond_name, target_text in TARGETS:
            print(f"\n--- [{vec_name}] {cond_name}: {target_text!r} ---")
            ids = model.tokenize(target_text)
            seq_len = int(ids.shape[1])
            final_pos = seq_len - 1

            results[vec_name][cond_name] = {}
            baseline_p_cap = None
            for alpha in ALPHAS:
                interventions = []
                if alpha != 0.0:
                    interventions = [Patch.add(
                        layer=LAYER, position=final_pos, value=vec, alpha=alpha,
                    )]
                probs = model.run(ids, interventions=interventions or None).last_probs()
                top1 = int(np.argmax(probs))
                top1_tok = model.tokenizer.decode([top1])
                top1_prob = float(probs[top1])
                p_cap = float(probs[cap_id_array].sum())
                if alpha == 0.0:
                    baseline_p_cap = p_cap
                delta_p_cap = p_cap - (baseline_p_cap or 0.0)
                results[vec_name][cond_name][alpha] = {
                    "top1_id": top1,
                    "top1_tok": top1_tok,
                    "top1_prob": top1_prob,
                    "p_capital": p_cap,
                    "delta_p_capital": delta_p_cap,
                }
                print(f"  alpha={alpha:>4.1f}  top1={top1_tok!r:>16s} p={top1_prob:.3f}  "
                      f"p(capital_cities)={p_cap:.4f}  Δ={delta_p_cap:+.4f}")

    # ---- Verdict ----
    print(f"\n{'=' * 70}")
    print(f"Verdict")
    print(f"{'=' * 70}")
    for vec_name, _, _ in steering_vectors:
        print(f"\n##### Steering vector: {vec_name}")
        for cond_name, _ in TARGETS:
            row = results[vec_name][cond_name]
            baseline = row[0.0]["p_capital"]
            max_alpha_p = max(row[a]["p_capital"] for a in ALPHAS if a > 0)
            peak_alpha = max(ALPHAS,
                              key=lambda a: row[a]["p_capital"] if a > 0 else -1)
            ratio = (max_alpha_p / max(baseline, 1e-9))
            print(f"\n  {cond_name}:")
            print(f"    baseline p(capital_cities):     {baseline:.4f}")
            print(f"    peak under injection:           {max_alpha_p:.4f}  "
                  f"(at alpha={peak_alpha})")
            ever_capital = any(row[a]["top1_id"] in cap_ids for a in ALPHAS if a > 0)
            baseline_top_was_capital = row[0.0]["top1_id"] in cap_ids
            if baseline_top_was_capital:
                print(f"    baseline top-1 was already a capital city")
            elif ever_capital:
                for a in ALPHAS:
                    if a > 0 and row[a]["top1_id"] in cap_ids:
                        print(f"    top-1 BECAME a capital city at alpha={a}: "
                              f"{row[a]['top1_tok']!r}")
                        break
            else:
                print(f"    top-1 never became a capital city under any injection")

    # ---- Plot: 2 rows (vec) x 3 cols (cond) of p(cap) vs alpha ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for row_idx, (vec_name, _, vec_np) in enumerate(steering_vectors):
        for col_idx, (cond_name, target_text) in enumerate(TARGETS):
            ax = axes[row_idx, col_idx]
            row = results[vec_name][cond_name]
            ys = [row[a]["p_capital"] for a in ALPHAS]
            color = "#1f77b4" if vec_name == "raw" else "#2ca02c"
            ax.plot(ALPHAS, ys, marker="o", linewidth=2, color=color)
            ax.set_xlabel("injection scale alpha")
            ax.set_ylabel("p(capital-city tokens)")
            for a, y, top1 in zip(ALPHAS, ys, [row[a]["top1_tok"] for a in ALPHAS]):
                ax.annotate(f"{top1!r}", xy=(a, y), xytext=(5, 8),
                            textcoords="offset points", fontsize=8)
            short = target_text[:38] + ("..." if len(target_text) > 38 else "")
            ax.set_title(f"[{vec_name}] {cond_name}\n{short}", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, max(0.05, max(ys) * 1.2))

    fig.suptitle(
        f"Representation injection at layer {LAYER}, final position\n"
        f"top row = raw v_capital (norm {float(np.linalg.norm(v_raw_np)):.0f});  "
        f"bottom row = mean-subtracted v_capital "
        f"(norm {float(np.linalg.norm(v_sub_np)):.1f})",
        fontsize=11,
    )
    plt.tight_layout()
    out_path = OUT_DIR / "representation_injection.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
