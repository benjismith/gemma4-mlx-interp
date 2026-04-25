"""Step 33 — DLA across FACTUAL_15 on Gemma 4 E2B.

For each prompt, compute (target - distractor) logit at every layer's
resid_post via direct logit attribution. The E4B step_33 finding
showed a clean pattern: mid-layer residual *prefers the distractor*
(negative diff in the L9-L24 band on E4B), then flips to target-
preference around L25 — i.e., the model 'commits' to the target
answer at the boundary between fresh-K/V and KV-shared globals.

For E2B, the framing predicts the analogous commit happens at the
L14 → L15 boundary (first_kv_shared = 15). This script tests that
prediction.

Outputs JSON to mechbench-experiments/caches/ + a heatmap PNG.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.prompts.factual import FACTUAL_15  # noqa: E402
from mechbench_core import (  # noqa: E402
    Capture,
    Model,
    accumulated_resid,
    logit_attrs,
)


MODEL_ID = "mlx-community/gemma-4-E2B-it-bf16"
PREDICTED_BOUNDARY = 14  # commit predicted at L14 → L15


DISTRACTORS: dict[str, str] = {
    "Paris": "London",
    "Tokyo": "Kyoto",
    "China": "Japan",
    "Brazil": "Peru",
    "Africa": "Asia",
    "oxygen": "nitrogen",
    "meters": "miles",
    "Au": "Ag",
    "Shakespeare": "Marlowe",
    "Leonardo": "Michelangelo",
    "five": "six",
    "Wednesday": "Thursday",
    "cold": "warm",
    "blue": "gray",
    "pets": "animals",
}


def first_token_id(model, word: str) -> int:
    ids = model.tokenizer.encode(" " + word, add_special_tokens=False)
    return int(ids[0])


def main() -> None:
    print(f"Loading {MODEL_ID}...")
    model = Model.load(MODEL_ID)
    arch = model.arch
    n_layers = arch.n_layers
    global_layers = list(arch.global_layers)
    first_kv_shared = arch.first_kv_shared_layer
    print(
        f"E2B: {n_layers} layers, globals at {global_layers}, "
        f"first_kv_shared={first_kv_shared}"
    )

    n_prompts = len(FACTUAL_15.prompts)
    diffs = np.zeros((n_prompts, n_layers), dtype=np.float32)
    labels: list[str] = []

    for p_idx, prompt in enumerate(FACTUAL_15.prompts):
        target = prompt.target
        distractor = DISTRACTORS[target]
        labels.append(f"{target}/{distractor}")

        ids = model.tokenize(prompt.text)
        t_id = first_token_id(model, target)
        d_id = first_token_id(model, distractor)

        interventions = [Capture.residual(range(n_layers), point="post")]
        result = model.run(ids, interventions=interventions)
        # accumulated_resid defaults to E4B's N_LAYERS=42; pass the
        # variant's n_layers explicitly so it iterates the right range.
        stack = accumulated_resid(result.cache, layers=range(n_layers))
        attrs = logit_attrs(model, stack, [t_id, d_id])
        diffs[p_idx] = attrs[:, 0] - attrs[:, 1]

        # Find the commit layer = last layer where diff is still negative.
        negs = np.where(diffs[p_idx] < 0)[0]
        last_neg = int(negs[-1]) if len(negs) > 0 else -1
        commit_layer = last_neg + 1 if last_neg >= 0 else 0
        print(
            f"  [{p_idx + 1:2d}/{n_prompts}] {target:12s} vs {distractor:12s}  "
            f"commit at L{commit_layer:02d}  (final diff = {diffs[p_idx][-1]:+.2f})"
        )

    mean_diff = diffs.mean(axis=0)
    median_diff = np.median(diffs, axis=0)
    n_negative = (diffs < 0).sum(axis=0)

    print("\n--- Per-layer aggregates across 15 prompts ---")
    print("  L     type    mean    median  #neg")
    for i in range(n_layers):
        t = "GLOBAL" if i in global_layers else "local "
        print(
            f"  L{i:02d}  {t}  {mean_diff[i]:>+7.2f}  {median_diff[i]:>+7.2f}  "
            f"{n_negative[i]:>2d}/{n_prompts}"
        )

    # Commit-layer distribution.
    commit_layers: list[int] = []
    for p_idx in range(n_prompts):
        negs = np.where(diffs[p_idx] < 0)[0]
        commit_layers.append(int(negs[-1]) + 1 if len(negs) > 0 else 0)

    print(f"\nCommit-layer distribution across 15 prompts:")
    for cl, lab in zip(commit_layers, labels):
        rel = "(at boundary)" if cl == PREDICTED_BOUNDARY + 1 else ""
        print(f"  L{cl:02d}  {lab:32s} {rel}")

    cl_arr = np.array(commit_layers)
    cl_at_boundary = (cl_arr == PREDICTED_BOUNDARY + 1).sum()
    cl_within_2 = (np.abs(cl_arr - (PREDICTED_BOUNDARY + 1)) <= 2).sum()
    cl_after_boundary = (cl_arr > PREDICTED_BOUNDARY).sum()
    print(f"\n  At predicted boundary (L{PREDICTED_BOUNDARY + 1}):     "
          f"{cl_at_boundary}/{n_prompts}")
    print(f"  Within ±2 of predicted boundary: {cl_within_2}/{n_prompts}")
    print(f"  After predicted boundary:        {cl_after_boundary}/{n_prompts}")
    print(f"  Median commit layer:             L{int(np.median(cl_arr)):02d}")

    out_dir = ROOT / "caches"
    out_dir.mkdir(exist_ok=True)
    out_json = out_dir / "step_33_dla_factual_sweep_e2b.json"
    out_json.write_text(json.dumps({
        "model_id": MODEL_ID,
        "n_layers": n_layers,
        "global_layers": global_layers,
        "first_kv_shared": first_kv_shared,
        "predicted_boundary": PREDICTED_BOUNDARY,
        "n_prompts": n_prompts,
        "labels": labels,
        "diffs": [[round(float(v), 4) for v in row] for row in diffs],
        "mean": [round(float(v), 4) for v in mean_diff],
        "median": [round(float(v), 4) for v in median_diff],
        "commit_layers": commit_layers,
    }, indent=2))
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
