"""Step 38 — DLA across FACTUAL_15 on Gemma 3 4B.

Gemma 3 4B counterpart of step_33 (E4B) and step_37 (E2B). Tests
the depth-fraction candidate from task 000190: across both Gemma 4
sizes, the median commit layer in step_33 lands at ~0.66 of network
depth (E4B 0.69, E2B 0.66, Δ = 0.033). If Gemma 3 4B's commit
distribution lands in the same band, the framing graduates from
"observation in the E-series" to "fixed depth-fraction commit
across the family." If commits scatter or land elsewhere, the
candidate fails predictively and we know to look harder.

Through mechbench-core's now-real Gemma 3 hook system (000192).

Outputs JSON to mechbench-experiments/caches/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

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


MODEL_ID = "mlx-community/gemma-3-4b-it-bf16"


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
    print(
        f"Gemma 3 4B: {n_layers} layers, globals at {list(arch.global_layers)}"
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

        result = model.run(
            ids,
            interventions=[Capture.residual(range(n_layers), point="post")],
        )
        stack = accumulated_resid(result.cache)
        attrs = logit_attrs(model, stack, [t_id, d_id])
        diffs[p_idx] = attrs[:, 0] - attrs[:, 1]

        negs = np.where(diffs[p_idx] < 0)[0]
        last_neg = int(negs[-1]) if len(negs) > 0 else -1
        commit_layer = last_neg + 1 if last_neg >= 0 else 0
        print(
            f"  [{p_idx + 1:2d}/{n_prompts}] {target:12s} vs {distractor:12s}  "
            f"commit at L{commit_layer:02d}  (final diff = {diffs[p_idx][-1]:+.2f})"
        )

    commit_layers: list[int] = []
    for p_idx in range(n_prompts):
        negs = np.where(diffs[p_idx] < 0)[0]
        commit_layers.append(int(negs[-1]) + 1 if len(negs) > 0 else 0)

    fracs = np.array([c / n_layers for c in commit_layers], dtype=np.float64)
    print("\n--- Commit-layer fractions (depth-fraction candidate, 000190) ---")
    print(f"  n_layers={n_layers}, n_prompts={n_prompts}")
    print(f"  fractions: {[round(float(f), 3) for f in fracs]}")
    print(f"  median: {float(np.median(fracs)):.3f}")
    print(f"  mean:   {float(np.mean(fracs)):.3f}")
    print(f"  std:    {float(np.std(fracs)):.3f}")
    print()
    print("  Reference (000188 / 000190):")
    print(f"    E4B median frac: 0.690  (n_layers=42)")
    print(f"    E2B median frac: 0.657  (n_layers=35)")
    print(f"    Gemma 3 4B:      {float(np.median(fracs)):.3f}  (n_layers={n_layers})")

    out_dir = ROOT / "caches"
    out_dir.mkdir(exist_ok=True)
    out_json = out_dir / "step_38_dla_factual_sweep_gemma3_4b.json"
    out_json.write_text(json.dumps({
        "model_id": MODEL_ID,
        "n_layers": n_layers,
        "global_layers": list(arch.global_layers),
        "first_kv_shared": arch.first_kv_shared_layer,
        "n_prompts": n_prompts,
        "labels": labels,
        "diffs": [[round(float(v), 4) for v in row] for row in diffs],
        "commit_layers": commit_layers,
        "commit_fractions": [round(float(f), 4) for f in fracs],
    }, indent=2))
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
