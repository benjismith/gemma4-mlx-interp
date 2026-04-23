"""Export step_33 (DLA across FACTUAL_15) data as JSON for mechbench-ui.

Mirrors step_33_dla_factual_sweep.py's computation, but emits a single
JSON file to ../mechbench-ui/public/data/ instead of plotting. This is the
first instance of the Python -> UI data pipeline; the output shape is
hand-designed here and will migrate to mechbench-schema when that emission
layer is wired up.

Assumes the mechbench family repos are laid out side-by-side:
  ~/dev/mechbench/mechbench-experiments/
  ~/dev/mechbench/mechbench-ui/          <- output lands here
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.prompts.factual import FACTUAL_15  # noqa: E402
from mechbench_core import (  # noqa: E402
    Capture,
    GLOBAL_LAYERS,
    Model,
    N_LAYERS,
    accumulated_resid,
    logit_attrs,
)


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


def resolve_output_path() -> Path:
    here = Path(__file__).resolve()
    experiments_repo = here.parent.parent
    tree_root = experiments_repo.parent
    ui_data_dir = tree_root / "mechbench-ui" / "public" / "data"
    ui_repo = tree_root / "mechbench-ui"
    if not ui_repo.exists():
        raise SystemExit(
            f"mechbench-ui not found at {ui_repo}. "
            "Confirm the mechbench repo family is laid out side-by-side."
        )
    ui_data_dir.mkdir(parents=True, exist_ok=True)
    return ui_data_dir / "step_33_dla_factual_sweep.json"


def main() -> None:
    output_path = resolve_output_path()
    print(f"Output target: {output_path}")

    print("Loading model...")
    model = Model.load()

    n_prompts = len(FACTUAL_15.prompts)
    diffs = np.zeros((n_prompts, N_LAYERS), dtype=np.float32)
    prompt_records: list[dict] = []

    for p_idx, prompt in enumerate(FACTUAL_15.prompts):
        target = prompt.target
        distractor = DISTRACTORS[target]
        ids = model.tokenize(prompt.text)
        t_id = first_token_id(model, target)
        d_id = first_token_id(model, distractor)

        print(f"[{p_idx + 1:2d}/{n_prompts}] {target:12s} vs {distractor:12s}")

        interventions = [Capture.residual(range(N_LAYERS), point="post")]
        result = model.run(ids, interventions=interventions)
        stack = accumulated_resid(result.cache)
        attrs = logit_attrs(model, stack, [t_id, d_id])
        diff_vec = (attrs[:, 0] - attrs[:, 1]).astype(np.float32)
        diffs[p_idx] = diff_vec

        prompt_records.append(
            {
                "target": target,
                "distractor": distractor,
                "text": prompt.text,
                "target_token_id": t_id,
                "distractor_token_id": d_id,
                "category": prompt.category or "",
                "diffs": [round(float(v), 4) for v in diff_vec],
            }
        )

    mean_vec = diffs.mean(axis=0)
    median_vec = np.median(diffs, axis=0)

    payload = {
        "experiment": "step_33_dla_factual_sweep",
        "description": (
            "Direct logit attribution across FACTUAL_15: per-prompt "
            "(target - distractor) logit difference at each layer's "
            "resid_post, projected through the tied unembed (no final-norm "
            "fold). Positive values indicate the cumulative residual prefers "
            "the target token; negative indicates distractor."
        ),
        "n_layers": N_LAYERS,
        "global_layers": list(GLOBAL_LAYERS),
        "model": "mlx-community/gemma-4-E4B-it-bf16",
        "prompts": prompt_records,
        "aggregates": {
            "mean": [round(float(v), 4) for v in mean_vec],
            "median": [round(float(v), 4) for v in median_vec],
        },
    }

    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {output_path} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
