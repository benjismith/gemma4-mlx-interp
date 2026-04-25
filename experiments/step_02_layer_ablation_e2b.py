"""Step 02 — layer-ablation damage curve on Gemma 4 E2B.

Predictive validation pass for the KV-boundary framing from task
000125: the framing predicts the **pivot layer is the global
immediately upstream of `first_kv_shared`** — for E2B that's L14
(globals at [4, 9, 14, 19, 24, 29, 34], `first_kv_shared = 15`).
The original L23-pivot story was post-hoc on E4B; running the
analogous experiment on E2B without baking the prediction into
the methodology is what task 000188 demands.

Outputs:
  - mechbench-ui/public/data/step_02_layer_ablation_e2b.json
    (LayerAblationPayload, same shape as the E4B exporter)
  - stdout summary with the L14 prediction explicitly checked.

Mirrors `experiments/step_02_layer_ablation.py` and
`export_step_02_for_ui.py`, but uses `model.arch.<field>` instead
of E4B-default module constants so it adapts to whichever variant
gets loaded.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mechbench_core import Ablate, Model  # noqa: E402
from experiments.prompts.factual import FACTUAL_15  # noqa: E402
from mechbench_schema import (  # noqa: E402
    AblationPrompt,
    LayerAblationPayload,
    LayerAggregates,
)


MODEL_ID = "mlx-community/gemma-4-E2B-it-bf16"
MIN_CONFIDENCE = 0.5

# E2B globals + KV-boundary prediction (000125 reframe).
PREDICTED_PIVOT = 14            # global immediately upstream of first_kv_shared = 15
NULL_FRESH_KV_GLOBAL = 9        # a fresh-K/V global; framing predicts unremarkable
NEXT_GLOBAL_AFTER_PIVOT = 19    # first KV-shared global; framing makes no claim


def _last_logp(logits: mx.array) -> np.ndarray:
    last = logits[0, -1, :].astype(mx.float32)
    lp = last - mx.logsumexp(last)
    mx.eval(lp)
    return np.array(lp)


def resolve_output_path() -> Path:
    here = Path(__file__).resolve()
    tree_root = here.parent.parent.parent
    ui_data_dir = tree_root / "mechbench-ui" / "public" / "data"
    ui_data_dir.mkdir(parents=True, exist_ok=True)
    return ui_data_dir / "step_02_layer_ablation_e2b.json"


def main() -> None:
    output_path = resolve_output_path()
    print(f"Output target: {output_path}")

    print(f"Loading {MODEL_ID}...")
    t0 = time.perf_counter()
    model = Model.load(MODEL_ID)
    print(f"Loaded in {time.perf_counter() - t0:.1f}s")

    arch = model.arch
    n_layers = arch.n_layers
    global_layers = list(arch.global_layers)
    print(
        f"E2B arch: {n_layers} layers, globals at {global_layers}, "
        f"first_kv_shared={arch.first_kv_shared_layer}"
    )

    # Validation pass.
    prompts = FACTUAL_15.prompts
    validated: list[tuple[str, str, int, float]] = []
    print(f"\nValidating {len(prompts)} prompts (top-1 prob >= {MIN_CONFIDENCE})...")
    for prompt in prompts:
        ids = model.tokenize(prompt.text)
        result = model.run(ids)
        lp = _last_logp(result.logits)
        top1_id = int(np.argmax(lp))
        top1_prob = float(np.exp(lp[top1_id]))
        top1_tok = model.tokenizer.decode([top1_id])
        keep = top1_prob >= MIN_CONFIDENCE
        marker = "✓" if keep else "✗"
        print(
            f"  {marker} prob={top1_prob:.3f} top1={top1_tok!r:>16}  "
            f"target={prompt.target!r:>16}  '{prompt.text[:55]}'"
        )
        if keep:
            validated.append(
                (prompt.text, prompt.target, top1_id, float(lp[top1_id]))
            )

    n = len(validated)
    print(f"{n}/{len(prompts)} prompts validated.\n")

    damage = np.zeros((n_layers, n), dtype=np.float32)
    print(f"Running {n_layers} × {n} = {n_layers * n} ablated forward passes...")
    t0 = time.perf_counter()
    for layer_idx in range(n_layers):
        ablation = Ablate.layer(layer_idx)
        for j, (text, _target, top1_id, baseline_lp) in enumerate(validated):
            ids = model.tokenize(text)
            result = model.run(ids, interventions=[ablation])
            lp = _last_logp(result.logits)
            damage[layer_idx, j] = float(lp[top1_id]) - baseline_lp
        if (layer_idx + 1) % 5 == 0 or layer_idx == n_layers - 1:
            mean_d = float(damage[layer_idx].mean())
            elapsed = time.perf_counter() - t0
            eta = elapsed / (layer_idx + 1) * (n_layers - layer_idx - 1)
            print(
                f"  layer {layer_idx:>2}: mean Δlogp = {mean_d:>+7.3f}  "
                f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]"
            )

    payload = LayerAblationPayload(
        experiment="step_02_layer_ablation_e2b",
        description=(
            "Per-layer zero-ablation on Gemma 4 E2B (35 layers, globals at "
            "[4, 9, 14, 19, 24, 29, 34], first_kv_shared=15). Predictive test "
            "of the KV-boundary framing from task 000125: pivot should be L14, "
            "the global immediately upstream of first_kv_shared. Null check at "
            "L9 (a fresh-K/V global) and contrast against L19 (first "
            "KV-shared global)."
        ),
        model=MODEL_ID,
        n_layers=n_layers,
        global_layers=global_layers,
        prompts=[
            AblationPrompt(
                text=text,
                target=target,
                top1_id=top1_id,
                baseline_logprob=round(baseline_lp, 4),
                damage=[round(float(v), 4) for v in damage[:, j]],
            )
            for j, (text, target, top1_id, baseline_lp) in enumerate(validated)
        ],
        aggregates=LayerAggregates(
            mean=[round(float(v), 4) for v in damage.mean(axis=1)],
            median=[round(float(v), 4) for v in np.median(damage, axis=1)],
        ),
    )

    output_path.write_text(
        json.dumps(payload.model_dump(mode="json"), indent=2, ensure_ascii=False)
        + "\n"
    )
    print(f"\nWrote {output_path} ({output_path.stat().st_size} bytes)")

    # ---- predictive-validation summary ----
    mean = damage.mean(axis=1)
    median = np.median(damage, axis=1)
    peak = int(np.argmin(mean))
    order = np.argsort(mean)
    top5 = [(int(i), round(float(mean[i]), 3)) for i in order[:5]]

    print("\n--- Predictive-validation summary (000188) ---")
    print(f"  Predicted pivot:                L{PREDICTED_PIVOT} "
          f"(global immediately upstream of first_kv_shared)")
    print(f"  Observed peak (most damaging):  L{peak}  "
          f"mean Δlogp = {mean[peak]:+.3f}")
    print(f"  Top-5 by mean Δlogp:            {top5}")
    print()
    print(f"  Comparison points:")
    for label, idx in [
        (f"L{PREDICTED_PIVOT} (predicted pivot)", PREDICTED_PIVOT),
        (f"L{NULL_FRESH_KV_GLOBAL} (null: fresh-K/V global)", NULL_FRESH_KV_GLOBAL),
        (f"L{NEXT_GLOBAL_AFTER_PIVOT} (first KV-shared global)", NEXT_GLOBAL_AFTER_PIVOT),
    ]:
        is_global = idx in global_layers
        marker = "GLOBAL" if is_global else "local"
        print(
            f"    {label:50s} mean Δlogp = {mean[idx]:>+7.3f}  "
            f"median = {median[idx]:>+7.3f}  ({marker})"
        )

    # Honest verdict.
    pivot_in_top5 = PREDICTED_PIVOT in {i for i, _ in top5}
    null_unremarkable = abs(mean[NULL_FRESH_KV_GLOBAL]) < 0.5 * abs(mean[peak])
    if peak == PREDICTED_PIVOT and null_unremarkable:
        verdict = "PREDICTION CONFIRMED"
    elif pivot_in_top5 and null_unremarkable:
        verdict = "PREDICTION PARTIALLY CONFIRMED (pivot in top-5 but not peak)"
    else:
        verdict = "PREDICTION FAILS"
    print(f"\n  Verdict (this experiment alone): {verdict}")


if __name__ == "__main__":
    main()
