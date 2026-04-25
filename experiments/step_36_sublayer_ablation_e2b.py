"""Step 36 — sublayer ablation (attention vs MLP) on Gemma 4 E2B.

The cleanest E4B signal for the L23 pivot was step_04: MLP
ablation drags damage across most of the network, but **attention
ablation specifically peaks at L23**. That's the experiment most
directly testing the KV-boundary framing's pivot prediction.

For E2B (000188) the framing predicts attention ablation should
peak at **L14** (the global immediately upstream of
`first_kv_shared = 15`). Null check: L9, a fresh-K/V global the
framing says should look unremarkable.

Outputs raw values to mechbench-experiments/caches/ and prints a
predictive-validation summary; no ui-public-data emit because we
don't have a chart for sublayer ablation yet.
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


MODEL_ID = "mlx-community/gemma-4-E2B-it-bf16"
PREDICTED_PIVOT = 14
NULL_FRESH_KV_GLOBAL = 9
NEXT_GLOBAL_AFTER_PIVOT = 19


def _last_logp(logits: mx.array) -> np.ndarray:
    last = logits[0, -1, :].astype(mx.float32)
    lp = last - mx.logsumexp(last)
    mx.eval(lp)
    return np.array(lp)


def main() -> None:
    out_dir = ROOT / "caches"
    out_dir.mkdir(exist_ok=True)
    out_json = out_dir / "step_36_sublayer_ablation_e2b.json"

    print(f"Loading {MODEL_ID}...")
    t0 = time.perf_counter()
    model = Model.load(MODEL_ID)
    print(f"Loaded in {time.perf_counter() - t0:.1f}s")

    arch = model.arch
    n_layers = arch.n_layers
    global_layers = list(arch.global_layers)
    print(
        f"E2B: {n_layers} layers, globals at {global_layers}, "
        f"first_kv_shared={arch.first_kv_shared_layer}"
    )

    print("\nValidating FACTUAL_15...\n")
    valid = FACTUAL_15.validate(model)
    n = len(valid)
    print(f"\n{n} prompts validated.")

    attn_delta = np.zeros((n_layers, n), dtype=np.float64)
    mlp_delta = np.zeros((n_layers, n), dtype=np.float64)

    total = n_layers * n * 2
    print(f"\nRunning {total} ablated forward passes...")
    t0 = time.perf_counter()
    for layer_idx in range(n_layers):
        attn_abl = Ablate.attention(layer_idx)
        mlp_abl = Ablate.mlp(layer_idx)
        for j, vp in enumerate(valid):
            r = model.run(vp.input_ids, interventions=[attn_abl])
            attn_delta[layer_idx, j] = (
                float(_last_logp(r.logits)[vp.target_id]) - vp.baseline_lp
            )
            r = model.run(vp.input_ids, interventions=[mlp_abl])
            mlp_delta[layer_idx, j] = (
                float(_last_logp(r.logits)[vp.target_id]) - vp.baseline_lp
            )
        if (layer_idx + 1) % 5 == 0 or layer_idx == n_layers - 1:
            elapsed = time.perf_counter() - t0
            eta = elapsed / (layer_idx + 1) * (n_layers - layer_idx - 1)
            print(
                f"  layer {layer_idx:>2}: attn Δ={np.mean(attn_delta[layer_idx]):>+7.3f}  "
                f"mlp Δ={np.mean(mlp_delta[layer_idx]):>+7.3f}  "
                f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]"
            )

    mean_attn = np.mean(attn_delta, axis=1)
    median_attn = np.median(attn_delta, axis=1)
    mean_mlp = np.mean(mlp_delta, axis=1)

    out_json.write_text(json.dumps({
        "model_id": MODEL_ID,
        "n_layers": n_layers,
        "global_layers": global_layers,
        "first_kv_shared": arch.first_kv_shared_layer,
        "n_prompts": n,
        "attn_mean": [round(float(v), 4) for v in mean_attn],
        "attn_median": [round(float(v), 4) for v in median_attn],
        "mlp_mean": [round(float(v), 4) for v in mean_mlp],
        "mlp_median": [
            round(float(v), 4) for v in np.median(mlp_delta, axis=1)
        ],
    }, indent=2))
    print(f"\nWrote {out_json}")

    print("\n--- Per-layer attention vs MLP ---")
    print(f"{'L':>3} {'type':>8} {'attn':>8} {'mlp':>8} {'dominant':>9}")
    for i in range(n_layers):
        t = "GLOBAL" if i in global_layers else "local"
        if abs(mean_attn[i]) < 0.01 and abs(mean_mlp[i]) < 0.01:
            dom = "-"
        elif abs(mean_attn[i]) > abs(mean_mlp[i]):
            dom = "attn"
        else:
            dom = "MLP"
        print(
            f"{i:>3} {t:>8} {mean_attn[i]:>+8.3f} {mean_mlp[i]:>+8.3f} {dom:>9}"
        )

    # Predictive-validation: where does ATTENTION ablation peak?
    attn_peak = int(np.argmin(mean_attn))
    attn_top5 = sorted(
        [(int(i), round(float(mean_attn[i]), 3)) for i in range(n_layers)],
        key=lambda x: x[1],
    )[:5]
    print("\n--- Attention-ablation predictive summary (000188) ---")
    print(f"  Predicted pivot:        L{PREDICTED_PIVOT}")
    print(f"  Observed attn peak:     L{attn_peak}  mean Δ={mean_attn[attn_peak]:+.3f}")
    print(f"  Top-5 attn peaks:       {attn_top5}")
    print()
    for label, idx in [
        (f"L{PREDICTED_PIVOT} (predicted pivot)", PREDICTED_PIVOT),
        (f"L{NULL_FRESH_KV_GLOBAL} (null: fresh-K/V global)", NULL_FRESH_KV_GLOBAL),
        (f"L{NEXT_GLOBAL_AFTER_PIVOT} (first KV-shared global)", NEXT_GLOBAL_AFTER_PIVOT),
    ]:
        is_global = idx in global_layers
        marker = "GLOBAL" if is_global else "local"
        print(
            f"    {label:48s} attn Δ={mean_attn[idx]:>+7.3f}  "
            f"median={median_attn[idx]:>+7.3f}  ({marker})"
        )


if __name__ == "__main__":
    main()
