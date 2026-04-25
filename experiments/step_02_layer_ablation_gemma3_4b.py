"""Step 02 (Gemma 3 4B port): layer-ablation damage curve.

Mirrors `export_step_02_for_ui.py` against Gemma 3 4B instead of
Gemma 4 E4B. The methodology is identical:

  - Validate FACTUAL_15 prompts under MIN_CONFIDENCE = 0.5.
  - For each validated prompt, ablate each of the 34 layers and
    record Δ log p of that prompt's top-1.
  - Aggregate to mean and median per layer; emit a
    LayerAblationPayload to mechbench-ui/public/data/.

Bypasses mechbench-core's hook system because mechbench-core is
gemma4-only at the forward-path level. Ablation is implemented
by replacing `lm.layers[i]` with an identity callable in the
Python list mlx-vlm's Gemma3Model iterates.

This is the FACTUAL_15-battery follow-up to the single-prompt
probe at bin/probe_gemma3_4b_ablation.py (task 000187), per
task 000189.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_vlm import load

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.prompts.factual import FACTUAL_15  # noqa: E402
from mechbench_schema import (  # noqa: E402
    AblationPrompt,
    LayerAblationPayload,
    LayerAggregates,
)


MODEL_ID = "mlx-community/gemma-3-4b-it-bf16"
MIN_CONFIDENCE = 0.5

# Gemma 3 4B has globals at [5, 11, 17, 23, 29] (period 6, 5:1
# local:global). 34 layers total; the last layer (33) is sliding —
# unlike Gemma 4 where the last layer is always global.
GEMMA3_4B_GLOBAL_LAYERS = (5, 11, 17, 23, 29)


def last_logp(logits: mx.array) -> np.ndarray:
    last = logits[0, -1, :].astype(mx.float32)
    lp = last - mx.logsumexp(last)
    mx.eval(lp)
    return np.array(lp)


def chat_encode(tokenizer, text: str) -> mx.array:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return mx.array([tokenizer.encode(rendered, add_special_tokens=False)])


def resolve_output_path() -> Path:
    here = Path(__file__).resolve()
    tree_root = here.parent.parent.parent
    ui_data_dir = tree_root / "mechbench-ui" / "public" / "data"
    ui_data_dir.mkdir(parents=True, exist_ok=True)
    return ui_data_dir / "step_02_layer_ablation_gemma3_4b.json"


def main() -> None:
    output_path = resolve_output_path()
    print(f"Output target: {output_path}")

    print(f"Loading {MODEL_ID}...")
    t0 = time.time()
    model, processor = load(MODEL_ID)
    tokenizer = processor.tokenizer
    print(f"Loaded in {time.time() - t0:.1f}s")

    lm = model.language_model.model
    layers = lm.layers
    n_layers = len(layers)
    print(f"{n_layers} transformer layers")

    def run_forward(ids: mx.array) -> mx.array:
        return model.language_model(ids).logits

    # Validation pass — keep prompts where top-1 prob >= MIN_CONFIDENCE.
    prompts = FACTUAL_15.prompts
    validated: list[tuple[str, str, int, float]] = []
    print(f"Validating {len(prompts)} prompts (top-1 prob >= {MIN_CONFIDENCE})...")
    for prompt in prompts:
        ids = chat_encode(tokenizer, prompt.text)
        result = run_forward(ids)
        lp = last_logp(result)
        top1_id = int(np.argmax(lp))
        top1_prob = float(np.exp(lp[top1_id]))
        top1_token = tokenizer.decode([top1_id])
        keep = top1_prob >= MIN_CONFIDENCE
        marker = "✓" if keep else "✗"
        print(
            f"  {marker} prob={top1_prob:.3f} top1={top1_token!r:>16}  "
            f"target={prompt.target!r:>16}  '{prompt.text[:60]}'"
        )
        if keep:
            validated.append((prompt.text, prompt.target, top1_id, float(lp[top1_id])))

    n = len(validated)
    print(f"{n}/{len(prompts)} prompts validated.")

    # Ablation pass.
    damage = np.zeros((n_layers, n), dtype=np.float32)

    def identity(h, *_args, **_kwargs):
        return h

    print(f"Running {n_layers} x {n} = {n_layers * n} ablated forward passes...")
    for layer_idx in range(n_layers):
        original = layers[layer_idx]
        layers[layer_idx] = identity  # type: ignore[assignment]
        try:
            for j, (text, _target, top1_id, baseline_lp) in enumerate(validated):
                ids = chat_encode(tokenizer, text)
                result = run_forward(ids)
                lp = last_logp(result)
                damage[layer_idx, j] = float(lp[top1_id]) - baseline_lp
        finally:
            layers[layer_idx] = original
        if (layer_idx + 1) % 5 == 0 or layer_idx == n_layers - 1:
            print(
                f"  layer {layer_idx:2d}: mean Δlogp = "
                f"{float(damage[layer_idx].mean()):+.3f}"
            )

    payload = LayerAblationPayload(
        experiment="step_02_layer_ablation_gemma3_4b",
        description=(
            "Per-layer ablation: replace each of Gemma 3 4B's 34 transformer "
            "blocks with the identity (residual unchanged) and measure Δ log "
            "p of the model's own top-1 prediction across validated factual-"
            "recall prompts. More negative = more damaging to ablate. "
            "Counterpart to the Gemma 4 E4B step_02 figure."
        ),
        model=MODEL_ID,
        n_layers=n_layers,
        global_layers=list(GEMMA3_4B_GLOBAL_LAYERS),
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

    mean = damage.mean(axis=1)
    peak = int(np.argmin(mean))
    print(f"Peak mean damage at L{peak}: {float(mean[peak]):+.3f}")
    order = np.argsort(mean)
    top5 = [(int(i), round(float(mean[i]), 3)) for i in order[:5]]
    print(f"Top-5 most damaging (by mean): {top5}")


if __name__ == "__main__":
    main()
