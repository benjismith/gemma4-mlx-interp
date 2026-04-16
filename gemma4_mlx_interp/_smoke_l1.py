"""Smoke test for L1 declarative interventions.

For each intervention type, compare against the corresponding hand-rolled
forward in the original experiment scripts. The new interventions should
produce numerically identical logits — that's the strongest possible
"these are equivalent" test.

Eight checks:
  1. Ablate.layer(14)             vs step_02.run_ablated_forward
  2. Ablate.attention(23)         vs step_04.run_sublayer_ablated (attn)
  3. Ablate.mlp(14)               vs step_04.run_sublayer_ablated (mlp)
  4. Ablate.head(29, head=7)      vs step_07.run_head_ablated
  5. Ablate.side_channel()        vs step_03.run_side_channel_ablated (all)
  6. Capture.attn_weights([23])   vs step_05.run_with_attention_weights
  7. Patch.position(10, 13, ...)  vs step_09.forward_with_patch
  8. Composition: Ablate.head + Capture.per_head_out at the same layer
     -> captured tensor's ablated head slice is all zeros

Run from project root with the venv active:
    python -m gemma4_mlx_interp._smoke_l1
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reference paths from the existing experiments.
from experiments.step_02_layer_ablation import run_ablated_forward  # noqa: E402
from experiments.step_03_side_channel_ablation import run_side_channel_ablated  # noqa: E402
from experiments.step_04_sublayer_ablation import run_sublayer_ablated  # noqa: E402
from experiments.step_05_attention_patterns import run_with_attention_weights  # noqa: E402
from experiments.step_07_single_head_ablation import run_head_ablated  # noqa: E402
from experiments.step_09_causal_tracing import forward_with_patch  # noqa: E402
from hooks import run_with_cache  # noqa: E402

from . import Ablate, Capture, Model, Patch  # noqa: E402

PROMPT = "Complete this sentence with one word: The Eiffel Tower is in"
PROMPT_CLEAN = PROMPT
PROMPT_CORRUPT = "Complete this sentence with one word: The Great Wall is in"

# bf16 path identity should give bitwise-equal logits for identical arithmetic.
TOLERANCE = 1e-3


def _last_logits_np(logits: mx.array) -> np.ndarray:
    return np.array(logits[0, -1, :].astype(mx.float32))


def _check(name: str, ref: np.ndarray, run: np.ndarray) -> bool:
    delta = float(np.max(np.abs(ref - run)))
    arg_match = int(np.argmax(ref)) == int(np.argmax(run))
    ok = delta < TOLERANCE and arg_match
    marker = "OK" if ok else "FAIL"
    print(f"  [{marker:>4}] {name:<60s}  max|Δ|={delta:.6e}  argmax_match={arg_match}")
    return ok


def main() -> int:
    print("Loading model...")
    t0 = time.perf_counter()
    model = Model.load()
    print(f"Loaded in {time.perf_counter() - t0:.1f}s.\n")

    ids = model.tokenize(PROMPT)
    ids_clean = model.tokenize(PROMPT_CLEAN)
    ids_corrupt = model.tokenize(PROMPT_CORRUPT)

    all_pass = True

    # ---- 1. Ablate.layer(14) ----
    ref = _last_logits_np(run_ablated_forward(model._model, ids, ablate_layer=14))
    run = _last_logits_np(model.run(ids, interventions=[Ablate.layer(14)]).logits)
    all_pass &= _check("Ablate.layer(14)", ref, run)

    # ---- 2. Ablate.attention(23) ----
    ref = _last_logits_np(
        run_sublayer_ablated(model._model, ids, ablate_attn_layer=23)
    )
    run = _last_logits_np(model.run(ids, interventions=[Ablate.attention(23)]).logits)
    all_pass &= _check("Ablate.attention(23)", ref, run)

    # ---- 3. Ablate.mlp(14) ----
    ref = _last_logits_np(
        run_sublayer_ablated(model._model, ids, ablate_mlp_layer=14)
    )
    run = _last_logits_np(model.run(ids, interventions=[Ablate.mlp(14)]).logits)
    all_pass &= _check("Ablate.mlp(14)", ref, run)

    # ---- 4. Ablate.head(29, head=7) ----
    ref = _last_logits_np(
        run_head_ablated(model._model, ids, ablate_layer=29, ablate_head=7)
    )
    run = _last_logits_np(
        model.run(ids, interventions=[Ablate.head(29, head=7)]).logits
    )
    all_pass &= _check("Ablate.head(29, head=7)", ref, run)

    # ---- 5. Ablate.side_channel() (all layers) ----
    ref = _last_logits_np(
        run_side_channel_ablated(model._model, ids, ablate_layers=None)
    )
    run = _last_logits_np(
        model.run(ids, interventions=[Ablate.side_channel()]).logits
    )
    all_pass &= _check("Ablate.side_channel() (all layers)", ref, run)

    # ---- 6. Capture.attn_weights([23]) -> shape + numerical equivalence ----
    _, ref_attn = run_with_attention_weights(model._model, ids, target_layers=[23])
    ref_w = np.array(ref_attn[23].astype(mx.float32))
    result = model.run(ids, interventions=[Capture.attn_weights([23])])
    run_w = np.array(result.cache["blocks.23.attn.weights"].astype(mx.float32))
    delta_w = float(np.max(np.abs(ref_w - run_w)))
    shape_match = ref_w.shape == run_w.shape
    ok_w = shape_match and delta_w < TOLERANCE
    print(f"  [{'OK' if ok_w else 'FAIL':>4}] Capture.attn_weights([23])"
          f"{'':<32s}  shape={run_w.shape}  max|Δ|={delta_w:.6e}")
    all_pass &= ok_w

    # ---- 7. Patch.position(layer=10, pos=13) for causal tracing ----
    # Build clean cache via L1 Capture, then patch into corrupt run.
    clean_result = model.run(
        ids_clean, interventions=[Capture.residual(layers=range(42), point="post")],
    )
    # Reference: step_09's forward_with_patch needs clean_resid_post as a
    # dict[layer_idx -> tensor]. Build it from the prototype hooks path.
    _, prototype_clean_cache = run_with_cache(model._model, ids_clean)
    clean_resid_dict = {
        i: prototype_clean_cache[f"blocks.{i}.resid_post"] for i in range(42)
    }
    ref = _last_logits_np(forward_with_patch(
        model._model, ids_corrupt,
        clean_resid_post=clean_resid_dict, patch_layer=10, patch_position=13,
    ))
    run = _last_logits_np(model.run(ids_corrupt, interventions=[
        Patch.position(layer=10, position=13, source=clean_result.cache),
    ]).logits)
    all_pass &= _check("Patch.position(layer=10, pos=13)", ref, run)

    # ---- 8. Composition: Ablate.head + Capture.per_head_out same layer ----
    result = model.run(ids, interventions=[
        Ablate.head(29, head=7),
        Capture.per_head_out([29]),
    ])
    captured = np.array(result.cache["blocks.29.attn.per_head_out"].astype(mx.float32))
    head7_zero = bool(np.all(captured[:, 7, :, :] == 0))
    other_heads_nonzero = bool(np.any(captured[:, 0, :, :] != 0))
    ok_comp = head7_zero and other_heads_nonzero
    print(f"  [{'OK' if ok_comp else 'FAIL':>4}] Composition: Ablate.head + Capture.per_head_out"
          f"  head7_all_zero={head7_zero}  other_heads_have_signal={other_heads_nonzero}")
    all_pass &= ok_comp

    print()
    if not all_pass:
        print("L1 SMOKE TEST FAILED. Some intervention does not match its reference.")
        return 1

    print("L1 smoke test passed. All interventions are numerically equivalent")
    print("to their hand-rolled reference forwards. L1 is ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
