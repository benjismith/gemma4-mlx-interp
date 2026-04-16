"""Per-layer zero-ablation experiment on Gemma 4 E4B.

For each of the 42 layers, skip that layer's contribution to the residual
stream and measure the resulting loss (negative log-prob of the model's own
top-1 prediction) on a battery of 15 factual-recall prompts.

42 layers × 15 prompts = 630 forward passes (~2 minutes on M2 Pro).

Run from project root:
    python experiments/step_02_layer_ablation.py
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gemma4_mlx_interp import Ablate, GLOBAL_LAYERS, Model, N_LAYERS  # noqa: E402

OUT_DIR = ROOT / "caches"
MIN_CONFIDENCE = 0.5

PROMPTS = [
    "Complete this sentence with one word: The Eiffel Tower is in",
    "Complete this sentence with one word: The capital of Japan is",
    "Complete this sentence with one word: The Great Wall is in",
    "Complete this sentence with one word: The Amazon River flows through",
    "Complete this sentence with one word: The Sahara Desert is in",
    "Complete this sentence with one word: Water is made of hydrogen and",
    "Complete this sentence with one word: The speed of light is measured in",
    "Complete this sentence with one word: The chemical symbol for gold is",
    "Complete this sentence with one word: Romeo and Juliet was written by",
    "Complete this sentence with one word: The Mona Lisa was painted by",
    "Complete this sentence with one word: One, two, three, four,",
    "Complete this sentence with one word: Monday, Tuesday,",
    "Complete this sentence with one word: The opposite of hot is",
    "Complete this sentence with one word: The color of the sky on a clear day is",
    "Complete this sentence with one word: Cats are popular household",
]


def _last_logp(logits: mx.array) -> np.ndarray:
    """Final-position log-probabilities, normalized. Shape [vocab_size]."""
    last = logits[0, -1, :].astype(mx.float32)
    lp = last - mx.logsumexp(last)
    mx.eval(lp)
    return np.array(lp)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading model...")
    model = Model.load()

    # ---- Validate prompts: keep ones the model is confident about ----
    print(f"\nValidating {len(PROMPTS)} prompts...\n")
    valid = []  # (input_ids, target_id, baseline_lp)
    for prompt in PROMPTS:
        ids = model.tokenize(prompt)
        result = model.run(ids)
        lp_np = _last_logp(result.logits)
        top1_id = int(np.argmax(lp_np))
        top1_prob = float(np.exp(lp_np[top1_id]))
        top1_tok = model.tokenizer.decode([top1_id])
        status = "OK" if top1_prob >= MIN_CONFIDENCE else "SKIP"
        print(f"  [{status}] {prompt[:55]:55s}  top1={top1_tok!r:15s} p={top1_prob:.3f}")
        if top1_prob >= MIN_CONFIDENCE:
            valid.append((ids, top1_id, float(lp_np[top1_id])))

    print(f"\n{len(valid)} / {len(PROMPTS)} prompts validated.\n")

    # ---- Sweep: ablate each layer, measure damage on each prompt ----
    loss_delta = np.zeros((N_LAYERS, len(valid)), dtype=np.float64)
    print(f"Running {N_LAYERS} × {len(valid)} = {N_LAYERS * len(valid)} ablated forward passes...")
    t0 = time.perf_counter()

    for i in range(N_LAYERS):
        ablation = Ablate.layer(i)
        for j, (ids, target_id, baseline_lp) in enumerate(valid):
            result = model.run(ids, interventions=[ablation])
            lp_np = _last_logp(result.logits)
            loss_delta[i, j] = float(lp_np[target_id]) - baseline_lp

        if (i + 1) % 6 == 0 or i == N_LAYERS - 1:
            elapsed = time.perf_counter() - t0
            eta = elapsed / (i + 1) * (N_LAYERS - i - 1)
            mean_d = float(np.mean(loss_delta[i]))
            print(f"  layer {i:>2}: mean Δlogp = {mean_d:>+8.3f}  "
                  f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]")

    total = time.perf_counter() - t0
    print(f"\nDone in {total:.0f}s ({total / (N_LAYERS * len(valid)) * 1000:.0f}ms per pass)")

    # ---- Aggregate + summary table ----
    mean_delta = np.mean(loss_delta, axis=1)

    print(f"\n{'layer':>5}  {'type':>7}  {'mean_Δlogp':>11}  {'median_Δlogp':>13}")
    print("-" * 45)
    for i in range(N_LAYERS):
        kind = "GLOBAL" if i in GLOBAL_LAYERS else "local"
        med = float(np.median(loss_delta[i]))
        print(f"{i:>5}  {kind:>7}  {mean_delta[i]:>+11.3f}  {med:>+13.3f}")

    # Summary stats
    global_idx = list(GLOBAL_LAYERS)
    global_deltas = mean_delta[global_idx]
    local_mask = np.ones(N_LAYERS, dtype=bool)
    local_mask[global_idx] = False
    local_deltas = mean_delta[local_mask]

    print(f"\n--- Summary ---")
    print(f"  Global layers (n={len(GLOBAL_LAYERS)}):  mean Δlogp = {np.mean(global_deltas):>+.3f}, "
          f"median = {np.median(global_deltas):>+.3f}")
    print(f"  Local layers  (n={N_LAYERS - len(GLOBAL_LAYERS)}): mean Δlogp = {np.mean(local_deltas):>+.3f}, "
          f"median = {np.median(local_deltas):>+.3f}")

    most_damaging = np.argsort(mean_delta)[:5]
    print(f"\n  5 most damaging layers to ablate:")
    for idx in most_damaging:
        kind = "GLOBAL" if idx in GLOBAL_LAYERS else "local"
        print(f"    layer {idx:>2} ({kind:>6}): mean Δlogp = {mean_delta[idx]:>+.3f}")
    n_global = sum(1 for idx in most_damaging if idx in GLOBAL_LAYERS)
    print(f"\n  {n_global} / 5 most damaging are global-attention layers")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#d62728" if i in GLOBAL_LAYERS else "#1f77b4" for i in range(N_LAYERS)]
    ax.bar(range(N_LAYERS), mean_delta, color=colors, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("layer index")
    ax.set_ylabel("mean Δ log p(target)  ← more damaging")
    ax.set_title("Layer ablation impact — Gemma 4 E4B (15 prompts)")
    ax.set_xticks(range(0, N_LAYERS, 3))
    ax.grid(True, alpha=0.3, axis="y")

    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(color="#d62728", label="global attention"),
                 Patch(color="#1f77b4", label="local (sliding window)")],
        loc="lower left",
    )

    plt.tight_layout()
    out_path = OUT_DIR / "layer_ablation.png"
    fig.savefig(out_path, dpi=140)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
