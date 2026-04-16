"""Multi-prompt logit lens: test whether the global-attention inflection pattern
generalizes across diverse prompts.

Runs the logit lens on a battery of prompts with known target completions,
aggregates per-layer rank and log-probability trajectories, and produces:
  1. Individual per-prompt rank trajectories (thin lines)
  2. Geometric-mean rank trajectory (bold line)
  3. Mean log-probability trajectory
  4. A summary table

The key hypothesis: the big inflection points in the lens trajectory land on
or just after the global-attention layers (5, 11, 17, 23, 29, 35, 41).

Run from project root:
    python experiments/logit_lens_batch.py
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forward import load_model, _tokenize, top_k_tokens  # noqa: E402
from hooks import run_with_cache  # noqa: E402
from mlx_vlm.models.gemma4.language import logit_softcap  # noqa: E402

GLOBAL_LAYERS = [5, 11, 17, 23, 29, 35, 41]
N_LAYERS = 42
OUT_DIR = ROOT / "caches"

# Prompts only — target token is auto-detected as the model's top-1
# prediction at the final layer. We skip any prompt where the model isn't
# confident (top-1 prob < 0.5) since a weak prediction makes the lens
# trajectory noisy and hard to interpret.
PROMPTS = [
    # Factual recall — geography
    "Complete this sentence with one word: The Eiffel Tower is in",
    "Complete this sentence with one word: The capital of Japan is",
    "Complete this sentence with one word: The Great Wall is in",
    "Complete this sentence with one word: The Amazon River flows through",
    "Complete this sentence with one word: The Sahara Desert is in",
    # Factual recall — science
    "Complete this sentence with one word: Water is made of hydrogen and",
    "Complete this sentence with one word: The speed of light is measured in",
    "Complete this sentence with one word: The chemical symbol for gold is",
    # Factual recall — culture
    "Complete this sentence with one word: Romeo and Juliet was written by",
    "Complete this sentence with one word: The Mona Lisa was painted by",
    # Linguistic / pattern completion
    "Complete this sentence with one word: One, two, three, four,",
    "Complete this sentence with one word: Monday, Tuesday,",
    # Common sense
    "Complete this sentence with one word: The opposite of hot is",
    "Complete this sentence with one word: The color of the sky on a clear day is",
    "Complete this sentence with one word: Cats are popular household",
]

MIN_CONFIDENCE = 0.5


def project_to_logits(model, resid: mx.array) -> mx.array:
    lm = model.language_model
    tm = lm.model
    h = tm.norm(resid)
    logits = tm.embed_tokens.as_linear(h)
    if lm.final_logit_softcapping is not None:
        logits = logit_softcap(lm.final_logit_softcapping, logits)
    return logits


def compute_lens_trajectory(model, cache, target_id: int):
    """Return (ranks, logprobs) arrays of shape [N_LAYERS]."""
    ranks = np.zeros(N_LAYERS, dtype=np.int64)
    logprobs = np.zeros(N_LAYERS, dtype=np.float64)

    for i in range(N_LAYERS):
        resid = cache[f"blocks.{i}.resid_post"]
        logits_i = project_to_logits(model, resid)
        last = logits_i[0, -1, :].astype(mx.float32)
        lp = last - mx.logsumexp(last)
        mx.eval(lp)
        lp_np = np.array(lp)

        target_lp = float(lp_np[target_id])
        rank = int(np.sum(lp_np > target_lp))
        ranks[i] = rank
        logprobs[i] = target_lp

    return ranks, logprobs


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading model...")
    model, processor = load_model()
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    # Run each prompt, auto-detect target as top-1 at the final layer,
    # keep only prompts where the model is confident.
    valid_prompts = []
    print(f"\nValidating {len(PROMPTS)} prompts...\n")

    for prompt in PROMPTS:
        input_ids = _tokenize(processor, model, prompt)
        logits, cache = run_with_cache(model, input_ids)

        last_lp = logits[0, -1, :].astype(mx.float32)
        probs = mx.softmax(last_lp)
        mx.eval(probs)
        probs_np = np.array(probs)

        top1_id = int(np.argmax(probs_np))
        top1_prob = float(probs_np[top1_id])
        top1_tok = tokenizer.decode([top1_id])

        status = "OK" if top1_prob >= MIN_CONFIDENCE else "SKIP"
        print(f"  [{status}] {prompt[:55]:55s}  top1={top1_tok!r:15s} p={top1_prob:.3f}")

        if top1_prob >= MIN_CONFIDENCE:
            valid_prompts.append((prompt, top1_tok, top1_id, cache))

    print(f"\n{len(valid_prompts)} / {len(PROMPTS)} prompts validated.\n")

    if len(valid_prompts) < 3:
        print("Too few valid prompts to draw conclusions. Exiting.")
        return

    # Compute per-layer trajectories for all valid prompts.
    all_ranks = np.zeros((len(valid_prompts), N_LAYERS), dtype=np.float64)
    all_logprobs = np.zeros((len(valid_prompts), N_LAYERS), dtype=np.float64)

    for j, (prompt, target_tok, target_id, cache) in enumerate(valid_prompts):
        ranks, logprobs = compute_lens_trajectory(model, cache, target_id)
        all_ranks[j] = ranks
        all_logprobs[j] = logprobs

    # Aggregate: geometric mean of (rank + 1) then subtract 1, to handle rank=0.
    log_rank_plus1 = np.log(all_ranks + 1)
    geomean_rank = np.exp(np.mean(log_rank_plus1, axis=0)) - 1
    mean_logprob = np.mean(all_logprobs, axis=0)

    # --- Summary table ---
    print(f"\n{'layer':>5}  {'type':>7}  {'geomean_rank':>13}  {'mean_logp':>10}")
    print("-" * 42)
    table_layers = sorted(set(list(range(0, N_LAYERS, 6)) + [N_LAYERS - 1]))
    for i in table_layers:
        kind = "GLOBAL" if i in GLOBAL_LAYERS else "local"
        print(f"{i:>5}  {kind:>7}  {geomean_rank[i]:>13.1f}  {mean_logprob[i]:>10.3f}")

    # --- Compute per-layer rank deltas to find biggest inflection points ---
    rank_delta = np.diff(geomean_rank)
    biggest_drops = np.argsort(rank_delta)[:5]
    print(f"\nLargest rank drops (layer i → i+1):")
    for idx in biggest_drops:
        kind = "GLOBAL" if (idx + 1) in GLOBAL_LAYERS else "local"
        print(f"  layer {idx:>2} → {idx+1:>2} ({kind:>6}): "
              f"geomean rank {geomean_rank[idx]:.0f} → {geomean_rank[idx+1]:.0f}  "
              f"(Δ = {rank_delta[idx]:.0f})")

    n_global_in_top5 = sum(1 for idx in biggest_drops if (idx + 1) in GLOBAL_LAYERS)
    print(f"\n  {n_global_in_top5} / 5 biggest drops land on global-attention layers")

    # --- Plot ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    layers_x = np.arange(N_LAYERS)

    # Panel 1: rank trajectories
    ax = axes[0]
    for j in range(len(valid_prompts)):
        rank_plot = np.where(all_ranks[j] == 0, 0.5, all_ranks[j])
        ax.semilogy(layers_x, rank_plot, color="#1f77b4", alpha=0.2, linewidth=0.8)
    geomean_plot = np.where(geomean_rank < 0.5, 0.5, geomean_rank)
    ax.semilogy(layers_x, geomean_plot, color="#d62728", linewidth=2.5,
                label=f"geometric mean (n={len(valid_prompts)})")
    ax.set_ylabel("rank of target token (log)")
    ax.set_title(f"Logit lens across {len(valid_prompts)} prompts — Gemma 4 E4B")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel 2: logprob trajectories
    ax = axes[1]
    for j in range(len(valid_prompts)):
        ax.plot(layers_x, all_logprobs[j], color="#2ca02c", alpha=0.2, linewidth=0.8)
    ax.plot(layers_x, mean_logprob, color="#d62728", linewidth=2.5,
            label=f"mean (n={len(valid_prompts)})")
    ax.set_ylabel("log p(target)")
    ax.set_xlabel("layer index")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    for a in axes:
        for g in GLOBAL_LAYERS:
            a.axvline(g, color="#999999", linestyle="--", linewidth=0.7, alpha=0.6)

    plt.tight_layout()
    out_path = OUT_DIR / "logit_lens_batch.png"
    fig.savefig(out_path, dpi=140)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
