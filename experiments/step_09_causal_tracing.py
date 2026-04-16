"""Causal tracing (activation patching) on Gemma 4 E4B.

For each paired (clean, corrupt) prompt:
  1. Run CLEAN, capture residual stream at every (layer, position).
  2. Run CORRUPT, but at one chosen (layer, position), patch in the clean
     resid_post and let the rest of the forward pass proceed.
  3. Measure recovery of the clean answer's probability.

Sweep all (layer, position) cells -> heatmap of where the factual
information causally lives. Per finding 09: two sharp hotspots — subject
position in early/middle layers, final position in late layers — with a
clean handoff around layer 29-30.

Run from project root:
    python experiments/step_09_causal_tracing.py
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

from gemma4_mlx_interp import (  # noqa: E402
    Capture, GLOBAL_LAYERS, Model, N_LAYERS, Patch, position_heatmap,
)

OUT_DIR = ROOT / "caches"

# (clean_text, corrupt_text). Both must tokenize to the same length so
# positions align under patching.
PAIRS = [
    ("Complete this sentence with one word: The Eiffel Tower is in",
     "Complete this sentence with one word: The Great Wall is in"),
    ("Complete this sentence with one word: The capital of Japan is",
     "Complete this sentence with one word: The capital of France is"),
    ("Complete this sentence with one word: Romeo and Juliet was written by",
     "Complete this sentence with one word: Pride and Prejudice was written by"),
]


def _top1(model, logits: mx.array) -> tuple[int, str, float]:
    last = logits[0, -1, :].astype(mx.float32)
    probs = mx.softmax(last)
    mx.eval(probs)
    probs_np = np.array(probs)
    idx = int(np.argmax(probs_np))
    return idx, model.tokenizer.decode([idx]), float(probs_np[idx])


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading model...")
    model = Model.load()

    capture_all = Capture.residual(layers=range(N_LAYERS), point="post")

    for pair_idx, (clean_text, corrupt_text) in enumerate(PAIRS):
        print(f"\n{'=' * 70}")
        print(f"Pair {pair_idx}: CLEAN   = {clean_text!r}")
        print(f"            CORRUPT = {corrupt_text!r}")

        clean_ids = model.tokenize(clean_text)
        corrupt_ids = model.tokenize(corrupt_text)
        seq_len = clean_ids.shape[1]
        assert clean_ids.shape[1] == corrupt_ids.shape[1], "Prompt lengths must match"

        token_labels_clean = [model.tokenizer.decode([int(t)]) for t in clean_ids[0]]
        token_labels_corrupt = [model.tokenizer.decode([int(t)]) for t in corrupt_ids[0]]

        diff_pos = [
            i for i in range(seq_len)
            if int(clean_ids[0, i]) != int(corrupt_ids[0, i])
        ]
        print(f"Diff positions: {diff_pos} "
              f"({[token_labels_clean[p] for p in diff_pos]} vs "
              f"{[token_labels_corrupt[p] for p in diff_pos]})")

        # Clean and corrupt baseline runs
        clean_result = model.run(clean_ids, interventions=[capture_all])
        corrupt_result = model.run(corrupt_ids)
        clean_top1_id, clean_answer, clean_prob = _top1(model, clean_result.logits)
        _, corrupt_answer, corrupt_prob = _top1(model, corrupt_result.logits)

        last_corrupt = corrupt_result.last_logits.astype(mx.float32)
        corrupt_probs = mx.softmax(last_corrupt)
        mx.eval(corrupt_probs)
        baseline_clean_in_corrupt = float(np.array(corrupt_probs)[clean_top1_id])

        print(f"Clean answer:   {clean_answer!r} (p={clean_prob:.3f})")
        print(f"Corrupt answer: {corrupt_answer!r} (p={corrupt_prob:.3f})")
        print(f"p(clean answer) in corrupt run baseline: {baseline_clean_in_corrupt:.4f}")

        # Sweep (layer, position) patches
        patch_results = np.zeros((N_LAYERS, seq_len), dtype=np.float64)
        print(f"\nRunning {N_LAYERS * seq_len} patched forward passes...")
        t0 = time.perf_counter()

        for L in range(N_LAYERS):
            for P in range(seq_len):
                patch = Patch.position(layer=L, position=P, source=clean_result.cache)
                r = model.run(corrupt_ids, interventions=[patch])
                last = r.last_logits.astype(mx.float32)
                probs = mx.softmax(last)
                mx.eval(probs)
                patch_results[L, P] = float(np.array(probs)[clean_top1_id])

            if (L + 1) % 7 == 0 or L == N_LAYERS - 1:
                elapsed = time.perf_counter() - t0
                eta = elapsed / (L + 1) * (N_LAYERS - L - 1)
                print(f"  layer {L:>2} done  [{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]")

        print(f"Done in {time.perf_counter() - t0:.0f}s")

        recovery = patch_results - baseline_clean_in_corrupt

        # Top patches
        flat = recovery.flatten()
        top_indices = np.argsort(-flat)[:10]
        print(f"\nTop-10 patches that most recover p({clean_answer!r}):")
        print(f"  {'layer':>5}  {'pos':>4}  {'token':>12}  {'recovery':>9}  {'p(clean)':>9}")
        for idx in top_indices:
            L, P = int(idx) // seq_len, int(idx) % seq_len
            print(f"  {L:>5}  {P:>4}  {token_labels_corrupt[P]:>12s}  "
                  f"{recovery[L, P]:>+8.3f}   {patch_results[L, P]:>7.3f}")

        # Two-panel heatmap using position_heatmap
        fig, axes = plt.subplots(1, 2, figsize=(max(12, seq_len * 0.6), 8))

        position_heatmap(
            patch_results, token_labels_corrupt, ax=axes[0],
            cmap="Greens", vmin=0, vmax=max(0.01, float(np.max(patch_results))),
            mark_positions=diff_pos, mark_layers=GLOBAL_LAYERS,
            colorbar_label="p",
            title=f"p({clean_answer!r}) after patching (corrupt -> clean)\n"
                  f"baseline: {baseline_clean_in_corrupt:.3f} in corrupt, "
                  f"{clean_prob:.3f} in clean",
        )

        max_abs = float(np.max(np.abs(recovery)))
        position_heatmap(
            recovery, token_labels_corrupt, ax=axes[1],
            cmap="RdBu_r", vmin=-max_abs * 0.5, vmax=max_abs * 0.5,
            mark_positions=diff_pos, mark_layers=GLOBAL_LAYERS,
            colorbar_label="Δp",
            title=f"recovery: Δ p({clean_answer!r}) vs unpatched corrupt run",
        )

        fig.suptitle(
            f"Causal tracing: clean = {clean_text[:40]}.. corrupt = {corrupt_text[:40]}..",
            fontsize=10,
        )
        plt.tight_layout()
        out_path = OUT_DIR / f"causal_trace_{pair_idx}.png"
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
