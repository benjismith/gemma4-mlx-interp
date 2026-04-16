"""Attention pattern analysis for global layers in Gemma 4 E4B.

Captures attention weights at each of the 7 global-attention layers across
6 factual-recall prompts. For each (prompt, global-layer) pair, plots the
attention from the final sequence position (mean over heads).

Key finding (per docs/findings/step_05_attention_patterns.md): the global
layers attend predominantly to chat-template tokens (`user`, `<|turn>`,
`<turn|>`, `model`, newlines), not to subject-entity content tokens.

Run from project root:
    python experiments/step_05_attention_patterns.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gemma4_mlx_interp import (  # noqa: E402
    GLOBAL_LAYERS, Capture, Model, Prompt, PromptSet,
)

OUT_DIR = ROOT / "caches"

# Six FACTUAL_15-style prompts spanning landmark / capital / author /
# element / opposite / sequence categories — chosen to surface any
# prompt-specific structure in the attention patterns.
ATTN_DEMO = PromptSet(name="ATTN_DEMO", prompts=(
    Prompt(text="Complete this sentence with one word: The Eiffel Tower is in", target="Paris"),
    Prompt(text="Complete this sentence with one word: The capital of Japan is", target="Tokyo"),
    Prompt(text="Complete this sentence with one word: Romeo and Juliet was written by", target="Shakespeare"),
    Prompt(text="Complete this sentence with one word: The chemical symbol for gold is", target="Au"),
    Prompt(text="Complete this sentence with one word: The opposite of hot is", target="cold"),
    Prompt(text="Complete this sentence with one word: Monday, Tuesday,", target="Wednesday"),
))


def get_token_labels(tokenizer, input_ids: mx.array) -> list[str]:
    """Per-token decoded labels, truncated for plot readability."""
    labels = []
    for tid in input_ids[0].tolist():
        tok = tokenizer.decode([tid])
        labels.append(tok[:10] + ".." if len(tok) > 12 else tok)
    return labels


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading model...")
    model = Model.load()
    capture = Capture.attn_weights(layers=list(GLOBAL_LAYERS))

    for prompt_idx, prompt in enumerate(ATTN_DEMO):
        print(f"\n{'=' * 60}")
        print(f"Prompt: {prompt.text!r}")
        ids = model.tokenize(prompt.text)
        token_labels = get_token_labels(model.tokenizer, ids)
        seq_len = ids.shape[1]
        print(f"Tokens ({seq_len}): {token_labels}")

        result = model.run(ids, interventions=[capture])

        # Sanity: top-1 prediction
        last = result.last_logits.astype(mx.float32)
        probs = mx.softmax(last)
        mx.eval(probs)
        probs_np = np.array(probs)
        top1_id = int(np.argmax(probs_np))
        top1_tok = model.tokenizer.decode([top1_id])
        print(f"Prediction: {top1_tok!r} (p={float(probs_np[top1_id]):.3f})")

        # Plot: one row per global layer, attention from FINAL position (mean over heads).
        fig, axes = plt.subplots(
            len(GLOBAL_LAYERS), 1,
            figsize=(max(10, seq_len * 0.5), len(GLOBAL_LAYERS) * 1.8),
        )
        if len(GLOBAL_LAYERS) == 1:
            axes = [axes]

        for row, layer_idx in enumerate(GLOBAL_LAYERS):
            ax = axes[row]
            w = result.cache[f"blocks.{layer_idx}.attn.weights"]  # [1, n_heads, L, S_kv]
            w_np = np.array(w[0, :, -1, :].astype(mx.float32))  # [n_heads, S_kv] @ final pos
            mean_w = np.mean(w_np, axis=0)
            ax.bar(
                range(seq_len), mean_w,
                color="#d62728" if layer_idx == 23 else "#1f77b4",
                alpha=0.8,
            )
            ax.set_ylabel(f"L{layer_idx}", fontsize=9, rotation=0, labelpad=25)
            ax.set_ylim(0, min(1.0, np.max(mean_w) * 1.3 + 0.01))
            ax.set_xlim(-0.5, seq_len - 0.5)
            if row == 0:
                short_prompt = prompt.text[:60] + ("..." if len(prompt.text) > 60 else "")
                ax.set_title(
                    f"Attention from final position (mean over heads)\n"
                    f"{short_prompt} → {top1_tok!r}",
                    fontsize=10,
                )
            if row == len(GLOBAL_LAYERS) - 1:
                ax.set_xticks(range(seq_len))
                ax.set_xticklabels(token_labels, rotation=60, ha="right", fontsize=7)
            else:
                ax.set_xticks([])
            ax.tick_params(axis="y", labelsize=7)

        plt.tight_layout()
        out_path = OUT_DIR / f"attn_pattern_{prompt_idx}.png"
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_path}")

        # Print L23 specifically: top-5 attended positions
        w23 = result.cache["blocks.23.attn.weights"]
        w23_np = np.array(w23[0, :, -1, :].astype(mx.float32))
        mean_w23 = np.mean(w23_np, axis=0)
        top_pos = np.argsort(-mean_w23)[:5]
        print(f"\n  Layer 23 — top-5 attended positions (from final token):")
        for pos in top_pos:
            print(f"    pos {pos:>2}: {token_labels[pos]:>15s}  weight={mean_w23[pos]:.4f}")

    # Summary plot: layer 23 attention across all 6 prompts
    print(f"\n{'=' * 60}")
    print("Generating summary plot (layer 23 across all prompts)...")

    fig, axes = plt.subplots(len(ATTN_DEMO), 1, figsize=(12, len(ATTN_DEMO) * 1.5))
    if len(ATTN_DEMO) == 1:
        axes = [axes]

    l23_capture = Capture.attn_weights(layers=[23])
    for prompt_idx, prompt in enumerate(ATTN_DEMO):
        ids = model.tokenize(prompt.text)
        token_labels = get_token_labels(model.tokenizer, ids)
        seq_len = ids.shape[1]

        result = model.run(ids, interventions=[l23_capture])
        last = result.last_logits.astype(mx.float32)
        probs = mx.softmax(last)
        mx.eval(probs)
        top1_tok = model.tokenizer.decode([int(np.argmax(np.array(probs)))])

        w23 = result.cache["blocks.23.attn.weights"]
        w23_np = np.array(w23[0, :, -1, :].astype(mx.float32))
        mean_w23 = np.mean(w23_np, axis=0)

        ax = axes[prompt_idx]
        ax.bar(range(seq_len), mean_w23, color="#d62728", alpha=0.85)
        ax.set_ylabel(f"→ {top1_tok!r}", fontsize=8, rotation=0, labelpad=45)
        ax.set_ylim(0, min(1.0, np.max(mean_w23) * 1.3 + 0.01))
        ax.set_xlim(-0.5, max(25, seq_len) - 0.5)
        ax.set_xticks(range(seq_len))
        if prompt_idx == len(ATTN_DEMO) - 1:
            ax.set_xticklabels(token_labels, rotation=60, ha="right", fontsize=7)
        else:
            ax.set_xticklabels(token_labels, rotation=60, ha="right", fontsize=6, alpha=0.5)
        ax.tick_params(axis="y", labelsize=7)

        if prompt_idx == 0:
            ax.set_title("Layer 23 attention from final position (mean over heads)",
                         fontsize=11)

    plt.tight_layout()
    out_path = OUT_DIR / "attn_layer23_summary.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
