"""Position-wise logit lens: where does the answer appear in the residual stream?

Instead of only projecting the final token's residual through the unembed
(standard logit lens), projects EVERY position at EVERY layer. Produces a
heatmap of the answer token's log-probability across [layer × position].

Key hypothesis: the answer (e.g. "Paris") appears at the subject-entity
position (e.g. "Tower") in the middle layers (10-24), before it appears at
the final prediction position. This would be evidence that MLPs write
associative-memory lookups at the subject position.

Run from project root:
    python experiments/step_08_position_logit_lens.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forward import load_model, _tokenize  # noqa: E402
from hooks import run_with_cache  # noqa: E402
from mlx_vlm.models.gemma4.language import logit_softcap  # noqa: E402

GLOBAL_LAYERS = [5, 11, 17, 23, 29, 35, 41]
N_LAYERS = 42
OUT_DIR = ROOT / "caches"

PROMPTS_WITH_SUBJECTS = [
    ("Complete this sentence with one word: The Eiffel Tower is in",
     ["Eiffel", "Tower"], "Paris"),
    ("Complete this sentence with one word: The capital of Japan is",
     ["capital", "Japan"], "Tokyo"),
    ("Complete this sentence with one word: Romeo and Juliet was written by",
     ["Romeo", "Juliet"], "Shakespeare"),
    ("Complete this sentence with one word: The chemical symbol for gold is",
     ["chemical", "symbol", "gold"], "Au"),
    ("Complete this sentence with one word: The opposite of hot is",
     ["opposite", "hot"], "cold"),
    ("Complete this sentence with one word: Monday, Tuesday,",
     ["Monday", "Tuesday"], "Wednesday"),
]


def project_to_logits(model, resid: mx.array) -> mx.array:
    lm = model.language_model
    tm = lm.model
    h = tm.norm(resid)
    logits = tm.embed_tokens.as_linear(h)
    if lm.final_logit_softcapping is not None:
        logits = logit_softcap(lm.final_logit_softcapping, logits)
    return logits


def get_token_labels(tokenizer, input_ids: mx.array) -> list:
    ids = input_ids[0].tolist()
    return [tokenizer.decode([tid]) for tid in ids]


def find_positions(token_labels, substrings):
    positions = []
    for i, label in enumerate(token_labels):
        if any(sub.lower() in label.lower() for sub in substrings):
            positions.append(i)
    return positions


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading model...")
    model, processor = load_model()
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    for prompt_idx, (prompt, subject_words, expected_answer) in enumerate(PROMPTS_WITH_SUBJECTS):
        print(f"\n{'=' * 60}")
        print(f"Prompt: {prompt!r}")

        input_ids = _tokenize(processor, model, prompt)
        token_labels = get_token_labels(tokenizer, input_ids)
        seq_len = input_ids.shape[1]
        subject_pos = find_positions(token_labels, subject_words)

        print(f"Tokens ({seq_len}): {token_labels}")
        print(f"Subject positions: {subject_pos} ({[token_labels[p] for p in subject_pos]})")

        # Find the actual target token ID (auto-detect from model output)
        logits_full, cache = run_with_cache(model, input_ids)
        last = logits_full[0, -1, :].astype(mx.float32)
        probs = mx.softmax(last)
        mx.eval(probs)
        probs_np = np.array(probs)
        target_id = int(np.argmax(probs_np))
        target_tok = tokenizer.decode([target_id])
        print(f"Target: {target_tok!r} (id={target_id}, p={float(probs_np[target_id]):.3f})")

        # Build the position × layer heatmap of the target token's log-probability.
        # heatmap[layer, position] = log p(target) at that (layer, position)
        heatmap_logp = np.full((N_LAYERS, seq_len), np.nan, dtype=np.float64)
        heatmap_rank = np.full((N_LAYERS, seq_len), np.nan, dtype=np.float64)

        for i in range(N_LAYERS):
            resid = cache[f"blocks.{i}.resid_post"]  # [1, seq_len, d_model]
            logits_i = project_to_logits(model, resid)  # [1, seq_len, vocab]
            logits_f32 = logits_i[0].astype(mx.float32)  # [seq_len, vocab]
            lp = logits_f32 - mx.logsumexp(logits_f32, axis=-1, keepdims=True)
            mx.eval(lp)
            lp_np = np.array(lp)  # [seq_len, vocab]

            for pos in range(seq_len):
                target_lp = float(lp_np[pos, target_id])
                heatmap_logp[i, pos] = target_lp
                heatmap_rank[i, pos] = int(np.sum(lp_np[pos] > target_lp))

        # --- Print: where does the answer first appear at subject positions? ---
        print(f"\n  Target token rank at subject positions across layers:")
        print(f"  {'layer':>5}  ", end="")
        for p in subject_pos:
            print(f"  {token_labels[p]:>12s}", end="")
        print(f"  {'[final pos]':>12s}")
        print(f"  {'-' * (7 + 14 * (len(subject_pos) + 1))}")

        for i in list(range(0, N_LAYERS, 3)) + [N_LAYERS - 1]:
            print(f"  {i:>5}  ", end="")
            for p in subject_pos:
                rank = int(heatmap_rank[i, p])
                print(f"  {rank:>12d}", end="")
            rank_final = int(heatmap_rank[i, -1])
            print(f"  {rank_final:>12d}")

        # Find the layer where target first enters top-10 at each subject position
        print(f"\n  Layer where target first enters top-10:")
        for p in subject_pos:
            first_layer = None
            for i in range(N_LAYERS):
                if heatmap_rank[i, p] < 10:
                    first_layer = i
                    break
            tok = token_labels[p]
            if first_layer is not None:
                print(f"    pos {p:>2} ({tok:>12s}): layer {first_layer}")
            else:
                print(f"    pos {p:>2} ({tok:>12s}): never")

        first_final = None
        for i in range(N_LAYERS):
            if heatmap_rank[i, -1] < 10:
                first_final = i
                break
        print(f"    final position:          layer {first_final}")

        # --- Heatmap plot ---
        fig, axes = plt.subplots(1, 2, figsize=(max(12, seq_len * 0.6), 8))

        # Log-probability heatmap
        ax = axes[0]
        vmin, vmax = np.nanpercentile(heatmap_logp, [1, 99])
        im = ax.imshow(heatmap_logp, aspect="auto", cmap="RdYlGn",
                       vmin=max(vmin, -30), vmax=min(vmax, 0),
                       origin="lower", interpolation="nearest")
        ax.set_xlabel("token position")
        ax.set_ylabel("layer")
        ax.set_title(f"log p({target_tok!r}) at each position × layer")
        ax.set_xticks(range(seq_len))
        ax.set_xticklabels(token_labels, rotation=70, ha="right", fontsize=6)
        plt.colorbar(im, ax=ax, shrink=0.6, label="log p")

        # Mark subject positions
        for p in subject_pos:
            ax.axvline(p, color="red", linewidth=1.5, alpha=0.7, linestyle="--")
        # Mark global layers
        for g in GLOBAL_LAYERS:
            ax.axhline(g, color="gray", linewidth=0.5, alpha=0.5, linestyle=":")

        # Rank heatmap (log scale)
        ax = axes[1]
        log_rank = np.log10(heatmap_rank + 1)
        im = ax.imshow(log_rank, aspect="auto", cmap="RdYlGn_r",
                       vmin=0, vmax=5.5,
                       origin="lower", interpolation="nearest")
        ax.set_xlabel("token position")
        ax.set_title(f"rank of {target_tok!r} (log₁₀ scale)")
        ax.set_xticks(range(seq_len))
        ax.set_xticklabels(token_labels, rotation=70, ha="right", fontsize=6)
        plt.colorbar(im, ax=ax, shrink=0.6, label="log₁₀(rank + 1)")

        for p in subject_pos:
            ax.axvline(p, color="red", linewidth=1.5, alpha=0.7, linestyle="--")
        for g in GLOBAL_LAYERS:
            ax.axhline(g, color="gray", linewidth=0.5, alpha=0.5, linestyle=":")

        fig.suptitle(f"{prompt}\n→ {target_tok!r}  (red dashes = subject positions)",
                     fontsize=10)
        plt.tight_layout()
        out_path = OUT_DIR / f"position_logit_lens_{prompt_idx}.png"
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Wrote {out_path}")


if __name__ == "__main__":
    main()
