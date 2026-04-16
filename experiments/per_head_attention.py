"""Per-head attention analysis at global layers in Gemma 4 E4B.

Previous experiment averaged attention across heads and found template
tokens dominate. This experiment breaks open individual heads to check
for specialization — content-attending heads vs template-attending heads.

Also computes a "subject attention score" for every head at every global
layer: what fraction of attention from the final position lands on the
subject-entity tokens vs the chat-template tokens?

Run from project root:
    python experiments/per_head_attention.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forward import load_model, _tokenize  # noqa: E402
from experiments.attention_patterns import run_with_attention_weights, get_token_labels  # noqa: E402

GLOBAL_LAYERS = [5, 11, 17, 23, 29, 35, 41]
OUT_DIR = ROOT / "caches"

# (prompt, subject_token_substrings) — the subject tokens are identified
# by substring match against the token labels. This is imperfect but good
# enough for our 6 prompts.
PROMPTS_WITH_SUBJECTS = [
    ("Complete this sentence with one word: The Eiffel Tower is in",
     ["Eiffel", "Tower"]),
    ("Complete this sentence with one word: The capital of Japan is",
     ["capital", "Japan"]),
    ("Complete this sentence with one word: Romeo and Juliet was written by",
     ["Romeo", "Juliet", "written"]),
    ("Complete this sentence with one word: The chemical symbol for gold is",
     ["chemical", "symbol", "gold"]),
    ("Complete this sentence with one word: The opposite of hot is",
     ["opposite", "hot"]),
    ("Complete this sentence with one word: Monday, Tuesday,",
     ["Monday", "Tuesday"]),
]

TEMPLATE_SUBSTRINGS = ["<bos>", "<|turn>", "user", "<turn|>", "model"]


def find_token_positions(token_labels, substrings):
    """Return indices where any token label contains any of the substrings."""
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

    # Collect per-head attention weights for all prompts at all global layers.
    # Shape: [n_prompts, n_global_layers, n_heads, seq_len]
    all_data = []

    for prompt, subjects in PROMPTS_WITH_SUBJECTS:
        input_ids = _tokenize(processor, model, prompt)
        token_labels = get_token_labels(tokenizer, input_ids)
        seq_len = input_ids.shape[1]

        logits, attn_dict = run_with_attention_weights(
            model, input_ids, target_layers=GLOBAL_LAYERS
        )

        last = logits[0, -1, :].astype(mx.float32)
        probs = mx.softmax(last)
        mx.eval(probs)
        top1_tok = tokenizer.decode([int(np.argmax(np.array(probs)))])

        subject_pos = find_token_positions(token_labels, subjects)
        template_pos = find_token_positions(token_labels, TEMPLATE_SUBSTRINGS)

        prompt_data = {
            "prompt": prompt,
            "prediction": top1_tok,
            "token_labels": token_labels,
            "subject_pos": subject_pos,
            "template_pos": template_pos,
            "attn": {},  # layer_idx -> [n_heads, seq_len]
        }

        for layer_idx in GLOBAL_LAYERS:
            # [B, n_heads, L, S_kv] -> [n_heads, S_kv] at final position
            w = np.array(attn_dict[layer_idx][0, :, -1, :].astype(mx.float32))
            prompt_data["attn"][layer_idx] = w

        subj_labels = [token_labels[p] for p in subject_pos]
        print(f"  {prompt[:50]:50s} → {top1_tok!r:12s}  "
              f"subject pos: {subject_pos} ({subj_labels})")

        all_data.append(prompt_data)

    # --- Plot 1: All 8 heads at layer 23 for first prompt (Eiffel Tower) ---
    d = all_data[0]
    w23 = d["attn"][23]  # [n_heads, seq_len]
    n_heads = w23.shape[0]
    seq_len = len(d["token_labels"])

    fig, axes = plt.subplots(n_heads, 1, figsize=(max(10, seq_len * 0.5), n_heads * 1.5))
    for h in range(n_heads):
        ax = axes[h]
        colors = []
        for pos in range(seq_len):
            if pos in d["subject_pos"]:
                colors.append("#e74c3c")  # red for subject
            elif pos in d["template_pos"]:
                colors.append("#999999")  # gray for template
            else:
                colors.append("#3498db")  # blue for other
        ax.bar(range(seq_len), w23[h], color=colors, alpha=0.85)
        ax.set_ylabel(f"H{h}", fontsize=9, rotation=0, labelpad=20)
        ax.set_ylim(0, min(1.0, np.max(w23[h]) * 1.3 + 0.01))
        ax.set_xlim(-0.5, seq_len - 0.5)
        if h == 0:
            ax.set_title(f"Layer 23 per-head attention — {d['prompt'][:55]}.. → {d['prediction']!r}\n"
                         f"(red = subject entity, gray = template, blue = other)",
                         fontsize=10)
        if h == n_heads - 1:
            ax.set_xticks(range(seq_len))
            ax.set_xticklabels(d["token_labels"], rotation=60, ha="right", fontsize=7)
        else:
            ax.set_xticks([])
        ax.tick_params(axis="y", labelsize=7)

    plt.tight_layout()
    out_path = OUT_DIR / "per_head_layer23_eiffel.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")

    # --- Compute subject vs template attention scores per head per layer ---
    # For each head: what fraction of attention goes to subject tokens vs template tokens?
    n_globals = len(GLOBAL_LAYERS)

    # Aggregate across prompts: [n_globals, n_heads, 3] for (subject, template, other)
    subject_scores = np.zeros((n_globals, n_heads))
    template_scores = np.zeros((n_globals, n_heads))

    for d in all_data:
        for gi, layer_idx in enumerate(GLOBAL_LAYERS):
            w = d["attn"][layer_idx]  # [n_heads, seq_len]
            for h in range(n_heads):
                subj_attn = sum(w[h, p] for p in d["subject_pos"])
                tmpl_attn = sum(w[h, p] for p in d["template_pos"])
                subject_scores[gi, h] += subj_attn
                template_scores[gi, h] += tmpl_attn

    subject_scores /= len(all_data)
    template_scores /= len(all_data)

    # --- Print the subject-attention leaderboard ---
    print(f"\n{'=' * 60}")
    print("Subject-entity attention leaderboard (averaged over 6 prompts)")
    print(f"{'=' * 60}")
    print(f"\n{'layer':>5}  {'head':>4}  {'subject_attn':>13}  {'template_attn':>14}  {'ratio':>7}")
    print("-" * 50)

    # Flatten and sort by subject attention
    entries = []
    for gi, layer_idx in enumerate(GLOBAL_LAYERS):
        for h in range(n_heads):
            entries.append((layer_idx, h, subject_scores[gi, h], template_scores[gi, h]))

    entries.sort(key=lambda x: -x[2])  # sort by subject attention descending

    for layer_idx, h, subj, tmpl in entries[:15]:
        ratio = subj / tmpl if tmpl > 0 else float("inf")
        print(f"  L{layer_idx:>2}     H{h}     {subj:>10.4f}      {tmpl:>11.4f}    {ratio:>6.2f}")

    print(f"\n... (showing top 15 of {len(entries)} head×layer combinations)")

    # --- Plot 2: Heatmap of subject attention by layer × head ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    ax = axes[0]
    im = ax.imshow(subject_scores, aspect="auto", cmap="Reds")
    ax.set_yticks(range(n_globals))
    ax.set_yticklabels([f"L{l}" for l in GLOBAL_LAYERS])
    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)])
    ax.set_title("Subject-entity attention\n(mean over 6 prompts)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1]
    im = ax.imshow(template_scores, aspect="auto", cmap="Greys")
    ax.set_yticks(range(n_globals))
    ax.set_yticklabels([f"L{l}" for l in GLOBAL_LAYERS])
    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)])
    ax.set_title("Template-token attention\n(mean over 6 prompts)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    out_path = OUT_DIR / "head_specialization_heatmap.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Wrote {out_path}")

    # --- Plot 3: Layer 23 all 8 heads across all 6 prompts (compact grid) ---
    fig, axes = plt.subplots(len(all_data), n_heads, figsize=(n_heads * 2.5, len(all_data) * 1.8))

    for row, d in enumerate(all_data):
        w = d["attn"][23]
        seq_len = len(d["token_labels"])
        for h in range(n_heads):
            ax = axes[row, h]
            colors = []
            for pos in range(seq_len):
                if pos in d["subject_pos"]:
                    colors.append("#e74c3c")
                elif pos in d["template_pos"]:
                    colors.append("#999999")
                else:
                    colors.append("#3498db")
            ax.bar(range(seq_len), w[h], color=colors, alpha=0.85)
            ax.set_ylim(0, min(1.0, np.max(w[h]) * 1.3 + 0.02))
            ax.set_xlim(-0.5, seq_len - 0.5)
            ax.set_xticks([])
            ax.tick_params(axis="y", labelsize=5)

            if row == 0:
                ax.set_title(f"H{h}", fontsize=9)
            if h == 0:
                ax.set_ylabel(f"→{d['prediction']!r}", fontsize=7, rotation=0, labelpad=40)

    fig.suptitle("Layer 23 — all heads × all prompts\n"
                 "(red = subject, gray = template, blue = other)", fontsize=11)
    plt.tight_layout()
    out_path = OUT_DIR / "layer23_heads_grid.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
