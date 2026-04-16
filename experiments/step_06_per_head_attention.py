"""Per-head attention analysis at global layers in Gemma 4 E4B.

Breaks open the 8 individual attention heads at each of the 7 global
layers across 6 factual-recall prompts. For each (layer, head), computes
the fraction of attention from the final position landing on subject-
entity tokens vs chat-template tokens.

Key finding (per docs/findings/step_06_per_head_attention.md): L29 H7 has
the highest subject-entity attention of any head at any global layer
(subject/template ratio 0.86). M07 then showed this head is causally
expendable — high attention does NOT imply high importance.

Run from project root:
    python experiments/step_06_per_head_attention.py
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
    Capture, GLOBAL_LAYERS, Model, Prompt, PromptSet,
)

OUT_DIR = ROOT / "caches"
N_HEADS = 8
TEMPLATE_SUBSTRINGS = ("<bos>", "<|turn>", "user", "<turn|>", "model")

# Per-prompt subject substrings stored in metadata. The framework's
# Prompt.subject is a single string; multi-token subjects use metadata.
PROMPTS_WITH_SUBJECTS = PromptSet(name="PER_HEAD_DEMO", prompts=(
    Prompt(text="Complete this sentence with one word: The Eiffel Tower is in",
           target="Paris", metadata={"subjects": ("Eiffel", "Tower")}),
    Prompt(text="Complete this sentence with one word: The capital of Japan is",
           target="Tokyo", metadata={"subjects": ("capital", "Japan")}),
    Prompt(text="Complete this sentence with one word: Romeo and Juliet was written by",
           target="Shakespeare", metadata={"subjects": ("Romeo", "Juliet", "written")}),
    Prompt(text="Complete this sentence with one word: The chemical symbol for gold is",
           target="Au", metadata={"subjects": ("chemical", "symbol", "gold")}),
    Prompt(text="Complete this sentence with one word: The opposite of hot is",
           target="cold", metadata={"subjects": ("opposite", "hot")}),
    Prompt(text="Complete this sentence with one word: Monday, Tuesday,",
           target="Wednesday", metadata={"subjects": ("Monday", "Tuesday")}),
))


def _decoded_token_labels(model, ids: mx.array) -> list[str]:
    return [model.tokenizer.decode([int(t)]) for t in ids[0]]


def _find_positions(token_labels: list[str], substrings) -> list[int]:
    """Return positions whose decoded label contains any of `substrings` (case-insensitive)."""
    return [
        i for i, label in enumerate(token_labels)
        if any(s.lower() in label.lower() for s in substrings)
    ]


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading model...")
    model = Model.load()
    capture = Capture.attn_weights(layers=list(GLOBAL_LAYERS))

    all_data = []
    print()
    for prompt in PROMPTS_WITH_SUBJECTS:
        ids = model.tokenize(prompt.text)
        token_labels = _decoded_token_labels(model, ids)
        seq_len = ids.shape[1]
        result = model.run(ids, interventions=[capture])

        last = result.last_logits.astype(mx.float32)
        probs = mx.softmax(last)
        mx.eval(probs)
        top1_tok = model.tokenizer.decode([int(np.argmax(np.array(probs)))])

        subject_pos = _find_positions(token_labels, prompt.metadata["subjects"])
        template_pos = _find_positions(token_labels, TEMPLATE_SUBSTRINGS)

        attn = {
            layer_idx: np.array(
                result.cache[f"blocks.{layer_idx}.attn.weights"][0, :, -1, :].astype(mx.float32)
            )
            for layer_idx in GLOBAL_LAYERS
        }

        subj_labels = [token_labels[p] for p in subject_pos]
        print(f"  {prompt.text[:50]:50s} → {top1_tok!r:12s}  "
              f"subject pos: {subject_pos} ({subj_labels})")

        all_data.append({
            "prompt": prompt,
            "prediction": top1_tok,
            "token_labels": token_labels,
            "subject_pos": subject_pos,
            "template_pos": template_pos,
            "attn": attn,
        })

    # ---- Plot 1: all 8 heads at L23 for the Eiffel Tower prompt ----
    d = all_data[0]
    w23 = d["attn"][23]
    seq_len = len(d["token_labels"])

    fig, axes = plt.subplots(N_HEADS, 1, figsize=(max(10, seq_len * 0.5), N_HEADS * 1.5))
    for h in range(N_HEADS):
        ax = axes[h]
        colors = []
        for pos in range(seq_len):
            if pos in d["subject_pos"]:
                colors.append("#e74c3c")
            elif pos in d["template_pos"]:
                colors.append("#999999")
            else:
                colors.append("#3498db")
        ax.bar(range(seq_len), w23[h], color=colors, alpha=0.85)
        ax.set_ylabel(f"H{h}", fontsize=9, rotation=0, labelpad=20)
        ax.set_ylim(0, min(1.0, np.max(w23[h]) * 1.3 + 0.01))
        ax.set_xlim(-0.5, seq_len - 0.5)
        if h == 0:
            ax.set_title(f"Layer 23 per-head attention — {d['prompt'].text[:55]}.. → {d['prediction']!r}\n"
                         f"(red = subject entity, gray = template, blue = other)",
                         fontsize=10)
        if h == N_HEADS - 1:
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

    # ---- Subject vs template attention scores per (layer, head) ----
    n_globals = len(GLOBAL_LAYERS)
    subject_scores = np.zeros((n_globals, N_HEADS))
    template_scores = np.zeros((n_globals, N_HEADS))
    for d in all_data:
        for gi, layer_idx in enumerate(GLOBAL_LAYERS):
            w = d["attn"][layer_idx]
            for h in range(N_HEADS):
                subject_scores[gi, h] += sum(w[h, p] for p in d["subject_pos"])
                template_scores[gi, h] += sum(w[h, p] for p in d["template_pos"])
    subject_scores /= len(all_data)
    template_scores /= len(all_data)

    # Leaderboard
    print(f"\n{'=' * 60}")
    print("Subject-entity attention leaderboard (averaged over 6 prompts)")
    print(f"{'=' * 60}")
    print(f"\n{'layer':>5}  {'head':>4}  {'subject_attn':>13}  {'template_attn':>14}  {'ratio':>7}")
    print("-" * 50)
    entries = [
        (layer_idx, h, float(subject_scores[gi, h]), float(template_scores[gi, h]))
        for gi, layer_idx in enumerate(GLOBAL_LAYERS)
        for h in range(N_HEADS)
    ]
    entries.sort(key=lambda x: -x[2])
    for layer_idx, h, subj, tmpl in entries[:15]:
        ratio = subj / tmpl if tmpl > 0 else float("inf")
        print(f"  L{layer_idx:>2}     H{h}     {subj:>10.4f}      {tmpl:>11.4f}    {ratio:>6.2f}")
    print(f"\n... (showing top 15 of {len(entries)} head x layer combinations)")

    # ---- Plot 2: heatmaps subject vs template ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    ax = axes[0]
    im = ax.imshow(subject_scores, aspect="auto", cmap="Reds")
    ax.set_yticks(range(n_globals))
    ax.set_yticklabels([f"L{l}" for l in GLOBAL_LAYERS])
    ax.set_xticks(range(N_HEADS))
    ax.set_xticklabels([f"H{h}" for h in range(N_HEADS)])
    ax.set_title("Subject-entity attention\n(mean over 6 prompts)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1]
    im = ax.imshow(template_scores, aspect="auto", cmap="Greys")
    ax.set_yticks(range(n_globals))
    ax.set_yticklabels([f"L{l}" for l in GLOBAL_LAYERS])
    ax.set_xticks(range(N_HEADS))
    ax.set_xticklabels([f"H{h}" for h in range(N_HEADS)])
    ax.set_title("Template-token attention\n(mean over 6 prompts)")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    out_path = OUT_DIR / "head_specialization_heatmap.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Wrote {out_path}")

    # ---- Plot 3: 6x8 grid of L23 across prompts/heads ----
    fig, axes = plt.subplots(len(all_data), N_HEADS,
                              figsize=(N_HEADS * 2.5, len(all_data) * 1.8))
    for row, d in enumerate(all_data):
        w = d["attn"][23]
        seq_len = len(d["token_labels"])
        for h in range(N_HEADS):
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
                ax.set_ylabel(f"→{d['prediction']!r}", fontsize=7,
                              rotation=0, labelpad=40)

    fig.suptitle("Layer 23 — all heads x all prompts\n"
                 "(red = subject, gray = template, blue = other)", fontsize=11)
    plt.tight_layout()
    out_path = OUT_DIR / "layer23_heads_grid.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
