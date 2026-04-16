"""Position-wise logit lens: where in the residual stream does the answer
appear at every token position, at every layer?

For each prompt, projects every layer's resid_post through the unembed at
EVERY position (not just the final one). Produces a [layers x positions]
heatmap of the answer token's log-probability and rank.

Per finding 08: the answer is never decodable at the subject position. It
crystallizes at the final position around layers 29-30 and gets stronger
through layer 41. The MLPs at subject positions write features that
later layers compose into the answer — they don't write the answer token
itself.

Run from project root:
    python experiments/step_08_position_logit_lens.py
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
    Capture, GLOBAL_LAYERS, Model, N_LAYERS, Prompt, PromptSet,
    logit_lens_per_position, position_heatmap,
)

OUT_DIR = ROOT / "caches"

# A subset of FACTUAL_15 plus the BIG_SWEEP_96 conventions for subject
# substrings. Each prompt has a single-substring subject for position lookup.
PROMPTS_WITH_SUBJECTS = PromptSet(name="POS_LENS_DEMO", prompts=(
    Prompt(text="Complete this sentence with one word: The Eiffel Tower is in",
           target="Paris", subject="Tower"),
    Prompt(text="Complete this sentence with one word: The capital of Japan is",
           target="Tokyo", subject="Japan"),
    Prompt(text="Complete this sentence with one word: Romeo and Juliet was written by",
           target="Shakespeare", subject="Juliet"),
    Prompt(text="Complete this sentence with one word: The chemical symbol for gold is",
           target="Au", subject="gold"),
    Prompt(text="Complete this sentence with one word: The opposite of hot is",
           target="cold", subject="hot"),
    Prompt(text="Complete this sentence with one word: Monday, Tuesday,",
           target="Wednesday", subject="Tuesday"),
))


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading model...")
    model = Model.load()

    capture = Capture.residual(layers=range(N_LAYERS), point="post")

    for prompt_idx, prompt in enumerate(PROMPTS_WITH_SUBJECTS):
        print(f"\n{'=' * 60}")
        print(f"Prompt: {prompt.text!r}")

        ids = model.tokenize(prompt.text)
        token_labels = [model.tokenizer.decode([int(t)]) for t in ids[0]]
        seq_len = ids.shape[1]

        # Find subject positions (substring may match multiple tokens)
        subject_pos = [
            i for i, t in enumerate(token_labels)
            if prompt.subject.lower() in t.lower()
        ]
        print(f"Tokens ({seq_len}): {token_labels}")
        print(f"Subject positions: {subject_pos} ({[token_labels[p] for p in subject_pos]})")

        # Run with full residual capture and detect the model's actual top-1.
        result = model.run(ids, interventions=[capture])
        last = result.last_logits.astype(mx.float32)
        probs = mx.softmax(last)
        mx.eval(probs)
        target_id = int(np.argmax(np.array(probs)))
        target_tok = model.tokenizer.decode([target_id])
        target_prob = float(np.array(probs)[target_id])
        print(f"Target: {target_tok!r} (id={target_id}, p={target_prob:.3f})")

        # Position-wise lens: matrices [N_LAYERS, seq_len].
        ranks, logprobs = logit_lens_per_position(model, result.cache, target_id)

        # Print per-layer ranks at subject positions vs final
        print(f"\n  Target token rank at subject positions across layers:")
        print(f"  {'layer':>5}  ", end="")
        for p in subject_pos:
            print(f"  {token_labels[p]:>12s}", end="")
        print(f"  {'[final pos]':>12s}")
        print(f"  {'-' * (7 + 14 * (len(subject_pos) + 1))}")
        for i in list(range(0, N_LAYERS, 3)) + [N_LAYERS - 1]:
            print(f"  {i:>5}  ", end="")
            for p in subject_pos:
                print(f"  {int(ranks[i, p]):>12d}", end="")
            print(f"  {int(ranks[i, -1]):>12d}")

        # Layer where target first enters top-10 at each subject + final pos
        def first_top10(pos: int) -> str:
            for i in range(N_LAYERS):
                if ranks[i, pos] < 10:
                    return f"layer {i}"
            return "never"

        print(f"\n  Layer where target first enters top-10:")
        for p in subject_pos:
            print(f"    pos {p:>2} ({token_labels[p]:>12s}): {first_top10(p)}")
        print(f"    final position:          {first_top10(seq_len - 1)}")

        # Heatmap: 2-panel (logprob + log10 rank)
        fig, axes = plt.subplots(1, 2, figsize=(max(12, seq_len * 0.6), 8))

        position_heatmap(
            logprobs, token_labels, ax=axes[0],
            cmap="RdYlGn", vmin=-30, vmax=0,
            mark_positions=subject_pos, mark_layers=GLOBAL_LAYERS,
            colorbar_label="log p",
            title=f"log p({target_tok!r}) at each position x layer",
        )
        position_heatmap(
            ranks, token_labels, ax=axes[1],
            cmap="RdYlGn_r", vmin=0, vmax=5.5,
            mark_positions=subject_pos, mark_layers=GLOBAL_LAYERS,
            colorbar_label="log10(rank + 1)",
            log_scale=True,
            title=f"rank of {target_tok!r} (log10 scale)",
        )

        fig.suptitle(
            f"{prompt.text}\n→ {target_tok!r}  (red dashes = subject positions)",
            fontsize=10,
        )
        plt.tight_layout()
        out_path = OUT_DIR / f"position_logit_lens_{prompt_idx}.png"
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Wrote {out_path}")


if __name__ == "__main__":
    main()
