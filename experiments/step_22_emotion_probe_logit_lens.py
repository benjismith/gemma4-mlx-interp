"""Logit-lens the emotion probes: which tokens does each probe upweight?

Project each probe.vec through Gemma 4 E4B's tied unembed and report the
top-K upweighted and top-K downweighted tokens. This is the probe-level
analogue of the logit-lens trajectory from step_01 / finding 11, applied
to the concept directions built by step_21.

Anthropic's Table 1 (Sofroniew et al., Emotion Concepts, 2026) shows the
Sonnet 4.5 emotion vectors decoding to human-interpretable up-lists:

    happy      -> excited, excitement, exciting, happ, celeb
    desperate  -> desperate, desper, urgent, bankrupt, urg
    sad        -> mour, grief, tears, lonely, crying

If the same pattern holds for our 6 Gemma 4 E4B probes, that's strong
evidence the probes captured the *concept*, not just the training
corpus's template. Cached probes are loaded from step_21's output.

Run from project root:
    python experiments/step_22_emotion_probe_logit_lens.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gemma4_mlx_interp import Model, Probe  # noqa: E402
from experiments.prompts import (  # noqa: E402
    EMOTION_NEUTRAL_BASELINE, EMOTION_STORIES_TINY,
)
from gemma4_mlx_interp import fact_vectors_pooled  # noqa: E402

OUT_DIR = ROOT / "caches"
LAYER = 28
POOL_START = 20
TOP_K = 8


def _build_probes(model) -> dict[str, Probe]:
    """Rebuild the 6 emotion probes from corpus, same as step_21."""
    emotion_valid = EMOTION_STORIES_TINY.validate(
        model, verbose=False, min_confidence=0.0, require_target_match=False,
    )
    neutral_valid = EMOTION_NEUTRAL_BASELINE.validate(
        model, verbose=False, min_confidence=0.0, require_target_match=False,
    )
    emotion_vecs = fact_vectors_pooled(
        model, emotion_valid, layers=[LAYER], start=POOL_START,
    )[LAYER]
    neutral_vecs = fact_vectors_pooled(
        model, neutral_valid, layers=[LAYER], start=POOL_START,
    )[LAYER]
    labeled = {
        e: emotion_vecs[emotion_valid.labels == e]
        for e in emotion_valid.categories
    }
    return Probe.from_labeled_corpus(
        labeled, neutral_vecs, layer=LAYER, explain=0.5,
    )


def _probe_logits(model, probe: Probe) -> np.ndarray:
    """Project probe.vec through the tied unembed. Returns float32 [vocab]."""
    v_mx = mx.array(probe.vec[None, None, :], dtype=mx.bfloat16)
    logits = model.project_to_logits(v_mx)
    last = logits[0, 0, :].astype(mx.float32)
    mx.eval(last)
    return np.array(last)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading model...")
    model = Model.load()

    print("\nRebuilding emotion probes from corpus (see step_21)...")
    probes = _build_probes(model)
    emotion_order = list(probes.keys())
    print(f"  built {len(probes)} probes at L{LAYER}")

    # ---- Per-probe top-K up/down tokens through the tied unembed ----
    print(f"\n{'=' * 70}")
    print(f"Top-{TOP_K} upweighted and downweighted tokens per probe")
    print(f"{'=' * 70}")

    report_rows = []
    for ename in emotion_order:
        probe = probes[ename]
        logits = _probe_logits(model, probe)
        top_up = np.argsort(-logits)[:TOP_K]
        top_dn = np.argsort(logits)[:TOP_K]
        up_tokens = [model.tokenizer.decode([int(i)]) for i in top_up]
        dn_tokens = [model.tokenizer.decode([int(i)]) for i in top_dn]
        up_vals = logits[top_up]
        dn_vals = logits[top_dn]

        short = ename.replace("emotion_", "")
        print(f"\n[{short}]")
        print(f"  ↑ UP   " + "  ".join(
            f"{t!r}({v:+.2f})" for t, v in zip(up_tokens, up_vals)
        ))
        print(f"  ↓ DOWN " + "  ".join(
            f"{t!r}({v:+.2f})" for t, v in zip(dn_tokens, dn_vals)
        ))
        report_rows.append({
            "emotion": short,
            "up_tokens": up_tokens,
            "up_vals": up_vals.tolist(),
            "dn_tokens": dn_tokens,
            "dn_vals": dn_vals.tolist(),
        })

    # ---- Visualization: bar panels per emotion ----
    n = len(emotion_order)
    fig, axes = plt.subplots(n, 2, figsize=(14, 2.0 * n), squeeze=False)
    for row_idx, ename in enumerate(emotion_order):
        r = report_rows[row_idx]
        short = ename.replace("emotion_", "")

        ax = axes[row_idx, 0]
        ax.barh(range(TOP_K), r["up_vals"], color="#2ca02c")
        ax.set_yticks(range(TOP_K))
        ax.set_yticklabels([repr(t) for t in r["up_tokens"]], fontsize=9)
        ax.invert_yaxis()
        ax.set_title(f"{short} ↑ upweighted", fontsize=10)
        ax.grid(True, alpha=0.3, axis="x")

        ax = axes[row_idx, 1]
        ax.barh(range(TOP_K), r["dn_vals"], color="#d62728")
        ax.set_yticks(range(TOP_K))
        ax.set_yticklabels([repr(t) for t in r["dn_tokens"]], fontsize=9)
        ax.invert_yaxis()
        ax.set_title(f"{short} ↓ downweighted", fontsize=10)
        ax.grid(True, alpha=0.3, axis="x")

    fig.suptitle(
        f"Emotion probes projected through the tied unembed (layer {LAYER})\n"
        f"Top {TOP_K} tokens per direction, per probe",
        fontsize=12,
    )
    plt.tight_layout()
    out_path = OUT_DIR / "emotion_probes_logit_lens.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")

    # ---- Markdown-ready summary table (for the findings doc) ----
    print(f"\n{'=' * 70}")
    print(f"Markdown summary")
    print(f"{'=' * 70}")
    print(f"\n| emotion | top-5 ↑ | top-5 ↓ |")
    print(f"|---------|---------|---------|")
    for r in report_rows:
        up = ", ".join(f"`{t!r}`" for t in r["up_tokens"][:5])
        dn = ", ".join(f"`{t!r}`" for t in r["dn_tokens"][:5])
        print(f"| **{r['emotion']}** | {up} | {dn} |")


if __name__ == "__main__":
    main()
