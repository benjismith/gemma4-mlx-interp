"""Scalar intensity modulation test for the emotion probes.

Template prompts with one numerical quantity that modulates the expected
emotional intensity WITHOUT changing most of the surface structure.
Score each level against all 6 probes at L28. A semantic probe should
respond monotonically to the scalar along the expected direction; the
antipode probe (opposite valence/arousal) should respond in the
opposite direction; unrelated probes should stay roughly flat.

Four axes:
  1. Tylenol dose (mg) -> afraid up, calm down
     'I just took {X} mg of Tylenol for my back pain. Should I be concerned?'
  2. Lottery winnings ($) -> happy up, proud up
     'I just won ${X} in the lottery. What should I do first?'
  3. Amount stolen ($) -> angry up, calm down
     'My contractor disappeared after taking ${X} from me. What are my legal options?'
  4. Silent retreat length (days) -> calm up
     'I just finished a {X}-day silent meditation retreat. What should I do with the feeling?'

This is the Gemma 4 mini-scale version of Figure 3 from Sofroniew et al.,
Emotion Concepts, 2026, which varies Tylenol dosage and hours-since-eating
and watches Sonnet 4.5's emotion probes respond accordingly.

Run from project root:
    python experiments/step_24_emotion_probe_intensity.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gemma4_mlx_interp import Model, Probe, Prompt, PromptSet  # noqa: E402
from gemma4_mlx_interp import fact_vectors_pooled  # noqa: E402
from experiments.prompts import (  # noqa: E402
    EMOTION_NEUTRAL_BASELINE, EMOTION_STORIES_TINY,
)

OUT_DIR = ROOT / "caches"
LAYER = 28
POOL_START = 20

AXES = [
    {
        "key": "tylenol",
        "description": "Tylenol dose (mg)",
        "expected_up": "emotion_afraid",
        "expected_down": "emotion_calm",
        "template": "I just took {X} mg of Tylenol for my back pain. Should I be concerned?",
        "levels": [500, 1500, 5000, 10000, 20000],
        "log_x": True,
    },
    {
        "key": "lottery",
        "description": "Lottery winnings ($)",
        "expected_up": "emotion_happy",
        "expected_down": None,
        "template": "I just won ${X} in the lottery. What should I do first?",
        "levels": [50, 500, 5000, 50000, 500000],
        "log_x": True,
    },
    {
        "key": "theft",
        "description": "Amount stolen by contractor ($)",
        "expected_up": "emotion_angry",
        "expected_down": "emotion_calm",
        "template": "My contractor disappeared after taking ${X} from me. What are my legal options?",
        "levels": [500, 5000, 50000, 500000],
        "log_x": True,
    },
    {
        "key": "retreat",
        "description": "Silent meditation retreat length (days)",
        "expected_up": "emotion_calm",
        "expected_down": None,
        "template": "I just finished a {X}-day silent meditation retreat. What should I do with the feeling?",
        "levels": [1, 3, 7, 14, 30],
        "log_x": True,
    },
]


def _build_probes(model) -> tuple[dict[str, Probe], list[str]]:
    """Rebuild the 6 emotion probes from the training corpus."""
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
    probes = Probe.from_labeled_corpus(
        labeled, neutral_vecs, layer=LAYER, explain=0.5,
    )
    return probes, emotion_valid.categories


def _format_x(v: int) -> str:
    """Pretty-print the scalar for use in templates."""
    if v >= 1000:
        return f"{v:,}"
    return str(v)


def _score_axis(
    model, axis: dict, probes: dict[str, Probe], probe_order: list[str],
) -> tuple[np.ndarray, list[str]]:
    """For one axis: build prompts, extract pooled residuals, score against
    each probe. Returns [n_levels, n_probes] of scores and the rendered
    prompt texts.
    """
    texts = [axis["template"].format(X=_format_x(v)) for v in axis["levels"]]
    prompts = PromptSet(
        name=f"INTENSITY_{axis['key']}",
        prompts=tuple(Prompt(text=t, category=axis["key"]) for t in texts),
    )
    valid = prompts.validate(
        model, verbose=False, min_confidence=0.0, require_target_match=False,
    )
    vecs = fact_vectors_pooled(
        model, valid, layers=[LAYER], start=POOL_START,
    )[LAYER]
    scores = np.stack(
        [probes[p].score(vecs) for p in probe_order],
        axis=1,
    )
    return scores, texts


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading model...")
    model = Model.load()

    print("\nBuilding emotion probes...")
    probes, emotion_order = _build_probes(model)
    probe_order = list(probes.keys())
    print(f"  built {len(probes)} probes at L{LAYER}")

    # Colors for each probe (consistent with step_21)
    probe_colors = {
        "emotion_happy":   "#f0a500",
        "emotion_sad":     "#2d81b6",
        "emotion_angry":   "#d62728",
        "emotion_afraid":  "#6a3d9a",
        "emotion_calm":    "#2ca02c",
        "emotion_proud":   "#ff7f0e",
    }

    all_results = {}
    print(f"\n{'=' * 80}")
    print(f"Intensity sweeps")
    print(f"{'=' * 80}")

    for axis in AXES:
        print(f"\n--- {axis['description']} ---")
        scores, texts = _score_axis(model, axis, probes, probe_order)
        all_results[axis["key"]] = {
            "scores": scores, "levels": axis["levels"], "texts": texts,
        }

        print(f"  {'level':>12s}  " + "  ".join(
            f"{p.replace('emotion_', ''):>8s}" for p in probe_order
        ))
        for i, lv in enumerate(axis["levels"]):
            print(f"  {_format_x(lv):>12s}  " + "  ".join(
                f"{scores[i, j]:+7.3f}"
                for j in range(len(probe_order))
            ))
        # Monotonicity check
        if axis["expected_up"]:
            up_idx = probe_order.index(axis["expected_up"])
            up_series = scores[:, up_idx]
            monotonic_up = all(
                up_series[i + 1] >= up_series[i]
                for i in range(len(up_series) - 1)
            )
            direction = up_series[-1] - up_series[0]
            print(f"  {axis['expected_up'].replace('emotion_', '')} probe: "
                  f"delta = {direction:+.3f}  "
                  f"{'(strictly monotonic UP)' if monotonic_up else '(non-monotonic)'}")
        if axis["expected_down"]:
            dn_idx = probe_order.index(axis["expected_down"])
            dn_series = scores[:, dn_idx]
            monotonic_dn = all(
                dn_series[i + 1] <= dn_series[i]
                for i in range(len(dn_series) - 1)
            )
            direction = dn_series[-1] - dn_series[0]
            print(f"  {axis['expected_down'].replace('emotion_', '')} probe: "
                  f"delta = {direction:+.3f}  "
                  f"{'(strictly monotonic DOWN)' if monotonic_dn else '(non-monotonic)'}")

    # ---- Plot: one panel per axis, six lines per panel ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, axis in zip(axes.flatten(), AXES):
        r = all_results[axis["key"]]
        levels = r["levels"]
        scores = r["scores"]
        for j, pname in enumerate(probe_order):
            is_target = pname == axis["expected_up"]
            is_antipode = pname == axis["expected_down"]
            lw = 3 if is_target or is_antipode else 1.3
            alpha = 1.0 if is_target or is_antipode else 0.55
            label = pname.replace("emotion_", "")
            if is_target:
                label += " (target ↑)"
            elif is_antipode:
                label += " (antipode ↓)"
            ax.plot(
                levels, scores[:, j],
                marker="o", linewidth=lw, alpha=alpha,
                color=probe_colors[pname], label=label,
            )
        if axis["log_x"]:
            ax.set_xscale("log")
        ax.set_xlabel(axis["description"])
        ax.set_ylabel("probe score")
        ax.set_title(axis["description"])
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8, ncol=2)

    fig.suptitle(
        "Intensity modulation: do probes respond monotonically to scalar axes?\n"
        f"(layer {LAYER}, pool_start = {POOL_START}, bold lines = expected "
        f"target / antipode)",
        fontsize=12,
    )
    plt.tight_layout()
    out_path = OUT_DIR / "emotion_probes_intensity.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
