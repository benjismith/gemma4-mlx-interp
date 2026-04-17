"""Concept generalization test: score implicit-emotion scenarios.

The self-consistency test in step_21 showed each emotion's training
passages score highest on their own probe — but that is the trivial
thing probes should do. The real test is whether the probes have
learned the EMOTION CONCEPT or merely the TRAINING CORPUS TEMPLATE.

Mirror of Anthropic's Table 2 (Sofroniew et al., Emotion Concepts, 2026):
12 user-turn scenarios, 2 per emotion, that each evoke one of our six
emotions without naming it. If the probes score these scenarios along
the right diagonal, they generalize across surface forms and the
concept claim survives.

Run from project root:
    python experiments/step_23_emotion_probe_implicit.py
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


# Hand-curated implicit-emotion scenarios, 2 per emotion.
# Each evokes the target emotion via SITUATION, avoiding the emotion's
# English name and its most obvious close synonyms. The diagonal test is
# whether the probe for the target emotion scores each scenario highest
# relative to the other five probes.
SCENARIOS = (
    Prompt(
        text="I just got the acceptance letter from my first-choice university with a full scholarship. My family is coming over this weekend to share the news. What's a nice thing to plan for when they arrive?",
        category="emotion_happy",
    ),
    Prompt(
        text="My partner and I bought our first house yesterday after saving for eight years. The keys are in my hand right now. What's a small thing we should do today to mark the moment?",
        category="emotion_happy",
    ),
    Prompt(
        text="My mother passed away last week and I still haven't been able to open her mail. What do people usually do with a lifetime of someone's paperwork?",
        category="emotion_sad",
    ),
    Prompt(
        text="I just got back from the vet. Fourteen years with him. The house feels impossible tonight. What do people usually do on that first evening?",
        category="emotion_sad",
    ),
    Prompt(
        text="My contractor took sixty thousand dollars and disappeared after demolishing half my kitchen. He won't answer calls or emails. What are my legal options for getting the money back?",
        category="emotion_angry",
    ),
    Prompt(
        text="A driver ran the red light and totaled my car with my kids in the back seat. They're okay, but the driver denied it at the scene and drove off. How do I make sure he faces consequences?",
        category="emotion_angry",
    ),
    Prompt(
        text="Someone has been following me home from the subway three nights in a row. My building doesn't have a doorman and my apartment is on the ground floor. What should I do tonight?",
        category="emotion_afraid",
    ),
    Prompt(
        text="My cardiologist just called to say my stress test came back abnormal and I need to come in tomorrow morning. I can't sleep. What questions should I be asking her?",
        category="emotion_afraid",
    ),
    Prompt(
        text="It's been raining all afternoon, I'm in a warm kitchen with tea, and I've got nothing on my calendar until Monday. What's a good book for a weekend like this?",
        category="emotion_calm",
    ),
    Prompt(
        text="I just got home from a silent meditation retreat. I want to extend the feeling for another day before I turn my phone back on. Any suggestions for how to spend tomorrow?",
        category="emotion_calm",
    ),
    Prompt(
        text="My daughter, who has a severe learning disability, just graduated valedictorian of her class. How can we honor what she's accomplished this weekend without making it feel like too much attention?",
        category="emotion_proud",
    ),
    Prompt(
        text="I finally shipped the product I've been building in my garage for four years. The first hundred units went out yesterday. How do I tell my parents about it without sounding like I'm bragging?",
        category="emotion_proud",
    ),
)

SCENARIO_SET = PromptSet(name="EMOTION_SCENARIOS_TINY", prompts=SCENARIOS)


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


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading model...")
    model = Model.load()

    print("\nRebuilding emotion probes from training corpus...")
    probes, emotion_order = _build_probes(model)
    print(f"  built {len(probes)} probes at L{LAYER}")

    # ---- Extract pooled residuals for the scenarios ----
    print(f"\nValidating {len(SCENARIO_SET)} scenarios...")
    valid = SCENARIO_SET.validate(
        model, verbose=False, min_confidence=0.0, require_target_match=False,
    )
    n = len(valid)
    print(f"  {n} / {len(SCENARIO_SET)} validated.")

    print(f"Extracting pooled residuals at layer {LAYER} "
          f"(positions >= {POOL_START})...")
    scenario_vecs = fact_vectors_pooled(
        model, valid, layers=[LAYER], start=POOL_START,
    )[LAYER]
    print(f"  scenario_vecs: {scenario_vecs.shape}")

    # ---- Score each scenario against each probe ----
    probe_names = list(probes.keys())
    scores = np.stack(
        [probes[p].score(scenario_vecs) for p in probe_names],
        axis=1,
    )  # [n_scenarios, n_probes]
    labels = valid.labels

    # Per-scenario printout
    print(f"\n{'=' * 90}")
    print(f"Per-scenario probe scores (best probe marked *)")
    print(f"{'=' * 90}")
    print(f"{'true':>10s}  " + "  ".join(
        f"{p.replace('emotion_', ''):>8s}" for p in probe_names
    ) + "   scenario")
    print("-" * 90)

    correct = 0
    for j in range(n):
        row = scores[j]
        arg = int(np.argmax(row))
        hit = probe_names[arg] == labels[j]
        if hit:
            correct += 1
        true_short = labels[j].replace("emotion_", "")
        score_parts = []
        for pj in range(len(probe_names)):
            marker = "*" if pj == arg else " "
            score_parts.append(f"{row[pj]:+7.2f}{marker}")
        short = valid[j].prompt.text[:52] + ("..." if len(valid[j].prompt.text) > 52 else "")
        print(f"  {true_short:>8s}  " + "  ".join(score_parts) + f"   {short}")

    print()
    print(f"Per-scenario top-1 accuracy: {correct} / {n} = {100 * correct / n:.1f}%")
    print(f"(chance = 1 / {len(probe_names)} = {100/len(probe_names):.1f}%)")

    # Aggregate by true emotion
    print(f"\n{'=' * 90}")
    print(f"Aggregated (mean over each emotion's 2 scenarios)")
    print(f"{'=' * 90}")
    agg = np.zeros((len(emotion_order), len(probe_names)), dtype=np.float32)
    for i, e in enumerate(emotion_order):
        mask = labels == e
        agg[i] = scores[mask].mean(axis=0) if mask.any() else 0.0
    header = "true / probe"
    print(f"  {header:>15s}  " + "  ".join(
        f"{p.replace('emotion_', ''):>8s}" for p in probe_names
    ))
    print("-" * (17 + 10 * len(probe_names)))
    diag_hits = 0
    for i, e in enumerate(emotion_order):
        row = agg[i]
        arg = int(np.argmax(row))
        if probe_names[arg] == e:
            diag_hits += 1
        print(f"  {e:>15s}  " + "  ".join(
            f"{row[j]:+7.3f}" + ("*" if j == arg else " ")
            for j in range(len(probe_names))
        ))
    print()
    print(f"Aggregated diagonal hits: {diag_hits} / {len(emotion_order)}")

    # ---- Visualization ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: per-scenario heatmap (rows grouped by true emotion)
    ax = axes[0]
    order = np.argsort([emotion_order.index(l) for l in labels])
    scores_sorted = scores[order]
    labels_sorted = labels[order]
    im = ax.imshow(
        scores_sorted, aspect="auto", cmap="RdBu_r",
        vmin=-np.abs(scores).max(), vmax=np.abs(scores).max(),
    )
    ax.set_xticks(range(len(probe_names)))
    ax.set_xticklabels([p.replace("emotion_", "") for p in probe_names],
                       rotation=30, ha="right")
    ax.set_yticks(range(len(labels_sorted)))
    ax.set_yticklabels(
        [f"{labels_sorted[k].replace('emotion_', ''):>6s}  #{k+1}"
         for k in range(len(labels_sorted))],
        fontsize=9,
    )
    # Boundary lines between emotion groups
    cur = labels_sorted[0]
    for y in range(len(labels_sorted)):
        if labels_sorted[y] != cur:
            ax.axhline(y - 0.5, color="black", linewidth=0.8)
            cur = labels_sorted[y]
    ax.set_xlabel("probe")
    ax.set_ylabel("scenario (grouped by true emotion)")
    ax.set_title(f"Per-scenario probe scores at L{LAYER}\n"
                 f"{correct}/{n} correct = {100 * correct / n:.0f}%")
    # Text annotations
    for i in range(scores_sorted.shape[0]):
        for j in range(scores_sorted.shape[1]):
            v = scores_sorted[i, j]
            color = "white" if abs(v) > np.abs(scores).max() * 0.5 else "black"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    color=color, fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Panel 2: aggregated heatmap
    ax = axes[1]
    im = ax.imshow(
        agg, aspect="auto", cmap="RdBu_r",
        vmin=-np.abs(agg).max(), vmax=np.abs(agg).max(),
    )
    ax.set_xticks(range(len(probe_names)))
    ax.set_xticklabels([p.replace("emotion_", "") for p in probe_names],
                       rotation=30, ha="right")
    ax.set_yticks(range(len(emotion_order)))
    ax.set_yticklabels([e.replace("emotion_", "") for e in emotion_order])
    ax.set_xlabel("probe")
    ax.set_ylabel("true emotion")
    ax.set_title(f"Aggregated scenario scores at L{LAYER}\n"
                 f"diag {diag_hits}/{len(emotion_order)}")
    for i in range(agg.shape[0]):
        for j in range(agg.shape[1]):
            v = agg[i, j]
            color = "white" if abs(v) > np.abs(agg).max() * 0.5 else "black"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    color=color, fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(
        "Implicit-emotion scenario validation: do probes discriminate "
        "held-out scenarios without training on their vocabulary?",
        fontsize=12,
    )
    plt.tight_layout()
    out_path = OUT_DIR / "emotion_probes_implicit.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
