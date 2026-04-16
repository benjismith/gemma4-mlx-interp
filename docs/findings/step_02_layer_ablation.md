# Per-Layer Zero-Ablation on Gemma 4 E4B

**Date:** 2026-04-15
**Script:** `experiments/step_02_layer_ablation.py`
**Plot:** `caches/layer_ablation.png`

## Setup

For each of the 42 layers, we skipped the layer's contribution to the residual stream and measured how much the model's log-probability of its own top-1 answer dropped, averaged across 15 prompts. "Skipping" means the residual stream passes through unmodified: `resid_post = resid_pre`. For layers 0–23, which own KV cache entries that downstream layers may read from, we still ran attention to populate the cache — we just didn't add any of the layer's output (attention, MLP, per-layer gate) to the residual. For layers 24–41 (KV-shared, read-only), we skipped entirely.

This gives us a causal measure of each layer's importance: a large negative Δ log p means removing that layer destroys the model's confidence in its answer.

## Results

### Layer 0 dominates

Ablating layer 0 produces a mean Δ log p of -16.0, by far the largest. This isn't surprising: layer 0 transforms raw (scaled) token embeddings into the representation space the rest of the network expects. Without it, every subsequent layer receives input that's statistically unlike anything it was trained on. Layer 1 is a distant second at -2.5, and layers 2–4 are near zero.

### The "invisible middle" is actually the most important

The logit lens experiment found that layers 0–24 look inert through the unembed projection — the residual stream carries essentially no information about the final answer during those layers. Ablation tells the opposite story: layers 10–24 contain the *most damaging* layers to remove (14, 19, 23, 16, 13, 10, all with mean Δ log p below -2.5).

This is a clean complementary result. The logit lens measures what's visible through a specific probe (the tied unembed); ablation measures causal impact. The discrepancy means these middle layers are doing *preparation* — building representations that the later layers (27–36) then convert into the final answer. The work is real but invisible to the lens because the unembed isn't the right projection for intermediate representations. This is exactly the kind of phenomenon the "tuned lens" literature (Belrose et al., 2023) was designed to address: a learned per-layer affine probe recovers much more information from early/middle layers than the raw unembed does.

### Global vs. local: modest difference, not categorical

| Metric | Global (n=7) | Local (n=35) |
|--------|-------------:|-------------:|
| Mean Δ log p | -1.68 | -1.47 |
| Median Δ log p | -1.88 | -0.30 |

The medians tell the story better: global layers are slightly more consistently important (median -1.88 vs -0.30), but the distributions overlap heavily. Specific globals matter a lot (layer 23: -4.1), others barely at all (layer 5: -0.18, layer 35: +0.07). The same is true for locals — layer 14 (-6.3) is more important than any global, while many locals in the 25–41 range are near zero.

The global-attention layers are not a categorically special class in terms of causal importance. They're slightly more likely to matter, but the variation within each group swamps the between-group difference.

### Most damaging layers to ablate

| Rank | Layer | Type | Mean Δ log p |
|-----:|------:|------|-------------:|
| 1 | 0 | local | -15.96 |
| 2 | 14 | local | -6.33 |
| 3 | 19 | local | -4.08 |
| 4 | 23 | GLOBAL | -4.07 |
| 5 | 16 | local | -3.74 |

1 out of 5 is a global layer. Nothing about the every-6th-layer placement makes those layers systematically more valuable than their neighbors.

### Late layers are mostly dispensable

Layers 25–41 show small ablation impacts, with a few exceptions (layer 29 at -1.9, layer 41 at -1.1). The model's late layers are doing refinement — calibrating confidence, selecting surface forms — not heavy retrieval or reasoning. This is consistent with the logit lens finding that the answer is essentially locked in by layer 36.

## Synthesis with the logit lens

Combining both experiments produces a two-phase picture of how Gemma 4 E4B processes a factual-recall prompt:

1. **Layers 0–24: invisible but essential.** The residual stream is being shaped into a rich internal representation. The logit lens can't see it (projecting through the unembed returns noise), but removing any of these layers causes significant damage. The model is doing its hardest computational work here.

2. **Layers 25–41: visible but dispensable.** The answer appears in the logit lens around layer 27–30 and sharpens through layer 36. But ablation says most of these layers contribute little — they're reading out and polishing a representation that was already built in the first phase. The exception is the final global layer (41), which does meaningful work on surface-form selection and confidence calibration.

The hybrid global/local architecture doesn't create a clean separation between "global information" and "local processing" layers, at least not one that's visible in these two experiments. Whatever role the every-6th-layer cadence plays, it's subtler than "globals do the important stuff."

## Follow-up ideas

1. **Per-layer ablation conditioned on prompt type.** Our 15 prompts are all factual recall. Global layers might matter more for tasks that require longer-range dependencies (multi-hop reasoning, coreference). The current experiment can't distinguish.

2. **Ablation at sub-layer granularity.** Instead of skipping the entire layer, ablate just the attention branch or just the MLP branch. This would tell us whether the critical layers 10–24 are important for their attention (information routing) or their MLP (knowledge retrieval).

3. **Tuned lens.** Fit a per-layer affine probe and re-run the logit lens to see whether the "invisible middle" becomes visible with a better projection. If it does, the two experiments converge; if not, the middle layers are doing something genuinely orthogonal to the output space.
