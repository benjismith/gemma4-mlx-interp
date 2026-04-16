# Single-Head Ablation: L29 H7 Is Not the Factual Recall Bottleneck

**Date:** 2026-04-15
**Script:** `experiments/step_07_single_head_ablation.py`
**Plot:** `caches/single_head_ablation.png`

## Hypothesis

L29 H7 has the highest subject-entity attention of any head at any global layer (subject/template ratio 0.86, finding 06). We tested whether ablating this single head would destroy the model's factual recall ability — whether one head out of 336 is the mechanism that retrieves factual answers.

## Results

### L29 H7 has essentially zero causal impact

| Metric | Value |
|--------|------:|
| Mean Δ log p | -0.010 |
| Rank among L29 heads | 3rd/8 |
| Prompts where target remains top-1 | 14/15 |

The one prompt that changes (Amazon River → "Brazil" instead of "South") is arguably an improvement. Removing the head that most attends to subject entities doesn't damage factual recall at all.

### No single head is a bottleneck

The largest single-head ablation effect across all 16 heads tested:

| Layer | Head | Mean Δ log p | Role (from finding 06) |
|------:|-----:|-------------:|-----------------------|
| L23 | H0 | -0.20 | Template-dominant |
| L29 | H3 | -0.11 | Mixed |
| L23 | H2 | -0.06 | General-purpose |
| L23 | H3 | -0.06 | Template-dominant |
| L23 | H6 | -0.04 | Late-sequence |

The most important head (L23 H0 at -0.20) is a template-attending head, not a content-attending head. And even -0.20 is tiny: the full L23 attention ablation from finding 04 cost -4.96, meaning the 8 heads collectively are ~25x more important than the single most important head. The heads function as a redundant ensemble.

### Where attention looks ≠ what it contributes

This is the central lesson. L29 H7's attention pattern clearly shows it attending to subject-entity tokens more than any other head. But removing it doesn't matter, because:

1. **The residual stream already contains the answer.** By layer 29, the MLPs in layers 10–23 have written factual knowledge into the residual stream at every position. The information L29 H7 reads from the subject tokens is already available to downstream layers through the residual stream itself — the attention-based copy is redundant.

2. **Other heads compensate.** Even if L29 H7 provides some marginal value, the other 7 heads at L29 (and the 8 heads at each of 6 other global layers) provide enough redundancy that no single head is a bottleneck.

3. **Attention patterns reflect what information is *available*, not what is *needed*.** The head attends to subject tokens because those positions contain rich representations (thanks to the MLPs). But "this position has useful information" doesn't mean "this head is the only way to access it."

## Implications for mechanistic interpretability

This result pushes back against the "crisp circuit" narrative that mechanistic interpretability often aspires to. The common template — find a specific head that attends to the right tokens, trace the information flow, declare you've found the circuit — doesn't work here. The mechanism for factual recall in Gemma 4 E4B is:

- **Knowledge storage**: Distributed across MLPs in layers 10–24, no single layer or MLP dominates (finding 04, where the most damaging single MLP ablation was layer 14 at -9.4, but 10+ MLPs each contributed meaningfully).
- **Knowledge routing**: Distributed across all attention heads collectively, no single head is a bottleneck (this finding).
- **Structural context**: Distributed across template-attending heads at multiple global layers (finding 05/06).

The model has built robustness through redundancy. This makes it harder to reverse-engineer but may be part of why it works so well — no single point of failure for factual recall.

## Revised functional map

The map from findings 01–06 stands, with one correction: we should NOT assign specific computational roles to individual heads. The division of labor operates at the level of *layer types* (MLP vs attention) and *layer regions* (foundation / engine room / readout), not at the level of individual heads.

| Component | Role | Granularity of importance |
|-----------|------|--------------------------|
| MLPs (layers 10–24) | Knowledge storage & retrieval | Per-layer (some layers critical, others dispensable) |
| Attention (layer 23) | Structural bridging | Collective (all heads together, no single bottleneck) |
| Attention (layer 29) | Content reading | Collective (heads attend to content but individually expendable) |
| Side-channel | Token identity grounding | Per-layer (concentrated at globals 11, 17, 23, 29) |
| Late layers (25–41) | Readout & calibration | Mostly dispensable individually |
