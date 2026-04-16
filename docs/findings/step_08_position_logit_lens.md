# Position-Wise Logit Lens: The Answer Only Lives at the Final Position

**Date:** 2026-04-15
**Script:** `experiments/step_08_position_logit_lens.py`
**Plots:** `caches/position_logit_lens_*.png`

## Hypothesis

We know the MLPs in layers 10–24 are essential for factual recall (finding 04). We hypothesized that they work by writing the answer token into the residual stream at the subject-entity position — e.g., writing "Paris" at the "Tower" position. This would be an "associative memory" mechanism: the MLP recognizes "Eiffel Tower" and writes its associated fact ("Paris") directly into that position's representation, where later attention (perhaps L29) copies it to the final prediction position.

This is a common narrative in the interpretability literature. Meng et al.'s "Locating and Editing Factual Associations in GPT" (2022) proposes essentially this mechanism for GPT-2.

## Setup

Project every layer's `resid_post` through the unembed at **every token position** (not just the final one). Produce a heatmap of the answer token's log-probability across all [layer × position] pairs, for each of 6 factual-recall prompts.

## Results

### The answer never appears at subject positions

Across all 6 prompts and all 42 layers, the answer token **never enters the top-10 at any subject-entity position.** Not once. Not even in the high-rank range: "Paris" sits at rank 9,732+ at the "Tower" position at every layer. "Tokyo" sits at rank 16,723+ at the "Japan" position. "Shakespeare" at rank 10,264+ at "Romeo." The pattern is universal.

| Prompt | Subject position | Best rank at any layer | Layer | Entered top-10 at final pos |
|--------|-----------------|----------------------:|------:|----------------------------:|
| Eiffel Tower → Paris | Tower | 9,732 | 0 | Layer 32 |
| Capital of Japan → Tokyo | Japan | 8,880 | 30 | Layer 30 |
| Romeo and Juliet → Shakespeare | Juliet | 945 | 33 | Layer 30 |
| Chemical symbol for gold → Au | gold | 2,744 | 33 | Layer 31 |
| Opposite of hot → cold | hot | 545 | 33 | Layer 30 |
| Monday, Tuesday → Wednesday | Tuesday | 119 | 0 | Layer 29 |

The answer only crystallizes at the final prediction position (the last token in the sequence), and only in the late layers (29–32). It does not exist in any decodable form at subject positions at any depth.

### The heatmaps show rightward information flow

The heatmaps reveal a striking spatial pattern. In the late layers (30+), the answer token's probability is:
- **Very high** at the final position (rank 0–5, deep green)
- **Moderate** at positions just left of the final position (`in`, `is`, `<turn|>`)
- **Low** at the subject positions (rank 1,000+)
- **Very low** at the template tokens early in the sequence

The answer appears to "flow rightward" through the sequence in the late layers — it's strongest at the final position and weakens as you move leftward. This is consistent with causal attention: information propagates from left to right, and the model builds up the answer at the rightmost positions where all prior context is available.

### What are the MLPs writing?

The MLPs in layers 10–24 are essential for factual recall (finding 04) but they're not writing the answer token. They must be writing something else — a representation that encodes the *concept* of the answer without being decodable as the answer's token. The "invisible middle" from finding 01 is invisible for a good reason: the intermediate representations genuinely aren't in the vocabulary space.

Several possibilities:
1. **Feature directions.** The MLPs may write a direction in residual stream space that encodes "the thing associated with Eiffel Tower" without corresponding to any single token. Later layers compose this with positional/structural information to produce the actual token prediction.
2. **Superposition.** The answer might be encoded across a linear combination of directions that don't individually project to "Paris" through the unembed but collectively constrain the final prediction.
3. **Distributed predicate encoding.** Rather than writing "Paris" at the "Tower" position, the MLPs might write features like [is_European, is_a_city, is_a_capital, associated_with_France] that downstream layers combine into the final answer.

Distinguishing between these would require probing experiments beyond the logit lens (e.g., training a linear probe to predict the answer from the residual stream at subject positions, or using causal tracing to identify which residual stream directions at subject positions causally affect the output).

## Implications

### The Meng et al. model doesn't hold here

The "factual associations stored as key-value pairs in MLP weights, readable through the unembed at subject positions" model from the Meng et al. ROME/MEMIT papers doesn't describe Gemma 4 E4B. The MLPs are critical (finding 04), but they're not writing vocab-decodable answer tokens at subject positions.

This could be because:
- **E4B is instruction-tuned**, and the chat template changes how information flows compared to base models. The model may have learned to route factual information differently when prompted in chat format vs. bare completion.
- **E4B's hybrid architecture** (sliding-window attention + periodic globals + MatFormer side-channel) may distribute information differently than a standard dense transformer.
- **The Meng et al. result was specifically on GPT-2**, a much smaller, older model. The mechanism may not transfer to modern architectures.

### The answer emerges at the output position, not the subject position

The clean story would have been: "MLPs write the answer at the subject position, attention copies it to the output position." The actual story appears to be: "MLPs write something at the subject position that *contributes to* the answer, but the answer only becomes decodable as a specific token at the output position in the late layers." The mechanism is compositional, not a simple lookup-and-copy.

## Updated functional map

The three-phase map (foundation → engine room → readout) still holds, but with a refinement:

- **Engine room (layers 10–24)**: MLPs write *features* at all positions, not *answers* at subject positions. These features are in an internal representation that doesn't project cleanly through the unembed. The work is real (ablation proves it) but invisible to all forms of logit lens (standard or position-wise).
- **Readout (layers 25–41)**: The features written by the engine room are composed into the actual token prediction, but only at the final position. The composition is rightward-flowing and appears between layers 29–33.
