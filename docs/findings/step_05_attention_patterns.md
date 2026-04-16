# Attention Patterns at Global Layers in Gemma 4 E4B

**Date:** 2026-04-15
**Script:** `experiments/step_05_attention_patterns.py`
**Plots:** `caches/attn_pattern_*.png`, `caches/attn_layer23_summary.png`

## Hypothesis

Layer 23 emerged from previous experiments as a convergence point: the most attention-critical layer (finding 04), a top-5 whole-layer ablation (finding 02), and side-channel-dependent (finding 03). The last global-attention layer before KV sharing begins, it's the model's final opportunity to compute fresh attention over the full sequence.

We hypothesized that layer 23's attention at the final position would disproportionately attend to subject-entity tokens — that for "The Eiffel Tower is in → Paris," the model would be looking at "Eiffel Tower" to retrieve the answer. This would complete a clean mechanistic story: MLPs write knowledge at subject positions, global attention copies it to the prediction position.

## Results

The hypothesis was wrong. The actual finding is more interesting.

### Layer 23 attends to template structure, not content

Across all 6 prompts, layer 23's attention from the final position is dominated by **chat template tokens**:

| Prompt | Top-3 attended tokens |
|--------|----------------------|
| Eiffel Tower → Paris | `user` (0.16), newline (0.15), `<turn\|>` (0.10) |
| Capital of Japan → Tokyo | newline (0.16), `user` (0.14), `<turn\|>` (0.10) |
| Romeo and Juliet → Shakespeare | newline (0.14), `user` (0.14), `<turn\|>` (0.09) |
| Chemical symbol for gold → Au | newline (0.17), `user` (0.13), `<turn\|>` (0.12) |
| Opposite of hot → cold | newline (0.17), `user` (0.15), `<turn\|>` (0.11) |
| Monday, Tuesday → Wednesday | `user` (0.16), newline (0.15), `<turn\|>` (0.09) |

The actual content tokens ("Eiffel", "Tower", "Japan", "Romeo", "gold", "hot") receive modest, relatively uniform attention — typically 2–5% each. The subject entities are not singled out. The model is not "looking at the Eiffel Tower to predict Paris."

### Global layers show a structural attention progression

The per-prompt 7-layer plots reveal a clear progression in what each global layer attends to:

**L5, L11** (early globals): Heavy attention on `<bos>`, `<|turn>`, `user` — the opening template tokens. These layers appear to be establishing "this is a chat turn" context. Attention is fairly diffuse across content tokens.

**L17** (middle global): More distributed attention. Starts engaging with content-adjacent tokens like `:` and `word`. The most evenly spread pattern of any global layer.

**L23** (critical global): Bimodal pattern — strong attention on template tokens at BOTH the start (`user`, `<|turn>`) AND end (`<turn|>`, `model`, newline) of the sequence. This layer bridges the user turn and the model turn, attending to the structural markers that delimit them.

**L29** (late global): Attention reconcentrates on `user` and early template tokens. Content tokens lose weight.

**L35, L41** (final globals): Increasingly concentrated on just `<bos>`, `<|turn>`, `user`. By L41, almost all attention mass is on the first 3 tokens. These layers appear to be doing "position anchoring" — attending to fixed reference points rather than content.

### Reinterpretation: attention does structure, MLPs do content

This resolves an apparent contradiction in our previous findings. Layer 23 is the most attention-critical layer (ablating its attention hurts more than any other layer), yet it doesn't attend to the content tokens that carry the answer. How can it be critical without looking at the subject entity?

The answer: **by layer 23, the MLPs have already written the factual information into the residual stream at every position.** The model doesn't need to "look at" the Eiffel Tower tokens because that information has been distributed through the residual stream by layers 10–22's MLPs. Layer 23's attention is doing something orthogonal: managing the structural context of the chat format, ensuring the generation position is properly connected to the turn boundaries.

This is consistent with the side-channel finding (03): the global layers rely on the per-layer-input side-channel for token-identity information, which frees their attention to focus on structural/positional signals rather than redundantly attending to content.

The complete mechanistic picture:

1. **Layers 0–9 (foundation)**: Layer 0's MLP transforms raw embeddings. Early globals (L5) establish basic chat-turn context.
2. **Layers 10–22 (engine room)**: MLPs store and retrieve factual knowledge, writing it into the residual stream. The information propagates through local residual connections — no global attention needed.
3. **Layer 23 (structural bridge)**: Global attention connects the user-turn structure to the model-turn structure, without needing to attend to specific content tokens. This is the last layer with fresh KV computation.
4. **Layers 24–41 (readout)**: KV-shared layers translate internal representations into the output vocabulary. Late globals (L35, L41) do position anchoring on fixed reference tokens. The answer crystallizes in the logit lens around layers 27–36.

### The role of attention in transformers

This finding aligns with emerging work in the interpretability community suggesting that attention in large instruction-tuned models is less about "retrieving information from specific source positions" and more about "managing representational structure." The attention heads aren't database lookups — they're routing infrastructure. The actual knowledge lives in the MLPs.

For Gemma 4 E4B specifically, this is sharpened by the hybrid architecture: the global layers get exactly 7 chances to do full-sequence routing, and they spend those chances on structural signals rather than content retrieval. The local sliding-window layers handle position-by-position information flow, and the MLPs do the heavy computational work at each position.

## Limitations

- Averaging attention across heads obscures head-level specialization. Individual heads might attend to content — it's the average that's dominated by template tokens. A per-head analysis could reveal "subject-attending" heads alongside "template-attending" heads.
- Only 6 prompts, all in the "Complete this sentence" format. A different template or prompt style might shift the pattern.
- We're only looking at attention from the *final* position. Intermediate positions might show very different patterns — for example, the position right after "Eiffel Tower" might attend back to "Eiffel" at the global layers.
