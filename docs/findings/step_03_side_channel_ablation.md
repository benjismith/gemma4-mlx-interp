# MatFormer Per-Layer-Input Side-Channel Ablation

**Date:** 2026-04-15
**Script:** `experiments/step_03_side_channel_ablation.py`
**Plot:** `caches/side_channel_ablation.png`

## Background

Gemma 4 E4B uses a MatFormer-style architecture where every decoder block receives a side-channel input beyond the main residual stream. Each block has a `per_layer_input_gate` and `per_layer_projection` that combine a per-layer token embedding (from a separate 262,144 × 10,752 embedding table) with a projection of the current residual stream, gated through GELU and added back to the residual. This is architecturally unusual — most transformers route all information through the single residual stream.

The question: is this side-channel doing real work, or is it closer to vestigial?

## Experiment

Two tests on the same 15-prompt battery used in previous experiments:

1. **Full ablation**: zero out the side-channel gate in ALL 42 layers simultaneously. Measure mean Δ log p of the model's own top-1 prediction.
2. **Per-layer ablation**: zero out the side-channel gate in ONE layer at a time. Identify which layers depend most on the side-channel.

## Results

### The side-channel is massively load-bearing

Full ablation produces a mean Δ log p of **-30.4**. For comparison, the most damaging single *entire-layer* ablation (layer 0, which removes all attention + MLP + gate) was -16.0. Removing just the side-channel — a relatively small linear pathway at each layer — is nearly twice as destructive as removing the single most important layer wholesale.

Every single prompt's answer was destroyed. The model doesn't degrade gracefully to a worse-but-related answer; it produces incoherent garbage (`' St'`, `'个'`, `' maq'`, `'�'`). Whatever information flows through the side-channel, the model has no redundant pathway for it.

This decisively answers the motivating question: the MatFormer side-channel is not vestigial. It's core infrastructure.

### The side-channel matters most at global-attention layers

Per-layer ablation reveals a striking pattern. The 5 layers most affected by single-layer gate ablation:

| Rank | Layer | Type | Mean Δ log p |
|-----:|------:|------|-------------:|
| 1 | 15 | local | -5.72 |
| 2 | 17 | GLOBAL | -5.23 |
| 3 | 11 | GLOBAL | -2.79 |
| 4 | 23 | GLOBAL | -2.40 |
| 5 | 29 | GLOBAL | -1.20 |

4 out of 5 are global-attention layers. This is the first experiment in this project where the global layers clearly stand out as a structurally distinct class. They don't stand out in the logit lens (finding 01) or in whole-layer ablation (finding 02), but they *depend on the side-channel disproportionately*.

The interpretation: global-attention layers use the per-layer-input side-channel to receive token-identity information that supplements what's available through the residual stream. Local layers, which attend only to nearby tokens, apparently get enough token-level information from their sliding window. Global layers, which attend to the full sequence, seem to rely on the side-channel to ground their attention in per-token content.

### Layer 15 is an outlier

The single most affected layer is 15, a local layer sitting midway between globals 11 and 17. Its Δ log p of -5.7 exceeds even the worst global (17 at -5.2). This might indicate that layer 15 plays a bridging role — aggregating information that globals 11 and 17 need to integrate. Or it might be an artifact of 15 prompts. Worth revisiting with a larger corpus.

### Early and late layers barely use the side-channel

Layers 0–9 and 30–41 (except global layer 41 at -0.32) show near-zero impact from single-layer gate ablation. The side-channel's critical contribution is concentrated in layers 11–29, overlapping with the "invisible middle" identified in the layer ablation experiment (finding 02). The same band of layers that's most important for whole-layer ablation is also where the side-channel is most important — but through a different mechanism.

## Synthesis across three experiments

| Experiment | What it measures | Where the action is | Global layers special? |
|------------|-----------------|--------------------|-----------------------|
| Logit lens | Where the answer becomes visible | Layers 27–36 | No |
| Layer ablation | Which layers are causally important | Layers 10–24 | Modestly |
| Side-channel ablation | Which layers depend on the MatFormer gate | Layers 11–29 | **Yes, strongly** |

The side-channel experiment adds a new dimension to the picture. The global-attention layers aren't special in terms of what they produce (logit lens) or how much damage removing them causes (layer ablation). They're special in terms of *how they work* — they depend on the per-layer-input side-channel in a way that local layers don't. The side-channel appears to be an architectural mechanism specifically serving the global layers' need for per-token grounding when attending over long ranges.

## Follow-up ideas

1. **What information flows through the side-channel?** The per-layer embedding table maps token IDs to a 256-dimensional vector per layer. Is this carrying lexical identity, positional information, or something more abstract? One probe: compute the cosine similarity between per-layer embeddings for semantically similar vs. dissimilar tokens and see if the embedding space is organized semantically or syntactically.

2. **Does the side-channel carry redundant information with the main residual stream?** If we *amplify* the gate (scale by 2x instead of zeroing), does performance improve or degrade? If it degrades, the gate's contribution is finely calibrated; if it improves, the model is under-using the side-channel.

3. **Ablate attention vs. MLP in the critical middle layers (11–29)** while leaving the side-channel intact. This would disentangle whether the side-channel is compensating for attention limitations or MLP limitations in those layers.
