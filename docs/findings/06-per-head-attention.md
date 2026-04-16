# Per-Head Attention Analysis at Global Layers

**Date:** 2026-04-15
**Script:** `experiments/per_head_attention.py`
**Plots:** `caches/per_head_layer23_eiffel.png`, `caches/head_specialization_heatmap.png`, `caches/layer23_heads_grid.png`

## Setup

The previous experiment averaged attention across heads and found that global layers attend primarily to chat template tokens rather than subject entities. This experiment breaks open the 8 individual attention heads at each global layer. For each head, we compute what fraction of attention from the final position lands on subject-entity tokens vs. template tokens, averaged across 6 prompts.

Subject tokens were identified by substring matching ("Eiffel", "Tower" for the Paris prompt; "Japan", "capital" for the Tokyo prompt; etc.). Template tokens are the chat-format markers: `<bos>`, `<|turn>`, `user`, `<turn|>`, `model`.

## Results

### Layer 29 is the content-attention layer, not layer 23

The subject-attention heatmap shows a clear winner: **L29** has the highest subject-entity attention of any global layer, across almost every head. The subject-attention leaderboard:

| Layer | Head | Subject attn | Template attn | Ratio |
|------:|-----:|-------------:|--------------:|------:|
| L29 | H7 | 0.252 | 0.294 | 0.86 |
| L29 | H2 | 0.229 | 0.493 | 0.47 |
| L41 | H3 | 0.212 | 0.664 | 0.32 |
| L35 | H5 | 0.163 | 0.532 | 0.31 |
| L29 | H1 | 0.160 | 0.489 | 0.33 |

L29 H7 has a subject/template ratio of 0.86 — nearly equal attention to content and template tokens. No other head at any layer comes close to this balance. L29 is the global layer where the model actually looks at what the prompt is about.

This is a correction to finding 05, which focused on layer 23 and concluded that globals don't attend to content. Layer 23 doesn't — but layer 29 does, and it's the first global layer *after* the KV-sharing boundary. Layers 24–41 share KV caches with layers 22 and 23, meaning layer 29's attention is computed with fresh queries against keys written by layers 22–23. It's reading from the representation built by the "engine room" layers.

### Head specialization is dramatic at layer 23

The 8-head breakdown for the Eiffel Tower prompt at L23 reveals clear functional differentiation:

- **H0, H1, H3, H7** — Template-dominant heads. Concentrated on `user`, `<bos>`, `<|turn>`. These do structural routing — anchoring the generation context to the chat format.
- **H4, H6** — Late-sequence heads. Most weight on `<turn|>`, `model`, newline. These bridge the user turn to the model turn, attending to the response-boundary markers.
- **H2, H5** — The most distributed heads. They give the most weight to content tokens (including `Eiffel`, `Tower`) of any heads at L23, but are still not strongly subject-focused. These appear to be "general-purpose" heads doing a mix of structural and content work.

This pattern is consistent across all 6 prompts, as shown in the 6×8 grid plot. The same heads play the same roles regardless of prompt content — H0 always looks at template tokens, H4 always looks at late-sequence positions, etc. The specialization is a learned structural property of the heads, not prompt-dependent.

### Template attention concentrates deeper in the network

The template-attention heatmap (gray scale) reveals a clear depth gradient:

- **L5**: Template attention 0.3–0.6 per head. Relatively balanced.
- **L11, L17, L23**: Template attention 0.3–0.7. Gradually increasing.
- **L35, L41**: Template attention 0.6–0.9. Heavily concentrated on template tokens.

The deeper globals become increasingly narrow, spending most of their attention on a few fixed reference tokens. This is consistent with the "position anchoring" interpretation from finding 05: the late globals are doing less content processing and more structural stabilization.

**L29 is the exception.** It has the lightest template attention among the late globals (0.3–0.7, comparable to L17). This is precisely why it has room for content attention — it hasn't yet narrowed down to the position-anchoring pattern that dominates L35 and L41.

### Revised functional map of the global layers

| Layer | Role | Key evidence |
|------:|------|-------------|
| L5 | Chat-format initialization | Template-dominant but diffuse |
| L11 | Early structural context | Side-channel-dependent (finding 03) |
| L17 | Distributed processing | Balanced attention, side-channel-dependent |
| L23 | User→model structural bridge | Attention-critical (finding 04), template-focused with per-head specialization |
| L29 | **Content attention** | Highest subject-entity attention; content/template ratio near 1.0 at H7 |
| L35 | Position anchoring | Template-concentrated, nearly free to ablate |
| L41 | Surface-form selection | Narrow position anchoring, final readout |

Layer 23's importance (from the sub-layer ablation) is about managing the structural transition between user and model turns. Layer 29's role is to actually look at the content — to read from the subject entity's position in the representation built by layers 10–23. These are complementary functions: L23 sets up the structural context, L29 reads the content within that context.

## Synthesis

Six experiments in, the mechanistic picture has sharpened considerably:

1. **Layers 0–9**: Foundation. Layer 0's MLP is essential. Early globals establish chat format.
2. **Layers 10–23**: Engine room. MLPs write factual knowledge into the residual stream. Layer 23's attention manages user→model structural bridging with per-head specialization. The side-channel feeds token identity into the globals.
3. **Layer 29**: The content reader. The one global layer that substantially attends to subject-entity tokens. Reads from the representation built by the engine room.
4. **Layers 30–41**: Readout and anchoring. The answer becomes visible in the logit lens. Late globals narrow down to position anchoring. Layer 41 does surface-form selection.

The division of labor is clean: **MLPs store knowledge, attention routes structure, one specific global layer (29) bridges the gap by reading content from the MLP-enriched residual stream.**
