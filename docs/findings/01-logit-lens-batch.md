# Logit Lens Across 15 Prompts on Gemma 4 E4B

**Date:** 2026-04-15
**Script:** `experiments/logit_lens_batch.py`
**Plot:** `caches/logit_lens_batch.png`

## Setup

We ran a logit lens — projecting every layer's residual stream through the final RMSNorm and tied unembed — on 15 prompts covering factual recall (geography, science, culture), pattern completion, and common sense. For each prompt, we auto-detected the model's top-1 prediction at the final layer and tracked that token's rank and log-probability backward through all 42 layers. All 15 prompts had top-1 confidence above 0.5 (range: 0.52–1.00), so the trajectories are tracking tokens the model is genuinely committed to.

Gemma 4 E4B has a hybrid attention pattern: 7 global-attention layers at positions 5, 11, 17, 23, 29, 35, and 41 (every 6th layer), with the remaining 35 using local sliding-window attention. One of our motivating questions was whether the global layers would show up as visible inflection points in the logit-lens trajectory — discrete moments where long-range information arrives and the model's intermediate prediction jumps.

## Results

### The phase transition is sharp and late

The geometric mean rank of the target token across all 15 prompts stays in the 60,000–145,000 range (out of a 262,144-token vocabulary) for layers 0 through 24. The model's intermediate representations carry essentially no information about the final answer for the first 25 layers, at least as measured by projecting through the unembed. Then between layers 27 and 30, the geomean rank crashes from ~48,000 to ~16. By layer 36 it's at 2. By layer 41 it's 0 (top-1 for all prompts).

This is consistent with published logit-lens results on GPT-2, Pythia, and Llama: the residual stream is "inert" (from the lens's perspective) for a substantial early portion of the network, then the answer crystallizes in a concentrated band of layers. On E4B that band is roughly layers 27–36, or about 25% of the network's depth.

| Layer | Type   | Geomean Rank | Mean log p(target) |
|------:|--------|-------------:|-------------------:|
|     0 | local  |     126,670  |            -31.2   |
|     6 | local  |     112,909  |            -23.6   |
|    12 | local  |      63,202  |            -29.6   |
|    18 | local  |      62,321  |            -26.7   |
|    24 | local  |      71,209  |            -28.7   |
|    30 | local  |          16  |             -5.2   |
|    36 | local  |           2  |             -2.6   |
|    41 | GLOBAL |           0  |             -0.1   |

### Global-attention layers are not the inflection points

The five largest single-layer rank drops (layer i → i+1) in the geomean trajectory are:

| Transition | Layer type at i+1 | Geomean rank drop |
|-----------:|-------------------|------------------:|
|   21 → 22  | local             |          -73,365  |
|   23 → 24  | local             |          -48,393  |
|   26 → 27  | local             |          -46,626  |
|   27 → 28  | local             |          -42,521  |
|    7 → 8   | local             |          -41,064  |

Zero out of five land on global-attention layers. The single-prompt Eiffel Tower experiment had suggested that global layers might be where the model's prediction "jumps" — that was a coincidence of one trajectory, not a structural property.

This doesn't mean the global layers are doing nothing. It means whatever they contribute isn't visible as a discrete jump in the logit-lens projection. Several possibilities remain open: the globals may distribute information that the *following* local layers then exploit (making the locals look like the inflection points in the lens); the globals may contribute to representation quality in ways the tied unembed can't detect (the "tuned lens" literature argues the raw logit lens underestimates early-layer computation); or the globals may matter more for some prompt types than others, and our 15-prompt sample is too small to see it.

The layer-ablation experiment (issue `grh`) is the right next step to test whether globals are disproportionately load-bearing despite not showing up in the lens.

### Individual trajectories converge tightly

The plot shows individual prompt trajectories as thin lines and the aggregate as a bold line. In the early layers (0–24), the individual trajectories fan out wildly — different prompts place their target token at very different ranks in the intermediate distributions, with no coherent pattern. After layer 30, all 15 trajectories have converged into a tight bundle heading toward rank 0. By layer 36 the spread is minimal.

This convergence is expected (all prompts were chosen to have confident final predictions), but its sharpness is worth noting. The model doesn't gradually narrow in on an answer across many layers — it goes from "no idea" to "extremely confident" in about 10 layers, and it does this consistently across diverse prompt types.

## Observations worth following up

1. **The early layers (0–24) aren't just noise — they're structured noise.** Geomean rank fluctuates non-monotonically, dipping at layer 12, rising at layer 24. Whatever computation is happening in those layers, it's reorganizing the residual stream in ways the logit lens can partially detect. Whether this reflects "genuine" intermediate computation or just artifacts of the unembed projection is an open question the tuned-lens literature addresses.

2. **Surface-form token switching at the final layer.** In the single-prompt experiment, the model's top-1 switched from `' Paris'` (space-prefixed) at layer 36 to `'Paris'` (no space) at layer 41. The batch experiment confirmed this is systematic: for all 15 prompts, the model's final top-1 token was the no-space variant, while intermediate layers often preferred the space-prefixed variant. The final global-attention layer appears to do surface-form selection — choosing between tokenizer variants that encode the same word. This is a small but concrete observation about what the last block is for (issue `020`).

3. **The phase transition band (layers 27–36) spans exactly the gap between global layers 23 and 29.** The semantic heavy lifting happens *between* two global layers, not at them. This is suggestive — it could mean the global layer at 23 provides the long-range context that the subsequent local layers (24–28) use to assemble the answer, with the global at 29 then "confirming" or redistributing the result. But this is speculation from one experiment; ablation data would help.
