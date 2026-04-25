# Gemma 4 global-attention spacing: 5:1 in the family, 4:1 only for E2B

The motivation for this writeup, from task [000125](https://github.com/mechbench/mechbench/blob/main/tasks/mechbench-core/done/000125-investigate-gemma-4-global-attention-spacing-rule-e4b-every.md): both Gemma 4 E4B and E2B have exactly 7 global-attention layers with the *last* layer always global, but the spacing between globals differs (E4B every 6th, E2B every 5th). Several hypotheses to disambiguate — depth-fraction of the pivot, count of KV-shared trailing layers, count of fresh-K/V globals, depth-band targeting, fixed compute budget. Web research closes the question by adding two more datapoints.

## What the configs say

| model | n_layers | global indices | period (local:global) | num_kv_shared_layers | source |
|---|---|---|---|---|---|
| Gemma 3 1B | 26 | every 6th (`sliding_window_pattern: 6`) | 5:1 | n/a | [config](https://huggingface.co/mlx-community/gemma-3-1b-it-bf16/raw/main/config.json) |
| Gemma 3 4B | 34 | 5:1 | 5:1 | n/a | [config](https://huggingface.co/mlx-community/gemma-3-4b-it-bf16/raw/main/config.json) |
| Gemma 3 12B | 48 | 5:1 | 5:1 | n/a | [config](https://huggingface.co/mlx-community/gemma-3-12b-it-bf16/raw/main/config.json) |
| Gemma 3 27B | 62 | 5:1 | 5:1 | n/a | [config](https://huggingface.co/mlx-community/gemma-3-27b-it-bf16/raw/main/config.json) |
| **Gemma 4 E2B** | 35 | [4, 9, 14, 19, 24, 29, 34] | **4:1** | **20** | [config](https://huggingface.co/mlx-community/gemma-4-E2B-it-bf16/raw/main/config.json) |
| **Gemma 4 E4B** | 42 | [5, 11, 17, 23, 29, 35, 41] | 5:1 | 18 | [config](https://huggingface.co/mlx-community/gemma-4-E4B-it-bf16/raw/main/config.json) |
| Gemma 4 26B-A4B (MoE) | 30 | [5, 11, 17, 23, 29] | 5:1 | 0 | [config](https://huggingface.co/mlx-community/gemma-4-26B-A4B-it-bf16/raw/main/config.json) |
| Gemma 4 31B (dense) | 60 | [5, 11, 17, …, 59] (10 globals) | 5:1 | 0 | [config](https://huggingface.co/mlx-community/gemma-4-31B-it-bf16/raw/main/config.json) |

(Two corrections to the working notes that motivated this task: E2B has `num_kv_shared_layers = 20`, not 21; and E2B's spacing is best read as period-5 / 4:1 ratio, not "every 5th".)

## What the docs say

The Gemma **3** tech report ([arxiv 2503.19786](https://arxiv.org/html/2503.19786v1) §2.2) names the design choice and motivates it:

> The architecture was changed to reduce the KV-cache memory that tends to explode with long context by increasing the ratio of local to global attention layers, and keeping the span on local attention short.

…with empirical finding that perplexity is robust to that change, and standard sliding-window of 1024 tokens. So the **5:1 ratio is a Gemma-3-era decision driven by long-context KV memory**, kept across the family.

The Gemma **4** technical material is thinner. The HF blog and the Google Developers blog cover Gemma 4 broadly but don't spell out the spacing rule per-variant; the `layer_types` arrays in the configs are the authoritative artifact. The widely-cited [Maarten Grootendorst visual guide](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4) summarizes the *observable* rule:

> The 4:1 pattern, however, is only for the E2B as all other variants have a 5:1 pattern… Global attention is always the last layer.

I did not find a *primary Google source* stating the "E2B is 4:1, everyone else is 5:1" rule or the "last layer is always global" rule. Both are visible in the HF configs but unattributed in the Gemma 4 papers / blogs I could locate.

## Disambiguating the hypotheses

Re-evaluating the original five with the larger sample:

| hypothesis | verdict |
|---|---|
| (a) Fixed *fraction* of depth at the L23-equivalent pivot | **Ruled out.** E4B 23/42 ≈ 0.55, E2B 14/35 = 0.40, Gemma 3 27B's analogous "second-to-last fresh global" ≈ 41/62 ≈ 0.66. No fixed fraction. |
| (b) Fixed count of KV-shared trailing layers | **Ruled out.** E4B 18, E2B 20, 26B-A4B 0, 31B 0. KV-shared-layers is an on-device-only feature of the E-series. |
| (c) Fixed count of fresh-K/V globals | **Ruled out.** E4B has 4 fresh-K/V globals, E2B has 3 (globals 4,9,14 are upstream of `first_kv_shared = 15`). |
| (d) Spacing tuned for a depth band | **Ruled out.** No band the data fits; the pivot depth varies with size, not period. |
| (e) Spacing chosen for a compute budget | **Closest, but reframed.** What the data actually supports: a *fixed period in local layers per global* (5:1 across the family for KV-cache reasons, deliberately tightened to 4:1 only on the smallest-on-device variant E2B), with the last layer pinned global as a separate stylistic choice. |

The right summary is therefore not "different spacing rules for E4B vs E2B" but: **Gemma 3 picked 5:1 for KV-cache reasons; Gemma 4 inherited that everywhere except E2B, where 4:1 must trade some local-layer count for either better long-range modeling or cleaner KV reuse on the smallest model**. The "every 6th" / "every 5th" framing is downstream of period 6 vs period 5.

## What this means for the L23-pivot story

We previously claimed the architectural pivot at L23 in E4B "generalizes across the family" because the analogous layer in E2B (L14) showed the same convergence. With the wider config table:

- The pivot lands consistently at **2 globals before the end** in the E-series: E4B [5,11,17,23,29,35,41] — pivot at 23 = position 4 of 7; E2B [4,9,14,19,24,29,34] — pivot at 14 = position 3 of 7. Not the same *index*, not the same *depth fraction*, not the same *count of KV-shared layers downstream*. The cleanest invariant we have is "the global immediately upstream of the first KV-shared region": E4B's `first_kv_shared = 24`, pivot at L23 = global immediately before. E2B's `first_kv_shared = 15`, pivot at L14 = global immediately before. **This is the new candidate generalization to test.**
- The 26B-A4B and 31B variants have `num_kv_shared_layers = 0`, so this candidate doesn't even define a pivot for them. Either the pivot phenomenon is specific to the on-device E-series (because it's an artifact of where fresh-K/V globals end and KV-reuse begins), or it generalizes to the larger non-E variants in some other way that we haven't found yet.

Both are independently interesting. The pivot-as-KV-boundary reframing is the more disciplined hypothesis and is testable on the larger E-series checkpoints if any are released; the pivot-as-something-else reframing requires running the existing experiments against 26B-A4B or 31B and seeing whether *any* layer shows the convergence.

## Sources

- [Gemma 3 Technical Report (arxiv 2503.19786)](https://arxiv.org/html/2503.19786v1)
- [Google Developers Blog — Gemma 3 explained](https://developers.googleblog.com/gemma-explained-whats-new-in-gemma-3/)
- [HuggingFace blog — Welcome Gemma 3](https://huggingface.co/blog/gemma3)
- [HuggingFace blog — Welcome Gemma 4](https://huggingface.co/blog/gemma4)
- [Maarten Grootendorst — A Visual Guide to Gemma 4](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4)
- [Google AI Developers — Gemma 4 model overview](https://ai.google.dev/gemma/docs/core)
- HF configs cited inline in the table.
