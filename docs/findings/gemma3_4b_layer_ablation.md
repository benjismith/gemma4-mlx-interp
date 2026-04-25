# No L23-style pivot in Gemma 3 4B; damage is front-loaded in early sliding layers

A first probe for task [000187](https://github.com/mechbench/mechbench/blob/main/tasks/mechbench-experiments/open/000187-test-l23-pivot-generalization-on-26b-a4b-and-31b.md): does the L23-style architectural pivot reproduce in a Gemma family member that has *no* `num_kv_shared_layers`? The motivating Gemma 4 candidates (26B-A4B and 31B) don't fit on this 32 GB Mac at bf16; the 12B Gemma 3 hung indefinitely inside `mlx_vlm.load`. Falling back to **Gemma 3 4B** (~7 GB at bf16, 34 layers, globals at [5, 11, 17, 23, 29], no KV sharing) gets us a Gemma-family non-E-series datapoint.

## What I did

Single-prompt layer ablation on `mlx-community/gemma-3-4b-it-bf16`:

- Prompt: `"Complete this sentence with one word: The Eiffel Tower is in"`, rendered through `tokenizer.apply_chat_template` (Gemma 3 instruct demands the chat template; bare-prompt encoding produced a `' ____'` placeholder top-1 in a first attempt).
- Method: replace `lm.layers[i]` in mlx-vlm's Gemma 3 layer list with an identity callable, run the forward, record Δ log p of the baseline top-1. Repeat for each of the 34 layers.
- Note: this bypasses `mechbench-core` because mechbench-core's forward path is hardcoded to `mlx_vlm.models.gemma4`. A small one-off in `bin/probe_gemma3_4b_ablation.py`.

## Result

Baseline top-1 = `'Paris'` with **log p = 0.0** (probability ≈ 1.0 — the prompt is fully overdetermined). Per-layer Δ log p of the baseline top-1 across the 34 layers:

| layer | type | Δ log p |
|---|---|---|
| 0 | sliding | −17.66 |
| 1 | sliding | −5.01 |
| 2 | sliding | 0.00 |
| 3 | sliding | **−30.88** ← peak |
| 4 | sliding | −15.58 |
| 5 | **global** | 0.00 |
| 6 | sliding | −9.75 |
| 7 | sliding | 0.00 |
| 8 | sliding | −12.50 |
| 9 | sliding | 0.00 |
| 10 | sliding | −1.50 |
| 11 | **global** | −0.01 |
| 12 | sliding | −0.00 |
| 13–32 | mixed | 0.00 (within rounding) |
| 33 | sliding | −0.01 |

**No pivot.** The damage curve is front-loaded with peak at **L3** (a sliding layer); after L10 every layer shows essentially zero damage. The Gemma-3-4B globals at L5, L11, L17, L23, L29 do *not* form a peak — L5 and L11 are quiet, the rest are completely silent. Compare to E4B's "invisible middle" (L10-24 all damaging) and L23 pivot: nothing analogous here.

## The big caveat

Baseline log p = 0.0 means the model is *certain* the answer is "Paris". A fully-confident prediction has no room to be hurt by ablating layers that operate on a residual stream that already encodes the answer. The damage we *do* see in layers 0-10 is presumably the early computation that establishes "Paris" as the answer; once it's locked in, the late layers contribute zero because they're polishing an already-correct answer.

In other words: **this single prompt cannot distinguish "no pivot exists in Gemma 3 4B" from "the prompt is too easy to expose any late-layer involvement at all."** The original L23 finding rested on the FACTUAL_15 battery with a `MIN_CONFIDENCE = 0.5` filter — prompts where the model is uncertain enough that late-layer involvement is observable.

## What this is and isn't evidence for

**It is** evidence that:
- The methodology works (chat-templated prompt produces sensible top-1; identity-layer ablation produces meaningful damage curves where there's room for damage).
- **Late-layer involvement in Gemma 3 4B is conspicuously absent on a fully-confident prompt**, in a way that didn't hold for E4B even on confident prompts (FACTUAL_15 had the L10-24 damage band). Whether this is a model-size effect, an architecture-family effect, or just the prompt is open.

**It is not** evidence that:
- Gemma 3 4B has no L23-style pivot (we didn't measure that — we measured a single prompt at logp=0.0).
- The KV-boundary framing is right or wrong (4B has no KV sharing, but the prompt-confidence confound dominates).

## Follow-up: FACTUAL_15 battery confirms the picture

Task 000189 ran the proper FACTUAL_15 battery against Gemma 3 4B (15/15 prompts validated under `MIN_CONFIDENCE = 0.5`, top-1 probabilities 0.62–1.00 — including a couple of less-confident ones like Romeo & Juliet → "William" at p=0.62). The mean Δ log p over the 15 prompts:

| layer | type | mean Δ log p | median |
|---|---|---|---|
| 0 | sliding | −14.27 | −13.87 |
| 1 | sliding | −4.88 | −0.42 |
| 2 | sliding | −1.01 | 0.00 |
| 3 | sliding | **−23.13** ← peak | −24.06 |
| 4 | sliding | −12.37 | −13.07 |
| 5 | **global** | −1.01 | 0.00 |
| 6 | sliding | −9.94 | −9.75 |
| 7 | sliding | −0.30 | 0.00 |
| 8 | sliding | −16.29 | −16.75 |
| 9 | sliding | −0.83 | 0.00 |
| 10 | sliding | −6.34 | −4.76 |
| 11 | **global** | −4.17 | −0.46 |
| 12–22 | mixed | −0.01 to −3.14 | ~0 |
| 23 | **global** | −0.23 | 0.00 |
| 24–28 | sliding | small | ~0 |
| 29 | **global** | +0.03 | 0.00 |
| 30–32 | sliding | −0.4 to −0.97 | 0.00 |
| 33 | sliding | −0.24 | −0.06 |

The top-5 most damaging layers are **L3, L8, L0, L4, L6** — every one sliding, every one in the early third of the network. The result holds when the prompt is no longer overdetermined.

**Three things this resolves:**

1. **No L23-style pivot in Gemma 3 4B.** Globals at [5, 11, 17, 23, 29] do not form a peak. L11 is mildly more damaging than its sliding neighbors (−4.17 vs ~−1) but it's not even in the top 5, let alone a load-bearing pivot.

2. **No "invisible middle" plateau.** E4B's L10–24 band of mid-network damage doesn't appear here at all. Past L11, ablating any one layer barely registers (mean abs Δ log p < 1.6 for layers 12–32, with most at ~0).

3. **The 000187 single-prompt picture was real, not a confound.** L3 remained the peak across the battery; the front-loaded shape is genuine.

**What this means for the 000125 framing.** The KV-boundary candidate ("pivot = global immediately upstream of first_kv_shared") *predicts* no pivot in Gemma 3 4B because Gemma 3 has no `num_kv_shared_layers`. The data is consistent with that framing — but it's also consistent with a simpler model-scale story (4B is small enough that mid-network refinement isn't needed). To distinguish, we'd need a non-E-series Gemma 4 (26B-A4B or 31B; both currently infeasible at bf16 on this hardware) or a Gemma 3 large enough to need late-layer computation.

**Heavy-tailed mean vs median.** Several layers have substantially more negative mean than median (e.g. L11: mean −4.17, median −0.46) — a few prompts pay a lot, the rest pay nothing. The "harder" prompts (Romeo and Juliet → William at p=0.62 baseline; Mona Lisa → Leonardo at p=1.00 but morphologically tricky) drag the mean down where the easy ones are unmoved. So: the late layers DO matter for some queries; they just don't matter on average for FACTUAL_15. A more semantically-difficult battery would likely shift the curve.

## Side observation worth folding back

Gemma 3 4B's last layer (33) is *sliding*, not global. The "last layer is always global" rule we noted in `gemma4_global_spacing.md` is therefore Gemma-4-only — Gemma 3 doesn't follow it.

## Sources

- Probe script: [`bin/probe_gemma3_4b_ablation.py`](../../bin/probe_gemma3_4b_ablation.py)
- Battery script: [`experiments/step_02_layer_ablation_gemma3_4b.py`](../../experiments/step_02_layer_ablation_gemma3_4b.py)
- Single-prompt raw data: `caches/gemma3_4b_layer_ablation.json` (gitignored)
- Battery raw data: `mechbench-ui/public/data/step_02_layer_ablation_gemma3_4b.json`
- Gemma 3 4B [config](https://huggingface.co/mlx-community/gemma-3-4b-it-bf16/raw/main/config.json)
- Original L23 finding context: [`docs/findings/gemma4_global_spacing.md`](gemma4_global_spacing.md)
