# Depth-fraction candidate refuted on Gemma 3 4B (000190)

A predictive validation pass for the depth-fraction reframing that emerged from task 000188. The 000188 work had refuted the KV-boundary framing on E2B; left behind in the data was a candidate replacement: across both Gemma 4 sizes, the median commit layer in step_33 lands at ~0.66 of network depth (E4B L25/42 ≈ 0.595 originally; recomputed here as 0.690 with slightly different boundary handling — see appendix). The candidate said: if the L23-style pivot is *fixed at a depth fraction*, then any sufficiently-deep transformer should show a commit distribution centered at that fraction, regardless of architectural mechanism. Gemma 3 4B was the test — it has no `num_kv_shared_layers`, no MatFormer, lacks the mechanism intersection 000125 originally invoked.

The framing required a Gemma 3 4B median commit fraction in the 0.6-0.7 band.

## What I found

step_38 — Gemma 3 4B FACTUAL_15 DLA sweep, run through the now-real mechbench-core hook system (000192 having landed in the same session).

| model | n_layers | median commit frac | distribution |
|---|---|---|---|
| Gemma 4 E4B | 42 | **0.690** | tight cluster L26-L31, std ≈ 0.10 |
| Gemma 4 E2B | 35 | **0.657** | cluster L20-L27 + 2 final-layer outliers |
| **Gemma 3 4B** | 34 | **0.088** | bimodal: **7/15 at L0, 1/15 at L33, rest scattered** |

Gemma 3 4B's per-prompt commit distribution by fraction:

```
Paris/London      0.529   (L18)
Tokyo/Kyoto       0.000   (L00)
China/Japan       0.088   (L03)
Brazil/Peru       0.588   (L20)
Africa/Asia       0.000   (L00)
oxygen/nitrogen   0.765   (L26)
meters/miles      0.000   (L00)
Au/Ag             0.676   (L23)
Shakespeare/Marl  0.000   (L00)
Leonardo/Michel   0.000   (L00)
five/six          0.706   (L24)
Wednesday/Thurs   0.000   (L00)
cold/warm         0.029   (L01)
blue/gray         0.147   (L05)
pets/animals      0.971   (L33)
```

Median 0.088, mean 0.300, std 0.345. **Categorically different from the E-series 0.66 cluster** — not a shifted version, not a wider version, a structurally different distribution.

Seven of fifteen prompts commit at **L0** — meaning the residual stream prefers the target over the distractor *from the embed layer onward*, with no intermediate "Berlin phase" to commit out of. This is what an extremely confident model on easy prompts looks like under the (target − distractor) DLA reduction. It's also what 000189's layer-ablation finding already implied at the qualitative level: Gemma 3 4B has very front-loaded computation, and on FACTUAL_15 it's locking in answers before any mid-network refinement is needed.

## Verdict

**The depth-fraction candidate fails.** It predicted a 0.6-0.7 median for Gemma 3 4B. The actual median is 0.088. This isn't a measurement-noise gap; it's a distribution-shape difference (E-series cluster + tail at one mode; Gemma 3 4B bimodal at L0 and L-final).

What this means, charted against everything filed today:

| candidate | first proposed | tested how | verdict |
|---|---|---|---|
| KV-boundary specific pivot ("L14 is E2B's L23") | 000125 | 000188 — E2B 4-experiment battery | **rejected** (step_02 fails outright; step_33 commits cluster far from L15) |
| Depth-fraction commit (~0.66 of n_layers) | 000188 leftover data | 000190 — Gemma 3 4B step_38 | **rejected** (Gemma 3 4B median 0.088 vs E-series 0.66) |
| Group-level fresh-K/V > KV-shared damage | 000125 | 000188 step_35/36 | **survives** (5× ratio in E2B layer ablation; KV-shared globals near-invisible to attn ablation) |
| Original L23 in E4B (5+ angles converge at one layer) | §23 of essay | the entire essay | **survives** for E4B; E2B replicates partially; Gemma 3 4B does not show analogous pattern |

Two candidate generalizations of the L23 pivot have now been tested and rejected. **The L23 pivot is increasingly looking like a Gemma-4-E-series-specific phenomenon, downstream of the fresh-K/V × KV-shared mechanism intersection specifically.** It does not generalize to Gemma 3 (no KV-sharing), and the depth-fraction reading that would have made it transferable across architecture families doesn't hold up.

What this leaves us with is a sharper, narrower claim: the L23 architectural-pivot story is real *for Gemma 4 E*, isn't a depth-fraction artifact, isn't a KV-boundary specific-pivot, and *is* something about how training pressure distributes mechanism-dependent computation in models that have a fresh-K/V → KV-shared transition. Models without that transition (Gemma 3, presumably also Qwen 2.5 once 000201 lands) won't reproduce it. Models that *do* have it but at different scales (the elusive 26B-A4B and 31B Gemma 4 variants) might or might not reproduce it depending on whether the mechanism-pressure story holds at scale — that question stays blocked on remote compute (000194).

## Methodological note

Gemma 3 has no `final_logit_softcapping`; the per-layer (target − distractor) magnitudes are 100-1000× larger than E4B's (E4B final diffs ~1-5, Gemma 3 4B final diffs +648 to +2840). Since softcap is monotonic, the **sign-change** point — which is what defines the commit layer — is unaffected. The comparison of commit-layer distributions across the family is methodologically valid; only the *magnitudes* don't compare.

The bimodal distribution (7/15 at L0) does invite the alternative hypothesis that the commit-layer metric is too easy a target for confident models. A harder battery (less-overdetermined prompts, semantically tricky lateral competitors) would test this. Filing as a possible follow-up rather than a rebuttal.

## Sources

- Battery script: [`experiments/step_38_dla_factual_sweep_gemma3_4b.py`](../../experiments/step_38_dla_factual_sweep_gemma3_4b.py)
- Raw data: `caches/step_38_dla_factual_sweep_gemma3_4b.json` (gitignored)
- E4B reference: [`mechbench-ui/public/data/step_33_dla_factual_sweep.json`](../../../mechbench-ui/public/data/step_33_dla_factual_sweep.json)
- E2B reference: `caches/step_33_dla_factual_sweep_e2b.json` (gitignored)
- Spacing finding (now requires another update): [`docs/findings/gemma4_global_spacing.md`](gemma4_global_spacing.md)

## Appendix — recomputing E4B's commit fraction

§26 of the essay reported "E4B median ≈ L25/42 = 0.595" for the commit fraction. Recomputing from the raw step_33 JSON using the same "first layer past the last negative diff" rule used here gives a median of L29/42 = 0.690. The discrepancy is whether to count the *last layer with a negative diff* (which gives the §26 number when adding 1 for some prompts) or the *first layer after the sign change* (which gives this number); the methods differ by ±2 layers per prompt depending on prompt structure. Either way, the E-series cluster is in the 0.6-0.7 band and Gemma 3 4B at 0.088 falls clearly outside it.
