# E2B pivot validation against the KV-boundary framing — step_02 (1 of 5)

Predictive validation pass for task [000188](https://github.com/mechbench/mechbench/blob/main/tasks/mechbench-experiments/open/000188-pivot-as-kv-boundary-validate-on-e2b-with-fresh-data.md). The 000125 reframe claimed: *"the pivot is the global immediately upstream of `first_kv_shared`"* — for E2B that's **L14** (globals at [4, 9, 14, 19, 24, 29, 34], `first_kv_shared = 15`). The original L23-pivot story landed by **convergence across five experiments** (steps 02, 04, 20, 30, 33); validating its E2B analogue requires running all five and looking for that same convergence on L14, plus a null test on L9 (a fresh-K/V global the framing predicts unremarkable).

This doc covers **step_02 (layer ablation) only — 1 of 5 experiments**. Subsequent experiments will land additional rows in the table at the bottom.

## What I found on step_02

E2B, FACTUAL_15 with `MIN_CONFIDENCE = 0.5`. 14 of 15 prompts validated (Eiffel Tower → Paris was the lone reject at p=0.41 — E2B is less confident than E4B and Gemma 3 4B on the bare prompt).

Damage curve highlights:

```
L0  local   −9.82
L2  local  −14.19
L4  GLOBAL  −4.66    ← fresh-K/V global
L6  local  −15.13    ← peak
L8  local  −10.44
L9  GLOBAL −11.08    ← fresh-K/V global
L14 GLOBAL  −9.79    ← predicted pivot
L19 GLOBAL  −2.64    ← first KV-shared global
L24 GLOBAL  −1.58
L29 GLOBAL  −0.45
L34 GLOBAL  −2.15
```

**Top 5 most damaging (by mean Δ log p):** L6 (−15.1), L2 (−14.2), L9 (−11.1), L8 (−10.4), L0 (−9.8). All sliding-attention except L9.

## Verdict on this experiment alone

**The KV-boundary framing's L14 prediction fails on step_02.** The peak is L6 (sliding, well outside the predicted set). Worse, the *null* test point — L9, a fresh-K/V global the framing said should look unremarkable — is **more damaging than L14**: −11.08 vs −9.79 in mean, with median going the same direction.

The framing predicted: "L14 is the special pivot global; L9 is unremarkable." The data says: L9 outranks L14, and the early sliding layers outrank both.

## A useful honesty correction about E4B

I had the L23-pivot story lodged in my head as "L23 peaks in step_02." Re-reading E4B's step_02 numbers, **L23 isn't the peak there either**. The E4B step_02 top 5 by mean is **L0 (−15.95), L14 (−6.33), L19 (−4.08), L23 (−4.07), L16 (−3.74)**. L23 is #3-#4, not #1 — and L14 (a *local* layer in E4B) outranks it.

So:
- **E4B step_02 alone never named L23 the unique peak.** L23 was the *most damaging global* in the L10-24 mid-network band, but L14 (local!) outranked it on this metric. The L23 pivot claim was always a convergence claim.
- **The KV-boundary framing's expectation that step_02 would peak at the boundary global was already a mistake of memory** before any new data. step_02 was never the cleanest signal for the pivot — that was step_04 (sublayer ablation: "MLPs dominate; only L23 is attention-critical") and step_03 (side-channel ablation, the headline finding).

This doesn't rescue the framing — but it reframes step_02-fails as a one-of-five datapoint, not a refutation.

## A real signal that does survive

There IS a clean distinction visible in the E2B data: **fresh-K/V globals carry substantially more damage than KV-shared globals**.

| layer | fresh-K/V or shared | mean Δ log p |
|---|---|---|
| L4 | fresh-K/V | −4.66 |
| L9 | fresh-K/V | −11.08 |
| L14 | fresh-K/V | −9.79 |
| L19 | KV-shared | −2.64 |
| L24 | KV-shared | −1.58 |
| L29 | KV-shared | −0.45 |
| L34 | KV-shared | −2.15 |

Mean of fresh-K/V globals: −8.51. Mean of KV-shared globals: −1.71. **5× ratio.** That's a real architectural distinction the data supports — KV-shared globals are doing markedly less work than fresh-K/V ones in E2B's compute. But this is a coarser, group-level claim than the framing's "specific boundary global is uniquely special" claim, and not the one being tested.

## Step_04 — sublayer ablation (attention vs MLP)

E4B's strongest L23 evidence: "MLPs dominate; **only L23 is attention-critical** and weakly L17." E2B run, 9 prompts validated (more rejections than step_02 because some baseline top-1s are placeholder/markdown tokens like `'**'`):

Attention-ablation top-5: **L0 (−10.54), L14 (−7.26), L12 (−6.90), L13 (−6.38), L2 (−4.77)**.

Comparison points:

| layer | role | attn Δ | median attn Δ |
|---|---|---|---|
| L14 | predicted pivot (GLOBAL) | **−7.26** | **−6.90** |
| L9 | null: fresh-K/V global | −3.04 | −1.25 |
| L19 | first KV-shared global | −0.88 | +0.00 |

**This time the framing's predictions partially hold:**

- L14 is #2 by mean attention damage (after L0, which is the trivial baseline — first layer is always damaging in any layer-ablation experiment).
- **L9 behaves as a null** at less than half L14's damage.
- KV-shared globals (L19+) are nearly invisible to attention ablation — L19 attn Δ = −0.88, L29 = +0.03, L34 = +0.02.

**But the prediction isn't clean.** L12 and L13 — *sliding* layers immediately upstream of L14 — are also attention-critical at Δ ≈ −6.5. So what the data actually supports is **a "L12-L14 attention-critical band", not an isolated peak at L14**. L14 is the *most* attention-critical layer in that band, and it happens to be the global one, but the framing's "specific boundary global is uniquely special" claim is fuzzy at best.

Also notable: **MLP ablation dominates almost everywhere**, with peak at L0 (−19.2), then a wide band L2–L11 all in the −10 to −17 range. The "MLPs dominate" half of E4B's step_04 finding reproduces cleanly on E2B; the "only L23 is attention-critical" half maps to "L14 ± neighbors are attention-critical" — partial reproduction.

## Convergence table (in progress)

| experiment | E4B peak (re-checked) | E2B peak | E2B null L9 | E2B verdict |
|---|---|---|---|---|
| step_02 layer ablation | L23 (#3-#4) | L6 (sliding) | **L9 outranks L14** | **fails** |
| step_04 sublayer ablation (attn) | "only L23 attn-critical" | L14 #2 (after baseline L0); L12, L13 also high | L9 unremarkable as predicted | **partial** |
| step_20 homonym layer ablation | — | — | — | not yet run |
| step_30/31 perplexity probe | R² peak L21, rotation L22→L23 | R² peak **L12**, rotation **L13→L14** (test R²=0.619) | (probe doesn't have a null point) | **partial** |
| step_33 DLA factual sweep | — | — | — | not yet run |

## Step_30/31 — perplexity probe

`step_31_perplexity_probe_e2b.py` already ran in an earlier session with its own finding at [`docs/findings/step_31_perplexity_probe_e2b.md`](step_31_perplexity_probe_e2b.md). Folding its result into the convergence picture:

- **R² peak at L12** (test R² = 0.619), not at L14 (predicted) — peak is **two layers before the predicted pivot**.
- **Sharpest rotation L13 → L14** (cosine 0.0152) — at the predicted pivot.
- This same offset pattern holds in E4B: there, the R² peak is at L21 (two before the predicted L23), with the rotation at L22 → L23. **The pattern transfers**: R² peaks just before the KV-boundary global, then the residual rotates *at* the boundary.

So step_30/31 says: **the predicted pivot is the rotation site, but not the peak of surprisal-decodability**. That's a much narrower architectural claim than "L14 is uniquely special" — the actual story is "L12-13 host peak surprisal info; L14 is the boundary where the basis rotates." Partial support, with the rotation aligned to the predicted pivot but the magnitude peak two layers upstream.

## Step_33 — DLA factual sweep

For each of the 15 FACTUAL_15 prompts, compute (target − distractor) at every layer's resid_post via direct logit attribution, find the *commit layer* (last layer where the diff is still negative — i.e. the residual still prefers the distractor — plus one).

E4B's analogous step_33 finding: commits cluster around L25, just after the L23-L24 KV-boundary. The framing predicts: E2B commits should cluster at L15, just after L14.

**E2B commit-layer distribution across 15 prompts:**

| layer | #prompts |
|---|---|
| L14 | 1 (Shakespeare/Marlowe) |
| L20 | 2 |
| L21 | 2 |
| L23 | 4 ← median |
| L24 | 1 |
| L25 | 1 |
| L26 | 1 |
| L27 | 1 |
| L35 | 2 (final-layer commits — Brazil/Peru, meters/miles, both with weak final diffs) |

- At predicted L15: **0/15**
- Within ±2 of L15: 1/15
- After L15 (which is most of network): 14/15
- **Median commit layer: L23**

The strongest DLA peaks (where the residual is most target-preferring) are L28 (+7.60) and L29 (+6.67, GLOBAL — a *KV-shared* global). L14 itself sits at mean +0.03, indistinguishable from zero.

**Verdict: fails outright.** The commits don't cluster anywhere near L15, and L14 is invisible in the DLA curve. What the data actually shows is *late-network commitment* — the model decides in L25-L33, with the **KV-shared globals (L29, L34)** being where target-preference is strongest, the opposite of what the framing's "fresh-K/V globals are special" intuition suggested.

## Where this leaves the framing (4 of 5 experiments)

| experiment | E2B verdict | what it says about L14 |
|---|---|---|
| step_02 (layer ablation) | **fails** | L14 part of a fresh-K/V cluster but L9 outranks it |
| step_04 (sublayer ablation) | **partial** | L14 is the most attention-critical non-trivial layer (L12-13 also high); L9 unremarkable as predicted |
| step_30/31 (perplexity probe) | **partial** | R² peaks two layers *before* L14; the rotation is *at* L14 |
| step_33 (DLA factual sweep) | **fails** | Commits cluster at L20-27, median L23 (far from predicted L15); L14 is invisible in DLA curve |

## Verdict on the KV-boundary framing

**Two fails (step_02, step_33), two partials (step_04, step_30/31).** The framing's specific claim — "L14 is the architectural pivot in E2B" — does not hold up to predictive testing. step_33 in particular is decisive: the model's factual commit happens deep in the KV-shared region, not at the KV boundary.

What's left of the framing:

- **The fresh-K/V vs KV-shared distinction is real** at the group level. step_02 showed fresh-K/V globals carry 5× the damage of KV-shared globals; step_04 showed KV-shared globals are near-invisible to attention ablation.
- **Something does happen at L14**, but it's a rotation site rather than a metric peak — step_30/31 shows the cosine basis changes at L13→L14, even though magnitude peaks at L12.
- **The "specific layer is uniquely the pivot" framing fails.** L14 isn't the peak of layer-ablation damage, attention-criticality, perplexity-probe R², OR commit timing in E2B. The L23 framing for E4B suffers the same problem when re-examined carefully — L23 isn't the peak in step_02 or in DLA either.

## What story the data actually supports

Across both E4B and E2B, the most defensible architectural claim is:

> **Compute character changes across the fresh-K/V → KV-shared boundary, with fresh-K/V globals doing more attention-critical work and KV-shared globals doing more late-stage residual integration. The "pivot layer" abstraction is a smoothed-over version of a coarser story about which globals are operating in which regime.**

This is much weaker than "L14 = E2B's L23." It's also more honest: the KV-boundary claim survives in coarse-grained form (fresh-K/V vs KV-shared globals are doing different work), but the precise-layer claim doesn't.

The remaining open question — does any specific layer in E2B carry an L23-style multi-experiment convergence — has a different answer: **L23 in E2B** turns up as a quiet but consistent damage point and is the median commit layer in step_33. **L23/35 ≈ 0.657** of network depth; **E4B commit median ≈ L25/42 ≈ 0.595**. Comparable fractions. Maybe the right reframe isn't KV-boundary at all but **"the commit happens at a fixed depth fraction of ~0.6"**, which would make this a model-scale property, not an architecture-family one. Filing as a candidate for follow-up.

## Step_20 not run

Given two fails and two partials in steps 02/04/30/33, step_20 (homonym layer ablation) wasn't expected to flip the verdict. The framing is already refuted on the strongest tests it had. Skipping it for now; can come back if a future framing wants to use it.

## Action on `gemma4_global_spacing.md`

Demote the "pivot = global immediately upstream of first_kv_shared" framing from "candidate generalization" to "rejected (predictive test failed on E2B)". Note the surviving group-level claim: fresh-K/V globals carry more attention-critical work than KV-shared globals.

## Sources

- Battery script: [`experiments/step_35_layer_ablation_e2b.py`](../../experiments/step_35_layer_ablation_e2b.py)
- E2B raw data: `mechbench-ui/public/data/step_35_layer_ablation_e2b.json`
- E4B raw data (re-read): `mechbench-ui/public/data/step_02_layer_ablation.json`
- Original framing: [`docs/findings/gemma4_global_spacing.md`](gemma4_global_spacing.md)
