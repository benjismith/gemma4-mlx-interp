# Cohesion Doesn't Predict Decoding Sharpness

**Date:** 2026-04-16
**Script:** `experiments/step_18_cohesion_analysis.py`
**Plot:** `caches/cohesion_vs_decoding.png`

## The metric

**Cohesion** of a cluster: the mean cosine similarity between each cluster member and the cluster's centroid.

```
cohesion(C) = (1/|C|) * sum_{m in C} cos(m, mean(C))
```

Range [-1, 1]; higher = members tightly grouped around their own mean. It's distinct from the intra-cluster mean pairwise cosine that we already use (which is member-to-member); cohesion is member-to-centroid (n similarities rather than n(n-1)/2). Cohesion is generally ≥ the pairwise mean since the centroid is the angular best-fit anchor for the cluster.

## The hypothesis

Open. Three a priori possibilities:

1. **Positive correlation** — tighter clusters give sharper, more concentrated centroid decodings. This is the obvious bet: if all members agree on a direction, the centroid is a clean prototype of that direction, and projecting it through the unembed should produce a peaked token distribution.
2. **Null** — cohesion measures something different from decoding sharpness; no relationship.
3. **Negative correlation** — tighter clusters give noisier decodings. Counterintuitive but possible if very tight clusters reflect template-specific consensus rather than abstract-concept consensus.

## Test corpus

16 clusters across two corpora at two readout depths:

- **BIG_SWEEP_96**: 12 categorical clusters (capital, element, author, landmark, opposite, past_tense, plural, french, profession, animal_home, color_mix, math), 8 prompts each. Subject = the operand position (country, element name, book title, etc.).
- **HOMONYM_CAPITAL_ALL**: 4 sense clusters (city, finance, uppercase, punishment), 8 prompts each. Subject = the *capital* token itself.

For each cluster, at layer 30 and layer 41 (the canonical decoding depths from findings 12 and 17 respectively), compute:
- Cohesion at that layer.
- Top-1 decoded probability of the mean-subtracted centroid (subtracting the corpus-wide mean to remove template common-mode).

Then plot cohesion vs top-1 probability and compute the Pearson correlation.

## Results

### The correlation flips sign between layers

| Layer | Pearson r | Cohesion range | Top-1 prob range |
|------:|----------:|---------------:|-----------------:|
| 30    | **+0.306**  | [0.94, 1.00]   | [0.008, 0.385]   |
| 41    | **−0.244**  | [0.94, 1.00]   | [0.004, 0.944]   |

Both correlations are weak. Neither survives a strict statistical test at n=16. But the qualitative reading is interesting: at the layer where the original centroid-decoding finding was sharpest (L30, the canonical anchor depth from finding 12), there's a slight positive trend. At the layer where the homonym-sense decodings became single-token-dominated (L41), the trend slightly inverts.

### Cohesion alone doesn't tell you which centroids decode well

The two most striking outliers at layer 41:

- **math**: cohesion 0.996 (highest in the corpus), top-1 prob **0.944** on `' equals'` — sharpest decoder, also the most cohesive. Confirms hypothesis 1 in this case.
- **punishment**: cohesion 0.937 (**lowest** in the corpus), top-1 prob **0.802** on `' punishment'` — second-sharpest decoder, least cohesive. Confirms hypothesis 3 in this case.

These two points sit on opposite extremes of the cohesion axis but at nearly the same height on the decoding-sharpness axis. Whatever determines decoding sharpness, it isn't simply how tight the cluster is.

### The HOMONYM clusters are systematically less cohesive than BIG_SWEEP

The four HOMONYM clusters all sit in cohesion 0.94–0.97 at both layers. The 12 BIG_SWEEP clusters all sit in 0.96–0.998. The HOMONYM cohesion range is shifted left by ~0.02 and never overlaps the highest BIG_SWEEP values.

Why: BIG_SWEEP prompts share a fixed template (*"Complete this sentence with one word: ..."*) — eight prompts vary only in the operand. Template structure dominates the residual stream, so all eight vectors at the operand position look very similar. HOMONYM prompts vary template (*"The capital of France is Paris."*, *"Paris serves as the capital of France."*, *"Tokyo became the capital of Japan in 1868."*) — eight prompts vary in sentence structure. More template diversity, looser cluster.

But the HOMONYM clusters span almost the *entire* range of decoding sharpness at L41 — from `' city'` at p=0.007 (nearly flat) to `' punishment'` at p=0.802 (extremely sharp). So *more* template diversity didn't uniformly hurt the centroids, and *less* template diversity (BIG_SWEEP) didn't uniformly help. The specific concept being averaged matters more than the cluster's compactness.

### Some clusters sharpen between L30 and L41; others flatten

Per-cluster top-1 probability shifts from L30 to L41:

| Cluster | L30 top-1 prob | L41 top-1 prob | direction |
|---------|---------------:|---------------:|:---------:|
| math (BIG_SWEEP) | 0.030 | **0.944** | sharpens 31× |
| punishment (HOMONYM) | 0.060 | **0.802** | sharpens 13× |
| uppercase (HOMONYM) | 0.022 | **0.594** | sharpens 27× |
| capital (BIG_SWEEP) | 0.111 | 0.189 | sharpens 1.7× |
| profession (BIG_SWEEP) | 0.012 | 0.178 | sharpens 15× |
| plural (BIG_SWEEP) | 0.272 | **0.014** | flattens 19× |
| opposite (BIG_SWEEP) | 0.385 | **0.028** | flattens 14× |
| french (BIG_SWEEP) | 0.066 | 0.004 | flattens 16× |
| city (HOMONYM) | 0.032 | 0.007 | flattens 5× |

Some clusters' centroids dramatically tighten as we move toward the output; others dramatically flatten. The same cluster's *cohesion* changes very little across these layers (e.g., plural cohesion is 0.987 at L30 and 0.991 at L41 — barely moved), but its decoding sharpness collapsed by 19×. That's a clear demonstration that decoding sharpness is *not* a function of cohesion alone.

### What does predict decoding sharpness, then?

Looking at the patterns, the layer-41 sharp decoders share a property: **the cluster's concept word is something the model would naturally generate as a next token in the prompts' contexts**.

- math centroid → `' equals'`: the prompts (*"Two plus two equals"*) literally end on or near "equals" and the next-token continuations involve arithmetic.
- punishment centroid → `' punishment'`: the prompts (*"capital punishment is controversial"*, *"capital crime"*) discuss punishment as their topic; "punishment" is a likely continuation token in many of those contexts.
- uppercase centroid → `' letters'`: the prompts mention writing in capital letters; "letters" is the natural completion of "capital ".

The flat decoders share the opposite property: **the cluster's concept word is *referenced* in the prompts but isn't a likely next-token output**.

- plural centroid → `'plural'` only at p=0.014 at L41: the prompts (*"The plural of mouse is"*) end on "is" and the model is about to generate the plural form ("mice"), not the word "plural" itself.
- opposite centroid → `'going'` at p=0.028 at L41: the prompts ask for an antonym; the model is about to generate the antonym, not the word "opposite".
- french centroid → `' translated'` at p=0.004 at L41: the prompts ask for translations; the model is about to generate the French word, not "translated".

So at L41 — the very last hidden layer, where the model has decisively committed to its next-token distribution — centroid decoding sharpness measures something specific: *how concentrated the model's about-to-output distribution is on a single token across the cluster's prompts*. Cohesion in activation space and concentration in vocabulary space are different geometric facts about the same vector, and the relationship between them runs through the tied unembed in a way that depends entirely on what the model is preparing to say.

At L30 the picture is murkier because L30 is earlier in the readout phase — the representations are partially decodable but haven't yet committed to specific tokens. The slight positive cohesion-vs-decoding-sharpness correlation there might reflect the simpler "tighter cluster → cleaner concept averaging" intuition, before the late-layer next-token dynamics start dominating.

## Verdict

Cohesion is a real, simple-to-compute metric of cluster compactness. It is *not*, by itself, a predictor of how cleanly the cluster's centroid will decode through the model's tied unembed. The relationship between the two metrics is weak (|r| ≤ 0.31 at either layer in this corpus) and *changes sign* between layer 30 and layer 41.

What does predict decoding sharpness, especially at the deepest readout layer, is **whether the cluster's prompts cause the model to converge on a single canonical next-token across cluster members**. That's a property of the model's expected output behavior on those prompts, not of the cluster's geometric tightness in activation space.

This is a useful negative result for the framework as a whole. The centroid-decoding technique from findings 11/12/17 has a known cost (mean subtraction) and a known benefit (multilingual concept tokens come out the other end). What it doesn't have is a reliable predictor of which centroids will decode cleanly. Cohesion was a reasonable candidate; it doesn't fill that role.

It also reframes finding 17's headline a bit. The sense-disambiguation centroids at layer 41 (' punishment' at 0.802, ' letters' at 0.594) decode unusually cleanly partly because they're in the HOMONYM corpus's looser-cohesion regime — meaning the cluster averages over more template variation, and the centroid is closer to the "what is this prompt about" abstraction rather than the "what specific operand is being asked" specifics. That's a virtue of the homonym setup, not a property the centroid would have inherited from a tighter cluster.

## Caveats and follow-ups

- n=16 clusters is small. The correlation values shouldn't be over-interpreted; the structural pattern (sign flip between layers, role of next-token-output identity) is the qualitative finding.
- Layer 30 and 41 only. Cohesion was tighter at the engine-room peak (L12) per finding 17; an extension would be to compute cohesion at L12 and decoding at L41 (cross-layer comparison), to test whether engine-room compactness predicts readout-layer decoding sharpness.
- Top-1 probability is one of several plausible "decoding sharpness" measures. Top-5 mass and decoded entropy were also computed and behave similarly. A semantically-grounded measure (does the top-K decoded set include the cluster's concept-name token?) would be more meaningful for some categories but requires manual labeling.
- The "next-token convergence" interpretation is a hypothesis. A clean test would be to construct two cohorts: one where every prompt naturally elicits the same single next-token, one where prompts elicit varied next-tokens but share an abstract concept. If the first decodes more sharply at L41 than the second, regardless of cohesion, the hypothesis is confirmed.
