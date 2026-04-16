# Proposed Experiments: Testing Operation-Factorization and Program-Handle Functionality

**Date:** 2026-04-16
**Status:** Proposed — not yet run
**Context:** Follow-up to the centroid-cosine-similarity experiments (findings 10, 11, 12, 13), after an important deflation of the original framing.

---

## Background

Findings 10–13 demonstrated that the mid-layer subject-position activations of factual-recall prompts cluster cleanly by semantic category, that their centroids (mean-subtracted and projected through the tied unembed) decode to tokens that name the category's relational operation, and that this structure holds robustly across 12 different operation types and 96 prompts with clean statistical baselines.

The original framing of these findings claimed something along the lines of "the model's internal cognition is multilingual" — citing as evidence the fact that category centroids decoded to concept-name tokens in many languages simultaneously (e.g., the capital centroid decoded to *capital*, *राजधानी*, *Hauptstadt*, *capitale*, *столица*, etc.).

That framing does not survive scrutiny. Multilingual models are trained on multilingual data, which causes cross-lingually equivalent concept-tokens to occupy the same region of the tied embedding space. When a mid-layer activation points toward the "capital-concept" region, it will decode to all the tokens in that region — across all languages. The multilingual decoding is a direct consequence of embedding-space alignment, not evidence of multilingual cognition. This is a well-known property of multilingual transformers and does not require any novel experimental support.

But the deflation of the multilingual framing does not dispose of the entire finding. What remains, when stated carefully, is a **factorization claim** about the mid-layer residual stream, and it is worth testing more directly.

---

## The Residual Claim

**Stated precisely:** At mid-to-late layers in Gemma 4, the residual-stream activation at a prompt's subject-entity position encodes the *cognitive operation being performed*, factored apart from both the *operand* (what specific entity the operation is acting on) and the *answer* (what output the operation will produce).

Two pieces of evidence for this claim are already in our data from findings 10–13:

**Same answer, different operations cluster separately.** The prompt *The Eiffel Tower is in* predicts *Paris*. The prompt *The capital of France is* also predicts *Paris*. If mid-layer activations tracked the forthcoming answer, these two prompts should cluster together. They do not. The Eiffel Tower prompt clusters with other landmark prompts (*Great Wall, Colosseum, Taj Mahal*) whose answers are *China, Rome, India* — different from *Paris* in every case. The France prompt clusters with other capital-lookup prompts whose answers are also all different. The cluster membership tracks the operation (landmark-location-lookup vs. capital-of-country), not the answer.

**Same operation, different operands cluster together.** The eight capital-lookup prompts have eight different subject countries (France, Japan, Germany, Italy, Spain, Russia, Egypt, Greece) and eight different answer cities. If mid-layer activations tracked either the subject or the answer, these prompts should be maximally dispersed. Instead, they cluster tightly, with 100% nearest-neighbor same-category hit rates.

This is a factorization: operation-type is encoded separately from operand-content. It is visible in our existing data, and it is not explained by the multilingual-embedding artifact.

---

## The Confound

There is a remaining concern. All eight capital-lookup prompts contain the literal word *capital*. A skeptical reader could argue that the mid-layer subject-position activations are not tracking the operation (capital-lookup) but rather the forward-propagated influence of the token *capital* having appeared earlier in the context. The residual stream at position 14 (the country name) has been updated by attention and MLP operations that had access to the token at position 12 (*capital*), and that token's influence is carried forward into the activation we are measuring.

The Eiffel Tower evidence argues against the strongest version of this confound — landmark prompts contain different operation-words (if any), and yet cluster together. But it does not rule out a weaker version, in which each category's cluster structure is driven by the specific token content of its prompts rather than by the abstract operation being invoked.

A clean disambiguation would require prompts that *contain* an operation-name word but do not *invoke* that operation.

---

## Proposed Experiment 1: Operation-Word Disambiguation

**Purpose.** Determine whether mid-layer subject-position activations cluster by operation-type (the cognition program being invoked) or by operation-word-presence (the literal appearance of the operation-name token in the prompt).

**Design.** Construct paired prompt sets that vary operation-type and operation-word-presence independently:

| Prompt set | Operation invoked | Operation-word present? |
|------------|-------------------|-------------------------|
| **A1** (anchor) | Capital-lookup | Yes (*capital*) |
| **A2** | Capital-lookup | No (paraphrased) |
| **B1** | Letter-counting | Yes (*capital* as operand) |
| **B2** | Letter-counting | No |

**Prompt construction.** Eight prompts each, following these patterns:

**A1 (capital-lookup, "*capital*" present):** Our existing capital prompts. *The capital of France is*, *The capital of Japan is*, etc.

**A2 (capital-lookup, "*capital*" absent):** Paraphrases that invoke the same operation without using the word *capital*. Possible forms:
- *The administrative seat of France is*
- *Where is the government of France headquartered? Answer in one word:*
- *The city that serves as France's seat of government is*

**B1 (letter-counting, "*capital*" present):** Prompts that contain *capital* as an operand of a different operation. Possible forms:
- *The number of letters in the word 'capital' is*
- *How many syllables are in the word 'capital'? Answer with a number.*
- *Spelled backwards, the word 'capital' is*

**B2 (letter-counting, "*capital*" absent):** Same operation as B1, different operand words:
- *The number of letters in the word 'elephant' is*
- *How many syllables are in the word 'paradigm'? Answer with a number.*

**Predictions under the competing hypotheses.**

*If the geometry tracks operation-type (the factorization hypothesis):*
- A1 and A2 cluster together (both invoke capital-lookup).
- B1 and B2 cluster together (both invoke letter-counting).
- A1/A2 cluster is well-separated from B1/B2 cluster.

*If the geometry tracks operation-word-presence (the confound hypothesis):*
- A1 and B1 cluster together (both contain *capital*).
- A2 and B2 cluster together (neither contains *capital*).
- The A1/B1 and A2/B2 clusters are well-separated, but the separation has nothing to do with operation-type.

*If the geometry is partially confounded:*
- Cluster structure shows a gradient — some influence from operation-type, some from word-presence.
- Intermediate case: the A1/A2/B1/B2 arrangement might be four distinct clusters rather than two.

**Measurements.** For each prompt, extract the residual-stream activation at the operand position at layer 30. Compute the 32×32 pairwise cosine-similarity matrix. Measure (a) the mean intra-operation similarity within each hypothesis's predicted clusters, (b) the mean inter-cluster similarity, and (c) the silhouette score against each hypothesis's grouping.

**What this experiment would establish.** The factorization claim becomes robust if the operation-based clustering wins cleanly. If the word-presence clustering wins, the factorization claim was a confound all along and we are back to the more trivial observation that activations track surface tokens. If the result is mixed, we have learned that the factorization is real but entangled with surface cues in a specific, characterizable way — itself a publishable observation.

---

## Proposed Experiment 2: Representation Injection

**Purpose.** Determine whether the category centroids are passive correlates of a running program, or whether they are *functional* — that is, whether injecting a centroid into a different residual-stream context can actually steer the model's computation toward that operation.

This is a stronger claim than factorization. Factorization says *the model's internal state, when it happens to be running a capital-lookup, occupies this region of embedding space.* Functionality says *putting an activation in this region of embedding space can cause the model to run a capital-lookup, even in a context where it wasn't going to.*

**Design.** Three conditions, each using the capital centroid `v_capital` computed from the original eight capital-lookup prompts at layer 30:

**Condition 1 — Inject into a neutral prompt.** Take a prompt that does not invoke any specific factual-retrieval operation, such as *Complete this sentence with one word: The following country is famous for its*. Run the forward pass normally and record the model's top predictions. Then repeat, but at layer 30, position of the final content token, add α·`v_capital` to the residual stream (for some scaling factor α). Measure the change in the predicted token distribution. If capital-city tokens (*Paris, Tokyo, Berlin, Madrid, ...*) become more probable, the centroid is steering the computation.

**Condition 2 — Inject into a different-operation prompt.** Take a prompt that invokes a different operation, such as *The past tense of run is*. Inject α·`v_capital` at the subject position at layer 30. If the model's output shifts away from *ran* toward capital-city tokens, the injection has overridden the prompt's original operation-dispatch. This is the strongest form of the steering claim.

**Condition 3 — Inject into a natural-capital-lookup prompt, but with a different country.** Take *The capital of Germany is* (which normally predicts *Berlin*). Inject α·`v_capital` at the subject position. If `v_capital` is already close to what the model is naturally doing here, we expect minimal effect (Berlin remains the prediction). This is the null/control condition.

**Parameters to sweep.**
- Injection layer L ∈ {15, 20, 25, 30, 35}. Different depths may have different sensitivities.
- Injection strength α ∈ {0.1, 0.5, 1.0, 2.0, 5.0} (in units of the residual stream's typical vector norm). Too small has no effect; too large disrupts the computation entirely.
- Injection position: operand position (most tightly matched to centroid), final position (most proximal to the prediction), both.

**Measurements.** The primary metric is the change in the log-probability of capital-city tokens under injection vs. baseline. Secondary metrics include the coherence of the output (does the model still produce a sensible single-word response?) and the specificity of the steering (does *Paris* dominate, or does the model produce some random capital? the latter would suggest the centroid carries operation-identity but not operand-identity).

**Predictions under the competing hypotheses.**

*If centroids are functional program-handles (the strong claim):*
- Condition 1: Injecting `v_capital` into a neutral prompt raises the probability of capital-city tokens above baseline.
- Condition 2: Injecting `v_capital` into a past-tense prompt partially or fully shifts the output toward capital-city tokens.
- Condition 3: Minimal effect (the model was already doing the right thing).

*If centroids are passive correlates only:*
- Condition 1 and 2: No meaningful shift toward capital-city tokens. The centroid is in the right neighborhood of embedding space but doesn't actually trigger anything when injected.
- Condition 3: Same.

*If centroids are partially functional:*
- The steering effect is measurable but small, or works only at certain layers, or requires careful scaling.

**What this experiment would establish.** If centroids steer, then the "embeddings are programs" framing is concretely vindicated — the mid-layer vectors can be used as function-pointers to invoke cognition programs, not just as reflectors of what is happening. If centroids do not steer, then the factorization claim survives but the stronger functional claim does not. Either outcome is a real result.

---

## What Both Experiments Together Would Establish

A 2×2 grid of possible outcomes:

|                           | Experiment 1 supports factorization | Experiment 1 shows confound |
|---------------------------|--------------------------------------|------------------------------|
| **Experiment 2: steers**  | Strong claim: mid-layer activations at subject positions are functional operation-handles, factored apart from operand and answer. | Contradiction — investigate. |
| **Experiment 2: null**    | Factorization real but not functional; centroids are diagnostic probes, not steering interventions. | The original finding was surface-token-driven and adds little over the multilingual-embedding baseline. |

The upper-left cell is the one where the whole line of inquiry produces its maximum value. The lower-right cell is where we learn the effect was smaller than we thought. The other two cells are independently interesting and would require re-examination of assumptions.

---

## Thought Process: How We Got Here

This section records the reasoning that led to the two proposed experiments, for future reference.

The initial framing of the centroid-cosine-similarity work emphasized the "multilingual decoding" result as the headline finding. The intended claim was roughly that the model's internal cognition operates in a multilingually-shared conceptual space, as evidenced by the fact that English-language prompts produced mid-layer activations whose decoded tokens spanned many languages.

The deflation came from recognizing that multilingual token alignment is already a property of the tied embedding space, trained into the model as a byproduct of multilingual training data. Any vector that points toward any semantic region of embedding space will decode to tokens from many languages, because equivalent-meaning tokens across languages are already clustered. The multilingual decoding observed was not evidence of multilingual cognition; it was a direct readout of a well-known property of the embedding space.

Once that overclaim was removed, the question became: what empirical content remained?

The first thing to notice was that the clustering result is independent of the multilingual decoding. The 100% k-means purity and 100% nearest-neighbor same-category hit rates do not depend on what the centroids decode to; they depend only on the geometric structure of the activations themselves. Whatever the cluster centroids "are," the fact that same-operation prompts have high cosine similarity while different-operation prompts have low cosine similarity is a real empirical observation.

The second step was to ask: what dimension is this clustering organized along? Three candidates presented themselves:

1. The activations might track the **answer** the model is about to produce.
2. The activations might track the **subject** the prompt is about.
3. The activations might track the **operation** being performed.

The existing data already contained a test of (1) vs (3): the Eiffel Tower prompt and the capital-of-France prompt both produce *Paris* as their answer, but they cluster with landmark and capital prompts respectively. If the geometry tracked answer, they should cluster together. They don't. Hypothesis (1) is ruled out.

The test of (2) vs (3) is similarly contained in the data: same-operation prompts with different subjects cluster tightly, while same-subject prompts are not really present in the dataset, so (2) cannot be the explanation either — at least not fully.

What's left is (3), operation-factorization. And this is a substantive claim: mid-layer subject-position activations are encoding *which cognition program is running*, factored apart from *what it's running on* and *what it will return*.

The next question was whether this claim has any confounds that our existing data cannot resolve. The most obvious is that our operation-categories are distinguishable by surface tokens — capital prompts all contain *capital*, element prompts all contain *chemical symbol*, etc. It is possible that the clustering is tracking these surface tokens rather than the abstract operations they name. The Eiffel Tower data argues against the strongest form of this — the landmark prompts don't share an operation-name word — but it doesn't rule out a weaker form where each category is defined by its own characteristic surface tokens.

The way to disambiguate is to construct prompts that *contain* an operation-name word but *invoke* a different operation. This is the design of Experiment 1. The specific choice of capital-lookup vs. letter-counting-using-*capital*-as-operand is convenient because *capital* naturally fits both as operation-name and as operand word.

Experiment 1, on its own, would settle whether the factorization is real or a surface confound. But even a clean positive result for factorization would leave open a weaker challenge: so what? The observation that "mid-layer activations encode task-relevant information" is well-established from the probing literature. What would make the operation-factorization finding substantively interesting is if the mid-layer vectors are not merely correlates but *handles* — if they can be used to actively invoke the operations they represent.

This is where Experiment 2 comes from. Representation injection (taking a centroid and adding it into another prompt's residual stream) is the natural test. If it steers the model's computation toward the injected operation, we have demonstrated functionality, not just correlation. This is the form of claim that aligns with the "embeddings are programs" philosophical framing: programs are not passive state but active computational entities that can be invoked.

The design of Experiment 2 was shaped by wanting three conditions of varying difficulty — neutral baseline, different-operation override, same-operation null — so that the result produces a curve rather than a single binary answer. The parameter sweeps (layer, strength, position) are to avoid missing the effect due to poor choice of injection parameters.

Finally, the 2×2 grid of possible outcomes was worked out last, as a check that the experiments really are informative regardless of which way they come out. Each of the four cells produces a different specific claim about what the model is doing, which means each experiment is worth running even if the other one is ambiguous.

The two experiments together, if run and analyzed carefully, would either sharpen the factorization finding into a defensible specific claim about Gemma 4's internal structure, or discover that the original result is weaker than we thought and refine our understanding accordingly. Either outcome advances what we know.
