# Fact Vector Geometry: Semantic Structure in Vocab-Opaque Representations

**Date:** 2026-04-15
**Script:** `experiments/fact_vector_geometry.py`
**Plot:** `caches/fact_vector_geometry.png`

## Setup

Causal tracing (finding 09) established that the residual stream at the last subject-entity token in early-to-middle layers causally determines the factual answer, even though the answer token isn't vocab-decodable there (finding 08). This raised a natural question, borrowed from the word2vec literature: **does that opaque representation have geometric structure?**

We built a labeled corpus of 40 factual-recall prompts in 5 semantic categories (8 prompts each):

- **capital**: "The capital of {country} is" → city
- **element**: "The chemical symbol for {element} is" → symbol
- **author**: "{book title} was written by" → author
- **landmark**: "{landmark} is in" → location
- **opposite**: "The opposite of {word} is" → antonym

For each prompt, we extracted the 2560-dim residual stream vector at the last subject-entity token position at three depths: layer 5 (pre-engine-room), layer 15 (in the engine room, the most critical region per finding 04), and layer 30 (post-handoff, after the information has been routed to the final position).

Then: cosine similarity analysis, PCA projection, k-means clustering, nearest-neighbor retrieval, and silhouette scoring.

## Results

### Clean category structure emerges and sharpens with depth

| Layer | Intra-cat cos | Inter-cat cos | Separation | K-means purity | Silhouette |
|------:|-------------:|-------------:|-----------:|---------------:|-----------:|
| 5 | +0.883 | +0.852 | +0.030 | 0.925 | +0.158 |
| 15 | +0.925 | +0.865 | +0.060 | 1.000 | +0.383 |
| 30 | +0.966 | +0.889 | +0.077 | 1.000 | +0.613 |

At layer 5, the category structure is detectable (k-means purity 0.925 vs chance 0.2) but messy in PCA projection. By layer 15, the five categories cluster perfectly (100% k-means purity). By layer 30, the clusters are tight, well-separated, and clearly visible in the 2D PCA projection.

### Nearest-neighbor retrieval is 100% at layer 15

At layer 15, for every one of the 40 prompts, the vector's nearest neighbor (by cosine similarity) is another prompt from the same category. Chance is 7/39 ≈ 17.9%. Across all categories, every semantic neighbor is correctly identified.

Selected examples:
- `France` → nearest neighbor: `Italy` (capital, cos=0.977)
- `gold` → nearest neighbor: `silver` (element, cos=0.977)
- `Juliet` → nearest neighbor: `Prejudice` (author, cos=0.892)
- `Tower` → nearest neighbor: `House` (landmark, cos=0.936)
- `hot` → nearest neighbor: `fast` (opposite, cos=0.955)

The model has organized its fact vectors such that prompts asking the same type of question (e.g. "capital of X") produce residual-stream vectors that cluster regardless of what X is. This is analogous to the category structure that emerges in word2vec embeddings — but it's happening inside a transformer's residual stream, at a position the unembed can't decode.

### High absolute similarity, small but reliable category separation

All pairs of fact vectors have cosine similarity in the 0.85–0.98 range. The vectors all live in a narrow cone in 2560-dim space — they share a common "this is a factual-recall prompt" direction. The category structure is encoded in small but consistent deviations around this common centroid. This parallels word2vec after mean-subtraction or PCA-whitening: the interesting signal is in the residual differences, not absolute positions.

### The geometry sharpens even past the causal handoff

The most surprising aspect of the result is that **the cluster structure continues to improve from layer 15 to layer 30**, even though causal tracing (finding 09) showed that by layer 20+, the subject-position vector no longer causally determines the output. Two interpretations:

1. **Compression.** Once the information has been routed to the final position (layers 20–28), the subject-position vector is free to be "cleaned up" — noise and task-irrelevant content are discarded, and the remaining representation sharpens around the category prototype. The vector at layer 30 carries a compressed, distilled version of the category identity.

2. **Redundant encoding.** The model keeps a copy of the categorical structure at the subject position even after routing, possibly as a redundant signal that could be read by future layers or that serves as a "handle" for downstream processing.

Either interpretation supports the broader claim that the model uses **compositional, vocab-opaque representations**: the fact vector's job isn't to be decodable as the answer (finding 08), it's to participate in a geometric space where semantically similar facts cluster together.

### Comparison to word2vec-style embeddings

This result generalizes the word2vec phenomenon from surface tokens to internal representations. In word2vec:
- Words with similar meanings cluster in embedding space
- Related word groups (animals, countries, colors) form identifiable clusters
- Arithmetic relations exist (king − man + woman ≈ queen)

In E4B's residual stream at subject positions:
- Prompts asking similar questions cluster together
- Semantic categories (capitals, elements, authors) form identifiable clusters
- Whether arithmetic relations hold is an open question we can test next

The key difference: these are not *word* embeddings but **prompt-type representations** at a specific position within the forward pass. They encode "what kind of factual recall this is" rather than "what this word means." But the geometric structure — labeled populations clustering around category-specific centroids — is directly analogous to the vividness-lexicon approach of using cosine distances to labeled clusters as a categorization mechanism.

## Implications

**The vocab-opaque representations have exploitable structure.** We can build tools on top of this: a new prompt's subject-position vector can be compared to the existing cluster centroids to determine what *kind* of factual recall it represents, even before the model produces an output. This is genuinely useful for probing and analysis.

**Meng et al.'s ROME/MEMIT localizes facts; this shows they're also geometrically organized.** The two views are complementary: causal tracing says *where* the fact lives (subject position, middle layers); cosine geometry says *how* it's organized (clustered by relational type, in a cone of highly correlated vectors around a common centroid).

**The fact that clustering is 100% pure at layer 15 is a strong signal.** K-means is unsupervised — we didn't tell it the category labels. That the five semantic categories naturally pop out of the geometry without supervision means the category structure is a strong dominant signal, not a weak artifact. This would not be possible in a space dominated by noise or random direction.

## Follow-up ideas

1. **Vector arithmetic**: does `v(Eiffel Tower) - v(France) + v(Japan)` land near a Japanese landmark? If so, the model encodes relational structure linearly.
2. **Centroid unembed projection**: project each category's centroid through the unembed. Individual vectors aren't decodable, but the centroid might project toward category-concept words like "capital" or "city" or "element".
3. **Compare to token embeddings**: does `v_fact(Eiffel Tower)` have a systematic relationship to `v_embed(Paris)` (the tied token embedding of the answer)?
4. **Scale to 200+ prompts across 15+ categories**: the 40-prompt experiment is a proof of concept. With more data, hierarchical structure might emerge (fine-grained subcategories within broader ones, e.g., "European capitals" vs "Asian capitals").
5. **Apply to non-factual prompts**: does the same clean clustering structure exist for other prompt types (creative writing, arithmetic, coreference), or is it specific to factual recall?
