# Category Centroids Decode to Multilingual Relational Concepts

**Date:** 2026-04-15
**Script:** `experiments/step_11_centroid_and_arithmetic.py`

## Setup

Finding 10 established that the vocab-opaque fact vectors at subject positions cluster cleanly by semantic category. This raised three follow-up questions from the word2vec / text-embedding tradition:

1. **Centroid decoding**: what does each category's *centroid* look like when projected through the model's own output head? Individual vectors are vocab-opaque, but averaging cancels per-instance noise and should amplify whatever shared signal defines the category.
2. **Within-category linear structure**: do pairwise diff vectors inside a category share a consistent direction (king − queen-style linearity), or are they fact-specific and roughly orthogonal?
3. **Cross-category alignment**: for pairs of prompts from different categories that share an answer (e.g., "Eiffel Tower → Paris" is a landmark and "France → Paris" is a capital), do they get closer in similarity after subtracting the category centroid from each?

## Result 1: centroid projection reveals multilingual relational concepts

This is the headline result. At layers 30+, the mean-subtracted centroid of each category, when projected through the tied unembed, decodes to a cluster of tokens that are **multilingual equivalents of the category's relational frame**.

Selected top-tokens for each category's mean-subtracted centroid:

| Category | Layer | Top tokens (mean-subtracted centroid → unembed) |
|----------|------:|--------------------------------------------------|
| capital | 30 | `' தலைநக'` (Tamil), `' city'`, `' embassy'`, `' เมือง'` (Thai), `' cities'`, `' ialah'` (Malay) |
| capital | 35 | `' capital'`, `' capitals'`, `'Capital'`, `' capitale'` (Italian), `' राजधानी'` (Hindi), `' Capital'` |
| capital | 41 | `' столи'` (Russian), `' राजधानी'` (Hindi), `' capitale'` (Italian), `' capital'`, `' Hauptstadt'` (German), `' தலைநக'` (Tamil) |
| element | 30 | `'原子'` (Chinese: atom), `' chemical'`, `'Atomic'`, `' shorthand'`, `' alphanumeric'`, `'atomic'` |
| element | 35 | `' is'`, `'Elements'`, `' element'`, `'元素'` (Chinese: element), `'Element'`, `'element'` |
| element | 41 | `' is'`, `' chemical'`, `' symbol'`, `' chemically'`, `' element'`, `' atomic'` |
| landmark | 30 | `'矗'` (Chinese: stands tall), `' monument'`, `' 기념'` (Korean: commemorate), `'这座'` (Chinese: this [classifier]), `' commemorates'`, `' commemorate'` |
| landmark | 35 | `'矗'`, `' berdiri'` (Indonesian: stand), `'屹'` (Chinese: stand firm), `' stands'`, `' stood'` |
| landmark | 41 | `' overlooks'`, `' welcomed'`, `' stood'`, `' guards'`, `' berdiri'`, `' greeted'` |
| opposite | 30 | `'闇'` (Japanese: darkness), `'逆に'` (Japanese: conversely), `' reverse'`, `'ness'`, `'反'` (Chinese: opposite), `'反対'` (Japanese: opposite) |
| opposite | 35 | `'नेस'` (Hindi suffix), `'逆に'`, `'icky'`, `' backward'`, `'ness'`, `' inverse'` |
| opposite | 41 | `'falling'`, `'going'`, `'coming'`, `'thirsty'`, `'pointing'`, `'running'` |

The vocab-opaque vectors become interpretable when we decode their centroid. The individual "France → Paris" fact vector has no decodable "Paris" or "capital" content. But the centroid of 8 capital-asking prompts decodes directly to **the word "capital" in English, Italian, Russian, German, Hindi, Tamil**, plus related concepts like "city," "embassy," "cities." The representation is a multilingual concept, not an English lexical item.

This is a clean result methodologically because it doesn't require training any probes or classifiers — we're just using the model's own output head to decode an averaged internal representation. The fact that it works tells us something real about how the model organizes its middle-layer computations.

### What exactly is being decoded?

Each category's centroid appears to encode the *relational frame* — the "kind of question being asked" — rather than any specific answer or subject:

- **capital centroid** → words meaning "capital" / "city" / "embassy" across many languages
- **element centroid** → words meaning "element" / "atom" / "chemical" / "symbol"
- **landmark centroid** → words about monuments standing, commemorating, overlooking
- **opposite centroid** → words meaning "opposite" / "reverse" / "inverse" / "backward"
- **author centroid** → narrative/story verbs ("wrote," "fought," "pursued," "recounted")

Each centroid has converted a structured semantic operation ("ask for the capital of a country") into a distributed multilingual representation that the model's output vocabulary can partially decode. The mean-subtraction step is essential: without it, the centroids project toward generic continuation tokens like ` is` (the last word in every prompt). With it, we see the category signature.

### Why the multilingual spread is significant

Gemma 4 E4B is a multilingual model. The fact that averaging over 8 English prompts yields a representation whose top-decoded tokens include Chinese, Japanese, Hindi, Tamil, Italian, German, Russian, Malay, Indonesian, Thai, and Korean equivalents of the concept demonstrates that **the model's internal representation of a relational frame is genuinely multilingual** — the same middle-layer "capital-asking" direction activates all these equivalents simultaneously.

This is stronger evidence for multilingual conceptual alignment than looking at token embeddings directly. Token embeddings for "capital" and "राजधानी" might be near each other by training objective; seeing them both emerge from averaging mid-computation residuals shows that the model is processing the query in a language-agnostic way.

## Result 2: within-category diff vectors are mostly fact-specific

Mean pairwise cosine between diff vectors within each category (e.g., v(France) − v(Japan) vs v(Germany) − v(Italy)):

| Layer | capital | element | author | landmark | opposite |
|-------|--------:|--------:|-------:|---------:|---------:|
| 15 | +0.12 | +0.06 | +0.09 | +0.06 | +0.06 |
| 25 | +0.17 | +0.10 | +0.08 | +0.06 | +0.07 |
| 30 | +0.16 | +0.11 | +0.08 | +0.07 | +0.06 |
| 35 | +0.15 | +0.11 | +0.09 | +0.06 | +0.05 |
| 41 | +0.14 | +0.11 | +0.08 | +0.11 | +0.05 |

There's a small consistent "between-facts" direction in each category — non-zero, but weak (0.05–0.17). Most of the variation between two different capital prompts is fact-specific, not aligned with a common axis.

This matches the reality of embedding spaces: the clean king-queen-style arithmetic is the exception, not the rule. Capitals have the highest within-category diff consistency because all capital prompts essentially differ in one dimension (which country is named). Opposites have the lowest diff consistency because "hot/cold" and "up/down" share no structure beyond the shared category direction.

## Result 3: modest same-answer alignment after centroid subtraction

For cross-category pairs sharing an answer (France↔Eiffel Tower both → Paris; Italy↔Colosseum both → Rome), compared with control pairs that differ in both category AND answer:

| Layer | Same-answer pairs: cos after sub | Control pairs: cos after sub |
|-------|---------------------------------:|------------------------------:|
| 15 | +0.080 | −0.002 |
| 25 | +0.169 | +0.003 |
| 30 | +0.114 | −0.015 |
| 35 | +0.053 | −0.008 |
| 41 | +0.120 | +0.026 |

After subtracting the category centroid, same-answer pairs retain a small positive cosine (~0.05–0.17), while control pairs drop to near zero. The signal is real — the answer identity leaves a residual trace after removing the category — but it's modest. Only 2 qualifying pairs in our 40-prompt corpus, so this is suggestive rather than conclusive.

## Synthesis

Three levels of structure in the fact-vector geometry at subject positions:

1. **The "prompt type" direction** dominates. All fact vectors sit in a narrow cone (cosine 0.85–0.98 with each other) that encodes "this is a factual-recall prompt in this template."
2. **The "category" direction** is the strongest discriminator within that cone. K-means recovers it at 100% purity (finding 10). Centroids are multilingual encodings of the category's relational frame (this finding).
3. **The "specific fact" direction** is what's left. It's weak (mean pairwise diff-cosine ~0.05–0.17) and fact-specific. No clean king-queen-style arithmetic, but a small consistent residual that distinguishes same-answer from different-answer pairs.

The geometry has clean second-order structure (categories) but messy third-order structure (specific facts). That's consistent with how real embedding spaces behave: clean at the coarse semantic level, noisy at the fine-grained instance level.

## The centroid-projection technique more generally

What we just did is genuinely methodologically novel (as far as I can tell from the interp literature): **use the model's own output head to decode the *prototype* of a semantic operation represented in mid-layer activations.** The individual internal representations are unreadable, but their average over many instances of the same relational frame decodes to multilingual verbalizations of that frame.

This could be a useful general technique:

- Given any batch of prompts instantiating the same relational operation, average their mid-layer residuals and project through the unembed to see what the model "thinks the operation is about."
- Different layer depths expose different aspects: layer 15's centroid decodes to noisy gibberish, layer 30's centroid decodes to the pure concept, layer 41's centroid decodes to continuation-token mechanics. The operation is getting compiled down from abstract concept to specific lexical form.
- The multilingual spread at the concept layer gives a direct probe of how multilingual the model's representations actually are for a given concept.

In the context of your Shaxpir vividness work: this is close in spirit to using centroids of labeled clusters as category prototypes, but with a transformer twist — the centroid is decoded back through the model's own vocabulary rather than compared against other labeled vectors. It's a "speak yourself back through yourself" operation, and it reveals a lot.
