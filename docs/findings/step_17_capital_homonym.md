# Within-Token Sense Disambiguation: 'capital' Across Four Senses

**Date:** 2026-04-16
**Script:** `experiments/step_17_capital_homonym.py`
**Plots:** `caches/homonym_capital_depth_profile.png`, `caches/homonym_capital_pca_grid.png`

## Hypothesis

The factorization story (findings 10–15) said mid-layer subject-position activations encode the cognitive operation the model is preparing to run. A natural extension: when the same surface token has multiple senses, does the model's *within-token* representation factorize by sense as cleanly as its *cross-prompt* representation factorizes by operation?

The word *capital* is a near-ideal test case. It has at least four robust senses in English:

- **city**: "the capital of France", "Tokyo became the capital of Japan"
- **finance**: "raised capital from investors", "venture capital firms"
- **uppercase**: "write in capital letters", "starts with a capital letter"
- **punishment**: "a capital offense", "capital crime", "capital punishment"

We built 8 prompts per sense (32 total), each containing the word *capital* used unambiguously in that sense. Each cohort uses varied sentence templates internally so that template structure isn't a confound between cohorts. For every prompt, we extract the residual stream at the *capital* token's position at every layer, and ask:

1. At what depth does sense disambiguation kick in (vs the embedding-only L0 baseline)?
2. How clean does the separation get?
3. Do the per-sense centroids decode through the unembed to sense-relevant tokens?

## Results

### Sense disambiguation is fast, then stable

Selected per-layer separation metrics for the 4-sense grouping:

| Layer | NN same-sense | k-means purity | Silhouette | Notes |
|------:|--------------:|---------------:|-----------:|-------|
|  0    | 0.594         | 0.469          | +0.0228    | embedding only — almost identical vectors |
|  3    | 0.688         | 0.688          | +0.0594    | |
|  4    | 0.844         | 0.781          | +0.1300    | first layer above sil > 0.1 |
|  5    | 0.812         | 0.719          | +0.2070    | first global layer |
|  9    | 0.844         | 0.875          | +0.2840    | |
| **12** | **0.844**    | **0.781**      | **+0.3351** | **peak silhouette** |
| 18    | 0.812         | 0.656          | +0.1644    | engine-room separation softens |
| 30    | 0.844         | 0.688          | +0.1662    | |
| 35    | 0.812         | 0.656          | +0.2293    | secondary peak |
| 41    | 0.875         | 0.688          | +0.1192    | final layer |

K-means purity > chance (0.25) at every layer past the first. NN same-sense rate is around 0.84 from layer 4 onward and stays there through the final layer. The four senses are genuinely separable in the residual stream at the *capital* position from very early in the network.

### Two-peak depth profile

Silhouette has a primary peak at layers 7–13 (0.27–0.34) and a secondary peak around layers 33–38 (0.20–0.23). This isn't an artifact — the same shape appears in the intra-inter separation curve:

- Geometric tightness peaks in the engine room (the layers finding 02 identified as causally most important)
- Then softens through the middle layers
- Then sharpens again at the readout layers

Per finding 08, the engine room represents factual content in vocab-opaque feature space. Here we see that same engine room also doing the heaviest sense-disambiguation work for *capital* — and at its peak (L12), the engine room makes the four senses maximally distinguishable as raw geometric clusters.

### Centroid decoding: the senses become vocab-decodable at the readout layers

The headline. Each sense's mean-subtracted centroid, projected through the tied unembed, decodes to multilingual tokens for that sense.

**Layer 30** (mean-subtracted centroid, top-8 tokens per sense):

- city: `' 도시'` (Korean), `' municipio'`, `' city'`, `' mayors'`, `' şehr'` (Turkish), `' municipality'`, `' arrondissement'`, `'都市'` (Japanese)
- finance: `' investments'`, `' 투자'` (Korean), `'投資'` (Japanese), `' инвести'` (Russian), `' invest'`, `' investment'`, `' investasi'` (Indonesian/Malay)
- uppercase: `' italics'`, `' huruf'` (Indonesian: letter), `' حروف'` (Arabic: letters), `' uppercase'`, `' alphanumeric'`, `' italic'`, `' lowercase'`
- punishment: `' crimes'`, `' felony'`, `' Punishment'`, `'罪'` (Chinese: crime), `' Crimes'`, `' punishment'`, `'โทษ'` (Thai: punishment), `' delitos'` (Spanish)

**Layer 35**:

- city: `' शहर'` (Hindi: city), `' city'`, `' 도시'`, `' cities'`, `' kota'` (Indonesian/Malay: city), `' cidade'` (Portuguese), `' City'`, `'city'`
- finance: `' investments'`, `' funds'`, `' investment'`, `' Funds'`, `' investing'`, `' investors'`
- uppercase: `' uppercase'`, `' letter'`, `' letters'`, `' huruf'`, `' letras'` (Spanish), `' буквы'` (Russian: letters), `' lettering'`
- punishment: `' crimes'`, `' punishment'`, `'โทษ'`, `' crime'`, `' Crimes'`, `'罪'`, `' delitos'`

**Layer 41** (final layer — extreme concentration):

- city: `' city'` (0.007), `' of'` (0.005), `' शहर'`, `'city'`, `' cidade'`, `' городе'` (Russian), `' مدينة'` (Arabic: city), `' ciudad'` (Spanish)
- finance: `' funds'` (0.024), `' ratios'`, `' Funds'`, `' investments'`
- uppercase: **`' letters'` (0.594)**, `' huruf'` (0.103), `' letter'` (0.103), `' uppercase'` (0.055), `' letras'`, `' capitals'`, `' caps'`, `' CAPS'`
- punishment: **`' punishment'` (0.802)**, `' crime'` (0.109), `'crime'` (0.040), `' felony'`, `' Punishment'`, `'โทษ'`, `' offense'`

The uppercase and punishment centroids at L41 decode their primary token at over 50% / 80% probability respectively. That's an order of magnitude tighter than the multilingual concept decodings in finding 12 (where the strongest single-token signal was ~0.10 for the capital category).

This is the cleanest centroid-decoding result in the project so far.

### Why two peaks?

The geometric-cluster peak (L12) and the vocab-decode peak (L35–41) measure different things:

- **Geometric clustering** asks "are the four senses in distinct regions of activation space?" That's maximized in the engine room where the sense-feature representation is sharpest before the network starts compressing it for output.
- **Vocab decodability** asks "does the centroid project through the tied unembed onto sense-relevant tokens?" That's maximized in the readout layers where the representation has been transformed into vocab-space.

This mirrors the broader two-phase architecture from findings 02 / 08: the engine room (L10–24) builds vocab-opaque feature representations; the readout (L25–41) projects them into vocab space. Sense disambiguation follows the same pattern: built in the engine room, decoded in the readout.

## Interpretation

Within-token sense disambiguation in Gemma 4 E4B is a real, geometrically clean phenomenon. By layer 4 the model has already separated the four senses to ~84% nearest-neighbor accuracy. By layer 12 the silhouette score is +0.34 — well above the threshold for "obviously distinct clusters". By layer 41 the centroids decode to single dominant tokens with probabilities up to 0.802.

This extends the operation-factorization story (finding 15) downward into the within-token regime:

- Finding 15: cross-prompt activations factorize by *which cognitive operation* the prompt invokes (capital lookup vs letter counting), independent of operand identity.
- Finding 17 (this): within-token activations at the position of a homonym factorize by *which sense* the surrounding context disambiguates to, independent of the surface token (which is identical across all 32 prompts).

The mechanism is presumably: attention layers gather context from surrounding positions; MLPs at the *capital* position transform the residual to encode "this token, in this context, means the {city/finance/uppercase/punishment} sense"; that representation persists through the rest of the network as the basis for whatever the model needs to do next.

### One subtle caveat about the peak silhouette being in the engine room

Finding 08 said the engine room representations are vocab-opaque — meaning if you project them through the unembed you get nonsense. We see that here: the layer-12 centroid decodings ARE noise (`'ToInt'`, `'væ'`, `'Cass'`, `' sqm'`). But the *geometric structure* exists at layer 12 even though it isn't projecting into vocab space. The structure becomes vocab-decodable later, at the readout layers, even though the geometric clustering itself has softened by then.

This is consistent with the broader picture: the engine room does the work; the readout makes it visible. Sense disambiguation gets done in the engine room (visible only as opaque geometric clusters), then carried forward and ultimately projected onto sense-relevant vocabulary in the late layers.

## Implications for the bigger picture

- **Operation-factorization scales down to within-token sense factorization.** The same machinery that distinguishes "capital lookup" from "letter counting" at the operand position also distinguishes "city sense of capital" from "finance sense of capital" at the *capital* position itself. Same kind of residual-stream geometry, same multilingual decoding pattern, same two-phase (engine room → readout) timing.
- **The centroid-decoding technique generalizes.** Finding 12 showed it for cognitive-operation centroids. Finding 17 shows it for sense centroids. Both produce clean multilingual concept-token decodings at the readout layers. This suggests centroid decoding is a robust general technique for reading any "what is the model thinking about" signal that's encoded as a direction in the residual stream.
- **The 'centroids are passive correlates' result from finding 16 is reinforced**, not undermined. Centroids encode rich, decodable, sense-discriminative information — but that doesn't make them functional handles for invoking the relevant cognition in another prompt. Reading vs writing remain different operations.

## Caveats

- One word, four senses. *capital* is unusually rich in sense diversity; results might be sharper for it than for typical homonyms. *bank* (only really two senses) might give a smaller silhouette gap.
- 8 prompts per sense; templates are varied within cohort but a larger cohort would tighten the centroid estimates.
- Subject-position resolution finds the LAST occurrence of "capital" in each prompt. A few prompts have multiple matches (e.g., "Capital punishment is a capital crime") would behave inconsistently — none of our 32 prompts have this, but a larger cohort would need to handle it.
- Engine-room peak silhouette is +0.34, which is good but not as tight as the +0.66 from finding 12's 12-category clustering. Sense disambiguation is real but less cleanly separable than operation-type disambiguation.
- Layer 41 centroid decodings are specifically tuned to chat-template start-of-line context per finding 14 (the surface-form switching layer). The exact decoded probabilities probably depend on that.
