# Logit-Lens of the Emotion Probes: Every Direction Decodes

**Date:** 2026-04-17
**Script:** `experiments/step_22_emotion_probe_logit_lens.py`
**Plot:** `caches/emotion_probes_logit_lens.png`

## The setup

Take each of the six emotion probes built in step_21, project the probe's unit-normalized direction vector through Gemma 4 E4B's tied unembed (RMSNorm + embedding-as-linear + logit-softcap), and report the top-K tokens that the direction most upweights and most downweights. This is the same analysis Anthropic ran in Table 1 of their 'Emotion Concepts' paper, for the same diagnostic purpose: if the probe captured the concept rather than a template artifact, its unembed projection should be human-interpretable and semantically aligned with the concept.

## Top tokens

| emotion | top-5 ↑ (upweighted) | top-5 ↓ (downweighted) |
|---------|----------------------|------------------------|
| **happy** | `' triumphant'`, `' celebratory'`, `' overjoyed'`, `' delighted'`, `' ecstatic'` | `' inaction'`, `' günd'`, `' Evil'`, `' протяжении'`, `'気になる'` |
| **sad** | `' memories'`, `' remembered'`, `' Erinner'`, `'💔'`, `' memory'` | `' elated'`, `' loudly'`, `' cautiously'`, `' đạt'`, `' osią'` |
| **angry** | `' grievance'`, `'😠'`, `'愤'`, `' aggrieved'`, `' frustrated'` | `' brighten'`, `' joyful'`, `' brightening'`, `' welcomed'`, `' fragrant'` |
| **afraid** | `' alarmed'`, `' emergency'`, `' danger'`, `' alarma'`, `' 紧急'` | `' proudly'`, `' celebrating'`, `' joyful'`, `' beautiful'`, `' 자랑'` |
| **calm** | `' atmospheric'`, `' atmosphere'`, `' ambient'`, `' moonlight'`, `' soothing'` | `' horrified'`, `' indignant'`, `' angrily'`, `' shocked'`, `' outraged'` |
| **proud** | `' 자랑'`, `' proudly'`, `' proud'`, `' celebratory'`, `' triumphant'` | `'챗'`, `' inconvenient'`, `'連'`, `' async'`, `' directions'` |

Top-8 versions are in the script output and plot.

## Observations

### Every probe's UP direction is semantically coherent

This is the primary result. Without exception, the five tokens each probe most-upweights are vocabulary a fluent speaker would associate with that emotion. `happy` puts its top weight on *triumphant, celebratory, overjoyed, delighted, ecstatic*. `angry` puts it on *grievance, aggrieved, frustrated*. `afraid` on *alarmed, emergency, danger*. `proud` on *proudly, proud, celebratory, triumphant*. These probes are not tracking a corpus template; they are tracking the concept the corpus was written to illustrate.

### Multilingual structure is prominent, as in section 11

The upweighted lists are not English-only. Every probe has at least one non-English token in its top 8:

- `angry` → `'愤'` (Chinese: anger), `'😠'`, `'🤬'`, `'😡'`
- `afraid` → `' 紧急'` (Chinese: urgent), `' hiểm'` (Vietnamese: dangerous)
- `proud` → `' 자랑'` (Korean: pride/boasting, and this is the #1 upweighted token)
- `sad` → `' Erinner'` (German: memory-related), `'思い出'` (Japanese: memory), `'💔'`
- `happy` → (all English in top 5, but emoji-like celebratory tokens appear further down)

This matches what section 11 of the experiment-narrative essay found for categorical centroids: Gemma 4's tied embedding aligns cross-lingual equivalents of a concept to nearby positions, so any vector pointing toward the concept region decodes multilingually. We are seeing the emotion-probe version of the same phenomenon, across six emotions.

### The DOWN direction is as informative as the UP

A clean negative correlate is strong evidence that the probe is a *direction*, not just a cluster position. Three patterns:

- **Opposite-valence decay.** `angry` and `afraid` both down-weight the positive-emotion vocabulary (*brighten, joyful, welcomed, proudly, celebrating, beautiful, 자랑, 만족*). `calm` down-weights the high-arousal-negative vocabulary (*horrified, indignant, angrily, shocked, outraged*). Each probe's negative pole is its psychological antipode, consistent with the valence/arousal cross-score structure from step_21.
- **Anti-nostalgia for `sad`.** The `sad` probe points *away from* confident/assertive words (*elated, loudly, cautiously, đạt, osią* [Polish and Vietnamese: achieve-related]). Combined with its UP list of memory-tokens, this suggests Gemma 4's sad direction is more nostalgic-memory-of-loss than acute-grief — a reflection of our hand-curated corpus, which leaned heavily on passages involving reminiscence (empty hallways, old photographs, lingering absence) rather than immediate sorrow.
- **Noise for `proud`.** The proud probe's DOWN direction is weak: the tokens are programming / technical noise (`async`, `OSP`, `unresponsive`, `챗`). Unlike the other probes, proud has no clear semantic antipode in our corpus. This is a corpus-size artifact — with only 16 training passages per emotion, the probe's negative pole has less to discriminate from than its positive pole.

### `calm` leans on ambiance rather than emotional state

Calm's top-5 UP tokens are *atmospheric, atmosphere, ambient, moonlight, soothing*. That's descriptive of *scenes* evoking calm rather than direct emotion vocabulary like *peaceful, tranquil, serene*. Looking at the training corpus reveals the source: our 16 calm passages are ambiance-heavy (lakes at dawn, rain on the garden, tea at the windowsill, snow on an empty park, the quiet of a meditation hall). Gemma 4 read what we wrote and built the probe accordingly. The probe is working, but what it captures is *what our calm passages are about*, not necessarily *what "calm" means in the abstract*.

That's a useful caveat for anyone using difference-of-means probes: the probe direction is as coherent as the corpus allows. If the corpus emphasizes scene ambiance, the probe encodes scene ambiance. A larger, more lexically-diverse corpus (step_21's `beads-f4k` follow-up) might sharpen the concept-vs-scene distinction.

### The `proud` probe's top token is Korean

`' 자랑'` (Korean: *pride, boasting*) is the single most upweighted token by any probe in the set (+29.88), ahead of English *proudly, proud, celebratory, triumphant*. It's not an artifact — the probe direction in activation space is closer to the 자랑 row of the embedding table than to the English *proud* row.

This matters for the mechbench product goal: a workbench that visualizes probe decoding should be multilingual-aware by default. Sorting by English only would miss the top signal for several emotions and would make the probes look weaker than they are.

## Verdict

All six probes decode cleanly through the unembed to concept-appropriate multilingual vocabulary. The UP and DOWN directions together carry psychological structure that matches the valence/arousal cross-score pattern from step_21. The probes are semantic, not template-surface.

Two caveats worth flagging for future work:

1. **Corpus-shape leakage.** The calm probe tracks ambiance because our calm corpus is ambiance-heavy. The sad probe tracks memory because our sad corpus leans on remembrance. The probe is a faithful read of what we wrote.
2. **Asymmetric negative poles.** Emotions with clear semantic opposites in the set (calm-vs-angry, happy-vs-sad variants) produce cleaner DOWN lists; proud's DOWN is near-random because no cohort in our six opposes pride directly.

These are both addressable by (a) more diverse corpora and (b) adding opposing-emotion cohorts. Neither undermines the core result that difference-of-means + PC-orthogonalization on 100 passages is enough to build probes whose logit-lens projections read like what a human annotator would have chosen.

## For the mechbench product

Two visualization primitives emerge from this experiment that are product-ready:

1. **Probe unembed-projection bar chart.** Horizontal bar chart of top-K up/down tokens per probe, with multilingual rendering support. Every difference-of-means concept vector has this representation; every GUI that exposes `Probe` should offer this view.
2. **Per-probe semantic summary.** A fluent-human glance at the top-up / top-down list is the fastest sanity check that the probe captured the concept. The table in this document is exactly the shape that summary view should take.

Neither is specific to emotions. Sentiment, register, modality, formality, any difference-of-means probe workflow uses the same unembed-projection diagnostic.
