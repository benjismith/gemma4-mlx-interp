# OV-Circuit Activation Trajectories: Static vs Dynamic Head Behavior

**Date:** 2026-04-17
**Script:** `experiments/step_27_ov_trajectories.py`
**Output:** `caches/step_27_output.txt` (verbatim tables for all 5 prompts × 5 heads × 2 views)

## The setup

Complement to step_26's static weight-level analysis. Step_26 found that specific heads have clean multilingual concept structure in their OV circuit singular components — L5 h3 rank-1 writes *"nation"* across languages, L23 h7 rank-1 writes *"movement"*, L40 h6 rank-4 writes *"created"* in 4 European languages. But those are **potential** patterns under SVD decomposition of the frozen weights. This experiment measures what each head is **actually** writing at each position during a live forward pass.

Two new framework functions in `head_weights.py`:

- **`head_ov_position_writes(model, ids, layer, head)`** — for each position *p*, projects `V[group, p]` through `W_O[h-slice]` and the tied unembed. Returns the tokens this head **would** write if it attended fully to position *p*. Independent of the actual attention pattern. Think of it as the head's "potential output" at each input position.

- **`head_ov_actual_writes(model, ids, layer, head)`** — for each query position *q*, projects `per_head_out[h, q]` (the `softmax(QK) @ V` output) through `W_O[h-slice]` and the unembed. Returns the tokens this head **actually** writes at *q* under its real attention pattern.

Five heads (the ones step_26 identified as interpretable at some singular rank) × five prompts (chosen to exercise each head's predicted concept) × both views = 50 per-position token-list tables.

## Headline finding

**Most heads are "attention sinks" in practice, even when they have clean concept structure in their weights.**

The static weight-level analysis and the activation-level analysis *disagree systematically*. Static analysis shows what each head is capable of writing in vocabulary terms. Activation analysis shows that in real forward passes, most heads concentrate their attention on a small set of early/boundary positions (typically `<bos>`, `<|turn>`, or `user`) and therefore contribute a nearly-constant output — the value vector from those attended-to positions — at *every* query position.

Three specific patterns emerged:

### 1. "Potential writer" heads whose actual output is attention-sink noise

Example: **L40 h6** ("created" rank-4 writer from step_26).

On the prompt *"The committee created a comprehensive new set of criteria."*:

| position | query token | **potential** top-4 writes |
|---------:|-------------|---------------------------|
| 6 | ` created` | `' lasting'`, `' havoc'`, `'lasting'`, `'zos'` |
| 12 | ` criteria` | `'ção'` (Portuguese), `'šnje'` (Slavic), `' sviluppo'` (Italian: development), `' lancement'` (French: launch) |
| 17 | `model` | `'assign'`, `'assigned'`, `'Assign'`, `' Assigned'` |

The potential writes show real semantic structure: multilingual launch/development vocabulary at the `criteria` position, an assignment cluster at the `model` (chat-template) position. The head's weights encode meaningful directions that WOULD produce those writes if it attended to those positions.

| position | query token | **actual** top-4 writes |
|---------:|-------------|---------------------------|
| 6 | ` created` | `'}}_{\\'`, `'</h4>'`, `'}}^{'`, `' Defensa'` |
| 12 | ` criteria` | `'}}_{\\'`, `'\\|_{\\'`, `')}_{\\'`, `'}}^{'` |
| 17 | `model` | `'}}_{\\'`, `'}}^{'`, `')}_{\\'`, `'</h4>'` |

LaTeX fragments at every position. The head's attention is locked on positions 0-2 (`<bos>`, `<|turn>`, `user`), whose V vectors decode to LaTeX-like tokens. The actual contribution is basically a constant bias, not the meaningful "created" direction the static analysis found.

### 2. "Detector" heads that confirm the static picture — clean inputs, noisy outputs, both in static and dynamic

Example: **L7 h3** (step_26 identified it as a multilingual *"system"* detector via clean inputs at rank 0).

On the prompt *"The operating system reboots automatically at midnight."*, the actual writes at most positions are cryptic tokens (`'whenever'`, `'ropath'`, `'닙'`, `'ən'`). The head doesn't cleanly write anything the unembed can decode.

This matches the static finding: L7 h3's OV circuit has clean *inputs* (what it detects — forms of "system" in many languages) but noisy *outputs* (what it writes). In activation-level analysis, this pattern reproduces: the head fires on system-adjacent positions but emits a residual-space signal that doesn't project cleanly to any vocabulary direction. Its actual computational role is probably to write an opaque residual-stream signature that downstream layers use — not to push toward any specific output tokens.

### 3. "Partial" writers with some position-variation

Example: **L5 h6** (step_26 identified it as a multilingual *"model"* writer via clean outputs at rank 0).

On the prompt *"The statistical model captures the structural patterns of the data."*:

- **Potential at position 0** (`<bos>`): `' fitted', ' модель' (Russian: model), ' candles', 'モデル' (Japanese: model)` — the "model" concept IS in this head's weights, anchored at the `<bos>` position.
- **Actual at position 4** (`'The'`): `' проводи' (Russian: conducting), ' цветов' (Russian: flowers), 'itores', ' ivy'` — varied, but no "model."
- **Actual at position 6** (`' model'`): `'adam', ' fume', ' виде' (Russian: in the form), 'erce'` — different from positions 7-14, suggests some position-specific behavior.
- **Actual at position 18** (`'model'` chat-template token): `' ivy', ' 퍼' (Korean), ' цветов' (Russian), ' få' (Swedish: get)` — completely different mix from positions 7-14.

So there IS position-variation in L5 h6's actual writes (unlike L40 h6's uniform LaTeX), but the writes aren't obviously semantic. The head's attention pattern is diffuse rather than locked on a single sink, but the resulting per-query contributions don't map cleanly to vocabulary.

### 4. Pure attention sinks

Example: **L5 h3** ("nation" writer at rank 1 per step_26).

On the prompt *"The French nation celebrates its independence on Bastille Day."*, the actual writes at essentially every query position are `'锡'` (Chinese: tin), `'稀'` (Chinese: rare), `'essential'`, `'ანი'` (Georgian) — literally the same tokens at every position. This is the textbook attention-sink pattern: the head's attention is pinned to one specific position (position 0 in this case), so `per_head_out[h, q, :]` is approximately `V[group, 0, :]` regardless of `q`. The "nation" concept the static analysis found at rank 1 is effectively dormant during normal inference.

## Why static analysis overstates heads' roles

Three compounding factors explain the divergence:

1. **Attention sinks dominate.** In decoder-only transformers trained at scale, many heads develop strong attention to `<bos>` and early tokens — it's a learned "default position" that stabilizes the residual stream when the head has nothing to do. Research (Xiao et al. 2023, Darcet et al. 2024) has documented this extensively. Our results show it at per-head granularity.

2. **SVD reveals all subspaces, use or not.** The rank-k singular components of a head's OV matrix include every direction the head is *capable* of writing, including directions that are rarely engaged in practice. A head can have a clean semantic direction at rank 3 that only fires when its attention lands on specific tokens, which happens rarely in typical prompts.

3. **Attention filters the subspace.** A head's `W_V` projects residuals into its small head-dim subspace. Which specific directions within that subspace are actually activated depends on *which positions* the head attends to. An attention-sink head activates the subspace position 0's value inhabits; a context-matching head activates the subspace of the attended content.

## Implications for interpretation

**Static weight-level maps (like step_26's) are necessary but not sufficient.** They tell us what each head *could* do. To know what each head *does*, we need activation-level analysis on real prompts.

**Activation-level analysis tells a different story.** Many "interesting" static heads are quiet contributors in practice — or contribute constant biases that other heads and MLPs depend on. The architectural fact of 336 heads, each with 5+ concept directions in weights, doesn't mean 336 × 5 = 1680 active specialized subcomputations. Most of them are latent.

**Mixed pictures are common.** L5 h6 shows some position-variation but not clean semantics. L5 h3 and L40 h6 are attention sinks. L7 h3 is a detector with noisy writes. None of the five heads we looked at acted as a clean "writer" of its static-analysis concept in actual inference. Finding such a head would require more search — probably across many heads and many prompts.

## For the mechbench product

Three viz primitives that fall out of this work:

1. **Potential-vs-actual diff view.** For a chosen (head, prompt), render both trajectories side-by-side. Rows = positions; columns = top-k writes for potential and actual. Diverging patterns are the interesting cases.
2. **Attention-sink detector.** A head whose actual writes are near-constant across positions is probably an attention sink. Flag them automatically. (Heuristic: cosine similarity between write vectors at adjacent positions; high constant similarity = sink.)
3. **Position-write heatmap.** For each head, render a [layer × head × position] volume colored by the top-1 actual write token. Browseable, clickable, filterable by "token = X across multiple positions" or "token varies per position."

## Caveats and follow-ups

- Five heads × five prompts is a tiny sample. The patterns are representative but not exhaustive.
- We looked only at heads step_26 flagged as interpretable. Heads the static analysis rated as noisy might show surprising activation patterns — worth a sweep across all 336 heads on a standardized prompt set.
- We didn't measure the *magnitude* of each head's contribution (it's embedded in per_head_out but we only looked at top-token rankings). A head whose actual writes are `'}}_{\\'` at every position might still have tiny magnitude — in which case it's not really doing anything, and the tokens we're seeing are just noise floor.
- The "potential" view assumes attention = 1.0 on a single position. Actual heads often attend over several positions with weights that produce constructive/destructive interference. A richer analysis would compute the *cumulative* actual write per query position (which is what the `actual` view does) and compare that to a few *counterfactual* "if you attended to position X" views to see how much the attention pattern matters.
- Detection of attention sinks deserves its own primitive (see mechbench-5pu epic).

## Verdict

The framework now has two complementary views of attention-head behavior: the static weight-level view (step_26) and the dynamic activation-level view (this step). Together they reveal the big gap between capability and use — most heads with clean static structure are attention sinks or noisy detectors in actual forward passes.

This is the kind of finding that the static-only view would systematically miss. It's exactly the kind of observation a mechbench product should surface as a default diagnostic: any time a user is excited about a head's weight-level concept, the GUI should offer a one-click check of whether that concept is active during inference, at which positions, and how consistently.
