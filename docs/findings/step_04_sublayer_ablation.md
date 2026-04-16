# Sub-Layer Ablation: Attention vs MLP Across Gemma 4 E4B

**Date:** 2026-04-15
**Script:** `experiments/step_04_sublayer_ablation.py`
**Plot:** `caches/sublayer_ablation.png`

## Setup

For each of the 42 layers, we independently ablated the attention branch (zeroed out its residual contribution while still populating KV caches for downstream layers) and the MLP branch (zeroed out its residual contribution). This disentangles the two main computational pathways in each transformer block: attention routes information between positions; MLPs transform information at each position.

Same 15-prompt battery as previous experiments. 42 layers × 15 prompts × 2 branches = 1,260 forward passes (~3.5 minutes on M2 Pro).

## Results

### MLPs dominate overwhelmingly

The headline: **MLP ablation is far more damaging than attention ablation at almost every layer.** The difference is dramatic and consistent:

| Region | Attention ablation (mean Δ log p) | MLP ablation (mean Δ log p) |
|--------|----------------------------------:|----------------------------:|
| All layers | -0.293 | -1.593 |
| Global (n=7) | -0.966 | -1.913 |
| Local (n=35) | -0.180 | -1.523 |
| Middle (10–24) | -0.554 | -2.594 |

Across the critical middle layers (10–24), MLP ablation is almost 5x more damaging than attention ablation. The model's factual recall lives in the MLPs, consistent with findings on GPT-2 and Llama (Meng et al., 2022; Geva et al., 2021).

### The most critical MLPs

The 5 most damaging individual MLP ablations:

| Layer | Type | MLP Δ log p |
|------:|------|------------:|
| 0 | local | -17.58 |
| 14 | local | -9.36 |
| 11 | GLOBAL | -5.67 |
| 12 | local | -3.76 |
| 9 | local | -3.30 |

Layer 0's MLP is catastrophic to remove (Δ = -17.6), which explains why layer 0 dominated the whole-layer ablation in finding 02 — it was almost entirely about the MLP. Layer 14's MLP (Δ = -9.4) is the second most critical, and it barely showed up in attention ablation (Δ = +0.008). This layer is doing essentially pure knowledge retrieval with no important attention contribution.

### Attention only matters at two layers

Attention ablation is near-zero for most layers. Only two layers show substantial attention impact:

| Layer | Type | Attention Δ log p |
|------:|------|------------------:|
| 23 | GLOBAL | -4.96 |
| 0 | local | -3.02 |

Layer 23 is the one layer where attention ablation is *more* damaging than MLP ablation (-4.96 vs -2.31). This is the last global-attention layer before the KV-sharing boundary (layers 24+ share KV caches). It's the model's last opportunity to route information across the full sequence through fresh attention computations — layers 24+ reuse layer 23's (and 22's) KV caches. Removing its attention cuts off the primary long-range information highway.

Layer 17 (another global) shows roughly equal attention and MLP impact (-1.32 vs -1.19), suggesting it serves a dual role.

### The bottom panel tells the story

The difference plot (attention Δ − MLP Δ) is almost entirely positive across layers 0–24 — meaning MLP ablation is worse than attention ablation — then flattens near zero for layers 25–41. The one prominent negative spike is layer 23, where attention dominates. The visual is striking: a wall of blue (MLP-dominant) from layers 0–24, punctuated by a single red spike (attention-dominant) at layer 23.

### Reconciling with previous findings

**Why the logit lens shows nothing in layers 0–24 (finding 01):** The MLPs in these layers are storing and retrieving factual knowledge, but in an internal representation the tied unembed can't decode. The computation is real (ablation proves it) but invisible to the logit lens because the unembed is the wrong projection. These layers are writing to the residual stream in a "language" that only later layers can read.

**Why whole-layer ablation showed layers 10–24 as critical (finding 02):** That result was almost entirely driven by MLP importance. Attention contributes little in those layers — the model is mostly doing position-wise knowledge retrieval, not cross-position information routing.

**Why the side-channel matters most at global layers (finding 03):** The global layers are the ones doing cross-sequence attention, and the side-channel provides per-token identity information to ground that attention. The MLP-heavy local layers don't need the side-channel as much because they're doing position-wise computation where the residual stream already carries sufficient token-level information.

**The special role of layer 23:** It's the convergence point of three findings. It's the most attention-critical layer (this experiment), one of the top-5 most damaging whole-layer ablations (finding 02), and one of the layers most dependent on the side-channel (finding 03). As the last global layer before KV sharing begins, it appears to be the primary bottleneck for long-range information integration.

## The emerging picture of Gemma 4 E4B

Four experiments in, a coherent functional map is forming:

- **Layers 0–9**: Foundation. Layer 0's MLP does essential embedding transformation. Layers 1–9 contribute modestly; ablating any single one barely matters.
- **Layers 10–24**: The engine room. MLPs here store and retrieve factual knowledge. Attention is mostly irrelevant except at global layers 17 and especially 23. The side-channel feeds token identity into the globals. The logit lens can't see any of this because the representations are in an internal format.
- **Layers 25–41**: Readout. The answer becomes visible in the logit lens here (layers 27–36), but ablation says these layers contribute little new computation. They're translating the internal representation built by layers 10–24 into the output vocabulary. The final global layer (41) does surface-form selection.

This is a clean story that synthesizes all four experiments. The work happens in the middle, the readout happens at the end, and the logit lens can only see the readout.
