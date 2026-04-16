# gemma4-mlx-interp

Mechanistic interpretability experiments on Google's Gemma 4 models, running locally on Apple Silicon via MLX. A weekend-curiosity project, not a research program — the goal is learning, producing a few concrete findings worth writing up, and having fun poking at a new architecture.

## Environment

- **Python**: 3.11, in a venv at `./.venv`. Always activate before running anything (`source .venv/bin/activate`).
- **Key packages**: `mlx`, `mlx-lm`, `mlx-vlm` (currently 0.4.4), `numpy`, `transformers`. Don't install into system Python; it has 2023-era packages that will conflict.
- **Model**: `mlx-community/gemma-4-E4B-it-bf16`, downloaded to the HF cache. This is the instruction-tuned 4B variant in bfloat16 — roughly 8 GB on disk, ~16 GB peak unified memory during inference. Do NOT switch to 4-bit or 8-bit quantized variants without explicit discussion; quantization distorts the activations we're trying to study.
- **Hardware budget**: ~16 GB of unified memory headroom after the model loads. Don't get cavalier with batching — a batch of 8 prompts at 512 tokens each will start to bite. When in doubt, run sequentially.

## Architecture facts about Gemma 4 E4B (confirmed by model structure dump, keep handy)

- 42 transformer layers, `d_model = 2560`, MLP hidden 10240, vocab 262144.
- **Hybrid attention pattern**: layers 5, 11, 17, 23, 29, 35, 41 are *global* attention (q_proj output 4096, head_dim 512, `ProportionalRoPE`). The other 35 layers are *local sliding-window* (q_proj output 2048, head_dim 256, standard `RoPE`). The final layer is always global. That's 7 global layers out of 42, exactly every 6th, which is a non-obvious design choice worth investigating in its own right.
- **Unembed is tied to `embed_tokens`** — there is no separate `lm_head` module. For logit lens, project through `model.language_model.model.embed_tokens.as_linear(x)` or equivalent.
- **MatFormer per-layer embedding side-channel**: every decoder block has `per_layer_input_gate` and `per_layer_projection` modules, and the top-level model has a giant `embed_tokens_per_layer(262144, 10752)` table. This is a real side-input into every block beyond the residual stream, computed via `get_per_layer_inputs(input_ids)` in `mlx_vlm/models/gemma4/gemma4.py`. **It's load-bearing**: calling the language model without populating this path produces coherent-shaped garbage, not NaNs. The hook harness must route through whatever entry point computes per-layer inputs correctly.
- **`v_norm` is `RMSNormNoScale`** on every attention module — unusual normalization choice, may or may not matter for interp, flag if it comes up.

## Project status

**Foundational work is done.** The forward-path bug noted in earlier versions of this file (`model(input_ids)` returning garbage) is resolved — see "The framework" section below; `Model.run` is the working forward path, packaged for hook-based interp.

**Findings landed** in `docs/findings/step_NN_*.md`:

1. Logit lens phase transition (sharp crash from rank ~100k+ to 0 in layers 27–36)
2. Layer-ablation: layers 10–24 are the "invisible middle" — most causally important, least visible to the lens
3. **MatFormer side-channel ablation: load-bearing, concentrated at global layers** — the headline finding
4. Sub-layer ablation: MLPs dominate; only L23 is attention-critical
5. Attention patterns: globals attend to chat-template structure, not content
6. Per-head: L29 H7 has highest subject-attention; M07 then showed it's still expendable
7. Single-head ablation: no single head is a bottleneck
8. Position-wise lens: answer never decodable at the subject position
9. Causal tracing: two clean hotspots (subject pos early, final pos late)
10–13. Fact-vector geometry, centroid decoding, big sweep, stress tests — culminating in the centroid-decoding technique

**Next direction** is in `docs/proposals/factorization-experiments.md`: deflate the "multilingual cognition" framing as an embedding-space artifact, test the surviving operation-factorization claim with (a) operation-word disambiguation prompts and (b) representation-injection (steering) experiments.

**Out of scope for now**:

- The 26B-A4B MoE variant (interesting but MoE adds complications we don't need yet).
- The 31B dense model (won't fit on this hardware at bf16).
- Vision or audio encoder interp — text-only.
- Multi-architecture support in the framework. The package is Gemma-4-E4B-specific by design; if/when we have a second model worth probing, we generalize. Until then, keeping the constants hard-coded keeps the code honest.

## The framework: `gemma4_mlx_interp/`

The package at `gemma4_mlx_interp/` is the canonical interface to the model. **Every experiment script imports from it; nothing in the project should call `model.language_model(input_ids)` directly anymore.** It was extracted from the prototype `forward.py` + `hooks.py` modules during the migration epic (closed: `qbf` + 17 children).

Public surface (everything below is re-exported from the top-level package):

- **Forward + hooks:** `Model.load()`, `Model.run(input_ids, hooks={}, capture=[])`, `ActivationCache`. The hook-aware forward pass with 294 named hook points (`blocks.{i}.resid_pre/mid/post/attn_out/mlp_out/gate_out/attn.weights/attn.per_head_out`). TransformerLens-style callback contract.
- **Declarative interventions:** `Ablate` / `Capture` / `Patch`, plus `compose`. Pass a list as `interventions=[...]` to `Model.run`. Multiple interventions on the same hook point chain in declaration order.
- **Prompt tooling:** `Prompt` / `PromptSet` / `validate`, the canonical prompt sets (`FACTUAL_15`, `BIG_SWEEP_96`, `STRESS_TEMPLATE_VAR/CROSS_LINGUAL/CREATIVE`).
- **Analysis helpers:** `logit_lens_final` / `logit_lens_per_position`, `fact_vectors` / `fact_vectors_at`, `centroid_decode`, plus `cosine_matrix` / `cluster_purity` / `silhouette_cosine` / `nearest_neighbor_purity` / `intra_inter_separation`.
- **Plot helpers:** `bar_by_layer`, `lens_trajectory`, `logprob_trajectory`, `position_heatmap`, `pca_scatter`, `similarity_heatmap`. Conventions baked in; every plot can still be hand-rolled.

Quickstart:

```python
from gemma4_mlx_interp import Model, Ablate, Capture

model = Model.load()
ids = model.tokenize("Complete this sentence with one word: The Eiffel Tower is in")

result = model.run(ids)                                    # bare run
result = model.run(ids, interventions=[Ablate.layer(14)])  # ablation
result = model.run(ids, interventions=[                    # capture + ablation
    Ablate.head(29, head=7),
    Capture.attn_weights(layers=[23, 29]),
])
```

See `gemma4_mlx_interp/README.md` for the full API tour and worked examples.

Smoke tests live next to the package: `python -m gemma4_mlx_interp._smoke` (forward path), `_smoke_interventions` (composition), `_smoke_analysis` (reproduces published findings 01/11/12 numbers), `_smoke_plots` (plot helpers vs synthetic data).

## Code style and conventions

- **Save activation caches to disk** (`.npz` or `.safetensors`) when an experiment runs more than a few seconds of forward passes. Recomputing E4B activations is cheap-ish but not free, and being able to re-analyze without re-running is worth the disk space.
- **bf16 throughout the cache, float32 only at the analysis boundary.** Cast with `.astype(mx.float32)` right before going to numpy — or use `cache.to_float32()`. MLX → numpy conversions on bf16 arrays will crash with a PEP 3118 buffer format error; this is a known footgun.
- **Prefer reading mlx-vlm's source over guessing at its API.** The package is small, the Gemma 4 model file is a few hundred lines of readable Python, and the upstream docs are sparse. When in doubt, view the file.
- **`mx.eval()` before reading values.** MLX is lazy. The framework's `Model.run` evals the cache + logits in a single batch before returning, so users typically don't need to think about this — but if you build your own forward path, remember it.
- **No `localStorage`, no browser APIs, no web frontends** — this is a CLI/notebook project. Any visualization is matplotlib, Plotly, or (at most) writing an HTML artifact opened manually.

## Debugging principle

When something isn't working, **read the source of whatever is working first before theorizing.** If there's a working path and a broken path, diff them at the source level rather than guessing at causes.

## Files and directories

```
gemma4-mlx-interp/
├── .venv/                     # Python 3.11 venv (don't commit)
├── CLAUDE.md                  # This file
├── benchmark.py               # Latency benchmarks for Model.run + capture configs
├── gemma4_mlx_interp/         # The framework
│   ├── __init__.py            # Public API re-exports
│   ├── _arch.py               # E4B architectural constants + hook registry
│   ├── _forward.py            # Canonical hook-aware forward pass (THE forward)
│   ├── model.py               # Model.load / Model.run / RunResult
│   ├── cache.py               # ActivationCache
│   ├── hooks.py               # HookInfo + name parser
│   ├── interventions.py       # Ablate / Capture / Patch / compose
│   ├── lens.py                # logit_lens_final / logit_lens_per_position
│   ├── geometry.py            # fact_vectors / centroid_decode / stats
│   ├── plot.py                # bar_by_layer / lens_trajectory / heatmaps / pca
│   ├── prompts/               # Prompt + PromptSet + canonical sets
│   │   ├── _core.py
│   │   ├── factual.py         # FACTUAL_15
│   │   ├── big_sweep.py       # BIG_SWEEP_96 (12 categories)
│   │   └── stress.py          # STRESS_TEMPLATE_VAR / CROSS_LINGUAL / CREATIVE
│   ├── errors.py              # InvalidHookName / CacheKeyError / etc.
│   ├── _smoke.py              # Forward-path smoke test
│   ├── _smoke_interventions.py  # Composition smoke test
│   ├── _smoke_analysis.py     # Reproduces findings 01/11/12 numbers
│   ├── _smoke_plots.py        # Plot helpers vs synthetic data
│   └── README.md              # User-facing framework docs
├── experiments/               # Numbered scripts using the framework
│   ├── step_01_logit_lens_batch.py
│   ├── step_02_layer_ablation.py
│   └── ...                    # through step_13_stress_tests.py
├── docs/
│   ├── findings/              # step_NN_*.md write-ups (one per experiment)
│   ├── essays/                # Long-form narratives
│   └── proposals/             # Next-direction experiment designs
├── caches/                    # Saved activation caches + plots (gitignored)
└── notes/                     # Scratch observations (gitignored)
```

## Useful commands

```bash
# Activate the venv every session
source .venv/bin/activate

# Sanity-check the framework loads + answers a basic prompt
python -m gemma4_mlx_interp._smoke

# Reproduce any experiment's published findings
python experiments/step_02_layer_ablation.py
python experiments/step_12_big_sweep.py

# Find mlx-vlm source for reading (still useful when something surprises you)
python -c "import mlx_vlm, os; print(os.path.dirname(mlx_vlm.__file__))"
```


<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
