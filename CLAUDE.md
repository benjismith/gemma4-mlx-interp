# mechbench-experiments (pre-rename: gemma4-mlx-interp)

Mechanistic interpretability experiments on Google's Gemma 4 models, running locally on Apple Silicon via MLX. A weekend-curiosity project that grew into the first chapter of the [mechbench](https://github.com/mechbench/mechbench) project family — the research-script + findings repo that consumes `mechbench-core` as its compute engine.

**This repo is scheduled to be renamed to `mechbench-experiments`.** Treat the existing `gemma4-mlx-interp` name as provisional; new documentation and commit messages should reference the destination name.

## The mechbench family

This repo is one of eight in the family. See the [meta repo](https://github.com/mechbench/mechbench) for the full map:

- [`mechbench`](https://github.com/mechbench/mechbench) — vision, philosophy, cross-repo task backlog.
- [`mechbench-core`](https://github.com/mechbench/mechbench-core) — the Python compute engine. **A snapshot of its source lives in this repo at `gemma4_mlx_interp/`** (see below).
- [`mechbench-schema`](https://github.com/mechbench/mechbench-schema) — typed emission contract (Pydantic + generated TS).
- [`mechbench-ui`](https://github.com/mechbench/mechbench-ui), [`mechbench-agent`](https://github.com/mechbench/mechbench-agent), [`mechbench-remote`](https://github.com/mechbench/mechbench-remote), [`mechbench-memo`](https://github.com/mechbench/mechbench-memo), [`mechbench-skills`](https://github.com/mechbench/mechbench-skills) — scoped but not yet populated.

If a user asks for work that belongs in one of those repos, push back. Research scripts, findings, essays, and prompt collections belong here; framework-level primitives belong in `mechbench-core`.

## The framework snapshot at `gemma4_mlx_interp/`

The package at `gemma4_mlx_interp/` is a **local snapshot** of the framework code that has since been lifted into the standalone [`mechbench-core`](https://github.com/mechbench/mechbench-core) repo (where the package is now called `mechbench_core`). This repo still has 32 experiment scripts that import from `gemma4_mlx_interp`, so the local copy stays for now.

**Migrating the experiments to import from `mechbench-core` is a follow-up task** tracked in the meta repo's `tasks/mechbench-experiments/`. Until that migration lands:

- Framework bugfixes should go into `mechbench-core` first, then optionally be synced here. Don't let the two diverge in behavior without a reason.
- New framework features belong in `mechbench-core`, not here.
- Experiment code should continue to work against the local `gemma4_mlx_interp/` package. When the migration task lands, all `from gemma4_mlx_interp import …` statements get rewritten to `from mechbench_core import …` in one pass.

Public surface of the framework (snapshot; authoritative version is in `mechbench-core`):

- **Forward + hooks:** `Model.load()`, `Model.run(input_ids, hooks={}, capture=[])`, `ActivationCache`. Hook-aware forward pass with 294 named hook points.
- **Declarative interventions:** `Ablate` / `Capture` / `Patch`, plus `compose`. Pass as `interventions=[...]`.
- **Prompt tooling:** `Prompt` / `PromptSet` / `validate`. Project-specific collections live in `experiments/prompts/`.
- **Analysis helpers:** `logit_lens_final` / `logit_lens_per_position`, `fact_vectors` / `centroid_decode`, geometry stats.
- **Plot helpers:** `bar_by_layer`, `lens_trajectory`, heatmaps, `pca_scatter`, etc.

Quickstart:

```python
from gemma4_mlx_interp import Model, Ablate, Capture

model = Model.load()
ids = model.tokenize("Complete this sentence with one word: The Eiffel Tower is in")

result = model.run(ids)
result = model.run(ids, interventions=[Ablate.layer(14)])
```

Smoke tests: `python -m gemma4_mlx_interp._smoke` (forward path), `_smoke_interventions`, `_smoke_plots`. The integration test that reproduces findings 01/11/12 against this repo's prompt collections is in `experiments/smoke_analysis.py`.

## Environment

- **Python**: 3.11, in a venv at `./.venv`. Always activate before running anything (`source .venv/bin/activate`).
- **Key packages**: `mlx`, `mlx-lm`, `mlx-vlm` (currently 0.4.4), `numpy`, `transformers`. Don't install into system Python.
- **Model**: `mlx-community/gemma-4-E4B-it-bf16`, downloaded to the HF cache (~8 GB on disk, ~16 GB peak unified memory during inference). Also supports the E2B variant. Do NOT switch to 4-bit or 8-bit quantized variants without explicit discussion — quantization distorts the activations we're studying.
- **Hardware budget**: ~16 GB of unified memory headroom after the model loads.

## Architecture facts about Gemma 4 E4B (keep handy)

- 42 transformer layers, `d_model = 2560`, MLP hidden 10240, vocab 262144.
- **Hybrid attention pattern**: layers 5, 11, 17, 23, 29, 35, 41 are *global* attention (q_proj output 4096, head_dim 512, `ProportionalRoPE`). The other 35 layers are *local sliding-window* (q_proj output 2048, head_dim 256, standard `RoPE`). 7 global layers out of 42 — exactly every 6th — a non-obvious design choice.
- **Unembed is tied to `embed_tokens`** — no separate `lm_head`. For logit lens, project through `model.language_model.model.embed_tokens.as_linear(x)`.
- **MatFormer per-layer embedding side-channel**: every decoder block has `per_layer_input_gate` and `per_layer_projection`; the top-level model has a giant `embed_tokens_per_layer(262144, 10752)` table. Calling the language model without populating this path produces coherent-shaped garbage, not NaNs. The hook harness must route through the correct per-layer-input entry point.
- **`v_norm` is `RMSNormNoScale`** on every attention module — unusual normalization.

E2B replicates the pattern with different constants (30 layers, global every 5th). See `_arch.py` for the adapter.

## Findings landed

All in `docs/findings/step_NN_*.md`:

1. Logit lens phase transition (sharp crash from rank ~100k+ to 0 in layers 27–36).
2. Layer-ablation: layers 10–24 are the "invisible middle" — most causally important, least visible to the lens.
3. **MatFormer side-channel ablation: load-bearing, concentrated at global layers** — the headline finding.
4. Sub-layer ablation: MLPs dominate; only L23 is attention-critical.
5. Attention patterns: globals attend to chat-template structure, not content.
6. Per-head: L29 H7 has highest subject-attention; still expendable (see step 7).
7. Single-head ablation: no single head is a bottleneck.
8. Position-wise lens: answer never decodable at the subject position.
9. Causal tracing: two clean hotspots (subject pos early, final pos late).
10–13. Fact-vector geometry, centroid decoding, big sweep, stress tests.
14+. Homonym disambiguation, emotion probes, perplexity probe replication on E4B and E2B, L23 architectural-pivot essay.

Long-form narrative in `docs/essays/experiment-narrative.md`.

## Task tracking

**Tasks for this repo live in the meta repo** at [`mechbench/tasks/mechbench-experiments/`](https://github.com/mechbench/mechbench/tree/main/tasks/mechbench-experiments), not here. The centralization is deliberate — it lets `depends_on:` references resolve across the family via `grep`.

When a PR in this repo closes a task:

1. PR description references the task id (e.g., "Closes 000119").
2. In the meta repo, `git mv tasks/mechbench-experiments/open/<id>-*.md tasks/mechbench-experiments/done/` in a concurrent commit.

This repo previously used [beads](https://github.com/steveyegge/beads) for task tracking. Beads has been removed; the 138 existing tasks were migrated to file-based markdown in the meta repo. Do not reinstall `bd`, do not use `TodoWrite`, `TaskCreate`, or markdown TODO lists. File-based tasks in the meta repo are the only source of truth.

## Code style and conventions

- **Save activation caches to disk** (`.npz` or `.safetensors`) when an experiment runs more than a few seconds of forward passes.
- **bf16 throughout the cache, float32 only at the analysis boundary.** Cast with `.astype(mx.float32)` before going to numpy — or use `cache.to_float32()`. MLX → numpy conversions on bf16 arrays will crash with a PEP 3118 buffer format error.
- **Prefer reading mlx-vlm's source over guessing at its API.** The Gemma 4 model file is a few hundred lines of readable Python.
- **`mx.eval()` before reading values.** MLX is lazy. `Model.run` handles this internally; if you build your own forward path, remember it.
- **No comments by default.** Only when the WHY is non-obvious.
- **No speculative abstractions.** Build for the second consumer, not the hypothetical tenth.

## Debugging principle

When something isn't working, **read the source of whatever is working first before theorizing.** If there's a working path and a broken path, diff them at the source level rather than guessing.

## Files and directories

```
gemma4-mlx-interp/              # (will become mechbench-experiments)
├── .venv/                      # Python 3.11 venv (gitignored)
├── .claude/                    # editor/agent settings
├── CLAUDE.md                   # This file
├── benchmark.py                # Latency benchmarks for Model.run + capture configs
├── gemma4_mlx_interp/          # Local snapshot of mechbench-core (migration pending)
│   ├── __init__.py
│   ├── _arch.py                # E4B + E2B architectural constants
│   ├── _forward.py             # Canonical hook-aware forward pass
│   ├── model.py                # Model.load / Model.run / RunResult
│   ├── cache.py                # ActivationCache
│   ├── hooks.py                # HookInfo + name parser
│   ├── interventions.py        # Ablate / Capture / Patch / compose
│   ├── lens.py                 # logit_lens_final / logit_lens_per_position
│   ├── geometry.py             # fact_vectors / centroid_decode / stats
│   ├── probes.py               # Probe primitive
│   ├── head_weights.py         # Static W_Q/W_K/W_V/W_O analysis
│   ├── generate.py             # Text generation helpers
│   ├── plot.py                 # Matplotlib helpers with project conventions
│   ├── prompts.py              # Prompt + PromptSet
│   ├── errors.py               # Framework errors
│   ├── _smoke.py               # Forward-path smoke test
│   ├── _smoke_e2b.py           # E2B variant smoke test
│   ├── _smoke_interventions.py # Composition smoke test
│   ├── _smoke_plots.py         # Plot helpers vs synthetic data
│   └── README.md               # User-facing framework docs (snapshot)
├── experiments/                # Numbered scripts + project-specific data
│   ├── prompts/                # Project-specific prompt collections
│   │   ├── __init__.py
│   │   ├── factual.py          # FACTUAL_15
│   │   ├── big_sweep.py        # BIG_SWEEP_96 (12 categories)
│   │   └── stress.py           # STRESS_TEMPLATE_VAR / CROSS_LINGUAL / CREATIVE
│   ├── smoke_analysis.py       # Integration test: framework + prompts -> findings
│   ├── step_01_logit_lens_batch.py
│   ├── step_02_layer_ablation.py
│   └── ...                     # through step_31_e2b_perplexity_probe.py
├── docs/
│   ├── findings/               # step_NN_*.md write-ups (one per experiment)
│   ├── essays/                 # Long-form narratives (experiment-narrative.md)
│   └── proposals/              # Next-direction experiment designs
├── caches/                     # Saved activation caches + plots (gitignored)
└── notes/                      # Scratch observations (gitignored)
```

## Useful commands

```bash
source .venv/bin/activate

python -m gemma4_mlx_interp._smoke                 # smoke-test the framework
python experiments/step_02_layer_ablation.py       # reproduce an experiment
python -c "import mlx_vlm, os; print(os.path.dirname(mlx_vlm.__file__))"
```

## Session close

Commit with specific file paths (not `-A`). Push before ending the session. No beads commands, no Dolt push — those tools have been removed.
