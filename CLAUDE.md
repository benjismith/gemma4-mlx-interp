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

## Known open bug (unblock this first)

As of the state of this project's creation: calling `model(input_ids)` or `model.language_model(input_ids)` directly on the loaded E4B model produces garbage output (top token `'**'` at p=0.31 for "The Eiffel Tower is in", regardless of chat templating or bare input). **However, `mlx_vlm.generate(model, processor, prompt, ...)` produces correct output ("Paris") on the same model object in the same process.** So the weights load correctly and the forward path works *when called through `generate`*; something in the direct-call path is missing a setup step that `generate` performs.

The MatFormer per-layer-input machinery is NOT the culprit — a config probe confirmed `hidden_size_per_layer_input=256` is set and `get_per_layer_inputs` exists, and `Model.__call__` in `gemma4.py` routes through `get_input_embeddings` which computes per-layer inputs correctly. The bug is something else: possibly a cache object that `generate` initializes that the layer code reads as non-optional state, possibly processor-side input preparation, possibly something in `language.py`'s `LanguageModel.__call__` signature mismatch.

**First task in any new session on this project**: read `mlx_vlm/generate.py`, `mlx_vlm/utils.py`, and `mlx_vlm/models/gemma4/{gemma4,language}.py` in order. Trace a single forward pass through `mlx_vlm.generate` and identify exactly what `generate` feeds into the model that a direct `model(x)` call doesn't. Then write a minimal reproduction of the working forward path that we control end-to-end. Don't write hooks until the basic forward path is understood and reproducible.

## Project goals

**Short-term** (get this working):

1. Resolve the forward-path bug above. Produce a minimal `forward.py` that runs a prompt through E4B and returns logits, matching what `mlx_vlm.generate` produces.
2. Build a TransformerLens-style hook harness (`hooks.py`) that caches, per layer: `resid_pre`, `attn_out`, `mlp_out`, `resid_post`, and the `per_layer_input_gate` output. Cache should stay in bf16 and only cast to float32 at analysis time.
3. Implement a logit lens as the first experiment — project every layer's `resid_post` through the tied unembed and visualize the trajectory of a target token's rank/probability across depth. Classic sanity check; if it looks nothing like published logit lens results on other models, something is still wrong with the harness.

**Medium-term** (things actually worth looking at):

- **Global vs. local layer specialization.** Do the 7 global-attention layers (5, 11, 17, 23, 29, 35, 41) do qualitatively different computational work from the 35 local-attention layers? Candidate probes: compare attention-pattern entropy, zero-ablate each layer individually and measure loss impact on a corpus of prompts, logit-lens the residual stream *into* and *out of* each global layer specifically.
- **Is the per-layer-input side-channel doing real work?** Ablate `per_layer_input_gate` (set the gate output to zero in every block) and measure degradation on a held-out corpus. If degradation is small, the MatFormer side-channel may be closer to vestigial than Google's paper suggests. If it's large, the next question is *what kind of information* flows through it — is it per-token content, positional, or something else?

Both of the above are writeup-shaped if they yield clean results. Neither has been done publicly as of the project's creation, as far as we know.

**Out of scope for now**:

- The 26B-A4B MoE variant (interesting but MoE adds complications we don't need yet).
- The 31B dense model (won't fit on this hardware at bf16).
- Vision or audio encoder interp — text-only.
- Writing a general-purpose interp library. This is a scrappy project harness, not `TransformerLens-MLX`. Keep the abstraction surface thin and the code honest.

## Code style and conventions

- **Small, composable scripts over frameworks.** `forward.py`, `hooks.py`, numbered `experiments/step_NN_*.py` files, etc. Not a monorepo with a setup.py.
- **Save activation caches to disk** (`.npz` or `.safetensors`) when an experiment runs more than a few seconds of forward passes. Recomputing E4B activations is cheap-ish but not free, and being able to re-analyze without re-running is worth the disk space.
- **bf16 throughout the cache, float32 only at the analysis boundary.** Cast with `.astype(mx.float32)` right before going to numpy. MLX → numpy conversions on bf16 arrays will crash with a PEP 3118 buffer format error; this is a known footgun.
- **Prefer reading mlx-vlm's source over guessing at its API.** The package is small, the Gemma 4 model file is a few hundred lines of readable Python, and the upstream docs are sparse. When in doubt, `view` the file.
- **`mx.eval()` before reading values.** MLX is lazy; caching `x` in a dict without calling `mx.eval()` stores a graph node, not a value. The harness should eval the full cache after each forward pass, not on every individual insertion.
- **No `localStorage`, no browser APIs, no web frontends** — this is a CLI/notebook project. Any visualization is matplotlib, Plotly, or (at most) writing an HTML artifact opened manually.

## Debugging principle

When something isn't working, **read the source of whatever is working first before theorizing.** If there's a working path and a broken path, diff them at the source level rather than guessing at causes.

## Files and directories (as they develop)

```
gemma4-mlx-interp/
├── .venv/              # Python 3.11 venv (don't commit)
├── CLAUDE.md           # This file
├── README.md           # Human-facing project description (write after first results)
├── forward.py          # Minimal working forward pass, used by everything else
├── hooks.py            # Activation cache + hook harness
├── experiments/
│   ├── logit_lens.py
│   ├── layer_ablation.py
│   └── ...
├── caches/             # Saved activation caches (gitignore)
└── notes/              # Scratch observations, half-baked hypotheses, TODOs
```

## Useful commands

```bash
# Activate the venv every session
source .venv/bin/activate

# Sanity-check that mlx and the model load
python -c "from mlx_vlm import load; m, p = load('mlx-community/gemma-4-E4B-it-bf16'); print('ok')"

# Find mlx-vlm source for reading
python -c "import mlx_vlm, os; print(os.path.dirname(mlx_vlm.__file__))"

# Run a known-working generation as ground truth for any forward-path work
python -c "
from mlx_vlm import load, generate
m, p = load('mlx-community/gemma-4-E4B-it-bf16')
print(generate(model=m, processor=p,
               prompt='Complete this sentence with one word: The Eiffel Tower is in',
               max_tokens=5, temperature=0.0).text)
"
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
