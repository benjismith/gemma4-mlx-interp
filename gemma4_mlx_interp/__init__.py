"""gemma4_mlx_interp — a mechanistic-interpretability framework for
Google's Gemma 4 E4B running locally on Apple Silicon via MLX.

Quick start:

    from gemma4_mlx_interp import Model

    model = Model.load()
    ids = model.tokenize("Complete this sentence with one word: The Eiffel Tower is in")

    # Forward pass, no instrumentation:
    result = model.run(ids)
    for tok, p in result.top_k(model.tokenizer, k=5):
        print(f'{tok!r:20s} p={p:.4f}')

Capture activations:

    result = model.run(ids, capture=['blocks.23.attn.weights',
                                      'blocks.14.mlp_out'])
    weights = result.cache['blocks.23.attn.weights']  # [1, 8, S, S], bf16
    mlp = result.cache['blocks.14.mlp_out']           # [1, S, 2560], bf16

Modify activations with a hook:

    import mlx.core as mx

    def zero_layer_14_mlp(act, info):
        return mx.zeros_like(act)

    result = model.run(ids, hooks={'blocks.14.mlp_out': zero_layer_14_mlp})

Declarative interventions (Ablate / Capture / Patch), prompt tooling,
logit-lens + geometry helpers, and matplotlib plot helpers are all
re-exported from this module — see README.md for the full API tour.

The full list of hook points is at gemma4_mlx_interp.all_hook_names().
"""

from ._arch import (
    DEFAULT_MODEL_ID,
    D_MODEL,
    GLOBAL_LAYERS,
    LAYER_HOOK_POINTS,
    N_HEADS,
    N_LAYERS,
    VOCAB_SIZE,
    all_hook_names,
    layer_type,
)
from .cache import ActivationCache
from .errors import (
    CacheKeyError,
    InterpError,
    InvalidHookName,
    LayerIndexOutOfRange,
)
from .geometry import (
    centroid_decode,
    cluster_purity,
    cohesion,
    cosine_matrix,
    fact_vectors,
    fact_vectors_at,
    intra_inter_separation,
    nearest_neighbor_purity,
    silhouette_cosine,
)
from .hooks import HookFn, HookInfo, parse_hook_name
from .interventions import Ablate, Capture, Intervention, Patch, compose
from .lens import logit_lens_final, logit_lens_per_position
from .model import Model, RunResult
from .plot import (
    bar_by_layer,
    lens_trajectory,
    logprob_trajectory,
    pca_scatter,
    position_heatmap,
    similarity_heatmap,
)
from .prompts import (
    Prompt,
    PromptSet,
    ValidatedPrompt,
    ValidatedPromptSet,
)

__version__ = "0.4.0"

__all__ = [
    # Main API
    "Model",
    "RunResult",
    "ActivationCache",
    # Declarative interventions
    "Ablate",
    "Capture",
    "Patch",
    "Intervention",
    "compose",
    # Prompts (specific prompt collections live in experiments.prompts)
    "Prompt",
    "PromptSet",
    "ValidatedPrompt",
    "ValidatedPromptSet",
    # Logit lens
    "logit_lens_final",
    "logit_lens_per_position",
    # Fact vectors + geometry
    "fact_vectors",
    "fact_vectors_at",
    "centroid_decode",
    "cohesion",
    "cosine_matrix",
    "intra_inter_separation",
    "cluster_purity",
    "silhouette_cosine",
    "nearest_neighbor_purity",
    # Plot helpers
    "bar_by_layer",
    "lens_trajectory",
    "logprob_trajectory",
    "position_heatmap",
    "pca_scatter",
    "similarity_heatmap",
    # Hook types (for users writing raw callbacks)
    "HookInfo",
    "HookFn",
    "parse_hook_name",
    # Architecture facts
    "DEFAULT_MODEL_ID",
    "N_LAYERS",
    "D_MODEL",
    "N_HEADS",
    "VOCAB_SIZE",
    "GLOBAL_LAYERS",
    "LAYER_HOOK_POINTS",
    "layer_type",
    "all_hook_names",
    # Errors
    "InterpError",
    "InvalidHookName",
    "LayerIndexOutOfRange",
    "CacheKeyError",
]
