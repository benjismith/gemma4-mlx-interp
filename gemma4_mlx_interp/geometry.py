"""Fact-vector extraction + geometric analyses.

Three layers of functionality:

1. Extraction: fact_vectors / fact_vectors_at run prompts through the model
   and pull the residual stream at chosen positions and layers.

2. Decoding: centroid_decode projects the (mean-subtracted) centroid of a
   group of vectors through the model's tied unembed to read off the
   relational concept that group represents (the technique from finding 11/12).

3. Pure-numpy stats: cosine_matrix, cluster_purity, silhouette_cosine,
   nearest_neighbor_purity, intra_inter_separation. No model dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import mlx.core as mx
import numpy as np

from ._arch import D_MODEL
from .interventions import Capture


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _find_subject_position(token_labels: list[str], subject: str) -> int:
    """Last token whose decoded label contains the subject substring,
    or whose stripped form is itself contained in the subject (handles cases
    like German genitive 'Frankreichs' which tokenizes as ' Frankreich' + 's').
    """
    s = subject.lower()
    # Pass 1: exact substring match (subject is in token).
    for i in range(len(token_labels) - 1, -1, -1):
        if s in token_labels[i].lower():
            return i
    # Pass 2: reverse — non-trivial token stripped is contained in subject.
    for i in range(len(token_labels) - 1, -1, -1):
        t = token_labels[i].strip().lower()
        if len(t) >= 3 and t in s:
            return i
    raise ValueError(
        f"Subject substring {subject!r} not found in any token. "
        f"Tokens were: {token_labels}"
    )


def _resolve_position(
    model, validated_prompt, position: str
) -> int:
    """Resolve a position spec to a concrete index for one prompt."""
    if position == "final":
        return int(validated_prompt.input_ids.shape[1]) - 1
    if position == "subject":
        subj = validated_prompt.prompt.subject
        if subj is None:
            raise ValueError(
                f"position='subject' but Prompt.subject is None for: "
                f"{validated_prompt.prompt.text!r}"
            )
        token_labels = [
            model.tokenizer.decode([int(t)])
            for t in validated_prompt.input_ids[0]
        ]
        return _find_subject_position(token_labels, subj)
    raise ValueError(
        f"position must be 'subject' or 'final', got {position!r}"
    )


def fact_vectors(
    model,
    validated,
    layer: int,
    *,
    position: str = "subject",
) -> np.ndarray:
    """Extract residual-stream vectors at a chosen position for each prompt.

    Args:
        model: Model instance.
        validated: ValidatedPromptSet.
        layer: Layer index whose resid_post to extract.
        position: 'subject' (use Prompt.subject substring) or 'final'.

    Returns:
        np.ndarray of shape [n_prompts, D_MODEL], dtype float32.
    """
    n = len(validated)
    out = np.zeros((n, D_MODEL), dtype=np.float32)
    cap = Capture.residual(layer, point="post")
    key = f"blocks.{layer}.resid_post"
    for j, vp in enumerate(validated):
        result = model.run(vp.input_ids, interventions=[cap])
        pos = _resolve_position(model, vp, position)
        v = result.cache[key][0, pos, :].astype(mx.float32)
        mx.eval(v)
        out[j] = np.array(v)
    return out


def fact_vectors_at(
    model,
    validated,
    layers: Iterable[int],
    *,
    position: str = "subject",
) -> dict[int, np.ndarray]:
    """Multi-layer fact-vector extraction in one model pass per prompt.

    Strictly more efficient than calling fact_vectors() once per layer: each
    prompt is run through the model exactly once, with all requested layers
    captured at once. Use when you need vectors at several depths.

    Args:
        model: Model instance.
        validated: ValidatedPromptSet.
        layers: Iterable of layer indices.
        position: 'subject' or 'final'.

    Returns:
        dict {layer: np.ndarray[n_prompts, D_MODEL]} in float32.
    """
    layers_list = list(layers)
    n = len(validated)
    out = {L: np.zeros((n, D_MODEL), dtype=np.float32) for L in layers_list}
    cap = Capture.residual(layers_list, point="post")
    for j, vp in enumerate(validated):
        result = model.run(vp.input_ids, interventions=[cap])
        pos = _resolve_position(model, vp, position)
        for L in layers_list:
            v = result.cache[f"blocks.{L}.resid_post"][0, pos, :].astype(mx.float32)
            mx.eval(v)
            out[L][j] = np.array(v)
    return out


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def centroid_decode(
    model,
    vectors: np.ndarray,
    *,
    k: int = 10,
    mean_subtract: bool = True,
    overall_mean: Optional[np.ndarray] = None,
) -> list[tuple[str, float]]:
    """Project the (mean-subtracted) centroid of `vectors` through the tied
    unembed, return top-k decoded tokens.

    The technique from finding 11/12: individual fact vectors at subject
    positions are vocab-opaque, but their averaged centroid (after removing
    the prompt-template common-mode via mean-subtraction) decodes to a
    multilingual cluster of tokens that names the relational frame.

    Args:
        model: Model instance.
        vectors: np.ndarray[n, D_MODEL] in float32. The vectors to centroid.
        k: How many top tokens to return.
        mean_subtract: If True, subtract `overall_mean` (or vectors.mean if
            overall_mean is None) from the centroid before projecting.
        overall_mean: Optional baseline mean to subtract. Pass the
            corpus-wide mean (over all categories) when decoding individual
            categories — using only this category's mean would zero out the
            very signal you're trying to decode. If None, defaults to
            vectors.mean(), which is correct only if `vectors` is the whole
            corpus.

    Returns:
        list of (decoded_token_string, probability) sorted by descending p.
    """
    centroid = vectors.mean(axis=0)
    if mean_subtract:
        if overall_mean is None:
            overall_mean = centroid  # degenerate, but documented above
        centroid = centroid - overall_mean

    v = mx.array(centroid[None, None, :], dtype=mx.bfloat16)
    logits = model.project_to_logits(v)
    last = logits[0, 0, :].astype(mx.float32)
    probs = mx.softmax(last)
    mx.eval(probs)
    probs_np = np.array(probs)
    top_idx = np.argsort(-probs_np)[:k]
    return [
        (model.tokenizer.decode([int(i)]), float(probs_np[int(i)]))
        for i in top_idx
    ]


# ---------------------------------------------------------------------------
# Pure-numpy stats
# ---------------------------------------------------------------------------


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity. Returns [N, N]."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normed = vectors / np.clip(norms, 1e-12, None)
    return normed @ normed.T


def intra_inter_separation(
    vectors: np.ndarray, labels: np.ndarray | list
) -> tuple[float, float, float]:
    """Mean intra-category cosine, mean inter-category cosine, and their
    difference (the 'separation' score). Higher separation = more distinct
    categories.

    Returns (intra_mean, inter_mean, separation).
    """
    labels = np.asarray(labels)
    sim = cosine_matrix(vectors)
    intra, inter = [], []
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            (intra if labels[i] == labels[j] else inter).append(sim[i, j])
    intra_mean = float(np.mean(intra)) if intra else 0.0
    inter_mean = float(np.mean(inter)) if inter else 0.0
    return intra_mean, inter_mean, intra_mean - inter_mean


def cluster_purity(true_labels, pred_labels) -> float:
    """Maximum achievable accuracy after relabeling clusters to match truth.

    For each predicted cluster, we 'relabel' it to whichever true label is
    most common in it; we then count how many points are correctly classified.
    Pure clusters (every member shares a true label) score 1.0; uniformly-
    mixed clusters score chance.
    """
    n = len(true_labels)
    clusters = set(pred_labels)
    correct = 0
    for c in clusters:
        in_cluster = [t for t, p in zip(true_labels, pred_labels) if p == c]
        if in_cluster:
            most_common = max(in_cluster.count(label) for label in set(in_cluster))
            correct += most_common
    return correct / n if n else 0.0


def silhouette_cosine(vectors: np.ndarray, labels) -> float:
    """Silhouette score (cosine metric) of the labeling. Range [-1, 1];
    higher = better-separated clusters. Wraps sklearn for the heavy lifting."""
    from sklearn.metrics import silhouette_score
    labels_arr = np.asarray(labels)
    # silhouette_score needs integer labels
    unique = list(dict.fromkeys(labels_arr.tolist()))
    int_labels = np.array([unique.index(l) for l in labels_arr])
    return float(silhouette_score(vectors, int_labels, metric="cosine"))


def nearest_neighbor_purity(
    vectors: np.ndarray, labels
) -> tuple[float, np.ndarray]:
    """For each vector, find its cosine-nearest neighbor (excluding self) and
    check whether that neighbor shares its label. Returns (hit_rate, hits)
    where hits[i] is True iff vectors[i]'s nearest neighbor matches.

    This is the cleanest 'is the geometry semantically organized' test —
    chance for a balanced k-class set is roughly 1/k.
    """
    labels_arr = np.asarray(labels)
    sim = cosine_matrix(vectors)
    np.fill_diagonal(sim, -np.inf)
    nn_idx = np.argmax(sim, axis=1)
    hits = labels_arr[nn_idx] == labels_arr
    return float(hits.mean()), hits


def cohesion(vectors: np.ndarray) -> float:
    """Mean cosine similarity between each vector and the cluster centroid.

    Measures how tightly a group of vectors is clustered around its own mean.
    Range [-1, 1]; higher = tighter cluster.

    Distinct from the mean intra-cluster pairwise cosine that you'd compute
    via cosine_matrix + masking: cohesion is member-to-centroid (n similarities)
    rather than member-to-member (n*(n-1)/2). Cohesion is generally >= the
    intra-cluster pairwise mean since the centroid is the maximum-likelihood
    'anchor' that minimizes the angular spread.
    """
    if len(vectors) == 0:
        return 0.0
    centroid = vectors.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm < 1e-12:
        return 0.0
    centroid_unit = centroid / centroid_norm
    member_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    member_units = vectors / np.clip(member_norms, 1e-12, None)
    sims = member_units @ centroid_unit
    return float(sims.mean())


# ---------------------------------------------------------------------------
# Vocabulary-space concentration
#
# Given a residual vector projected through the tied unembed and softmaxed,
# how concentrated is the resulting distribution? "Concentrated" can mean
# several things; these primitives give the standard framings, and
# vocab_concentration returns them all in a single structured result.
#
# The primitives work on ANY probability distribution, not just vocab ones —
# the "vocab" framing is about the typical use case (project a residual,
# softmax, ask how peaked the result is) but the math is generic.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VocabConcentration:
    """Concentration metrics for a probability distribution (typically a
    vocabulary-space distribution produced by projecting a residual vector
    through the tied unembed and softmaxing).

    Attributes:
        top1: probability of the single most-likely token.
        top_k_mass: sum of the top-k probabilities. Use for robustness to
            ties and to capture how 'peaked' the distribution is at the high end.
        k: the k that top_k_mass was computed at (so the result is self-describing).
        entropy_bits: Shannon entropy in bits. Lower = more concentrated.
            Range [0, log2(vocab_size)]; for Gemma 4 E4B's 262,144-token vocab,
            uniform distribution = log2(262144) = 18.0 bits.
        effective_vocab_size: exp(entropy_in_nats). Roughly 'how many tokens
            worth of probability mass is the distribution effectively spread
            over'. A delta gives 1; a uniform distribution gives the vocab size.
    """

    top1: float
    top_k_mass: float
    k: int
    entropy_bits: float
    effective_vocab_size: float


def top_k_mass(probs: np.ndarray, k: int = 10) -> float:
    """Sum of the top-k probabilities in a discrete distribution."""
    if probs.size == 0 or k <= 0:
        return 0.0
    k = min(k, probs.size)
    # np.partition is O(n); avoids a full sort
    return float(np.partition(probs, -k)[-k:].sum())


def entropy_bits(probs: np.ndarray) -> float:
    """Shannon entropy of a probability distribution, in bits.

    Lower = more concentrated. A delta distribution has entropy 0;
    a uniform distribution over N items has entropy log2(N).
    """
    p = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(p * np.log2(p)))


def effective_vocab_size(probs: np.ndarray) -> float:
    """exp(entropy_in_nats). The 'how many tokens worth' the distribution
    is effectively spread over.

    A delta gives 1.0; a uniform over N items gives N. Equivalent to
    perplexity of a model whose distribution this is.
    """
    nats = entropy_bits(probs) * np.log(2)
    return float(np.exp(nats))


def vocab_concentration(probs: np.ndarray, k: int = 10) -> VocabConcentration:
    """Compute the standard concentration metrics for a probability
    distribution in one pass, returning a structured VocabConcentration.

    Use when you want all four metrics together (the common case for the
    "how concentrated is this decoded distribution" question). Use the
    individual primitives (top_k_mass / entropy_bits / effective_vocab_size)
    when you want just one.
    """
    if probs.size == 0:
        return VocabConcentration(
            top1=0.0, top_k_mass=0.0, k=k,
            entropy_bits=0.0, effective_vocab_size=0.0,
        )
    top1 = float(probs.max())
    tk = top_k_mass(probs, k=k)
    h = entropy_bits(probs)
    ev = float(np.exp(h * np.log(2)))
    return VocabConcentration(
        top1=top1, top_k_mass=tk, k=k,
        entropy_bits=h, effective_vocab_size=ev,
    )
