"""OV-circuit activation trajectories: what each head writes, per position.

Complement to step_26's static weight-level analysis. That found that
specific heads have clean multilingual concept structure in their OV
circuit singular components (L5 h3 rank-1 writes 'nation' across
languages, L23 h7 rank-1 writes 'movement', L40 h6 rank-4 writes
'created'). But those are POTENTIAL patterns under SVD decomposition.
This experiment measures what each head is ACTUALLY writing at each
position during a live forward pass.

Two views per (prompt, layer, head):

  POTENTIAL writes (head_ov_position_writes):
    For each position p, what tokens would this head write if it
    attended fully to position p? Uses V[p] projected through W_O and
    then through the tied unembed. Independent of the actual attention
    pattern.

  ACTUAL writes (head_ov_actual_writes):
    For each query position q, what tokens is this head actually
    writing at q given its attention pattern? Uses per_head_out (the
    weighted softmax(QK) @ V).

The difference between POTENTIAL and ACTUAL reveals how the head's
attention pattern shapes its output. A head whose potential writes are
clean but actual writes are messy is 'capable but quiet'. A head whose
potential writes are messy but actual writes are clean is 'the
attention is doing the interpretation.'

Run from project root:
    python experiments/step_27_ov_trajectories.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gemma4_mlx_interp import (  # noqa: E402
    Model, head_ov_actual_writes, head_ov_position_writes,
)
from gemma4_mlx_interp.head_weights import _embed_matrix_f32  # noqa: E402

OUT_DIR = ROOT / "caches"

# Heads of interest from step_26's findings.
HEADS = [
    (5, 3, "step_26: rank-1 OV wrote 'nation' multilingually"),
    (5, 6, "step_26: rank-0 OV wrote 'model' multilingually"),
    (7, 3, "step_26: rank-0 OV detected 'system' multilingually"),
    (23, 7, "step_26: rank-1 OV wrote 'movement' multilingually"),
    (40, 6, "step_26: rank-4 OV wrote 'created' in 4 European langs"),
]

# Prompts chosen to plausibly exercise the heads above.
PROMPTS = [
    "The French nation celebrates its independence on Bastille Day.",
    "The statistical model captures the structural patterns of the data.",
    "The operating system reboots automatically at midnight.",
    "The civil rights movement transformed American politics.",
    "The committee created a comprehensive new set of criteria.",
]


def _print_table(label: str, writes, k_show=4):
    print(f"\n[{label}]")
    print(f"  {'pos':>3}  {'query_token':>15}  top-{k_show} writes")
    for pw in writes:
        top_str = "  ".join(f"{t!r}" for t, _ in pw.top_tokens[:k_show])
        # Display printable version of the query token
        qt = pw.query_token.replace("\n", "\\n")
        print(f"  {pw.position:>3}  {qt!r:>15}  {top_str}")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading model...")
    model = Model.load()
    print("Precomputing embed matrix...")
    E = _embed_matrix_f32(model)

    for prompt in PROMPTS:
        print(f"\n{'=' * 80}")
        print(f"Prompt: {prompt!r}")
        print(f"{'=' * 80}")
        ids = model.tokenize(prompt)
        n_tokens = int(ids.shape[1])
        print(f"Tokens after chat-template: {n_tokens}")
        # Show the tokens themselves for reference
        decoded = [model.tokenizer.decode([int(ids[0, i])]) for i in range(n_tokens)]
        print("Positions: " + "  ".join(f"[{i}]{t!r}" for i, t in enumerate(decoded)))

        for (L, h, note) in HEADS:
            print(f"\n--- L{L} h{h}: {note} ---")
            potential = head_ov_position_writes(
                model, ids, L, h, k=6, embed=E,
            )
            actual = head_ov_actual_writes(
                model, ids, L, h, k=6, embed=E,
            )
            _print_table("POTENTIAL writes (if attended fully)", potential, k_show=4)
            _print_table("ACTUAL writes (with attention weights)", actual, k_show=4)


if __name__ == "__main__":
    main()
