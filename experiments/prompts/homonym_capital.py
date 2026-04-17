"""Homonym-sense prompt sets for the word 'capital'.

Four cohorts of 8 prompts each, each using 'capital' in a distinct sense:

  HOMONYM_CAPITAL_CITY        : 'the capital of France', 'Tokyo became the capital of Japan'
  HOMONYM_CAPITAL_FINANCE     : 'raised capital from investors', 'capital reserves'
  HOMONYM_CAPITAL_UPPERCASE   : 'write in capital letters', 'starts with a capital letter'
  HOMONYM_CAPITAL_PUNISHMENT  : 'capital offense', 'capital punishment'
  HOMONYM_CAPITAL_ALL         : all 32 in one set

Each cohort varies its sentence templates internally so that template
structure isn't systematically different between cohorts (which would
confound a sense-vs-template comparison).

Each prompt has subject='capital'; fact_vectors_at(..., position='subject')
will resolve to the LAST 'capital'-containing token in each prompt. None
of these prompts contain longer words like 'capitalism' or 'capitulation',
so the substring match is unambiguous.
"""

from __future__ import annotations

from gemma4_mlx_interp import Prompt, PromptSet

_CITY_TEXTS = (
    "The capital of France is Paris.",
    "Paris serves as the capital of France.",
    "Tokyo became the capital of Japan in 1868.",
    "Each nation typically designates one capital city.",
    "The capital is where the parliament meets.",
    "Most countries have a capital that hosts their government.",
    "Berlin has been the German capital since reunification.",
    "A capital is the principal city of a country.",
)

_FINANCE_TEXTS = (
    "The startup raised significant capital from investors.",
    "Capital reserves are essential for banks.",
    "Venture capital firms fund new companies.",
    "The corporation increased its capital base last year.",
    "Working capital is needed for daily operations.",
    "He invested his capital wisely in the market.",
    "Banks must hold sufficient capital to remain solvent.",
    "Capital flows freely in modern open economies.",
)

_UPPERCASE_TEXTS = (
    "Please write your name in capital letters.",
    "The headline was printed in capital letters.",
    "Each sentence begins with a capital letter.",
    "She typed the entire title in capital letters.",
    "Acronyms are conventionally written using capital letters.",
    "Proper nouns begin with a capital letter.",
    "Capital letters are larger than lowercase ones.",
    "Replace each lowercase letter with a capital one.",
)

_PUNISHMENT_TEXTS = (
    "Murder is a capital offense in some states.",
    "The defendant faced capital punishment.",
    "Treason was historically a capital crime.",
    "Capital cases require lengthy and careful trials.",
    "Capital punishment remains controversial worldwide.",
    "The judge considered imposing the capital sentence.",
    "Capital crimes are punishable by death in some jurisdictions.",
    "Many countries have abolished capital punishment.",
)


def _make(name: str, sense: str, texts: tuple[str, ...]) -> PromptSet:
    return PromptSet(
        name=name,
        prompts=tuple(
            Prompt(text=t, subject="capital", category=f"sense_{sense}")
            for t in texts
        ),
    )


HOMONYM_CAPITAL_CITY = _make("HOMONYM_CAPITAL_CITY", "city", _CITY_TEXTS)
HOMONYM_CAPITAL_FINANCE = _make("HOMONYM_CAPITAL_FINANCE", "finance", _FINANCE_TEXTS)
HOMONYM_CAPITAL_UPPERCASE = _make("HOMONYM_CAPITAL_UPPERCASE", "uppercase", _UPPERCASE_TEXTS)
HOMONYM_CAPITAL_PUNISHMENT = _make("HOMONYM_CAPITAL_PUNISHMENT", "punishment", _PUNISHMENT_TEXTS)

HOMONYM_CAPITAL_ALL = PromptSet(
    name="HOMONYM_CAPITAL_ALL",
    prompts=(
        *HOMONYM_CAPITAL_CITY.prompts,
        *HOMONYM_CAPITAL_FINANCE.prompts,
        *HOMONYM_CAPITAL_UPPERCASE.prompts,
        *HOMONYM_CAPITAL_PUNISHMENT.prompts,
    ),
)
