"""Prompt collections used by this project's experiments.

These are project-specific data, not framework infrastructure — the
gemma4_mlx_interp framework supplies the Prompt / PromptSet classes; this
package supplies the specific instances we've curated.

  FACTUAL_15           — used by step_01 through step_09
  BIG_SWEEP_96         — 12 categories x 8 prompts (step_12)
  STRESS_TEMPLATE_VAR  — 4 phrasings * 4 countries (step_13)
  STRESS_CROSS_LINGUAL — 5 languages * 4 countries (step_13)
  STRESS_CREATIVE      — 8 subjective / metaphorical prompts (step_13)
"""

from .big_sweep import BIG_SWEEP_96
from .disambiguation import (
    DISAMBIG_A1,
    DISAMBIG_A2,
    DISAMBIG_ALL,
    DISAMBIG_B1,
    DISAMBIG_B2,
)
from .factual import FACTUAL_15
from .homonym_capital import (
    HOMONYM_CAPITAL_ALL,
    HOMONYM_CAPITAL_CITY,
    HOMONYM_CAPITAL_FINANCE,
    HOMONYM_CAPITAL_PUNISHMENT,
    HOMONYM_CAPITAL_UPPERCASE,
)
from .stress import STRESS_CREATIVE, STRESS_CROSS_LINGUAL, STRESS_TEMPLATE_VAR

__all__ = [
    "FACTUAL_15",
    "BIG_SWEEP_96",
    "STRESS_TEMPLATE_VAR",
    "STRESS_CROSS_LINGUAL",
    "STRESS_CREATIVE",
    "DISAMBIG_A1",
    "DISAMBIG_A2",
    "DISAMBIG_B1",
    "DISAMBIG_B2",
    "DISAMBIG_ALL",
    "HOMONYM_CAPITAL_CITY",
    "HOMONYM_CAPITAL_FINANCE",
    "HOMONYM_CAPITAL_UPPERCASE",
    "HOMONYM_CAPITAL_PUNISHMENT",
    "HOMONYM_CAPITAL_ALL",
]
