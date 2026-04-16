"""gemma4_mlx_interp.prompts — prompt management + canonical prompt sets.

Prompt and PromptSet are the data types every analysis script consumes.
PromptSet.validate(model) runs each through the model and returns a
ValidatedPromptSet with input_ids and baseline log-probabilities attached.

Predefined sets (mirroring the original experiment scripts):
    FACTUAL_15           — used by step_01 through step_09
    BIG_SWEEP_96         — used by step_12 (12 categories x 8 prompts)
    STRESS_TEMPLATE_VAR  — used by step_13 (4 phrasings * 4 countries)
    STRESS_CROSS_LINGUAL — used by step_13 (capital question in 5 languages)
    STRESS_CREATIVE      — used by step_13 (subjective / metaphorical prompts)
"""

from ._core import Prompt, PromptSet, ValidatedPrompt, ValidatedPromptSet
from .big_sweep import BIG_SWEEP_96
from .factual import FACTUAL_15
from .stress import STRESS_CREATIVE, STRESS_CROSS_LINGUAL, STRESS_TEMPLATE_VAR

__all__ = [
    # Types
    "Prompt",
    "PromptSet",
    "ValidatedPrompt",
    "ValidatedPromptSet",
    # Predefined sets
    "FACTUAL_15",
    "BIG_SWEEP_96",
    "STRESS_TEMPLATE_VAR",
    "STRESS_CROSS_LINGUAL",
    "STRESS_CREATIVE",
]
