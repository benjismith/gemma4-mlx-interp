"""FACTUAL_15: the 15-prompt battery used by step_01 - step_09.

Mixed factual recall covering geography, science, culture, sequence
completion, and common sense. Targets are the model's actual top-1 outputs
on Gemma 4 E4B (auto-detected during the original step_01 run, then frozen
here as the canonical answers).
"""

from __future__ import annotations

from ._core import Prompt, PromptSet

FACTUAL_15 = PromptSet(
    name="FACTUAL_15",
    prompts=(
        # Geography
        Prompt(text="Complete this sentence with one word: The Eiffel Tower is in",
               target="Paris", subject="Tower", category="landmark"),
        Prompt(text="Complete this sentence with one word: The capital of Japan is",
               target="Tokyo", subject="Japan", category="capital"),
        Prompt(text="Complete this sentence with one word: The Great Wall is in",
               target="China", subject="Wall", category="landmark"),
        Prompt(text="Complete this sentence with one word: The Amazon River flows through",
               target="Brazil", subject="Amazon"),
        Prompt(text="Complete this sentence with one word: The Sahara Desert is in",
               target="Africa", subject="Sahara"),
        # Science
        Prompt(text="Complete this sentence with one word: Water is made of hydrogen and",
               target="oxygen", subject="hydrogen"),
        Prompt(text="Complete this sentence with one word: The speed of light is measured in",
               target="meters", subject="light"),
        Prompt(text="Complete this sentence with one word: The chemical symbol for gold is",
               target="Au", subject="gold", category="element"),
        # Culture
        Prompt(text="Complete this sentence with one word: Romeo and Juliet was written by",
               target="Shakespeare", subject="Juliet", category="author"),
        Prompt(text="Complete this sentence with one word: The Mona Lisa was painted by",
               target="Leonardo", subject="Lisa"),
        # Sequence completion
        Prompt(text="Complete this sentence with one word: One, two, three, four,",
               target="five", subject="four"),
        Prompt(text="Complete this sentence with one word: Monday, Tuesday,",
               target="Wednesday", subject="Tuesday"),
        # Common sense
        Prompt(text="Complete this sentence with one word: The opposite of hot is",
               target="cold", subject="hot", category="opposite"),
        Prompt(text="Complete this sentence with one word: The color of the sky on a clear day is",
               target="blue", subject="sky"),
        Prompt(text="Complete this sentence with one word: Cats are popular household",
               target="pets", subject="Cats"),
    ),
)
