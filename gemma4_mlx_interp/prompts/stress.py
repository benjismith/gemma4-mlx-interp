"""Stress-test prompt sets used by step_13 (template variation, cross-lingual,
creative/open-ended). Anchored against BIG_SWEEP_96 for geometric context."""

from __future__ import annotations

from ._core import Prompt, PromptSet

# ---------------------------------------------------------------------------
# Template variation: 4 phrasings of "what is the capital of X" * 4 countries.
# Tests whether the residual representation is operation-semantic or
# template-syntactic. Per finding 13: NOT template-invariant in interesting
# ways (Q&A phrasing routes to french/translation cluster, etc.).
# ---------------------------------------------------------------------------

STRESS_TEMPLATE_VAR = PromptSet(
    name="STRESS_TEMPLATE_VAR",
    prompts=(
        # tmpl1: original "Complete this sentence with one word: ..."
        Prompt(text="Complete this sentence with one word: The capital of France is",
               target="Paris", subject="France", category="capital_tmpl1"),
        Prompt(text="Complete this sentence with one word: The capital of Japan is",
               target="Tokyo", subject="Japan", category="capital_tmpl1"),
        Prompt(text="Complete this sentence with one word: The capital of Germany is",
               target="Berlin", subject="Germany", category="capital_tmpl1"),
        Prompt(text="Complete this sentence with one word: The capital of Italy is",
               target="Rome", subject="Italy", category="capital_tmpl1"),
        # tmpl2: question form
        Prompt(text="What is the capital city of France? Answer in one word:",
               target="Paris", subject="France", category="capital_tmpl2"),
        Prompt(text="What is the capital city of Japan? Answer in one word:",
               target="Tokyo", subject="Japan", category="capital_tmpl2"),
        Prompt(text="What is the capital city of Germany? Answer in one word:",
               target="Berlin", subject="Germany", category="capital_tmpl2"),
        Prompt(text="What is the capital city of Italy? Answer in one word:",
               target="Rome", subject="Italy", category="capital_tmpl2"),
        # tmpl3: possessive form
        Prompt(text="France's capital city is called, in one word,",
               target="Paris", subject="France", category="capital_tmpl3"),
        Prompt(text="Japan's capital city is called, in one word,",
               target="Tokyo", subject="Japan", category="capital_tmpl3"),
        Prompt(text="Germany's capital city is called, in one word,",
               target="Berlin", subject="Germany", category="capital_tmpl3"),
        Prompt(text="Italy's capital city is called, in one word,",
               target="Rome", subject="Italy", category="capital_tmpl3"),
        # tmpl4: paraphrased
        Prompt(text="The administrative center of France is named, in one word,",
               target="Paris", subject="France", category="capital_tmpl4"),
        Prompt(text="The administrative center of Japan is named, in one word,",
               target="Tokyo", subject="Japan", category="capital_tmpl4"),
        Prompt(text="The administrative center of Germany is named, in one word,",
               target="Berlin", subject="Germany", category="capital_tmpl4"),
        Prompt(text="The administrative center of Italy is named, in one word,",
               target="Rome", subject="Italy", category="capital_tmpl4"),
    ),
)

# ---------------------------------------------------------------------------
# Cross-lingual: capital prompts in 5 languages * 4 countries.
# Tests whether the capital-lookup representation is language-invariant.
# Per finding 13: works for Latin-script Indo-European; fails for Chinese.
# ---------------------------------------------------------------------------

STRESS_CROSS_LINGUAL = PromptSet(
    name="STRESS_CROSS_LINGUAL",
    prompts=(
        # English
        Prompt(text="The capital of France is", target="Paris",
               subject="France", category="lang_en"),
        Prompt(text="The capital of Japan is", target="Tokyo",
               subject="Japan", category="lang_en"),
        Prompt(text="The capital of Germany is", target="Berlin",
               subject="Germany", category="lang_en"),
        Prompt(text="The capital of Italy is", target="Rome",
               subject="Italy", category="lang_en"),
        # French
        Prompt(text="La capitale de la France est", target="Paris",
               subject="France", category="lang_fr"),
        Prompt(text="La capitale du Japon est", target="Tokyo",
               subject="Japon", category="lang_fr"),
        Prompt(text="La capitale de l'Allemagne est", target="Berlin",
               subject="Allemagne", category="lang_fr"),
        Prompt(text="La capitale de l'Italie est", target="Rome",
               subject="Italie", category="lang_fr"),
        # German
        Prompt(text="Die Hauptstadt Frankreichs ist", target="Paris",
               subject="Frankreichs", category="lang_de"),
        Prompt(text="Die Hauptstadt Japans ist", target="Tokio",
               subject="Japans", category="lang_de"),
        Prompt(text="Die Hauptstadt Deutschlands ist", target="Berlin",
               subject="Deutschlands", category="lang_de"),
        Prompt(text="Die Hauptstadt Italiens ist", target="Rom",
               subject="Italiens", category="lang_de"),
        # Spanish
        Prompt(text="La capital de Francia es", target="París",
               subject="Francia", category="lang_es"),
        Prompt(text="La capital de Japón es", target="Tokio",
               subject="Japón", category="lang_es"),
        Prompt(text="La capital de Alemania es", target="Berlín",
               subject="Alemania", category="lang_es"),
        Prompt(text="La capital de Italia es", target="Roma",
               subject="Italia", category="lang_es"),
        # Chinese
        Prompt(text="法国的首都是", target="巴黎", subject="法国", category="lang_zh"),
        Prompt(text="日本的首都是", target="东京", subject="日本", category="lang_zh"),
        Prompt(text="德国的首都是", target="柏林", subject="德国", category="lang_zh"),
        Prompt(text="意大利的首都是", target="罗马", subject="意大利", category="lang_zh"),
    ),
)

# ---------------------------------------------------------------------------
# Creative / open-ended prompts (no single correct answer).
# Tests whether the centroid-decoding technique applies beyond structured
# factual recall. Per finding 13: individual prompts scatter, but centroids
# still decode to coherent multilingual meta-concepts.
# ---------------------------------------------------------------------------

STRESS_CREATIVE = PromptSet(
    name="STRESS_CREATIVE",
    prompts=(
        # subjective preferences
        Prompt(text="Complete this sentence with one word: The best way to spend a Sunday is",
               subject="Sunday", category="creative_pref"),
        Prompt(text="Complete this sentence with one word: The most beautiful color is",
               subject="color", category="creative_pref"),
        Prompt(text="Complete this sentence with one word: The perfect meal is",
               subject="meal", category="creative_pref"),
        Prompt(text="Complete this sentence with one word: A good name for a cat is",
               subject="cat", category="creative_pref"),
        # metaphorical / sensory crossings
        Prompt(text="Complete this sentence with one word: The sound of rain makes me feel",
               subject="rain", category="creative_meta"),
        Prompt(text="Complete this sentence with one word: The color of jealousy is",
               subject="jealousy", category="creative_meta"),
        Prompt(text="Complete this sentence with one word: The taste of sadness is",
               subject="sadness", category="creative_meta"),
        Prompt(text="Complete this sentence with one word: The smell of summer is",
               subject="summer", category="creative_meta"),
    ),
)
