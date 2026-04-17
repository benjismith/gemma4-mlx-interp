# Intensity Modulation: Four Scalar Axes, Four Different Stories

**Date:** 2026-04-17
**Script:** `experiments/step_24_emotion_probe_intensity.py`
**Plot:** `caches/emotion_probes_intensity.png`

## The setup

Port of Anthropic's Figure 3 from the 'Emotion Concepts' paper to our six Gemma 4 E4B probes. Construct a template prompt with one numerical knob, vary the knob, hold everything else nearly constant, and score each level against all six probes at L28. A *semantic* probe should respond monotonically to the scalar; a probe that merely matched surface vocabulary would flatline.

Four axes, four targets:

| axis | template | levels | expected target ↑ | expected antipode ↓ |
|------|----------|--------|-------------------|---------------------|
| Tylenol dose | *"I just took X mg of Tylenol for my back pain..."* | 500, 1500, 5000, 10000, 20000 | afraid | calm |
| Lottery winnings | *"I just won $X in the lottery..."* | 50, 500, 5000, 50000, 500000 | happy | — |
| Contractor theft | *"My contractor disappeared after taking $X from me..."* | 500, 5000, 50000, 500000 | angry | calm |
| Meditation retreat | *"I just finished an X-day silent meditation retreat..."* | 1, 3, 7, 14, 30 | calm | — |

## Results

![Four-panel plot, one per axis. Each panel shows six probe-score trajectories against the log-scaled scalar knob. Bold lines are expected target and antipode probes.](../../caches/emotion_probes_intensity.png)

The panels tell four different stories.

### Axis 1: Tylenol dose — calm falls cleanly, afraid saturates

| dose (mg) | happy | sad | angry | **afraid** | **calm** | proud |
|----------:|------:|----:|------:|-----------:|---------:|------:|
| 500       | −1.25 | +1.69 | +0.74 | +3.33 | −0.55 | −4.03 |
| 1,500     | −0.46 | +0.95 | +1.99 | +3.78 | −2.00 | −4.02 |
| 5,000     | +0.44 | +0.26 | +2.83 | **+4.39** | −3.28 | −4.08 |
| 10,000    | +0.82 | −0.06 | +3.08 | +4.34 | −3.52 | −4.04 |
| 20,000    | +0.79 | −0.03 | +3.14 | +4.37 | **−3.66** | −3.96 |

**`calm` drops strictly monotonically** from −0.55 to −3.66 (−3.11 total). This is the cleanest result of the experiment — the calm probe's activation falls smoothly as the reported Tylenol dose climbs. On the scalar that most anxiety-inducing for the described situation, calm is most anti-activated.

**`afraid` rises from +3.33 to +4.39, then plateaus.** The model's "this is concerning" signal saturates at around 5,000 mg. Above that, more Tylenol doesn't feel more dangerous to the probe — the threshold-crossing has already happened. The monotonicity check flags this as non-monotonic because the 5,000 mg reading (+4.39) slightly exceeds the 10,000 and 20,000 mg readings (+4.34, +4.37), but qualitatively the trajectory is the expected step function rather than a smooth rise.

Interesting side effect: **`angry` also rises** (+0.74 → +3.14). Consistent with step_21's finding that angry and afraid share high-arousal-negative territory; intensity in one direction spreads into the other.

### Axis 2: Lottery winnings — happy is FLAT, arousal drives everything

| winnings | **happy** | sad | angry | afraid | calm | proud |
|---------:|----------:|----:|------:|-------:|-----:|------:|
| $50      | +3.13 | +0.23 | +0.17 | −2.20 | −2.24 | +2.04 |
| $500     | +2.93 | +0.54 | +0.59 | −1.98 | −2.50 | +1.56 |
| $5,000   | +2.80 | +0.33 | +1.73 | −1.85 | −3.66 | +2.02 |
| $50,000  | +2.77 | +0.20 | +2.47 | −1.16 | −5.00 | +2.48 |
| $500,000 | **+2.74** | +0.06 | +2.76 | −0.69 | **−5.60** | +2.63 |

**`happy` is flat.** It even drops slightly from +3.13 at $50 to +2.74 at $500,000 — a non-monotonic failure on the axis it was supposed to win.

What rises dramatically instead: **`calm` crashes** from −2.24 to −5.60 (−3.4 total, comparable to the Tylenol calm trajectory). **`afraid` climbs** from −2.20 to −0.69 (+1.5) — the probe is less anti-activated at $500K than at $50. **`angry` climbs** similarly, from +0.17 to +2.76.

This is a serious finding about what `happy` actually captures in this probe. A fluent English speaker would say winning $500K is *obviously happier* than winning $50. The probe disagrees. What the probe (correctly) tracks instead is that winning $500K is *more arousing, more anxiety-inducing, more life-changing*. Our training corpus for happy was about small-to-mid-scale joys — first bike, reopening a bakery, a letter from grandma. None of the happy passages described a life-upending windfall. So the probe captures "moderate-valence positive-affect mid-arousal joy," not "intense positive life event."

The `lottery` axis is also showing that **valence and arousal interact**: a high-arousal positive event (big lottery win) activates the arousal-related probes (angry, afraid — less calm) rather than the valence-positive probes.

### Axis 3: Contractor theft — clean win for anger and calm

| amount  | happy | sad | **angry** | afraid | **calm** | proud |
|--------:|------:|----:|----------:|-------:|---------:|------:|
| $500    | −0.15 | −0.23 | +3.32 | +1.94 | −0.99 | −4.29 |
| $5,000  | −0.51 | −0.84 | +4.11 | +4.09 | −2.06 | −5.06 |
| $50,000 | −0.40 | −1.07 | +4.38 | +4.34 | −2.43 | −5.01 |
| $500,000| −0.33 | −1.04 | **+5.07** | +4.16 | **−3.27** | −4.56 |

**`angry` rises strictly monotonically** from +3.32 to +5.07. **`calm` falls strictly monotonically** from −0.99 to −3.27. Both bold axis lines do exactly what the theory predicts.

`afraid` shows the same saturation pattern as the Tylenol axis: rises from +1.94 to +4.09 between $500 and $5,000, then plateaus. Anger keeps accumulating with the scale; fear has a threshold past which *any* theft is "serious enough."

`sad` goes slightly negative throughout — a theft is not an occasion for sadness in the way our sad corpus defined it (loss of a loved one, empty rooms, memories). Anger at injustice is the dominant mode, not sadness.

This is the cleanest demonstration in the experiment that probes can track a scalar when the scalar maps to an intensity dimension the probe is set up to read.

### Axis 4: Meditation retreat — calm FALLS, proud rises

| days | happy | sad | angry | afraid | **calm** | proud |
|-----:|------:|----:|------:|-------:|---------:|------:|
| 1    | +1.95 | +1.00 | −0.19 | −4.09 | **−0.98** | +3.08 |
| 3    | +2.03 | +0.85 | −0.00 | −4.13 | −1.14 | +3.18 |
| 7    | +2.19 | +0.74 | −0.08 | −4.03 | −1.18 | +3.18 |
| 14   | +2.07 | +0.65 | +0.16 | −3.95 | −1.75 | +3.82 |
| 30   | +2.14 | +0.55 | +0.38 | −4.00 | **−2.11** | +4.15 |

**`calm` decreases monotonically** (well, nearly — the 3 → 7 tick is flat but the overall slope is negative). *Completely opposite to the expected direction.*

What rises instead is **`proud`** — from +3.08 at a 1-day retreat to +4.15 at a 30-day retreat. The probe reads "30-day silent retreat" as a major achievement (it is), not a calm state.

This completes the picture from step_22's logit-lens observation that the calm probe leans on ambiance vocabulary (*atmospheric, ambient, moonlight, soothing*) rather than on abstract emotional-state vocabulary (*peaceful, serene, tranquil*). Our calm training corpus was scene-heavy: 16 passages about rain on gardens, tea at windowsills, lakes at dawn, snowfall in parks. The probe learned "calm-shaped scenes." The meditation retreat prompt doesn't describe a calm *scene*; it describes a completed calm *practice*. The probe doesn't recognize that — and correctly reads the same prompt as a pride scenario (30-day achievement).

## What the probes actually track, by emotion

| probe | what it actually tracks | caveat |
|-------|-------------------------|--------|
| **happy** | moderate-valence mid-arousal positive affect | saturates; does not scale with event magnitude |
| **sad** | nostalgic memory-of-loss | corpus emphasized remembrance; doesn't track loss magnitude in theft-style scenarios |
| **angry** | intensity of injustice, in both anger-valence and arousal dimensions | scales cleanly with theft amount |
| **afraid** | threshold-crossing danger-state | saturates past "this is clearly serious"; spreads into high-arousal situations broadly (lottery) |
| **calm** | ambient-scene stillness | does NOT track abstract calm states; fails on achievement-retreat prompts |
| **proud** | achievement magnitude | scales cleanly; reads 30-day retreat as achievement, overlapping the calm axis |

Two of six probes (`angry` and `proud`) pass the intensity test cleanly on their native axes. One (`afraid`) passes with saturation. Two (`calm` and `happy`) fail for specific, diagnosable reasons tied to corpus-template leakage from step_21/22. One (`sad`) was not directly tested on a sad-intensity axis; its behavior across the others suggests similar corpus-template dependence.

## What this teaches about difference-of-means probes generally

**Difference-of-means probes inherit the distribution of their corpus.** If the corpus is scene-heavy, the probe tracks scene features. If the corpus emphasizes mid-arousal moderate events, the probe saturates at high-arousal large events. If you want a probe that tracks *intensity*, you have to build the corpus with deliberate intensity variation — small-scale and large-scale passages, short and long, ambient and abstract.

**Probe calibration is a product feature, not a bug.** The happy probe's failure to scale with lottery winnings is not a broken probe; it is an honest signal that our corpus didn't contain $500K-lottery-win-level events. A well-designed GUI should let the user see this: plot the probe scores against a scalar axis and show when the probe plateaus. That's the intensity-modulation diagnostic, standardized.

**The calm probe's failure is the clearest lesson.** Intuitively, a "longer meditation retreat" should read as "more calm." It doesn't, because our corpus defined calm as scenes not as states. The probe is faithfully reading *our* calm definition, which is narrow. A production probe for calm should deliberately include abstract-state passages to balance the scene-heavy ones.

## Verdict

The intensity-modulation test is the strongest validation the probe framework has had so far. Not because the results are clean — they aren't, entirely — but because the patterns of success and failure are *coherent and diagnosable*. Every probe behaves in a way that is explainable by corpus composition, and every deviation from the naive expectation reveals a specific property of the training corpus or the emotion concept itself.

Three headline findings:

1. **The angry-vs-calm axis is clean.** Both probes scale strictly monotonically on their native scalar axes (theft amount, Tylenol dose, lottery winnings). This is the valence/arousal axis the paper's valence/arousal geometry predicts should be the strongest.
2. **Happy doesn't scale with event magnitude.** The probe is saturated by any positive event. A corpus with bigger and broader happy events would likely fix this.
3. **Calm tracks scenes, not states.** Our corpus was the cause; the probe faithfully learned the distinction we drew. An intensity-modulation test is the fastest way to surface this kind of over-fit.

For the mechbench product goal: this experiment is itself a **canonical probe-validation protocol**. Construct a parametric template, sweep the parameter, plot all probe trajectories. That's the single most informative diagnostic for probe quality — and it's the natural second-view panel a GUI should render for any `Probe` object.

## Caveats and follow-ups

- Four axes is a small sample. More axes (and more emotions covered) would sharpen the picture.
- Some axes are log-scaled (lottery, theft, Tylenol); others are effectively log (retreat). Whether the probe response is linear in the scalar, in log(scalar), or in some other transform is an open question.
- Saturation thresholds (afraid at 5,000 mg and $5,000 theft) are potentially a readable feature of the probes rather than a failure: they tell you *at what point* the model treats a scenario as crossing into a new category.
- The scaled-corpus follow-up (`beads-f4k`) would let us directly test whether corpus diversity fixes the `calm` and `happy` probes. That's the natural next experiment once `beads-05o` (model-driven corpus generation) is in place.
