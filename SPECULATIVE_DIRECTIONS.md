# Speculative Directions

A seedbed of pie-in-the-sky ideas for mechbench's long-range evolution.

This is the companion document to `PHILOSOPHY_AND_DIRECTION.md`. That document states firm directional commitments — what mechbench is, who it is for, how its architecture should be organized. This document is for the weirder ideas: possibilities that are speculative, partially formed, or dependent on capabilities that don't yet exist. None of them are promises. Some may be years out; some may turn out to be wrong; some might produce the next distinguishing feature of the platform once their technical preconditions arrive.

The purpose is not to enumerate a roadmap. It is to give future contributors — human researchers, AI agents, skeptical readers — a place to see the imaginative envelope the project is trying to inhabit. Good speculative ideas are worth recording even when they are not scheduled.

Contributors are encouraged to add to this document when a provocative idea arrives and doesn't obviously belong in the roadmap or the codebase yet.

---

## 1. Interpretability as live instrumentation

Interp today is retrospective: run an experiment, analyze results, publish findings. Mechbench could attach to a running production model and emit interp data slices in real time. While a user is in conversation with Claude, a side panel shows which heads are firing, which concept probes are activating, which path through a known architectural pivot is being taken. Not debugging in the traditional sense — *watching cognition happen*. The closest analog is real-time brain-imaging dashboards: a new literacy for end-users, a new data source for researchers.

## 2. The probe market

Researchers publish concept probes the way people publish datasets on Hugging Face. Someone builds a high-quality "moral-dilemma detector" probe, uploads it, and any mechbench analysis can pull it in and apply it to its own corpus. The network effect is real: the more probes that exist in the ecosystem, the more useful mechbench becomes, because every new experiment can leverage every existing probe as a lens. Technically lightweight — a probe vector is kilobytes — but the social and infrastructural surface needed to make probe-sharing reliable (versioning, model-compatibility, validation) is nontrivial.

## 3. Probe algebra

If `happy` and `proud` are concept probes, what is `happy ∩ proud`? What is `happy \ proud` (happy-but-not-proud)? What is `happy ∘ surprise` composed through some operator? Concept directions in a shared embedding space permit operations, and the *semantics* of those operations are a genuinely unexplored research area. A probe-algebra interface where users compose concept vectors with set-like operations — and then see which tokens, passages, or positions activate the composed direction — would, as far as I can tell, be novel. It is also the kind of feature that unlocks a different style of question than current interp tools support.

## 4. Cross-model translation via shared concept basis

Build emotion probes on Gemma 4 E4B. Build emotion probes on Llama 3 8B. The probe *directions* are model-specific; the probe *semantics* (happy, sad, angry) are shared. A translation function that maps a concept in one model's residual space to the equivalent concept in another model's residual space would permit questions like "does model A represent happiness the same way as model B?" — and the geometry of that approximation is its own research topic. Mechbench, with multi-model support already designed in, is one of the few places this kind of work could be done at scale.

## 5. Interpretability as a consumer product

Imagine a user-facing app where you paste a conversation you had with an AI and it shows you what the model was doing at each step. Not the raw logits — an interpretable narrative: "the model was in a high-certainty factual-recall mode through tokens 1-40, then shifted into a branching-creative-expansion mode after you said 'brainstorm', with probe activations for 'encouragement' and 'hedging' rising in tandem." The interp equivalent of a fitness tracker: the ambient-awareness form of the technology. Someone just learning the field could use it to develop intuition about how models actually think.

## 6. "Save game" files for cognitive states

Capture a full activation snapshot at a particular step of generation — residuals at every layer, attention patterns, side-channel state, cache state — and save it as a reproducible artifact. Load it back later into the model and resume generation from that exact cognitive state. A researcher could save "the state the model was in right before it made a mistake" and then perturb it — flip a probe, ablate a head, rotate a residual direction — to see what would have changed the outcome. Useful for debugging agents that went off the rails; useful for counterfactual-behavior research; useful as a data format for interp-for-safety investigations.

## 7. Interpretability-guided fine-tuning

If probe activations reveal *why* the model produced a particular output, those activations can become training signal. The `a0r` epic (randomness / entropy / dice-throwing) is already walking toward this — the probe on "branching mode" becomes a reward signal for training the model to enter that mode in appropriate contexts. Generalize: any interpretable behavioral property can become a fine-tuning target by way of a probe. This blurs the line between "understanding" and "intervening" in a productive way. Every probe the platform can measure becomes a lever the platform can pull.

## 8. Collaborative 3D spatial workspaces

Genuinely pie-in-the-sky. A VR or AR space where a team of researchers walks together through a model's residual-stream manifold. The 42-layer stack is a physical corridor; attention patterns are visible edges between positions; probe directions are arrows you can point at. One researcher drags a probe-vector across the stack and asks "what if we rotate this 30° toward the K-space direction at L23?" — their collaborator watches the rotation in real time and sees the downstream effects on logit attribution. This is a form of collaboration that doesn't exist yet for any kind of work, and dense high-dimensional-model internals is genuinely the right domain for it.

## 9. Semantic weight diff — git-for-models

When a researcher fine-tunes a model, mechbench computes the WEIGHT changes and shows them *semantically*. Not "3.7 million parameters changed" but "this specific attention head now writes sadness tokens more strongly, this MLP direction got stronger for moral-reasoning contexts, this residual subspace rotated 14° toward the empathy cluster." Versioned model artifacts with meaningful-diff visualizations. Related: time-lapse animations of training, where you can scrub through checkpoint history and watch concepts emerge, consolidate, and specialize.

## 10. Interpretability firewall for agent safety

Run a model under mechbench instrumentation and configure it to reject any inference step where a specified probe activates above a threshold — a deception probe, a reward-hacking probe, a self-exfiltration-intent probe. The firewall isn't just behavioral (blocking certain output tokens) but *cognitive* (blocking the internal states that produce those outputs). Much closer to actual safety than anything currently in the production AI stack. Strongly speculative — depends on probe quality improving dramatically — but compellingly aligned with the AI safety research agenda, and potentially a route to institutional adoption and funding.

## 11. Structured-interview interface for the model

Instead of chat, the user interacts with the model through a paired UI: the model's reply on the left, a real-time mechbench view on the right. Click a word in the reply and see which heads contributed to it. Highlight a phrase and ask "what was the model attending to when it said this?" A layered-transparency UX that the current chat interface simply doesn't offer. Not a replacement for chat — an alternative for users who want to understand the model at a deeper level than its surface output.

## 12. Functional interpretability via perturbation

Run the same prompt through a model under different mechbench-instrumented conditions — add a probe, suppress a head, inject a concept direction, rotate a residual — and see how the response changes. The model under perturbation is how researchers actually build intuitions about its cognition, the way therapists build intuition about clients by how they respond to specific prompts. A tool that makes this kind of investigation fluid and fast is genuinely different from anything currently available; it turns interpretability from a static analysis into a conversational probe.

## 13. A theorem-proving interface for interp claims

A researcher states a claim: "the model uses L23 as the last fresh-K/V-global layer for content integration." Mechbench runs a battery of pre-designed tests: does ablating L23 hurt downstream tasks? Does the activation direction rotate at L22→L23? Does the effect disappear when L17 is substituted? Each test is a primitive; the claim is a composition of passing tests. Over time, a library of *proved interp claims* accumulates — each one a reproducible experiment attached to a specific assertion. Closer to how interpretability should work as a field, versus its current form of "one person's notebook showing an interesting plot."

## 14. Backward-pass and training-dynamics interpretability

Almost all interp work today — this project included — focuses on the forward pass: what the model computes given a fixed set of weights. But models are not static artifacts; they are the end state of a training process that flows gradients backward through the same architecture forward inference uses. The backward pass is where *learning* actually happens, and it is a largely unexplored interpretability surface.

Several specific directions open up once a framework supports backward-pass instrumentation:

**Training-pressure mapping.** The narrative of L23's load-bearing role in Gemma 4 E4B is, at root, a claim about gradient flow: training pressure concentrates at architectural bottlenecks where specific mechanisms (fresh-K/V global attention on a deep residual) are available. With backward hooks we could observe this directly — measure per-layer gradient magnitude during training, watch it concentrate at L23 over time, and confirm (or refute) that the cross-task L23 pivot is indeed an emergent consequence of gradient competition for a narrow architectural resource.

**Concept emergence across checkpoints.** Probe vectors are stable with respect to a fixed model. But they are not stable across training checkpoints: at the start of training, probes for "happy" or "nation" don't yet exist as learnable directions. Somewhere during pretraining, the concept emerges — maybe gradually, maybe in a phase transition, maybe non-monotonically. A mechbench-instrumented training run could checkpoint probe geometry every N steps and produce a time-series of concept formation. The L23 structural hub presumably wasn't present at step 0; when did it crystallize? What training-data features were being seen at that moment?

**Per-example influence analysis.** For a specific training example, which weights receive the largest gradients? Which parts of the model are being updated most, and which pass through unchanged? A version of influence functions but computed during actual training rather than retrofitted afterward. This is the right way to understand what a specific data point is teaching the model.

**RL-specific dynamics.** Reinforcement learning shapes gradients differently than supervised fine-tuning. The KL penalty to a reference policy produces one gradient signature; the reward-model signal produces another. Watching how different RL components distribute their influence across layers, heads, and residual subspaces would turn training-time choices (PPO vs DPO, KL coefficient, reward-model architecture) into empirically-observable interventions rather than hyperparameter tweaks.

**Catastrophic-forgetting and repurposing diagnostics.** During fine-tuning, which representations get overwritten? Which concept probes shift, and by how much? Diff the probe geometry before and after a fine-tuning run and you have a direct measurement of what was lost, what was preserved, and what was added. Useful for alignment research (understanding what RLHF actually changes) and for practical fine-tuners (diagnosing when a fine-tune has eroded a capability that should have been preserved).

**Curriculum and data-mix effects.** Different training data distributes its influence differently. Watching gradient magnitudes and directions during training on code vs. prose vs. math vs. multilingual text would turn the "what does pretraining data actually do to the model" question into a measurable time series. Useful for training-data curation choices; useful for understanding how multimodal and multilingual capabilities are acquired.

**Live training dashboard.** Real-time instrumentation during training runs — probes, gradient flow, per-layer update magnitudes — rendered as a live mechbench view. Instead of watching training loss and guessing at what the model is doing, watch concept probes emerge, see gradient pressure shift across epochs, detect training pathologies (mode collapse, gradient explosion, KL runaway) as they happen rather than at checkpoint-inspection time.

**MoE routing dynamics under backprop.** Mixture-of-Experts models are a special case where backward-pass interpretability is especially rich. The gating network decides which experts process each token; the gradient that flows back through a forward pass is *routed* — concentrated in the specific experts that were activated for each token, and in the gating weights that sent it there. Watching this flow during training reveals a whole set of phenomena that dense-model interp can't surface: how expert specialization emerges (does the "math expert" crystallize gradually, or does it win a competitive phase?); whether gating collapse is already measurable from gradient signatures before it shows up in output behavior; how gradient load distributes across experts as a function of data mix (a code-heavy batch should show concentration in code-specialist experts); how auxiliary losses (load-balancing regularizers) interact with primary-task gradients to shape expert diversity. For the recent wave of MoE and conditional-compute models (Gemma 3 MoE variants, DeepSeek-V3, Mixtral, Qwen-MoE), training-time routing analysis might be the *primary* interp question: understanding what each expert learned is understanding what the model learned. A mechbench view over MoE training that shows per-expert gradient activation, per-token routing decisions, and per-batch expert utilization — alongside the usual probe and activation views — would be directly useful for anyone building or fine-tuning these models, and there is currently no tooling at that level of specificity.

Technically, this extends mechbench's scope in a meaningful way. Interp primitives need to support backward hooks and gradient capture (beads-bge in the TL-parity epic already addresses this). The dual-path forward-pass selection extends to a dual-path training step: one fast path for standard training, one instrumented path for gradient analysis. The memoization layer has to account for weight state at specific checkpoints. The emission schema needs to represent gradient-valued data in addition to activation-valued data.

The payoff is that mechbench becomes a tool for *both* kinds of interpretability: understanding a trained model (today's focus) and understanding *how it became trained* (this direction). That second mode is closer to where AI safety research actually needs to live — you can't align behavior you don't understand, and you can't understand behavior whose training dynamics you can't observe. The forward-pass-only framing leaves that half of the problem untouched.

## 15. Agent self-instrumentation: interp primitives as model-callable tools

Expose mechbench's primitives as callable tools an agent can invoke on its own forward pass. Before answering a hard question, the agent calls `mechbench.capture(layers=[17, 23, 29], hook_points=['resid_post', 'attn.q'])` to instrument its next inference step. Then in a follow-up tool call, it retrieves the captured artifacts and reasons about them: "I notice my L23 residual in this response was unusually far from the nearest emotion-probe centroid — let me reconsider whether I'm answering in the mode this user expected." A form of metacognition that isn't just chain-of-thought over text, but literal introspection over one's own activations.

This is qualitatively different from #1 (live instrumentation as an external observer watching the model from outside). Self-instrumentation is the model instrumenting *itself*, with tool results that flow back into its own context as legitimate reasoning material. It opens questions that don't currently have good answers: if a model can read its own L23 probe activations, can it learn to modulate them? If it can see which heads were firing during a previous response, can it use that signal to debug its own reasoning? If an agent in a multi-turn conversation notices its "deception probe" activated at turn three, what does it do with that information?

Technically, this requires a forward-pass harness that can be interrupted, instrumented, and resumed — the "save-game" state from #6 is a prerequisite. It also requires a careful account of what activation data is safe to expose to the model itself, which is a nontrivial safety question. Some activation signals (uncertainty, concept activations) seem fine to surface. Others (internal representations of the user, self-evaluation circuits) might create feedback loops whose consequences are harder to reason about.

But if it works, it changes what it means to be an AI agent. Current agents reason in text over their external observations. A self-instrumented agent reasons in text over its own internal states, grounded in real measurements rather than in verbal approximations of those states. The gap between "I think I was confident about that answer" and "I measured my L23 entropy-probe activation and it was 0.72, which is higher than my typical confident-answer baseline of 0.34" is the gap between introspection-as-confabulation and introspection-as-measurement.

A mechbench platform that exposes its primitives as tools would be the natural place to prototype this. The primitives are there; the emission schema is there; the only missing piece is the tool-call wrapping and the harness that routes the results back into the agent's context window.

---

## Rough near-term viability

The ideas above sit at different distances from current capability. As a rough sorting:

- **Technically immediate** given the existing primitive layer: the probe market (#2), probe algebra (#3), save-game cognitive states (#6), functional interpretability via perturbation (#12).
- **Medium-range** — requires sustained framework maturation plus new infrastructure: interp-guided fine-tuning (#7), cross-model concept translation (#4), theorem-proving interp (#13), semantic weight diff (#9), backward-pass and training-dynamics interp (#14), agent self-instrumentation (#15).
- **Long-range** — dependent on external capability advances, larger user bases, or substantial product surfaces that don't yet exist: live instrumentation (#1), consumer interp product (#5), VR workspaces (#8), interp firewall (#10), structured-interview UI (#11).

Any of these could move up the list quickly if the project finds itself wanting to build in that direction. Equally, any of them could be removed from the list as experience shows they're not actually worth pursuing. This is not a roadmap — it's an imagination budget.

---

## Closing note

Speculative ideas compound. An idea recorded here that seemed fanciful in 2026 may find its moment in 2028 when a capability prerequisite suddenly exists; an idea that seemed essential may turn out to have been answered by something else entirely. The value of the document is not that any particular idea on it is correct, but that the act of writing ideas down keeps the project's imaginative range wider than its execution range — which is the natural condition of a healthy research-adjacent engineering effort.

Contributors: when a speculative idea arrives, add it here. Keep the intro and the near-term-viability grouping current. Delete entries that have been executed (and link to their execution), or entries that have been shown to be wrong (and link to the evidence).
