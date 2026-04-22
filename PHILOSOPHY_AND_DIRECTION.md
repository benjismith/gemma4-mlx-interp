# Philosophy and Direction

This document describes what **mechbench** is, what it is for, and how the project should evolve. It is written so that a new contributor — or a returning one after a long gap — can orient quickly on the north star and the design choices that follow from it.

The repository is currently named `gemma4-mlx-interp`, reflecting where the work started: mechanistic interpretability experiments on Google's Gemma 4 family, running locally via MLX on Apple Silicon. That name will be superseded. The destination is **`mechbench.ai`** — an open-source workbench for mechanistic interpretability.

---

## The premise

Mechanistic interpretability has matured into a field with real techniques: linear probes over activations, attention-pattern analysis, causal tracing, head-composition circuits, sparse-autoencoder features, logit-attribution decompositions. The techniques themselves are not secret. What is missing is the *infrastructure for thinking with them*.

The canonical Python library — **TransformerLens** — is excellent for researchers already fluent in Jupyter notebooks, hook callbacks, and numpy. It supports 50+ architectures and has a mature test suite. But it is a library, not a workbench. Its users drive it from scripts.

**Neuronpedia** is a polished hosted dashboard, beautiful in the places where it is beautiful, but narrow: it browses sparse-autoencoder features for a specific set of models, and that is mostly what it does.

Between these two poles — research-library-you-script-against and narrow-browseable-dashboard — there is a large gap. The daily work of *doing* interpretability happens in unshared Jupyter notebooks, cryptic variable names, and scripts that live on individual researchers' laptops. Every analysis reinvents its own plotting code. Every new technique requires wiring glue between the model, the cache, the metric, and the visualization. The methods are composable in principle; they are not composable in practice.

Mechbench is an answer to that gap.

---

## What mechbench is

A workbench for mechanistic interpretability, with three parts that interoperate cleanly:

1. **A primitive layer in Python.** Not hundreds of functions: a curated set of roughly thirty to fifty primitives at the right level of abstraction — `Probe`, `Cluster`, `fact_vectors_at_hook`, hook-aware forward passes, intervention composition, residual decomposition, circuit SVD. Each primitive expresses a specific cognitive move. They compose.

2. **A structured emission layer.** Every primitive produces typed data slices with schemas rich enough for a frontend to render without redoing analysis — per-element metadata for tooltips, linkage keys for cross-view brushing, semantic typing for colormap and axis defaults. The emission layer is the interface contract between compute and presentation.

3. **An interactive visualization layer.** Data slices are rendered as beautiful, navigable visualizations. Static charts where static is right; animated transitions where motion reveals structure; three-dimensional navigable views for dense activation data (residual-stream trajectories as flowing particles, attention patterns as luminous edge graphs) where two dimensions are not enough. The visualization layer is a separate codebase, built against the emission schema, likely web-first so that results can be shared by URL.

These three layers are logically and physically separable. Compute can run locally on the user's Apple Silicon hardware. Compute can also run on a remote H100 cluster, or any other scaled-compute endpoint, with the same emission schema shipping results back to the same frontend. The user drives the experiment from the GUI; the compute backend is a configuration choice, not a hard-coded assumption.

---

## What distinguishes mechbench

**Visualization as thinking, not decoration.** Interpretability produces dense, high-dimensional data — 42 layers × 8 heads × thousands of tokens × 2560 residual dimensions. Conventional chart libraries compress that into tiny heatmaps and distribution plots. Mechbench treats visualization as the act of reading: every chart is *about* something specific, outliers are annotated by default, interactions reveal structure that is not visible statically, motion is purposeful (a PCA scatter rearranging when the basis rotates reveals geometry; a fade does not). The 3D and particle-system roadmap is where the real differentiation lands — visualizing dense activation data in navigable spaces where human perception can actually engage with it.

**Composable primitives.** The interface is small and deliberate. Each primitive expresses one idea. Users wire them together in a node-graph GUI or in Python scripts — same objects, same semantics, two surfaces. New primitives can be added as the field evolves, because the emission schema is the stable contract.

**Extensibility as a first-class affordance.** The GUI is where standard experiments get designed and tracked. But new techniques will always outpace any curated primitive set. A new primitive in mechbench consists of a small, well-defined bundle: a Python snippet that performs the operation and exposes results through the emission schema, and a TypeScript snippet that consumes the emission and renders it in the GUI. These extensions drop into the platform without requiring a fork.

**Collaboration as a first-class affordance.** Interp is a team sport. A user should be able to share a live experiment with a teammate, let others annotate charts and comment on findings, fork an analysis to explore a variant, or merge annotations back into a shared workspace. These primitives are as important to the product as the compute primitives — a bench without collaborators is a notebook with extra steps.

**Local-first with remote-optional.** Running on the user's own hardware is the simplest, fastest, most private path for most interp work — especially on Apple Silicon, where MLX gives native performance on models that would otherwise require a rented GPU. Local-first is the default and the demo. But remote compute is a first-class mode: mechbench orchestrates the experiment, the emission schema travels, the frontend renders the same way.

**Multi-model from day one.** Interpretability findings gain credibility by replicating across models. A framework that supports one model is a research substrate; a framework that supports many is a workbench. Mechbench adopts TransformerLens 3.0's `TransformerBridge` adapter pattern: one adapter per model family, with per-variant dimensions read from the loaded model's own config. The current repo already supports Gemma 4 E4B and E2B through a shared `Arch` dataclass; the architecture is in place to expand as new models become relevant.

---

## Who mechbench is for

Three concentric rings of user.

**Mech-interp researchers** who already know the techniques and want better infrastructure. They see figures produced by mechbench and notice the quality. They try it on a model they care about and find that standard workflows — build a probe, sweep a hyperparameter, decompose a logit — take fewer steps than they expected. This group's adoption is what turns mechbench from a personal tool into a piece of field infrastructure.

**ML engineers and AI-safety practitioners** at small labs, startups, and applied teams. They need interpretability capability but don't want to become TransformerLens experts. The node-graph GUI, standard experiments as first-class artifacts, and shareable results make interp accessible to people whose primary job is elsewhere.

**Curious technical users** with an Apple Silicon laptop and an interest in how these models work. No existing tool serves this audience well. Running locally with no API keys, watching attention patterns render in real time, following the findings narrative — mechbench is how somebody learns the field by playing with it.

---

## What the long-term vision implies for near-term work

The current repo is already walking toward this destination. Several organizing principles follow from the vision above.

**The compute primitive layer is the priority.** Every new experiment should land as composable primitives. The framework's `Probe`, `fact_vectors_at_hook`, `head_weights`, intervention composition, plot helpers, and hook system are the substrate the GUI will eventually run on. Making those primitives clean, well-named, and stable is the highest-leverage work available today.

**The emission schema is the critical bridge.** Separating data generation from rendering — even before the GUI exists — means that experiment scripts emit structured artifacts, not matplotlib figures. Scripts can still call the current plot helpers to produce PNGs for the narrative, but the underlying data should always be serializable to a declarative specification that a future frontend can render natively.

**Multi-model support is not a post-MVP concern.** The repo has already begun generalizing within the Gemma 4 family via a `TransformerBridge`-style adapter pattern. Extending to other MLX-compatible model families (and, eventually, remote-compute model families) is a day-one design constraint, not a rewrite.

**Narrative-style thinking remains essential while findings accumulate.** The `experiment-narrative.md` essay currently serves a specific purpose: it forces every experiment to connect to a story, which keeps the work pointed at real questions rather than drifting into curiosities. That essay will not live in the repo forever — once the project stabilizes into its product form, the essay will move to publication as a standalone artifact. Until then, it stays here as the thinking surface.

**Publication findings from mechbench will appear at `machinecreativity.substack.com`** — the project owner's writing venue — as long as the mechbench work and the Machine Creativity framing remain connected. If mechbench grows large enough to warrant its own publication surface, it will get one.

---

## What mechbench is not

It is not trying to be TransformerLens. TransformerLens is an excellent library with a mature breadth of model support and a committed community; duplicating it is pointless. Mechbench's path is depth on a smaller set of models with first-class visualization, composability, and collaboration — a different product category.

It is not a hosted service, at least not by default. A user should be able to clone the repository, install the Python package, and run interp experiments on their own hardware with no external dependencies. A hosted tier might exist someday; it is not the product.

It is not purely a research project. The intent is a usable workbench, which implies the polish, documentation, onboarding, and stability that distinguish a tool from a prototype.

It is not tied to any particular model family forever. Gemma 4 is the starting substrate because it runs cleanly on Apple Silicon and has architectural features (hybrid attention, KV-sharing, the MatFormer side-channel) that are interesting to study. Other model families will follow as the adapter pattern is exercised.

---

## Open design questions

These remain genuinely open and will be answered by experiment rather than declaration.

- **Frontend stack.** Web-native (shareable by URL, multi-user friendly) is a strong default. Electron desktop (richer local hardware integration) is a plausible alternative. The emission layer is designed to be frontend-agnostic; the choice can be made later.

- **Extension primitive format.** The shape of a Python+TypeScript bundle that extends the platform — how it is declared, where it is stored, how it is installed — needs prototyping before it is specified.

- **Commercialization path, if any.** Mechbench will remain open-source. Whether a hosted tier, enterprise features, or a managed collaboration surface emerges on top of the open core is a question for a later year. Keeping the open-source substrate clean and useful is the current priority; commercial layering is optional and deferred.

- **Relationship to existing tools.** Integrations with TransformerLens (import a TL-captured activation cache?) and Neuronpedia (load an SAE feature set?) may be worth building over time. These are interop questions, not architectural ones.

---

## Summary

Mechbench is an open-source workbench for mechanistic interpretability, structured as three separable layers — composable Python primitives, a typed emission schema, and an interactive visualization frontend — designed so that compute can run locally or remotely, experiments can be shared and forked among collaborators, and new techniques can be added as first-class extensions as the field evolves. It starts from a research substrate targeting Gemma 4 on MLX; it expands via an adapter pattern to support the models and backends that users actually care about. The visualization layer aspires to make dense interpretability data legible in a way no existing tool currently achieves.

The technical work happens one primitive at a time. The vision, written down here, is what the primitives are for.
