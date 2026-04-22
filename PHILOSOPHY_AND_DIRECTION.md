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

**Dependency-aware data pipeline, memoization by default.** Interpretability analyses form natural dataflow hierarchies — forward pass → residual extraction at chosen layers → probe construction over a labeled corpus → scoring against held-out prompts → aggregating by a grouping dimension. Each step is expensive to compute and cheap to store, and each step's output is a deterministic function of its inputs that can be reused the next time any analysis asks for the same thing. Jupyter notebooks, the current default tooling, have no shared memoization: every kernel reload recomputes, every teammate's copy recomputes, every fork of an analysis starts the chain from scratch. Mechbench is built around a dataflow DAG that memoizes at every level. A forward pass on a specific (model, prompt) is cached and reusable; a probe built from a specific (corpus, layer, hook_point, head) is keyed by its input signature; sweeps, decompositions, centroid projections, and every other composable primitive record their results into a shared cache that persists across sessions and is accessible to teammates and agents alike. The practical effect is that a researcher joining a team inherits the team's accumulated computational work; an agent running a new analysis pays only for the steps that are genuinely new; an analysis forked from an existing one reruns only the diverged branch. This is the concrete technical answer to "why mechbench instead of a pile of Jupyter notebooks": the cache is a first-class artifact of the platform, not a side-effect of a session.

**Dual-path forward with automatic selection.** Interpretability instrumentation has a cost: capturing activations at every hook point, forcing the manual-softmax attention path, running with backward hooks enabled — these are all slower than the fused, optimized kernels the model uses in production inference. For a framework that wants to support both *fast uninstrumented inference* and *rich instrumented analysis*, the right architecture is a dual-path forward pass where the faster path is always used unless instrumentation explicitly requires otherwise. Mechbench's forward harness already works this way at a coarse level — requests that target attention-internal hooks force the manual attention path, while requests that only need residual captures stay on the fused kernel. The long-term commitment is to generalize this: every primitive declares what instrumentation it requires, and the harness chooses the cheapest forward path that supplies all the required data. An agent, a GUI session, or a batch experiment that asks only for final logits pays the fast-path cost. A request that additionally asks for L23 head-4 Q-vectors pays only the incremental slow-path cost for L23 specifically. A request that wants backward hooks pays only where gradients are needed. This matters especially for agent self-instrumentation and for real-time instrumentation of running models: the platform has to stay close to uninstrumented inference speed in the common case, and only pay for introspection when introspection is actually being used.

**Extensibility as a first-class affordance.** The GUI is where standard experiments get designed and tracked. But new techniques will always outpace any curated primitive set. A new primitive in mechbench consists of a small, well-defined bundle: a Python snippet that performs the operation and exposes results through the emission schema, and a TypeScript snippet that consumes the emission and renders it in the GUI. These extensions drop into the platform without requiring a fork.

**Collaboration as a first-class affordance.** Interp is a team sport. A user should be able to share a live experiment with a teammate, let others annotate charts and comment on findings, fork an analysis to explore a variant, or merge annotations back into a shared workspace. These primitives are as important to the product as the compute primitives — a bench without collaborators is a notebook with extra steps.

**Local-first with remote-optional.** Running on the user's own hardware is the simplest, fastest, most private path for most interp work — especially on Apple Silicon, where MLX gives native performance on models that would otherwise require a rented GPU. Local-first is the default and the demo. But remote compute is a first-class mode: mechbench orchestrates the experiment, the emission schema travels, the frontend renders the same way.

**Multi-model from day one.** Interpretability findings gain credibility by replicating across models. A framework that supports one model is a research substrate; a framework that supports many is a workbench. Mechbench adopts TransformerLens 3.0's `TransformerBridge` adapter pattern: one adapter per model family, with per-variant dimensions read from the loaded model's own config. The current repo already supports Gemma 4 E4B and E2B through a shared `Arch` dataclass; the architecture is in place to expand as new models become relevant.

**First-class support for Mixture-of-Experts architectures.** MoE models (Mixtral, DeepSeek-V3, Qwen-MoE, Gemma 3 MoE variants, future GPT-style MoEs) are not just dense models with a gating step — they introduce a categorical difference in how computation is structured. Tokens are routed to specific experts; different tokens in the same forward pass follow different computational paths; the gating decision is itself a high-information signal about how the model classifies each token. A framework designed around dense-model assumptions — "the residual at layer L, position P" is a single well-defined object — has to stretch uncomfortably to represent MoE behavior. Mechbench commits to treating MoE as a first-class target rather than a grafted-on extension. The primitive layer exposes gating decisions, per-expert activations, and routing patterns as typed artifacts; probes can be built over specific experts rather than over the aggregated layer output; visualizations surface which experts fired for which tokens and how routing distributes across a batch. The central interp question for an MoE model shifts from "what does each layer represent?" to "what does each expert specialize in, and how is the gating network deciding?" — and the platform's forward-pass primitives are designed to answer that question directly, not by analogy to dense-model tooling.

**Agent-native by design.** Interpretability is no longer the exclusive domain of solo researchers or even human-only teams. AI agents — working with Claude Code, Codex, Cursor, or similar coding-agent harnesses — are already conducting large portions of interp work in collaboration with humans, and the trajectory is toward agents taking on more of the experimental design, execution, and analysis loop. A modern interp platform has to be legible and operable to agents, not just to humans clicking through a GUI. Mechbench treats this as a first-class design constraint: its primitives have structured, discoverable interfaces that agents can introspect; its emission schema is self-describing so an agent can reason about results without having to render them visually; its experiments can be declared, run, and compared through command-line and scripting surfaces that parallel the GUI. Skill bundles — small reusable packages that encode "how to run this kind of analysis" — are installable into both human and agent workflows, so the same workflow that a researcher executes by hand can be kicked off by an agent participating in a team's research agenda. The platform is a workbench for humans, for agents, and for teams composed of both.

**Two visualization surfaces, not one.** The GUI renders data slices for humans: navigable, interactive, visually rich, optimized for perceptual grasp via shape, color, motion, and three-dimensional space. But an AI agent operating alongside a human team does not consume the same surface the same way. Even multimodal agents with visual sensors and computer-use skills don't navigate a WebGL particle system the way a human clicks and drags through it — the visual affordances that make a chart legible to a person (tooltips on hover, camera rotation, brush-selected linkage between panels) are not primary affordances for an agent that can inspect the underlying numbers directly. An agent reading a mechbench emission artifact is better served by structured tables, ranked summaries, typed metrics, and queryable data — the raw material of the chart, not the chart itself. Conversely, agents have perception surfaces humans don't: they can read a 2560×2560 matrix of cosine similarities as a table and notice block structure without needing a heatmap to make it legible; they can hold a per-layer trajectory of 42 probe-score scalars in working memory and compare it against a reference without plotting it; they can detect outliers in raw numeric arrays in ways that don't require color or motion to become visible. The agent-facing surface should expose these affordances directly — dense numerical views, JSON queries, structured natural-language summaries — rather than translating every finding through the human visual channel. Practically, this means mechbench's emission layer serves two rendering targets: the human-facing GUI (rich visualizations, navigable) and the agent-facing interface (structured data, queryable, textually summarizable). The same underlying emission artifact feeds both. The chart the human looks at and the table the agent reads are two views of one result.

---

## Who mechbench is for

Three concentric rings of user.

**Mech-interp researchers** who already know the techniques and want better infrastructure. They see figures produced by mechbench and notice the quality. They try it on a model they care about and find that standard workflows — build a probe, sweep a hyperparameter, decompose a logit — take fewer steps than they expected. This group's adoption is what turns mechbench from a personal tool into a piece of field infrastructure.

**ML engineers and AI-safety practitioners** at small labs, startups, and applied teams. They need interpretability capability but don't want to become TransformerLens experts. The node-graph GUI, standard experiments as first-class artifacts, and shareable results make interp accessible to people whose primary job is elsewhere.

**Curious technical users** with an Apple Silicon laptop and an interest in how these models work. No existing tool serves this audience well. Running locally with no API keys, watching attention patterns render in real time, following the findings narrative — mechbench is how somebody learns the field by playing with it.

**AI research agents operating alongside human teams.** Coding agents like Claude Code, Codex, and Cursor are increasingly a participant in research workflows — not just as editors that draft code but as collaborators that run experiments, analyze results, suggest follow-ups, and keep track of where a research agenda sits. Mechbench should be one of the tools those agents reach for. Its primitives are discoverable, its schema is self-describing, its experiments are declarable in code and shareable as artifacts. An agent reading an experiment's emission artifact can reason about the result directly; an agent running a new experiment can compose primitives the same way a human would. The practical implication is that mechbench ships with skill bundles — portable workflows that encode common interp tasks ("build emotion probes over this corpus," "sweep activation patching across these positions," "compare this model to its sibling") — which human researchers and AI agents both execute by the same mechanism. Teams in which humans and agents are both doing interp work should find that mechbench speaks both idioms natively, and that results produced by either flow back into the shared workspace indistinguishably.

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
