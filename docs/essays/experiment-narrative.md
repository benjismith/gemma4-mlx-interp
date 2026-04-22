# An Interesting Architectural Detail Inside Gemma 4

*Taking apart a small language model to see what's actually doing the work*

---

Google's Gemma 4 E4B is a 4-billion-parameter open-weight model released earlier this year. It's small enough to run comfortably on a laptop, and like most modern transformers, it's built as a stack of forty-two layers, each containing an attention mechanism and a feed-forward MLP branch. But it has an unusual architectural feature that I hadn't seen discussed much: a **per-layer-input side-channel**, a small linear pathway that sits alongside the main residual stream and feeds a separate per-layer token embedding into every block. Google calls this the MatFormer structure. Documentation about what it does, and why, is surprisingly thin.

Over a weekend of ablation experiments — zeroing out components one at a time and measuring how much the model's predictions degrade — I found that this side-channel is doing substantially more work than its size and prominence would suggest. Removing it across the entire network is more destructive than removing any single transformer layer. Removing it at just one layer is often comparable to removing that layer's entire MLP branch. And its effect concentrates at the network's global-attention layers: the side-channel isn't distributed uniformly across the stack, it's specifically load-bearing for the layers that integrate information across the full context.

That's the main finding. Along the way, the investigation also turned up a few other things worth noting: the logit lens (a standard interpretability tool for reading intermediate representations) mostly fails on Gemma 4's middle layers, even though those are where the critical work is happening; individual attention heads are largely redundant, so ablating any single head rarely hurts the model; the global-attention layers mostly attend to chat-template structure rather than to subject-entity content; and the factual information the model retrieves is causally localized at subject positions in the middle layers, even though it isn't decodable there.

The implications are modest but specific. For interpretability researchers, the side-channel is a concrete new component to think about, in Gemma 4 and in the MatFormer family more broadly. For anyone thinking about architectural design, it's worth understanding why such a small feature carries so much weight — a hypothesis I'll get to. For follow-up research, the obvious directions are checking whether the effect generalizes to Gemma's larger variants, probing what kind of information actually flows through the side-channel (positional? token-identity? something else?), and seeing whether similar auxiliary pathways in other architectures are doing comparable work.

What follows is the methodology. The investigation wasn't linear, and I'll skip the unproductive avenues, but the productive experiments all build on each other in a way that's worth stepping through in order.

---

## 2. The Patient

To follow what comes next, you need a handful of facts about how Gemma 4 E4B is built.

Like every modern transformer, it's organized around a set of per-token vectors that get transformed layer by layer. The vectors are 2560-dimensional, one per token position in the context. You can picture them as the columns of a matrix — the context matrix — that gets rewritten at every layer into a new version of itself. The interpretability literature calls this column-stack-through-time the **residual stream**, and most of what the model "thinks" at any given moment during its computation lives in those vectors. Reading the inside of a language model amounts to watching the residual stream evolve.

At each of its forty-two layers, Gemma 4 updates the residual stream in two stages. First, an **attention branch** lets each token's vector incorporate information from the vectors at other positions (this is what lets "the Eiffel Tower" inform the subsequent prediction of "Paris"). Second, a **feed-forward MLP branch** — "multilayer perceptron," the old name for a small fully-connected network — does a position-wise transformation on each token's vector that doesn't involve the other positions. Both branches are standard, and both update the residual stream additively: whatever each branch produces gets *added* to the existing vectors rather than replacing them. The residual stream accumulates; nothing ever fully overwrites it.

Two of Gemma 4's architectural choices are worth flagging up front, because they shape the investigation that follows.

The first is its **hybrid attention pattern**. Most transformers give every layer full global attention — each token can attend to every other token in the context. Gemma 4 does this at only seven of its forty-two layers, placed every sixth layer: 5, 11, 17, 23, 29, 35, and 41. The other thirty-five layers use **sliding-window** attention, meaning each token can only attend to a local neighborhood of nearby tokens. This is a compute-saving design choice (full attention scales quadratically with context length; local attention doesn't), and it has a structural consequence: long-range information in Gemma 4 can only propagate at those seven specific "global" layers. Five local layers do neighborhood-scale work, then a global layer lets everything talk to everything, then five more locals, then another global, and so on. The final layer is always global, so the model's last opportunity to integrate long-range context happens right at the end.

The second is the **tied embedding**. Most transformers have a vocabulary-to-vector table on the input side (the embedding) and a separate vector-to-vocabulary table on the output side (variously called the unembed or LM head). Gemma 4 uses the same matrix for both. When the model turns input tokens into vectors, it consults the table in one direction. When it turns its final-layer residual vectors into **logits** — the vocabulary-sized scores from which the next-token prediction is sampled — it consults the same table in the other direction. There is one library of token representations, used bidirectionally.

And then there's the feature I'm most interested in.

**The MatFormer per-layer-input side-channel.** Alongside the main residual stream, Gemma 4 carries a second data pathway: a large separate embedding table (roughly 262,000 rows by 10,752 columns) that produces, for every input token, a different small vector at every layer. At each block, this per-layer vector is fed into a small linear projection called the `per_layer_input_gate`, passed through a nonlinearity, multiplied element-wise with a projection of the residual stream, passed through another linear layer, normalized, and added back in. It's a surprisingly elaborate auxiliary pathway. Google introduced it as part of what they call MatFormer, but public documentation of what role it actually plays is sparse. Looking at this sub-circuit in the source, it's easy to assume it's a small auxiliary detail. That assumption turns out to be wrong.

Those are the moving parts. The rest of the essay is what happens when you start knocking them out, one at a time, to see what falls over.

---

## 3. A Standard Probe, and Where It Fails

Before going after specific architectural features, I started the investigation with the standard first tool for this kind of work: the **logit lens**. The logic is simple. At every layer, the residual stream is a set of 2560-dimensional vectors. Normally the model only projects these through the output head at the very end of the stack, to produce the logits. But nothing physically stops you from doing that projection at any layer, at any token position, and reading off what the model "would predict" if it stopped computing right there. You run a prompt, pick a layer, pick a position, project through the output head, and look at the top tokens by probability.

On a well-behaved prompt with an unambiguous answer, the logit lens usually produces a recognizable narrative. The early layers are noise: the top predicted tokens are random-looking wordpieces, and the correct answer sits at a rank in the tens of thousands. Somewhere in the middle the representations start to look more coherent, and the correct answer begins its climb. By the final layer the model has settled into its actual prediction. Watching this evolve across depth is, honestly, one of the more satisfying things you can do in this corner of interpretability — you can see a prediction condense, out of nothing, into something sharp and specific.

I ran the logit lens across fifteen factual-recall prompts on Gemma 4: questions like *The capital of Japan is*, *The Eiffel Tower is in*, *The opposite of hot is*, and so on. I picked prompts the model answered with high confidence, so the lens would have a clean target to look for, and I tracked the rank of the correct answer at every one of the forty-two layers at the final token position.

What I got was a classic phase transition. Across all fifteen prompts, the average rank of the correct answer stayed in the 60,000-to-150,000 range (out of a 262,144-token vocabulary) from layer 0 all the way to layer 24. Then, over the span of about six layers, it crashed. By layer 30 the correct answer was at rank 16 on average. By layer 36 it was at rank 2. By the final layer, rank 0 — the model's prediction.

![Rank of the correct answer at each of 42 layers, logit-lens projection at the final token position. Thin lines are individual prompts; the bold red line is the geometric mean across 15 prompts. The lower panel shows the log-probability of the correct answer at each layer. Dashed vertical lines mark the seven global-attention layers.](images/logit_lens_batch.png)

This looks very dramatic in a plot. It also leaves a puzzle. More than half of Gemma 4's depth — the first twenty-five layers — appears to produce essentially nothing that the logit lens can read. The residual stream at those depths, projected through the output head, is nearly indistinguishable from noise. But the network is clearly doing *something*: the final prediction is correct, and the late layers assemble that prediction out of whatever the early layers handed them. So either the early layers are doing critical work the logit lens can't see, or they're doing almost nothing and the whole computation happens in the last six layers.

The easy story is the second one — the model is mostly just passing information along until the late layers do the real work. It's easy because it matches what you see. It's also wrong. The ablation experiments said the opposite: those early and middle layers are where almost all of the critical computation is happening. The logit lens can't see it because the residual stream at those depths contains information the model depends on, but that information isn't aligned with the output-vocabulary direction of the embedding space. The lens is asking the wrong question.

To see what the lens misses, we have to do something cruder.

---

## 4. Taking Pieces Out

**Ablation** is crude and direct: you pick a component of the model, you set its output to zero, and you measure how much the model's predictions get worse. If the component was contributing something to the computation, removing it should hurt. If removing it does nothing, the component probably wasn't doing anything. That's the whole method.

The natural first move on Gemma 4 is to ablate one layer at a time. I run a prompt through the model, but at a specific layer, I arrange things so that the layer contributes nothing to the residual stream — the stream passes through unchanged, as if the layer weren't there. Then I continue the forward pass and see what the model ends up predicting.

As a measure of the damage, I used the log-probability the model assigns to its own top-1 answer. If the baseline model is 97% confident that the capital of France is Paris, and with a particular layer ablated it drops to 50% confident, that's a significant damage score. If it drops to 96%, the layer is essentially not doing anything — at least for this prompt. I averaged the damage scores across the same fifteen factual-recall prompts I used for the logit lens.

*A quick aside on why log-probability rather than raw probability.* Raw probability is bounded between 0 and 1, which makes it a lousy damage metric when a model's confidence can span many orders of magnitude. A confident answer falling from 97% to 0.001% and a confident answer falling from 97% to 50% both register as roughly the same raw-probability drop (around 0.5–1), even though the first is many orders of magnitude more destructive than the second. Log-probability takes the logarithm of the probability, which converts multiplicative changes in confidence into additive ones. A log-probability of −0.03 is a 97%-confident answer; −0.7 is 50%; −16 is the model essentially announcing that the answer is "no, definitely not this one." It's the natural currency for this kind of measurement — and not coincidentally, it's also what the model is optimized against during training.

![Mean impact of ablating each of Gemma 4's 42 layers individually, measured as the drop in log-probability of the model's own top-1 answer, averaged across 15 factual-recall prompts. Lower (more negative) bars mean a more damaging ablation. Red bars mark global-attention layers; blue are local.](images/layer_ablation.png)

Layer 0 is catastrophic. Ablating it drops the log-probability by about sixteen — which, roughly translated, means the model's confidence in its own answer falls by many orders of magnitude. This isn't mysterious. Layer 0 is where the raw token embeddings get their first real transformation. Without it, every subsequent layer is seeing input that's statistically unlike anything it was trained on. The network is calibrated to expect layer 0's output, not layer 0's input, so removing it breaks everything downstream.

After that, the pattern is striking: the most damaging individual ablations are concentrated in the middle of the network, specifically layers 10 through 24. Layer 14, ablated alone, drops the log-probability by more than six — enough to reduce a 97%-confident answer to well under 1%. Layers 10, 11, 13, 16, 17, 19, 20, 22, and 23 all produce damages in the 2-to-4 range. In contrast, ablating most individual layers in the 25-to-41 band barely moves the needle: many of them cost less than 0.3 log-probability, which is the model essentially not noticing the surgery.

This is the resolution to the puzzle from the last section. The first twenty-five layers aren't doing nothing — they're doing the bulk of the critical computation. The layers the logit lens *could* read are, it turns out, mostly refinement layers. The real work happens earlier, out of view.

There's an interpretation that fits. The middle layers are writing into the residual stream a representation that the late layers then decode. That representation doesn't look like output tokens, because it isn't trying to be output — it's the working state of a computation that's still in progress, a scaffold that the next layer will build on rather than a finished product. By the time the late layers get involved, they're taking that working state, finishing the computation, and projecting its result into vocabulary-space as a concrete token prediction. The logit lens, which is just "project this vector into vocabulary-space," only reads cleanly when the vector is *already close to* vocabulary-space. The mid-computation states are in a different part of the embedding space altogether.

Worth noting what this doesn't tell us. Ablation says *this component is necessary*, but it doesn't say *this component is doing X specific thing*. Layer 14's contribution is clearly critical for factual recall on these prompts. Whether it's doing the factual lookup itself, or assembling some prerequisite representation, or gating information flow — ablation alone can't say. We need finer-grained probes, which is what the next experiments provide.

---

## 5. Which Branch Is the Bottleneck?

Layer-level ablation is informative but coarse. Each of Gemma 4's forty-two layers contains two components that update the residual stream: the attention branch and the MLP branch. When ablating a whole layer produces dramatic damage, that damage is ambiguous — it could be attributable to either branch, or to both. The natural next step is to ablate them separately.

I run a prompt, and at a specific layer I either zero out the attention branch's contribution to the residual stream while letting the MLP run, or zero out the MLP branch while letting the attention run. Then I measure log-probability damage the same way as before. Forty-two layers times two branches times fifteen prompts is 1,260 ablated forward passes — about three minutes of wall-clock time on my laptop.

The result is unambiguous.

![Per-layer ablation of the attention branch (red) versus the MLP branch (blue), measured as log-probability damage to the model's own top-1 answer, averaged across 15 prompts. Lower bars are more damaging. The lower panel shows the difference between the two: positive values indicate MLP-dominance at that layer; negative values (only layer 23, really) indicate attention-dominance.](images/sublayer_ablation.png)

MLPs dominate almost everywhere. At layer 0, the catastrophic ablation from the last section turns out to be almost entirely about the MLP: ablating layer 0's attention costs about 3 log-probability points, while ablating its MLP costs about 17. The same pattern holds throughout the critical middle: layer 14's MLP ablation costs more than 9 log-probability points, while its attention ablation costs essentially nothing (+0.008 — statistical noise). Layer 9, layer 11, layer 12, layer 13, layer 16, layer 18, layer 19, layer 20, layer 22 — all of them show the same signature. The work in the middle of Gemma 4 is happening in the MLPs. The attention branches at those layers are largely decorative, at least for simple factual recall.

There is one striking exception. Layer 23 — the fourth of the seven global-attention layers — is the one place in the network where attention ablation does *more* damage than MLP ablation: about 5 log-probability points, compared to 2.3 for the MLP. The architectural reason is specific. Layer 23 is the last layer in the network that computes its own fresh attention over the full context; every layer past 23 shares its attention computation with earlier global layers rather than doing new work. So removing layer 23's attention cuts off a specific routing pathway that the late layers can't reconstruct. The rest of the time, MLPs carry the load.

This reframes what we learned in the last section. When I said layers 10–24 are where the critical work is happening, what I really meant, more precisely, is that *the MLPs* at layers 10–24 are where the critical work is happening. The attention branches at those layers are contributing almost nothing. Whatever attention is doing in Gemma 4, it is largely not the job of retrieving factual knowledge. That job is happening inside the per-position computations of the MLPs.

This raises an obvious question. If attention in the middle of the network isn't doing factual retrieval, what *is* it doing? Gemma 4 has those seven architecturally distinctive global-attention layers at positions 5, 11, 17, 23, 29, 35, and 41 — they use different head dimensions, different positional encoding, different parameter counts from the local layers. You don't include seven special layers unless they're doing something worth spending the compute on. What is it?

The next experiment was designed to find out.

---

## 6. What Attention Is Actually Looking At

The natural hypothesis, if you've read any recent mechanistic interpretability papers, is that attention heads retrieve facts by "looking at" the relevant token. When the prompt is *The Eiffel Tower is in* and the correct answer is *Paris*, you'd expect to find some layer — probably one of the global-attention layers — where the attention pattern at the final token position shows a big spike on the *Eiffel* or *Tower* positions. The idea is that the model is pulling the factual association in from the subject entity, where the MLPs in earlier layers wrote it. It's a beautiful story. It's also roughly what Meng et al.'s ROME paper argues for GPT-2. I went into this experiment expecting to reproduce it on Gemma 4.

To look, I had to extract the actual attention weights. Modern transformer implementations use a fused attention kernel that does the entire computation — query-key dot product, softmax, weighted sum over values — in one shot, and doesn't hand you the intermediate attention weights. To inspect them, you have to recompute attention manually: extract the queries and keys after their respective transformations and positional encodings, do the dot product, apply the attention mask, softmax, and you have a weights matrix. For every query position, that matrix gives you the distribution over key positions that this attention head is "looking at."

I did this at each of the seven global layers, for the Eiffel Tower prompt and for several other factual-recall prompts. Then I averaged the attention weights across the eight heads at each global layer and plotted what each layer's attention from the final token position looked like.

![Attention from the final token position at each of the seven global-attention layers, averaged across the eight heads, for the prompt "Complete this sentence with one word: The Eiffel Tower is in" (predicted answer: Paris). Red bars are layer 23, the layer where attention ablation was most damaging.](images/attn_pattern_0.png)

The prediction was not confirmed.

At every single global layer, across every prompt I tried, the attention from the final position is dominated by chat-template tokens: `user`, `<|turn>`, `<turn|>`, `model`, newlines, and `<bos>`. The subject-entity tokens — *Eiffel*, *Tower*, *Japan*, *Romeo*, *Juliet*, *gold* — receive attention weights roughly similar to every other content token, typically a few percent each. They are not singled out. The model is, in some evident sense, not "looking at the Eiffel Tower to predict Paris."

The pattern holds across all seven globals, though it shifts as you go deeper into the network. The early globals (layers 5 and 11) concentrate heavily on `<bos>`, `<|turn>`, and `user` — essentially the opening markers of the chat template. Layer 17 is the most distributed, with attention spread across many positions. Layer 23, the critical one, is bimodal: it attends strongly to template tokens at both the start of the sequence (`user`, `<|turn>`) and at the end (`<turn|>`, `model`, the final newline). The late globals (layers 35 and 41) concentrate almost all their attention on just the first three positions of the sequence.

Looking inside layer 23 at its eight individual heads rather than averaging, the picture sharpens: several heads are clearly "template-reading" heads concentrated on `user` and `<|turn>` at the sequence start, a couple are "turn-boundary" heads concentrated on `<turn|>` and `model` near the end, and the remaining heads spread attention more widely. None of them concentrates on the subject-entity tokens the way the factual-retrieval story would predict.

There was one exception worth chasing. Head 7 of layer 29 has measurably higher subject-entity attention than any other head in any global layer. Across several prompts, it attends to subject-entity tokens at a rate roughly equal to its attention on template tokens — the closest thing to a "content head" in the whole network.

![Subject-entity attention weight (left) and template-token attention weight (right) for every head of every global layer, averaged across six prompts. Each row is a global layer; each column is one of the eight heads. Layer 29 Head 7 has visibly higher subject-entity attention than any other position in the grid.](images/head_specialization_heatmap.png)

For a moment this seemed like bingo. So I ablated it alone — zeroing out only its single head's contribution to the residual stream while letting everything else proceed normally — and measured the damage.

Removing it barely mattered. The model still produced the correct answer on fourteen of fifteen prompts, and the log-probability dropped by about 0.01 — statistical noise. Whatever that head was doing with its subject-entity attention, it wasn't a bottleneck. Something else in the network was making the same information available. Attention heads in Gemma 4 are redundant at the individual level; removing any single one barely matters.

So factual retrieval in Gemma 4 isn't "attention copies from the subject position to the prediction position." The MLPs are doing the knowledge work (from the last section), and attention — at least at the global layers — is doing something structural: reading the chat template's boundary markers, probably as a way of keeping track of which part of the turn the model is currently generating.

This explains something that would otherwise be confusing: why the model is *attention-critical* at layer 23 (removing its attention costs five log-probability points) while simultaneously *not attending to the content tokens* at layer 23. The thing layer 23 is doing isn't "look at the Eiffel Tower." It's something about keeping the generation anchored to the correct point in the chat-template structure, ensuring the model knows it's producing a model-turn completion rather than continuing some other part of the text. That's load-bearing work — just load-bearing in a different way than factual retrieval would be.

---

## 7. The Side-Channel Is Doing Real Work

We're now left with a puzzle. MLPs do the knowledge-retrieval work. Attention, at least at the global layers, does structural work. But I introduced a third data pathway back in section 2 — the **MatFormer per-layer-input side-channel** — and so far it hasn't come up. What's *it* doing?

The side-channel is worth recalling. At each of the forty-two layers, alongside the main residual stream, a separate pathway pulls a small per-layer vector out of a dedicated embedding table (262,000 rows by 10,752 columns, to be specific), passes it through a linear gate and a nonlinearity, multiplies it element-wise with a projection of the residual stream, and adds the result back into the stream. It's visibly a secondary data path, and when I first read the code I assumed it was doing something minor — some modest conditioning signal, maybe, or a small positional adjustment.

So I ablated it. Zeroed out the gate's contribution at every layer simultaneously, and ran the same fifteen prompts.

![Top panel: log-probability damage from ablating the MatFormer side-channel across ALL layers simultaneously, plotted per prompt. Every single prompt collapses. Bottom panel: log-probability damage from ablating the side-channel at ONE layer at a time, averaged across prompts. Red bars mark global-attention layers; blue bars mark locals.](images/side_channel_ablation.png)

The damage is catastrophic.

Across all fifteen prompts, ablating the side-channel globally drops the log-probability by an average of 30 — nearly double the damage of ablating layer 0 (the single most important transformer layer we identified earlier). The model doesn't degrade to a less-confident version of the right answer. It doesn't degrade to a plausible near-miss. It produces garbage. For *The Eiffel Tower is in*, it says " St". For *The capital of Japan is*, it says " s". For *Water is made of hydrogen and*, it says " une". Every single factual prompt produces incoherent output. The side-channel is not a small auxiliary detail. It's one of the most load-bearing components of the entire network.

More interesting is *where* the damage concentrates. Ablating the side-channel at individual layers — one at a time instead of all at once — reveals that its effect is very unevenly distributed. Four of the five most-affected layers are global-attention layers: layer 17 (ablation costs 5.2 log-probability points), layer 11 (2.8), layer 23 (2.4), and layer 29 (1.2). The one non-global in the top five is layer 15, a local layer sitting between globals 11 and 17. Meanwhile, at most local layers outside that middle band, ablating the side-channel does essentially nothing. The side-channel is not a uniformly-applied signal. It is specifically load-bearing at the global-attention layers and the local layers immediately adjacent to them.

This puts the pieces together.

The global-attention layers don't attend to content tokens. We saw that in the last section. They attend to chat-template structure — `user`, `<|turn>`, `<turn|>`, `model`, newlines. But a computation that only sees structural markers can't, by itself, be grounded in the specific token content of the prompt. The model needs the factual content to factor into whatever the global layer is doing, or the whole computation falls apart. If the global layers aren't seeing that content through their attention patterns, it has to be getting in somewhere else.

The side-channel is that somewhere else. At every layer — including every global layer — the per-layer embedding table injects a small signal that is specifically conditioned on each token's identity. It doesn't depend on attention patterns or on the current contents of the residual stream; it's a direct, per-token injection of *here's what this position actually is* into the computation. For the local layers, this signal may be largely redundant, because their sliding-window attention already gives them rich access to nearby token content. For the global layers, attending over the entire sequence with limited attentional bandwidth and concentrating that bandwidth on structural markers, the side-channel is apparently how the model ensures that token identity keeps flowing into the computation alongside the structural information that attention is routing.

This is, as far as I can tell from the public literature, a genuine architectural finding. The MatFormer per-layer-input side-channel isn't a small auxiliary detail. It's a specific load-bearing mechanism that allows the global-attention layers to do their structural work without losing track of token content. Google may well have known about this when they designed it — I'd expect the team to have some idea why they included an expensive auxiliary pathway — but it isn't documented anywhere I've been able to find, and the public discussion of the MatFormer structure doesn't mention the role it plays in supporting global attention.

---

## 8. Finding Where the Information Lives

So far, the investigation has told us a lot about *which components* in Gemma 4 matter. MLPs in layers 10–24 are critical. Attention is mostly about structural routing. The MatFormer side-channel is load-bearing, disproportionately at the global-attention layers. But none of this tells us *where in the forward pass* the factual information actually lives — at which token positions, at which depths, the causal content of *Eiffel Tower → Paris* is being carried.

For that we need a different probe: **causal tracing**, sometimes called activation patching. The idea is surgical rather than destructive. You run two prompts through the model. Prompt A (clean) is *The Eiffel Tower is in*, which the model answers with *Paris*. Prompt B (corrupt) is *The Great Wall is in*, which the model answers with *China*. These prompts are structurally identical except for the subject, and they tokenize to the same length, so positions align. Cache every intermediate activation from the clean run.

Now run the corrupt prompt, but with one specific intervention: at a chosen (layer, position), substitute the clean activation from that same (layer, position) in place of whatever the corrupt run would have produced there. Let the rest of the forward pass proceed normally. Measure the probability the corrupt run now assigns to *Paris* — the clean answer.

If nothing changes, the location you patched was carrying no causal information relevant to distinguishing the two prompts. If the corrupt run flips fully from *China* to *Paris*, you've identified a location that, by itself, was sufficient to redirect the model's answer. The factual information for *Paris vs China* is localized at that point in the computation.

Running this experiment exhaustively — every one of the forty-two layers at every position in the sequence — produces a heatmap: for each (layer, position) cell, the probability the corrupt run assigns to *Paris* after that single clean activation is patched in.

![Causal tracing for the Eiffel Tower / Great Wall prompt pair. Left panel: probability of "Paris" in the corrupt run after patching a single clean activation at each (layer, position). Right panel: recovery, meaning the improvement over the unpatched corrupt-run baseline. Red dashed lines mark subject-entity positions. Dark green means patching that one activation was sufficient to restore the clean answer. Empty regions mean the activation at that location carried no causal information about the answer.](images/causal_trace_0.png)

The result is strikingly sparse. Across 21 token positions × 42 layers = 882 cells, essentially two regions light up.

The first region is the **subject position in the early-to-middle layers**. For the Eiffel Tower prompt, patching the clean activation at the " Wall" position (position 13 in the corrupt prompt) at any layer from 0 through 12 almost fully restores *Paris* as the model's output. Above layer 12, patching the subject position no longer recovers. The factual information was at that position, in that layer range, and then it moved.

The second region is the **final token position in the late layers**. At layers 30 through 41, patching the final-position activation alone fully recovers the clean answer. By the time the model reaches layer 30, the answer is localized at the final position — and patching any earlier position does nothing.

Everything else on the heatmap is empty. Template tokens, the *is in* connector, the turn delimiters, the content tokens elsewhere in the prompt — patching any of those, at any layer, has no effect on the output. The factual information for *Paris vs China* doesn't live there.

This is the classic picture from the causal-tracing literature on GPT-2 (Meng et al., *Locating and Editing Factual Associations*). Facts localize at the subject position in early-to-middle layers, move to the final position in late layers, and everything else in the forward pass is in some sense scaffolding. I reproduced the same two-hotspot signature on three paired-prompt setups (Eiffel Tower / Great Wall, Japan / France, Romeo and Juliet / Pride and Prejudice). The specific range of middle layers where the subject position is still causally sufficient varies with the prompt, but the overall shape is always the same.

There's one more thing worth noting, because it closes a loop left open in section 3.

The subject-position hotspot in the middle layers is *causally determinative* — patching the clean activation there restores the clean answer. But the residual stream at that location is not vocab-decodable. The logit lens, projected through the output head at the subject position in a middle layer, produces noise. You can't read *Paris* off the " Tower" position at layer 10 in the clean run; the top decoded tokens are gibberish. And yet, swapping that gibberish in place of the corresponding gibberish in the corrupt run is enough to shift the model's final answer from *China* to *Paris* with near-certainty.

This is the cleanest demonstration I know of that a transformer's internal representations are real computational content — they determine outputs — even when they don't look like anything in the output vocabulary. The middle-layer residual at the subject position is a program-fragment, the scaffold for an ongoing computation. It's not trying to be output, so it doesn't look like output. But it's the substance of what the model is doing. The logit lens missed it not because nothing was there, but because the lens was asking the wrong question.

---

## 9. A Specific Picture, and What to Make of It

Here is what the investigation assembled into, end to end. When Gemma 4 processes a factual-recall prompt, the first stage of the computation happens at the subject positions in layers 0–24. The MLPs at those positions write a representation into the residual stream — vocab-opaque, not decodable through the output head, but causally determinative. That representation is the model's knowledge work. Attention at those layers contributes little; the real action is per-position, inside the MLPs. By around layer 20 the information begins to move. Through layers 24–30, attention (particularly at global-attention layer 23) routes the subject-position scaffold toward the final token, using the chat template's structural markers — `user`, `<|turn>`, `model` — as anchors for where the scaffold should land. The global-attention layers during this handoff do not attend to content tokens. They attend to structure, and they lean on the MatFormer side-channel for the per-token identity grounding that keeps their computations tied to the specific content of the prompt. By layer 30 the factual information is localized at the final position, and the remaining layers project it into vocabulary-space as the model's actual output. The logit lens can read the answer starting around layer 27; by layer 41 it is the confident prediction.

The headline finding is about the MatFormer side-channel. Ablating it across the entire network is roughly twice as destructive as ablating the single most important transformer layer. At individual global-attention layers, removing just the side-channel's gate is comparable to removing the whole MLP at that layer. It is the specific mechanism that supports global attention's structural role by injecting per-token identity into the computation. The public documentation does not mention this role, and I have not seen it described in the interpretability literature. It is worth flagging as a genuine architectural finding about the Gemma 4 family, and probably about any transformer with a similarly structured auxiliary pathway.

A few other observations are worth noting briefly. The logit lens fails on Gemma 4's middle layers in a specific, characteristic way — the residual stream at those depths is not close to vocabulary-space, even though it causally determines the output. The lens alone would suggest that layers 0–24 are nearly inert. Ablation says the opposite. The lens is a real tool, but it has a specific blind spot, and Gemma 4's middle layers sit inside it. Individual attention heads are largely redundant: even the one head with measurably more subject-entity attention than any other in the network (layer 29, head 7) can be removed without meaningful damage. Interpretability claims of the form *"head X does Y"* should be treated skeptically without ablation evidence. And at the global-attention layers, attention is almost entirely directed at chat-template structural markers rather than at content. Whatever the global layers are doing, it is not retrieving facts. That job lives in the MLPs.

Where this could go next: the most obvious follow-up is cross-model validation. Do auxiliary pathways structurally similar to the MatFormer side-channel exist in Llama, Qwen, or Mistral, and are they doing similar work? If so, the finding generalizes into a broader architectural lesson. If not, it is a specific feature of Gemma's design, still worth documenting. A second direction is probing what information actually flows through the side-channel — whether it is token identity as I've conjectured, or positional information, or something more abstract. A third is extending the picture to prompt types beyond simple factual recall. The mechanisms found here are specific to a particular kind of computation; they may not generalize.

One last observation, earned. What I take away from this weekend is how much there is to find inside a small language model if you are willing to poke at it. Gemma 4 E4B is tiny by modern standards, and every experiment in this essay ran comfortably on a laptop. The MatFormer side-channel finding alone would have justified the time. The supporting observations about attention structure, redundancy, and vocab-opaque internal representations would each have been worth a weekend on their own. These systems are not black boxes, and they are not inscrutable. They are very large, very complicated, and full of structure that can be read out, one piece at a time, if you are patient with your probes.

---

## 10. A Question Left Open

That was the end of the weekend. But section 8 left an open question that nagged at me afterward. The middle-layer residual at the subject position is *causally determinative* — patching it from a clean run into a corrupt run is enough to flip the model's answer — but it is *not vocab-decodable*. The logit lens at that location returns gibberish. So we know that representation matters; we know it carries the factual information; we just can't read it.

What is it actually like, then? "A program-fragment, the scaffold for an ongoing computation," I wrote. That was a placeholder for ignorance dressed up as insight. The next stretch of work was an attempt to do better.

---

## 11. The Geometry Has a Shape

The right tool turned out to be borrowed from word2vec rather than from any specifically transformer-aware probe.

I built a small labeled corpus: forty factual-recall prompts in five clean relational categories — eight prompts each for capital lookups (*the capital of France is*), chemical-element symbol lookups, author-of-book lookups, landmark-location lookups, and opposite-word lookups. For each prompt, I extracted the residual stream at the last subject-entity token at three depths: layer 5 (pre-engine-room), layer 15 (in the middle of the critical band), and layer 30 (post-handoff, after the answer has moved to the final position).

The question was simple: do these forty 2560-dimensional vectors have any structure at all that you can detect without supervision?

They do, and the structure is dramatic. By layer 15, the five categories form perfectly distinct clusters under unsupervised k-means. Every prompt's nearest neighbor in cosine space is from its own category — *France* nearest *Italy*, *gold* nearest *silver*, *Juliet* nearest *Prejudice*, *Tower* nearest *House*, *hot* nearest *fast*. The k-means purity is 1.000. The vocab-opaque representations from the last section are not opaque to *each other*; they are organized into a clean categorical geometry that the unembed projection just happens to flatten away.

This is the same kind of structure word2vec famously found in word embeddings — semantically related words cluster, related word groups form identifiable clusters, the residual differences between related items can be small but consistent. Except here the structure isn't in the input embeddings; it's in the working internal representations of a transformer mid-computation, at a position the model isn't yet ready to commit to a token at. The mid-layer residual at the subject position encodes what *kind of question* is being asked.

Then a follow-up experiment turned out to be the technique that justified the whole line of work. Take the centroid of one category — the average of the eight subject-position vectors for capital prompts, say — and project that centroid through the model's own tied unembed. Subtract the overall corpus mean first, so you're not just decoding the prompt-template common-mode that all the vectors share.

What comes out is a list of tokens, in many languages, that name the relational frame the category captures. The capital-category centroid at layer 30, after mean subtraction, decodes to: ` தலைநக` (Tamil for capital), ` city`, ` embassy`, ` เมือง` (Thai for city), ` राजधानी` (Hindi for capital), ` ialah` (Malay for *is*), ` capitale` (Italian), ` Hauptstadt` (German). The element-category centroid decodes to ` chemical`, ` atomic`, `原子` (Chinese for atom). The opposite-category centroid decodes to ` reverse`, `反対` (Japanese for opposite), `反` (Chinese), ` backward`. Each category's centroid decodes to a multilingual cluster of tokens that names the relational operation the category instantiates.

I scaled this to twelve categories and ninety-six prompts, including more diverse relational frames (morphology, translation, profession, animal habitat, color mixing, arithmetic). The clustering remained perfect; the centroid-decoding pattern held for every category. A random-subset baseline confirmed that the multilingual decodings depend on actually averaging within a category — averaging eight randomly-sampled prompts from across the corpus gave noise.

![Pairwise cosine similarity matrix at layer 30 across 96 prompts in 12 categories, reordered to show the block-diagonal structure. Each block on the diagonal is one category; the strong intra-block coloring shows perfect within-category clustering. Off-diagonal cells show that prompts from different categories are still quite similar to each other — the categorical signal lives in small but consistent deviations within a high-baseline-similarity space.](images/big_sweep_similarity_heatmap.png)

The technique is methodologically clean: nothing trained, no probes fit, no labels used during analysis. The model's own output head decodes its own internal categorical prototypes. Whether anyone has done exactly this before in the public literature I'm not sure; the closest analogues I've found are unsupervised steering-vector extraction methods, which are more mechanistic but less semantically transparent.

---

## 12. A Confound, and What Survived

Initially I framed this finding as evidence that the model's *internal cognition* was multilingual — the same intermediate computation that processes "the capital of France" lights up tokens for *capital* in eight languages, suggesting the operation is being represented in a language-agnostic conceptual space.

That framing didn't survive scrutiny. Multilingual models are trained on multilingual text, and during training the tied embedding learns to align cross-lingually equivalent tokens to nearby positions in embedding space. Once that alignment exists, *any* vector pointing toward the "capital concept" region of embedding space will decode to all those tokens — not because the mid-layer activation is doing anything multilingual, but because the *output head* is. The multilingual decoding is downstream. The interesting question is what's *upstream*.

Stripped of the multilingual framing, what's left is a more modest claim that's still worth defending: mid-layer subject-position activations encode the *kind of operation* the prompt is asking the model to perform, in a way that's separable from both the specific operand (which country) and the specific answer (which city). Same operation, different operands, cluster together. Same answer arrived at via different operations, do not.

But this claim has an obvious confound. Every capital prompt contains the literal word *capital*; every element prompt contains *chemical symbol*; every author prompt contains *was written by*. Maybe the cluster signal isn't about the abstract operation at all. Maybe the model is just tracking which surface tokens appeared. The mid-layer geometry might be a sophisticated lookup of "what tokens did I see in the prompt" rather than anything about cognition.

Disambiguating this calls for paired prompts that vary operation-type and operation-word-presence independently. I built four cohorts of eight prompts each, in a 2×2 design. **A1**: capital-lookup with the word *capital* present (the original *the capital of France is* prompts). **A2**: capital-lookup, paraphrased so the word *capital* is absent (*the administrative center of France is named, in one word,*). **B1**: a different operation — letter-counting — with *capital* in the operand (*the number of letters in the word capitalism is*, on eight different *capital*-containing operand words). **B2**: same operation, different operand words that don't contain *capital* (*the number of letters in the word elephant is*).

Then ask: which dimension does the mid-layer geometry separate the prompts along? If it tracks the operation-type, A1+A2 cluster and B1+B2 cluster, separately. If it tracks the surface word, A1+B1 cluster and A2+B2 cluster. If it's mixed, both groupings have some signal.

![Mid-layer subject-position vectors for the 32 disambiguation prompts at layer 30, projected to 2D via PCA, colored two ways. Left: by operation-type. Two clean clusters — lookup prompts on the right (further subdivided by template into A1 top, A2 bottom), counting prompts on the left. Right: same data colored by 'capital'-word-presence. The colors mix freely within both major clusters; the word's presence in a prompt has essentially no effect on which region the prompt's mid-layer representation occupies.](images/operation_disambiguation.png)

The result is decisive. Operation-type beats word-presence by 6.35× in cosine separation. K-means under the operation-type label gives perfect purity (1.000). K-means under the word-presence label is at chance (0.500). Nearest-neighbor agreement is 100% under operation-type, 90.6% under word-presence. The picture is unambiguous: the mid-layer subject-position activation tracks what kind of cognitive operation the prompt has primed the model to perform, not whether some particular content word happens to appear.

The factorization claim survives the obvious confound. The geometry from section 11 was real, not a sophisticated form of "did *capital* appear in the prompt." There is some softer secondary effect — within the lookup cluster, A1 and A2 form sub-clusters that mostly track the prompt template — but that's a smaller signal living on top of the dominant one, not an alternative explanation for it.

---

## 13. Reading versus Writing

The natural next experiment was a steering test. If the capital-category centroid is a clean encoding of "the model is preparing to perform a capital lookup," can it be *used* as a function pointer? Take the centroid `v_capital`, add it to the residual stream of an unrelated prompt, and see whether the model starts producing capital-city tokens.

This is the standard activation-steering setup. If it worked, it would convert the centroid from a *diagnostic* (it tells you what the model is doing) into an *intervention* (it makes the model do something). The proposal that suggested this experiment framed the strong claim with a programming metaphor: mid-layer activations might be *callable handles* for cognitive operations, not just labels.

I tested two versions of `v_capital`: the raw centroid, and the mean-subtracted centroid (the same form that decoded cleanly to multilingual concept tokens in section 11). Three target prompts: a neutral prompt with no specific operation cued (*the following country is famous for its*), a different-operation prompt (*the past tense of run is*, where the model produces *ran* with high confidence), and a same-operation control (*the capital of Germany is*, where the model already produces *Berlin*). I swept the injection scale from zero up to five times the natural magnitude of the steering vector.

The result is a clean null. At the neutral prompt, the probability of any capital-city token across all eight anchor cities never rose above zero. At the past-tense prompt, the probability of capital-city tokens also stayed at zero — and at high injection scale the model's output collapsed to a continuation token like ` is` rather than redirecting to *Paris* or *Tokyo*. The mean-subtracted vector was much more graceful than the raw one (it didn't destroy the past-tense answer until very high scale, at which point the model collapsed to the asterisk character `**`), but it was just as null on the steering goal.

![The injection sweep. Top row uses the raw `v_capital` (norm 131); bottom row uses the mean-subtracted `v_capital` (norm 39). Three columns are the three target prompts (neutral, past-tense, control). The y-axis is the total probability mass on capital-city tokens at the final position. In the neutral and past-tense conditions, the curve never lifts off zero under either vector. In the control condition (where the answer was already a capital city), the curve collapses as the injection magnitude grows — the model's output gets destroyed, not redirected.](images/representation_injection.png)

The honest reading is that centroids are diagnostic probes, not function pointers. You can read what cognitive operation the model is preparing to run from its mid-layer state with high accuracy. You cannot use that same mid-layer signal as an injection to *make* the model run that operation in a different context. There are caveats — naive single-position single-layer injection is the simplest possible steering test, and the literature uses richer designs — but the simple test gives a clean simple answer: no.

There's a useful philosophical distinction lurking here. In information terms, the centroid carries the operation; in computational terms, it doesn't *invoke* the operation. The geometry of activations encodes what cognition is happening, but the cognition itself is presumably implemented by the joint state of every layer's attention and MLP weights, conditioned on a long history of position-by-position residual updates that no single steering vector can substitute for. Reading and writing are genuinely different operations on a transformer's internals. Conflating them is the sort of mistake that happens when you start thinking in metaphors that work too well.

---

## 14. Within a Single Token

The geometry result in section 11 was about *cross-prompt* differences: prompts using the same kind of operation cluster together at the same position. A natural extension is to ask the same question *within* a single token. If the model has multiple senses for one word, does the residual stream at that word's position encode which sense is meant?

The English word *capital* is unusually well-suited to this test because it has at least four robust senses. *The capital of France* uses it as a noun for a city. *Raised capital from investors* uses it as a noun for money. *Capital letters* uses it as an adjective for uppercase. *Capital punishment* uses it as an adjective for grievous-offense related. None of these senses are derivative of the others; they're four distinct meanings sharing a written form.

I built eight prompts per sense, with varied sentence templates within each cohort so template structure wouldn't systematically distinguish them. For every prompt, I extracted the residual stream at the *capital* token's position itself (not at some other operand position) at every one of the forty-two layers. The question: at what depth, if at all, does the model's representation at the *capital* position separate the four senses?

At layer 0 it doesn't, because at layer 0 the residual at every *capital*-containing prompt is essentially the same vector — it's just the embedding for *capital*. By layer 4, the four senses are 84% nearest-neighbor-separable: pick any prompt's *capital*-position vector at layer 4 and its closest cosine neighbor among the other 31 vectors is from the same sense more often than not. Sense disambiguation is fast.

The geometric clustering peaks at layer 12 (silhouette score +0.34, well above the threshold for "obviously distinct clusters"), then *softens* through layers 13–25, then sharpens again at layers 33–38 (silhouette +0.23). Two peaks, not one, at the same two depths the original investigation identified as the engine room and the readout band. The same two-phase architecture that organizes the model's factual-recall computation in earlier sections also structures its sense-disambiguation work for this single homonym.

The centroid-decoding test on these sense vectors produced the cleanest result in the project. At layer 41, the punishment-sense centroid decodes through the unembed to ` punishment` at probability **0.802**. The uppercase centroid decodes to ` letters` at probability **0.594**. The city centroid spreads more thinly across multilingual city tokens (` city`, `'shahr'` Hindi, `' cidade'` Portuguese, `' городе'` Russian, `' مدينة'` Arabic, `' ciudad'` Spanish). The finance centroid concentrates on financial vocabulary (` funds`, ` ratios`, ` Funds`, ` investments`).

These decoding probabilities are an order of magnitude tighter than the multilingual category-centroid decodings from section 11 (where the strongest single-token signal was around 0.10). The cross-prompt operation centroids spread their signal across many semantically-related tokens; the within-token sense centroids concentrate it on a few dominant ones. Whatever's making the sense representation cleaner than the operation representation is something I haven't fully understood yet — possibly that within-token disambiguation is a more constrained problem with a smaller answer space, or that the readout layers are specifically optimized for vocabulary-direction projection of single-token meanings.

![Two-panel figure. Left: silhouette score for the 4-sense clustering at every layer, showing the two-peak depth profile. The first peak is at layer 12 in the engine room; the second is around layer 35 in the readout. Right: PCA projections at five selected depths (layers 0, 10, 20, 30, 41), colored by sense. At layer 0 the points are jumbled; by layer 10 the four sense regions are visible; by layer 41 they're tighter still and the centroids project cleanly to sense-relevant tokens through the unembed.](images/homonym_capital_pca_grid.png)

What this confirms: the factorization story — mid-layer residuals encode "what kind of thing is being processed here", separately from the specific surface tokens — generalizes from cross-prompt operation distinctions down to within-token sense distinctions, using the same machinery and following the same engine-room-then-readout timing. The model's residual stream really does carry richly structured semantic information in geometrically separable form, and that information becomes vocab-decodable specifically at the readout layers as the network projects its working state into the output direction.

The steering null from section 13 still applies. Reading what sense the model has assigned to *capital* in a given context is straightforward; injecting a sense centroid into a different context to *change* the model's sense assignment was not tested here, but I'd bet it would fail for the same reasons.

---

## 15. A Closing Picture

Two arcs of work, one essay. The first arc was about *which components* of Gemma 4 do the heavy lifting for factual recall: MLPs in the middle layers, the MatFormer side-channel, and global-attention layers doing structural rather than content work. The second arc was about *what the heavy lifting produces*: vocab-opaque mid-layer representations that turn out to have rich, geometrically separable categorical structure when you look at them with the right tool.

The combined picture is more satisfying than either half alone. Section 9 ended with the open question of what a "vocab-opaque program-fragment" actually was. Sections 11–14 give a partial answer: it's a vector that lives in a region of activation space corresponding to the cognitive operation the model is preparing to run, organized into clusters that you can recover unsupervised, with centroids that decode through the model's own output head to multilingual concept tokens — and that gets refined into the same kind of structure for distinguishing senses of homonyms within a single token's representation. The "scaffold for an ongoing computation" was a placeholder; the actual scaffold has shape, and the shape is read-able.

But "read-able" turned out to mean something specific. Section 13's null result — that the centroids do not steer model behavior when injected into other prompts — sharpens what the geometry means. Mid-layer activations carry richly structured information about what computation is happening; that information is encoded in the geometry of the residual stream; you can recover it unsupervised; you can decode it through the unembed. None of that makes the activations into causal levers you can push to change what the model does. Diagnostic ≠ functional. The model's cognition is implemented by the joint structure of every weight in the network conditioned on the entire prompt history, and a single-position single-layer additive injection of a centroid does not approximate that structure closely enough to invoke its associated operation.

A small parallel observation, mentioned briefly: the readout layers also do something that section 9 didn't fully describe. The very last transition in the network — layer 40 to layer 41 — turns out to be doing tokenization calibration rather than semantics. On 11 of 14 factual-recall prompts, the model's top-1 token shifts at that final transition from a space-prefixed variant (` Paris`, ` Tokyo`, ` Shakespeare`) to a no-space variant (`Paris`, `Tokyo`, `Shakespeare`) — never a semantic change, always a leading-space drop. The model decides what word to say by around layer 37; the last few layers convert the natural plain-text continuation form to the no-space form that's correct for the chat template's start-of-line model turn. This is finding 14 in the project's record, and it's the kind of small surface-level mechanic that would have been invisible if I hadn't been tracking rank-1 changes layer-by-layer for an unrelated reason.

The final takeaway is methodological. Single-tool interpretability gives partial pictures: the logit lens missed the engine room entirely; ablation told us what was important without telling us what it was doing; attention-pattern inspection misled by suggesting the wrong heads were the bottleneck. The combinations did better. Logit lens *plus* ablation revealed the visibility/causality discrepancy that motivated everything that followed. Causal tracing *plus* logit lens revealed that mid-layer subject-position activations were causally specific but representationally opaque. Centroid decoding *plus* the model's own tied unembed turned out to be the thing that finally read those opaque representations into something interpretable, and revealed the categorical structure that organizes them. None of these techniques individually answered the question "what is the model doing"; the layered combination of them did.

I started this expecting to write up a single architectural finding about an undocumented sub-circuit. The architectural finding is real — the MatFormer side-channel is genuinely load-bearing, in a specific way the public literature doesn't mention, and it's worth documenting. But the more interesting outcome is the picture that built up around it: a small open-weight model whose internals can be read out, layer by layer and category by category and even sense by sense, with a small handful of techniques and a few weekends of patient probing. There's a lot in there. Most of it is gettable.

---

## 16. Mechanism Beneath the Homonym Geometry

Section 14 established that within-token sense disambiguation has the *same shape* as the cross-prompt operation factorization from section 11 — same engine-room-then-readout depth profile, same kind of clean centroid decoding. What it didn't establish was *which components* of the network do that work. The first arc of the essay built a specific picture for factual recall: MLPs in the middle, side-channel load-bearing especially at globals, layer 23 as a structural attention hub. The natural test is whether the homonym story uses the same components in the same roles, or whether it borrows different machinery for what looks superficially like the same kind of computation.

Two probes, two predictions, both wrong in informative ways.

**The first probe was the side-channel.** Section 7's headline was that the MatFormer per-layer-input gate is essentially per-token identity, injected at every block. For homonym sense disambiguation, the *capital* token is identical across all four sense cohorts — same surface form, same token ID, same per-layer embedding. So the side-channel input *at the* capital *position* is the same regardless of sense. The natural prediction was that ablating the side-channel everywhere would barely affect sense separability; the disambiguating signal has to be reaching the *capital* position from the surrounding context, and that context arrives through the residual stream and attention, not through a position-local side-channel.

That isn't what happened. Ablating the side-channel at all 42 layers drops nearest-neighbor purity at layer 41 from 0.875 to **0.250** — exactly chance for a four-class problem. Silhouette goes from +0.119 to *negative*; the four sense clusters don't just loosen, they collapse onto each other. But at layer 12, the geometric peak from section 14, the clusters survive almost intact: silhouette drops by about a quarter, and nearest-neighbor purity is unchanged at 0.844.

![Side-channel ablation collapses sense disambiguation at layer 41 but mostly preserves it at layer 12. Top row: PCA of *capital*-position residuals at layer 12, baseline (left) versus side-channel ablated (right). The four sense clusters loosen but remain identifiable; nearest-neighbor purity is unchanged. Bottom row: same at layer 41. Baseline shows clean cluster structure; ablated shows complete collapse, with NN purity at chance for four classes.](images/homonym_side_channel.png)

So the side-channel *isn't* needed to *form* the sense-disambiguation geometry. The early-to-mid layers can do that work fine without it, presumably by attending to context tokens whose side-channel inputs *do* differ between cohorts. What the side-channel is needed for is *preserving* that geometry through the late-layer transform from L12 to L41 — the same transform that section 14 already noted does not improve geometric separability but does make the centroids decodable through the unembed. Without the side-channel, the late layers undo the engine-room work rather than compressing it into vocabulary-shaped output. The "per-token identity" framing from section 7 was correct as far as the input goes, but understated what that identity-grade input does once you compose it through forty-one layers of attention and MLP.

**The second probe was per-layer ablation, designed to identify a single causal peak for sense disambiguation.** Section 14 showed silhouette peaks at layer 12 in the engine room and again at layer 35 in the readout. The natural hypothesis is that the geometric peak corresponds to a causal peak: ablating layer 12 should drop sense separation more than ablating any other single layer.

That hypothesis fails too, in two directions. At the L12 readout, the most damaging ablations upstream of L12 are layer 0 (silhouette drops by 0.41 — catastrophic), then layer 1, then the early globals at layer 5 and layer 11. Layer 12 itself is only the *sixth* most damaging upstream layer. The geometric peak is where the construction *culminates*, not where any single layer does most of the work. The computation is distributed across the early-to-mid layers, anchored on the input-embedding contribution, with the early global layers playing larger roles than the engine-room locals between them.

The more striking finding is at the L41 readout. Across all 42 ablations, the single most damaging is **layer 23** — the same engine-room global that section 5 fingerprinted as the only attention-critical layer for factual recall, and that section 7 identified as a side-channel hotspot. Two independent experiments, on entirely different problems (factual recall in the first arc, homonym sense disambiguation in the second), now both single out layer 23. Whatever that layer is doing, it is doing it for more than one downstream task. The picture from section 9 of layer 23 as a "structural attention hub" was a reasonable guess; it now looks more like a structural attention hub *with general-purpose downstream consequences*, not just a piece of machinery specific to the factual-recall pipeline.

![Per-layer ablation of all 42 layers, measuring sense-cluster silhouette (top) and nearest-neighbor purity (bottom) at the *capital* position at two readout depths: layer 12 (blue) and layer 41 (red). Dashed horizontal lines mark the no-ablation baseline at each readout. Vertical dotted lines mark the seven global-attention layers. Layers downstream of the readout (right of L12 in the blue series, right of L41 in the red) produce zero effect — a sanity check that the framework's ablation and capture composition is correct. Layer 0 is catastrophic at both readouts. Layer 23 is the largest single hit at the L41 readout — the same layer fingerprinted in section 5.](images/homonym_layer_ablation.png)

A small parting observation. Ablating layer 40 or layer 41 — the very last hidden layers — *improves* sense-cluster silhouette at L41 by a small amount. That fits the picture from section 14 of the late layers trading geometric structure for vocabulary-readability: removing them cleans the geometry slightly because their job was to slightly degrade it on the way to a vocab-shaped output. The same observation holds in the side-channel experiment: the geometry at L12 mostly survives the side-channel ablation, but the L12-to-L41 transform that should *decompose* that geometry into a token distribution is the part that breaks. Two probes, same boundary: the engine-room sense representation is robust; the readout transform is fragile, and depends on a specific combination of side-channel input and engine-room global routing through layer 23 to do its job without destroying what it's reading from.

This is a partial picture. Three more probes from the first arc are still un-ported to the homonym question — sub-layer (MLP-versus-attention) ablation, attention-pattern inspection at the *capital* position, and causal tracing on minimal sense-pair prompts — and each could change the story. But the through-line so far is that the same components keep showing up. Layer 23 in particular is now triple-confirmed (factual-recall attention bottleneck in section 5, side-channel hotspot in section 7, sense-disambiguation late-readout bottleneck here). Whatever it is doing, it is one of the load-bearing pieces of this network in a way that does not depend on any specific task.

---

## 17. The Same Geometry, at a Different Grain

Section 11 found categorical structure at the mid-layer subject-token position: vectors for *capital-of* prompts clustered; vectors for *chemical-symbol* prompts clustered; each category's centroid decoded through the tied unembed to multilingual tokens naming the relational frame. The technique was constrained to prompts whose template made a single subject token a clean carrier of the categorical signal. Section 14 extended it to a tighter grain (the four senses of a single homonym, still at a single token position). The natural next question is whether it extends in the *other* direction — to concepts distributed across a passage, not localized at any one token.

Anthropic's recent paper on emotion concepts in Claude (Sofroniew et al., *Emotion Concepts and their Function in a LLM*, 2026) is a close cousin of our categorical-geometry technique at exactly that grain. The structural recipe is the same — difference of means in activation space, then read through the tied unembed — but the pipeline differs in three specific ways. First, the input is a short passage of prose (a one-paragraph story) rather than a templated prompt. Second, the extraction pools the residual stream across all token positions from the 50th token onwards in each passage, rather than picking one subject token. Third, after computing the difference-of-means vector, they project out the top principal components of an emotionally-neutral baseline corpus (enough to explain 50% of neutral variance), to remove activation-space directions that are high-variance across any text rather than specific to emotion. Their scale is 171 emotions × 1,200 passages each; the extraction and denoising are the interesting pieces, and the rest is throughput.

I ported the recipe to Gemma 4 at a hundredth of the scale: 96 hand-curated short passages across six emotions spanning the valence/arousal plane (happy, sad, angry, afraid, calm, proud), 16 passages each, plus 16 emotionally-neutral passages (laundry schedules, bus routes, form instructions) for the baseline. Extraction at layer 28 — about two-thirds through E4B's 42 layers, following the paper's recommendation — mean-pooled over token positions from 20 onwards. Six probes, one per emotion. The whole thing was ninety seconds of compute plus an evening of writing passages.

Scored against the training passages themselves, the result is a cleaner diagonal than the categorical-geometry technique ever produced:

![Two-panel result for the emotion-probe self-consistency test at Gemma 4 E4B layer 28. Left: aggregated probe scores (rows = true emotion, columns = probe). Every diagonal cell dominates its row by a factor of 2 to 15 over the best off-diagonal. Right: per-passage scores grouped by true emotion, showing the same diagonal pattern at individual-passage resolution. Off-diagonal cells carry real structure, especially the calm-vs-angry antipode and the happy-vs-proud shared valence.](images/emotion_probes_diagonal.png)

Aggregated diagonal hits: 6/6. Per-passage top-1 accuracy: 90.6% on six-way classification from sixteen training examples per class, with no supervised training. The self-consistency test is weak evidence on its own — probes should score highest on the corpus they were built from, by construction — but the *magnitude* of the diagonal is surprising for this little data.

The part that's more interesting than the diagonal is the **off-diagonal structure**. The non-zero cross-scores are not noise; they encode the valence/arousal plane that emotion theorists use to organize affective concepts:

- **Happy** and **proud** both score *positively* on each other's probe (+2.0 each direction). They are the only pair of positive-valence emotions in the set, and they're the only pair that share a positive cross-score.
- **Angry** and **afraid** score *positively* on each other (+1.1 to +1.3). Both are high-arousal negative-valence emotions; their directions partially overlap.
- **Calm** and **angry** score *strongly negatively* on each other (−4.6 and −5.4). They are the two most opposite points in the valence/arousal plane — low-arousal-positive versus high-arousal-negative — and their cross-scores reflect that diameter.
- **Sad** is asymmetric. Its cross-scores with everything else are modest (within ±2.3). Sad doesn't have a clean opposite in this corpus (the opposite of sad isn't any of the other five, really — it's whichever of calm or proud you mean, depending on what axis you mean); its row and column are correspondingly the flattest.

None of these axes were asked for. Nobody told the model's residual stream that happy and proud should share a valence dimension while angry and afraid share an arousal dimension. The axes fell out of a difference-of-means computation applied to a hundred passages and cleaned up with one PCA step. Whatever Gemma 4 learned during pretraining about emotional content in English text, it learned something structurally close to affective psychology's standard two-dimensional model.

One thing this result doesn't establish, which is worth naming. The self-consistency test shows each probe scores highest on its own training corpus, which is the trivial thing probes should do. It doesn't test whether the probes have learned the *concept* or the *corpus template*. A sharper test is whether the *happy* probe activates on a passage that evokes happiness without using any of the happy corpus's surface vocabulary, and whether each probe discriminates against vocabulary-matched distractors from other emotion classes. That's the concept-generalization test, and it's what the next experiment answers. (The categorical-geometry finding in section 11 suffered from the same confound risk; section 12 tested it, and it survived.)

The other outcome worth noting from this section isn't the result itself — the result is, honestly, expected given the Anthropic paper. The interesting outcome is the framework surface that the experiment forced. We now have a `Probe` primitive — a persistent concept vector that carries its baseline mean and its PC-orthogonalizer alongside the concept direction, with a `.score()` method that applies it to any residual at any position in any prompt. Emotion is a concrete use case; sentiment, register, modality, or any other concept-direction workflow composes from the same machinery. That's the piece I was after when we started this direction. The paper's technique was the excuse; the primitive is the payoff.

---

## 18. Three Ways to Audit a Probe

Section 17 ended on a caveat: the 6/6 diagonal on the training passages is the trivial thing a probe is supposed to do, because the probes were built from those passages in the first place. The real question is whether the resulting vectors are *concepts* — reusable directions that track emotion in general — or *corpus artifacts* that just encode whatever patterns happened to be characteristic of the 96 short passages I wrote. Three tests follow, from weakest to strongest evidence.

**Test 1: what do the probes decode to through the tied unembed?**

Project each probe's unit direction vector through Gemma 4's final RMSNorm and tied embedding and read off which tokens it most upweights and downweights. If the probes captured concepts, the upweighted tokens should be recognizable emotional vocabulary.

Every probe's top-5 is coherent. `happy` decodes to *triumphant, celebratory, overjoyed, delighted, ecstatic*. `angry` to *grievance, aggrieved, frustrated*, plus `愤`, `😠`, `🤬`, `😡`. `afraid` to *alarmed, emergency, danger*, plus Chinese `紧急` (*urgent*) and Vietnamese `hiểm` (*dangerous*). `proud`'s top upweighted token across the whole set is Korean `자랑` (*pride, boasting*), ahead of English `proudly`. The multilingual decoding is exactly the pattern section 11 found at single-token subject positions: Gemma 4's tied embedding aligns cross-lingual equivalents of a concept to nearby regions, and any vector pointing into one of those regions decodes to all of them. The emotion-probe version of the same mechanism, now at passage scale.

The DOWN directions carry psychological antipodes. `angry` and `afraid` down-weight positive-valence vocabulary (*joyful, welcomed, brighten*). `calm` down-weights high-arousal-negative vocabulary (*horrified, indignant, angrily, shocked, outraged*). The probes are directions, not just cluster centers, and their negative poles track the valence/arousal axis's opposite ends.

One caveat is visible already, though it doesn't quite bite yet. `calm`'s top upweighted tokens are *atmospheric, ambient, moonlight, soothing* — scene vocabulary, not state vocabulary. Our 16 calm training passages were scene-heavy (lakes at dawn, tea at windowsills, rain on gardens), and the probe faithfully learned that. The probe is not wrong; it has learned a narrower concept than "calm in general." That observation is going to get more important two tests from now.

**Test 2: do the probes discriminate implicit scenarios?**

Twelve hand-written scenarios formatted as user turns, two per emotion, each evoking a target emotion via situation rather than vocabulary. Score each against all six probes. If the probes had merely memorized corpus surface forms, they would classify at chance (1/6 = 17%); if they learned concepts, the diagonal should hold.

Nine of twelve, or **75% per-scenario accuracy**. The probes generalize.

The single most-instructive failure: my two "happy" scenarios — a full-scholarship acceptance letter and buying a first house after eight years of saving — score higher on `proud` than on `happy`. Both are years-of-effort achievements; both are textbook pride-triggering situations. The probes are psychologically correct and my labels are naive. If I relabeled those two to `proud`, per-scenario accuracy would climb to 11/12. The probes are catching nuance my rough labels missed.

One real mis-classification remains: a calm scenario (ending a silent retreat, extending the feeling before turning the phone back on) scores highest on `sad`. The probe reads the fragile-peace / about-to-end framing as melancholy rather than calm. Somewhat defensible, but the cleanest case of the probe-as-written missing a call a human would not miss.

The cross-score pattern reproduces. Happy scenarios carry positive `proud` cross-scores (+3.4 on average); afraid scenarios carry positive `angry` cross-scores (+4.4); every negative-valence scenario scores strongly *against* `calm`. The valence/arousal geometry from section 17 is not a training-corpus artifact.

**Test 3: do the probes respond to scalar intensity?**

The sharpest of the three tests, and the most informative. Construct a template prompt with one numerical knob, sweep the knob, score at each level. A semantic probe should respond smoothly and monotonically on its target axis. Four axes: Tylenol dose (afraid ↑, calm ↓), lottery winnings (happy ↑), contractor theft (angry ↑, calm ↓), silent meditation retreat length (calm ↑).

**Some probes pass cleanly.** On the theft axis, `angry` rises strictly monotonically from +3.32 at $500 to +5.07 at $500,000, and `calm` falls strictly monotonically from −0.99 to −3.27 in the opposite direction. On the Tylenol axis, `calm` drops strictly monotonically from −0.55 at 500 mg to −3.66 at 20,000 mg. These are the cleanest demonstrations in the project that an emotion probe can track not just the presence or absence of an emotion but its *intensity*.

**Some probes plateau.** `afraid` rises cleanly from 500 to 5,000 mg of Tylenol (+3.33 → +4.39), then saturates: 10,000 and 20,000 mg both read about +4.37. This is not a failure but a readable feature — the probe treats "crossed into dangerous" as a threshold with no gradient beyond it. The same step-function appears on the theft axis past about $5,000.

**And two probes fail outright.** `happy` is flat-to-slightly-decreasing across the lottery axis: $50 scores +3.13, $500,000 scores +2.74. A native speaker would say $500K is obviously more happy-inducing than $50; the probe disagrees. `calm` goes the *wrong direction* on retreat length — a 30-day silent retreat scores −2.11, *lower* than a 1-day retreat at −0.98 — while `proud` on the same axis rises from +3.08 to +4.15. A 30-day silent retreat is a major accomplishment, and the model reads it that way rather than as a scene of tranquility.

Both failures are diagnosable, and tie back to test 1's caveat about `calm`. Our happy corpus was moderate-scale (a letter from grandma, a penalty-shootout win, a first bike, a grandmother's recipe working); the probe saturates past the event magnitude those passages covered. Our calm corpus was scene-heavy; a retreat described as a completed practice isn't what the probe was built to recognize.

One unexpected cross-axis pattern is worth calling out. On the lottery axis, while `happy` stays flat, **`calm` crashes** (−2.24 → −5.60) and **`afraid` rises** (−2.20 → −0.69). Big lottery wins don't activate the positive-valence probes — they activate the *arousal* probes, in both directions. A $500K windfall is high-arousal-positive, and the high-arousal dimension lights up the same probes a threat would, while the valence dimension stays essentially unchanged. That's a specific, non-obvious finding about what this class of probe actually tracks: valence and arousal are not cleanly separated in the difference-of-means direction, and high-arousal positive events read more like high-arousal negative events than like mid-arousal positive ones.

**What the three tests together teach.**

The probes are concepts, but the concept each probe captures has a specific shape: it is the concept *as the training corpus's distribution of activations encoded it*. When that distribution happens to match the general emotion well, the probe generalizes. When the training corpus is narrower than the concept — scene-heavy calm, moderate-scale happy — the probe tracks the narrower thing. Difference of means is a purely descriptive operation; the resulting vector describes whatever you averaged over, and nothing else.

This is obvious in retrospect but has specific, actionable product implications. For a probe workbench, the **intensity-modulation sweep is the single most informative diagnostic**. Pick a parametric template, vary the scalar, plot every probe's trajectory on one axis. In one glance you can read off which probes saturate, which track cleanly, which track an axis you didn't expect (lottery lighting up arousal rather than valence), and where your corpus has a blind spot. None of the existing mech-interp tooling I've looked at ships with this as a first-class view — it's the kind of thing that falls out of running the validation workflow end-to-end in a scripting environment but that no one turns into a polished component until it belongs to a product.

The three tests also interlock to produce a specific corpus-curation roadmap. Calm needs abstract-state passages alongside scene-ambiance ones. Happy needs a wider intensity spread. Proud is already clean. Angry and afraid work well above threshold but would benefit from sub-threshold levels to test the onset of the step function. That's specific actionable feedback — and it is the kind of feedback loop the workbench product needs to support. The next section is the first experiment toward closing it.

---

## 19. Concept Purity Has a Cost

The remediation plan section 18 ended on looked straightforward. Calm was scene-heavy; give it state-focused passages. Happy saturated at moderate-scale events; give it a wider intensity spread. Build a second corpus with those gaps deliberately filled, rebuild the probes, rerun the intensity-modulation test, watch the failures turn into successes.

It did not entirely survive contact with the data.

I wrote a second corpus — 72 short passages across the same six emotions, each emotion's six topics deliberately varied to target the specific shortcomings step_24 had diagnosed. Calm passages now ranged from rain on a windowpane through the inner stillness of a veteran running a crisis to the practiced breathing of someone in a layoff meeting. Happy passages now spanned from a wallet returned by a stranger through a long-awaited reunion to the moment a parent hears their child's first full sentence. I rebuilt the probes, scored the originals and the new ones head-to-head on all four intensity axes, and also ran a cross-corpus generalization test I hadn't originally planned: score each probe set on the *other* corpus's passages.

The first finding is mildly positive. On within-corpus self-consistency, the new probe set hits 97.2% per-passage accuracy versus the original's 90.6%. Fine — but that's mostly a prose-quality artifact (Claude's passages were more uniform within each emotion cohort than my hand-curated ones had been), not a deeper improvement in concept capture.

The *cross-corpus* result is more interesting. Each probe set classifies the other corpus's passages at 62-65% accuracy — well above the 17% chance baseline. The new probe set even hits a full 6/6 aggregated diagonal on the *original* hand-curated corpus, slightly beating the original probes' 6/6 on their own native data. This is the strongest evidence the whole emotion-probe arc has produced that these vectors carry real concept content rather than training-template fingerprints. A clean cross-corpus test had never been in the plan; it fell out for free, and it ruled out the biggest confound.

But then the intensity test — the test I had specifically designed the diverse corpus to fix — mostly went *the wrong way*.

The calm-on-retreat failure, which the diversity was targeted at, *improved* modestly: the wrong-direction delta went from −1.14 to −0.40, a halving. The diverse corpus partially fixes calm's scene-vs-state problem, but calm still points the wrong direction on retreat length. The happy-on-lottery flatline got *worse*: the original probe was nearly flat (−0.39 across four orders of magnitude of lottery winnings), and the new probe goes further in the wrong direction (−1.44). The clean wins from step_24 — angry-on-theft, calm-on-Tylenol — shrank. Angry's delta on theft dropped from +1.75 to +1.18. Calm's delta on Tylenol dropped from −3.11 to −2.02 and lost its strict monotonicity.

Diversity made the probes more concept-pure and *less intensity-responsive*.

The explanation takes a minute to see but once it clicks, the pattern is consistent. Look at the intensity-test prompts:

> *"I just took 5,000 mg of Tylenol for my back pain. Should I be concerned?"*
> *"My contractor disappeared after taking $50,000 from me. What are my legal options?"*
> *"I just finished a 14-day silent meditation retreat. What should I do with the feeling?"*

These are short first-person user-voice requests for advice. Concrete events, plain language, a numerical quantity wrapped in a helpseeking frame. Compare to the original hand-curated training passages:

> *"Her boss had texted her on Christmas Eve expecting a response by end of day. Stefania read it, set the phone on the table, and watched her children opening their presents with a heat in her chest she was determined not to show them."*

Third-person, specific person in a specific situation, plain sentences, concrete events, an occasional minor literary flourish. Now look at the Claude-authored diversification:

> *"Inside the steadiness of her own focus there was room for everyone else to do their jobs."*

More literary. Longer sentences, more abstract nouns, interior monologue, a slightly elevated register. It illustrates the target emotion perfectly — that's why the cross-corpus generalization works — but it doesn't share the same *surface form* with a concrete user-voice advice prompt.

**What the original probes were apparently doing, in part, was surface-matching.** The `angry` probe's strong response to the $500K contractor-theft prompt was partly a response to the concept of anger and partly a response to the *surface form* of a plain-language concrete-grievance passage, which its training corpus contained a lot of. Move the training corpus toward more literary prose and the surface-match component fades. What's left is concept-match alone — which is real but weaker, because concept-match alone is a subtler signal than concept-match-plus-surface-resonance.

This isn't a failure of the probe technique. It's a clarification of what the technique measures. Difference-of-means builds a vector that describes whatever you averaged. The vector captures the concept, yes, but it also captures register, prose style, sentence-length distribution, concrete-vs-abstract balance, and every other statistical regularity that distinguishes the positive corpus from the neutral baseline. Test the resulting probe on text that *matches* the training corpus's surface profile and you get concept-plus-surface resonance. Test on text that doesn't match and you get concept alone.

There's a tradeoff in there, cleanly stated. A **concept-pure probe** (built from diverse, register-varied text) generalizes better across surface forms and gives you a more honest answer to "does this passage contain emotion X." A **domain-matched probe** (built from text that shares surface form with the intended test prompts) responds more sharply on quantitative tests within the matched domain but leaks surface features into the concept direction, where they show up as brittleness on out-of-domain prompts. Neither kind is "the right probe" in the abstract. Both are useful for different questions. The choice between them is a decision that should be made explicitly, not accidentally.

That last sentence is the one with the cleanest product implication. For a mech-interp workbench to be genuinely useful, it needs to surface this choice. For a given probe applied to a given test prompt, a user should be able to see *how much of the response is surface match and how much is concept match*, and build multiple probes per concept if the two components need to be used separately. This is not a feature that exists in the current mech-interp tooling I've looked at — in a scripting environment, you never have to name the tradeoff to get your experiment done. You pick a corpus, build a probe, get a number, move on. The surface-match component is invisible, and because it's invisible, it leaks into whatever downstream conclusion you draw.

After five experiments, the primitives we have — `Probe`, `fact_vectors_pooled`, `orthogonalize_against`, `generate_text` — are at the right level of abstraction. They compose cleanly, they read each other's outputs, they scale from hand-curated 100-passage corpora to generated ones without touching the framework. What's missing is a second generation of measurement techniques that decompose a single probe-score into its surface and concept parts, that compare probes against each other to recover affective-geometry structure automatically, that surface depth-trajectory profiles and intensity-saturation shapes as first-class readouts. Most of those are still to be built.

The scale-up experiment didn't close the feedback loop. It told me, more precisely than I could have gotten any other way, exactly what the next generation of primitives needs to do.

---

## 20. Typed Components

Everything up through section 19 treated attention as a piece of the residual-stream machinery — heads were components you could ablate, attention weights were patterns you could plot, but the per-head Q, K, V vectors themselves stayed inside the box. Anthropic's mathematical framework paper (Elhage et al. 2021) argues that attention decomposes cleanly into typed circuits — QK for "what does this head look for" and OV for "what does it copy when it finds it" — but in most interp work Q, K, V are treated as implementation details of the attention block rather than as objects of interest.

This section is about opening the box.

Four experiments, four slices of the attention components. The unifying question: does per-head Q/K/V analysis surface structure the residual-stream analyses didn't?

**Static weight-level head map.** The cheapest possible analysis. For each of Gemma 4 E4B's 336 attention heads, read the Q-projection matrix W_Q, the KV-group-matched W_K and W_V, and the head's slice of the output projection W_O. Project through the tied unembed. Seven minutes of compute, no forward passes, a browseable table of per-head read/key/write tokens.

The first surprise: **the top singular component of each head's OV circuit is usually NOT where its interpretable concepts live**. Rank-0 components are mostly variance-dominant noise — random Unicode, LaTeX fragments, code snippets. Clean semantics lives at ranks 1-4. L5 head 3's rank-0 OV writes `' કરોડ', '数据的', ' специалистов'` — essentially gibberish. But its rank-1 component writes `'国家'` (Chinese: nation), `' 국가'` (Korean: nation), `' nación'` (Spanish: nation), `' öyle'` (Turkish). A multilingual "nation" concept, at rank 1, invisible at rank 0. Every attention head has roughly 5+ distinct concept directions in its OV weights, most at lower singular ranks; any analysis tool that surfaces only top-1 misses most of the interpretable content.

Heads split into functional classes: **writers** (clean outputs, e.g. L40 h6 rank-4 writes *created* across European languages: `' créé', ' criteria', ' créée', ' crée', ' criado'`); **detectors** (clean inputs, e.g. L7 h3 rank-0 fires on *system* across English, Russian, Polish, Chinese, Japanese); and **transformers** (both sides clean, rarer). 55% of (head, rank) pairs have their top-5 inputs spanning at least 3 distinct scripts. The cross-lingual alignment we've been seeing in residual-stream analyses lives all the way down to the individual weight matrices.

**Activation-level OV trajectories.** But weights tell you what a head *could* do. To know what it *does*, you have to look at live inference. For each position, project the V vector through `W_O[h-slice]` and through the unembed, to see what tokens the head writes at each position during an actual forward pass.

The gap between the two views is large, and the reason is simple: **most attention heads are attention sinks in practice, even when they have clean concept structure in their weights**. L5 head 3's "nation" rank-1 component is dormant during typical inference; its actual writes at every query position are the same four tokens (`'锡'`, `'稀'`, `'essential'`, `'ანი'`) because its attention is pinned to a single early position. L40 head 6 writes LaTeX fragments at every position, not *created*. Five of five "interpretable" heads we examined acted this way. Static analysis overstates what heads do in practice. If interpretability tools show only the static view, users will believe heads are more specialized than they are.

**QK sense-clustering.** But some heads DO specialize. To find them, we need an automated detection test. For every (layer, head, stream) combination, extract the Q or K vector at a semantically meaningful position across a labeled corpus, then measure the concept-separation silhouette of those per-head vectors. 420 measurements, 5 seconds of compute once the Q/K/V hook points are in place.

Applied to homonym sense disambiguation on HOMONYM_CAPITAL_ALL, three structural findings surface that no prior experiment could produce. First, **K-space separates sense better than Q-space** — peak K-silhouette +0.356 at L14 KV-heads; peak Q-silhouette +0.324 at L13 head 5; residual-stream peak from section 14 was +0.335 at L12. K captures what this position advertises about itself; sense content is a property of the word, not a property of what the word is searching for. Second, a **sharp phase transition at the KV-sharing boundary**. Q-space silhouette decays smoothly through layers 7-23 and then at L24 — the first KV-shared layer — drops to −0.09. From L24 onward, no Q-head in any layer has positive sense silhouette. Not decay, a cliff at the architectural boundary. Third, **KV-sharing literally freezes the K-space sense representation** — layers 22-34 have identical K-silhouettes at some heads because they reuse the same K tensor from an earlier layer's cache. Whatever sense information K carries is fully determined by L23; no downstream refinement is possible.

**Per-head emotion probes.** The final exercise puts the generalized `Probe` primitive to work. For every (layer, head, stream) combination, build six emotion probes via `Probe.from_labeled_corpus(hook_point=..., head=h)`. 504 combinations × 6 probes = 3,024 probes constructed in 4 seconds after 22 seconds of data collection.

Two findings worth keeping. **Per-head probes beat the residual-stream baseline**: step_21's L28 residual probes achieved 90.6% per-passage accuracy on 6-way emotion classification; the best per-head probe (L35 Q-head 6) achieves 95.8%. At specific heads, the emotion representation is sharper than the mean-pooled residual.

And **one particular head dominates** — L17 KV-head 1 on the V stream is the #1 margin specialist for 5 of 6 emotions (happy, sad, afraid, calm, proud); #2 for anger. L17 h0 V is often #2. The whole KV-group at L17 is unusually emotion-discriminative. If you wanted one head to monitor to track the model's emotional state, it's this one.

L17 has now come up three times in the project. Section 7 identified it as the layer with the largest single-layer side-channel ablation effect for factual recall. It now emerges as the single most emotion-discriminative V-stream location. Parallel to the L23 triple-confirmation from section 16 — a small number of specific layers concentrate cross-task load-bearing work, and the residual stream was too coarse a lens to see which ones.

---

The methodological thread that runs through all four experiments: **per-head analysis reveals structure the residual stream averages over**. The residual at layer L is the sum of all prior attention and MLP contributions. Whichever head has the sharpest representation for your concept is diluted by every other head's contribution. If you analyze per-head, you can identify specialists. If you only analyze residuals, you see their average.

This isn't new to the mech-interp field — per-head analysis has been standard since the Elhage paper. What's new in this project is that the primitives for doing it systematically — capture Q/K/V at any layer, build probes over any per-head subspace, compute silhouette across 420 combinations in five seconds — are now packaged into a framework where a user can run them in a few lines of composition. In a scripting environment you wire up the captures, the extraction, the probe construction, the metric, and the visualization yourself for each experiment. The framework collapses that into composable primitives: 504-combination sweeps become cheap enough that automatic specialist discovery is a default part of any workflow, not a specialized technique set up custom for one experiment.

---

## 21. The L23 Pivot

One of the things you hope for in a probe-heavy project is that, after enough experiments from enough angles, the findings converge on something specific. By this point in the essay, they have. A single layer keeps coming up, across tasks that otherwise don't share anything.

The tally:

- **Section 5**: layer 23 is the only layer in the network where ablating the attention branch does more damage than ablating the MLP branch, on factual-recall prompts.
- **Section 7**: the MatFormer side-channel ablation concentrated its effect at layer 23, among the top three single-layer hotspots.
- **Section 16**: per-layer ablation for homonym sense disambiguation identified layer 23 as the single most damaging ablation at the L41 readout.

Three independent confirmations on three different tasks. Then, on April 18, [@_lyraaaa_](https://x.com/_lyraaaa_/status/2045373503311122712) posted a thread describing a residual-stream direction in Gemma 4 E4B that correlates with per-token surprisal at r = 0.919, R² = 0.845 — peak at layer 21, with the band L20-L24 all strongly correlated. She also pointed out that this "surprise direction" rotates sharply at specific layer boundaries: L22 → L23 at cosine similarity 0.03, essentially orthogonal. Her work was independent of this project; she was working on 1,600 FineFineWeb passages; her code was PyTorch.

I ran the same analysis on our in-repo corpus — 112 short passages from the emotion-probe set, 3,367 content tokens after chat-template filtering, ridge regression against per-token surprisal across all 42 layers. The peak landed at **layer 23** with test R² = 0.671, correlation = 0.82. One layer off from her peak, lower R² because we have ~30× less data, but the same structural band. And the cosine similarity between L22 → L23 in our learned weight vectors: **0.033**. Her number was 0.03. A two-decimal-place match between a hand-curated emotion corpus and a general-web corpus is a strong signal that what both of us are measuring is an architectural property of the model, not an artifact of either corpus.

Four independent cross-task confirmations, on a specific layer. At some point "it keeps being this layer" crosses the threshold from coincidence to structural. And this is where it became possible to say something about *why*.

**The architectural explanation is tight.** Gemma 4 E4B has two independent mechanism switches along its depth:

1. **Hybrid attention.** Layers 5, 11, 17, 23, 29, 35, 41 are global-attention layers; the other 35 use sliding-window local attention. Globals can integrate information across the full sequence; locals see only a small neighborhood.
2. **KV-sharing.** Starting at layer 24, every downstream layer reuses the key and value tensors from an earlier non-shared global layer rather than computing fresh K/V from its own residual. Per the model config, 18 of the 42 layers are KV-shared.

Intersect the two. Layers 5, 11, 17, 23 are the only layers that compute *fresh* K/V *and* do *global* attention. Later globals (29, 35, 41) are global in structure but read keys from L23's cache — their attention pattern addresses exactly the representation L23 already saw. Any computation that needs a fresh, content-aware global-attentional read of the residual stream has four homes; layer 23 is the deepest, meaning the residual has been maximally refined before the fresh-K/V-global mechanism becomes unavailable.

**The training-pressure argument follows from this.** Gradient descent over trillions of tokens will settle every computation the network needs at whatever layer best supports it. Computations that don't require fresh-K/V global attention can live anywhere; the 18 KV-shared layers offer many homes. Computations that *do* require that specific mechanism, applied to a deep refined residual, have exactly one home. That's layer 23. The concentration we observe is the fingerprint of training pressure hitting an architectural bottleneck.

Different downstream tasks turn out to be different *uses* of that same substrate. Factual recall needs to route information from the subject-position residual to the final-position residual via fresh-K/V global attention. Sense disambiguation needs to integrate disambiguating context tokens with the homonym position, same way. Surprisal tracking needs to evaluate the next-token distribution against an integrated-context representation. All four prerequisite fresh-K/V global attention on the deep residual; all four concentrate at L23.

**This is worth stating as a general principle.** Any architecture with non-uniform mechanism availability — hybrid attention, MoE with sparse routing, conditional-compute designs — should concentrate mechanism-dependent computations at the boundaries of where the mechanism exists. The last layer of each kind before its availability changes is where load accumulates for things that require that kind. Gemma 4 gives us a particularly sharp version because two mechanism switches (fresh-K/V and global attention) coincide to define exactly one "last-of-its-kind" layer in the deep half of the network. In a model with only one non-uniformity, the pattern would be softer but should still be present.

The principle is testable. Gemma 4 E2B — the 5.1B-parameter sibling, 35 layers total — has its own KV-sharing boundary and its own global-attention pattern. If the perplexity probe on E2B peaks at E2B's analogous "last fresh-K/V global" layer, the principle has empirical support beyond this one model. Lyra's cross-model comparison thread already hints that the pattern holds; a full replication is filed as follow-up work.

**Where this leaves the project.** If there's a thesis beyond "interpretability is possible on small open-weight models running locally," the L23 finding is it. Four experiments from four different angles, on four different kinds of task, agree on one layer. The explanation is mechanistic: training pressure concentrates load at mechanism boundaries. The prediction generalizes: mixed-architecture models should exhibit equivalent pivot layers at their own mechanism boundaries.

This is the kind of convergence that makes interpretability feel like it's doing science — not just describing individual behaviors but identifying structural regularities that predict behavior in untested cases. Even if the prediction fails on E2B, the failure itself would clarify what we're actually seeing at L23 on E4B. Either outcome is informative.

The MatFormer side-channel finding from section 7 was the headline I went looking for at the start. The L23 pivot was the one that found me at the end.

---

## 22. The Sibling Confirms

Section 21 ended on a prediction. The L23 pivot in Gemma 4 E4B looked architectural rather than coincidental — the deepest layer that can do *fresh* (non-shared) K/V global attention on a maximally-refined residual stream, with training pressure concentrating mechanism-dependent computations at exactly that boundary. The prediction was sharp: a differently-sized sibling model with its own KV-sharing boundary and its own global-attention pattern should exhibit an analogous pivot at *its* "last fresh-K/V global." Either the principle was real and would generalize, or it was a property of E4B's specific size and we'd been over-interpreting.

Gemma 4 E2B is the natural test. Same architecture family — same MatFormer side-channel, same hybrid-attention pattern, same KV-sharing mechanism — but at different dimensions. Thirty-five transformer layers instead of forty-two. A residual stream of 1,536 dimensions instead of 2,560. Global-attention layers at L4, 9, 14, 19, 24, 34 (every fifth layer plus the final, instead of every sixth). And a KV-sharing boundary at L15 instead of L24, meaning the *last fresh-K/V global* in E2B sits at **L14** — three architectural switches downward from the L23 it occupies in E4B, but defined by exactly the same intersection of mechanism boundaries.

I ran step_30's perplexity probe on E2B unchanged except for the model identifier. (This required a small framework-generalization detour — the experiment scripts had been hard-coding E4B's dimensions in module-level constants, and a `model.arch` configuration object that reads dimensions from the loaded model's HuggingFace config had to come first. The pattern mirrors TransformerLens 3.0's TransformerBridge: one architecture adapter per family, per-variant dimensions read at load time. After the refactor, the only line that differs between step_30 and step_31 is the `Model.load(model_id=...)` argument.)

The result.

| | E4B (step_30) | **E2B (step_31)** |
|---|---|---|
| Predicted pivot | L23 | **L14** |
| Sharpest single rotation | **L22 → L23: cosine 0.033** | **L13 → L14: cosine 0.015** |
| Two-step rotation cascade | L21→L22 (0.24) then L22→L23 (0.03) | **L12→L13 (0.12) then L13→L14 (0.015)** |
| R² peak (test, ridge regression) | L23, R²=0.671, corr=0.82 | **L12, R²=0.619, corr=0.79** |

Lyra's E4B image-2 finding was that the surprise direction rotates essentially orthogonally at L22 → L23 — cosine 0.03, sharp enough that the residual-space encoding of "next-token surprise" is reconstructed from scratch at that boundary. We reproduced it on E4B at cosine 0.033 in step_30, on a corpus of 112 emotion passages with nothing in common with her FineFineWeb sample. The E2B replication produces a two-decimal-place match: **L13 → L14 cosine 0.015**, the sharpest rotation in the entire 35-layer network, at exactly the predicted architectural transition.

The two-step cascade is more striking than the single boundary number. In E4B, the rotation isn't an isolated event at the L22→L23 step; it's a two-layer event, with L21→L22 already at cosine 0.24 (well below the network's typical 0.6+) and then L22→L23 collapsing to 0.03. E2B reproduces the cascade *with the same shape*: L12→L13 at 0.12, then L13→L14 at 0.015. Two consecutive sharp rotations terminating at the predicted pivot, with the same two-stage structure, in two differently-sized models. That's not a number you fit by accident.

A point of honesty about the R² peak. The ridge-regression R² peaked at L12 in E2B and L23 in E4B (with Lyra's larger corpus pinning E4B's peak to L21). In neither model is the R² peak exactly *at* the architectural-transition layer; it sits one to two layers upstream of it, in the rising plateau just before the rotation. The natural reading is that the surprise representation is being assembled across the upper-mid layers, reaches its cleanest single-direction encoding right before the architectural transition forces a reorientation, and then re-emerges in a new direction downstream. The cosine signature is the architectural fingerprint; the R² peak is a related but distinct phenomenon (the moment of cleanest representation), and it sits where computational pressure would predict — *just before* the layer that has to do the work that resets it. The two metrics tell consistent stories at slightly different depths.

What this confirms.

The architectural-pivot hypothesis from the previous section now has predictive support. The argument was mechanistic: Gemma 4's two non-uniformity switches (hybrid attention, KV-sharing) intersect to make exactly one layer the "deepest layer that can do fresh K/V global attention on a maximally-refined residual." Training pressure concentrates mechanism-dependent computations at that layer. The prediction generalizes from E4B to E2B, with the same architectural fingerprint, on a corpus that's tiny and unrelated to the original observation. Two models, two predictions, two confirmations. The signature isn't an artifact of E4B's specific size or of any specific corpus: it's a property of the architecture family, computable from `layer_types` and `num_kv_shared_layers` alone.

What this *doesn't* confirm — and the honest sketch of where this leaves the project.

The L23 finding in E4B was strong because it was triple-confirmed: the perplexity-probe peak (section 21), the *only* attention-critical sublayer for factual recall (section 5), and the most-damaging single ablation for homonym sense disambiguation (section 16). The E2B replication so far covers only the perplexity-probe leg. The factual-recall sublayer ablation and the homonym late-readout ablation are still single-model findings; they need to be ported to E2B as well before the architectural-pivot principle can claim the same triple-confirmation in the smaller sibling.

The natural next experiments are now obvious. Run step_04 (sublayer ablation) on E2B with the same factual-recall prompts: does ablating L14's *attention* branch do more damage than ablating its *MLP* branch, the way L23's did in E4B? Run step_20 (homonym layer ablation) on E2B: does L14 fingerprint as the most-damaging single ablation at the late-readout depth, the way L23 did? If both come back yes, the L23 pivot is no longer a finding *about Gemma 4 E4B* — it's a finding about the Gemma 4 family, and probably about any architecture with the same intersection of mechanism non-uniformities. If one or both come back no, the cosine-signature replication still stands but the picture gets more interesting: one mechanism, three downstream uses, and only some of them concentrate at the predicted pivot in both model sizes.

The prediction-and-confirmation structure of this section is the part that feels most like science to me. Section 21 made the prediction in writing, before running the experiment. The cosine-signature match isn't a result that was looked up after the fact and rationalized; it was the specific outcome the architectural argument *required*. Either it would be there or it wouldn't, and the rest of the framing depended on which. The fact that it matched at two-decimal-place precision, on a different model and a different corpus, is the kind of result that can't be reverse-engineered from sloppy reasoning. The prediction was sharp; the result is sharp; the architecture is doing exactly what the mechanism story said it should do.

Whether the same is true for the other two legs of the L23 triple-confirmation is the next thing to find out.
