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
