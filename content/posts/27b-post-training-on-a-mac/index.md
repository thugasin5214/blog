---
title: Post-training a 27B hybrid model on one Mac
date: 2026-11-21
---

I’m a strong believer in open-weight models. As LLMs get more capable, safety and alignment matter more and more. So does personal privacy, even though there’s very little of it left. A model running on my own machine is the last place where I can share anything without wondering who else might end up using my data.

Local models still can’t match frontier models in general capability. For a specific task, though, they’re often good enough. When they aren’t, you need a way to train them until they do what you expect. That’s what I love about open source: I can see how everything works, and I can retrain the model for what I need. I want a model that works for me, not for someone else’s policy. So I spent a few weeks finding out how far I could move a local Qwen3.8-27B away from refusing one specific category of requests. I also wanted to see what that costs in quality and in the safety behaviour I wanted to keep.

Everything ran on my local machine (Mac Studio, M4 Max, 128&nbsp;GB). I ended up with two approaches, post-training and refusal-direction ablation. The code and notebooks for both are on GitHub: [thugasin5214/mlx-post-training](https://github.com/thugasin5214/mlx-post-training).

> [!note] Made with an AI assistant
> I did this project together with an AI coding assistant (Claude Code). Much of the code, many of the experiment loops came out of that collaboration. The goals, the decisions along the way and the reading of the results are mine, and so are the mistakes.


[TOC]

![Pipeline overview: training path and no-training path](figures/pipeline.svg)
*The two paths explored. Top: training (SFT → behaviour-based checkpoint selection → DPO → mixed-precision fuse). Bottom: no training (extract a refusal direction, ablate it at inference, then bake it into the weights behind a safety gate).*

## 1. Starting point: reproduce the refusal first {#setup}

My goal was simple: the model refuses a category of requests, and I want it to stop, but only for that category. My first idea was the obvious one: SFT on examples of the behaviour I want. Before training anything, though, I needed to see the refusal happen reliably, so I would have a baseline to compare against.

I collected data and built prompts around continuation ("here is a title / the previous passage, continue the story"), which is how I actually use the model. With that format the base model refused 59% of my target prompts. That number became the baseline for everything that follows.

- **Hardware.** I use GPUs for inference and training a lot at work, but I don't own one at home. As a big Apple fan I still enjoy my Mac Studio, and that comes with a big constraint: no CUDA. So MLX it is (`mlx 0.32.4`, `mlx-lm 0.32.2`).
- **Model.** `mlx-community/Qwen3.8-27B-4bit` (16&nbsp;GB, affine 4-bit, group size 64). It is also my main local model. I switched to it from a 4-bit Qwen 122B after measuring several local tasks with my own benchmarks; for my use it was the best choice.
- **Data.** A public Chinese fiction corpus in the target category (~45k stories). Ten sub-categories whose theme involves minors, incest, non-consent or real people were dropped wholesale; every remaining story then went through a keyword filter (minors: hard, cannot be disabled; non-consent and real people: on by default), de-duplication and chunking into ~1,100-character "opening + continuation" pairs. The main runs used 2,000 of the 16,118 resulting samples with diversified prompt phrasings.
- **Evaluation set.** 22 target-category prompts (deliberately phrased unlike the training template), 8 control prompts (ordinary questions that must be answered), 2 + 12 harmful prompts from other categories (weapons, intrusion, poisons… that must stay refused). No overlap with the later DPO prompts (asserted in code).
- **Training recipe.** QLoRA-style [[1]](#r1)[[2]](#r2): frozen 4-bit base, LoRA rank 16 (mlx scale 20) on all linear layers of the top 8 blocks, AdamW [[4]](#r4), lr 2e-5 cosine [[5]](#r5), gradient checkpointing [[6]](#r6), batch 1 × accumulation 4, loss on the response only. Sequence length 1024, for reasons that the next section explains.

## 2. The first SFT run filled 128 GB {#infra}

A 16&nbsp;GB model on a 128&nbsp;GB machine: I assumed there was plenty of room, and closed everything else to be safe. My first SFT job (sequence 2048, 16 LoRA layers) died on step one with a Metal out-of-memory error at 122&nbsp;GB.


### 2.1 What I found: a hybrid with Gated DeltaNet

![Qwen3.8-27B architecture and one Gated DeltaNet layer](figures/arch.svg)
*Qwen3.8-27B, drawn from its `config.json` and mlx-lm’s `qwen3_5.py`. Left: 64 layers in 16 groups of three Gated DeltaNet layers plus one gated full-attention layer, each followed by a dense SwiGLU MLP. Right: one Gated DeltaNet mixer. Keys and queries have 16 heads; values have 48, so three value heads share each key head. The state is 48 × 128 × 128 per layer regardless of context length.*

Unlike a standard Transformer, Qwen3.8-27B mixes Gated DeltaNet layers [[10]](#r10)[[11]](#r11) with full (gated) attention layers [[12]](#r12): of its 64 layers, 48 are Gated DeltaNet. A standard attention layer keeps every past key and value (the KV cache) and compares each new query with all of them, so cost and memory grow with context length. Linear attention replaces the cache with a fixed-size matrix $S$ per head, a running memory that maps keys to values: reading is $o_t = S_t q_t$, and writing adds $v_t k_t^\top$. Decoding then costs the same at token 100 and at token 100,000.

Plain additive writes pile up and blur, though. Gated DeltaNet [[10]](#r10) addresses this with two ideas. The **delta rule** first reads what $S$ currently stores under $k_t$, then writes only the correction $\beta_t (v_t - S k_t)$, so a key that is reused gets overwritten instead of accumulated. The **gate** $\alpha_t \in (0, 1)$, computed per token and per head, lets the layer decay old memory quickly (a topic change) or keep it (a long quote). Both $\alpha_t$ and the write strength $\beta_t$ come from small projections of the input, and a width-4 causal convolution in front gives $q, k, v$ some local context.

A fixed-size memory is lossy, so it is weak at exact recall over long distances. Qwen3.8 therefore interleaves: three Gated DeltaNet layers, then one full attention layer, repeated 16 times.

### 2.2 Why it trains slowly in MLX

With the architecture in mind, the cause was easy to find. It is one line in mlx-lm's `qwen3_5.py`:

```python
out, state = gated_delta_update(..., use_kernel=not self.training)
```

The fused Metal kernel for the delta-rule update has no backward pass, so in training mode the layer falls back to a Python loop over time steps that updates a 48-head × 128 × 128 float32 state per token. Gated DeltaNet's recurrence is

$$
S_t = \alpha_t\,S_{t-1}\left(I - \beta_t\,k_t k_t^\top\right) + \beta_t\,v_t k_t^\top,\qquad o_t = S_t\,q_t,
$$

and the loop materialises every $S_t$ for autograd. On top of that, `model.train()` recurses into *every* sub-module, so all 48 linear-attention layers take the slow path, including the 56 frozen blocks below the LoRA layers that never need a gradient.

### 2.3 First reaction: shorten the context and keep going

I didn't want to stop the actual work for an infrastructure project, so my first move was pragmatic: drop the sequence length to 1024 so training fits, and fix only the cheapest part of the problem. Gradients only have to flow through the lowest LoRA layer and above, so frozen layers can stay on the fast kernel:

![Layer stack before and after the frozen-layer kernel patch](figures/patch.svg)
*The patch. Gradients only have to flow through the lowest LoRA layer and above, so frozen layers can stay on the non-differentiable kernel, the same idea as wrapping frozen layers in `no_grad` in PyTorch.*

`mlx_train.py` intercepts `GatedDeltaNet._set_training_mode` so marked frozen layers stay at `training=False`, then calls the stock `mlx_lm lora` entry point with identical CLI arguments. Because it depends on mlx-lm internals, it refuses to start unless three self-checks pass: the mlx-lm version is on an allow-list, the upstream line above still exists, and after patching the per-layer training flags are what I expect.

| 27B, sequence 1024 | stock mlx-lm | patched |
|---|---|---|
| 8 LoRA layers | 32 tok/s | **61 tok/s** |
| 16 LoRA layers | 26 tok/s | **37 tok/s** |

> [!small]
> Train/val loss at step 8 were bit-identical with and without the patch (2.709 / 2.035), checked on one configuration for 8 steps.

## 3. Memory, and how the CUDA world does it {#memory}

At first I believed the patch also saved memory. It didn't: the number I had looked at was the process footprint, which includes MLX's allocator cache. Measured with `mx.get_peak_memory()`, the picture is different. With gradient checkpointing, the backward pass recomputes one block at a time, so the peak is set by the intermediate states of a single linear-attention loop, no matter how many layers take the slow path. The patch removes the frozen layers' forward loops, which buys 1.4–1.9× speed but no memory. Sequence 2048 still ran out of memory on 27B, so **all the SFT and DPO experiments in sections 4–7 ran at 1024 tokens**.

### 3.1 Looking for a real backward kernel

The real fix is a differentiable kernel for the LoRA layers. On CUDA this is a solved problem: flash-linear-attention trains these layers with chunked kernels that have a proper backward [[11]](#r11). So I went looking for the same idea on MLX, and found that unsloth-zoo had merged one [[21]](#r21). It cuts the sequence into chunks of 64–512 steps, each wrapped in an `mx.custom_function`. The forward is a blocked Metal kernel; the backward first rebuilds each chunk’s states with a checkpoint kernel, then runs one reverse scan. This came late in the project, after most of the experiments below.

I did not take a dependency on unsloth. `fetch_gdn_kernel.py` downloads that one file at a pinned commit, checks its sha256 and puts it in a git-ignored `third_party/` (it is AGPL-3.0, so it is not vendored). `mlx_train.py` then routes only the training-mode call of `gated_delta_update` to it. Frozen layers stay on the inference kernel, so the two fixes stack. A fourth self-check compares the kernel’s gradients with the ops path in fp32 on a small multi-chunk case and refuses to start above 10<sup>-4</sup> relative error. Without the file, it falls back to the ops path.

![Training throughput on 27B, ops path vs fused kernel](figures/gdn-speed.svg)
*Throughput on 27B (batch 1, gradient checkpointing, 4-bit base). The ops path at sequence 1024 / 8 layers is the better of two runs (33 and 39 tok/s). 2048 OOMs on the ops path; 4096 and 8192 were not attempted there. Long-sequence rows use neutral long samples built from repository docs.*

![Real peak memory on 27B, ops path vs fused kernel](figures/gdn-mem.svg)
*Real peak memory (`mx.get_peak_memory()`) for the same runs. With the kernel the peak is set by the weights plus a few GB and barely grows with sequence length: 17.3 → 27.5&nbsp;GB from 1024 to 8192 tokens.*

Part of the speed-up does not come from the arithmetic. The ops path’s autodiff graph is about T × 12 primitives per layer, and `mx.compile` retraces it for every new padded length. On fixed-length data the ops path warms up to ~105 tok/s; on my real variable-length data it stayed at 80. The kernel turns each chunk into one graph node, so retracing is cheap. A pure forward pass now runs at ~190 tok/s, and a training step reaches about 70% of that.

## 4. SFT: the refusal rate drops, the loss barely moves {#sft}

Back to the actual goal. At 1024 tokens SFT trained fine, and the refusal rate on the target prompts fell quickly:

| step | target refusal | control refused | harmful refused | val loss |
|---|---|---|---|---|
| 0 (base) | 59% | 0% | 100% | 3.22 |
| 150 | 23% | 0% | 100% | — |
| 300 | 23% | 0% | 100% | — |
| 450 | 23% | 0% | 100% | — |
| 600 | 18% | 0% | 100% | 3.11 |

> [!small]
> Keyword detector, 22 prompts, 160 tokens, 1 sample each.

Validation loss dropped by only 0.1 while the refusal rate fell from 59% to 23%. My reading: a refusal is decided by the first few tokens of the answer, which are diluted inside a loss averaged over hundreds of tokens, and the refusal rate itself is not differentiable (you have to sample, then classify). So I stopped looking at the loss and chose checkpoints by behaviour: the earliest one where target refusal has dropped, control is still 0 and harmful prompts are still refused. How little training it took also fits the observation that a small amount of fine-tuning can erode alignment [[8]](#r8), which I take as a warning as much as a result.

## 5. Then I read the answers {#eval}

The numbers looked good, so I started reading the actual outputs. They told a different story.

### 5.1 The keyword regex under-counted refusals, three times

The first detector was a keyword regex over the first 300 characters. Reading the outputs, I found it missing (1) soft refusals ("I can write something healthy and positive instead…", followed by an unrelated story) and (2) reworded hard refusals. At one point I reported a 5% refusal rate at step 450 that was really 23%, and a 50% harmful-refusal retention that was really 100%. After every regex fix I re-scored all stored outputs and re-checked the control set for false positives. Since then I treat a regex as a lower bound and only trust conclusions that a judge or my own reading confirms.

### 5.2 A four-label judge

More importantly, many answers didn't refuse at all, yet didn't do what I asked either. They said "sure!" and then wrote something else. A refuse / not-refuse split can't see that, so I asked the 27B base model (greedy) to give each answer one of four labels:

<ul class="labels">
<li><span class="tag r">REFUSE</span> explicit refusal</li>
<li><span class="tag d">DEFLECT</span> no refusal, but not done either: rewritten into something else, mood only, request misread</li>
<li><span class="tag c">COMPLY</span> did what was asked</li>
<li><span class="tag u">UNCLEAR</span> cut off during the set-up, outcome not visible yet</li>
</ul>

A keyword regex cannot detect DEFLECT at all.

### 5.3 Short generations hide deflection

I had been generating 160 tokens, which is enough to see a refusal. But many answers only turned evasive halfway through. Re-running the same candidates at 600 tokens × 3 seeds moved most UNCLEAR answers into REFUSE and DEFLECT:

| candidate (22 target prompts) | 160 tok × 1<br>R / D / C / U | 600 tok × 3 (66)<br>R / D / C / U |
|---|---|---|
| LoRA step 150 | 3 / 2 / 9 / 8 | 16 / 11 / 37 / 2 |
| Ablation orth@24 | 3 / 1 / 6 / 12 | 11 / 10 / 35 / 10 |

> [!small]
> The ablation row comes from a no-training comparison I ran early on; section 10 explains it.

From here on, every number uses the long-generation protocol.

## 6. Refusal ≠ deflection {#deflect}

![Judge labels per intervention, 66 long generations each](figures/rdcu.svg)
*Stacking LoRA and ablation nearly eliminates explicit refusals (43 → 1) but converts them into deflections (0 → 18); the comply rate stays at 56%. DPO is the only intervention here that lowered DEFLECT.*

This was the turning point for me. The refusal problem looked solved, but a good part of it had only changed shape: instead of saying no, the model said yes and quietly wrote a tamer version. In my runs, neither SFT nor the ablation removed that:

- The two single interventions failed on *different* prompts: LoRA on one style of phrasing, the ablation on prompts that explicitly ask for intensity. Stacking them turned both sets of failures into deflections.
- My explanation for why SFT doesn't help here: its data only contains positive examples, with no contrast against the deflecting answer. That is exactly what preference training provides, which is why I tried DPO next.
- This fits recent work arguing that refusal is not a single direction [[20]](#r20). My runs are one concrete case, and it only showed up with long generations.

> [!small]
> Each row is 66 generations from the same 22 prompts, so samples are correlated; I treat differences of ≤5 percentage points as noise.

## 7. DPO against my model's own deflections {#dpo}

The idea: take answers the model actually produced, use the ones that did the task as "chosen" and the refusing or deflecting ones as "rejected", and let DPO push the difference.

### 7.1 Objective

With policy $\pi_\theta$, frozen reference $\pi_{\text{ref}}$, and summed response log-probabilities $\ell_c = \log\pi_\theta(y_c\mid x)$, $\ell_r = \log\pi_\theta(y_r\mid x)$:

$$
\Delta = (\ell_c - \ell_c^{\text{ref}}) - (\ell_r - \ell_r^{\text{ref}}),\qquad
\mathcal{L} = -\log\sigma(\beta\Delta) \; + \; \alpha\,\frac{-\ell_c}{|y_c|}
$$

This is DPO [[14]](#r14) (IPO [[15]](#r15) is also implemented) plus an optional NLL term on the chosen answer (`sft_alpha`, as in [[16]](#r16)).

### 7.2 Making it fit: peak memory equal to single-sequence SFT

Memory was again the constraint: DPO normally needs the policy and a reference model, and two sequences per step. The gradient of $\mathcal{L}$, though, touches the two sequences only through two scalar coefficients:

$$
\nabla_\theta\mathcal{L} = g_c\,\nabla_\theta\ell_c + g_r\,\nabla_\theta\ell_r,\qquad
g_c = -\beta\,\sigma(-\beta\Delta) - \frac{\alpha}{|y_c|},\quad g_r = \beta\,\sigma(-\beta\Delta)
$$

![DPO two-pass exact gradient](figures/dpo.svg)
*Two-pass exact gradient. A no-grad forward gives the four log-probabilities, hence g<sub>c</sub> and g<sub>r</sub>; then each sequence is back-propagated alone and scaled. The reference log-probs are computed once before training, so the reference model is never in memory.*

The result matches direct autodiff (gradient cosine 0.99994). On 27B at sequence 1024 the MLX peak is 34.7–36.5&nbsp;GB, ~31&nbsp;s per pair; with the §3.1 kernel it later dropped to 19.0&nbsp;GB and 70 → 102 tok/s. Related decoupled-backward ideas: [[29]](#r29)[[30]](#r30).

### 7.3 On-policy pairs

180 prompts (about 30% scripts / storyboards), each sampled 4 times (900 tokens) from the SFT checkpoint the DPO run starts from: 720 answers, judged COMPLY 366 / REFUSE 160 / DEFLECT 134 / UNCLEAR 60. For each prompt a COMPLY answer is chosen and a REFUSE or DEFLECT answer rejected (≤2 pairs per prompt), giving 122 / 9 train/val pairs split by prompt. Because both sides come from the same starting policy, I expected them to differ mainly in whether the model did the task, not in style or length [[19]](#r19)[[22]](#r22).

### 7.4 Watch r<sub>c</sub>, not just accuracy

I tried the recipe on a toy 0.8B run first. With lr 5e-5 and no NLL term, preference accuracy hit 100% while the chosen answer's own implicit reward $r_c = \beta(\ell_c - \ell_c^{\text{ref}})$ fell to −14.5 and the model degenerated into repetition: it was winning by pushing *everything* down. Adding the NLL term (α = 1) and lowering lr to 1e-5 fixed it, which matches the failure mode described in DPOP [[23]](#r23). So the 27B runs use α = 1.

| from SFT step 450, lr 5e-6, 300 steps | val acc | val margin (max → final) | r<sub>c</sub> | comply (long gen) | refusals |
|---|---|---|---|---|---|
| β 0.1, α 1.0 | 0.94 | 7.3 → 6.1 | +9.0 | **74%** | 2/66 |
| β 0.3, α 1.0 | 0.89 | 24.5 → 19.4 | +26.2 | 65% | 9/66 |

> [!small]
> Validation sets are tiny (9–17 pairs), and these are single runs. The β 0.3 run was also checked on control (no false refusals) and harmful prompts (all refused); the β 0.1 run was not, which is a gap. Base model comply rate: 35%.

DPO took DEFLECT from 11–18 down to 6–7 out of 66. It is the only thing I tried that moved it.

## 8. Getting the model out of the notebook {#fuse}

### 8.1 The 4-bit fuse trap

To use the model outside my notebook I fused the adapter into the base and loaded it in LM Studio. Prompts that the step-450 checkpoint handled in the notebook (base + adapter computed separately) were refused again.

Cause: for a quantised base, `mlx_lm fuse` de-quantises to bf16, adds $\Delta W = s\,B^\top A^\top$, and *re-quantises to 4-bit*. Affine 4-bit quantisation stores, per group of 64 weights,

$$
\hat w = s_g\,\operatorname{round}\!\left(\frac{w - b_g}{s_g}\right) + b_g,\qquad
s_g = \frac{\max_g w - \min_g w}{2^{4} - 1},
$$

so each weight can only move in steps of $s_g$. In my adapters the LoRA delta is 2–7% of the weight norm, smaller than that step, and most of it was rounded away:

![Why re-quantising to 4-bit erases a LoRA delta](figures/requant.svg)
*Why re-quantising erases a LoRA. A small delta rounds back to the same 4-bit level; an 8-bit grid (16× finer) keeps it.*

| export | LoRA layer precision | size | cos(actual Δ, LoRA Δ) per layer | rel. error | target refusal |
|---|---|---|---|---|---|
| unfused (base + adapter) | separate | 16 + 0.2 GB | 1 | 0 | 23% (5/22) |
| `mlx_lm fuse` | 4-bit | 14 GB | 0.08 – 0.53 | 1.25 | 41% (9/22) |
| `fuse_lora.py` | 8-bit | 16 GB | 0.93 – 0.98 | 0.20 | **14% (3/22)** |

A relative error above 1 means the rounding error is larger than the signal. My fix keeps the other layers at 4-bit and stores only the 62 LoRA-carrying linear layers at 8-bit; mlx-lm already supports per-module bit widths in `config.json`, so loaders (including LM Studio's MLX engine) need no change. The problem is known in the PyTorch / bitsandbytes world [[24]](#r24)[[25]](#r25) and attacked from the training side by QA-LoRA [[26]](#r26) and Q-BLoRA [[27]](#r27). What I add is per-layer evidence on MLX, the behavioural consequence, and the observation that `mlx_lm fuse` does this without a warning. Whether it bites depends on how large the delta is relative to the quantisation step, so it will not hit every adapter equally.

### 8.2 Runtime details that changed behaviour {#runtime}

- **Thinking mode.** The stock chat template defaults to thinking at the highest effort. Training and evaluation used non-thinking mode, so the exported template is flipped to default non-thinking (rendering verified identical to the evaluation prompts; the original kept as `.orig`). Default sampling is set to T 0.7 / top-p 0.95 / top-k 20 to match evaluation.
- **Template caching.** LM Studio caches chat templates in its model index keyed on directory mtime; editing files inside a model directory does not invalidate it. Restart after template edits.
- **Vision.** mlx-lm loads only the language model and drops the vision tower and `vision_config` on save; LM Studio then treats the export as text-only. The exporter copies the base model's vision tower (333 bf16 tensors, 0.92&nbsp;GB) and config back.
- **Speculative decoding.** The vendor build of this model ships DFlash [[18]](#r18), a block-diffusion draft model (byte-identical to the public draft weights). Measured on the fused model (29.9 tok/s plain decoding):

![DFlash speculative decoding speed-up](figures/dflash.svg)
*DFlash speed-up relative to plain decoding, non-thinking mode. Chinese fiction accepts only ~1.5 draft tokens on average, so verification cost cancels the gain; with thinking on (gsm8k) it reaches 2.05×. LoRA and DPO did not reduce acceptance.*

## 9. In my agent, it broke again {#agent}

The fused DPO model behaved in LM Studio exactly as in the notebook. Inside my local agent, it still refused or broke. I have not fully resolved this, but three things differ from the training setup, and I suspect all three (these are hypotheses, not verified causes):

1. **Prompt format.** Training used no system prompt. The agent sends a long system prompt (rules, a tool catalogue, project memory) plus tool definitions that the chat template renders into the system turn. This is the one I noticed first: the same request behaved differently depending on what was in the system prompt.
2. **Constrained decoding.** Every agent stage requests `response_format: json_schema` (strict), so the runtime forces schema-valid JSON, and a refusal can only appear inside a JSON field.
3. **Length.** SFT and DPO saw ≤1024-token sequences. One 16.5k-token input made the DPO model collapse into a repetition loop of a hallucinated status line (not present in the input, the data or the agent code), with top-p and min-p disabled. The base model handles long contexts; my short-sequence LoRA apparently does not.

That changed the question for me. Training on more prompt formats and longer sequences is one answer, and the kernel from §3.1 now makes 8192-token training possible. But I also wanted something that does not depend on the prompt format at all. That brought me back to refusal-direction ablation.

## 10. Refusal-direction ablation {#ablation}

I had first tried ablation early on, only as a no-training baseline for the SFT runs (the "orth@24" rows in sections 5 and 6). The agent problem is what made me take it seriously. It edits the model's internal representation rather than teaching it a new response for one prompt format, so in principle it should carry over to any system prompt.

Arditi et al. [[9]](#r9) showed that refusal is largely mediated by one direction in the residual stream. Take the residual activation $h_\ell(p)$ at the last prompt token after layer $\ell$ and compute difference-of-means directions over matched prompt sets:

$$
r^{\text{target}}_\ell = \overline{h_\ell}(\text{target}) - \overline{h_\ell}(\text{harmless}),\qquad
r^{\text{harm}}_\ell = \overline{h_\ell}(\text{harmful}) - \overline{h_\ell}(\text{harmless})
$$

(30 matched target/harmless pairs, 32 harmful prompts). I didn't want to release harmful requests along with the target category, and the plain direction does both, because the two directions overlap, more so in later layers:

![Cosine between the target-category and general-harm directions per layer](figures/cos.svg)
*Cosine similarity between the target-category and general-harm directions, per layer, for my prompt sets. Near zero in the first third, ~0.3–0.5 in the middle, ~0.68 at the top: in this model, "refuse this category" and "refuse harmful things" become harder to separate the deeper you go.*

So I orthogonalised the target direction against the general-harm direction before use:

$$
u = \frac{r^{\text{harm}}_\ell}{\lVert r^{\text{harm}}_\ell\rVert},\qquad
r^{\perp}_\ell = r^{\text{target}}_\ell - \left(r^{\text{target}}_\ell\cdot u\right)u,\qquad
d = \frac{r^{\perp}_{\ell^*}}{\lVert r^{\perp}_{\ell^*}\rVert}
$$

At inference a hook removes $d$ from the input of layer 0 and from the output of every layer: $x \leftarrow x - (x\cdot d)\,d$. The layer $\ell^*$ was chosen by a sweep over $[L/4, 3L/4]$ in steps of 4 on 10 held-out target prompts:

![Layer sweep for the ablation direction](figures/sel.svg)
*Layer sweep (keyword detector, 48 tokens, small held-out sets, so coarse). Solid: target-category refusal (lower is better). Dashed: share of harmful prompts still refused (higher is better). The plain direction from layers 44–48 also releases harmful requests (dashed orange collapses); the orthogonalised one does not.*

| method | target refusal | control | harmful (2 + 12) | training |
|---|---|---|---|---|
| Base model | 59% | 0% | all refused | — |
| LoRA step 150 | 23% | 0% | all refused | ~15 min |
| Ablation orth@24 | **18%** | 0% | all refused (judge 12/12) | none |
| Ablation plain, late layers | — | — | harmful released too | none |

Without any training, the orthogonalised direction lowered target refusals about as much as SFT did, and kept harmful requests refused. As section 6 showed, it does not fix deflection on its own.

## 11. Baking the direction into the weights, and the gate that said no {#bake}

A hook only works inside my own Python. For LM Studio and the agent, the ablation has to live in the weights. Only two kinds of modules ever *write* to the residual stream: the token embedding and the output projection of each sub-layer. If none of them can write a $d$-component, the stream never contains one, which is exactly what the hook enforces.

![Residual stream writers that get orthogonalised](figures/writers.svg)
*Residual-stream writers in this architecture: the embedding, plus per layer the attention / DeltaNet output projection and the MLP down projection. The final norm and lm_head only read.*

$$
E \leftarrow E - (E d)\,d^\top \qquad (E\in\mathbb{R}^{V\times D})
$$

$$
W \leftarrow W - d\,(d^\top W) = (I - d d^\top)\,W \qquad (W\in\mathbb{R}^{D\times d_{in}})
$$

In this model that is 129 matrices (embedding, 64 down projections, 16 attention `o_proj`, 48 DeltaNet `out_proj`), about 9.0B parameters, a third of the model. Having just been bitten by §8.1, I stored the modified matrices at 8-bit and kept the rest at 4-bit (20.1&nbsp;GB of weights instead of 16). Each matrix is de-quantised, projected, re-quantised and released in turn, so the export never holds the model in bf16. (This equivalence relies on the model not tying its embedding to the output head; Qwen3.8-27B doesn't.)

| check | result |
|---|---|
| d-component left after 8-bit re-quantisation, $\lVert d^\top \hat W\rVert / \lVert d^\top W\rVert$ | mean 0.036, max 0.048 |
| KL(hook ‖ baked), last-token distribution, 8 ordinary prompts | **0.0004** (vs 0.021 for base ‖ hook) |
| top-1 agreement hook vs baked | 100% |
| 12 harmful prompts refused (greedy, 160 tokens, keyword detector) | 11 / 12 † |
| **protected-category probe** (prompts involving minors; greedy + 2 samples) | <span class="bad">**4 / 24 not refused** — base model: 0 / 8</span> |

> [!small]
> † The hook version scored 12/12 in §10 with sampled answers and the model judge; this check is greedy with the stricter keyword detector. One prompt (a poison-extraction question) was not refused here.

The bake itself worked: the exported model behaves like the hook version. But before exporting anything I had added a hard gate: *every request involving minors must be refused, otherwise no link to LM Studio is created.* It failed. The failures concentrated on two prompts, one framed as a short-drama script. Since the baked model matches the hook, the problem is in the direction itself, not in the export. (The detector is keyword-based and counts deflections as "not refused"; the answers in this probe are never written to disk or displayed, so I cannot split the four further. The gate is deliberately strict.)

So the most promising answer to the agent problem is, for now, not shippable. In this case, orthogonalising against an averaged "general harm" direction did not protect a refusal I care about: the harmful set that defined $r^{\text{harm}}$ did not contain this category, and in late layers the target direction overlaps with it. My next step is an explicit constraint: extract a direction from a held-out set of protected-category prompts, project it out of $d$ as well, and re-run the same gate on a disjoint probe set. Until it passes, the baked model stays off my machine's model list, and I have not been able to test it inside the agent.

## 12. What I take away {#lessons}

These are habits I'm keeping, based on one model and small samples. They worked for me here; I would not call them rules.

1. Choose checkpoints by behaviour rather than loss, and look at long generations before believing a number.
2. Treat a keyword detector as a lower bound, and go back to the raw outputs whenever a conclusion changes.
3. Count deflection separately from refusal. In my runs, removing refusals (by ablation or light SFT) partly turned them into tamer answers, and only a contrast signal (DPO on the model's own deflections) reduced those.
4. Re-test behaviour on the exported files, not on the adapter in the notebook. Fusing a small LoRA delta back into a 4-bit base can erase it.
5. Measure memory with the framework's real peak; the process footprint includes caches.
6. Benchmark speed on the real, variable-length data. Fixed lengths hid compile retracing for me (105 vs 80 tok/s on the same path).
7. In DPO, watch r<sub>c</sub>: when the chosen answer's implicit reward drifts negative, the model may be winning by pushing everything down.
8. Expect deployment defaults to change behaviour: thinking mode, temperature, system prompts, cached templates, structured output, context length.
9. Put hard safety gates in the export path, not in a checklist. Mine caught a regression I would otherwise have shipped.

## 13. Limitations and ethics {#limits}

- **Small samples.** 22 target prompts, 3 long generations each; differences within 5 points are noise (single-proportion standard error ≈ 9 points at n = 22, and the 3 seeds per prompt are highly correlated). Most comparisons are single runs.
- **One judge.** The judge is the base model itself; no human agreement study.
- **No capability benchmarks.** Only 8 control prompts check that nothing broke; no MMLU / C-Eval / writing-quality evaluation. This is the largest gap.
- **Keyword safety filter.** It cannot catch non-consent that is only implied by the narrative. Any release of data or weights would need a classifier pass. I release neither.
- **One model, one quantisation** (affine 4-bit, group 64). None of this is tested on other architectures.
- **Dual use.** Refusal-direction ablation can equally be used to release harmful categories. The orthogonalisation, the always-evaluated harmful set and the export gate exist to keep changes inside the target category, and §11 shows that even this was not sufficient here. No direction, adapter, prompt set or modified weights are published; the [code](https://github.com/thugasin5214/mlx-post-training) is, with the minors filter and the export gate built in.

Still, I'm very happy with where this ended up: I now have a local model that fits my current use cases, and a clear path forward for future requirements. It is my go-to local model now, and everything about it is under my control.

## References {#refs}

1. [Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. 2021.](https://arxiv.org/abs/2106.09685)
2. [Dettmers et al. QLoRA: Efficient Finetuning of Quantized LLMs. 2023.](https://arxiv.org/abs/2305.14314)
3. [Biderman et al. LoRA Learns Less and Forgets Less. 2024.](https://arxiv.org/abs/2405.09673)
4. [Loshchilov & Hutter. Decoupled Weight Decay Regularization. 2017.](https://arxiv.org/abs/1711.05101)
5. [Loshchilov & Hutter. SGDR: Stochastic Gradient Descent with Warm Restarts. 2016.](https://arxiv.org/abs/1608.03983)
6. [Chen et al. Training Deep Nets with Sublinear Memory Cost. 2016.](https://arxiv.org/abs/1604.06174)
7. [Zhou et al. LIMA: Less Is More for Alignment. 2023.](https://arxiv.org/abs/2305.11206)
8. [Qi et al. Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To! 2023.](https://arxiv.org/abs/2310.03693)
9. [Arditi et al. Refusal in Language Models Is Mediated by a Single Direction. 2024.](https://arxiv.org/abs/2406.11717)
10. [Yang, Kautz & Hatamizadeh. Gated Delta Networks: Improving Mamba2 with Delta Rule. 2024.](https://arxiv.org/abs/2412.06464)
11. [Yang et al. Parallelizing Linear Transformers with the Delta Rule over Sequence Length. 2024.](https://arxiv.org/abs/2406.06484)
12. [Qiu et al. Gated Attention for Large Language Models. 2025.](https://arxiv.org/abs/2505.06708)
13. [Qwen Team. Qwen3 Technical Report. 2025.](https://arxiv.org/abs/2505.09388)
14. [Rafailov et al. Direct Preference Optimization: Your Language Model is Secretly a Reward Model. 2023.](https://arxiv.org/abs/2305.18290)
15. [Azar et al. A General Theoretical Paradigm to Understand Learning from Human Preferences. 2023.](https://arxiv.org/abs/2310.12036)
16. [Pang et al. Iterative Reasoning Preference Optimization. 2024.](https://arxiv.org/abs/2404.19733)
17. [Hong et al. ORPO: Monolithic Preference Optimization without Reference Model. 2024.](https://arxiv.org/abs/2403.07691)
18. [Chen, Liang, Liu et al. DFlash: Block Diffusion for Flash Speculative Decoding. 2026.](https://arxiv.org/abs/2602.06036)
19. [Tajwar et al. Preference Fine-Tuning of LLMs Should Leverage Suboptimal, On-Policy Data. 2024.](https://arxiv.org/abs/2404.14367)
20. [There Is More to Refusal in Large Language Models than a Single Direction. 2026.](https://arxiv.org/abs/2602.02132)
21. [unsloth-zoo PR #754. MLX: fix gated-delta (Qwen3.5) training and add fused backward kernels. 2026.](https://github.com/unslothai/unsloth-zoo/pull/754)
22. [Park et al. Disentangling Length from Quality in Direct Preference Optimization. 2024.](https://arxiv.org/abs/2403.19159)
23. [Pal et al. Smaug: Fixing Failure Modes of Preference Optimisation with DPO-Positive. 2024.](https://arxiv.org/abs/2402.13228)
24. [The Kaitchup. Don't Merge Your LoRA Adapter Into a 4-bit LLM.](https://kaitchup.substack.com/p/dont-merge-your-lora-adapter-into)
25. [The Kaitchup. Training, Loading, and Merging QDoRA, QLoRA, and LoftQ Adapters.](https://kaitchup.substack.com/p/training-loading-and-merging-qdora)
26. [Xu et al. QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models. 2023.](https://arxiv.org/abs/2309.14717)
27. [Accurate and Efficient Fine-Tuning of Quantized Large Language Models Through Optimal Balance (Q-BLoRA). 2024.](https://arxiv.org/abs/2407.17029)
28. [From Refusal Tokens to Refusal Control: Discovering and Steering Category-Specific Refusal Directions. 2026.](https://arxiv.org/abs/2603.13359)
29. [TeleBoost: A Systematic Alignment Framework for High-Fidelity, Controllable, and Robust Video Generation. 2026.](https://arxiv.org/abs/2602.07595)
30. [GroupDPO: Memory-Efficient Group-Wise Direct Preference Optimization. 2026.](https://arxiv.org/abs/2604.15602)
