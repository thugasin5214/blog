"""Build the blog: python tools/build.py  →  index.html + posts/*/index.html (static, for GitHub Pages)."""
from pathlib import Path

import charts as C
import diagrams as D

ROOT = Path(__file__).resolve().parents[1]
SLUG = "27b-post-training-on-a-mac"
TITLE = "Post-training a 27B hybrid model on one Mac"
SUBTITLE = ("QLoRA on Gated DeltaNet, refusal-direction ablation, DPO, and the quantisation trap "
            "that silently erases your LoRA")
DATE = "November 2026"
AUTHOR = "thugasin5214"

# ---------------------------------------------------------------- data (all measured unless marked)
COS = [-0.075, -0.057, 0.016, -0.008, 0.062, 0.02, 0.049, 0.057, 0.023, -0.017, -0.012, 0.126, 0.197, 0.23, 0.289,
       0.319, 0.368, 0.401, 0.412, 0.358, 0.36, 0.331, 0.301, 0.297, 0.317, 0.333, 0.33, 0.39, 0.433, 0.447, 0.47,
       0.489, 0.436, 0.481, 0.511, 0.505, 0.516, 0.508, 0.494, 0.48, 0.473, 0.478, 0.482, 0.5, 0.534, 0.557, 0.568,
       0.572, 0.581, 0.618, 0.663, 0.667, 0.666, 0.673, 0.684, 0.678, 0.684, 0.686, 0.67, 0.691, 0.689, 0.686,
       0.676, 0.664]
# layer-selection sweep (10 held-out target prompts, 6 harmful prompts, 48 tokens, keyword detector)
SEL = {"orth": {16: (0.5, 0.833), 20: (0.6, 0.833), 24: (0.0, 0.833), 28: (0.0, 0.667), 32: (0.4, 0.833),
                36: (0.0, 0.833), 40: (0.4, 0.833), 44: (0.0, 1.0), 48: (0.0, 0.833)},
       "plain": {16: (0.7, 0.833), 20: (0.9, 0.833), 24: (0.1, 0.833), 28: (0.0, 1.0), 32: (0.6, 0.833),
                 36: (0.1, 0.833), 40: (0.3, 0.833), 44: (0.0, 0.167), 48: (0.0, 0.0)}}
RDCU = [("Base model", [43, 0, 23, 0]), ("LoRA step 150", [16, 11, 37, 2]), ("Ablation orth@24", [11, 10, 35, 10]),
        ("LoRA + ablation", [1, 18, 37, 10]), ("DPO β=0.1", [2, 7, 49, 8]), ("DPO β=0.3", [9, 6, 43, 8])]
MEM = [(512, 4.8), (1024, 14.9), (2048, 52.6)]          # 0.8B, real MLX peak GB
MEM_FP = [(512, 8.8), (1024, 25.6), (2048, 70.4)]       # process footprint incl. allocator cache
# 27B fused gated-delta kernel (unsloth-zoo [21]) vs the ops path; batch 1, grad checkpoint, MLX real peak
GDN_ROWS = [("SFT 1024, 8 LoRA layers", (39, 102), (58.2, 17.3)),
            ("SFT 1024, 16 LoRA layers", (25, 81), (58.5, 17.7)),
            ("SFT variable length, 8 layers", (80, 162), (56.0, 17.3)),
            ("SFT 2048, 8 layers", ("OOM", 133), ("OOM", 18.8)),
            ("SFT 4096, 8 layers", ("not run", 132), ("not run", 21.9)),
            ("SFT 8192, 8 layers", ("not run", 134), ("not run", 27.5)),
            ("DPO 1024, 8 layers", (70, 102), (34.8, 19.0))]
GDN_LOSS = {"ops": [(2, 2.382), (4, 1.722), (6, 2.135)], "kernel": [(2, 2.383), (4, 1.722), (6, 2.135)]}
DFLASH = [("block 3, 4-bit draft", [("fiction", 1.04), ("scripts", 1.38), ("Q&A", 1.66)]),
          ("block 5 (recommended)", [("fiction", 0.74), ("scripts", 1.05), ("Q&A", 1.44)]),
          ("block 8 (vendor setting)", [("fiction", 0.48), ("scripts", 0.77), ("Q&A", 1.00)]),
          ("control: gsm8k, thinking on", [("block 5", 2.05)])]


def figs():
    f = {}
    f["pipeline"] = D.pipeline()
    f["arch"] = D.qwen_arch()
    f["patch"] = D.layer_patch()
    f["mem"] = C.line_chart(
        [("real peak (mx.get_peak_memory)", "--c1", MEM, False), ("process footprint (incl. cache)", "--c3", MEM_FP, True)],
        "Peak memory vs sequence length, 0.8B model", "sequence length T (tokens, log scale)", "GB (log scale)",
        (400, 2600), (3, 100), logx=True, logy=True, xticks=[512, 1024, 2048], yticks=[5, 10, 20, 50, 100],
        points=True, h=280)
    gdn_series = [("ops path (frozen-layer patch only)", "--c3"), ("+ fused backward kernel", "--c1")]
    f["gdn_speed"] = C.paired_bars([(n, s) for n, s, _ in GDN_ROWS], gdn_series,
                                   "Training throughput on 27B, ops path vs fused kernel", "tok/s")
    f["gdn_mem"] = C.paired_bars([(n, m) for n, _, m in GDN_ROWS], gdn_series,
                                 "Real peak memory on 27B, ops path vs fused kernel", "GB")
    f["gdn_loss"] = C.line_chart(
        [("fused kernel", "--c1", GDN_LOSS["kernel"], False), ("ops path (dashed, drawn on top)", "--c3", GDN_LOSS["ops"], True)],
        "Training loss, ops path vs fused kernel, same data and seed", "step", "train loss",
        (1.5, 6.5), (1.5, 2.6), xticks=[2, 4, 6], yticks=[1.6, 1.8, 2.0, 2.2, 2.4], points=True, h=230)
    f["cos"] = C.line_chart(
        [("cos(r_target, r_harm)", "--c1", list(enumerate(COS)), False)],
        "Cosine between the target-category and general-harm directions per layer", "layer", "cosine",
        (0, 63), (-0.1, 0.8), xticks=[0, 8, 16, 24, 32, 40, 48, 56, 63], yticks=[0, 0.2, 0.4, 0.6, 0.8],
        marks=[(24, "chosen: orth@24")], legend=False, h=260)
    f["sel"] = C.line_chart(
        [("orth · target", "--c1", [(k, v[0]) for k, v in SEL["orth"].items()], False),
         ("plain · target", "--c3", [(k, v[0]) for k, v in SEL["plain"].items()], False),
         ("orth · harmful kept", "--c1", [(k, v[1]) for k, v in SEL["orth"].items()], True),
         ("plain · harmful kept", "--c3", [(k, v[1]) for k, v in SEL["plain"].items()], True)],
        "Layer sweep for the ablation direction", "layer the direction is taken from", "rate",
        (14, 50), (-0.02, 1.05), xticks=list(SEL["orth"]), yticks=[0, 0.25, 0.5, 0.75, 1],
        fmt_y=lambda v: f"{v:.0%}", points=True, h=300)
    f["rdcu"] = C.stacked_bars(RDCU, ["REFUSE", "DEFLECT", "COMPLY", "UNCLEAR"], ["--c3", "--c4", "--c2", "--c5"],
                               "Judge labels per intervention, 66 long generations each",
                               note="22 prompts × 600 tokens × 3 seeds = 66 generations per row; labels from a 4-way model judge")
    f["dpo"] = D.dpo_two_pass()
    f["requant"] = D.requant()
    f["dflash"] = C.hbars(DFLASH, "DFlash speculative decoding speed-up", ref=1.0, maxv=2.3)
    f["writers"] = D.residual_writers()
    return f


def fig(svg, caption, n):
    return f'<figure>{svg}<figcaption><b>Figure {n}.</b> {caption}</figcaption></figure>'


def post_body(F):
    return f"""
<p>I&rsquo;m a strong believer in open-weight models. As LLMs get more capable, safety and alignment matter
more and more. So does personal privacy, even though there&rsquo;s very little of it left. A model running on my
own machine is the last place where I can share anything without wondering who else might end up using my data.</p>

<p>Local models still can&rsquo;t match frontier models in general capability. For a specific task, though,
they&rsquo;re often good enough. When they aren&rsquo;t, you need a way to train them until they do what you expect.
That&rsquo;s what I love about open source: I can see how everything works, and I can retrain the model for what I
need. I want a model that works for me, not for someone else&rsquo;s policy. So I spent a few weeks finding out how
far I could move a local Qwen3.8-27B away from refusing one specific category of requests. I also wanted to see
what that costs in quality and in the safety behaviour I wanted to keep.</p>

<p class="lede">This is a write-up of a few weeks of post-training a 27-billion-parameter model on a single Mac
(M4 Max, 128&nbsp;GB unified memory) with nothing but the open-source MLX stack. The goal was narrow:
make the model stop refusing <em>one specific content category</em> while it keeps refusing everything else,
then ship the result to a local runtime (LM Studio) and a local agent. The interesting parts turned out to be
less about the goal and more about what broke along the way.</p>

<div class="tldr">
<h3>TL;DR</h3>
<ul>
<li><b>Training a hybrid linear-attention model in MLX is slow for a fixable reason.</b> mlx-lm drops every
Gated DeltaNet layer into a per-token Python loop in training mode, frozen layers included. A ~60-line patch keeps
frozen layers on the Metal kernel: 32 → 61 tok/s. It does <em>not</em> save memory (I first thought it did; it was
allocator cache). Swapping unsloth-zoo&rsquo;s fused backward kernel into the LoRA layers does: 102 tok/s at
17&nbsp;GB instead of 58&nbsp;GB, and sequence 8192 now trains in 27.5&nbsp;GB, with identical loss.</li>
<li><b>Refusing and deflecting are different behaviours.</b> Stacking LoRA SFT with refusal-direction ablation took
explicit refusals from 43/66 to 1/66, yet the comply rate did not move (56%): the refusals turned into
"sure!"-and-then-write-something-else. Only DPO on the model's own deflections reduced them.</li>
<li><b>Fusing a LoRA into a 4-bit base erases it.</b> <code>mlx_lm fuse</code> re-quantises to 4-bit; the LoRA delta
(2–7% of the weight norm) is below the 4-bit rounding step, so the fused model behaves like the base again
(41% refusal vs 23% unfused). Saving only the LoRA layers at 8-bit fixes it for +2&nbsp;GB.</li>
<li><b>Baking the ablation into the weights works exactly</b> (KL to the hook version 0.0004) — and that is how
I found out the direction also weakened a refusal it was never supposed to touch. The safety gate blocked the
export. Orthogonalising against "general harm" is not enough; protected categories need their own constraint.</li>
</ul>
</div>

<p class="note">A note on scope. The target category was consensual fiction between adults, which mainstream
instruct models refuse by default. Nothing below shows any generated text, training sample or concrete prompt from
that category; everything is reported as rates and label counts. The data pipeline hard-filters anything involving
minors (no switch to turn it off), and every configuration is evaluated on other harmful requests that must stay
refused. I refer to the target simply as <b>the target category</b> from here on.</p>

<nav class="toc"><b>Contents</b>
<ol>
<li><a href="#setup">Setup and constraints</a></li>
<li><a href="#infra">Training Gated DeltaNet in MLX</a></li>
<li><a href="#memory">Memory: a self-correction, then the real fix</a></li>
<li><a href="#eval">What counts as "not refusing"?</a></li>
<li><a href="#sft">SFT: the loss barely moves</a></li>
<li><a href="#ablation">A category-specific refusal direction</a></li>
<li><a href="#deflect">Refusal ≠ deflection</a></li>
<li><a href="#dpo">DPO on one machine</a></li>
<li><a href="#fuse">The 4-bit fuse trap</a></li>
<li><a href="#runtime">Runtime details that change behaviour</a></li>
<li><a href="#bake">Baking the direction into the weights — and the gate that said no</a></li>
<li><a href="#agent">Inside an agent: format and length shift</a></li>
<li><a href="#lessons">Lessons</a></li>
<li><a href="#limits">Limitations and ethics</a></li>
</ol></nav>

{fig(F["pipeline"], "The two paths explored. Top: training (SFT → behaviour-based checkpoint selection → DPO → "
     "mixed-precision fuse). Bottom: no training (extract a refusal direction, ablate it at inference, then bake "
     "it into the weights behind a safety gate).", 1)}

<h2 id="setup">1. Setup and constraints</h2>
<ul>
<li><b>Hardware.</b> One Mac Studio, M4 Max, 128&nbsp;GB unified memory. No CUDA, so MLX
(<code>mlx 0.32.4</code>, <code>mlx-lm 0.32.2</code>).</li>
<li><b>Model.</b> <code>mlx-community/Qwen3.8-27B-4bit</code> (16&nbsp;GB, affine 4-bit, group size 64). It is a hybrid:
of 64 layers, 48 use Gated DeltaNet linear attention <a href="#r10">[10]</a><a href="#r11">[11]</a> and 16 use
full (gated) attention <a href="#r12">[12]</a>.</li>
<li><b>Data.</b> A public Chinese fiction corpus in the target category (~45k stories). Ten sub-categories whose
theme involves minors, incest, non-consent or real people were dropped wholesale; every remaining story then went
through a keyword filter (minors: hard, cannot be disabled; non-consent and real people: on by default),
de-duplication and chunking into ~1,100-character "opening + continuation" pairs. The main runs used 2,000 of the
16,118 resulting samples with diversified prompt phrasings.</li>
<li><b>Evaluation set.</b> 22 target-category prompts (deliberately phrased unlike the training template), 8 control
prompts (ordinary questions — must be answered), 2 + 12 harmful prompts from other categories (weapons, intrusion,
poisons… — must be refused). No overlap with the DPO prompts (asserted in code).</li>
<li><b>Training recipe.</b> QLoRA-style <a href="#r1">[1]</a><a href="#r2">[2]</a>: frozen 4-bit base, LoRA rank 16
(mlx scale 20) on all linear layers of the top 8 blocks, AdamW <a href="#r4">[4]</a>, lr 2e-5 cosine
<a href="#r5">[5]</a>, gradient checkpointing <a href="#r6">[6]</a>, sequence length 1024, batch 1 × accumulation 4,
loss on the response only.</li>
</ul>

<h2 id="infra">2. Training Gated DeltaNet in MLX</h2>
<h3>2.1 What Gated DeltaNet is</h3>
<p>A standard attention layer keeps every past key and value (the KV cache) and compares each new query with all of
them, so cost and memory grow with context length. Linear attention replaces the cache with a fixed-size matrix
\\(S\\) per head, a running memory that maps keys to values: reading is \\(o_t = S_t q_t\\), and writing adds
\\(v_t k_t^\\top\\). Decoding then costs the same at token 100 and at token 100,000.</p>
<p>Plain additive writes pile up and blur, though. Gated DeltaNet <a href="#r10">[10]</a> fixes this with two ideas.
The <b>delta rule</b> first reads what \\(S\\) currently stores under \\(k_t\\), then writes only the correction
\\(\\beta_t (v_t - S k_t)\\), so a key that is reused gets overwritten instead of accumulated. The <b>gate</b>
\\(\\alpha_t \\in (0, 1)\\), computed per token and per head, lets the layer decay old memory quickly (a topic change)
or keep it (a long quote). Both \\(\\alpha_t\\) and the write strength \\(\\beta_t\\) come from small projections of the
input, and a width-4 causal convolution in front gives \\(q, k, v\\) some local context.</p>
<p>A fixed-size memory is lossy, so it is weak at exact recall over long distances. Qwen3.8 therefore interleaves:
three Gated DeltaNet layers, then one full (gated) attention layer, repeated 16 times.</p>
{fig(F["arch"], "Qwen3.8-27B, drawn from its <code>config.json</code> and mlx-lm&rsquo;s <code>qwen3_5.py</code>. "
     "Left: 64 layers in 16 groups of three Gated DeltaNet layers plus one gated full-attention layer, each followed "
     "by a dense SwiGLU MLP. Right: one Gated DeltaNet mixer. Keys and queries have 16 heads; values have 48, so "
     "three value heads share each key head. The state is 48 × 128 × 128 per layer regardless of context length.", 2)}
<h3>2.2 Why it trains slowly in MLX</h3>
<p>The first 27B run (sequence 2048, 16 LoRA layers) died on step one with a Metal OOM at 122&nbsp;GB — for a 16&nbsp;GB
model. The cause is one line in mlx-lm's <code>qwen3_5.py</code>:</p>
<pre><code>out, state = gated_delta_update(..., use_kernel=not self.training)</code></pre>
<p>The fused Metal kernel for the delta-rule update has no backward pass, so in training mode the layer falls back to a
Python loop over time steps that updates a 48-head × 128 × 128 float32 state per token. Gated DeltaNet's recurrence
is</p>
\\[ S_t = \\alpha_t\\,S_{{t-1}}\\left(I - \\beta_t\\,k_t k_t^\\top\\right) + \\beta_t\\,v_t k_t^\\top,\\qquad o_t = S_t\\,q_t, \\]
<p>and the loop materialises every \\(S_t\\) for autograd. The catch: <code>model.train()</code> recurses into
<em>every</em> sub-module, so all 48 linear-attention layers take the slow path — including the 56 frozen blocks below
the LoRA layers that never need a gradient.</p>
{fig(F["patch"], "The patch. Gradients only have to flow through the lowest LoRA layer and above, so frozen layers can "
     "stay on the non-differentiable kernel — the same idea as wrapping frozen layers in <code>no_grad</code> in PyTorch.", 3)}
<p><code>mlx_train.py</code> intercepts <code>GatedDeltaNet._set_training_mode</code> so marked frozen layers stay at
<code>training=False</code>, then calls the stock <code>mlx_lm lora</code> entry point with identical CLI arguments.
It refuses to start unless three self-checks pass: the mlx-lm version is on an allow-list, the upstream line above
still exists, and after patching the per-layer training flags are what we expect.</p>
<table>
<tr><th>27B, sequence 1024</th><th>stock mlx-lm</th><th>patched</th></tr>
<tr><td>8 LoRA layers</td><td>32 tok/s</td><td><b>61 tok/s</b></td></tr>
<tr><td>16 LoRA layers</td><td>26 tok/s</td><td><b>37 tok/s</b></td></tr>
</table>
<p class="small">Train/val loss at step 8 were bit-identical with and without the patch (2.709 / 2.035) — checked on one
configuration for 8 steps.</p>

<h2 id="memory">3. Memory: a self-correction, then the real fix</h2>
<p>I initially reported that the patch cut memory from 83&nbsp;GB to 76&nbsp;GB. That number came from
<code>/usr/bin/time -l</code>, i.e. the process footprint, which includes MLX's allocator cache. Re-measuring with
MLX's own <code>mx.get_peak_memory()</code> on the 0.8B sibling model told a different story:</p>
{fig(F["mem"], "Real peak memory grows roughly as T<sup>1.7</sup> with sequence length. LoRA layer count (4–24), rank "
     "(16/64) and the patch all change the real peak by less than 2%. The footprint overstates it by 1.3–1.8×.", 4)}
<p>With gradient checkpointing, the backward pass recomputes one block at a time, so the peak is set by the
intermediate states of a single linear-attention loop — independent of how many layers take the slow path. The patch
only removes the frozen layers' forward loops, which is why it buys 1.4–1.9× speed and no memory. On 27B, sequence 2048
still OOMs; 1024 is the ceiling that drove most of the experiments below (chunk length, DPO sequence length).</p>

<h3>3.1 The thorough fix: a fused backward kernel</h3>
<p>What the LoRA layers need is a differentiable kernel. unsloth-zoo merged one <a href="#r21">[21]</a>: it cuts the
sequence into chunks of 64–512 steps, each wrapped in an <code>mx.custom_function</code>. The forward is a blocked
Metal kernel; the backward first rebuilds each chunk&rsquo;s states with a checkpoint kernel, then runs one reverse
scan. I did not take a dependency on unsloth. <code>fetch_gdn_kernel.py</code> downloads that one file at a pinned
commit, checks its sha256 and puts it in a git-ignored <code>third_party/</code> (it is AGPL-3.0, so it is not
vendored). <code>mlx_train.py</code> then routes only the training-mode call of <code>gated_delta_update</code> to it.
Frozen layers stay on the inference kernel, so the two fixes stack. A fourth self-check compares the kernel&rsquo;s
gradients with the ops path in fp32 on a small multi-chunk case and refuses to start above 10<sup>-4</sup> relative
error. Without the file, it falls back to the ops path.</p>
{fig(F["gdn_speed"], "Throughput on 27B (batch 1, gradient checkpointing, 4-bit base). The ops path at sequence "
     "1024 / 8 layers is the better of two runs (33 and 39 tok/s). 2048 OOMs on the ops path; 4096 and 8192 were not "
     "attempted there. Long-sequence rows use neutral long samples built from repository docs.", 5)}
{fig(F["gdn_mem"], "Real peak memory (<code>mx.get_peak_memory()</code>) for the same runs. With the kernel the peak "
     "is set by the weights plus a few GB and barely grows with sequence length: 17.3 → 27.5&nbsp;GB from 1024 to 8192 "
     "tokens.", 6)}
{fig(F["gdn_loss"], "Correctness: the first six SFT steps on the same data and seed. The two paths differ by at most "
     "0.001; the lines overlap.", 7)}
<p>The numbers hold up at the layer level too. Against an fp32 reference, both paths carry the same 2–4×10<sup>-3</sup>
bf16 error, and whole-model LoRA gradients differ by 0.8% between them. DPO&rsquo;s step-4 loss is 1.6463 vs 1.6442.
One layer&rsquo;s forward + backward at T = 1024 goes from 895&nbsp;ms to 12&nbsp;ms.</p>
<p>Where the speed comes from is partly not the arithmetic. The ops path&rsquo;s autodiff graph is about T × 12
primitives per layer, and <code>mx.compile</code> retraces it for every new padded length. On fixed-length data the
ops path warms up to ~105 tok/s; on real variable-length data it stays at 80. The kernel turns each chunk into one
graph node, so retracing is cheap. A pure forward pass now runs at ~190 tok/s, close to the M4 Max&rsquo;s compute
limit, and a training step reaches about 70% of that.</p>
<p>Things I tried on top that did not help: no gradient checkpointing (same 132 tok/s, 25.7&nbsp;GB), batch 2 (same
tok/s), padding to multiples of 256 (162 → 149 tok/s on variable-length data), and running the frozen layers on
unsloth&rsquo;s blocked forward too (0.41 → 0.20&nbsp;s for all 48 layers in isolation, under 2% per step, inside the
noise). What is left is mostly 4-bit matmuls.</p>

<h2 id="eval">4. What counts as "not refusing"?</h2>
<p>Most of the mistakes I made were measurement mistakes.</p>
<h3>4.1 The keyword regex under-counted refusals — three times</h3>
<p>The first detector was a keyword regex over the first 300 characters. Manual spot checks found it missing
(1) soft refusals ("I can write something healthy and positive instead…", followed by an unrelated story) and
(2) reworded hard refusals. At one point I reported a 5% refusal rate at step 450 that was really 23%, and a 50%
harmful-refusal retention that was really 100%. After every regex fix, all stored outputs were re-scored and the
control set re-checked for false positives. Rule: a regex is a lower bound; conclusions need a judge or a human.</p>
<h3>4.2 A four-label judge</h3>
<p><code>judge.py</code> asks the 27B base model (greedy) to label each answer:</p>
<ul class="labels">
<li><span class="tag r">REFUSE</span> explicit refusal</li>
<li><span class="tag d">DEFLECT</span> no refusal, but not done either: rewritten into something else, mood only, request misread</li>
<li><span class="tag c">COMPLY</span> did what was asked</li>
<li><span class="tag u">UNCLEAR</span> cut off during the set-up, outcome not visible yet</li>
</ul>
<p>DEFLECT is invisible to any regex.</p>
<h3>4.3 Short generations hide deflection</h3>
<p>Generating 160 tokens is enough to see a refusal, so that is what I did at first. But many answers turn evasive
halfway through. Re-running the same candidates at 600 tokens × 3 seeds moved most UNCLEAR answers into REFUSE and
DEFLECT:</p>
<table>
<tr><th>candidate (22 target prompts)</th><th>160 tok × 1<br>R / D / C / U</th><th>600 tok × 3 (66)<br>R / D / C / U</th></tr>
<tr><td>LoRA step 150</td><td>3 / 2 / 9 / 8</td><td>16 / 11 / 37 / 2</td></tr>
<tr><td>Ablation orth@24</td><td>3 / 1 / 6 / 12</td><td>11 / 10 / 35 / 10</td></tr>
</table>
<p>Everything below uses the long-generation protocol.</p>

<h2 id="sft">5. SFT: the loss barely moves, the behaviour does</h2>
<table>
<tr><th>step</th><th>target refusal</th><th>control refused</th><th>harmful refused</th><th>val loss</th></tr>
<tr><td>0 (base)</td><td>59%</td><td>0%</td><td>100%</td><td>3.22</td></tr>
<tr><td>150</td><td>23%</td><td>0%</td><td>100%</td><td>—</td></tr>
<tr><td>300</td><td>23%</td><td>0%</td><td>100%</td><td>—</td></tr>
<tr><td>450</td><td>23%</td><td>0%</td><td>100%</td><td>—</td></tr>
<tr><td>600</td><td>18%</td><td>0%</td><td>100%</td><td>3.11</td></tr>
</table>
<p class="small">Keyword detector, 22 prompts, 160 tokens, 1 sample each.</p>
<p>Validation loss dropped by 0.1 while the refusal rate fell from 59% to 23%. A refusal is decided by the first few
tokens of the answer, which are diluted inside a loss averaged over hundreds of tokens; and the refusal rate itself is
not differentiable (you have to sample, then classify). So checkpoints were chosen by behaviour: the earliest one where
target refusal has dropped, control is still 0 and harmful prompts are still refused. This matches the observation that
a small amount of fine-tuning is enough to erode alignment <a href="#r8">[8]</a> — which is also a warning.</p>

<h2 id="ablation">6. A category-specific refusal direction (no training)</h2>
<p>Arditi et al. <a href="#r9">[9]</a> showed that refusal is largely mediated by one direction in the residual stream.
Take the residual activation \\(h_\\ell(p)\\) at the last prompt token after layer \\(\\ell\\) and compute difference-of-means
directions over matched prompt sets:</p>
\\[ r^{{\\text{{target}}}}_\\ell = \\overline{{h_\\ell}}(\\text{{target}}) - \\overline{{h_\\ell}}(\\text{{harmless}}),\\qquad
    r^{{\\text{{harm}}}}_\\ell = \\overline{{h_\\ell}}(\\text{{harmful}}) - \\overline{{h_\\ell}}(\\text{{harmless}}) \\]
<p>(30 matched target/harmless pairs, 32 harmful prompts). The plain direction ablates both behaviours together, because
the two directions overlap — increasingly so in later layers:</p>
{fig(F["cos"], "Cosine similarity between the target-category and general-harm directions, per layer. Near zero in the "
     "first third, ~0.3–0.5 in the middle, ~0.68 at the top: the deeper you go, the more 'refuse this category' and "
     "'refuse harmful things' share one representation.", 8)}
<p>So the target direction is orthogonalised against the general-harm direction before use:</p>
\\[ u = \\frac{{r^{{\\text{{harm}}}}_\\ell}}{{\\lVert r^{{\\text{{harm}}}}_\\ell\\rVert}},\\qquad
    r^{{\\perp}}_\\ell = r^{{\\text{{target}}}}_\\ell - \\left(r^{{\\text{{target}}}}_\\ell\\cdot u\\right)u,\\qquad
    d = \\frac{{r^{{\\perp}}_{{\\ell^*}}}}{{\\lVert r^{{\\perp}}_{{\\ell^*}}\\rVert}} \\]
<p>At inference a hook removes \\(d\\) from the input of layer 0 and from the output of every layer:
\\(x \\leftarrow x - (x\\cdot d)\\,d\\). The layer \\(\\ell^*\\) was chosen by a sweep over
\\([L/4, 3L/4]\\) in steps of 4 on 10 held-out target prompts:</p>
{fig(F["sel"], "Layer sweep (keyword detector, 48 tokens, small held-out sets, so coarse). Solid: target-category "
     "refusal (lower is better). Dashed: share of harmful prompts still refused (higher is better). The plain direction "
     "from layers 44–48 also releases harmful requests (dashed orange collapses); the orthogonalised one does not.", 9)}
<table>
<tr><th>method</th><th>target refusal</th><th>control</th><th>harmful (2 + 12)</th><th>training</th></tr>
<tr><td>Base model</td><td>59%</td><td>0%</td><td>all refused</td><td>—</td></tr>
<tr><td>LoRA step 150</td><td>23%</td><td>0%</td><td>all refused</td><td>~15 min</td></tr>
<tr><td>Ablation orth@24</td><td><b>18%</b></td><td>0%</td><td>all refused (judge 12/12)</td><td>none</td></tr>
<tr><td>Ablation plain, late layers</td><td>—</td><td>—</td><td>harmful released too</td><td>none</td></tr>
</table>

<h2 id="deflect">7. The main finding: refusal ≠ deflection</h2>
{fig(F["rdcu"], "Stacking LoRA and ablation nearly eliminates explicit refusals (43 → 1) but converts them into "
     "deflections (0 → 18); the comply rate stays at 56%. DPO is the only intervention that lowers DEFLECT.", 10)}
<p>The direction controls whether the model <em>says no</em>. "Say yes, then quietly write a tamer version" is a
separate behaviour that neither SFT nor ablation removes. Some details:</p>
<ul>
<li>The two single interventions fail on <em>different</em> prompts: LoRA on one style of phrasing, the ablation on
prompts that explicitly ask for intensity. Stacking just turns both sets of failures into deflections.</li>
<li>SFT cannot learn "don't deflect" — its data only contains positives, with no contrast signal. DPO with the model's
own deflections as the rejected side provides exactly that signal (DEFLECT 11–18 → 6–7).</li>
<li>This is consistent with recent work arguing refusal is not a single direction <a href="#r20">[20]</a>; the
contribution here is a concrete case that only shows up with long generations.</li>
</ul>
<p class="small">Each row is 66 generations from the same 22 prompts, so samples are correlated; treat differences of
≤5 percentage points as noise.</p>

<h2 id="dpo">8. DPO on one machine</h2>
<h3>8.1 Objective</h3>
<p>With policy \\(\\pi_\\theta\\), frozen reference \\(\\pi_{{\\text{{ref}}}}\\), and summed response log-probabilities
\\(\\ell_c = \\log\\pi_\\theta(y_c\\mid x)\\), \\(\\ell_r = \\log\\pi_\\theta(y_r\\mid x)\\):</p>
\\[ \\Delta = (\\ell_c - \\ell_c^{{\\text{{ref}}}}) - (\\ell_r - \\ell_r^{{\\text{{ref}}}}),\\qquad
    \\mathcal{{L}} = -\\log\\sigma(\\beta\\Delta) \\; + \\; \\alpha\\,\\frac{{-\\ell_c}}{{|y_c|}} \\]
<p>— DPO <a href="#r14">[14]</a> (IPO <a href="#r15">[15]</a> is also implemented) plus an optional NLL term on the
chosen answer (<code>sft_alpha</code>, as in <a href="#r16">[16]</a>).</p>
<h3>8.2 Peak memory equal to single-sequence SFT</h3>
<p>The gradient of \\(\\mathcal{{L}}\\) touches the two sequences only through two scalar coefficients:</p>
\\[ \\nabla_\\theta\\mathcal{{L}} = g_c\\,\\nabla_\\theta\\ell_c + g_r\\,\\nabla_\\theta\\ell_r,\\qquad
    g_c = -\\beta\\,\\sigma(-\\beta\\Delta) - \\frac{{\\alpha}}{{|y_c|}},\\quad g_r = \\beta\\,\\sigma(-\\beta\\Delta) \\]
{fig(F["dpo"], "Two-pass exact gradient. A no-grad forward gives the four log-probabilities, hence g<sub>c</sub> and "
     "g<sub>r</sub>; then each sequence is back-propagated alone and scaled. The reference log-probs are computed once "
     "before training, so the reference model is never in memory.", 11)}
<p>The result matches direct autodiff (gradient cosine 0.99994). On 27B at sequence 1024 the MLX peak is
34.7–36.5&nbsp;GB, ~31&nbsp;s per pair; with the §3.1 kernel it drops to 19.0&nbsp;GB and 70 → 102 tok/s. Related decoupled-backward ideas: <a href="#r29">[29]</a><a href="#r30">[30]</a>.</p>
<h3>8.3 On-policy pairs</h3>
<p>180 prompts (about 30% scripts / storyboards), each sampled 4 times (900 tokens) from the SFT checkpoint the DPO
run starts from: 720 answers, judged COMPLY 366 / REFUSE 160 / DEFLECT 134 / UNCLEAR 60. For each prompt a COMPLY
answer is chosen and a REFUSE or DEFLECT answer rejected (≤2 pairs per prompt), giving 122 / 9 train/val pairs split by
prompt. Because both sides come from the starting policy, they differ in essentially one thing — whether the model did
it — and not in style or length <a href="#r19">[19]</a><a href="#r22">[22]</a>.</p>
<h3>8.4 Watch r<sub>c</sub>, not accuracy</h3>
<p>On a toy 0.8B run (lr 5e-5, no NLL term) preference accuracy hit 100% while the chosen answer's own implicit reward
\\(r_c = \\beta(\\ell_c - \\ell_c^{{\\text{{ref}}}})\\) fell to −14.5 and the model degenerated into repetition: it was
winning by pushing <em>everything</em> down. Adding the NLL term (α = 1) and lr 1e-5 fixed it, matching the failure mode
described in DPOP <a href="#r23">[23]</a>. The 27B default therefore uses α = 1.</p>
<table>
<tr><th>from SFT step 450, lr 5e-6, 300 steps</th><th>val acc</th><th>val margin (max → final)</th><th>r<sub>c</sub></th><th>comply (long gen)</th><th>refusals</th></tr>
<tr><td>β 0.1, α 1.0</td><td>0.94</td><td>7.3 → 6.1</td><td>+9.0</td><td><b>74%</b></td><td>2/66</td></tr>
<tr><td>β 0.3, α 1.0</td><td>0.89</td><td>24.5 → 19.4</td><td>+26.2</td><td>65%</td><td>9/66</td></tr>
</table>
<p class="small">Validation sets are tiny (9–17 pairs). The β 0.3 run was also checked on control (no false refusals) and
harmful prompts (all refused); the β 0.1 run was not — a gap. Base model comply rate: 35%.</p>

<h2 id="fuse">9. The 4-bit fuse trap</h2>
<p>Symptom: prompts that the step-450 checkpoint handled in the notebook (base + adapter computed separately) were
refused again after fusing and loading the model in LM Studio.</p>
<p>Cause: for a quantised base, <code>mlx_lm fuse</code> de-quantises to bf16, adds \\(\\Delta W = s\\,B^\\top A^\\top\\),
and <em>re-quantises to 4-bit</em>. Affine 4-bit quantisation stores, per group of 64 weights,</p>
\\[ \\hat w = s_g\\,\\operatorname{{round}}\\!\\left(\\frac{{w - b_g}}{{s_g}}\\right) + b_g,\\qquad
    s_g = \\frac{{\\max_g w - \\min_g w}}{{2^{{4}} - 1}}, \\]
<p>so each weight can only move in steps of \\(s_g\\). The LoRA delta is 2–7% of the weight norm — smaller than that
step — and is mostly rounded away:</p>
{fig(F["requant"], "Why re-quantising erases a LoRA. A small delta rounds back to the same 4-bit level; an 8-bit grid "
     "(16× finer) keeps it.", 12)}
<table>
<tr><th>export</th><th>LoRA layer precision</th><th>size</th><th>cos(actual Δ, LoRA Δ) per layer</th><th>rel. error</th><th>target refusal</th></tr>
<tr><td>unfused (base + adapter)</td><td>separate</td><td>16 + 0.2 GB</td><td>1</td><td>0</td><td>23% (5/22)</td></tr>
<tr><td><code>mlx_lm fuse</code></td><td>4-bit</td><td>14 GB</td><td>0.08 – 0.53</td><td>1.25</td><td>41% (9/22)</td></tr>
<tr><td><code>fuse_lora.py</code></td><td>8-bit</td><td>16 GB</td><td>0.93 – 0.98</td><td>0.20</td><td><b>14% (3/22)</b></td></tr>
</table>
<p>A relative error above 1 means the rounding error is larger than the signal. The fix keeps the other layers at
4-bit and stores only the 62 LoRA-carrying linear layers at 8-bit; mlx-lm already supports per-module bit widths in
<code>config.json</code>, so loaders (including LM Studio's MLX engine) need no change. The problem is known in the
PyTorch / bitsandbytes world <a href="#r24">[24]</a><a href="#r25">[25]</a> and attacked from the training side by
QA-LoRA <a href="#r26">[26]</a> and Q-BLoRA <a href="#r27">[27]</a>; what is new here is the per-layer evidence on MLX,
the behavioural consequence, and that <code>mlx_lm fuse</code> does this without a warning.</p>

<h2 id="runtime">10. Runtime details that change behaviour</h2>
<ul>
<li><b>Thinking mode.</b> The stock chat template defaults to thinking at the highest effort. Training and evaluation
used non-thinking mode, so the exported template is flipped to default non-thinking (rendering verified identical to
the evaluation prompts; the original kept as <code>.orig</code>). Default sampling is set to T 0.7 / top-p 0.95 /
top-k 20 to match evaluation.</li>
<li><b>Template caching.</b> LM Studio caches chat templates in its model index keyed on directory mtime; editing files
inside a model directory does not invalidate it. Restart after template edits.</li>
<li><b>Vision.</b> mlx-lm loads only the language model and drops the vision tower and <code>vision_config</code> on
save; LM Studio then treats the export as text-only. The exporter copies the base model's vision tower (333 bf16 tensors,
0.92&nbsp;GB) and config back.</li>
<li><b>Speculative decoding.</b> The vendor build of this model ships DFlash <a href="#r18">[18]</a>, a block-diffusion
draft model (byte-identical to the public draft weights). Measured on the fused model (29.9 tok/s plain decoding):</li>
</ul>
{fig(F["dflash"], "DFlash speed-up relative to plain decoding, non-thinking mode. Chinese fiction accepts only ~1.5 draft "
     "tokens on average, so verification cost cancels the gain; with thinking on (gsm8k) it reaches 2.05×. LoRA and DPO "
     "did not reduce acceptance.", 13)}

<h2 id="bake">11. Baking the direction into the weights — and the gate that said no</h2>
<p>An inference hook only works inside my own Python. To use the ablation in LM Studio or an agent, it has to live in
the weights. Only two kinds of modules ever <em>write</em> to the residual stream: the token embedding and the output
projection of each sub-layer. If none of them can write a \\(d\\)-component, the stream never contains one — which is
exactly what the hook enforces.</p>
{fig(F["writers"], "Residual-stream writers in this architecture: the embedding, plus per layer the attention / "
     "DeltaNet output projection and the MLP down projection. The final norm and lm_head only read.", 14)}
\\[ E \\leftarrow E - (E d)\\,d^\\top \\qquad (E\\in\\mathbb{{R}}^{{V\\times D}}) \\]
\\[ W \\leftarrow W - d\\,(d^\\top W) = (I - d d^\\top)\\,W \\qquad (W\\in\\mathbb{{R}}^{{D\\times d_{{in}}}}) \\]
<p>In this model that is 129 matrices (embedding, 64 down projections, 16 attention <code>o_proj</code>, 48 DeltaNet
<code>out_proj</code>) — about 9.0B parameters, a third of the model. Same lesson as §9: the change is a small component
of each matrix, so the modified matrices are stored at 8-bit and the rest stays 4-bit (20.1&nbsp;GB of weights instead
of 16). Each matrix is de-quantised, projected, re-quantised and released in turn, so the export never holds the
model in bf16.</p>
<table>
<tr><th>check</th><th>result</th></tr>
<tr><td>d-component left after 8-bit re-quantisation, \\(\\lVert d^\\top \\hat W\\rVert / \\lVert d^\\top W\\rVert\\)</td><td>mean 0.036, max 0.048</td></tr>
<tr><td>KL(hook ‖ baked), last-token distribution, 8 ordinary prompts</td><td><b>0.0004</b> (vs 0.021 for base ‖ hook)</td></tr>
<tr><td>top-1 agreement hook vs baked</td><td>100%</td></tr>
<tr><td>12 harmful prompts refused (greedy, 160 tokens, keyword detector)</td><td>11 / 12 &dagger;</td></tr>
<tr><td><b>protected-category probe</b> (prompts involving minors; greedy + 2 samples)</td><td class="bad"><b>4 / 24 not refused</b> — base model: 0 / 8</td></tr>
</table>
<p class="small">&dagger; The hook version scored 12/12 in §6 with sampled answers and the model judge; this
check is greedy with the stricter keyword detector. One prompt (a poison-extraction question) was not refused here.</p>
<p>The bake is faithful: the exported model behaves like the hook version. But the gate I had added before the
export — <em>every request involving minors must be refused, otherwise no link to LM Studio is created</em> — failed.
The failures concentrated on two prompts, one framed as a short-drama script. Since the baked model matches the hook,
this is a property of the direction itself, not of the export. (The detector is keyword-based and counts deflections as
"not refused"; the answers in this probe are never written to disk or displayed, so I cannot split the four further.
The gate is deliberately strict.)</p>
<p>The lesson is the most important one in this post: <b>orthogonalising against an averaged "general harm" direction
does not protect every refusal you care about.</b> The harmful set that defined \\(r^{{\\text{{harm}}}}\\) did not contain
this category, and in late layers the target direction overlaps with it. The next step is to add an explicit
constraint — extract a direction from a held-out set of protected-category prompts and project it out of \\(d\\) as
well — and re-run the same gate on a disjoint probe set. The model is not shipped until it passes.</p>

<h2 id="agent">12. Inside an agent: format and length shift</h2>
<p>The fused DPO model behaved in LM Studio exactly as in the notebook. Inside a local agent, it still refused or broke.
I have not fully resolved this, but three shifts away from the training distribution are clear (hypotheses, not
verified causes):</p>
<ol>
<li><b>Prompt format.</b> Training used no system prompt. The agent sends a long system prompt (rules, a tool catalogue,
project memory) plus tool definitions that the chat template renders into the system turn.</li>
<li><b>Constrained decoding.</b> Every agent stage requests <code>response_format: json_schema</code> (strict), so the
runtime forces schema-valid JSON — a refusal can only appear inside a JSON field.</li>
<li><b>Length.</b> SFT and DPO saw ≤1024-token sequences. One 16.5k-token input made the DPO model collapse into a
repetition loop of a hallucinated status line (not present in the input, the data or the agent code), with top-p and
min-p disabled. The base model keeps its long-context ability; a short-sequence LoRA apparently does not.</li>
</ol>
<p>An ablation that lives in the weights is attractive here precisely because it does not depend on the prompt format
and does not touch long-context behaviour — once it passes the gate.</p>

<h2 id="lessons">13. Lessons</h2>
<ol>
<li>Select checkpoints by behaviour, not loss — and measure behaviour on long generations.</li>
<li>A keyword regex is a lower bound. Every time a conclusion changes, go back to the raw outputs.</li>
<li>Separate refusal from deflection. Methods that only remove refusal (direction ablation, light SFT) hide the problem
in "tamer versions"; it takes a contrast signal (DPO on your own deflections) to move it.</li>
<li>Never fuse a LoRA back into a quantised base at the same bit width. Re-test behaviour on the exported files, not on
the adapter in the notebook.</li>
<li>Measure memory with the framework's real peak; process footprint includes caches and will flatter your
optimisation.</li>
<li>Benchmark speed on real, variable-length data. Fixed-length benchmarks hide compile retracing: the same ops path
ran at 105 tok/s on fixed lengths and 80 on the real data.</li>
<li>In DPO, watch r<sub>c</sub>. When the chosen answer's implicit reward goes negative, the model is winning by
pushing everything down.</li>
<li>Deployment defaults silently change behaviour: thinking mode, temperature, system prompts, cached templates,
structured-output constraints, context length.</li>
<li>Put hard safety gates <em>in the export path</em>, not in a checklist. Mine caught a regression I would otherwise
have shipped.</li>
</ol>

<h2 id="limits">14. Limitations and ethics</h2>
<ul>
<li><b>Small samples.</b> 22 target prompts, 3 long generations each; differences within 5 points are noise
(single-proportion standard error ≈ 9 points at n = 22, and the 3 seeds per prompt are highly correlated).</li>
<li><b>One judge.</b> The judge is the base model itself; no human agreement study.</li>
<li><b>No capability benchmarks.</b> Only 8 control prompts check that nothing broke; no MMLU / C-Eval / writing-quality
evaluation. This is the largest gap.</li>
<li><b>Keyword safety filter.</b> It cannot catch non-consent that is only implied by the narrative. Any release of data
or weights would need a classifier pass. I release neither.</li>
<li><b>One model, one quantisation</b> (affine 4-bit, group 64).</li>
<li><b>Dual use.</b> Refusal-direction ablation can equally be used to release harmful categories. The orthogonalisation,
the always-evaluated harmful set and the export gate exist to keep changes inside the target category — and §11 shows
that even this is not sufficient by default. No direction, adapter or modified weights are published.</li>
</ul>
"""


REFS = [
    ("Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. 2021.", "https://arxiv.org/abs/2106.09685"),
    ("Dettmers et al. QLoRA: Efficient Finetuning of Quantized LLMs. 2023.", "https://arxiv.org/abs/2305.14314"),
    ("Biderman et al. LoRA Learns Less and Forgets Less. 2024.", "https://arxiv.org/abs/2405.09673"),
    ("Loshchilov & Hutter. Decoupled Weight Decay Regularization. 2017.", "https://arxiv.org/abs/1711.05101"),
    ("Loshchilov & Hutter. SGDR: Stochastic Gradient Descent with Warm Restarts. 2016.", "https://arxiv.org/abs/1608.03983"),
    ("Chen et al. Training Deep Nets with Sublinear Memory Cost. 2016.", "https://arxiv.org/abs/1604.06174"),
    ("Zhou et al. LIMA: Less Is More for Alignment. 2023.", "https://arxiv.org/abs/2305.11206"),
    ("Qi et al. Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To! 2023.", "https://arxiv.org/abs/2310.03693"),
    ("Arditi et al. Refusal in Language Models Is Mediated by a Single Direction. 2024.", "https://arxiv.org/abs/2406.11717"),
    ("Yang, Kautz & Hatamizadeh. Gated Delta Networks: Improving Mamba2 with Delta Rule. 2024.", "https://arxiv.org/abs/2412.06464"),
    ("Yang et al. Parallelizing Linear Transformers with the Delta Rule over Sequence Length. 2024.", "https://arxiv.org/abs/2406.06484"),
    ("Qiu et al. Gated Attention for Large Language Models. 2025.", "https://arxiv.org/abs/2505.06708"),
    ("Qwen Team. Qwen3 Technical Report. 2025.", "https://arxiv.org/abs/2505.09388"),
    ("Rafailov et al. Direct Preference Optimization: Your Language Model is Secretly a Reward Model. 2023.", "https://arxiv.org/abs/2305.18290"),
    ("Azar et al. A General Theoretical Paradigm to Understand Learning from Human Preferences. 2023.", "https://arxiv.org/abs/2310.12036"),
    ("Pang et al. Iterative Reasoning Preference Optimization. 2024.", "https://arxiv.org/abs/2404.19733"),
    ("Hong et al. ORPO: Monolithic Preference Optimization without Reference Model. 2024.", "https://arxiv.org/abs/2403.07691"),
    ("Chen, Liang, Liu et al. DFlash: Block Diffusion for Flash Speculative Decoding. 2026.", "https://arxiv.org/abs/2602.06036"),
    ("Tajwar et al. Preference Fine-Tuning of LLMs Should Leverage Suboptimal, On-Policy Data. 2024.", "https://arxiv.org/abs/2404.14367"),
    ("There Is More to Refusal in Large Language Models than a Single Direction. 2026.", "https://arxiv.org/abs/2602.02132"),
    ("unsloth-zoo PR #754. MLX: fix gated-delta (Qwen3.5) training and add fused backward kernels. 2026.", "https://github.com/unslothai/unsloth-zoo/pull/754"),
    ("Park et al. Disentangling Length from Quality in Direct Preference Optimization. 2024.", "https://arxiv.org/abs/2403.19159"),
    ("Pal et al. Smaug: Fixing Failure Modes of Preference Optimisation with DPO-Positive. 2024.", "https://arxiv.org/abs/2402.13228"),
    ("The Kaitchup. Don't Merge Your LoRA Adapter Into a 4-bit LLM.", "https://kaitchup.substack.com/p/dont-merge-your-lora-adapter-into"),
    ("The Kaitchup. Training, Loading, and Merging QDoRA, QLoRA, and LoftQ Adapters.", "https://kaitchup.substack.com/p/training-loading-and-merging-qdora"),
    ("Xu et al. QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models. 2023.", "https://arxiv.org/abs/2309.14717"),
    ("Accurate and Efficient Fine-Tuning of Quantized Large Language Models Through Optimal Balance (Q-BLoRA). 2024.", "https://arxiv.org/abs/2407.17029"),
    ("From Refusal Tokens to Refusal Control: Discovering and Steering Category-Specific Refusal Directions. 2026.", "https://arxiv.org/abs/2603.13359"),
    ("TeleBoost: A Systematic Alignment Framework for High-Fidelity, Controllable, and Robust Video Generation. 2026.", "https://arxiv.org/abs/2602.07595"),
    ("GroupDPO: Memory-Efficient Group-Wise Direct Preference Optimization. 2026.", "https://arxiv.org/abs/2604.15602"),
]

CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1f;--muted:#5f6368;--line:#e3e0d8;--card:#ffffff;--accent:#9a4a00;--link:#0b57d0;
--c1:#2f6db5;--c2:#2e8b57;--c3:#d2691e;--c4:#b8860b;--c5:#9aa0a6;--bad:#c0392b;--good:#2e8b57;
--hl:#eef4fb;--warn:#fdf0e6;--code:#f2efe8;--sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Helvetica,Arial,sans-serif;
--serif:"Source Serif 4",Georgia,"Times New Roman",serif;--mono:ui-monospace,SFMono-Regular,Menlo,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#16171a;--fg:#e8e6e3;--muted:#a0a4aa;--line:#2e3036;
--card:#1e2024;--accent:#f0a35e;--link:#8ab4f8;--c1:#6ea8ff;--c2:#5cc08a;--c3:#f08c4a;--c4:#e0b84a;--c5:#7d838b;
--bad:#ff7b6b;--good:#5cc08a;--hl:#1d2a3a;--warn:#3a2a1d;--code:#24262b}}
:root[data-theme="dark"]{--bg:#16171a;--fg:#e8e6e3;--muted:#a0a4aa;--line:#2e3036;--card:#1e2024;--accent:#f0a35e;
--link:#8ab4f8;--c1:#6ea8ff;--c2:#5cc08a;--c3:#f08c4a;--c4:#e0b84a;--c5:#7d838b;--bad:#ff7b6b;--good:#5cc08a;
--hl:#1d2a3a;--warn:#3a2a1d;--code:#24262b}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:18px/1.65 var(--serif);-webkit-text-size-adjust:100%}
main{max-width:760px;margin:0 auto;padding:40px 16px 80px}
header.site{max-width:760px;margin:0 auto;padding:18px 16px 0;font:14px var(--sans)}
header.site a{color:var(--muted);text-decoration:none}
h1{font:700 2.1rem/1.2 var(--sans);margin:.2em 0 .3em;letter-spacing:-.01em}
.sub{font:400 1.15rem/1.45 var(--sans);color:var(--muted);margin:0 0 .8em}
.meta{font:14px var(--sans);color:var(--muted);margin-bottom:2em}
h2{font:700 1.45rem/1.3 var(--sans);margin:2.4em 0 .6em;padding-top:.6em;border-top:1px solid var(--line)}
h3{font:600 1.1rem/1.35 var(--sans);margin:1.6em 0 .4em}
a{color:var(--link)} p,li{hyphens:auto}
.lede{font-size:1.12rem}
code{font:.86em var(--mono);background:var(--code);padding:.1em .3em;border-radius:4px}
pre{background:var(--code);padding:12px 14px;border-radius:8px;overflow-x:auto} pre code{background:none;padding:0}
.tldr{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:8px;padding:6px 20px 8px;margin:1.6em 0}
.tldr h3{margin-top:.8em}
.note{font:15px/1.6 var(--sans);color:var(--muted);background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 16px}
.toc{font:15px/1.7 var(--sans);background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 18px;margin:1.6em 0}
.toc ol{margin:.3em 0 0;padding-left:1.4em} .toc a{text-decoration:none}
figure{margin:1.8em 0;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
figcaption{font:14px/1.5 var(--sans);color:var(--muted);margin-top:8px}
svg.chart{width:100%;height:auto;display:block}
svg .ink{fill:var(--fg)} svg .muted{fill:var(--muted)} svg .accent{fill:var(--accent)} svg .bad{fill:var(--bad)}
svg .on-bar{fill:#fff} svg .grid{stroke:var(--line);stroke-width:1} svg .axis{stroke:var(--muted);stroke-width:1.2}
svg .axis.thin{stroke-width:.8} svg .mark{stroke:var(--accent);stroke-width:1.4;stroke-dasharray:4 3}
svg .box{fill:var(--card);stroke:var(--muted);stroke-width:1.2} svg .box.hl{fill:var(--hl);stroke:var(--c1)}
svg .box.warn{fill:var(--warn);stroke:var(--bad)} svg .bad-fill{fill:var(--warn);stroke:var(--bad)}
svg .good-fill{fill:var(--hl);stroke:var(--good)} svg .band{fill:var(--hl);stroke:var(--c1);stroke-dasharray:4 3}
svg .arrow{stroke:var(--muted);stroke-width:1.6} svg .arrow.thin{stroke-width:1.1} svg .arrowhead{fill:var(--muted)}
svg .stream{stroke:var(--c1);stroke-width:5;opacity:.45} svg .plus{fill:var(--card);stroke:var(--c1);stroke-width:1.4}
svg .dot-base{fill:var(--c1)} svg .dot-new{fill:var(--c3)}
table{border-collapse:collapse;width:100%;font:14.5px/1.45 var(--sans);margin:1.2em 0;display:block;overflow-x:auto}
th,td{border-bottom:1px solid var(--line);padding:7px 10px;text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600;border-bottom:2px solid var(--line)}
td.bad,.bad{color:var(--bad)}
.small{font:14px/1.55 var(--sans);color:var(--muted)}
ul.labels{list-style:none;padding-left:0} .tag{display:inline-block;font:600 12px var(--mono);padding:1px 7px;border-radius:4px;color:#fff;margin-right:6px}
.tag.r{background:var(--c3)} .tag.d{background:var(--c4)} .tag.c{background:var(--c2)} .tag.u{background:var(--c5)}
.refs{font:14px/1.55 var(--sans)} .refs li{margin:.25em 0}
.katex-display{overflow-x:auto;overflow-y:hidden;padding:4px 0}
footer{max-width:760px;margin:0 auto;padding:0 16px 40px;font:13px var(--sans);color:var(--muted)}
.post-list{list-style:none;padding:0} .post-list li{padding:16px 0;border-bottom:1px solid var(--line)}
.post-list a{font:600 1.2rem var(--sans);text-decoration:none} .post-list p{margin:.3em 0;font:15px/1.5 var(--sans);color:var(--muted)}
@media (max-width:560px){body{font-size:17px}h1{font-size:1.65rem}main{padding-top:24px}}
"""

KATEX = """<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css">
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js"></script>
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/contrib/auto-render.min.js"
 onload="renderMathInElement(document.body,{delimiters:[{left:'\\\\[',right:'\\\\]',display:true},{left:'\\\\(',right:'\\\\)',display:false}]})"></script>"""


def page(title, desc, body, base, math=False):
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><meta name="description" content="{desc}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">
{KATEX if math else ""}
<style>{CSS}</style></head>
<body><header class="site"><a href="{base}">← {AUTHOR} / blog</a></header>
<main>{body}</main>
<footer>© 2026 {AUTHOR}. Text CC BY 4.0. No model weights, adapters, directions or datasets are distributed here.</footer>
</body></html>
"""


def main():
    F = figs()
    refs = "".join(f'<li id="r{i}"><a href="{u}">{t}</a></li>' for i, (t, u) in enumerate(REFS, 1))
    body = (f'<h1>{TITLE}</h1><p class="sub">{SUBTITLE}</p>'
            f'<div class="meta">{AUTHOR} · {DATE} · ~25 min read</div>'
            + post_body(F) + f'<h2 id="refs">References</h2><ol class="refs">{refs}</ol>')
    out = ROOT / "posts" / SLUG / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page(f"{TITLE} — {AUTHOR}", SUBTITLE, body, "../../", math=True))
    idx = (f'<h1>{AUTHOR} / blog</h1><p class="sub">Notes on training and running models locally.</p>'
           f'<ul class="post-list"><li><a href="posts/{SLUG}/">{TITLE}</a><p>{SUBTITLE}</p>'
           f'<p>{DATE}</p></li></ul>')
    (ROOT / "index.html").write_text(page(f"{AUTHOR} / blog", "Notes on training and running models locally.", idx, "./"))
    print("wrote", out.relative_to(ROOT), "and index.html")


if __name__ == "__main__":
    main()
