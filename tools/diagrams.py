"""Hand-drawn SVG diagrams for the post (flow charts and mechanism sketches)."""
from charts import _svg, _t


def box(x, y, w, h, title, sub="", cls="box", tcls="ink"):
    s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" class="{cls}"/>'
    if sub:
        s += _t(x + w / 2, y + h / 2 - 3, title, "middle", 12.5, tcls, "600")
        s += _t(x + w / 2, y + h / 2 + 13, sub, "middle", 10.5, "muted")
    else:
        s += _t(x + w / 2, y + h / 2 + 4, title, "middle", 12.5, tcls, "600")
    return s


def arrow(x1, y1, x2, y2, cls="arrow", dashed=False):
    d = ' stroke-dasharray="5 4"' if dashed else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="{cls}" marker-end="url(#ah)"{d}/>'


DEFS = ('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="arrowhead"/></marker></defs>')


def pipeline():
    b = [DEFS]
    W = 860
    # main (training) path
    y = 40
    b.append(_t(12, 24, "Training path", size=12, cls="accent", weight="700"))
    xs = [12, 172, 342, 512, 682]
    items = [("Base model", "Qwen3.8-27B, 4-bit"), ("SFT (QLoRA)", "top-8 layers, seq 1024"),
             ("Pick checkpoint", "by behaviour, not loss"), ("DPO", "on-policy pairs"),
             ("Mixed-precision fuse", "LoRA layers 8-bit")]
    for i, (x, (t, s)) in enumerate(zip(xs, items)):
        b.append(box(x, y, 150 if i < 4 else 166, 52, t, s, "box hl" if i == 4 else "box"))
        if i:
            b.append(arrow(xs[i - 1] + 150, y + 26, x - 4, y + 26))
    # eval harness band
    b.append('<rect x="172" y="112" width="490" height="34" rx="8" class="band"/>')
    b.append(_t(417, 134, "Evaluation harness: 600-token generations × 3 seeds → 4-label judge", "middle", 12, "ink", "600"))
    for x in (247, 417, 587):
        b.append(arrow(x, 108, x, 96, "arrow thin"))
    # deploy
    b.append(box(682, 112, 166, 44, "LM Studio / agent", "MLX engine"))
    b.append(arrow(765, 96, 765, 108))
    # no-training path
    y2 = 200
    b.append(_t(12, y2 - 14, "No-training path", size=12, cls="accent", weight="700"))
    xs2 = [12, 172, 342, 512, 682]
    items2 = [("Base model", ""), ("Refusal direction", "category ⟂ general harm"),
              ("Inference hook", "x ← x − (x·d)d"), ("Bake into weights", "129 matrices, 8-bit"),
              ("Safety gate", "protected refusals kept?")]
    for i, (x, (t, s)) in enumerate(zip(xs2, items2)):
        b.append(box(x, y2, 150 if i < 4 else 166, 52, t, s, "box warn" if i == 4 else "box"))
        if i:
            b.append(arrow(xs2[i - 1] + 150, y2 + 26, x - 4, y2 + 26))
    b.append(_t(765, y2 + 74, "failed → not shipped (§11)", "middle", 11.5, "bad", "600"))
    return _svg(W, 290, "".join(b), "Pipeline overview: training path and no-training path")


def layer_patch():
    b = [DEFS]
    W, H = 760, 300

    def stack(x0, title, lora_mode, frozen_mode, frozen_cls):
        s = _t(x0 + 140, 20, title, "middle", 13, "ink", "700")
        y = 34
        s += f'<rect x="{x0}" y="{y}" width="280" height="64" rx="8" class="box hl"/>'
        s += _t(x0 + 140, y + 26, "top 8 layers: LoRA attached", "middle", 12.5, "ink", "600")
        s += _t(x0 + 140, y + 46, lora_mode, "middle", 11.5, "muted")
        y = 108
        s += f'<rect x="{x0}" y="{y}" width="280" height="140" rx="8" class="box {frozen_cls}"/>'
        s += _t(x0 + 140, y + 60, "56 frozen layers", "middle", 12.5, "ink", "600")
        s += _t(x0 + 140, y + 80, frozen_mode, "middle", 11.5, "muted")
        return s

    b.append(stack(20, "stock mlx-lm (model.train())", "training=True → Python loop over T",
                   "training=True → Python loop over T", "bad-fill"))
    b.append(stack(440, "with the patch", "training=True → Python loop (needs backward)",
                   "pinned training=False → Metal kernel", "good-fill"))
    b.append(arrow(312, 150, 432, 150))
    b.append(_t(372, 140, "_frozen_fast", "middle", 11, "accent", "600"))
    b.append(_t(380, 280, "Gradients only flow through the LoRA layers and above; frozen layers below only run forward.",
                "middle", 11.5, "muted"))
    return _svg(W, H, "".join(b), "Layer stack before and after the frozen-layer kernel patch")


def dpo_two_pass():
    b = [DEFS]
    b.append(box(10, 20, 200, 60, "Pass 0: forward, no grad", "log p(y_c), log p(y_r)"))
    b.append(box(270, 20, 210, 60, "4 scalars → coefficients", "g_c = ∂L/∂ℓ_c,  g_r = ∂L/∂ℓ_r"))
    b.append(box(540, 4, 210, 40, "Pass 1: backward g_c · ∇ℓ_c", cls="box hl"))
    b.append(box(540, 56, 210, 40, "Pass 2: backward g_r · ∇ℓ_r", cls="box hl"))
    b.append(arrow(210, 50, 266, 50))
    b.append(arrow(480, 44, 536, 26))
    b.append(arrow(480, 56, 536, 76))
    b.append(box(10, 112, 200, 44, "Reference log p", "pre-computed once, cached"))
    b.append(arrow(110, 112, 300, 82, dashed=True))
    b.append(_t(240, 132, "Only one sequence is ever on the autograd tape,", "start", 11.5, "muted"))
    b.append(_t(240, 150, "so peak memory = single-sequence SFT.", "start", 11.5, "muted"))
    return _svg(760, 170, "".join(b), "DPO two-pass exact gradient")


def requant():
    """Why re-quantising to 4-bit erases a LoRA delta."""
    b = [DEFS]
    left, w = 40, 680
    y = 70
    b.append(f'<line x1="{left}" x2="{left + w}" y1="{y}" y2="{y}" class="axis"/>')
    step = w / 8
    for i in range(9):
        x = left + i * step
        b.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y - 8}" y2="{y + 8}" class="axis"/>')
    b.append(_t(left, 18, "4-bit grid inside one group of 64 weights (16 levels; 8 shown)", size=11.5, cls="muted"))
    xw = left + 3 * step
    b.append(f'<circle cx="{xw:.1f}" cy="{y}" r="6" class="dot-base"/>')
    b.append(_t(xw - 10, y - 16, "W (on grid)", "end", 11.5, "ink", "600"))
    xd = xw + step * 0.22
    b.append(f'<circle cx="{xd:.1f}" cy="{y}" r="6" class="dot-new"/>')
    b.append(_t(xd + 10, y - 16, "W + ΔW  (|ΔW| ≈ 2–7% of |W|)", "start", 11.5, "accent", "600"))
    b.append(f'<path d="M{xd:.1f},{y + 10} C{xd:.1f},{y + 40} {xw:.1f},{y + 40} {xw:.1f},{y + 12}" '
             f'class="arrow" fill="none" marker-end="url(#ah)"/>')
    b.append(_t(xw + 70, y + 50, "round to nearest → back to W: the delta is gone", "start", 11.5, "bad", "600"))
    y2 = 150
    b.append(f'<line x1="{left}" x2="{left + w}" y1="{y2}" y2="{y2}" class="axis"/>')
    for i in range(8 * 16 + 1):
        x = left + i * step / 16
        hh = 6 if i % 16 == 0 else 3
        b.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y2 - hh}" y2="{y2 + hh}" class="axis thin"/>')
    b.append(f'<circle cx="{xd:.1f}" cy="{y2}" r="6" class="dot-new"/>')
    b.append(_t(left, y2 + 28, "8-bit grid: steps are 16× finer, so W + ΔW survives (rel. error 1.25 → 0.20)",
                size=11.5, cls="muted"))
    return _svg(760, 190, "".join(b), "Why re-quantising to 4-bit erases a LoRA delta")


def residual_writers():
    b = [DEFS]
    y = 110
    b.append(f'<line x1="60" x2="720" y1="{y}" y2="{y}" class="stream"/>')
    b.append(_t(390, y + 30, "residual stream  h  (5120-d)", "middle", 12, "muted"))
    b.append(box(10, y - 22, 90, 44, "embedding", "E ← E − (Ed)dᵀ", "box hl"))
    xs = [170, 300, 470, 600]
    labels = [("attn / DeltaNet", "out_proj"), ("MLP", "down_proj"), ("attn / DeltaNet", "out_proj"), ("MLP", "down_proj")]
    for i, (x, (t, s)) in enumerate(zip(xs, labels)):
        b.append(box(x - 55, 14, 110, 44, t, s, "box hl"))
        b.append(arrow(x, 58, x, y - 6))
        b.append(f'<circle cx="{x}" cy="{y}" r="7" class="plus"/>')
        b.append(_t(x, y + 4, "+", "middle", 12, "ink", "700"))
    b.append(_t(385, 36, "· · ·", "middle", 16, "muted"))
    b.append(_t(235, 84, "layer 0", "middle", 11, "muted"))
    b.append(_t(535, 84, "layer 63", "middle", 11, "muted"))
    b.append(_t(742, y + 40, "norm → lm_head (reads only)", "end", 11, "muted"))
    b.append(_t(380, 180, "Every writer gets W ← W − d(dᵀW): no write can add a d-component, so h never contains d.",
                "middle", 11.5, "ink", "600"))
    b.append(_t(380, 198, "That is exactly what the inference hook x ← x − (x·d)d enforces after every layer.",
                "middle", 11.5, "muted"))
    return _svg(770, 210, "".join(b), "Residual stream writers that get orthogonalised")
