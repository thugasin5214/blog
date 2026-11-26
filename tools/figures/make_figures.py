"""Regenerate the post's figures as standalone, theme-aware SVG files.

    .venv/bin/python tools/figures/make_figures.py

The data below is the measured data behind each chart; edit it here, not in the SVGs. Each SVG carries a
<style data-standalone="1"> block so it also renders (light and dark) in Markdown editors; the site build inlines the SVG
and drops that block so the page's own CSS variables apply.
"""
from pathlib import Path

import charts as C
import diagrams as D

OUT = Path(__file__).resolve().parents[2] / "content" / "posts" / "27b-post-training-on-a-mac" / "figures"

STANDALONE = '<style data-standalone="1">svg{--bg:#fbfaf7;--fg:#1d1d1f;--muted:#5f6368;--line:#e3e0d8;--card:#ffffff;--accent:#9a4a00;--link:#0b57d0; --c1:#2f6db5;--c2:#2e8b57;--c3:#d2691e;--c4:#b8860b;--c5:#9aa0a6;--bad:#c0392b;--good:#2e8b57; --hl:#eef4fb;--warn:#fdf0e6;--code:#f2efe8;--sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Helvetica,Arial,sans-serif; --serif:"Source Serif 4",Georgia,"Times New Roman",serif;--mono:ui-monospace,SFMono-Regular,Menlo,monospace}@media (prefers-color-scheme:dark){svg{--bg:#16171a;--fg:#e8e6e3;--muted:#a0a4aa;--line:#2e3036;--card:#1e2024;--accent:#f0a35e; --link:#8ab4f8;--c1:#6ea8ff;--c2:#5cc08a;--c3:#f08c4a;--c4:#e0b84a;--c5:#7d838b;--bad:#ff7b6b;--good:#5cc08a; --hl:#1d2a3a;--warn:#3a2a1d;--code:#24262b}}svg .ink{fill:var(--fg)} svg .muted{fill:var(--muted)} svg .accent{fill:var(--accent)} svg .bad{fill:var(--bad)} svg .on-bar{fill:#fff} svg .grid{stroke:var(--line);stroke-width:1} svg .axis{stroke:var(--muted);stroke-width:1.2} svg .axis.thin{stroke-width:.8} svg .mark{stroke:var(--accent);stroke-width:1.4;stroke-dasharray:4 3} svg .box{fill:var(--card);stroke:var(--muted);stroke-width:1.2} svg .box.hl{fill:var(--hl);stroke:var(--c1)} svg .box.warn{fill:var(--warn);stroke:var(--bad)} svg .bad-fill{fill:var(--warn);stroke:var(--bad)} svg .good-fill{fill:var(--hl);stroke:var(--good)} svg .band{fill:var(--hl);stroke:var(--c1);stroke-dasharray:4 3} svg .arrow{stroke:var(--muted);stroke-width:1.6} svg .arrow.thin{stroke-width:1.1} svg .arrowhead{fill:var(--muted)} svg .stream{stroke:var(--c1);stroke-width:5;opacity:.45} svg .plus{fill:var(--card);stroke:var(--c1);stroke-width:1.4} svg .dot-base{fill:var(--c1)} svg .dot-new{fill:var(--c3)}</style>'

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




def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, svg in figs().items():
        svg = svg.replace('class="chart">', 'class="chart">' + STANDALONE, 1)
        (OUT / f"{name.replace('_', '-')}.svg").write_text(svg + "\n")
    print("wrote", len(figs()), "figures to", OUT)


if __name__ == "__main__":
    main()
