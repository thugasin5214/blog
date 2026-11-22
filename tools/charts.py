"""Tiny inline-SVG chart and diagram helpers. Colors come from CSS variables so charts follow light / dark mode."""
from html import escape

FONT = 'font-family="var(--sans)"'


def _svg(w, h, body, label):
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{escape(label)}" '
            f'xmlns="http://www.w3.org/2000/svg" class="chart">{body}</svg>')


def _t(x, y, s, anchor="start", size=12, cls="ink", weight="400", extra=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}" '
            f'font-weight="{weight}" class="{cls}" {FONT} {extra}>{escape(str(s))}</text>')


def stacked_bars(rows, cats, colors, label, total=66, w=720, note=""):
    """rows: [(name, [counts...])]; horizontal 100% stacked bars."""
    left, right, top, bh, gap = 150, 70, 36, 26, 12
    h = top + len(rows) * (bh + gap) + 30
    pw = w - left - right
    b = []
    # legend
    x = left
    for c, col in zip(cats, colors):
        b.append(f'<rect x="{x}" y="8" width="12" height="12" rx="2" fill="var({col})"/>')
        b.append(_t(x + 17, 18, c, size=12, cls="muted"))
        x += 30 + 7.2 * len(c) + 12
    for i, (name, vals) in enumerate(rows):
        y = top + i * (bh + gap)
        b.append(_t(left - 10, y + bh / 2 + 4, name, "end", 12.5))
        x = left
        for v, col in zip(vals, colors):
            ww = pw * v / total
            if ww > 0:
                b.append(f'<rect x="{x:.1f}" y="{y}" width="{ww:.1f}" height="{bh}" fill="var({col})"/>')
                if ww > 20:
                    b.append(_t(x + ww / 2, y + bh / 2 + 4, v, "middle", 11.5, "on-bar", "600"))
            x += ww
        comply = vals[2] / total
        b.append(_t(left + pw + 8, y + bh / 2 + 4, f"{comply:.0%}", size=12.5, weight="600"))
    b.append(_t(left + pw + 8, top - 8, "comply", size=11, cls="muted"))
    if note:
        b.append(_t(left, h - 6, note, size=11, cls="muted"))
    return _svg(w, h, "".join(b), label)


def line_chart(series, label, xlab, ylab, xr, yr, w=720, h=300, xticks=None, yticks=None,
               marks=(), logx=False, logy=False, fmt_y=lambda v: f"{v:g}", fmt_x=lambda v: f"{v:g}",
               legend=True, points=False):
    """series: [(name, color_var, [(x, y)...], dashed)]"""
    import math
    left, right, top, bottom = 56, 20, 30 if legend else 14, 44
    pw, ph = w - left - right, h - top - bottom
    fx = (lambda v: math.log10(v)) if logx else (lambda v: v)
    fy = (lambda v: math.log10(v)) if logy else (lambda v: v)
    X = lambda v: left + pw * (fx(v) - fx(xr[0])) / (fx(xr[1]) - fx(xr[0]))
    Y = lambda v: top + ph - ph * (fy(v) - fy(yr[0])) / (fy(yr[1]) - fy(yr[0]))
    b = []
    for v in yticks or []:
        b.append(f'<line x1="{left}" x2="{left + pw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>')
        b.append(_t(left - 8, Y(v) + 4, fmt_y(v), "end", 11, "muted"))
    for v in xticks or []:
        b.append(_t(X(v), top + ph + 18, fmt_x(v), "middle", 11, "muted"))
    b.append(f'<line x1="{left}" x2="{left + pw}" y1="{top + ph}" y2="{top + ph}" class="axis"/>')
    b.append(_t(left + pw / 2, h - 6, xlab, "middle", 11.5, "muted"))
    b.append(_t(14, top + ph / 2, ylab, "middle", 11.5, "muted", extra=f'transform="rotate(-90 14 {top + ph / 2})"'))
    for (mx_, txt) in marks:
        b.append(f'<line x1="{X(mx_):.1f}" x2="{X(mx_):.1f}" y1="{top}" y2="{top + ph}" class="mark"/>')
        b.append(_t(X(mx_) + 5, top + 12, txt, size=11, cls="accent", weight="600"))
    lx = left
    for name, col, pts, dashed in series:
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(x):.1f},{Y(y):.1f}" for i, (x, y) in enumerate(pts))
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        b.append(f'<path d="{d}" fill="none" stroke="var({col})" stroke-width="2.2"{dash}/>')
        if points:
            for x, y in pts:
                b.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3.2" fill="var({col})"/>')
        if legend:
            b.append(f'<line x1="{lx}" x2="{lx + 18}" y1="14" y2="14" stroke="var({col})" stroke-width="2.4"{dash}/>')
            b.append(_t(lx + 23, 18, name, size=12, cls="muted"))
            lx += 40 + 6.6 * len(name)
    return _svg(w, h, "".join(b), label)


def hbars(rows, label, w=720, unit="×", ref=None, maxv=None, color="--c1"):
    """rows: [(group, [(name, value)])] grouped horizontal bars."""
    left, right, top, bh, gap, ggap = 290, 60, 14, 16, 4, 16
    n = sum(len(r[1]) for r in rows)
    h = top + n * (bh + gap) + len(rows) * ggap + 26
    maxv = maxv or max(v for _, rs in rows for _, v in rs) * 1.1
    pw = w - left - right
    X = lambda v: left + pw * v / maxv
    b, y = [], top
    if ref is not None:
        b.append(f'<line x1="{X(ref):.1f}" x2="{X(ref):.1f}" y1="{top - 6}" y2="{h - 22}" class="mark"/>')
        b.append(_t(X(ref), h - 8, f"{ref:g}{unit} = no speed-up", "middle", 11, "accent"))
    for g, rs in rows:
        b.append(_t(8, y + bh - 3, g, size=12.5, weight="600"))
        for name, v in rs:
            b.append(_t(left - 8, y + bh - 4, name, "end", 11.5, "muted"))
            b.append(f'<rect x="{left}" y="{y}" width="{X(v) - left:.1f}" height="{bh}" rx="2" '
                     f'fill="var({"--c3" if ref is not None and v < ref else color})"/>')
            b.append(_t(X(v) + 6, y + bh - 4, f"{v:g}{unit}", size=11.5, weight="600"))
            y += bh + gap
        y += ggap
    return _svg(w, h, "".join(b), label)
