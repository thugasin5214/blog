"""Build the blog from Markdown: content/ + templates/ + static/  ->  _site/

    .venv/bin/python tools/build.py            # production build (drafts skipped)
    .venv/bin/python tools/serve.py            # live preview, see serve.py

Markdown conventions (all render sensibly in Typora / Obsidian / VS Code too):
  - YAML front matter: title, subtitle, description, date, author, tags, categories, draft, math, toc
  - `## Heading {#id}`         fixed anchor id (otherwise one is derived from the text)
  - `[TOC]` on its own line    table of contents built from the ## headings
  - `> [!tldr] Title`          callout; the type becomes the CSS class (tldr, note, lede, small, ...)
  - `![alt](figures/x.svg)` followed on the next line by `*caption*`  ->  numbered <figure>
    (SVGs whose root has class="chart" are inlined so they follow the page's light / dark theme)
  - `$inline$`, `$$display$$`  KaTeX;  ```lang fences  Pygments;  ```mermaid fences  Mermaid
  - `## References {#refs}` followed by a numbered list  ->  items get ids r1, r2, ... for `[[3]](#r3)` citations
"""
import datetime as dt
import html
import re
import shutil
import sys
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape
from markdown_it import MarkdownIt
from markdown_it.token import Token
from mdit_py_plugins.dollarmath import dollarmath_plugin
from mdit_py_plugins.footnote import footnote_plugin
from mdit_py_plugins.tasklists import tasklists_plugin
from pygments import highlight as pyg_highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound

ROOT = Path(__file__).resolve().parents[1]
CONTENT, TEMPLATES, STATIC, OUT = ROOT / "content", ROOT / "templates", ROOT / "static", ROOT / "_site"
FRONT = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)


# ---------------------------------------------------------------- markdown
def _highlight(code, lang, _attrs):
    if lang == "mermaid":
        return f'<pre class="mermaid">{html.escape(code)}</pre>'
    try:
        lexer = get_lexer_by_name(lang) if lang else None
    except ClassNotFound:
        lexer = None
    body = pyg_highlight(code, lexer, HtmlFormatter(nowrap=True)) if lexer else html.escape(code)
    cls = f' class="language-{html.escape(lang)}"' if lang else ""
    return f'<pre class="highlight"><code{cls}>{body}</code></pre>'


def _slug(text, seen):
    s = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    s = re.sub(r"[\s_]+", "-", s) or "section"
    base, n = s, 2
    while s in seen:
        s, n = f"{base}-{n}", n + 1
    seen.add(s)
    return s


def _heading_ids(state):
    """`## Title {#id}` -> id="id"; otherwise a slug of the text."""
    seen = set()
    toks = state.tokens
    for i, t in enumerate(toks):
        if t.type != "heading_open":
            continue
        inline = toks[i + 1]
        m = re.search(r"\s*\{#([\w-]+)\}\s*$", inline.content)
        if m:
            hid = m.group(1)
            seen.add(hid)
            inline.content = inline.content[:m.start()]
            last = inline.children[-1]
            last.content = re.sub(r"\s*\{#[\w-]+\}\s*$", "", last.content)
        else:
            hid = _slug(inline.content, seen)
        t.attrSet("id", hid)


def _callouts(state):
    """`> [!type] optional title` blockquotes -> <div class="callout type">[<h3>title</h3>] ...</div>."""
    toks = state.tokens
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.type == "blockquote_open" and i + 2 < len(toks) and toks[i + 2].type == "inline":
            inline = toks[i + 2]
            m = re.match(r"\[!([\w-]+)\][+-]?[ \t]*([^\n]*)", inline.content)
            if m:
                kind, title = m.group(1).lower(), m.group(2).strip()
                depth, j = 0, i
                while True:  # matching close
                    if toks[j].type == "blockquote_open":
                        depth += 1
                    elif toks[j].type == "blockquote_close":
                        depth -= 1
                        if depth == 0:
                            break
                    j += 1
                for k in (i, j):
                    toks[k].tag = "div"
                toks[i].attrSet("class", f"callout {kind}")
                kids = inline.children
                sb = next((k for k, c in enumerate(kids) if c.type in ("softbreak", "hardbreak")), len(kids))
                rest = kids[sb + 1:]
                inline.content = inline.content.split("\n", 1)[1] if "\n" in inline.content else ""
                new = []
                if title:
                    h_open = Token("heading_open", "h3", 1)
                    h_inl = Token("inline", "", 0)
                    h_inl.content, h_inl.children = title, state.md.parseInline(title)[0].children
                    new = [h_open, h_inl, Token("heading_close", "h3", -1)]
                if rest:
                    inline.children = rest
                    toks[i + 1:i + 1] = new
                else:  # first paragraph held only the marker line
                    toks[i + 1:i + 4] = new
                i += 1
                continue
        i += 1


def make_md():
    md = (MarkdownIt("commonmark", {"html": True, "highlight": _highlight})
          .enable(["table", "strikethrough"])
          .use(dollarmath_plugin, double_inline=True)
          .use(footnote_plugin)
          .use(tasklists_plugin))
    md.add_render_rule("math_inline", lambda self, tokens, idx, options, env:
                       f"\\({html.escape(tokens[idx].content)}\\)")
    md.add_render_rule("math_inline_double", lambda self, tokens, idx, options, env:
                       f"\\[{html.escape(tokens[idx].content)}\\]")
    md.add_render_rule("math_block", lambda self, tokens, idx, options, env:
                       f"\\[{html.escape(tokens[idx].content.strip())}\\]\n")
    md.core.ruler.push("heading_ids", _heading_ids)
    md.core.ruler.push("callouts", _callouts)
    return md


# ---------------------------------------------------------------- html post-processing
def _svg_inline(path):
    svg = path.read_text()
    if not re.match(r'\s*<svg[^>]*class="chart"', svg):
        return None
    return re.sub(r"<style data-standalone[^>]*>.*?</style>", "", svg, flags=re.S).strip()


def figures(body, src_dir):
    n = 0

    def one(m):
        nonlocal n
        src, alt, cap = html.unescape(m.group(1)), m.group(2), m.group(3)
        local = src_dir / src
        inner = _svg_inline(local) if src.endswith(".svg") and local.is_file() else None
        inner = inner or f'<img src="{m.group(1)}" alt="{alt}" loading="lazy">'
        if cap is None:
            return f"<figure>{inner}</figure>"
        n += 1
        return f"<figure>{inner}<figcaption><b>Figure {n}.</b> {cap}</figcaption></figure>"

    return re.sub(r'<p><img src="([^"]+)" alt="([^"]*)"\s*/?>(?:\s*<em>(.*?)</em>)?</p>', one, body, flags=re.S)


def references(body):
    m = re.search(r'<h2 id="refs">.*?</h2>\s*<ol>(.*?)</ol>', body, re.S)
    if not m:
        return body
    i = 0

    def li(_):
        nonlocal i
        i += 1
        return f'<li id="r{i}">'

    items = re.sub(r"<li>", li, m.group(1))
    return body[:m.start(1) - 4] + f'<ol class="refs">{items}</ol>' + body[m.end(1) + 5:]


def toc(body):
    items = [(hid, re.sub(r"<[^>]+>", "", t)) for hid, t in re.findall(r'<h2 id="([^"]+)">(.*?)</h2>', body)
             if hid != "refs"]
    lis = "".join(f'<li><a href="#{hid}">{re.sub(r"^\d+\.\s*", "", t)}</a></li>' for hid, t in items)
    nav = f'<nav class="toc"><b>Contents</b>\n<ol>{lis}</ol></nav>'
    return body.replace("<p>[TOC]</p>", nav)


# ---------------------------------------------------------------- site
def load_posts(md, drafts):
    posts = []
    for src in sorted(CONTENT.glob("posts/*/index.md")) + sorted(CONTENT.glob("posts/*.md")):
        text = src.read_text()
        m = FRONT.match(text)
        meta = yaml.safe_load(m.group(1)) if m else {}
        if meta.get("draft") and not drafts:
            continue
        slug = src.parent.name if src.name == "index.md" else src.stem
        body = md.render(text[m.end():] if m else text)
        body = toc(references(figures(body, src.parent)))
        words = len(re.sub(r"<svg.*?</svg>|<[^>]+>", " ", body, flags=re.S).split())
        date = meta.get("date")
        if isinstance(date, str):
            date = dt.date.fromisoformat(date)
        posts.append(dict(meta, slug=slug, src=src, body=body, date=date,
                          minutes=meta.get("reading_time") or max(1, round(words / 230)),
                          math=meta.get("math", "\\(" in body or "\\[" in body),
                          mermaid='class="mermaid"' in body))
    posts.sort(key=lambda p: p["date"] or dt.date.min, reverse=True)
    return posts


def build(dev=False, out=OUT):
    site = yaml.safe_load((ROOT / "site.yaml").read_text())
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]),
                      trim_blocks=True, lstrip_blocks=True)
    env.filters["date"] = lambda d, fmt=None: d.strftime(fmt or site.get("date_format", "%d %B %Y")) if d else ""
    md = make_md()
    posts = load_posts(md, drafts=dev)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copytree(STATIC, out / "assets")
    (out / "assets" / "pygments.css").write_text(pygments_css())
    (out / ".nojekyll").write_text("")
    common = dict(site=site, dev=dev, year=dt.date.today().year)
    for p in posts:
        dest = out / "posts" / p["slug"]
        shutil.copytree(p["src"].parent, dest, ignore=shutil.ignore_patterns("*.md", ".*"), dirs_exist_ok=True) \
            if p["src"].name == "index.md" else dest.mkdir(parents=True, exist_ok=True)
        (dest / "index.html").write_text(env.get_template("post.html").render(post=p, root="../../", **common))
    (out / "index.html").write_text(env.get_template("home.html").render(posts=posts, root="./", **common))
    return posts


def pygments_css():
    """Token colours for light and dark; the block background comes from style.css (var(--code))."""
    def rules(style, prefix=""):
        css = HtmlFormatter(style=style).get_style_defs(".highlight").splitlines()
        return [prefix + l for l in css if l.startswith(".highlight ") and not l.startswith(".highlight {")]
    auto_dark = ':root:not([data-theme="light"]) '
    return "\n".join(rules("friendly")
                     + ["@media (prefers-color-scheme:dark){"] + rules("github-dark", auto_dark) + ["}"]
                     + rules("github-dark", ':root[data-theme="dark"] ')) + "\n"


if __name__ == "__main__":
    posts = build(dev="--drafts" in sys.argv)
    print(f"built {len(posts)} post(s) -> {OUT.relative_to(ROOT)}/")
