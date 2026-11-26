# blog

A Markdown-driven static blog, published to GitHub Pages by GitHub Actions.
Markdown in `content/` is the only source of truth; the HTML in `_site/` is generated and never edited or committed.

## Layout

```
content/posts/<slug>/index.md     one post = one folder: the Markdown plus its images / figures
content/posts/<slug>/figures/     figures referenced from the post (SVG, PNG, ...)
templates/                        Jinja2 page templates (base, post, home)
static/style.css                  the site's CSS (published as assets/style.css)
site.yaml                         site title, author, description, footer, date format
tools/build.py                    Markdown -> _site/
tools/serve.py                    live preview
tools/figures/                    Python that draws the data charts / diagrams of the first post
.github/workflows/pages.yml       build + deploy on push to main
```

A post's URL is `/posts/<slug>/`, from its folder name.

## Setup (once)

```sh
uv venv && uv pip install -r requirements.txt      # or: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Writing

```sh
.venv/bin/python tools/serve.py        # http://127.0.0.1:4000, rebuilds and reloads on every save
```

Edit `content/posts/<slug>/index.md` in any editor (Typora, Obsidian, VS Code ...). Drafts (`draft: true`) show
up in the preview, marked as drafts, and are left out of the published site. Build errors appear in the
terminal and as an overlay in the browser.

New post: create `content/posts/my-new-post/index.md`:

```markdown
---
title: My new post
subtitle: One-line summary shown under the title and on the front page
description: For search engines and link previews (defaults to the subtitle)
date: 2026-12-01
tags: [mlx, lora]
categories: [post-training]
draft: true
---

Text ...
```

## Markdown features

| Write | Get |
|---|---|
| `$x^2$`, `$$ ... $$` | KaTeX math (inline / display) |
| ` ```python ` fences | syntax highlighting (Pygments), light and dark |
| ` ```mermaid ` fences | Mermaid diagrams |
| GitHub-style tables, `~~strike~~`, `- [ ] tasks`, footnotes `[^1]` | as usual |
| `![alt](figures/x.png)` and on the **next line** `*caption*` | numbered figure, "Figure N. caption" |
| `## Heading {#id}` | fixed anchor id (otherwise derived from the heading text) |
| `[TOC]` on its own line | contents list built from the `##` headings |
| `> [!tldr] TL;DR` / `> [!note]` / `> [!lede]` / `> [!small]` | callout boxes in the site style; any other type (e.g. `[!warning]`) gets the note box. Obsidian renders these natively |
| `## References {#refs}` + a numbered list | items get ids `r1`, `r2` ...; cite them as `[[3]](#r3)` |
| raw HTML | passed through (for the rare thing Markdown cannot express) |

SVG figures whose root element has `class="chart"` (the ones made by `tools/figures/`) are inlined into the page
so they follow the site's light / dark theme; they also carry their own fallback style, so editors preview them
too. Any other image is linked as a normal `<img>`.

## Charts of the first post

The measured numbers live in `tools/figures/make_figures.py`. After changing data or drawing code:

```sh
cd tools/figures && ../../.venv/bin/python make_figures.py      # rewrites content/posts/.../figures/*.svg
```

## Publishing

Push to `main`. The workflow installs `requirements.txt`, runs `tools/build.py` and deploys `_site/` to GitHub Pages
(repository Settings → Pages → Source must be "GitHub Actions"). A local production build is
`.venv/bin/python tools/build.py`.
