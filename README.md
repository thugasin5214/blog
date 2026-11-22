# blog

Static blog, published with GitHub Pages.

- Posts live in `posts/<slug>/index.html`; the front page is `index.html`.
- Both are generated: `python3 tools/build.py` (content, data and inline-SVG charts are in `tools/`).
- No external build dependencies; KaTeX and the web font are loaded from CDNs.
