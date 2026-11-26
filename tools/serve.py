"""Live preview: rebuild on every save and reload the browser.

    .venv/bin/python tools/serve.py [--port 4000]

Watches content/, templates/, static/, site.yaml and tools/build.py. Drafts (draft: true) are included here and
marked as such; the production build skips them. Build errors are shown as an overlay in the browser and in the
terminal, and the last good build stays up.
"""
import argparse
import importlib
import json
import shutil
import sys
import threading
import time
import traceback
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import build  # noqa: E402

ROOT = build.ROOT
SITE = ROOT / "_site"
WATCH = [ROOT / "content", ROOT / "templates", ROOT / "static", ROOT / "site.yaml", Path(build.__file__)]
STATE = {"version": 0, "error": None}

RELOAD_JS = """(() => {
  let v = null;
  const show = (err) => {
    let o = document.getElementById("__build_error");
    if (!err) { if (o) o.remove(); return; }
    if (!o) { o = document.createElement("pre"); o.id = "__build_error";
      o.style.cssText = "position:fixed;inset:auto 12px 12px 12px;max-height:50vh;overflow:auto;z-index:9999;" +
        "background:#2b0d0d;color:#ffd7d7;font:12px/1.4 ui-monospace,monospace;padding:12px;border-radius:8px";
      document.body.appendChild(o); }
    o.textContent = "build failed (last good build shown)\\n\\n" + err;
  };
  const poll = async () => {
    try {
      const s = await (await fetch("/__status", {cache: "no-store"})).json();
      if (v !== null && s.version !== v) { location.reload(); return; }
      v = s.version; show(s.error);
    } catch (e) {}
    setTimeout(poll, 700);
  };
  poll();
})();
"""


def snapshot():
    files = {}
    for p in WATCH:
        for f in ([p] if p.is_file() else p.rglob("*")):
            if f.is_file() and not f.name.startswith("."):
                files[f] = f.stat().st_mtime_ns
    return files


def rebuild():
    global build
    t0 = time.time()
    tmp = ROOT / "_site.tmp"
    try:
        build = importlib.reload(build)
        posts = build.build(dev=True, out=tmp)
        if SITE.exists():
            shutil.rmtree(SITE)
        tmp.rename(SITE)
        STATE["error"] = None
        print(f"[serve] built {len(posts)} post(s) in {time.time() - t0:.2f}s", flush=True)
    except Exception:
        STATE["error"] = traceback.format_exc()
        print(STATE["error"], file=sys.stderr, flush=True)
        shutil.rmtree(tmp, ignore_errors=True)
    STATE["version"] += 1


def watch():
    last = snapshot()
    while True:
        time.sleep(0.4)
        now = snapshot()
        if now != last:
            last = now
            rebuild()


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/__livereload.js":
            return self._send(RELOAD_JS.encode(), "application/javascript")
        if self.path == "/__status":
            return self._send(json.dumps(STATE).encode(), "application/json")
        return super().do_GET()

    def _send(self, data, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=4000)
    args = ap.parse_args()
    rebuild()
    threading.Thread(target=watch, daemon=True).start()
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=str(SITE)))
    print(f"[serve] http://127.0.0.1:{args.port}/  (Ctrl+C to stop)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
