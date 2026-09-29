#!/usr/bin/env python3
"""b2cviewer: serve the animated splat viewer over b2crig's work directory.

    python3 serve.py [--root ~/Projects/b2crig/work] [--port 8765] [--open]

Then open http://localhost:8765/. The viewer lists every subject under ROOT, its trained splats
(ROOT/<subject>/train/<tag>/scene_pruned_ev.ply or scene.ply, with the appearance MLP scene.app when there is one)
and its clips' posed cages (ROOT/<subject>/clips/<clip>/cage_v2.b2ccage or cage.b2ccage, like tools/dance_eval.py).
Files are served read-only from under ROOT; nothing is written.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import shutil
import struct
import threading
import webbrowser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

WEB = Path(__file__).resolve().parent / "web"
PLYS = ("scene_pruned_ev.ply", "scene.ply")
CAGES = ("cage_v2.b2ccage", "cage.b2ccage")


def cage_header(p: Path) -> dict | None:
    try:
        with open(p, "rb") as f:
            if f.read(8) != b"B2CCAGE1":
                return None
            nl, nv, nf, nfr = struct.unpack("<4i", f.read(16))
        return {"nv": nv, "nf": nf, "frames": nfr}
    except OSError:
        return None


def app_header(p: Path) -> dict | None:
    try:
        with open(p, "rb") as f:
            magic = f.read(8)
            if not magic.startswith(b"B2CAPP"):
                return None
            nv, nfeat, *_ = struct.unpack("<5i", f.read(20))
        return {"nv": nv, "version": magic.decode(), "nfeat": nfeat}
    except OSError:
        return None


def index(root: Path) -> dict:
    subjects = []
    for s in sorted(p for p in root.iterdir() if p.is_dir()):
        runs, clips = [], []
        for d in sorted((s / "train").glob("*/")) if (s / "train").is_dir() else []:
            ply = next((d / n for n in PLYS if (d / n).is_file()), None)
            if not ply:
                continue
            app = d / "scene.app"
            ah = app_header(app) if app.is_file() else None
            runs.append({"tag": d.name, "ply": str(ply.relative_to(root)), "ply_mb": round(ply.stat().st_size / 2**20),
                         "app": str(app.relative_to(root)) if ah else None, "app_nv": ah and ah["nv"],
                         "app_version": ah and ah["version"], "mtime": ply.stat().st_mtime})
        for d in sorted((s / "clips").glob("*/")) if (s / "clips").is_dir() else []:
            cage = next((d / n for n in CAGES if (d / n).is_file()), None)
            h = cage and cage_header(cage)
            if not h:
                continue
            wan = d / "wan"
            clips.append({"name": d.name, "cage": str(cage.relative_to(root)), **h,
                          "cameras": str((d / "cameras.json").relative_to(root)) if (d / "cameras.json").is_file() else None,
                          "wan": str(wan.relative_to(root)) if wan.is_dir() and any(wan.glob("*.png")) else None})
        if runs and clips:
            subjects.append({"name": s.name, "runs": runs, "clips": clips})
    return {"root": str(root), "subjects": subjects}


class Handler(SimpleHTTPRequestHandler):
    root: Path

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(WEB), **kw)

    def log_message(self, fmt, *args):   # quiet: only errors
        if args and str(args[1])[0] in "45":
            super().log_message(fmt, *args)

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path == "/api/index":
            body = json.dumps(index(self.root)).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path.startswith("/data/"):
            self.send_data(path[len("/data/"):])
        else:
            super().do_GET()

    def send_data(self, rel: str):
        p = (self.root / rel).resolve()
        if not p.is_relative_to(self.root) or not p.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(p.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(p.stat().st_size))
        self.end_headers()
        with open(p, "rb") as f:
            try:
                shutil.copyfileobj(f, self.wfile, 1 << 20)
            except (BrokenPipeError, ConnectionResetError):
                pass


mimetypes.add_type("text/javascript", ".js")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path.home() / "Projects" / "b2crig" / "work")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true", help="open the viewer in the default browser")
    a = ap.parse_args()
    Handler.root = a.root.expanduser().resolve()
    if not Handler.root.is_dir():
        ap.error(f"--root {Handler.root} is not a directory")
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    url = f"http://{'localhost' if a.host in ('127.0.0.1', '0.0.0.0') else a.host}:{srv.server_port}/"
    print(f"b2cviewer: {url}  (root {Handler.root})")
    if a.open:
        threading.Timer(0.5, webbrowser.open, (url,)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    os.chdir(WEB)
    main()
