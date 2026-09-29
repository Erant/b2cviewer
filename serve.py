#!/usr/bin/env python3
"""b2cviewer: serve the animated splat viewer over b2crig's work directory.

    python3 serve.py [--root ~/Projects/b2crig/work] [--port 8765] [--open]

Then open http://localhost:8765/. The viewer lists the b2c glTF files (~/Projects/b2cgltf/SPEC.md) under
ROOT/<subject>/gltf/: every subject file (a .glb that is not a clip, e.g. scene.glb), whether b2crig has rigged it,
and the clip files (<name>.clip.glb) beside it that belong to it (the same b2c_id). Files are served read-only from
under ROOT; nothing is written.
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


def glb_json(p: Path) -> dict | None:
    """The JSON chunk of a .glb (the header and first chunk only; the BIN chunk is not read)."""
    try:
        with open(p, "rb") as f:
            magic, version, _ = struct.unpack("<4sII", f.read(12))
            n, kind = struct.unpack("<II", f.read(8))
            if magic != b"glTF" or version != 2 or kind != 0x4E4F534A:
                return None
            return json.loads(f.read(n))
    except (OSError, ValueError, struct.error):
        return None


def index(root: Path) -> dict:
    subjects = []
    for d in sorted(p for p in root.glob("*/gltf") if p.is_dir()):
        clips = {}
        for c in sorted(d.glob("*.clip.glb")):
            js = glb_json(c)
            ext = js and js.get("extensions", {}).get("B2CRIG_clip")
            if ext:
                anim = js["animations"][0]
                acc = js["accessors"][anim["samplers"][0]["input"]] if anim.get("samplers") else {"count": 0}
                clips.setdefault(ext["subject"]["id"], []).append(
                    {"name": ext["name"], "path": str(c.relative_to(root)), "frames": acc["count"], "fps": ext["fps"]})
        for s in sorted(p for p in d.glob("*.glb") if not p.name.endswith(".clip.glb")):
            js = glb_json(s)
            sid = js and js.get("asset", {}).get("extras", {}).get("b2c_id")
            if not sid:
                continue
            name = d.parent.name if s.name == "scene.glb" else f"{d.parent.name}/{s.stem}"
            subjects.append({"name": name, "path": str(s.relative_to(root)), "mb": round(s.stat().st_size / 2**20),
                             "rigged": "B2CRIG_rig" in js.get("extensions", {}), "clips": clips.get(sid, [])})
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
        if not p.is_relative_to(self.root) or not p.is_file() or p.suffix != ".glb":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "model/gltf-binary")
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
