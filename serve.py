#!/usr/bin/env python3
"""b2cviewer: serve the animated splat viewer over b2crig's work directory.

    python3 serve.py [--root ~/Projects/b2crig/work] [--port 8765] [--open]

Then open http://localhost:8765/. The viewer lists the b2c glTF files (~/Projects/b2cgltf/SPEC.md) and their web
forms (b2cgltf web.py) under ROOT/<subject>/gltf/: every subject file (a .glb that is not a clip, e.g. scene.glb, or
a .sog: b2cgltf's web subject, or any plain SOG splat; or a <name>.lods/ folder: b2cgltf's streamed web subject with
its LOD levels), whether b2crig has rigged it, and the clip files (<name>.clip.glb or <name>.clip.zip) beside it
that belong to it (the same b2c_id). Files are served read-only from under ROOT; nothing is written.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import shutil
import struct
import threading
import zipfile
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


def glb_bytes_json(b: bytes) -> dict | None:
    """The JSON chunk of a .glb held in memory."""
    try:
        magic, version, _, n, kind = struct.unpack_from("<4sIIII", b)
        return json.loads(b[20:20 + n]) if magic == b"glTF" and version == 2 and kind == 0x4E4F534A else None
    except (ValueError, struct.error):
        return None


def zip_member(p: Path, name: str) -> bytes | None:
    try:
        with zipfile.ZipFile(p) as z:
            return z.read(name)
    except (OSError, KeyError, zipfile.BadZipFile):
        return None


def clip_json(p: Path) -> dict | None:
    """A clip file's glTF JSON: the .clip.glb itself, or the clip.glb inside a .clip.zip."""
    if p.name.endswith(".clip.glb"):
        return glb_json(p)
    b = zip_member(p, "clip.glb")
    return b and glb_bytes_json(b)


def sog_subject(p: Path) -> tuple[str | None, bool] | None:
    """(b2c_id, rigged) of a .sog: b2cgltf's web subject names its rig.glb in meta.json's `b2c`; a plain SOG has
    neither. None when it is not a SOG."""
    try:
        meta = json.loads(zip_member(p, "meta.json") or b"null")
    except ValueError:
        return None
    if not isinstance(meta, dict) or meta.get("version") != 2:
        return None
    rig = meta.get("b2c", {}).get("rig")
    js = rig and glb_bytes_json(zip_member(p, rig) or b"")
    if not js:
        return None, False
    return js.get("asset", {}).get("extras", {}).get("b2c_id"), "B2CRIG_rig" in js.get("extensions", {})


def lods_subject(p: Path) -> tuple[str | None, bool] | None:
    """(b2c_id, rigged) of a streamed web subject (b2cgltf SPEC 10.3): a <name>.lods/ folder whose lod-meta.json has
    b2cgltf's `b2c` key, naming the shared rig.glb when rigged. None otherwise."""
    try:
        meta = json.loads((p / "lod-meta.json").read_text())
    except (OSError, ValueError):
        return None
    b2c = meta.get("b2c") if isinstance(meta, dict) else None
    if not isinstance(b2c, dict):
        return None
    if not b2c.get("rig"):
        return None, False
    try:
        js = glb_bytes_json((p / b2c["rig"]).read_bytes())
    except OSError:
        return None
    return (js.get("asset", {}).get("extras", {}).get("b2c_id"), True) if js else None


def index(root: Path, dirs=None, where: str = "<subject>/gltf/*.glb") -> dict:
    """The /api/index JSON: the subject files in `dirs`, (name, directory) pairs under `root` (default: b2crig's
    layout, every ROOT/<subject>/gltf/ named <subject>), each with the clip files beside it that belong to it.

    A host serving another layout (b2crunner: <run>/ply/) passes its own `dirs`; `path` is relative to `root` and is
    what the page asks /data/ for. `where` only goes into the page's message when nothing is found."""
    if dirs is None:
        dirs = [(p.parent.name, p) for p in sorted(root.glob("*/gltf")) if p.is_dir()]
    subjects = []
    for base, d in dirs:
        clips = {}
        for c in sorted([*d.glob("*.clip.glb"), *d.glob("*.clip.zip")]):
            js = clip_json(c)
            ext = js and js.get("extensions", {}).get("B2CRIG_clip")
            if ext:
                anim = js["animations"][0]
                acc = js["accessors"][anim["samplers"][0]["input"]] if anim.get("samplers") else {"count": 0}
                same = clips.setdefault(ext["subject"]["id"], [])
                name = ext["name"]   # a web clip beside its .clip.glb gets its own name: the page keys clips by name
                if c.name.endswith(".clip.zip") and any(x["name"] == name for x in same):
                    name += " (zip)"
                same.append({"name": name, "path": str(c.relative_to(root)), "frames": acc["count"], "fps": ext["fps"]})
        for s in sorted(p for p in d.glob("*.glb") if not p.name.endswith(".clip.glb")):
            js = glb_json(s)
            sid = js and js.get("asset", {}).get("extras", {}).get("b2c_id")
            if not sid:
                continue
            name = base if s.name == "scene.glb" else f"{base}/{s.stem}"
            subjects.append({"name": name, "path": str(s.relative_to(root)), "mb": round(s.stat().st_size / 2**20),
                             "rigged": "B2CRIG_rig" in js.get("extensions", {}), "clips": clips.get(sid, [])})
        for s in sorted(d.glob("*.sog")):
            got = sog_subject(s)
            if got is None:
                continue
            sid, rigged = got
            subjects.append({"name": f"{base}/{s.name}", "path": str(s.relative_to(root)), "mb": round(s.stat().st_size / 2**20),
                             "rigged": rigged, "clips": clips.get(sid, []) if sid else []})
        for s in sorted(d.glob("*.lods")):
            got = lods_subject(s)
            if got is None:
                continue
            sid, rigged = got
            mb = round(sum(f.stat().st_size for f in s.rglob("*") if f.is_file()) / 2**20)
            subjects.append({"name": f"{base}/{s.name}", "path": str(s.relative_to(root)), "mb": mb, "rigged": rigged,
                             "streamed": True, "clips": clips.get(sid, []) if sid else []})
    return {"root": str(root), "where": where, "subjects": subjects}


DATA_TYPES = {".glb": "model/gltf-binary", ".sog": "application/zip", ".clip.zip": "application/zip",
              ".json": "application/json", ".webp": "image/webp"}


def data_type(p: Path) -> str:
    return DATA_TYPES[".clip.zip" if p.name.endswith(".clip.zip") else p.suffix]


def data_file(root: Path, rel: str) -> Path | None:
    """The file /data/REL names, or None when REL leaves ROOT or is not a .glb, .sog or .clip.zip file, or a .json /
    .webp file in a streamed web subject's <name>.lods/ folder."""
    p = (root / rel).resolve()
    in_lods = any(q.suffix == ".lods" for q in p.parents if q.is_relative_to(root))
    ok = p.suffix in (".glb", ".sog") or p.name.endswith(".clip.zip") or (in_lods and p.suffix in (".json", ".webp"))
    return p if p.is_relative_to(root) and p.is_file() and ok else None


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
        p = data_file(self.root, rel)
        if p is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", data_type(p))
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
