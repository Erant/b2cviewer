# b2cviewer

An interactive viewer for b2crig's animated gaussian splats. It plays the b2c glTF files
(`~/Projects/b2cgltf/SPEC.md`): a subject file and, once b2crig has rigged it, that subject's clip files.

```
python3 serve.py [--root ~/Projects/b2crig/work] [--port 8765] [--open]
```

Open http://localhost:8765/ in a browser with WebGL2 (Firefox, Chrome). `serve.py` is stdlib-only and serves
the files read-only.

- **Subject**: every subject file under `<root>/<subject>/gltf/` (`scene.glb`, or any other `.glb` that is not a
  clip), and every `.sog` there (listed as `<subject>/<file>`). A file b2crig has not rigged yet (straight from
  b2crunner) shows its splat as delivered, unposed, with no clips; `tools/export_gltf.py rig` in b2crig rigs it.
- **Clip**: the `<name>.clip.glb` and `<name>.clip.zip` files beside it that belong to it (the same `b2c_id`), or the
  rest pose. A clip made for another rig (its `cageSha256` differs) or another skeleton is refused, as SPEC 6
  requires; re-export it. A `.clip.zip` beside a `.clip.glb` of the same name is listed as `<name> (zip)`.
- **Web forms** (b2cgltf SPEC 10, `python -m b2cgltf.web`): a rigged subject's `.sog` (a SOG splat with the binding
  and the rig) and its `.clip.zip` clips play like the glTF files. Any other SOG (b2csite's splats, splat-transform
  output) is shown as a plain splat, turned 180 degrees about z as PlayCanvas shows SOG files. Reading a `.sog` needs
  WebGL2 in a worker (OffscreenCanvas) to decode its WebP textures without premultiplying alpha.
- **Streamed web subjects** (b2cgltf SPEC 10.3, `python -m b2cgltf.web subject-lods`): a `<name>.lods/` folder
  (`lod-meta.json` + one SOG folder per LOD level + the shared `rig.glb`) is listed as `<subject>/<name>.lods` and
  fetched one level at a time; a level fetched once is kept for later switches. It opens with Detail on auto. Served
  only (serve.py), not opened from local files.
- A file `b2cgltf.lod.strip_lods` made (its levels removed, `b2c_stripped`) opens like its source and plays its clips
  (it keeps the `b2c_id`).
- The selection is kept in the URL (`?subject=b24be4&clip=theater`), so links reopen the same view.
- **Open files…** (<kbd>Ctrl</kbd>+<kbd>O</kbd>) or dropping files onto the page views local files: a subject
  `.glb` or `.sog` and its `.clip.glb` or `.clip.zip` files, together or the clips later. They are read in the browser, not uploaded (the page
  still has to come from `serve.py`; any `--root` works).

A subject file b2crig has not rigged hides what only a rig drives: the clip picker, the posing options and the
timeline.

**Inside b2crunner**: its server serves this page at `/viewer/` over its runs (one subject per run's `ply/`, named
after the run; `pipeline/viewer.py` there), reusing `serve.index()` and `serve.data_file()` from this checkout. The
page's own requests are relative (`api/index`, `data/…`) so it works under such a prefix.

## Controls

| | |
|---|---|
| drag / right-drag or shift-drag / scroll | orbit / pan / zoom; double-click resets |
| <kbd>Space</kbd>, <kbd>←</kbd> <kbd>→</kbd> | play / pause, step a frame (the clip's own fps by default) |
| <kbd>R</kbd> | the clip's residual on / off: b2crig's posed cage, or plain skinning of the cage |
| <kbd>F</kbd> | follow the body (the orbit target tracks the smoothed body centroid) |
| <kbd>B</kbd> or **⇄** | swap the clip with its A/B partner, `CLIP` ⇄ `CLIP_fix`, at the same frame and view (a badge in the corner says which one is showing) |
| <kbd>S</kbd> or **⤓ PNG** | save the view as `b2cview_<subject>_<clip>_f<frame>_<time>.png` (Firefox: to Downloads). A `tEXt` chunk `b2cviewer` holds JSON: the subject and clip files, the frame, the render options and the camera as a b2crig cameras.json entry in the b2crunner frame; b2crig's `tools/viewer_shot.py` re-renders exactly that view with b2ctrain |
| <kbd>L</kbd> + drag | move the light (relighting on); dragging its dot does the same |

**Lighting** (experimental, viewer only): *Show* relit, the normals (glTF world as RGB) or the lighting alone on grey.
The captured colour is taken as albedo under an even light; it is multiplied by a hemisphere ambient (sky / ground
colours) plus one point, spot or sun light (power, colour temperature and tint, azimuth / elevation / distance around
the body, range, cone), with a wrap term, an optional Blinn-Phong specular, exposure and a soft highlight shoulder.
The light casts shadows of the posed cage (*Shadows*: strength, softness, bias; rigged subjects only).
Drag the light's dot (a hollow dot at the edge when it is off screen; faded when behind the body), or hold <kbd>L</kbd>
and drag anywhere, to move it over the sphere of its distance; scroll over the dot for the distance. The light follows
the body or stays put, and can move with the camera. The settings are kept in the URL (`?light=` JSON of the
non-default values) and in the PNG's `lighting`. With *ambient* 1, no light and no shoulder the relit view is the
captured one.

Also in the panel: the stretch fade (`--cage-fade-start/--cage-fade-end`), the max triangle growth
(`--cage-max-growth`), the SH degree and the background colour. The growth and fade start from the subject file's
render settings (`B2CRIG_rig.render`).

**Detail** (shown only for a splat with LOD levels, b2cgltf SPEC 4.7): the full splat or one of its levels
(`MSFT_lod` + `B2C_lod` / `B2CRIG_lod`), each labelled with its splat count and the largest on-screen height it is
meant for (`maxScreenHeightPx`). A rigged level is posed by its own binding and plays the same clips; switching keeps
the clip, the frame and the view. A chain that is stale (its `source` is not the base's) or broken is not offered, and
the info line says why. **auto** picks the coarsest level whose `maxScreenHeightPx` covers the subject's height on
screen (device pixels), once the wanted level holds for 0.4 s, and drops to a coarser level only 15 % below its limit;
a `.glb` opens at full detail and then switches. The level is kept in the URL (`?lod=1`, `?lod=auto`) and in the PNG's
`lod`. A `.sog` holds one level (`web.subject --level`); a streamed web subject holds them all.

## How it works

It mirrors `b2ctrain render --cage CAGE --cage-max-growth 1.15`:

- `web/gltf.js` reads the files. From the subject file it takes the current splat (SPEC 7.1), b2ctrain's binding of
  it (`B2CRIG_splat_cage`), the cage (`b2crig_cage`, one primitive per layer) and the skeleton. It checks a clip file
  against the subject (`b2c_id`, the rig hash, the skeleton copy) and poses a frame's cage as SPEC 7.2 says: skinning
  by the joints relative to the skeleton root, plus the clip's `B2CRIG_cage_residual`. All of this stays in the
  b2crunner frame; the orbit camera lives in glTF world and is mapped through `W`.
- `web/sog.js` reads the web forms: a ZIP reader (deflate through `DecompressionStream`) and a SOG v2 decoder that
  gives the splat in the same form as a subject file's, plus the binding textures and `rig.glb` of a web subject
  (gltf.js takes the rig from there as from a subject file).
- `web/rig.js` ports `gpu/cage.cu`'s posing: per frame it builds the posed triangle frames (with the growth clamp) and
  the posed splat centres.
- `web/rig-worker.js` runs the parsing, posing and depth sorting off the UI thread.
- `web/light.js` and `relight` in `web/renderer.js`: the relighting. Each splat's normal is the posed cage's
  (area-weighted vertex normals per frame, `Rig.frame`, interpolated with the splat's barycentrics) turned towards
  the splat's own shortest axis as far as the splat is flat (*Normals* slider). Splats without a cage face use their
  shortest axis, facing the camera; a normal facing away from the camera is reflected into the visible half (the cage
  has surfaces that face away, e.g. inside the mouth). Shadows: per frame the posed cage's triangles (from the face
  texture) go into a 2048² map of linear depth from the light, framed on the body's bounding sphere (perspective; a
  sun's orthographic). A bound splat looks its shadow up at its anchor on its cage triangle, so splats a little
  inside or outside the cage take the surface's shadow; a slope-scaled bias and a normal offset keep surfaces at a
  grazing angle from shadowing themselves, and 12 taps on a disc soften the edge.
- `web/renderer.js` poses every splat in the vertex shader from a per-face texture (`pose_bound` and the stretch
  fade), evaluates SH in the triangle's canonical frame and projects like b2ctrain's rasteriser (EWA, 0.3 px blur,
  1/255 cutoff). It blends into a half-float target.

Not supported yet: the dual binding (`alt*` in `B2CRIG_splat_cage`), the per-splat state (`_B2CRIG_OPEN_*`,
`_B2CRIG_CAGE_FILL`, `_B2CRIG_GATE_*`), the stretch fill, `B2CRIG_open_gate` and the appearance MLP
(`B2CRIG_cage_app`).
