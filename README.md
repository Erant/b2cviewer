# b2cviewer

An interactive viewer for b2crig's animated gaussian splats. It plays the b2c glTF files
(`~/Projects/b2cgltf/SPEC.md`): a subject file and, once b2crig has rigged it, that subject's clip files.

```
python3 serve.py [--root ~/Projects/b2crig/work] [--port 8765] [--open]
```

Open http://localhost:8765/ in a browser with WebGL2 (Firefox, Chrome). `serve.py` is stdlib-only and serves
the files read-only.

- **Subject**: every subject file under `<root>/<subject>/gltf/` (`scene.glb`, or any other `.glb` that is not a
  clip). A file b2crig has not rigged yet (straight from b2crunner) shows its splat as delivered, unposed, with no
  clips; `tools/export_gltf.py rig` in b2crig rigs it.
- **Clip**: the `<name>.clip.glb` files beside it that belong to it (the same `b2c_id`), or the rest pose. A clip
  made for another rig (its `cageSha256` differs) or another skeleton is refused, as SPEC 6 requires; re-export it.
- The selection is kept in the URL (`?subject=b24be4&clip=theater`), so links reopen the same view.
- **Open .glb files…** (<kbd>Ctrl</kbd>+<kbd>O</kbd>) or dropping files onto the page views local files: a subject
  `.glb` and its `.clip.glb` files, together or the clips later. They are read in the browser, not uploaded (the page
  still has to come from `serve.py`; any `--root` works).

## Controls

| | |
|---|---|
| drag / right-drag or shift-drag / scroll | orbit / pan / zoom; double-click resets |
| <kbd>Space</kbd>, <kbd>←</kbd> <kbd>→</kbd> | play / pause, step a frame (the clip's own fps by default) |
| <kbd>R</kbd> | the clip's residual on / off: b2crig's posed cage, or plain skinning of the cage |
| <kbd>F</kbd> | follow the body (the orbit target tracks the smoothed body centroid) |
| <kbd>B</kbd> or **⇄** | swap the clip with its A/B partner, `CLIP` ⇄ `CLIP_fix`, at the same frame and view (a badge in the corner says which one is showing) |
| <kbd>S</kbd> or **⤓ PNG** | save the view as `b2cview_<subject>_<clip>_f<frame>_<time>.png` (Firefox: to Downloads). A `tEXt` chunk `b2cviewer` holds JSON: the subject and clip files, the frame, the render options and the camera as a b2crig cameras.json entry in the b2crunner frame; b2crig's `tools/viewer_shot.py` re-renders exactly that view with b2ctrain |

Also in the panel: the stretch fade (`--cage-fade-start/--cage-fade-end`), the max triangle growth
(`--cage-max-growth`), the SH degree and the background colour. The growth and fade start from the subject file's
render settings (`B2CRIG_rig.render`).

## How it works

It mirrors `b2ctrain render --cage CAGE --cage-max-growth 1.15`:

- `web/gltf.js` reads the files. From the subject file it takes the current splat (SPEC 7.1), b2ctrain's binding of
  it (`B2CRIG_splat_cage`), the cage (`b2crig_cage`, one primitive per layer) and the skeleton. It checks a clip file
  against the subject (`b2c_id`, the rig hash, the skeleton copy) and poses a frame's cage as SPEC 7.2 says: skinning
  by the joints relative to the skeleton root, plus the clip's `B2CRIG_cage_residual`. All of this stays in the
  b2crunner frame; the orbit camera lives in glTF world and is mapped through `W`.
- `web/rig.js` ports `gpu/cage.cu`'s posing: per frame it builds the posed triangle frames (with the growth clamp) and
  the posed splat centres.
- `web/rig-worker.js` runs the parsing, posing and depth sorting off the UI thread.
- `web/renderer.js` poses every splat in the vertex shader from a per-face texture (`pose_bound` and the stretch
  fade), evaluates SH in the triangle's canonical frame and projects like b2ctrain's rasteriser (EWA, 0.3 px blur,
  1/255 cutoff). It blends into a half-float target.

Not supported yet: the dual binding (`alt*` in `B2CRIG_splat_cage`), the per-splat state (`_B2CRIG_OPEN_*`,
`_B2CRIG_CAGE_FILL`, `_B2CRIG_GATE_*`), the stretch fill, `B2CRIG_open_gate` and the appearance MLP
(`B2CRIG_cage_app`).
