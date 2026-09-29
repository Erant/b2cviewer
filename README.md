# b2cviewer

An interactive viewer for b2crig's animated gaussian splats: a trained subject splat driven by a clip's posed cage,
with the pose-dependent appearance MLP (`--cage-app`, the `.app` next to the ply).

```
python3 serve.py [--root ~/Projects/b2crig/work] [--port 8765] [--open]
```

Open http://localhost:8765/ in a browser with WebGL2 (Firefox, Chrome). `serve.py` is stdlib-only. It lists every
subject under the root that has both trained splats and clips, and serves the files read-only.

- **Splat**: `<subject>/train/<tag>/scene_pruned_ev.ply` (or `scene.ply`). Tags with a `scene.app` are marked
  `· MLP` and listed first.
- **Clip**: `<subject>/clips/<clip>/cage_v2.b2ccage` or `cage.b2ccage`, as `tools/dance_eval.py` picks it. The MLP
  is per cage vertex, so it only runs with cages of the same topology it was trained on. Clips with another cage
  are marked `· other cage`; they still play, but without the MLP.
- The selection is kept in the URL (`?subject=daef0c&run=dG8O1&clip=d_zorbas_G3`), so links reopen the same view.
- You can also drop local `.ply` / `.app` / `.b2ccage` (+ `cameras.json`) files onto the page.

## Controls

| | |
|---|---|
| drag / right-drag or shift-drag / scroll | orbit / pan / zoom; double-click resets |
| <kbd>Space</kbd>, <kbd>←</kbd> <kbd>→</kbd> | play / pause, step a frame |
| <kbd>A</kbd> | pose-dependent MLP on / off (compare the plain LBS splat) |
| <kbd>D</kbd> | show the MLP's blend weight as magenta (`--cage-app-debug`) |
| <kbd>C</kbd> | clip camera: the camera the WAN clip was generated from |
| <kbd>W</kbd> | the clip's WAN frame alongside the render |
| <kbd>F</kbd> | follow the body (the orbit target tracks the smoothed body centroid) |
| <kbd>B</kbd> or **⇄** | swap the clip with its A/B partner, `CLIP` ⇄ `CLIP_fix`, at the same frame and view (a badge in the corner says which one is showing); e.g. b2crig's retarget fixes baked next to the originals |
| <kbd>S</kbd> or **⤓ PNG** | save the view as `b2cview_<subject>_<run>_<clip>_f<frame>_<time>.png` (Firefox: to Downloads). A `tEXt` chunk `b2cviewer` holds JSON: subject, run, clip, frame, the render options and the camera as a b2crig cameras.json entry; b2crig's `tools/viewer_shot.py` re-renders exactly that view with b2ctrain |

Also in the panel: the stretch fade (`--cage-fade-start/--cage-fade-end`, off by default), the max triangle growth
(`--cage-max-growth`, default 1.15 as `b2crig/b2ctrain.py` always passes), the SH degree and the background colour.

## How it works

It mirrors `b2ctrain render --cage CAGE --cage-max-growth 1.15 --cage-app APP`:

- `web/formats.js` reads the ply (with `seg_label` / `seg_conf`), the cage (`B2CCAGE1`) and the MLP (`B2CAPP01..03`;
  older files are widened with zero weights as `CageApp::load` does).
- `web/rig.js` ports `gpu/cage.cu` and `gpu/cage_app.cu`. It binds each splat to the nearest vertex of the layer that
  owns its class, then to the nearest triangle among that vertex's first 32. Per frame it builds the posed triangle
  frames (with the growth clamp) and the MLP per cage vertex: 6 log-stretch features, 2 occlusion-change features and
  the latent, with the dead zone, minus the rest output.
- `web/rig-worker.js` runs the rig off the UI thread, caches the MLP output of every frame, and depth-sorts the
  posed splats.
- `web/renderer.js` poses every splat in the vertex shader from per-face and per-vertex textures. That covers
  `pose_bound`, the stretch fade and `splat_apply_kernel`'s colour blend, opacity, scale and offset. The shader then
  evaluates SH in the triangle's canonical frame and projects like b2ctrain's rasteriser (EWA, 0.3 px blur, 1/255
  cutoff). It blends into a half-float target.

Not supported: dual binding (`--alt-binding`), motions that are not yet cages (bake them with b2crig first, e.g. as
`tools/render_final.py` writes `cage.b2ccage`), and the training cage (`train/<tag>/cage.b2ccage` holds every
training view's pose and is too large to be worth loading).
