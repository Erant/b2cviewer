// b2cviewer UI: picks a subject's splat (+ appearance MLP) and a clip's posed cage from serve.py's index (or dropped
// files), plays the cage frames, and drives the rig worker (binding, per-frame MLP, sorting) and the renderer.
import { Renderer } from './renderer.js';

const $ = id => document.getElementById(id);
const canvas = $('canvas');
let renderer;
try { renderer = new Renderer(canvas); } catch (e) { fail(e.message); throw e; }
const worker = new Worker('rig-worker.js', { type: 'module' });

const S = {
  index: null,
  loaded: { ply: null, app: null, cage: null, cameras: null },   // source keys currently in the worker
  clip: null,            // the clip entry of the loaded cage (cameras, WAN frames), or null for dropped files
  info: null,            // the worker's 'loaded' message
  names: [], nframes: 0, bodies: null, cams: null,
  frame: 0, shown: -1, playing: false, playT0: 0,
  ready: false, busy: false, needFrame: false, needSort: false, dirty: true,
  orbit: null, gen: 0, cached: 0, timing: null, loading: false,
};

// ---------- status ----------

function status(msg, err = false) { $('status').textContent = msg; $('status').className = err ? 'err' : ''; }
function fail(msg) { status(msg, true); }
function progress(f) {
  const p = $('progress');
  p.classList.toggle('on', f !== null);
  if (f !== null) p.firstElementChild.style.width = `${Math.round(f * 100)}%`;
}
const fmt = x => x.toLocaleString('en-US');

function updateInfo() {
  const I = S.info;
  if (!I) { $('info').textContent = ''; return; }
  const lines = [
    `${fmt(I.n)} splats · SH ${I.degree}` + (I.hasLabels ? ` · ${fmt(I.nFallback)} bound across layers` : ' · no seg_label (any layer)') +
      (I.nUnbound ? ` · ${fmt(I.nUnbound)} unbound` : ''),
    `cage ${fmt(I.nv)} verts · ${fmt(I.nf)} tris · ${I.nframes} frames`,
  ];
  if (I.app) {
    const a = I.app;
    lines.push(`MLP ${a.version} · dead zone ${a.dz.toFixed(2)}` + (a.nfeat > 6 ? ' · occlusion' : '') +
      (a.maxDp > 0 ? ` · offset ≤ ${(a.maxDp * 100).toFixed(0)} cm` : '') + (a.maxDo <= 0 ? ' · opacity may only fall' : '') +
      `\nMLP cache ${S.cached}/${I.nframes} frames`);
  } else lines.push(I.appError ? `MLP not loaded: ${I.appError}` : 'no appearance MLP for this splat');
  if (S.timing) lines.push(`last frame: rig ${S.timing.rig.toFixed(0)} ms · sort ${S.timing.sort.toFixed(0)} ms`);
  $('info').textContent = lines.join('\n');
}

// ---------- worker ----------

let pendingLoad = null;
worker.onmessage = ({ data: m }) => {
  if (m.type === 'progress') { status(`${m.stage}…`); progress(m.f); }
  else if (m.type === 'loaded') { const p = pendingLoad; pendingLoad = null; p?.resolve(m); }
  else if (m.type === 'frame' || m.type === 'sort') {
    S.busy = false;
    if (m.type === 'frame') {
      renderer.setFace(m.face, S.info.nf);
      if (m.vout) renderer.setVout(m.vout, S.info.nv);
      S.shown = m.fr;
      S.timing = { rig: m.msRig, sort: m.msSort };
      updateInfo();
    }
    renderer.setOrder(m.order);
    S.dirty = true;
  } else if (m.type === 'precomputed') {
    if (m.gen === S.gen) { S.cached = m.fr + 1; updateInfo(); }
  } else if (m.type === 'error') {
    console.error(m.stack);
    S.busy = false;
    if (pendingLoad) { const p = pendingLoad; pendingLoad = null; p.reject(new Error(m.error)); } else fail(m.error);
  }
};

function workerLoad(msg, transfer) {
  return new Promise((resolve, reject) => { pendingLoad = { resolve, reject }; worker.postMessage(msg, transfer); });
}

async function fetchBuf(url, label) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${label}: HTTP ${r.status}`);
  const total = +r.headers.get('Content-Length') || 0;
  if (!total || !r.body) return r.arrayBuffer();
  const out = new Uint8Array(total), reader = r.body.getReader();
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    out.set(value, got); got += value.length;
    status(`loading ${label} (${(got / 2 ** 20).toFixed(0)} / ${(total / 2 ** 20).toFixed(0)} MB)…`); progress(got / total);
  }
  return out.buffer;
}

// sources: { ply, app, cage, cameras } each { key, get: () => Promise<ArrayBuffer | object> } (app may be null);
// only the ones whose key changed are fetched and sent.
async function load(src, clip) {
  if (S.loading) return;
  S.loading = true;
  setInputs(false);
  worker.postMessage({ type: 'cancel', gen: S.gen });
  try {
    const msg = { type: 'load' }, transfer = [];
    const changed = k => src[k] !== undefined && (src[k]?.key ?? null) !== S.loaded[k];
    for (const [k, label] of [['ply', 'splat'], ['cage', 'cage'], ['app', 'MLP']]) {
      if (!changed(k)) continue;
      msg[k] = src[k] ? await src[k].get(label) : null;
      if (msg[k]) transfer.push(msg[k]);
    }
    const newCams = changed('cameras');
    const cams = newCams && src.cameras ? await src.cameras.get() : null;
    progress(0);
    const info = await workerLoad(msg, transfer);
    for (const k of ['ply', 'cage', 'app', 'cameras']) if (src[k] !== undefined) S.loaded[k] = src[k]?.key ?? null;
    if (info.partial) { status('Drop the remaining files (a .ply and a .b2ccage are needed).'); return; }
    if (newCams) S.cams = cams ? indexCameras(cams) : null;
    S.clip = clip;
    if (info.splatTex) renderer.setSplats(info.splatTex, info.n);
    if (info.sh) renderer.setSH(info.sh, info.n, info.degree);
    renderer.hasVout = false;
    S.info = info;
    S.names = info.names; S.nframes = info.nframes;
    if (info.bodies) S.bodies = smoothBodies(info.bodies, info.nframes);
    if (!S.orbit) resetOrbit();   // first load or another subject; otherwise keep the user's view
    S.frame = Math.min(S.frame, S.nframes - 1);
    $('slider').max = S.nframes - 1;
    S.ready = true; S.needFrame = true; S.dirty = true; S.shown = -1;
    S.cached = 0; S.gen++;
    if (info.app) worker.postMessage({ type: 'precompute', gen: S.gen });
    $('clipCam').disabled = !S.cams;
    $('showRef').disabled = !clip?.wan;
    applyRefPanel();
    updateInfo(); updateFrameLabel();
    status(info.appError ? `Loaded without the MLP: ${info.appError}` : `Loaded in ${(info.ms / 1000).toFixed(1)} s (binding + MLP set-up).`, !!info.appError);
  } catch (e) {
    console.error(e);
    fail(e.message);
  } finally {
    progress(null);
    S.loading = false;
    setInputs(true);
  }
}

function setInputs(on) { for (const id of ['subject', 'run', 'clip']) $(id).disabled = !on; }

// ---------- server index ----------

function subjectOf() { return S.index?.subjects.find(s => s.name === $('subject').value); }

function fillSelect(sel, items, value) {
  sel.innerHTML = '';
  for (const it of items) {
    const o = document.createElement('option');
    o.value = it.value; o.textContent = it.label; o.title = it.title || '';
    sel.appendChild(o);
  }
  if (value !== undefined && items.some(i => i.value === value)) sel.value = value;
}

function fillRuns(pref) {
  const s = subjectOf();
  const runs = [...s.runs].sort((a, b) => (!!b.app - !!a.app) || b.mtime - a.mtime);
  fillSelect($('run'), runs.map(r => ({ value: r.tag, label: `${r.tag}${r.app ? ' · MLP' : ''}`, title: `${r.ply} (${r.ply_mb} MB)${r.app ? '\n' + r.app + ' ' + r.app_version : ''}` })), pref);
}

function fillClips(pref) {
  const s = subjectOf(), run = s.runs.find(r => r.tag === $('run').value);
  const ok = c => !run?.app_nv || c.nv === run.app_nv;
  fillSelect($('clip'), s.clips.map(c => ({ value: c.name, label: `${c.name} · ${c.frames}f${ok(c) ? '' : ' · other cage'}`,
    title: `${c.cage}\n${c.nv} vertices${ok(c) ? '' : ' (the MLP was trained on a cage with ' + run.app_nv + ': it will be off)'}` })), pref);
  if (!pref || !s.clips.some(c => c.name === pref)) {
    const best = s.clips.find(c => ok(c) && c.name.startsWith('d_')) || s.clips.find(ok) || s.clips[0];
    if (best) $('clip').value = best.name;
  }
}

// A/B partners: CLIP and CLIP_fix (e.g. a retarget fix baked next to the original clip). Swapping keeps the
// splat, the frame and the view; only the cage (and its cameras) reload.
function partnerOf(name) {
  const s = subjectOf(); if (!s || !name) return null;
  const other = name.endsWith('_fix') ? name.slice(0, -4) : `${name}_fix`;
  return s.clips.some(c => c.name === other) ? other : null;
}
function updateAB() {
  const name = $('clip').value, other = partnerOf(name), badge = $('abBadge');
  $('abSwap').disabled = !other;
  badge.classList.toggle('on', !!other);
  badge.classList.toggle('fix', name.endsWith('_fix'));
  badge.textContent = other ? `${name.endsWith('_fix') ? 'FIXED' : 'ORIGINAL'} · ${name}` : '';
}
function swapAB() {
  const other = partnerOf($('clip').value);
  if (!other || S.loading) return;
  $('clip').value = other;
  loadSelection();
}

function loadSelection() {
  const s = subjectOf(); if (!s) return;
  const run = s.runs.find(r => r.tag === $('run').value), clip = s.clips.find(c => c.name === $('clip').value);
  if (!run || !clip) return;
  const url = p => `/data/${p.split('/').map(encodeURIComponent).join('/')}`;
  const buf = p => ({ key: p, get: label => fetchBuf(url(p), label) });
  const q = new URLSearchParams({ subject: s.name, run: run.tag, clip: clip.name });
  history.replaceState(null, '', `?${q}`);
  updateAB();
  load({
    ply: buf(run.ply), app: run.app ? buf(run.app) : null, cage: buf(clip.cage),
    cameras: clip.cameras ? { key: clip.cameras, get: () => fetch(url(clip.cameras)).then(r => r.json()) } : null,
  }, { ...clip, wanUrl: clip.wan ? url(clip.wan) : null });
}

async function init() {
  try {
    S.index = await (await fetch('/api/index')).json();
  } catch {
    status('No index (open this page through serve.py to browse b2crig/work). Drop .ply / .app / .b2ccage files to view them.');
    return;
  }
  const subs = S.index.subjects;
  if (!subs.length) { status(`No subjects with trained splats and clips under ${S.index.root}.`); return; }
  const q = new URLSearchParams(location.search);
  const dflt = q.get('subject') || (subs.find(s => s.runs.some(r => r.app)) || subs[0]).name;
  fillSelect($('subject'), subs.map(s => ({ value: s.name, label: `${s.name} (${s.runs.length} trained, ${s.clips.length} clips)` })), dflt);
  fillRuns(q.get('run'));
  fillClips(q.get('clip'));
  status(`${subs.length} subjects under ${S.index.root}`);
  loadSelection();
}

$('subject').onchange = () => { fillRuns(); fillClips(); S.orbit = null; loadSelection(); };
$('run').onchange = () => { fillClips($('clip').value); loadSelection(); };
$('clip').onchange = () => loadSelection();
$('abSwap').onclick = swapAB;

// ---------- dropped files ----------

const view = $('view');
view.addEventListener('dragover', e => { e.preventDefault(); $('drop').classList.add('on'); });
view.addEventListener('dragleave', () => $('drop').classList.remove('on'));
view.addEventListener('drop', e => {
  e.preventDefault(); $('drop').classList.remove('on');
  const src = {};
  for (const f of e.dataTransfer.files) {
    const key = `local:${f.name}:${f.size}:${f.lastModified}`, get = () => f.arrayBuffer();
    if (/\.ply$/i.test(f.name)) src.ply = { key, get };
    else if (/\.app$/i.test(f.name)) src.app = { key, get };
    else if (/\.b2ccage$/i.test(f.name)) src.cage = { key, get };
    else if (/\.json$/i.test(f.name)) src.cameras = { key, get: async () => JSON.parse(await f.text()) };
  }
  if (!Object.keys(src).length) { fail('Drop .ply, .app, .b2ccage (and optionally a cameras.json) files.'); return; }
  if (src.cage && !src.cameras) src.cameras = null;
  history.replaceState(null, '', location.pathname);
  if (src.cage) { $('abBadge').classList.remove('on'); $('abSwap').disabled = true; }   // a dropped cage has no partner
  load(src, src.cage ? null : S.clip);
});

// ---------- cameras ----------

function smoothBodies(b, n) {
  // follow target: the body centroid, smoothed over +-0.5 s so the camera glides instead of shaking with the arms
  const out = new Float32Array(n * 3), k = 8;
  for (let f = 0; f < n; f++) {
    let s = [0, 0, 0], c = 0;
    for (let g = Math.max(0, f - k); g <= Math.min(n - 1, f + k); g++, c++) for (let j = 0; j < 3; j++) s[j] += b[g * 6 + j];
    for (let j = 0; j < 3; j++) out[f * 3 + j] = s[j] / c;
  }
  out.size = [b[3], b[4], b[5]];
  return out;
}

function resetOrbit() {
  const h = S.bodies ? Math.max(S.bodies.size[1], 0.5) : 1.8;
  const fov = +$('fov').value * Math.PI / 180;
  S.orbit = { az: 0, el: 5, dist: (0.62 * h) / Math.tan(fov / 2), pan: [0, 0, 0], target: bodyTarget(0) };
  S.needSort = true; S.dirty = true;
}

function bodyTarget(fr) {
  if (!S.bodies) return [0, 1, 0];
  const f = Math.min(Math.max(fr, 0), S.nframes - 1);
  return [S.bodies[f * 3], S.bodies[f * 3 + 1], S.bodies[f * 3 + 2]];
}

function indexCameras(j) {
  const byName = new Map(j.cameras.map(c => [c.name, c]));
  return { W: j.width, H: j.height, list: j.cameras, byName };
}

// OpenGL camera-to-world (columns right, up, back) + position -> the renderer's OpenCV world-to-camera
function glToCam(Rgl, pos) {
  const R = [Rgl[0][0], Rgl[1][0], Rgl[2][0], -Rgl[0][1], -Rgl[1][1], -Rgl[2][1], -Rgl[0][2], -Rgl[1][2], -Rgl[2][2]];
  const t = [0, 1, 2].map(r => -(R[r * 3] * pos[0] + R[r * 3 + 1] * pos[1] + R[r * 3 + 2] * pos[2]));
  return { R, t, pos };
}

function camera(fr, W, H) {
  if ($('clipCam').checked && S.cams) {
    const c = S.cams.byName.get(S.names[fr]) || S.cams.list[fr];
    if (c) {
      const s = Math.min(W / S.cams.W, H / S.cams.H);
      return { ...glToCam(c.rotation, c.position), fx: c.fx * s, fy: c.fy * s,
               cx: W / 2 + (c.cx - S.cams.W / 2) * s, cy: H / 2 + (c.cy - S.cams.H / 2) * s };
    }
  }
  const o = S.orbit || { az: 0, el: 5, dist: 3, pan: [0, 0, 0], target: [0, 1, 0] };
  const base = $('follow').checked ? bodyTarget(fr) : o.target;
  const tgt = base.map((x, j) => x + o.pan[j]);
  const az = o.az * Math.PI / 180, el = o.el * Math.PI / 180;
  const back = [Math.cos(el) * Math.sin(az), Math.sin(el), Math.cos(el) * Math.cos(az)];
  const pos = tgt.map((x, j) => x + o.dist * back[j]);
  const right = norm(cross([0, 1, 0], back)), up = cross(back, right);
  const Rgl = [[right[0], up[0], back[0]], [right[1], up[1], back[1]], [right[2], up[2], back[2]]];
  const f = (H / 2) / Math.tan(+$('fov').value * Math.PI / 360);
  return { ...glToCam(Rgl, pos), fx: f, fy: f, cx: W / 2, cy: H / 2, right, up };
}
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = a => { const l = Math.hypot(...a) || 1; return a.map(x => x / l); };

function sortView(fr) {
  const c = camera(fr, canvas.width, canvas.height);
  return [c.R[6], c.R[7], c.R[8], c.t[2]];
}

// orbit controls
let drag = null;
canvas.addEventListener('pointerdown', e => {
  canvas.setPointerCapture(e.pointerId);
  drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey || e.button === 1 };
});
canvas.addEventListener('pointermove', e => {
  if (!drag || !S.orbit) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  drag.x = e.clientX; drag.y = e.clientY;
  if ($('clipCam').checked && S.cams) leaveClipCam();
  const o = S.orbit;
  if (drag.pan) {
    const c = camera(S.shown < 0 ? 0 : S.shown, canvas.width, canvas.height);
    const k = o.dist * 2 * Math.tan(+$('fov').value * Math.PI / 360) / canvas.clientHeight;
    for (let j = 0; j < 3; j++) o.pan[j] += (-dx * c.right[j] + dy * c.up[j]) * k;
  } else {
    o.az -= dx * 0.3;
    o.el = Math.max(-89, Math.min(89, o.el + dy * 0.3));
  }
  S.needSort = true; S.dirty = true;
});
canvas.addEventListener('pointerup', () => { drag = null; });
canvas.addEventListener('contextmenu', e => e.preventDefault());
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  if (!S.orbit) return;
  if ($('clipCam').checked && S.cams) leaveClipCam();
  S.orbit.dist = Math.max(0.1, S.orbit.dist * Math.exp(e.deltaY * 0.001));
  S.needSort = true; S.dirty = true;
}, { passive: false });
canvas.addEventListener('dblclick', () => { $('clipCam').checked = false; resetOrbit(); });

function leaveClipCam() {
  // continue orbiting from where the clip camera was (azimuth / elevation / distance around the body)
  const fr = S.shown < 0 ? 0 : S.shown, c = S.cams.byName.get(S.names[fr]) || S.cams.list[fr];
  $('clipCam').checked = false;
  if (!c || !S.orbit) return;
  const tgt = $('follow').checked ? bodyTarget(fr) : S.orbit.target;
  const d = c.position.map((x, j) => x - tgt[j]), r = Math.hypot(...d);
  S.orbit.az = Math.atan2(d[0], d[2]) * 180 / Math.PI;
  S.orbit.el = Math.asin(d[1] / r) * 180 / Math.PI;
  S.orbit.dist = r; S.orbit.pan = [0, 0, 0];
}

$('follow').onchange = () => {
  if (!S.orbit) return;
  const fr = S.shown < 0 ? 0 : S.shown;
  if (!$('follow').checked) S.orbit.target = bodyTarget(fr).map((x, j) => x + S.orbit.pan[j]);
  S.orbit.pan = [0, 0, 0];
  S.needSort = true; S.dirty = true;
};
for (const id of ['clipCam', 'fov']) $(id).addEventListener('input', () => { S.needSort = true; S.dirty = true; });

// ---------- playback ----------

function fps() { return Math.max(1, +$('fps').value || 16); }
function setFrame(f) {
  if (!S.nframes) return;
  S.frame = ((f % S.nframes) + S.nframes) % S.nframes;
  S.needFrame = true;
  updateFrameLabel();
}
function updateFrameLabel() {
  $('slider').value = S.frame;
  $('frameLabel').textContent = S.nframes ? `${S.frame + 1} / ${S.nframes} · ${S.names[S.frame] ?? ''}` : '–';
}
function setPlaying(p) {
  S.playing = p && S.nframes > 1;
  S.playT0 = performance.now() - S.frame / fps() * 1000;
  $('play').textContent = S.playing ? '❚❚' : '▶';
}
$('play').onclick = () => setPlaying(!S.playing);
$('slider').oninput = () => { setPlaying(false); setFrame(+$('slider').value); };
$('fps').onchange = () => setPlaying(S.playing);

// ---------- options ----------

for (const id of ['useApp', 'maxGrowth']) $(id).addEventListener('change', () => { S.needFrame = true; S.dirty = true; });
for (const id of ['debugApp', 'fade', 'fadeStart', 'fadeEnd', 'degree', 'bg']) $(id).addEventListener('input', () => { S.dirty = true; });
$('showRef').onchange = applyRefPanel;
$('collapse').onclick = () => { $('panel').classList.toggle('collapsed'); $('collapse').textContent = $('panel').classList.contains('collapsed') ? '▸' : '▾'; };

function applyRefPanel() {
  const on = $('showRef').checked && !!S.clip?.wanUrl;
  $('ref').classList.toggle('on', on);
  if (on && !$('clipCam').checked && S.cams) { $('clipCam').checked = true; S.needSort = true; }
  S.refName = null; S.dirty = true;
}

function renderOpts() {
  const a = S.info?.app, bg = $('bg').value;
  return {
    app: $('useApp').checked ? ($('debugApp').checked ? 2 : 1) : 0,
    appLim: a ? [a.maxDo, a.maxDs, a.maxDp] : [0, 0, 0],
    fade: $('fade').checked ? [+$('fadeStart').value, +$('fadeEnd').value] : [0, 0],
    degree: +$('degree').value,
    bg: [1, 3, 5].map(i => parseInt(bg.slice(i, i + 2), 16) / 255),
  };
}

// ---------- save the view ----------

// A PNG of the canvas with a tEXt chunk "b2cviewer" holding JSON: what is loaded, the frame shown, the render options
// and the camera as a b2crig cameras.json entry (rotation = camera -> world with columns right, up, back), so
// b2crig's tools/viewer_shot.py can render exactly this view with b2ctrain.
const CRC = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; }
  return t;
})();
function crc32(bytes) {
  let c = 0xffffffff;
  for (const b of bytes) c = CRC[(c ^ b) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
function pngWithText(png, key, text) {
  const body = new TextEncoder().encode(`${key}\0${text}`);   // tEXt is Latin-1; JSON.stringify output here is ASCII
  const chunk = new Uint8Array(12 + body.length), dv = new DataView(chunk.buffer);
  dv.setUint32(0, body.length);
  chunk.set([0x74, 0x45, 0x58, 0x74], 4);   // "tEXt"
  chunk.set(body, 8);
  dv.setUint32(8 + body.length, crc32(chunk.subarray(4, 8 + body.length)));
  const ihdrEnd = 8 + 25;   // signature + IHDR (13-byte payload + length, type, crc)
  const out = new Uint8Array(png.length + chunk.length);
  out.set(png.subarray(0, ihdrEnd)); out.set(chunk, ihdrEnd); out.set(png.subarray(ihdrEnd), ihdrEnd + chunk.length);
  return out;
}

async function saveShot() {
  if (!S.ready || S.shown < 0) { status('Nothing to save yet.', true); return; }
  const W = canvas.width, H = canvas.height, fr = S.shown, cam = camera(fr, W, H);
  renderer.render(cam, renderOpts());   // the drawing buffer is preserved: read what this call drew
  const blob = await new Promise(r => canvas.toBlob(r, 'image/png'));
  const R = cam.R, sg = [1, -1, -1];   // OpenCV world -> camera (row-major) to OpenGL camera -> world columns
  const rotation = [0, 1, 2].map(i => [0, 1, 2].map(j => R[j * 3 + i] * sg[j]));
  const q = new URLSearchParams(location.search);
  const meta = {
    subject: q.get('subject'), run: q.get('run'), clip: q.get('clip'),
    ply: S.loaded.ply, cage: S.loaded.cage, app: S.loaded.app,
    frame: fr, frame_name: S.names[fr] ?? null, clip_camera: $('clipCam').checked && !!S.cams,
    width: W, height: H,
    camera: { name: S.names[fr] ?? String(fr), fx: cam.fx, fy: cam.fy, cx: cam.cx, cy: cam.cy, rotation, position: cam.pos },
    orbit: S.orbit, follow: $('follow').checked, fov_deg: +$('fov').value,
    options: { ...renderOpts(), maxGrowth: +$('maxGrowth').value || 0, useApp: $('useApp').checked && !!S.info?.app },
    url: location.href, saved: new Date().toISOString(),
  };
  const png = pngWithText(new Uint8Array(await blob.arrayBuffer()), 'b2cviewer', JSON.stringify(meta));
  const d = new Date(), p2 = x => String(x).padStart(2, '0');   // local time
  const stamp = `${d.getFullYear()}${p2(d.getMonth() + 1)}${p2(d.getDate())}-${p2(d.getHours())}${p2(d.getMinutes())}${p2(d.getSeconds())}`;
  const safe = s => String(s ?? 'local').replace(/[^\w.-]+/g, '_');
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([png], { type: 'image/png' }));
  a.download = `b2cview_${safe(meta.subject)}_${safe(meta.run)}_${safe(meta.clip)}_f${String(fr).padStart(4, '0')}_${stamp}.png`;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 10000);
  status(`Saved ${a.download}`);
}
$('shot').onclick = saveShot;

window.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' && e.target.type !== 'checkbox' && e.target.type !== 'range' || e.target.tagName === 'SELECT') return;
  const toggle = id => { const el = $(id); if (el.disabled) return; el.checked = !el.checked; el.dispatchEvent(new Event('change')); el.dispatchEvent(new Event('input')); };
  if (e.key === ' ') { e.preventDefault(); setPlaying(!S.playing); }
  else if (e.key === 'ArrowRight') { setPlaying(false); setFrame(S.frame + 1); }
  else if (e.key === 'ArrowLeft') { setPlaying(false); setFrame(S.frame - 1); }
  else if (e.key === 'a') toggle('useApp');
  else if (e.key === 'd') toggle('debugApp');
  else if (e.key === 'c') toggle('clipCam');
  else if (e.key === 'f') toggle('follow');
  else if (e.key === 'w') toggle('showRef');
  else if (e.key === 's') saveShot();
  else if (e.key === 'b') swapAB();
});

// ---------- loop ----------

new ResizeObserver(() => {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.round(canvas.clientWidth * dpr));
  canvas.height = Math.max(1, Math.round(canvas.clientHeight * dpr));
  S.needSort = true; S.dirty = true;
}).observe(canvas);

function pump() {
  if (!S.ready || S.busy || S.loading) return;
  if (S.needFrame) {
    S.needFrame = false; S.needSort = false; S.busy = true;
    worker.postMessage({ type: 'frame', fr: S.frame, view: sortView(S.frame),
                         opts: { maxGrowth: +$('maxGrowth').value || 0, useApp: $('useApp').checked && !!S.info.app } });
  } else if (S.needSort && S.shown >= 0) {
    S.needSort = false; S.busy = true;
    worker.postMessage({ type: 'sort', view: sortView(S.shown) });
  }
}

function tick(t) {
  if (S.playing && S.nframes) {
    const f = Math.floor((t - S.playT0) / 1000 * fps()) % S.nframes;
    if (f !== S.frame) setFrame(f);
  }
  pump();
  if (S.dirty && S.shown >= 0) {
    S.dirty = false;
    renderer.render(camera(S.shown, canvas.width, canvas.height), renderOpts());
    if ($('ref').classList.contains('on')) {
      const name = S.names[S.shown];
      if (name !== S.refName) { S.refName = name; $('refImg').src = `${S.clip.wanUrl}/${encodeURIComponent(name)}.png`; }
    }
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
init();

// for scripted checks (tools/compare.py): the state and a way to wait for a settled frame
window.b2cviewer = { S, setFrame, setPlaying, renderer,
  settled: () => S.ready && !S.loading && !S.busy && !S.needFrame && !S.needSort && S.shown === S.frame && !S.dirty };
