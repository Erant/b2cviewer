// b2cviewer UI: picks a rigged subject file and one of its clip files from serve.py's index (or dropped .glb files),
// plays the clip, and drives the rig worker (posing, sorting) and the renderer.
import { Renderer } from './renderer.js';

const $ = id => document.getElementById(id);
const canvas = $('canvas');
let renderer;
try { renderer = new Renderer(canvas); } catch (e) { fail(e.message); throw e; }
const worker = new Worker('rig-worker.js', { type: 'module' });

const S = {
  index: null,
  dropped: null,         // { subject: {key, get}, clips: [{name, key, get}] } from dropped files
  loaded: { subject: null, clip: null },   // source keys currently in the worker
  info: null,            // the worker's 'loaded' message
  nframes: 0, bodies: null, W: null, Winv: null,
  frame: 0, shown: -1, playing: false, playT0: 0,
  ready: false, busy: false, needFrame: false, needSort: false, dirty: true,
  orbit: null, timing: null, loading: false,
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
    `${fmt(I.n)} splats · SH ${I.degree}` + (I.nUnbound ? ` · ${fmt(I.nUnbound)} unbound` : ''),
    `cage ${fmt(I.nv)} verts · ${fmt(I.nf)} tris · ${I.layers.join(', ')}`,
    I.clip ? `clip ${I.clip.name} · ${I.nframes} frames at ${I.clip.fps} fps` + (I.clip.residual ? '' : ' · no residual (plain skinning)')
           : 'no clip: the rest pose',
  ];
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
      S.shown = m.fr;
      S.timing = { rig: m.msRig, sort: m.msSort };
      updateInfo();
    }
    renderer.setOrder(m.order);
    S.dirty = true;
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

// subject, clip: { key, get: label => Promise<ArrayBuffer> }; clip null = the rest pose. Only changed files are sent.
async function load(subject, clip) {
  if (S.loading) return;
  S.loading = true;
  setInputs(false);
  try {
    const msg = { type: 'load' }, transfer = [];
    const newSubject = subject.key !== S.loaded.subject;
    if (newSubject) { msg.subject = await subject.get('subject'); transfer.push(msg.subject); }
    if (newSubject || (clip?.key ?? null) !== S.loaded.clip) {
      msg.clip = clip ? await clip.get('clip') : null;
      if (msg.clip) transfer.push(msg.clip);
    }
    progress(0);
    const info = await workerLoad(msg, transfer);
    S.loaded = { subject: subject.key, clip: clip?.key ?? null };
    if (info.splatTex) renderer.setSplats(info.splatTex, info.n);
    if (info.sh) renderer.setSH(info.sh, info.n, info.degree);
    if (newSubject) {
      setW(info.W);
      const r = info.render;   // b2ctrain's render settings from B2CRIG_rig
      $('maxGrowth').value = r.maxGrowth ?? 1.15;
      $('fade').checked = r.fadeEnd > r.fadeStart && r.fadeStart > 0;
      if ($('fade').checked) { $('fadeStart').value = r.fadeStart; $('fadeEnd').value = r.fadeEnd; }
      S.orbit = null;
    }
    S.info = info;
    S.nframes = info.nframes;
    S.bodies = smoothBodies(info.bodies, info.nframes);
    if (!S.orbit) resetOrbit();   // another subject; otherwise keep the user's view
    S.frame = Math.min(S.frame, S.nframes - 1);
    if (info.clip) $('fps').value = info.clip.fps;
    $('slider').max = S.nframes - 1;
    $('residual').disabled = !info.clip?.residual;
    S.ready = true; S.needFrame = true; S.dirty = true; S.shown = -1;
    updateInfo(); updateFrameLabel();
    status(`Loaded in ${(info.ms / 1000).toFixed(1)} s.`);
  } catch (e) {
    console.error(e);
    fail(e.message);
    if (e.message.includes('clip')) S.loaded.clip = undefined;   // the worker kept its previous clip: reload next time
  } finally {
    progress(null);
    S.loading = false;
    setInputs(true);
  }
}

function setInputs(on) { for (const id of ['subject', 'clip']) $(id).disabled = !on; }

// ---------- subjects and clips ----------

const url = p => `/data/${p.split('/').map(encodeURIComponent).join('/')}`;
const served = p => ({ key: p, get: label => fetchBuf(url(p), label) });

// { name, subject: source, clips: [{ name, label, source }] } for the selected entry
function entryOf(name = $('subject').value) {
  if (name === '(dropped)' && S.dropped) return { name, subject: S.dropped.subject, clips: S.dropped.clips.map(c => ({ name: c.name, label: c.name, source: c })) };
  const s = S.index?.subjects.find(x => x.name === name);
  return s && s.rigged && { name: s.name, path: s.path, subject: served(s.path),
    clips: s.clips.map(c => ({ name: c.name, label: `${c.name} · ${c.frames}f`, path: c.path, source: served(c.path) })) };
}

function fillSelect(sel, items, value) {
  sel.innerHTML = '';
  for (const it of items) {
    const o = document.createElement('option');
    o.value = it.value; o.textContent = it.label; o.title = it.title || ''; o.disabled = !!it.disabled;
    sel.appendChild(o);
  }
  if (value !== undefined && items.some(i => i.value === value && !i.disabled)) sel.value = value;
}

function fillSubjects(pref) {
  const items = (S.index?.subjects || []).map(s => ({ value: s.name, disabled: !s.rigged, title: s.path,
    label: s.rigged ? `${s.name} (${s.clips.length} clips)` : `${s.name} (not rigged yet)` }));
  if (S.dropped) items.unshift({ value: '(dropped)', label: `dropped: ${S.dropped.subject.name}` });
  fillSelect($('subject'), items, pref);
}

function fillClips(pref) {
  const e = entryOf();
  fillSelect($('clip'), [{ value: '', label: 'rest pose' }, ...(e?.clips || []).map(c => ({ value: c.name, label: c.label, title: c.path || '' }))], pref);
  if (!pref && e?.clips.length) $('clip').value = e.clips[0].name;
}

// A/B partners: CLIP and CLIP_fix (e.g. a retarget fix exported next to the original clip). Swapping keeps the
// frame and the view; only the clip reloads.
function partnerOf(name) {
  const e = entryOf(); if (!e || !name) return null;
  const other = name.endsWith('_fix') ? name.slice(0, -4) : `${name}_fix`;
  return e.clips.some(c => c.name === other) ? other : null;
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
  const e = entryOf(); if (!e) return;
  const clip = e.clips.find(c => c.name === $('clip').value);
  if (e.name !== '(dropped)') history.replaceState(null, '', `?${new URLSearchParams({ subject: e.name, clip: clip?.name ?? '' })}`);
  else history.replaceState(null, '', location.pathname);
  updateAB();
  load(e.subject, clip?.source ?? null);
}

async function init() {
  try {
    S.index = await (await fetch('/api/index')).json();
  } catch {
    status('No index (open this page through serve.py to browse b2crig/work). Drop a subject .glb and its .clip.glb files to view them.');
    return;
  }
  const subs = S.index.subjects, rigged = subs.filter(s => s.rigged);
  if (!rigged.length) { status(`No rigged subject files under ${S.index.root} (<subject>/gltf/*.glb).`); fillSubjects(); return; }
  const q = new URLSearchParams(location.search);
  fillSubjects(q.get('subject') || rigged[0].name);
  fillClips(q.get('clip') ?? undefined);
  status(`${rigged.length} rigged subjects under ${S.index.root}`);
  loadSelection();
}

$('subject').onchange = () => { fillClips(); loadSelection(); };
$('clip').onchange = () => loadSelection();
$('abSwap').onclick = swapAB;

// ---------- dropped files ----------

const view = $('view');
view.addEventListener('dragover', e => { e.preventDefault(); $('drop').classList.add('on'); });
view.addEventListener('dragleave', () => $('drop').classList.remove('on'));
view.addEventListener('drop', e => {
  e.preventDefault(); $('drop').classList.remove('on');
  let subject = null;
  const clips = [];
  for (const f of e.dataTransfer.files) {
    const src = { key: `local:${f.name}:${f.size}:${f.lastModified}`, get: () => f.arrayBuffer() };
    if (/\.clip\.glb$/i.test(f.name)) clips.push({ ...src, name: f.name.replace(/\.clip\.glb$/i, '') });
    else if (/\.glb$/i.test(f.name)) subject = { ...src, name: f.name };
  }
  if (!subject && S.dropped && clips.length) subject = S.dropped.subject;   // more clips for the dropped subject
  if (!subject) { fail('Drop a rigged subject .glb (and its .clip.glb files).'); return; }
  S.dropped = { subject, clips: clips.sort((a, b) => a.name < b.name ? -1 : 1) };
  fillSubjects('(dropped)'); fillClips();
  loadSelection();
});

// ---------- cameras ----------

// W maps the b2crunner frame (the posed data) to glTF world (+Y up, subject facing +Z). The orbit lives in glTF
// world; the renderer gets it in the b2crunner frame.
function setW(W) {   // column-major 4x4, rigid
  const R = [0, 1, 2].map(r => [0, 1, 2].map(c => W[c * 4 + r])), t = [W[12], W[13], W[14]];
  S.W = { R, t };
  const Ri = [0, 1, 2].map(r => [0, 1, 2].map(c => R[c][r]));
  S.Winv = { R: Ri, t: Ri.map(row => -(row[0] * t[0] + row[1] * t[1] + row[2] * t[2])) };
}
const apply = (T, p) => T.R.map((row, r) => row[0] * p[0] + row[1] * p[1] + row[2] * p[2] + T.t[r]);

function smoothBodies(b, n) {
  // follow target: the body centroid in glTF world, smoothed over +-0.5 s so the camera glides instead of shaking
  const out = new Float32Array(n * 3), k = 8;
  for (let f = 0; f < n; f++) {
    let s = [0, 0, 0], c = 0;
    for (let g = Math.max(0, f - k); g <= Math.min(n - 1, f + k); g++, c++) for (let j = 0; j < 3; j++) s[j] += b[g * 6 + j];
    out.set(apply(S.W, s.map(x => x / c)), f * 3);
  }
  out.size = [0, 1, 2].map(r => Math.abs(S.W.R[r][0]) * b[3] + Math.abs(S.W.R[r][1]) * b[4] + Math.abs(S.W.R[r][2]) * b[5]);
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

// OpenGL camera-to-world (columns right, up, back) + position -> the renderer's OpenCV world-to-camera
function glToCam(Rgl, pos) {
  const R = [Rgl[0][0], Rgl[1][0], Rgl[2][0], -Rgl[0][1], -Rgl[1][1], -Rgl[2][1], -Rgl[0][2], -Rgl[1][2], -Rgl[2][2]];
  const t = [0, 1, 2].map(r => -(R[r * 3] * pos[0] + R[r * 3 + 1] * pos[1] + R[r * 3 + 2] * pos[2]));
  return { R, t, pos };
}

// The orbit camera in the b2crunner frame (right / up stay in glTF world for panning).
function camera(fr, W, H) {
  const o = S.orbit || { az: 0, el: 5, dist: 3, pan: [0, 0, 0], target: [0, 1, 0] };
  const base = $('follow').checked ? bodyTarget(fr) : o.target;
  const tgt = base.map((x, j) => x + o.pan[j]);
  const az = o.az * Math.PI / 180, el = o.el * Math.PI / 180;
  const back = [Math.cos(el) * Math.sin(az), Math.sin(el), Math.cos(el) * Math.cos(az)];
  const pos = tgt.map((x, j) => x + o.dist * back[j]);
  const right = norm(cross([0, 1, 0], back)), up = cross(back, right);
  const Wi = S.Winv || { R: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], t: [0, 0, 0] }, rot = v => apply({ R: Wi.R, t: [0, 0, 0] }, v);
  const [r, u, b] = [rot(right), rot(up), rot(back)];
  const Rgl = [[r[0], u[0], b[0]], [r[1], u[1], b[1]], [r[2], u[2], b[2]]];
  const f = (H / 2) / Math.tan(+$('fov').value * Math.PI / 360);
  return { ...glToCam(Rgl, apply(Wi, pos)), fx: f, fy: f, cx: W / 2, cy: H / 2, right, up };
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
  S.orbit.dist = Math.max(0.1, S.orbit.dist * Math.exp(e.deltaY * 0.001));
  S.needSort = true; S.dirty = true;
}, { passive: false });
canvas.addEventListener('dblclick', () => resetOrbit());

$('follow').onchange = () => {
  if (!S.orbit) return;
  const fr = S.shown < 0 ? 0 : S.shown;
  if (!$('follow').checked) S.orbit.target = bodyTarget(fr).map((x, j) => x + S.orbit.pan[j]);
  S.orbit.pan = [0, 0, 0];
  S.needSort = true; S.dirty = true;
};
$('fov').addEventListener('input', () => { S.needSort = true; S.dirty = true; });

// ---------- playback ----------

function fps() { return Math.max(1, +$('fps').value || 30); }
function setFrame(f) {
  if (!S.nframes) return;
  S.frame = ((f % S.nframes) + S.nframes) % S.nframes;
  S.needFrame = true;
  updateFrameLabel();
}
function updateFrameLabel() {
  $('slider').value = S.frame;
  $('frameLabel').textContent = S.nframes ? `${S.frame + 1} / ${S.nframes}` : '–';
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

for (const id of ['residual', 'maxGrowth']) $(id).addEventListener('change', () => { S.needFrame = true; S.dirty = true; });
for (const id of ['fade', 'fadeStart', 'fadeEnd', 'degree', 'bg']) $(id).addEventListener('input', () => { S.dirty = true; });
$('collapse').onclick = () => { $('panel').classList.toggle('collapsed'); $('collapse').textContent = $('panel').classList.contains('collapsed') ? '▸' : '▾'; };

function frameOpts() { return { maxGrowth: +$('maxGrowth').value || 0, residual: $('residual').checked }; }

function renderOpts() {
  const bg = $('bg').value;
  return {
    fade: $('fade').checked ? [+$('fadeStart').value, +$('fadeEnd').value] : [0, 0],
    degree: +$('degree').value,
    bg: [1, 3, 5].map(i => parseInt(bg.slice(i, i + 2), 16) / 255),
  };
}

// ---------- save the view ----------

// A PNG of the canvas with a tEXt chunk "b2cviewer" holding JSON: the files loaded (paths under serve.py's root), the
// frame shown, the render options and the camera as a b2crig cameras.json entry in the b2crunner frame (rotation =
// camera -> world with columns right, up, back), so b2crig's tools/viewer_shot.py can render exactly this view.
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
  const q = new URLSearchParams(location.search), e = entryOf();
  const meta = {
    subject: q.get('subject'), clip: S.info.clip?.name ?? null,
    subject_file: e?.path ?? S.loaded.subject, clip_file: e?.clips.find(c => c.name === $('clip').value)?.path ?? S.loaded.clip,
    frame: fr, width: W, height: H,
    camera: { name: String(fr), fx: cam.fx, fy: cam.fy, cx: cam.cx, cy: cam.cy, rotation, position: cam.pos },
    orbit: S.orbit, follow: $('follow').checked, fov_deg: +$('fov').value,
    options: { ...renderOpts(), ...frameOpts() },
    url: location.href, saved: new Date().toISOString(),
  };
  const png = pngWithText(new Uint8Array(await blob.arrayBuffer()), 'b2cviewer', JSON.stringify(meta));
  const d = new Date(), p2 = x => String(x).padStart(2, '0');   // local time
  const stamp = `${d.getFullYear()}${p2(d.getMonth() + 1)}${p2(d.getDate())}-${p2(d.getHours())}${p2(d.getMinutes())}${p2(d.getSeconds())}`;
  const safe = s => String(s ?? 'local').replace(/[^\w.-]+/g, '_');
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([png], { type: 'image/png' }));
  a.download = `b2cview_${safe(meta.subject)}_${safe(meta.clip ?? 'rest')}_f${String(fr).padStart(4, '0')}_${stamp}.png`;
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
  else if (e.key === 'r') toggle('residual');
  else if (e.key === 'f') toggle('follow');
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
    worker.postMessage({ type: 'frame', fr: S.frame, view: sortView(S.frame), opts: frameOpts() });
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
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
init();

// for scripted checks: the state and a way to wait for a settled frame
window.b2cviewer = { S, setFrame, setPlaying, renderer,
  settled: () => S.ready && !S.loading && !S.busy && !S.needFrame && !S.needSort && S.shown === S.frame && !S.dirty };
