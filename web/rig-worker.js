// Worker: parses the subject files, binds the splat to the cage, and per request computes a frame's face / vertex
// textures and a back-to-front splat order for the current camera.
import { parsePly, parseCage, parseApp } from './formats.js';
import { Rig } from './rig.js';
import { parseGlb } from './gltf.js';

let splat = null, cage = null, app = null, rig = null, minConf = 0.5;
let glb = null, binding = null;   // a loaded .glb (gltf.js) and b2ctrain's binding from it
let last = null;   // { fr, key, centres }: the centres of the frame last computed

const post = (msg, transfer = []) => self.postMessage(msg, transfer);
const progress = (stage, f) => post({ type: 'progress', stage, f });

function sort(centres, view) {
  // depth along the camera's forward axis (OpenCV: +z ahead); back to front with a 16-bit counting sort
  const n = centres.length / 3, [r0, r1, r2, tz] = view;
  const depth = new Float32Array(n);
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < n; i++) {
    const d = r0 * centres[i * 3] + r1 * centres[i * 3 + 1] + r2 * centres[i * 3 + 2] + tz;
    depth[i] = d;
    if (d < lo) lo = d;
    if (d > hi) hi = d;
  }
  const scale = hi > lo ? 65535 / (hi - lo) : 0;
  const key = new Uint16Array(n), count = new Uint32Array(65537);
  for (let i = 0; i < n; i++) { const k = 65535 - Math.floor((depth[i] - lo) * scale); key[i] = k; count[k + 1]++; }
  for (let k = 0; k < 65536; k++) count[k + 1] += count[k];
  const order = new Uint32Array(n);
  for (let i = 0; i < n; i++) order[count[key[i]]++] = i;
  return order;
}

function optsKey(o) { return `${o.maxGrowth}|${o.useApp}`; }

self.onmessage = async ({ data: m }) => {
  try {
    if (m.type === 'load') {
      const t0 = performance.now();
      let newSplat = false, newCage = false;
      if (m.ply) { progress('parsing splat', 0); splat = parsePly(m.ply); newSplat = true; glb = null; binding = null; }
      if (m.cage) { progress('parsing cage', 0); cage = parseCage(m.cage); newCage = true; }
      if (m.glb) {   // splat, binding and clips in one file; the clip is an animation of it
        progress('parsing glTF', 0); glb = await parseGlb(m.glb);
        splat = glb.splat; binding = glb.binding; newSplat = true; app = null;
      }
      if (glb && (m.glb || m.anim !== undefined)) {
        const ai = Math.max(0, glb.anims.indexOf(m.anim ?? glb.anims[0]));
        progress(`posing ${glb.anims[ai]}`, 0); cage = glb.cage(ai); newCage = true;
      }
      if ('app' in m) app = m.app ? parseApp(m.app) : null;
      if (m.minConf !== undefined && m.minConf !== minConf) { minConf = m.minConf; newCage = true; }
      if (!splat || !cage) { post({ type: 'loaded', partial: true }); return; }
      const msg = { type: 'loaded' }, transfer = [];
      if (newSplat || newCage || !rig) {
        rig = new Rig(splat, cage, { minConf, binding, progress: f => progress('binding splats to the cage', f) });
        last = null;
        const tex = rig.splatTexels();
        msg.splatTex = tex; transfer.push(tex.buffer);
        const bodies = new Float32Array(cage.nframes * 6);
        for (let fr = 0; fr < cage.nframes; fr++) {
          const b = rig.bodyCentre(fr);
          bodies.set(b.centre, fr * 6); bodies.set(b.size, fr * 6 + 3);
        }
        msg.bodies = bodies;
      }
      if (newSplat) { const sh = rig.shTexels(); msg.sh = sh; transfer.push(sh.data.buffer); }
      let appError = null;
      try { progress('preparing the MLP', 0); rig.setApp(app); } catch (e) { appError = e.message; rig.setApp(null); }
      last = null;
      Object.assign(msg, {
        n: splat.n, degree: splat.degree, nv: cage.nv, nf: cage.nf, nframes: cage.nframes, names: cage.names,
        nFallback: rig.nFallback, nUnbound: rig.nUnbound, hasLabels: !!splat.labels,
        app: rig.app ? { version: rig.app.version, nfeat: rig.app.nfeat, dz: rig.app.dz, maxDo: rig.app.maxDo, maxDs: rig.app.maxDs, maxDp: rig.app.maxDp } : null,
        appError, ms: performance.now() - t0,
        glb: glb ? { anims: glb.anims, render: glb.render, subject: glb.subject, residual: cage.residual, fps: cage.fps } : null,
      });
      post(msg, transfer);
    } else if (m.type === 'frame') {
      if (!rig) return;
      const t0 = performance.now();
      const fr = Math.min(Math.max(m.fr, 0), cage.nframes - 1);
      const { face, vout, centres } = rig.frame(fr, m.opts);
      const t1 = performance.now();
      last = { fr, key: optsKey(m.opts), centres };
      const order = sort(centres, m.view);
      const v = vout ? vout.slice() : null;
      const transfer = [face.buffer, order.buffer];
      if (v) transfer.push(v.buffer);
      post({ type: 'frame', id: m.id, fr, face, vout: v, order, msRig: t1 - t0, msSort: performance.now() - t1 }, transfer);
    } else if (m.type === 'sort') {
      if (!last) return;
      const t0 = performance.now();
      const order = sort(last.centres, m.view);
      post({ type: 'sort', id: m.id, fr: last.fr, order, msSort: performance.now() - t0 }, [order.buffer]);
    } else if (m.type === 'precompute') {
      // warm the MLP cache for every frame so playback does not stall on it
      if (!rig || !rig.app) return;
      const gen = m.gen;
      for (let fr = 0; fr < cage.nframes; fr++) {
        rig.vout(fr);
        post({ type: 'precomputed', gen, fr, of: cage.nframes });
        await new Promise(r => setTimeout(r, 0));   // let frame / sort requests in between
        if (self.cancelGen !== undefined && self.cancelGen >= gen) return;
      }
    } else if (m.type === 'cancel') {
      self.cancelGen = m.gen;
    }
  } catch (e) {
    post({ type: 'error', error: e.message, stack: e.stack });
  }
};
