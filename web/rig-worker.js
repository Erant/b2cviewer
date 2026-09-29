// Worker: parses the subject and clip files (gltf.js), and per request poses a clip frame's cage, builds the face
// texture and a back-to-front splat order for the current camera. Without a clip it shows the rest pose.
import { parseSubject, parseClip, poseCage } from './gltf.js';
import { Rig } from './rig.js';

let subject = null, clip = null, rig = null;
let last = null;   // { fr, centres }: the centres of the frame last computed

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

const nframes = () => clip ? clip.nframes : 1;
const posed = (fr, residual = true) => clip ? poseCage(subject, clip, fr, residual) : subject.cage.verts0;

self.onmessage = async ({ data: m }) => {
  try {
    if (m.type === 'load') {
      const t0 = performance.now(), msg = { type: 'loaded' }, transfer = [];
      if (m.subject) {
        progress('parsing the subject file', 0);
        subject = await parseSubject(m.subject); clip = null;
        rig = new Rig(subject.splat, subject.binding, subject.cage);
        const tex = rig.splatTexels(), sh = rig.shTexels();
        msg.splatTex = tex; msg.sh = sh; transfer.push(tex.buffer, sh.data.buffer);
      }
      if (!subject) throw new Error('load a subject file first (a rigged scene.glb)');
      if ('clip' in m) { progress('parsing the clip', 0); clip = m.clip ? parseClip(m.clip, subject) : null; }
      last = null;
      const T = nframes(), bodies = new Float32Array(T * 6);
      for (let fr = 0; fr < T; fr++) {
        if ((fr & 31) === 0) progress('posing the clip', fr / T);
        const b = rig.bodyCentre(posed(fr));
        bodies.set(b.centre, fr * 6); bodies.set(b.size, fr * 6 + 3);
      }
      const { splat, cage, skeleton } = subject;
      Object.assign(msg, {
        bodies, n: splat.n, degree: splat.degree, nv: cage.nv, nf: cage.nf, nframes: T, nUnbound: rig.nUnbound,
        layers: cage.layers.map(L => L.name), W: Array.from(skeleton.W), render: subject.render, id: subject.id,
        clip: clip ? { name: clip.name, fps: clip.fps, residual: !!clip.residual } : null, ms: performance.now() - t0,
      });
      post(msg, transfer);
    } else if (m.type === 'frame') {
      if (!rig) return;
      const t0 = performance.now();
      const fr = Math.min(Math.max(m.fr, 0), nframes() - 1);
      const { face, centres } = rig.frame(posed(fr, m.opts.residual), m.opts.maxGrowth);
      const t1 = performance.now();
      last = { fr, centres };
      const order = sort(centres, m.view);
      post({ type: 'frame', fr, face, order, msRig: t1 - t0, msSort: performance.now() - t1 }, [face.buffer, order.buffer]);
    } else if (m.type === 'sort') {
      if (!last) return;
      const order = sort(last.centres, m.view);
      post({ type: 'sort', fr: last.fr, order }, [order.buffer]);
    }
  } catch (e) {
    post({ type: 'error', error: e.message, stack: e.stack });
  }
};
