// A b2crig subject .glb (b2crig/export/gltf.py; layout in b2crig docs/gltf-export.md), read with glTF-Transform
// (web/vendor, MIT) into what the rest of the viewer already uses:
//   splat    parsePly's shape, from the KHR_gaussian_splatting primitive (+ seg labels from B2C_splat_cage)
//   binding  b2ctrain's binding (B2C_splat_cage: face, barycentrics, offset), so Rig skips its own
//   cage(i)  parseCage's shape for animation i: the skinned cage mesh posed by the animation's joint channels
//            (standard glTF skinning, up to 8 joints per vertex) plus the B2C_cage_residual when present
import { WebIO, Logger } from './vendor/gltf-transform-core.js';

const KHR = 'KHR_gaussian_splatting', RIG = 'B2C_rig', BIND = 'B2C_splat_cage', RES = 'B2C_cage_residual';

export async function parseGlb(buf) {
  const io = new WebIO().setLogger(new Logger(Logger.Verbosity.ERROR));   // our B2C_* extensions are read from the JSON
  const jsonDoc = await io.binaryToJSON(new Uint8Array(buf));
  const json = jsonDoc.json, ext = json.extensions?.[RIG];
  if (!ext) throw new Error('not a b2crig subject glTF (no B2C_rig extension)');
  const doc = await io.readJSON(jsonDoc), root = doc.getRoot();
  const acc = root.listAccessors(), nodes = root.listNodes();
  const arr = i => acc[i].getArray();

  // splat
  const sp = nodes[ext.splatNode].getMesh().listPrimitives()[0];
  const spJson = json.meshes[json.nodes[ext.splatNode].mesh].primitives[0];
  const A = s => { const a = sp.getAttribute(`${KHR}:${s}`); return a && a.getArray(); };
  const pos = Float32Array.from(sp.getAttribute('POSITION').getArray()), n = pos.length / 3;
  const rot = A('ROTATION'), scale = A('SCALE'), opac = A('OPACITY');
  const coefs = [A('SH_DEGREE_0_COEF_0')];
  for (const [d, m] of [[1, 3], [2, 5], [3, 7]]) {
    if (!A(`SH_DEGREE_${d}_COEF_0`)) break;
    for (let c = 0; c < m; c++) coefs.push(A(`SH_DEGREE_${d}_COEF_${c}`));
  }
  const K = coefs.length, degree = Math.round(Math.sqrt(K)) - 1;
  const op = new Float32Array(n), quat = new Float32Array(n * 4), ls = new Float32Array(n * 3), sh = new Float32Array(n * K * 3);
  for (let i = 0; i < n; i++) {
    const o = Math.min(Math.max(opac[i], 1e-7), 1 - 1e-7);
    op[i] = Math.log(o / (1 - o));
    quat[i * 4] = rot[i * 4 + 3]; quat[i * 4 + 1] = rot[i * 4]; quat[i * 4 + 2] = rot[i * 4 + 1]; quat[i * 4 + 3] = rot[i * 4 + 2];   // xyzw -> wxyz
    for (let k = 0; k < 3; k++) ls[i * 3 + k] = Math.log(scale[i * 3 + k]);
    for (let k = 0; k < K; k++) for (let c = 0; c < 3; c++) sh[(i * K + k) * 3 + c] = coefs[k][i * 3 + c];
  }
  const be = spJson.extensions[BIND];
  let labels = null;
  if (be.segLabel !== undefined) {
    const l = arr(be.segLabel), c = arr(be.segConf);
    labels = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) { labels[i * 2] = l[i]; labels[i * 2 + 1] = c[i]; }
  }
  const splat = { n, degree, K, pos, op, quat, ls, sh, labels };
  const binding = { face: new Int32Array(Uint32Array.from(arr(be.face)).buffer), bary: Float32Array.from(arr(be.barycentric)),
                    offset: Float32Array.from(arr(be.offset)) };

  // skeleton
  const skin = root.listSkins()[ext.skin], joints = skin.listJoints(), J = joints.length;
  const jointOf = new Map(joints.map((nd, j) => [nd, j]));
  const parent = joints.map(nd => { const p = nd.getParentNode(); return p && jointOf.has(p) ? jointOf.get(p) : -1; });
  const order = [], seen = new Uint8Array(J);   // parents before children
  const visit = j => { if (seen[j]) return; if (parent[j] >= 0) visit(parent[j]); seen[j] = 1; order.push(j); };
  for (let j = 0; j < J; j++) visit(j);
  const ibm = skin.getInverseBindMatrices().getArray();

  // cage (one primitive per layer, in b2ctrain's cage order)
  const cagePrims = nodes[ext.cageNode].getMesh().listPrimitives();
  const nv = ext.layers.reduce((s, L) => s + L.vertexCount, 0), nf = ext.layers.reduce((s, L) => s + L.faceCount, 0);
  const verts0 = new Float32Array(nv * 3), faces = new Int32Array(nf * 3), jIdx = new Uint16Array(nv * 8), jW = new Float32Array(nv * 8);
  const layers = ext.layers.map((L, l) => {
    const p = cagePrims[L.primitive], vo = L.vertexOffset;
    verts0.set(p.getAttribute('POSITION').getArray(), vo * 3);
    const idx = p.getIndices().getArray();
    for (let k = 0; k < idx.length; k++) faces[L.faceOffset * 3 + k] = idx[k] + vo;
    for (const [set, o] of [[0, 0], [1, 4]]) {
      const ja = p.getAttribute(`JOINTS_${set}`), wa = p.getAttribute(`WEIGHTS_${set}`);
      if (!ja) continue;
      const jv = ja.getArray(), wv = wa.getArray();
      for (let v = 0; v < L.vertexCount; v++) for (let k = 0; k < 4; k++) {
        jIdx[(vo + v) * 8 + o + k] = jv[v * 4 + k]; jW[(vo + v) * 8 + o + k] = wv[v * 4 + k];
      }
    }
    let bits = 0; for (const c of L.classes) bits |= 1 << c;
    return { vOff: vo, vCount: L.vertexCount, fOff: L.faceOffset, fCount: L.faceCount, classBits: bits >>> 0, name: L.name };
  });

  const anims = root.listAnimations(), animJson = json.animations || [];
  // Posed cage of animation `ai`: one key per frame (as the exporter writes it); joints without a channel keep their rest pose.
  function cage(ai) {
    const an = anims[ai], chans = an.listChannels();
    const T = chans.length ? chans[0].getSampler().getInput().getCount() : 1;
    const times = chans.length ? chans[0].getSampler().getInput().getArray() : [0];
    const rest = joints.map(nd => [nd.getRotation(), nd.getTranslation()]);
    const src = joints.map(() => ({ r: null, t: null }));
    for (const ch of chans) {
      const j = jointOf.get(ch.getTargetNode()); if (j === undefined) continue;
      const out = ch.getSampler().getOutput().getArray(), path = ch.getTargetPath();
      if (path === 'rotation') src[j].r = out; else if (path === 'translation') src[j].t = out;
    }
    const res = animJson[ai]?.extensions?.[RES];
    const rq = res ? arr(res.accessor) : null, rs = res ? res.scale / 32767 : 0;
    const posed = new Float32Array(T * nv * 3);
    const W = new Float64Array(J * 16), M = new Float64Array(J * 12), L = new Float64Array(16);
    for (let f = 0; f < T; f++) {
      for (const j of order) {
        const q = src[j].r ? src[j].r.subarray(f * 4, f * 4 + 4) : rest[j][0];
        const t = src[j].t ? src[j].t.subarray(f * 3, f * 3 + 3) : rest[j][1];
        trsMat(q, t, L);
        if (parent[j] >= 0) mul4(W, parent[j] * 16, L, 0, W, j * 16); else W.set(L, j * 16);
        // M_j = W_j IBM_j (glTF matrices are column-major), kept as 3x4 rows
        for (let r = 0; r < 3; r++) for (let c = 0; c < 4; c++) {
          let s = 0; for (let k = 0; k < 4; k++) s += W[j * 16 + k * 4 + r] * ibm[j * 16 + c * 4 + k];
          M[j * 12 + r * 4 + c] = s;
        }
      }
      const o = f * nv * 3;
      for (let v = 0; v < nv; v++) {
        const x = verts0[v * 3], y = verts0[v * 3 + 1], z = verts0[v * 3 + 2];
        let px = 0, py = 0, pz = 0;
        for (let k = 0; k < 8; k++) {
          const w = jW[v * 8 + k]; if (w === 0) continue;
          const m = jIdx[v * 8 + k] * 12;
          px += w * (M[m] * x + M[m + 1] * y + M[m + 2] * z + M[m + 3]);
          py += w * (M[m + 4] * x + M[m + 5] * y + M[m + 6] * z + M[m + 7]);
          pz += w * (M[m + 8] * x + M[m + 9] * y + M[m + 10] * z + M[m + 11]);
        }
        if (rq) { const r = o + v * 3; px += rq[r] * rs; py += rq[r + 1] * rs; pz += rq[r + 2] * rs; }
        posed[o + v * 3] = px; posed[o + v * 3 + 1] = py; posed[o + v * 3 + 2] = pz;
      }
    }
    const names = Array.from({ length: T }, (_, i) => String(i).padStart(4, '0'));
    return { nv, nf, nframes: T, layers, verts0, faces, names, posed, fps: T > 1 ? 1 / (times[1] - times[0]) : 0, residual: !!res };
  }
  return { splat, binding, cage, anims: anims.map((a, i) => a.getName() || `animation ${i}`), render: ext.render || {}, subject: ext.subject || '' };
}

function trsMat(q, t, out) {   // column-major 4x4 from a unit quaternion (x, y, z, w) and a translation
  const [x, y, z, w] = q;
  out[0] = 1 - 2 * (y * y + z * z); out[1] = 2 * (x * y + z * w); out[2] = 2 * (x * z - y * w); out[3] = 0;
  out[4] = 2 * (x * y - z * w); out[5] = 1 - 2 * (x * x + z * z); out[6] = 2 * (y * z + x * w); out[7] = 0;
  out[8] = 2 * (x * z + y * w); out[9] = 2 * (y * z - x * w); out[10] = 1 - 2 * (x * x + y * y); out[11] = 0;
  out[12] = t[0]; out[13] = t[1]; out[14] = t[2]; out[15] = 1;
}

function mul4(A, ao, B, bo, out, oo) {   // column-major out = A B (out may not alias B)
  const tmp = new Float64Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0; for (let k = 0; k < 4; k++) s += A[ao + k * 4 + r] * B[bo + c * 4 + k];
    tmp[c * 4 + r] = s;
  }
  out.set(tmp, oo);
}
