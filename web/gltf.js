// Reader for the b2c glTF files (~/Projects/b2cgltf/SPEC.md): a b2crig-rigged subject file and its clip files.
//   parseSubject  the current splat (7.1) with b2ctrain's binding (5.3), the cage (5.2) and the skeleton (4.3)
//   parseClip     one clip file (6), checked against its subject: same b2c_id, same rig hash, same skeleton
//   poseCage      a clip frame's posed cage (7.2 steps 1-2): the cage skinned by the joints relative to the skeleton
//                 root, plus the clip's residual
// Every array stays in the b2crunner frame (SPEC 2); the viewer applies W once, in its camera.

const KHR = 'KHR_gaussian_splatting', RIG = 'B2CRIG_rig', BIND = 'B2CRIG_splat_cage', CLIP = 'B2CRIG_clip',
  RES = 'B2CRIG_cage_residual';
const NCOMP = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT3: 9, MAT4: 16 };
const TYPED = { 5120: Int8Array, 5121: Uint8Array, 5122: Int16Array, 5123: Uint16Array, 5125: Uint32Array, 5126: Float32Array };

// GLB container: the JSON chunk and buffer 0 (the BIN chunk; SPEC R3 allows no other buffer).
function readGlb(buf) {
  const dv = new DataView(buf);
  if (buf.byteLength < 20 || dv.getUint32(0, true) !== 0x46546c67) throw new Error('not a .glb file');
  if (dv.getUint32(4, true) !== 2) throw new Error(`glTF container version ${dv.getUint32(4, true)} is not supported`);
  let o = 12, json = null, bin = null;
  while (o + 8 <= buf.byteLength) {
    const len = dv.getUint32(o, true), type = dv.getUint32(o + 4, true);
    if (type === 0x4e4f534a) json = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, o + 8, len)));
    else if (type === 0x004e4942) bin = new Uint8Array(buf, o + 8, len);
    o += 8 + len;
  }
  if (!json) throw new Error('the .glb has no JSON chunk');
  return { json, acc: i => accessor(json, bin, i) };
}

// Accessor i as a flat typed array of its component type (strided data de-interleaved; matrices stay column-major).
function accessor(json, bin, i) {
  const a = json.accessors[i], T = TYPED[a.componentType], nc = NCOMP[a.type], n = a.count * nc;
  if (!T || !nc) throw new Error(`accessor ${i}: unsupported type ${a.type}/${a.componentType}`);
  if (a.bufferView === undefined) return new T(n);
  const bv = json.bufferViews[a.bufferView];
  if ((bv.buffer ?? 0) !== 0 || !bin) throw new Error(`accessor ${i}: only buffer 0 (the BIN chunk) is supported`);
  const off = bin.byteOffset + (bv.byteOffset ?? 0) + (a.byteOffset ?? 0), el = nc * T.BYTES_PER_ELEMENT;
  if (bv.byteStride && bv.byteStride !== el) {
    const out = new T(n), src = new Uint8Array(bin.buffer), dst = new Uint8Array(out.buffer);
    for (let k = 0; k < a.count; k++) dst.set(src.subarray(off + k * bv.byteStride, off + k * bv.byteStride + el), k * el);
    return out;
  }
  if (off % T.BYTES_PER_ELEMENT === 0) return new T(bin.buffer, off, n);
  return new T(bin.buffer.slice(off, off + n * T.BYTES_PER_ELEMENT));
}

function version(ext, name, known = 1) {
  if (ext.version !== known) throw new Error(`${name} version ${ext.version} is not supported (this viewer reads ${known})`);
}

// the parent of every node (-1 for roots)
function parentOf(json) {
  const p = new Int32Array(json.nodes.length).fill(-1);
  json.nodes.forEach((nd, i) => (nd.children || []).forEach(c => { p[c] = i; }));
  return p;
}

// SPEC 7.1: the splat node the default scene lists (directly or below a listed root) that no other node supersedes.
function currentSplat(json) {
  const scene = json.scenes[json.scene ?? 0], listed = [], stack = [...scene.nodes];
  while (stack.length) { const i = stack.pop(); listed.push(i); stack.push(...(json.nodes[i].children || [])); }
  const superseded = new Set(json.nodes.map(nd => nd.extras?.supersedes).filter(x => x !== undefined));
  const splats = listed.filter(i => {
    const m = json.nodes[i].mesh;
    return m !== undefined && !superseded.has(i) && json.meshes[m].primitives.some(p => p.extensions?.[KHR]);
  });
  if (splats.length !== 1) throw new Error(`the default scene has ${splats.length} current splats (SPEC 7.1 wants one)`);
  return splats[0];
}

function skeletonOf(json, acc, skinIndex) {
  const skin = json.skins[skinIndex], joints = skin.joints, parent = parentOf(json);
  const jof = new Map(joints.map((n, j) => [n, j]));
  const parents = Int32Array.from(joints, n => jof.get(parent[n]) ?? -1);
  const root = skin.skeleton ?? parent[joints[parents.indexOf(-1)]];
  const order = [], seen = new Uint8Array(joints.length);   // parents before children
  const visit = j => { if (seen[j]) return; if (parents[j] >= 0) visit(parents[j]); seen[j] = 1; order.push(j); };
  for (let j = 0; j < joints.length; j++) visit(j);
  const restQ = new Float32Array(joints.length * 4), restT = new Float32Array(joints.length * 3);
  joints.forEach((n, j) => { restQ.set(json.nodes[n].rotation ?? [0, 0, 0, 1], j * 4); restT.set(json.nodes[n].translation ?? [0, 0, 0], j * 3); });
  const ibm = skin.inverseBindMatrices !== undefined ? Float32Array.from(acc(skin.inverseBindMatrices)) : identities(joints.length);
  return { joints, parents, order, root, W: nodeMatrix(json.nodes[root]), restQ, restT, ibm };
}

function identities(n) {
  const m = new Float32Array(n * 16);
  for (let j = 0; j < n; j++) m[j * 16] = m[j * 16 + 5] = m[j * 16 + 10] = m[j * 16 + 15] = 1;
  return m;
}

function nodeMatrix(nd) {   // column-major 4x4
  if (nd.matrix) return Float64Array.from(nd.matrix);
  const m = new Float64Array(16), s = nd.scale ?? [1, 1, 1];
  trsMat(nd.rotation ?? [0, 0, 0, 1], nd.translation ?? [0, 0, 0], m, 0);
  for (let c = 0; c < 3; c++) for (let r = 0; r < 3; r++) m[c * 4 + r] *= s[c];
  return m;
}

// ---------------------------------------------------------------- the subject file

// The splat (display form converted to the renderer's: logit opacity, log scales, w x y z rotation, SH
// coefficient-major per splat), b2ctrain's binding, the cage, the skeleton and b2ctrain's render settings.
export async function parseSubject(buf) {
  const { json, acc } = readGlb(buf);
  const id = json.asset?.extras?.b2c_id;
  if (!id) throw new Error('not a b2c subject file (no asset.extras.b2c_id)');
  if (json.asset.extras.b2c_stripped) throw new Error('this is a stripped distribution file (SPEC R8): it has no rig to play');
  const rig = json.extensions?.[RIG];
  if (!rig) throw new Error('this subject file has no rig yet (b2crig: tools/export_gltf.py rig)');
  version(rig, RIG);

  // the splat
  const si = currentSplat(json);
  if (si !== rig.splatNode) throw new Error(`the current splat (node ${si}) is not b2crig's rigged splat (node ${rig.splatNode})`);
  const prim = json.meshes[json.nodes[si].mesh].primitives.find(p => p.extensions?.[KHR]), at = prim.attributes;
  const be = prim.extensions?.[BIND];
  if (!be) throw new Error('the rigged splat has no B2CRIG_splat_cage binding');
  version(be, BIND);
  const A = s => at[`${KHR}:${s}`] !== undefined ? acc(at[`${KHR}:${s}`]) : null;
  const pos = Float32Array.from(acc(at.POSITION)), n = pos.length / 3;
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
    quat[i * 4] = rot[i * 4 + 3]; quat[i * 4 + 1] = rot[i * 4]; quat[i * 4 + 2] = rot[i * 4 + 1]; quat[i * 4 + 3] = rot[i * 4 + 2];
    for (let k = 0; k < 3; k++) ls[i * 3 + k] = Math.log(scale[i * 3 + k]);
    for (let k = 0; k < K; k++) for (let c = 0; c < 3; c++) sh[(i * K + k) * 3 + c] = coefs[k][i * 3 + c];
  }
  const splat = { n, degree, K, pos, op, quat, ls, sh };
  const binding = { face: Int32Array.from(acc(be.face)),   // 0xFFFFFFFF (unbound) wraps to -1
                    bary: Float32Array.from(acc(be.barycentric)), offset: Float32Array.from(acc(be.offset)), posing: be.posing };

  // the cage: one primitive per layer in b2ctrain's cage order, skinned by b2crunner's skin
  const cageNode = json.nodes[rig.cageNode], cagePrims = json.meshes[cageNode.mesh].primitives;
  const nv = rig.layers.reduce((s, L) => s + L.vertexCount, 0), nf = rig.layers.reduce((s, L) => s + L.faceCount, 0);
  const sets = Math.max(...cagePrims.map(p => Object.keys(p.attributes).filter(k => /^JOINTS_\d+$/.test(k)).length));
  const KW = 4 * sets;
  const verts0 = new Float32Array(nv * 3), faces = new Int32Array(nf * 3), jIdx = new Uint16Array(nv * KW), jW = new Float32Array(nv * KW);
  const layers = rig.layers.map(L => {
    const p = cagePrims[L.primitive], vo = L.vertexOffset, V = acc(p.attributes.POSITION);
    if (V.length !== L.vertexCount * 3) throw new Error(`cage layer ${L.name}: vertex count differs from B2CRIG_rig`);
    verts0.set(V, vo * 3);
    const idx = acc(p.indices);
    for (let k = 0; k < idx.length; k++) faces[L.faceOffset * 3 + k] = idx[k] + vo;
    for (let s = 0; s < sets; s++) {
      if (p.attributes[`JOINTS_${s}`] === undefined) continue;
      const jv = acc(p.attributes[`JOINTS_${s}`]), wa = json.accessors[p.attributes[`WEIGHTS_${s}`]], wv = acc(p.attributes[`WEIGHTS_${s}`]);
      const wn = wa.normalized ? 1 / (wa.componentType === 5121 ? 255 : 65535) : 1;
      for (let v = 0; v < L.vertexCount; v++) for (let k = 0; k < 4; k++) {
        jIdx[(vo + v) * KW + s * 4 + k] = jv[v * 4 + k]; jW[(vo + v) * KW + s * 4 + k] = wv[v * 4 + k] * wn;
      }
    }
    return { vOff: vo, vCount: L.vertexCount, fOff: L.faceOffset, fCount: L.faceCount, name: L.name };
  });
  const cage = { nv, nf, layers, verts0, faces, jIdx, jW, KW, body: rig.bodyLayer ?? 0 };
  const skeleton = skeletonOf(json, acc, cageNode.skin);
  const rigHash = await hashRig(acc, cagePrims, skeleton);
  return { id, splat, binding, cage, skeleton, rigHash, render: rig.render || {} };
}

// B2CRIG_clip.rig.cageSha256 (SPEC 6), as b2cgltf rig.rig_hash: per cage primitive its POSITION, indices, then
// JOINTS_n / WEIGHTS_n (n ascending, JOINTS first); then the inverse bind matrices (row-major, as numpy holds them),
// rest rotations and rest translations, all float32 little-endian.
async function hashRig(acc, prims, sk) {
  const bytes = a => new Uint8Array(a.buffer, a.byteOffset, a.byteLength), parts = [];
  for (const p of prims) {
    const at = p.attributes;
    const skin = Object.keys(at).filter(k => /^(JOINTS|WEIGHTS)_\d+$/.test(k))
      .sort((a, b) => (+a.split('_')[1] - +b.split('_')[1]) || (a < b ? -1 : 1));
    parts.push(bytes(acc(at.POSITION)), bytes(acc(p.indices)), ...skin.map(k => bytes(acc(at[k]))));
  }
  const J = sk.joints.length, ibmRows = new Float32Array(J * 16);
  for (let j = 0; j < J; j++) for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) ibmRows[j * 16 + r * 4 + c] = sk.ibm[j * 16 + c * 4 + r];
  parts.push(bytes(ibmRows), bytes(sk.restQ), bytes(sk.restT));
  const all = new Uint8Array(parts.reduce((s, x) => s + x.length, 0));
  let o = 0; for (const x of parts) { all.set(x, o); o += x.length; }
  const h = new Uint8Array(await crypto.subtle.digest('SHA-256', all));
  return Array.from(h, b => b.toString(16).padStart(2, '0')).join('');
}

// ---------------------------------------------------------------- clip files

// A clip file for `subject` (parseSubject's result), refused as SPEC 6 says a player must. Joint TRS come out as
// q [T * J * 4] (x y z w), t [T * J * 3]; joints without a channel keep their rest pose.
export function parseClip(buf, subject) {
  const { json, acc } = readGlb(buf);
  const c = json.extensions?.[CLIP];
  if (!c) throw new Error('not a clip file (no B2CRIG_clip)');
  version(c, CLIP);
  if (c.subject.id !== subject.id) throw new Error(`the clip belongs to subject ${c.subject.id}, not ${subject.id}`);
  if (c.rig.cageSha256 !== subject.rigHash) throw new Error('the clip was made for another rig (cageSha256 differs): re-export it');
  const sk = subject.skeleton, J = sk.joints.length;
  if (c.skeleton.joints.length !== J) throw new Error(`the clip has ${c.skeleton.joints.length} joints, the subject ${J}`);
  const parent = parentOf(json), jof = new Map(c.skeleton.joints.map((n, j) => [n, j]));
  for (let j = 0; j < J; j++) {
    const nd = json.nodes[c.skeleton.joints[j]], p = jof.get(parent[c.skeleton.joints[j]]) ?? -1;
    const q = Float32Array.from(nd.rotation ?? [0, 0, 0, 1]), t = Float32Array.from(nd.translation ?? [0, 0, 0]);
    if (p !== sk.parents[j] || q.some((x, k) => x !== sk.restQ[j * 4 + k]) || t.some((x, k) => x !== sk.restT[j * 3 + k]))
      throw new Error(`the clip's skeleton copy differs from the subject's at joint ${j}`);
  }
  const anim = json.animations[0], src = Array.from({ length: J }, () => ({ r: null, t: null }));
  let T = 0;
  for (const ch of anim.channels) {
    const j = jof.get(ch.target.node); if (j === undefined) continue;
    const out = acc(anim.samplers[ch.sampler].output);
    if (ch.target.path === 'rotation') { src[j].r = out; T = out.length / 4; }
    else if (ch.target.path === 'translation') { src[j].t = out; T = out.length / 3; }
  }
  if (!T) throw new Error('the clip has no joint channels');
  const q = new Float32Array(T * J * 4), t = new Float32Array(T * J * 3);
  for (let f = 0; f < T; f++) for (let j = 0; j < J; j++) {
    q.set(src[j].r ? src[j].r.subarray(f * 4, f * 4 + 4) : sk.restQ.subarray(j * 4, j * 4 + 4), (f * J + j) * 4);
    t.set(src[j].t ? src[j].t.subarray(f * 3, f * 3 + 3) : sk.restT.subarray(j * 3, j * 3 + 3), (f * J + j) * 3);
  }
  const re = anim.extensions?.[RES];
  let residual = null;
  if (re) {
    version(re, RES);
    if (re.frames !== T || re.vertices !== subject.cage.nv)
      throw new Error(`the residual is ${re.frames} x ${re.vertices}, the clip ${T} frames x ${subject.cage.nv} cage vertices`);
    residual = { data: acc(re.accessor), scale: re.scale };
  }
  return { name: c.name, fps: c.fps, nframes: T, q, t, residual };
}

// SPEC 7.2 steps 1-2 for frame f: skin the cage with W^-1 . joint_world . IBM (the joints relative to the skeleton
// root), then add the residual (normalised SHORT x scale). Returns Float32Array [nv * 3] in the b2crunner frame.
export function poseCage(subject, clip, f, residual = true, out = new Float32Array(subject.cage.nv * 3)) {
  const sk = subject.skeleton, J = sk.joints.length, { nv, verts0, jIdx, jW, KW } = subject.cage;
  const G = new Float64Array(J * 16), L = new Float64Array(16), M = new Float64Array(J * 12);
  for (const j of sk.order) {
    trsMat(clip.q.subarray((f * J + j) * 4, (f * J + j) * 4 + 4), clip.t.subarray((f * J + j) * 3, (f * J + j) * 3 + 3), L, 0);
    if (sk.parents[j] >= 0) mul4(G, sk.parents[j] * 16, L, G, j * 16); else G.set(L, j * 16);
    for (let r = 0; r < 3; r++) for (let c = 0; c < 4; c++) {   // M_j = G_j IBM_j, kept as 3x4 rows
      let s = 0; for (let k = 0; k < 4; k++) s += G[j * 16 + k * 4 + r] * sk.ibm[j * 16 + c * 4 + k];
      M[j * 12 + r * 4 + c] = s;
    }
  }
  const R = residual && clip.residual, rq = R ? R.data : null, rs = R ? R.scale / 32767 : 0, ro = f * nv * 3;
  for (let v = 0; v < nv; v++) {
    const x = verts0[v * 3], y = verts0[v * 3 + 1], z = verts0[v * 3 + 2];
    let px = 0, py = 0, pz = 0;
    for (let k = 0; k < KW; k++) {
      const w = jW[v * KW + k]; if (w === 0) continue;
      const m = jIdx[v * KW + k] * 12;
      px += w * (M[m] * x + M[m + 1] * y + M[m + 2] * z + M[m + 3]);
      py += w * (M[m + 4] * x + M[m + 5] * y + M[m + 6] * z + M[m + 7]);
      pz += w * (M[m + 8] * x + M[m + 9] * y + M[m + 10] * z + M[m + 11]);
    }
    if (rq) {   // normalised SHORT: max(c / 32767, -1)
      const r = ro + v * 3;
      px += Math.max(rq[r], -32767) * rs; py += Math.max(rq[r + 1], -32767) * rs; pz += Math.max(rq[r + 2], -32767) * rs;
    }
    out[v * 3] = px; out[v * 3 + 1] = py; out[v * 3 + 2] = pz;
  }
  return out;
}

function trsMat(q, t, out, o) {   // column-major 4x4 from a unit quaternion (x, y, z, w) and a translation
  const [x, y, z, w] = q;
  out[o] = 1 - 2 * (y * y + z * z); out[o + 1] = 2 * (x * y + z * w); out[o + 2] = 2 * (x * z - y * w); out[o + 3] = 0;
  out[o + 4] = 2 * (x * y - z * w); out[o + 5] = 1 - 2 * (x * x + z * z); out[o + 6] = 2 * (y * z + x * w); out[o + 7] = 0;
  out[o + 8] = 2 * (x * z + y * w); out[o + 9] = 2 * (y * z - x * w); out[o + 10] = 1 - 2 * (x * x + y * y); out[o + 11] = 0;
  out[o + 12] = t[0]; out[o + 13] = t[1]; out[o + 14] = t[2]; out[o + 15] = 1;
}

function mul4(A, ao, B, out, oo) {   // column-major out[oo..] = A[ao..] B (B must not alias out)
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0; for (let k = 0; k < 4; k++) s += A[ao + k * 4 + r] * B[c * 4 + k];
    out[oo + c * 4 + r] = s;
  }
}
