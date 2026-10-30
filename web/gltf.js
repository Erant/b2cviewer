// Reader for the b2c glTF files (~/Projects/b2cgltf/SPEC.md): a b2crig-rigged subject file and its clip files, and
// their web forms (sog.js): a subject .sog (SOG splat + binding textures + rig.glb) and .clip.zip files.
//   parseSubject  the current splat (7.1), or one of its LOD levels (4.7, 7.1a), with b2ctrain's binding (5.3), the
//                 cage (5.2) and the skeleton (4.3); a file b2crig has not rigged yet, or a plain .sog, gives the
//                 splat alone (shown unposed)
//   parseClip     one clip file (6), checked against its subject: same b2c_id, same rig hash, same skeleton
//   poseCage      a clip frame's posed cage (7.2 steps 1-2): the cage skinned by the joints relative to the skeleton
//                 root, plus the clip's residual
// Every array stays in the b2crunner frame (SPEC 2); the viewer applies W once, in its camera.
import { isZip, readZip, parseSog } from './sog.js';

const KHR = 'KHR_gaussian_splatting', RIG = 'B2CRIG_rig', BIND = 'B2CRIG_splat_cage', CLIP = 'B2CRIG_clip',
  RES = 'B2CRIG_cage_residual', LOD = 'MSFT_lod';
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
  const names = joints.map(n => json.nodes[n].name || '');
  return { joints, names, parents, order, root, W: nodeMatrix(json.nodes[root]), restQ, restT, ibm };
}

// The glTF world matrix of node i's parent chain (an unrigged splat sits under b2c_world, whose matrix is W).
function worldMatrix(json, i) {
  const parent = parentOf(json);
  let M = new Float64Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  for (let p = parent[i]; p >= 0; p = parent[p]) { const out = new Float64Array(16); mul4(nodeMatrix(json.nodes[p]), 0, M, out, 0); M = out; }
  return M;
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
// level: the member of the current splat's LOD chain to show (0 = the current splat itself); the result's `lods` is
// the chain (lodChain), `lodProblem` why a chain the file has is not used.
// decode: WebP decoder for a .sog (sog.js's WebGL2 readback by default; a web subject has no LOD levels).
// buf: the .glb or .sog bytes, or one level of a streamed web subject (b2cgltf SPEC 10.3), { files, lods, level }:
// that level's SOG files (+ the shared rig.glb under the name its meta.json gives) and the chain from lod-meta.json.
export async function parseSubject(buf, decode, level = 0) {
  if (buf.files) return { ...await parseWebSubject(buf.files, decode), lods: buf.lods, lodProblem: null, level: buf.level };
  if (isZip(buf)) return { ...await parseWebSubject(await readZip(buf), decode), lods: null, lodProblem: null };
  const { json, acc } = readGlb(buf);
  const id = json.asset?.extras?.b2c_id;
  if (!id) throw new Error('not a b2c subject file (no asset.extras.b2c_id)');
  const stripped = json.asset.extras.b2c_stripped;   // b2cgltf lod.strip_lods only removes levels: the rig is whole
  if (stripped && !(Array.isArray(stripped) && stripped.every(x => x?.what?.startsWith('LOD levels'))))
    throw new Error('this is a stripped distribution file (SPEC R8): it has no rig to play');
  const rig = json.extensions?.[RIG];
  if (rig) version(rig, RIG);

  // the splat: the current one or a level of its chain (in no scene; posed and placed as the current splat is, 7.1a)
  const base = currentSplat(json);
  if (rig && base !== rig.splatNode) throw new Error(`the current splat (node ${base}) is not b2crig's rigged splat (node ${rig.splatNode})`);
  const { lods, problem: lodProblem } = await lodChain(json, acc, base);
  if (!lods[level]) level = 0;   // e.g. ?lod= from another subject
  const si = lods[level].node, prim = splatPrim(json, si), at = prim.attributes;
  const be = prim.extensions?.[BIND];
  if (rig && !be) throw new Error(`the rigged splat${level ? `'s LOD level ${level}` : ''} has no B2CRIG_splat_cage binding`);
  if (be) version(be, BIND);
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
  const lod = { lods, lodProblem, level };
  if (!rig) return { id, splat, binding: null, cage: null, skeleton: null, rigHash: null, render: {}, W: worldMatrix(json, base), ...lod };
  const binding = { face: Int32Array.from(acc(be.face)),   // 0xFFFFFFFF (unbound) wraps to -1
                    bary: Float32Array.from(acc(be.barycentric)), offset: Float32Array.from(acc(be.offset)), posing: be.posing };
  return { id, splat, binding, ...await rigOf(json, acc, rig), ...lod };
}

const splatPrim = (json, i) => json.nodes[i].mesh === undefined ? undefined
  : json.meshes[json.nodes[i].mesh].primitives.find(p => p.extensions?.[KHR]);

// SPEC 4.7 / 7.1a: the chain behind the splat node `base`, [{ node, name, count, maxScreenHeightPx }], most detailed
// first (the base's height null). The base alone when it has no chain, or when the chain is stale or broken (a
// subset of SPEC 8's checks: what this viewer relies on); `problem` then says why.
async function lodChain(json, acc, base) {
  const count = i => json.accessors[splatPrim(json, i).attributes.POSITION].count;
  const lods = [{ node: base, name: json.nodes[base].name || `node ${base}`, count: count(base), maxScreenHeightPx: null }];
  const ext = json.nodes[base].extensions || {};
  if (!ext[LOD]) return { lods, problem: null };
  const ids = ext[LOD].ids || [], rec = ext.B2CRIG_lod ?? ext.B2C_lod, problems = [];
  if (!rec) problems.push('MSFT_lod but no B2C_lod / B2CRIG_lod record');
  else if (rec.version !== 1) problems.push(`level record version ${rec.version} (this viewer reads 1)`);
  else {
    const lv = rec.levels || [];
    if (lv.length !== ids.length || lv.some((r, k) => r.node !== ids[k])) problems.push("the level record's nodes do not match MSFT_lod.ids");
    let prev = lods[0].count, hPrev = Infinity;
    for (const r of lv) {
      if (!json.nodes[r.node] || !splatPrim(json, r.node)) { problems.push(`level node ${r.node} is not a splat`); continue; }
      const c = count(r.node);
      if (c !== r.count) problems.push(`level node ${r.node}: record count ${r.count}, splat ${c}`);
      if (c >= prev) problems.push(`level node ${r.node}: ${c} splats, not fewer than the ${prev} before it`);
      if (!(r.maxScreenHeightPx > 0 && r.maxScreenHeightPx < hPrev)) problems.push('maxScreenHeightPx does not strictly decrease');
      prev = c; hPrev = r.maxScreenHeightPx;
    }
    if (!problems.length) {
      const src = await sourceHash(json, acc, base);
      if (lv.some(r => r.source !== src)) problems.push(`stale: a level's source is not the base's (${src.slice(0, 19)}…)`);
    }
    if (!problems.length) for (const r of lv)
      lods.push({ node: r.node, name: json.nodes[r.node].name || `node ${r.node}`, count: r.count, maxScreenHeightPx: r.maxScreenHeightPx });
  }
  return { lods, problem: problems.length ? problems.join('; ') : null };
}

// SPEC 4.7 `source`: SHA-256 over the splat's POSITION, ROTATION, SCALE, OPACITY, then SH_DEGREE_d_COEF_n, as stored
// (little-endian float32, no stride padding). "sha256:" + lowercase hex.
async function sourceHash(json, acc, node) {
  const at = splatPrim(json, node).attributes, keys = ['POSITION', `${KHR}:ROTATION`, `${KHR}:SCALE`, `${KHR}:OPACITY`];
  for (let d = 0; at[`${KHR}:SH_DEGREE_${d}_COEF_0`] !== undefined; d++)
    for (let c = 0; c <= 2 * d; c++) keys.push(`${KHR}:SH_DEGREE_${d}_COEF_${c}`);
  return 'sha256:' + await digest(keys.map(k => {
    const a = acc(at[k]);
    if (!(a instanceof Float32Array)) throw new Error(`LOD source hash: ${k} is not float32`);
    return a;
  }));
}

// SHA-256 of the concatenated bytes of typed arrays, lowercase hex
async function digest(arrays) {
  const all = new Uint8Array(arrays.reduce((s, a) => s + a.byteLength, 0));
  let o = 0; for (const a of arrays) { all.set(new Uint8Array(a.buffer, a.byteOffset, a.byteLength), o); o += a.byteLength; }
  const h = globalThis.crypto?.subtle ? new Uint8Array(await crypto.subtle.digest('SHA-256', all)) : sha256(all);
  return Array.from(h, b => b.toString(16).padStart(2, '0')).join('');
}

// A web subject: a plain SOG is a splat alone (no b2c_id, no rig). PlayCanvas shows a SOG turned 180 degrees about
// z, and SOG files are written for that (b2csite's splats: tools/build.py), so W is that turn. b2cgltf's web subject
// adds the binding and rig.glb, the subject file stripped to its skeleton, skin and cage, whose root carries W.
async function parseWebSubject(files, decode) {
  const s = await parseSog(files, decode);
  if (!s.rig) return { id: null, splat: s.splat, binding: null, cage: null, skeleton: null, rigHash: null, render: {},
                       W: new Float64Array([-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]) };
  const { json, acc } = readGlb(s.rig);
  const id = json.asset?.extras?.b2c_id, rig = json.extensions?.[RIG];
  if (!id || !rig) throw new Error('the .sog\'s rig.glb is not a b2c rig (no b2c_id or B2CRIG_rig)');
  version(rig, RIG);
  return { id, splat: s.splat, binding: s.binding, ...await rigOf(json, acc, rig) };
}

// The cage, the skeleton, the rig hash and b2ctrain's render settings of a rigged file (B2CRIG_rig: rig).
async function rigOf(json, acc, rig) {
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
  return { cage, skeleton, rigHash, render: rig.render || {}, W: skeleton.W };
}

// B2CRIG_clip.rig.cageSha256 (SPEC 6), as b2cgltf rig.rig_hash: per cage primitive its POSITION, indices, then
// JOINTS_n / WEIGHTS_n (n ascending, JOINTS first); then the inverse bind matrices (row-major, as numpy holds them),
// rest rotations and rest translations, all float32 little-endian.
async function hashRig(acc, prims, sk) {
  const parts = [];
  for (const p of prims) {
    const at = p.attributes;
    const skin = Object.keys(at).filter(k => /^(JOINTS|WEIGHTS)_\d+$/.test(k))
      .sort((a, b) => (+a.split('_')[1] - +b.split('_')[1]) || (a < b ? -1 : 1));
    parts.push(acc(at.POSITION), acc(p.indices), ...skin.map(k => acc(at[k])));
  }
  const J = sk.joints.length, ibmRows = new Float32Array(J * 16);
  for (let j = 0; j < J; j++) for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) ibmRows[j * 16 + r * 4 + c] = sk.ibm[j * 16 + c * 4 + r];
  parts.push(ibmRows, sk.restQ, sk.restT);
  return digest(parts);
}

// SHA-256 (FIPS 180-4) for pages served over plain HTTP from a non-localhost address: browsers only expose
// crypto.subtle in secure contexts, and b2crunner on a pod is reached that way.
function sha256(msg) {
  const K = new Uint32Array([
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2]);
  const H = new Uint32Array([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]);
  const n = msg.length, padded = new Uint8Array(((n + 9 + 63) >> 6) << 6);
  padded.set(msg); padded[n] = 0x80;
  const dv = new DataView(padded.buffer);
  dv.setUint32(padded.length - 8, Math.floor(n / 0x20000000)); dv.setUint32(padded.length - 4, n * 8 >>> 0);
  const W = new Uint32Array(64), rotr = (x, r) => (x >>> r) | (x << (32 - r));
  for (let o = 0; o < padded.length; o += 64) {
    for (let i = 0; i < 16; i++) W[i] = dv.getUint32(o + i * 4);
    for (let i = 16; i < 64; i++) {
      const a = W[i - 15], b = W[i - 2];
      W[i] = W[i - 16] + (rotr(a, 7) ^ rotr(a, 18) ^ (a >>> 3)) + W[i - 7] + (rotr(b, 17) ^ rotr(b, 19) ^ (b >>> 10));
    }
    let [a, b, c, d, e, f, g, h] = H;
    for (let i = 0; i < 64; i++) {
      const t1 = h + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + W[i] | 0;
      const t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c)) | 0;
      h = g; g = f; f = e; e = d + t1 | 0; d = c; c = b; b = a; a = t1 + t2 | 0;
    }
    H[0] += a; H[1] += b; H[2] += c; H[3] += d; H[4] += e; H[5] += f; H[6] += g; H[7] += h;
  }
  const out = new Uint8Array(32), ov = new DataView(out.buffer);
  for (let i = 0; i < 8; i++) ov.setUint32(i * 4, H[i]);
  return out;
}

// ---------------------------------------------------------------- clip files

// A clip file (.clip.glb, or its web form .clip.zip) for `subject` (parseSubject's result), refused as SPEC 6 says a
// player must. Joint TRS come out as q [T * J * 4] (x y z w), t [T * J * 3]; joints without a channel keep their
// rest pose. The residual comes out as { data: integer array [T * nv * 3], unit: metres per step, min: lowest integer }.
export async function parseClip(buf, subject) {
  const files = isZip(buf) ? await readZip(buf) : null;
  if (files && !files.has('clip.glb')) throw new Error('not a b2c clip archive (no clip.glb)');
  const { json, acc } = readGlb(files ? files.get('clip.glb').slice().buffer : buf);
  const c = json.extensions?.[CLIP];
  if (!c) throw new Error('not a clip file (no B2CRIG_clip)');
  if (!subject.rigHash) throw new Error('the subject file has no rig yet, so it has no clips (b2crig: tools/export_gltf.py rig)');
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
    if (re.file !== undefined) residual = webResidual(files, re);
    else residual = { data: acc(re.accessor), unit: re.scale / 32767, min: -32767 };   // normalised SHORT: max(c / 32767, -1)
  }
  return { name: c.name, fps: c.fps, nframes: T, q, t, residual };
}

// The web clip's residual (b2cgltf web.py): int16 frame deltas (frame 0 absolute) in units of `step` metres, byte
// planes split (every low byte, then every high byte).
function webResidual(files, re) {
  const raw = files?.get(re.file);
  if (!raw) throw new Error(`the clip names its residual ${re.file}, which the archive does not contain`);
  const n = re.frames * re.vertices * 3;
  if (raw.length !== 2 * n) throw new Error(`${re.file} has ${raw.length} bytes, ${2 * n} expected`);
  const out = new Int32Array(n), per = re.vertices * 3;
  for (let i = 0; i < n; i++) {
    const d = (raw[i] | (raw[n + i] << 8)) << 16 >> 16;
    out[i] = i < per ? d : out[i - per] + d;
  }
  return { data: out, unit: re.step, min: -Infinity };
}

// SPEC 7.2 steps 1-2 for frame f: skin the cage with W^-1 . joint_world . IBM (the joints relative to the skeleton
// root), then add the residual (its integers, floored at min, x unit). Returns Float32Array [nv * 3] in the b2crunner frame.
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
  const R = residual && clip.residual, rq = R ? R.data : null, rs = R ? R.unit : 0, rmin = R ? R.min : 0, ro = f * nv * 3;
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
    if (rq) {
      const r = ro + v * 3;
      px += Math.max(rq[r], rmin) * rs; py += Math.max(rq[r + 1], rmin) * rs; pz += Math.max(rq[r + 2], rmin) * rs;
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
