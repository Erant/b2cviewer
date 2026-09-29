// Readers for the three files an animated b2crig subject needs (layouts documented in b2ctrain):
//   scene*.ply     the canonical splat (b2ctrain src/ply.cpp; seg_label / seg_conf pick each splat's cage layer)
//   *.b2ccage      cage layers + per-frame posed cage vertices (src/gpu/cage.h, written by b2crig/b2ctrain.py)
//   *.app          the pose-dependent appearance MLP (src/gpu/cage_app.h, B2CAPP01..03)

export const APP_FEAT = 8, APP_LAT = 8, APP_IN = APP_FEAT + APP_LAT, APP_HID = 32, APP_OUT = 9;
export const O_W1 = 0, O_B1 = O_W1 + APP_HID * APP_IN, O_W2 = O_B1 + APP_HID, O_B2 = O_W2 + APP_HID * APP_HID,
  O_W3 = O_B2 + APP_HID, O_B3 = O_W3 + APP_OUT * APP_HID, APP_NP = O_B3 + APP_OUT;

const PLY_TYPES = {
  float: ['getFloat32', 4], float32: ['getFloat32', 4], double: ['getFloat64', 8], float64: ['getFloat64', 8],
  uchar: ['getUint8', 1], uint8: ['getUint8', 1], char: ['getInt8', 1], int8: ['getInt8', 1],
  ushort: ['getUint16', 2], uint16: ['getUint16', 2], short: ['getInt16', 2], int16: ['getInt16', 2],
  uint: ['getUint32', 4], uint32: ['getUint32', 4], int: ['getInt32', 4], int32: ['getInt32', 4],
};

// Canonical splat: pos [n*3], opacity logit [n], quat (w, x, y, z) [n*4], log scales [n*3],
// sh [n*K*3] (coefficient-major per splat: k0 = DC, then f_rest), labels [n*2] (seg_label, seg_conf) or null.
export function parsePly(buf) {
  const bytes = new Uint8Array(buf);
  // b2crunner headers carry the body rig and orbit record as comments (>100 KB), so search the bytes for end_header
  // (the body offset must be a byte offset, not a decoded-string index)
  const marker = new TextEncoder().encode('end_header\n');
  let end = -1;
  for (let i = bytes.indexOf(marker[0]); i >= 0 && i <= bytes.length - marker.length; i = bytes.indexOf(marker[0], i + 1)) {
    let k = 1;
    while (k < marker.length && bytes[i + k] === marker[k]) k++;
    if (k === marker.length) { end = i; break; }
  }
  const head = new TextDecoder().decode(bytes.subarray(0, Math.max(end, 0)));
  if (!head.startsWith('ply') || end < 0) throw new Error('not a PLY file');
  const body = end + marker.length;
  let n = 0, fmt = '', inVertex = false;
  const props = [];
  for (const line of head.slice(0, end).split('\n')) {
    const w = line.trim().split(/\s+/);
    if (w[0] === 'format') fmt = w[1];
    else if (w[0] === 'element') { inVertex = w[1] === 'vertex'; if (inVertex) n = parseInt(w[2]); }
    else if (w[0] === 'property' && inVertex) {
      if (w[1] === 'list') throw new Error('PLY vertex list properties are not supported');
      const t = PLY_TYPES[w[1]];
      if (!t) throw new Error(`PLY property type ${w[1]} is not supported`);
      props.push({ name: w[2], get: t[0], size: t[1] });
    }
  }
  if (fmt !== 'binary_little_endian') throw new Error(`PLY format ${fmt} is not supported (binary_little_endian only)`);
  let stride = 0;
  const off = {};
  for (const p of props) { off[p.name] = { o: stride, get: p.get }; stride += p.size; }
  for (const k of ['x', 'y', 'z']) if (!off[k]) throw new Error(`PLY has no ${k}`);
  let nRest = 0; while (off[`f_rest_${nRest}`]) nRest++;
  const K = nRest / 3 + 1, degree = Math.round(Math.sqrt(K)) - 1;
  if ((degree + 1) * (degree + 1) !== K) throw new Error(`PLY: ${nRest} f_rest properties is not an SH layout`);

  const allFloat = props.every(p => p.get === 'getFloat32');
  let col;   // col(name) -> function(i) reading one property
  if (allFloat && body % 4 === 0) {
    const f = new Float32Array(buf, body, n * stride / 4);
    col = name => { const o = off[name].o / 4, s = stride / 4; return i => f[i * s + o]; };
  } else if (allFloat) {
    const f = new Float32Array(buf.slice(body, body + n * stride));
    col = name => { const o = off[name].o / 4, s = stride / 4; return i => f[i * s + o]; };
  } else {
    const dv = new DataView(buf, body);
    col = name => { const { o, get } = off[name]; return i => dv[get](i * stride + o, true); };
  }
  const opt = (name, dflt) => off[name] ? col(name) : () => dflt;

  const pos = new Float32Array(n * 3), op = new Float32Array(n), quat = new Float32Array(n * 4), ls = new Float32Array(n * 3);
  const sh = new Float32Array(n * K * 3);
  const X = col('x'), Y = col('y'), Z = col('z'), O = opt('opacity', 10);
  const S = [0, 1, 2].map(k => opt(`scale_${k}`, Math.log(0.01)));
  const Q = [0, 1, 2, 3].map(k => opt(`rot_${k}`, k === 0 ? 1 : 0));
  const DC = [0, 1, 2].map(k => opt(`f_dc_${k}`, 0));
  const R = []; for (let k = 0; k < nRest; k++) R.push(col(`f_rest_${k}`));
  const hasLabels = !!off.seg_label;
  const labels = hasLabels ? new Float32Array(n * 2) : null;
  const L = opt('seg_label', -1), C = opt('seg_conf', 1);
  for (let i = 0; i < n; i++) {
    pos[i * 3] = X(i); pos[i * 3 + 1] = Y(i); pos[i * 3 + 2] = Z(i);
    op[i] = O(i);
    for (let k = 0; k < 3; k++) ls[i * 3 + k] = S[k](i);
    for (let k = 0; k < 4; k++) quat[i * 4 + k] = Q[k](i);
    const b = i * K * 3;
    for (let c = 0; c < 3; c++) sh[b + c] = DC[c](i);
    // f_rest is channel-major: all R rest coefficients, then G, then B
    for (let c = 0; c < 3; c++) for (let k = 1; k < K; k++) sh[b + k * 3 + c] = R[c * (K - 1) + (k - 1)](i);
    if (labels) { labels[i * 2] = L(i); labels[i * 2 + 1] = C(i); }
  }
  return { n, degree, K, pos, op, quat, ls, sh, labels };
}

// B2CCAGE1: int32 n_layers, n_verts, n_faces, n_frames; per layer int32 v_off, v_count, f_off, f_count, uint32 class_bits;
// float3 verts[nv]; int3 faces[nf]; char names[n_frames][64]; float3 posed[n_frames][nv].
export function parseCage(buf) {
  const dv = new DataView(buf);
  const magic = new TextDecoder().decode(new Uint8Array(buf, 0, 8));
  if (magic !== 'B2CCAGE1') throw new Error('not a b2ctrain cage v1 (bad magic)');
  const nl = dv.getInt32(8, true), nv = dv.getInt32(12, true), nf = dv.getInt32(16, true), nframes = dv.getInt32(20, true);
  if (nl <= 0 || nl > 32 || nv <= 0 || nf <= 0 || nframes <= 0) throw new Error('cage: unsupported layout');
  let o = 24;
  const layers = [];
  for (let l = 0; l < nl; l++, o += 20)
    layers.push({ vOff: dv.getInt32(o, true), vCount: dv.getInt32(o + 4, true), fOff: dv.getInt32(o + 8, true),
                  fCount: dv.getInt32(o + 12, true), classBits: dv.getUint32(o + 16, true) });
  const need = o + nv * 12 + nf * 12 + nframes * 64 + nframes * nv * 12;
  if (buf.byteLength < need) throw new Error('cage is truncated');
  const verts0 = new Float32Array(buf, o, nv * 3); o += nv * 12;
  const faces = new Int32Array(buf, o, nf * 3); o += nf * 12;
  const names = [];
  const dec = new TextDecoder();
  for (let i = 0; i < nframes; i++, o += 64) {
    const b = new Uint8Array(buf, o, 64);
    const z = b.indexOf(0);
    names.push(dec.decode(z < 0 ? b : b.subarray(0, z)));
  }
  const posed = new Float32Array(buf, o, nframes * nv * 3);
  for (const L of layers)
    if (L.vOff < 0 || L.vCount <= 0 || L.vOff + L.vCount > nv || L.fOff < 0 || L.fCount <= 0 || L.fOff + L.fCount > nf)
      throw new Error('cage: layer range out of bounds');
  return { nv, nf, nframes, layers, verts0, faces, names, posed };
}

// B2CAPP0x: int32 nv, n_feat, n_lat, n_hid, n_out; (v2: float dz, max_do, max_ds; v3: + max_dp); float params[]; float z[nv][n_lat].
// Older files carry fewer features (6) / outputs (6): expanded into the current layout with zero weights, as CageApp::load does.
export function parseApp(buf) {
  const dv = new DataView(buf);
  const magic = new TextDecoder().decode(new Uint8Array(buf, 0, 8));
  if (!/^B2CAPP0[123]$/.test(magic)) throw new Error('not a b2ctrain cage appearance model');
  const h = [0, 1, 2, 3, 4].map(k => dv.getInt32(8 + 4 * k, true));
  const [nv, nfeat, nlat, nhid, nout] = h;
  const lim = [0, 1e30, 1, 0];
  const nlim = magic[7] === '3' ? 4 : magic[7] === '2' ? 3 : 0;
  let o = 28;
  for (let k = 0; k < nlim; k++, o += 4) lim[k] = dv.getFloat32(o, true);
  if (nfeat > APP_FEAT || nfeat < 6 || nlat !== APP_LAT || nhid !== APP_HID || nout > APP_OUT || nout < 6)
    throw new Error(`app layout (${h.join('/')}) is not supported`);
  const inF = nfeat + APP_LAT;
  const w1F = APP_HID * inF, w3F = w1F + APP_HID + APP_HID * APP_HID + APP_HID;
  const npFile = w3F + nout * APP_HID + nout;
  if (buf.byteLength < o + 4 * (npFile + nv * APP_LAT)) throw new Error('app file is truncated');
  const fp = new Float32Array(buf.slice(o, o + 4 * npFile)); o += 4 * npFile;
  const Z = new Float32Array(buf.slice(o, o + 4 * nv * APP_LAT));
  const P = new Float32Array(APP_NP);
  for (let hh = 0; hh < APP_HID; hh++) {
    for (let d = 0; d < nfeat; d++) P[O_W1 + hh * APP_IN + d] = fp[hh * inF + d];
    for (let d = 0; d < APP_LAT; d++) P[O_W1 + hh * APP_IN + APP_FEAT + d] = fp[hh * inF + nfeat + d];
  }
  P.set(fp.subarray(w1F, w3F + nout * APP_HID), O_B1);   // B1, W2, B2 and the file's W3 rows
  P.set(fp.subarray(w3F + nout * APP_HID), O_B3);
  return { version: magic, nv, nfeat, nout, dz: lim[0], maxDo: lim[1], maxDs: lim[2], maxDp: lim[3], P, Z };
}
