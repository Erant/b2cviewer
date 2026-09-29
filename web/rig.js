// The cage rig of b2ctrain (src/gpu/cage.cu, src/gpu/cage_app.cu), ported to JS for the viewer:
//   bind      every splat to a cage triangle (nearest vertex of its layer, then the nearest triangle around it)
//   frame     per posed cage frame: the triangle frames (-> face texture), the appearance MLP per cage vertex
//             (-> vertex texture) and the posed splat centres (depth sorting)
// The per-splat posing itself (position, rotation, scale, SH frame, MLP outputs) runs in the vertex shader
// (renderer.js) from the face / vertex textures, so it follows pose_bound() and splat_apply_kernel() there.
import { APP_FEAT, APP_LAT, APP_IN, APP_HID, APP_OUT, O_W1, O_B1, O_W2, O_B2, O_W3, O_B3 } from './formats.js';

export const SPLAT_TEXELS = 5;   // t0 pos.xyz op | t1 quat wxyz | t2 lscale.xyz face | t3 off.xyz b1 | t4 b2
export const FACE_TEXELS = 7;    // a.xyz ia | b.xyz ib | c.xyz ic | R row0, k | R row1, k0 | R row2, ratio | dq wxyz
export const VERT_TEXELS = 3;    // the 9 MLP outputs R (rest-subtracted where cage_app.h says)

const RING_MAX = 32;
const LOG_S_MAX = 2;
const OCC_R1 = 0.04, OCC_R2 = 0.08;

// Triangle frame (cage.cu tri_frame): R row-major with columns (e1, e2, n); returns k = sqrt(2 area), 0 when degenerate.
function triFrame(V, ia, ib, ic, R) {
  const ax = V[ia * 3], ay = V[ia * 3 + 1], az = V[ia * 3 + 2];
  const ux = V[ib * 3] - ax, uy = V[ib * 3 + 1] - ay, uz = V[ib * 3 + 2] - az;
  const wx = V[ic * 3] - ax, wy = V[ic * 3 + 1] - ay, wz = V[ic * 3 + 2] - az;
  const nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx;
  const nl = Math.hypot(nx, ny, nz), ul = Math.hypot(ux, uy, uz);
  if (!(nl > 1e-20) || !(ul > 1e-12)) { R.fill(0); R[0] = R[4] = R[8] = 1; return 0; }
  const e1x = ux / ul, e1y = uy / ul, e1z = uz / ul, n0 = nx / nl, n1 = ny / nl, n2 = nz / nl;
  const e2x = n1 * e1z - n2 * e1y, e2y = n2 * e1x - n0 * e1z, e2z = n0 * e1y - n1 * e1x;
  R[0] = e1x; R[1] = e2x; R[2] = n0;
  R[3] = e1y; R[4] = e2y; R[5] = n1;
  R[6] = e1z; R[7] = e2z; R[8] = n2;
  return Math.sqrt(nl);
}

// Rotation matrix (row-major) -> quaternion (w, x, y, z), as cage.cu mat_to_quat.
function matToQuat(m, out, o) {
  const tr = m[0] + m[4] + m[8];
  let w, x, y, z;
  if (tr > 0) { const s = Math.sqrt(tr + 1) * 2; w = 0.25 * s; x = (m[7] - m[5]) / s; y = (m[2] - m[6]) / s; z = (m[3] - m[1]) / s; }
  else if (m[0] > m[4] && m[0] > m[8]) { const s = Math.sqrt(1 + m[0] - m[4] - m[8]) * 2; w = (m[7] - m[5]) / s; x = 0.25 * s; y = (m[1] + m[3]) / s; z = (m[2] + m[6]) / s; }
  else if (m[4] > m[8]) { const s = Math.sqrt(1 + m[4] - m[0] - m[8]) * 2; w = (m[2] - m[6]) / s; x = (m[1] + m[3]) / s; y = 0.25 * s; z = (m[5] + m[7]) / s; }
  else { const s = Math.sqrt(1 + m[8] - m[0] - m[4]) * 2; w = (m[3] - m[1]) / s; x = (m[2] + m[6]) / s; y = (m[5] + m[7]) / s; z = 0.25 * s; }
  out[o] = w; out[o + 1] = x; out[o + 2] = y; out[o + 3] = z;
}

// Closest point on triangle (Ericson 5.1.5), as cage.cu closest_on_tri: sets bc[0..1] = (b1, b2), returns the squared distance.
function closestOnTri(px, py, pz, V, ia, ib, ic, bc) {
  const ax = V[ia * 3], ay = V[ia * 3 + 1], az = V[ia * 3 + 2];
  const bx = V[ib * 3], by = V[ib * 3 + 1], bz = V[ib * 3 + 2];
  const cx = V[ic * 3], cy = V[ic * 3 + 1], cz = V[ic * 3 + 2];
  const abx = bx - ax, aby = by - ay, abz = bz - az, acx = cx - ax, acy = cy - ay, acz = cz - az;
  const apx = px - ax, apy = py - ay, apz = pz - az;
  const d1 = abx * apx + aby * apy + abz * apz, d2 = acx * apx + acy * apy + acz * apz;
  let b1, b2;
  if (d1 <= 0 && d2 <= 0) { b1 = 0; b2 = 0; }
  else {
    const bpx = px - bx, bpy = py - by, bpz = pz - bz;
    const d3 = abx * bpx + aby * bpy + abz * bpz, d4 = acx * bpx + acy * bpy + acz * bpz;
    const cpx = px - cx, cpy = py - cy, cpz = pz - cz;
    const d5 = abx * cpx + aby * cpy + abz * cpz, d6 = acx * cpx + acy * cpy + acz * cpz;
    const vc = d1 * d4 - d3 * d2, vb = d5 * d2 - d1 * d6, va = d3 * d6 - d5 * d4;
    if (d3 >= 0 && d4 <= d3) { b1 = 1; b2 = 0; }
    else if (d6 >= 0 && d5 <= d6) { b1 = 0; b2 = 1; }
    else if (vc <= 0 && d1 >= 0 && d3 <= 0) { b1 = d1 / (d1 - d3); b2 = 0; }
    else if (vb <= 0 && d2 >= 0 && d6 <= 0) { b1 = 0; b2 = d2 / (d2 - d6); }
    else if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) { const t = (d4 - d3) / ((d4 - d3) + (d5 - d6)); b1 = 1 - t; b2 = t; }
    else { const den = 1 / (va + vb + vc); b1 = vb * den; b2 = vc * den; }
  }
  const qx = ax + abx * b1 + acx * b2 - px, qy = ay + aby * b1 + acy * b2 - py, qz = az + abz * b1 + acz * b2 - pz;
  bc[0] = b1; bc[1] = b2;
  return qx * qx + qy * qy + qz * qz;
}

// Uniform grid over a subset of points (CSR cells; each cell's items in ascending index order).
class Grid {
  constructor(P, ids, h) {
    this.P = P; this.h = h;
    let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const i of ids) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], P[i * 3 + k]); hi[k] = Math.max(hi[k], P[i * 3 + k]); }
    if (!ids.length) { lo = [0, 0, 0]; hi = [0, 0, 0]; }
    this.lo = lo;
    this.dim = hi.map((x, k) => Math.max(1, Math.floor((x - lo[k]) / h) + 1));
    const [dx, dy, dz] = this.dim, nc = dx * dy * dz;
    const cell = new Int32Array(ids.length), start = new Int32Array(nc + 1);
    ids.forEach((i, j) => { const c = this.cellOf(P[i * 3], P[i * 3 + 1], P[i * 3 + 2]); cell[j] = c; start[c + 1]++; });
    for (let c = 0; c < nc; c++) start[c + 1] += start[c];
    const fill = start.slice(0, nc), items = new Int32Array(ids.length);
    ids.forEach((i, j) => { items[fill[cell[j]]++] = i; });   // ids ascending -> each cell ascending
    this.start = start; this.items = items;
  }
  coord(x, k) { return Math.min(this.dim[k] - 1, Math.max(0, Math.floor((x - this.lo[k]) / this.h))); }
  cellOf(x, y, z) { return (this.coord(z, 2) * this.dim[1] + this.coord(y, 1)) * this.dim[0] + this.coord(x, 0); }
  // Nearest item to p (ties -> lowest index, as the CUDA scan in index order); -1 when the grid is empty.
  nearest(px, py, pz) {
    if (!this.items.length) return -1;
    const { P, h, dim, start, items } = this;
    const cx = this.coord(px, 0), cy = this.coord(py, 1), cz = this.coord(pz, 2);
    const rmax = Math.max(dim[0], dim[1], dim[2]);
    let best = Infinity, bi = -1;
    for (let r = 0; r <= rmax; r++) {
      for (let z = cz - r; z <= cz + r; z++) {
        if (z < 0 || z >= dim[2]) continue;
        for (let y = cy - r; y <= cy + r; y++) {
          if (y < 0 || y >= dim[1]) continue;
          const shell = z === cz - r || z === cz + r || y === cy - r || y === cy + r;
          for (let x = cx - r; x <= cx + r; x += (shell || r === 0) ? 1 : 2 * r) {
            if (x < 0 || x >= dim[0]) continue;
            const c = (z * dim[1] + y) * dim[0] + x;
            for (let j = start[c]; j < start[c + 1]; j++) {
              const i = items[j];
              const dx = P[i * 3] - px, dy = P[i * 3 + 1] - py, dz = P[i * 3 + 2] - pz, d2 = dx * dx + dy * dy + dz * dz;
              if (d2 < best || (d2 === best && i < bi)) { best = d2; bi = i; }
            }
          }
        }
      }
      // every unvisited cell is at least r * h away from p
      if (bi >= 0 && best <= (r * h) * (r * h)) break;
    }
    return bi;
  }
  // Number of items strictly within sqrt(r2a) and sqrt(r2b) (r2b >= r2a, the cell size >= sqrt(r2b)).
  count2(px, py, pz, r2a, r2b, out) {
    const { P, dim, start, items } = this;
    const cx = this.coord(px, 0), cy = this.coord(py, 1), cz = this.coord(pz, 2);
    let n1 = 0, n2 = 0;
    for (let z = Math.max(0, cz - 1); z <= Math.min(dim[2] - 1, cz + 1); z++)
      for (let y = Math.max(0, cy - 1); y <= Math.min(dim[1] - 1, cy + 1); y++)
        for (let x = Math.max(0, cx - 1); x <= Math.min(dim[0] - 1, cx + 1); x++) {
          const c = (z * dim[1] + y) * dim[0] + x;
          for (let j = start[c]; j < start[c + 1]; j++) {
            const i = items[j];
            const dx = P[i * 3] - px, dy = P[i * 3 + 1] - py, dz = P[i * 3 + 2] - pz, d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < r2b) { n2++; if (d2 < r2a) n1++; }
          }
        }
    out[0] = n1; out[1] = n2;
  }
}

export class Rig {
  // splat: parsePly output; cage: parseCage output; minConf as b2ctrain render --cage-min-conf.
  // binding: b2ctrain's own binding { face, bary, offset } (a .glb carries it, gltf.js), used instead of bind().
  constructor(splat, cage, { minConf = 0.5, progress = () => {}, binding = null } = {}) {
    this.splat = splat; this.cage = cage; this.app = null;
    const { nv, nf, faces, verts0 } = cage;
    // vertex -> incident faces (CSR, ascending face order as CageRig::bind builds it)
    const off = new Int32Array(nv + 1);
    for (let f = 0; f < nf * 3; f++) off[faces[f] + 1]++;
    for (let k = 0; k < nv; k++) off[k + 1] += off[k];
    const idx = new Int32Array(off[nv]), fill = off.slice(0, nv);
    for (let f = 0; f < nf; f++) for (let j = 0; j < 3; j++) idx[fill[faces[f * 3 + j]]++] = f;
    this.vfOff = off; this.vfIdx = idx;
    // canonical triangle frames
    this.k0 = new Float32Array(nf); this.q0 = new Float32Array(nf * 4);
    const R = new Float64Array(9);
    for (let f = 0; f < nf; f++) {
      this.k0[f] = triFrame(verts0, faces[f * 3], faces[f * 3 + 1], faces[f * 3 + 2], R);
      matToQuat(R, this.q0, f * 4);
    }
    if (binding) this.useBinding(binding);
    else this.bind(minConf, progress);
  }

  useBinding({ face, bary, offset }) {
    if (face.length !== this.splat.n) throw new Error(`the binding has ${face.length} splats, the splat ${this.splat.n}`);
    this.bindF = face; this.bindB = bary; this.bindOff = offset;
    let unbound = 0; for (let i = 0; i < face.length; i++) if (face[i] < 0) unbound++;
    this.nFallback = 0; this.nUnbound = unbound;
  }

  bind(minConf, progress) {
    const { splat, cage } = this, { n, pos, labels } = splat, { nv, layers, faces, verts0 } = cage;
    const vertLayer = new Int32Array(nv);   // 0 unless a layer claims it, -1 when no face uses it
    layers.forEach((L, l) => vertLayer.fill(l, L.vOff, L.vOff + L.vCount));
    for (let k = 0; k < nv; k++) if (this.vfOff[k + 1] === this.vfOff[k]) vertLayer[k] = -1;
    const h = 0.03;
    const all = [], per = layers.map(() => []);
    for (let k = 0; k < nv; k++) if (vertLayer[k] >= 0) { all.push(k); per[vertLayer[k]].push(k); }
    const gAll = new Grid(verts0, all, h), gLayer = per.map(ids => new Grid(verts0, ids, h));
    const bindF = new Int32Array(n), bindB = new Float32Array(n * 2), bindOff = new Float32Array(n * 3);
    let fallback = 0, unbound = 0;
    const bc = [0, 0], R = new Float64Array(9);
    for (let i = 0; i < n; i++) {
      if ((i & 0xffff) === 0) progress(i / n);
      let want = -1;
      if (labels) {
        const c = Math.round(labels[i * 2]), conf = labels[i * 2 + 1];
        if (c >= 0 && c < 32 && conf >= minConf)
          for (let l = 0; l < layers.length; l++) if ((layers[l].classBits >>> c) & 1) { want = l; break; }
      }
      if (want < 0) fallback++;
      const px = pos[i * 3], py = pos[i * 3 + 1], pz = pos[i * 3 + 2];
      const bi = (want >= 0 ? gLayer[want] : gAll).nearest(px, py, pz);
      let bf = -1, bb1 = 0, bb2 = 0, bd = Infinity;
      if (bi >= 0) {
        const e = Math.min(this.vfOff[bi + 1], this.vfOff[bi] + RING_MAX);
        for (let k = this.vfOff[bi]; k < e; k++) {
          const f = this.vfIdx[k];
          const d2 = closestOnTri(px, py, pz, verts0, faces[f * 3], faces[f * 3 + 1], faces[f * 3 + 2], bc);
          if (d2 < bd) { bd = d2; bf = f; bb1 = bc[0]; bb2 = bc[1]; }
        }
      }
      bindF[i] = bf; bindB[i * 2] = bb1; bindB[i * 2 + 1] = bb2;
      if (bf < 0) { unbound++; continue; }
      // offset from the foot point in the canonical triangle frame, / k0 (offset_kernel)
      const ia = faces[bf * 3], ib = faces[bf * 3 + 1], ic = faces[bf * 3 + 2];
      triFrame(verts0, ia, ib, ic, R);
      const w0 = 1 - bb1 - bb2;
      const dx = px - (verts0[ia * 3] * w0 + verts0[ib * 3] * bb1 + verts0[ic * 3] * bb2);
      const dy = py - (verts0[ia * 3 + 1] * w0 + verts0[ib * 3 + 1] * bb1 + verts0[ic * 3 + 1] * bb2);
      const dz = pz - (verts0[ia * 3 + 2] * w0 + verts0[ib * 3 + 2] * bb1 + verts0[ic * 3 + 2] * bb2);
      const kk = this.k0[bf] > 0 ? 1 / this.k0[bf] : 0;
      bindOff[i * 3] = (R[0] * dx + R[3] * dy + R[6] * dz) * kk;
      bindOff[i * 3 + 1] = (R[1] * dx + R[4] * dy + R[7] * dz) * kk;
      bindOff[i * 3 + 2] = (R[2] * dx + R[5] * dy + R[8] * dz) * kk;
    }
    this.bindF = bindF; this.bindB = bindB; this.bindOff = bindOff;
    this.nFallback = fallback; this.nUnbound = unbound;
  }

  // Static per-splat texture data (SPLAT_TEXELS RGBA texels per splat).
  splatTexels() {
    const { n, pos, op, quat, ls } = this.splat, out = new Float32Array(n * SPLAT_TEXELS * 4);
    for (let i = 0; i < n; i++) {
      const o = i * SPLAT_TEXELS * 4;
      out.set(pos.subarray(i * 3, i * 3 + 3), o); out[o + 3] = op[i];
      out.set(quat.subarray(i * 4, i * 4 + 4), o + 4);
      out.set(ls.subarray(i * 3, i * 3 + 3), o + 8); out[o + 11] = this.bindF[i];
      out.set(this.bindOff.subarray(i * 3, i * 3 + 3), o + 12); out[o + 15] = this.bindB[i * 2];
      out[o + 16] = this.bindB[i * 2 + 1];
    }
    return out;
  }

  // SH texture data: ceil(K * 3 / 4) texels per splat. Hands the splat's SH over (the rig itself never reads it).
  shTexels() {
    const { n, K, sh } = this.splat, per = Math.ceil(K * 3 / 4);
    if (!sh) throw new Error('the SH coefficients were already handed over');
    this.splat.sh = null;
    if (per * 4 === K * 3) return { per, data: sh };
    const out = new Float32Array(n * per * 4);
    for (let i = 0; i < n; i++) out.set(sh.subarray(i * K * 3, (i + 1) * K * 3), i * per * 4);
    return { per, data: out };
  }

  // --- appearance MLP (cage_app.cu) ---

  setApp(app) {
    this.app = null; this.voutCache = new Map();
    if (!app) return;
    const { nv, nf, faces, verts0 } = this.cage;
    if (app.nv !== nv) throw new Error(`the MLP has ${app.nv} cage vertices, this cage ${nv}`);
    // canonical edge-matrix inverses, face / vertex areas (CageApp::init)
    const inv = new Float64Array(nf * 4), area = new Float64Array(nf), varea = new Float64Array(nv);
    for (let f = 0; f < nf; f++) {
      const a = faces[f * 3], b = faces[f * 3 + 1], c = faces[f * 3 + 2];
      const ux = verts0[b * 3] - verts0[a * 3], uy = verts0[b * 3 + 1] - verts0[a * 3 + 1], uz = verts0[b * 3 + 2] - verts0[a * 3 + 2];
      const wx = verts0[c * 3] - verts0[a * 3], wy = verts0[c * 3 + 1] - verts0[a * 3 + 1], wz = verts0[c * 3 + 2] - verts0[a * 3 + 2];
      const nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx;
      const nl = Math.hypot(nx, ny, nz), ul = Math.hypot(ux, uy, uz);
      if (!(nl > 1e-20) || !(ul > 1e-12)) continue;
      const e1x = ux / ul, e1y = uy / ul, e1z = uz / ul, n0 = nx / nl, n1 = ny / nl, n2 = nz / nl;
      const e2x = n1 * e1z - n2 * e1y, e2y = n2 * e1x - n0 * e1z, e2z = n0 * e1y - n1 * e1x;
      const p = ul, q = wx * e1x + wy * e1y + wz * e1z, s = wx * e2x + wy * e2y + wz * e2z;
      if (!(s > 1e-12)) continue;
      inv[f * 4] = Math.fround(1 / p); inv[f * 4 + 1] = Math.fround(-q / (p * s)); inv[f * 4 + 3] = Math.fround(1 / s);
      area[f] = Math.fround(0.5 * nl);
      varea[a] += area[f]; varea[b] += area[f]; varea[c] += area[f];
    }
    for (let k = 0; k < nv; k++) varea[k] = Math.fround(varea[k]);
    const live = [];
    for (let k = 0; k < nv; k++) if (varea[k] > 0) live.push(k);
    this.appInv = inv; this.appArea = area; this.appVarea = varea; this.appLive = live;
    this.appCnt0 = this.occCounts(verts0);
    // the MLP at zero features (per vertex: it depends on the latent only)
    const rest = new Float32Array(nv * APP_OUT), x = new Float64Array(APP_IN), o = new Float64Array(APP_OUT);
    const h1 = new Float64Array(APP_HID), h2 = new Float64Array(APP_HID);
    for (let v = 0; v < nv; v++) {
      for (let d = 0; d < APP_LAT; d++) x[APP_FEAT + d] = app.Z[v * APP_LAT + d];
      mlp(app.P, x, h1, h2, o);
      for (let j = 0; j < APP_OUT; j++) rest[v * APP_OUT + j] = o[j];
    }
    this.appRest = rest;
    this.app = app;
  }

  occCounts(V) {
    const nv = this.cage.nv, g = new Grid(V, this.appLive, OCC_R2), out = new Float32Array(nv * 2), c = [0, 0];
    for (let k = 0; k < nv; k++) {
      g.count2(V[k * 3], V[k * 3 + 1], V[k * 3 + 2], OCC_R1 * OCC_R1, OCC_R2 * OCC_R2, c);
      out[k * 2] = c[0]; out[k * 2 + 1] = c[1];
    }
    return out;
  }

  // Per-vertex features of posed vertices V (CageApp::vert_forward): [nv][APP_FEAT].
  features(V) {
    const { nv, nf, faces } = this.cage, { appInv: inv, appArea: area, appVarea: varea, vfOff, vfIdx } = this;
    const feat = new Float32Array(nv * APP_FEAT), ff = new Float64Array(nf * 2), vt = new Float64Array(nv * 2);
    for (let f = 0; f < nf; f++) {
      let rx = 0, ry = 0;
      const a = faces[f * 3], b = faces[f * 3 + 1], c = faces[f * 3 + 2];
      if (area[f] > 0) {
        const ux = V[b * 3] - V[a * 3], uy = V[b * 3 + 1] - V[a * 3 + 1], uz = V[b * 3 + 2] - V[a * 3 + 2];
        const wx = V[c * 3] - V[a * 3], wy = V[c * 3 + 1] - V[a * 3 + 1], wz = V[c * 3 + 2] - V[a * 3 + 2];
        const nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx;
        const nl = Math.hypot(nx, ny, nz), ul = Math.hypot(ux, uy, uz);
        if (nl > 1e-20 && ul > 1e-12) {
          const e1x = ux / ul, e1y = uy / ul, e1z = uz / ul, n0 = nx / nl, n1 = ny / nl, n2 = nz / nl;
          const e2x = n1 * e1z - n2 * e1y, e2y = n2 * e1x - n0 * e1z, e2z = n0 * e1y - n1 * e1x;
          const p = ul, q = wx * e1x + wy * e1y + wz * e1z, s = wx * e2x + wy * e2y + wz * e2z;
          if (s > 1e-12) {
            const i00 = inv[f * 4], i01 = inv[f * 4 + 1], i10 = inv[f * 4 + 2], i11 = inv[f * 4 + 3];
            const f00 = p * i00 + q * i10, f01 = p * i01 + q * i11, f10 = s * i10, f11 = s * i11;
            const A = f00 * f00 + f10 * f10, B = f00 * f01 + f10 * f11, C = f01 * f01 + f11 * f11;
            const m = 0.5 * (A + C), d = Math.sqrt(Math.max(0.25 * (A - C) * (A - C) + B * B, 0));
            const l1 = m + d, l2 = Math.max(m - d, 1e-12);
            rx = Math.min(Math.max(0.5 * Math.log(l1), -LOG_S_MAX), LOG_S_MAX);
            ry = Math.min(Math.max(0.5 * Math.log(l2), -LOG_S_MAX), LOG_S_MAX);
          }
        }
      }
      ff[f * 2] = rx; ff[f * 2 + 1] = ry;
    }
    const gather = slot => {
      for (let k = 0; k < nv; k++) {
        let sx = 0, sy = 0;
        for (let j = vfOff[k]; j < vfOff[k + 1]; j++) { const f = vfIdx[j], w = area[f]; sx += w * ff[f * 2]; sy += w * ff[f * 2 + 1]; }
        const iv = varea[k] > 0 ? 1 / varea[k] : 0;
        vt[k * 2] = sx * iv; vt[k * 2 + 1] = sy * iv;
        if (slot >= 0) { feat[k * APP_FEAT + slot] = vt[k * 2]; feat[k * APP_FEAT + slot + 1] = vt[k * 2 + 1]; }
      }
    };
    gather(0);
    for (let it = 1; it <= 6; it++) {   // face mean <-> vertex mean: ~2 and ~6 rings
      for (let f = 0; f < nf; f++) {
        const a = faces[f * 3], b = faces[f * 3 + 1], c = faces[f * 3 + 2];
        ff[f * 2] = (vt[a * 2] + vt[b * 2] + vt[c * 2]) / 3; ff[f * 2 + 1] = (vt[a * 2 + 1] + vt[b * 2 + 1] + vt[c * 2 + 1]) / 3;
      }
      gather(it === 2 ? 2 : it === 6 ? 4 : -1);
    }
    if (this.app.nfeat > 6) {   // occlusion change (6 = older files: those inputs carry zero weights)
      const cnt = this.occCounts(V), c0 = this.appCnt0;
      for (let k = 0; k < nv; k++) if (varea[k] > 0) {
        feat[k * APP_FEAT + 6] = Math.log((cnt[k * 2] + 1) / (c0[k * 2] + 1));
        feat[k * APP_FEAT + 7] = Math.log((cnt[k * 2 + 1] + 1) / (c0[k * 2 + 1] + 1));
      }
    }
    return feat;
  }

  // MLP outputs R per vertex for frame fr (VERT_TEXELS RGBA texels per vertex), cached.
  vout(fr) {
    if (this.voutCache.has(fr)) return this.voutCache.get(fr);
    const { nv } = this.cage, { P, Z, dz } = this.app, rest = this.appRest;
    const V = this.cage.posed.subarray(fr * nv * 3, (fr + 1) * nv * 3);
    const feat = this.features(V);
    const out = new Float32Array(nv * VERT_TEXELS * 4);
    const x = new Float64Array(APP_IN), o = new Float64Array(APP_OUT), h1 = new Float64Array(APP_HID), h2 = new Float64Array(APP_HID);
    for (let v = 0; v < nv; v++) {
      let any = false;
      for (let d = 0; d < APP_FEAT; d++) {
        const f = feat[v * APP_FEAT + d], a = Math.max(Math.abs(f) - dz, 0);
        x[d] = f < 0 ? -a : a;
        any = any || a !== 0;
      }
      const r = rest.subarray(v * APP_OUT, (v + 1) * APP_OUT), b = v * VERT_TEXELS * 4;
      if (!any) { out[b + 1] = r[1]; out[b + 2] = r[2]; out[b + 3] = r[3]; continue; }   // features in the dead zone: rest
      for (let d = 0; d < APP_LAT; d++) x[APP_FEAT + d] = Z[v * APP_LAT + d];
      mlp(P, x, h1, h2, o);
      out[b] = o[0] - r[0]; out[b + 1] = o[1]; out[b + 2] = o[2]; out[b + 3] = o[3];
      for (let j = 4; j < APP_OUT; j++) out[b + j] = o[j] - r[j];
    }
    if (this.voutCache.size > 400) this.voutCache.delete(this.voutCache.keys().next().value);
    this.voutCache.set(fr, out);
    return out;
  }

  // --- per frame ---

  // Face texture data of frame fr, and the posed splat centres (for sorting). maxGrowth: --cage-max-growth.
  frame(fr, { maxGrowth = 1.15, useApp = true } = {}) {
    const { nv, nf, faces, posed } = this.cage, { k0, q0 } = this;
    const V = posed.subarray(fr * nv * 3, (fr + 1) * nv * 3);
    const face = new Float32Array(nf * FACE_TEXELS * 4), R = new Float64Array(9), q = new Float64Array(4);
    for (let f = 0; f < nf; f++) {
      const ia = faces[f * 3], ib = faces[f * 3 + 1], ic = faces[f * 3 + 2], o = f * FACE_TEXELS * 4;
      face[o] = V[ia * 3]; face[o + 1] = V[ia * 3 + 1]; face[o + 2] = V[ia * 3 + 2]; face[o + 3] = ia;
      face[o + 4] = V[ib * 3]; face[o + 5] = V[ib * 3 + 1]; face[o + 6] = V[ib * 3 + 2]; face[o + 7] = ib;
      face[o + 8] = V[ic * 3]; face[o + 9] = V[ic * 3 + 1]; face[o + 10] = V[ic * 3 + 2]; face[o + 11] = ic;
      let k = triFrame(V, ia, ib, ic, R);
      const ratio = k0[f] > 0 ? k / k0[f] : 1;
      if (maxGrowth > 0 && k0[f] > 0) k = Math.min(Math.max(k, k0[f] / maxGrowth), k0[f] * maxGrowth);
      face[o + 12] = R[0]; face[o + 13] = R[1]; face[o + 14] = R[2]; face[o + 15] = k;
      face[o + 16] = R[3]; face[o + 17] = R[4]; face[o + 18] = R[5]; face[o + 19] = k0[f];
      face[o + 20] = R[6]; face[o + 21] = R[7]; face[o + 22] = R[8]; face[o + 23] = ratio;
      // dq = quat(R) * conj(q0): the rotation R R0^T
      matToQuat(R, q, 0);
      const aw = q[0], ax = q[1], ay = q[2], az = q[3];
      const bw = q0[f * 4], bx = -q0[f * 4 + 1], by = -q0[f * 4 + 2], bz = -q0[f * 4 + 3];
      face[o + 24] = aw * bw - ax * bx - ay * by - az * bz;
      face[o + 25] = aw * bx + ax * bw + ay * bz - az * by;
      face[o + 26] = aw * by - ax * bz + ay * bw + az * bx;
      face[o + 27] = aw * bz + ax * by - ay * bx + az * bw;
    }
    const vout = this.app && useApp ? this.vout(fr) : null;
    // posed centres (pose_bound + the MLP's position channel)
    const { n, pos } = this.splat, { bindF, bindB, bindOff } = this, centres = new Float32Array(n * 3);
    const maxDp = vout ? this.app.maxDp : 0;
    for (let i = 0; i < n; i++) {
      const f = bindF[i];
      if (f < 0) { centres[i * 3] = pos[i * 3]; centres[i * 3 + 1] = pos[i * 3 + 1]; centres[i * 3 + 2] = pos[i * 3 + 2]; continue; }
      const o = f * FACE_TEXELS * 4, b1 = bindB[i * 2], b2 = bindB[i * 2 + 1], w0 = 1 - b1 - b2;
      const ox = bindOff[i * 3], oy = bindOff[i * 3 + 1], oz = bindOff[i * 3 + 2], k = face[o + 15];
      let x = face[o] * w0 + face[o + 4] * b1 + face[o + 8] * b2 + (face[o + 12] * ox + face[o + 13] * oy + face[o + 14] * oz) * k;
      let y = face[o + 1] * w0 + face[o + 5] * b1 + face[o + 9] * b2 + (face[o + 16] * ox + face[o + 17] * oy + face[o + 18] * oz) * k;
      let z = face[o + 2] * w0 + face[o + 6] * b1 + face[o + 10] * b2 + (face[o + 20] * ox + face[o + 21] * oy + face[o + 22] * oz) * k;
      if (maxDp > 0) {
        const ia = face[o + 3] * 12, ib = face[o + 7] * 12, ic = face[o + 11] * 12;
        const l = [0, 1, 2].map(j => maxDp * Math.tanh((w0 * vout[ia + 6 + j] + b1 * vout[ib + 6 + j] + b2 * vout[ic + 6 + j]) / maxDp));
        x += face[o + 12] * l[0] + face[o + 13] * l[1] + face[o + 14] * l[2];
        y += face[o + 16] * l[0] + face[o + 17] * l[1] + face[o + 18] * l[2];
        z += face[o + 20] * l[0] + face[o + 21] * l[1] + face[o + 22] * l[2];
      }
      centres[i * 3] = x; centres[i * 3 + 1] = y; centres[i * 3 + 2] = z;
    }
    return { face, vout, centres };
  }

  // Centroid and bounding-box size of the posed body layer (the camera's follow target and framing).
  bodyCentre(fr) {
    const { nv, posed, layers } = this.cage, L = layers[0];
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity], sum = [0, 0, 0];
    for (let k = L.vOff; k < L.vOff + L.vCount; k++)
      for (let j = 0; j < 3; j++) {
        const x = posed[(fr * nv + k) * 3 + j];
        sum[j] += x; if (x < lo[j]) lo[j] = x; if (x > hi[j]) hi[j] = x;
      }
    return { centre: sum.map(s => s / L.vCount), size: hi.map((x, j) => x - lo[j]) };
  }
}

const silu = a => a / (1 + Math.exp(-a));

// 16 -> 32 -> 32 -> 9, SiLU (cage_app.cu mlp_fwd).
function mlp(P, x, h1, h2, o) {
  for (let h = 0; h < APP_HID; h++) {
    let s = P[O_B1 + h];
    const w = O_W1 + h * APP_IN;
    for (let d = 0; d < APP_IN; d++) s += P[w + d] * x[d];
    h1[h] = silu(s);
  }
  for (let h = 0; h < APP_HID; h++) {
    let s = P[O_B2 + h];
    const w = O_W2 + h * APP_HID;
    for (let k = 0; k < APP_HID; k++) s += P[w + k] * h1[k];
    h2[h] = silu(s);
  }
  for (let j = 0; j < APP_OUT; j++) {
    let s = P[O_B3 + j];
    const w = O_W3 + j * APP_HID;
    for (let k = 0; k < APP_HID; k++) s += P[w + k] * h2[k];
    o[j] = s;
  }
}
