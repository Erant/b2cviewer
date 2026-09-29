// b2ctrain's cage posing (src/gpu/cage.cu pose_bound), ported to JS for the viewer. The binding is b2ctrain's own,
// read from the subject file (B2CRIG_splat_cage); per posed cage frame this builds the triangle frames (-> the face
// texture) and the posed splat centres (depth sorting). The per-splat posing itself (position, rotation, scale, SH
// frame) runs in the vertex shader (renderer.js) from the face texture.

export const SPLAT_TEXELS = 5;   // t0 pos.xyz op | t1 quat wxyz | t2 lscale.xyz face | t3 off.xyz b1 | t4 b2
export const FACE_TEXELS = 7;    // a.xyz | b.xyz | c.xyz | R row0, k | R row1, k0 | R row2, ratio | dq wxyz

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

export class Rig {
  // splat, binding, cage: parseSubject's (gltf.js).
  constructor(splat, binding, cage) {
    const { nf, faces, verts0 } = cage;
    if (binding.face.length !== splat.n) throw new Error(`the binding has ${binding.face.length} splats, the splat ${splat.n}`);
    this.splat = splat; this.cage = cage;
    this.bindF = binding.face; this.bindB = binding.bary; this.bindOff = binding.offset;
    this.nUnbound = 0;
    for (let i = 0; i < splat.n; i++) if (this.bindF[i] < 0 || this.bindF[i] >= nf) { this.bindF[i] = -1; this.nUnbound++; }
    // canonical triangle frames
    this.k0 = new Float32Array(nf); this.q0 = new Float32Array(nf * 4);
    const R = new Float64Array(9);
    for (let f = 0; f < nf; f++) {
      this.k0[f] = triFrame(verts0, faces[f * 3], faces[f * 3 + 1], faces[f * 3 + 2], R);
      matToQuat(R, this.q0, f * 4);
    }
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

  // Face texture data for the posed cage V [nv * 3], and the posed splat centres (for sorting).
  // maxGrowth: b2ctrain's --cage-max-growth.
  frame(V, maxGrowth = 1.15) {
    const { nf, faces } = this.cage, { k0, q0 } = this;
    const face = new Float32Array(nf * FACE_TEXELS * 4), R = new Float64Array(9), q = new Float64Array(4);
    for (let f = 0; f < nf; f++) {
      const ia = faces[f * 3], ib = faces[f * 3 + 1], ic = faces[f * 3 + 2], o = f * FACE_TEXELS * 4;
      face[o] = V[ia * 3]; face[o + 1] = V[ia * 3 + 1]; face[o + 2] = V[ia * 3 + 2];
      face[o + 4] = V[ib * 3]; face[o + 5] = V[ib * 3 + 1]; face[o + 6] = V[ib * 3 + 2];
      face[o + 8] = V[ic * 3]; face[o + 9] = V[ic * 3 + 1]; face[o + 10] = V[ic * 3 + 2];
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
    // posed centres (pose_bound)
    const { n, pos } = this.splat, { bindF, bindB, bindOff } = this, centres = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const f = bindF[i];
      if (f < 0) { centres[i * 3] = pos[i * 3]; centres[i * 3 + 1] = pos[i * 3 + 1]; centres[i * 3 + 2] = pos[i * 3 + 2]; continue; }
      const o = f * FACE_TEXELS * 4, b1 = bindB[i * 2], b2 = bindB[i * 2 + 1], w0 = 1 - b1 - b2;
      const ox = bindOff[i * 3], oy = bindOff[i * 3 + 1], oz = bindOff[i * 3 + 2], k = face[o + 15];
      centres[i * 3] = face[o] * w0 + face[o + 4] * b1 + face[o + 8] * b2 + (face[o + 12] * ox + face[o + 13] * oy + face[o + 14] * oz) * k;
      centres[i * 3 + 1] = face[o + 1] * w0 + face[o + 5] * b1 + face[o + 9] * b2 + (face[o + 16] * ox + face[o + 17] * oy + face[o + 18] * oz) * k;
      centres[i * 3 + 2] = face[o + 2] * w0 + face[o + 6] * b1 + face[o + 10] * b2 + (face[o + 20] * ox + face[o + 21] * oy + face[o + 22] * oz) * k;
    }
    return { face, centres };
  }

  // Centroid and bounding-box size of the posed body layer (the camera's follow target and framing).
  bodyCentre(V) {
    const L = this.cage.layers[this.cage.body];
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity], sum = [0, 0, 0];
    for (let k = L.vOff; k < L.vOff + L.vCount; k++)
      for (let j = 0; j < 3; j++) {
        const x = V[k * 3 + j];
        sum[j] += x; if (x < lo[j]) lo[j] = x; if (x > hi[j]) hi[j] = x;
      }
    return { centre: sum.map(s => s / L.vCount), size: hi.map((x, j) => x - lo[j]) };
  }
}
