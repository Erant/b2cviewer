// The armpit-crease heuristic (b2crig: b2crig/rig/crease.py, whose constants these are; keep them in step).
// A splat is crease wall when it lies within REACH of an arm vertex and of a torso vertex of the canonical body and
// the two vertex normals face each other; its value is the larger of the two distances. The renderer culls splats
// whose value is below the chosen radius (they were captured in shadow between the two walls and bloom into coloured
// wisps under a raised arm).

export const REACH = 0.10, FACING = -0.2, NONE = 1e9;
const ARM = ['l_uparm', 'r_uparm', 'l_lowarm', 'r_lowarm'], TORSO = ['c_spine', 'root'];

// splat: { n, pos }; cage: parseSubject's; names: the skeleton's joint names. Returns Float32Array [n] (NONE = not crease).
export function creaseValues(splat, cage, names) {
  const out = new Float32Array(splat.n).fill(NONE);
  const L = cage.layers[cage.body], { verts0, faces, jIdx, jW, KW } = cage, vo = L.vOff;
  // vertex normals of the body layer (area weighted, as crease.py vertex_normals)
  const nrm = new Float64Array(L.vCount * 3);
  for (let f = L.fOff; f < L.fOff + L.fCount; f++) {
    const a = faces[f * 3], b = faces[f * 3 + 1], c = faces[f * 3 + 2];
    const ux = verts0[b * 3] - verts0[a * 3], uy = verts0[b * 3 + 1] - verts0[a * 3 + 1], uz = verts0[b * 3 + 2] - verts0[a * 3 + 2];
    const wx = verts0[c * 3] - verts0[a * 3], wy = verts0[c * 3 + 1] - verts0[a * 3 + 1], wz = verts0[c * 3 + 2] - verts0[a * 3 + 2];
    const nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx;
    for (const v of [a, b, c]) { const o = (v - vo) * 3; nrm[o] += nx; nrm[o + 1] += ny; nrm[o + 2] += nz; }
  }
  for (let v = 0; v < L.vCount; v++) {
    const l = Math.max(Math.hypot(nrm[v * 3], nrm[v * 3 + 1], nrm[v * 3 + 2]), 1e-9);
    nrm[v * 3] /= l; nrm[v * 3 + 1] /= l; nrm[v * 3 + 2] /= l;
  }
  // arm / torso vertices by their dominant skin joint's name
  const kind = new Uint8Array(L.vCount);   // 1 arm, 2 torso
  for (let v = 0; v < L.vCount; v++) {
    let best = -1, bw = -Infinity;
    for (let k = 0; k < KW; k++) { const w = jW[(vo + v) * KW + k]; if (w > bw) { bw = w; best = jIdx[(vo + v) * KW + k]; } }
    const nm = names[best] || '';
    kind[v] = ARM.some(p => nm.startsWith(p)) ? 1 : TORSO.some(p => nm.startsWith(p)) ? 2 : 0;
  }
  // a hash grid of cell REACH per kind: any vertex within REACH of a splat lies in its 3x3x3 cells
  const grids = [null, new Map(), new Map()], key = (x, y, z) => (x * 73856093) ^ (y * 19349663) ^ (z * 83492791);
  const cell = x => Math.floor(x / REACH);
  for (let v = 0; v < L.vCount; v++) {
    if (!kind[v]) continue;
    const p = (vo + v) * 3, k = key(cell(verts0[p]), cell(verts0[p + 1]), cell(verts0[p + 2])), g = grids[kind[v]];
    (g.get(k) || g.set(k, []).get(k)).push(v);
  }
  const nearest = (g, x, y, z) => {   // [distance, vertex] of the nearest vertex within REACH, or [Infinity, -1]
    let bd = REACH * REACH, bv = -1;
    const cx = cell(x), cy = cell(y), cz = cell(z);
    for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) for (let k = -1; k <= 1; k++) {
      const vs = g.get(key(cx + i, cy + j, cz + k)); if (!vs) continue;
      for (const v of vs) {
        const p = (vo + v) * 3, dx = verts0[p] - x, dy = verts0[p + 1] - y, dz = verts0[p + 2] - z, d = dx * dx + dy * dy + dz * dz;
        if (d < bd) { bd = d; bv = v; }
      }
    }
    return bv < 0 ? [Infinity, -1] : [Math.sqrt(bd), bv];
  };
  const { pos } = splat;
  for (let i = 0; i < splat.n; i++) {
    const x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2];
    const [da, va] = nearest(grids[1], x, y, z); if (va < 0) continue;
    const [dt, vt] = nearest(grids[2], x, y, z); if (vt < 0) continue;
    const dot = nrm[va * 3] * nrm[vt * 3] + nrm[va * 3 + 1] * nrm[vt * 3 + 1] + nrm[va * 3 + 2] * nrm[vt * 3 + 2];
    if (dot < FACING) out[i] = Math.max(da, dt);
  }
  return out;
}
