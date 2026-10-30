// Relighting helpers (experimental). The captured splat colour is taken as albedo under an even light of 1; the
// renderer (renderer.js relight) multiplies it by a hemisphere ambient plus parametric lights and adds a specular.
// Lights are placed in glTF world (+Y up) around the body like the orbit camera; main.js maps them to the b2crunner
// frame for the renderer.

// Black-body colour of `K` kelvin (Tanner Helland's fit), brightest channel 1.
export function kelvinRGB(K) {
  const t = K / 100;
  const r = t <= 66 ? 255 : 329.698727446 * Math.pow(t - 60, -0.1332047592);
  const g = t <= 66 ? 99.4708025861 * Math.log(t) - 161.1195681661 : 288.1221695283 * Math.pow(t - 60, -0.0755148492);
  const b = t >= 66 ? 255 : t <= 19 ? 0 : 138.5177312231 * Math.log(t - 10) - 305.0447927307;
  const c = [r, g, b].map(x => Math.min(Math.max(x, 0), 255));
  const m = Math.max(...c);
  return c.map(x => x / m);
}

// '#rrggbb' -> [r, g, b] in 0..1, linearised (gamma 2.2) when `linear`
export function hexRGB(hex, linear) {
  const c = [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16) / 255);
  return linear ? c.map(x => Math.pow(x, 2.2)) : c;
}

// Unit vector from the target towards azimuth / elevation (degrees), glTF world: az 0 = in front (+Z), 90 = +X.
export function dirOf(az, el) {
  const a = az * Math.PI / 180, e = el * Math.PI / 180;
  return [Math.cos(e) * Math.sin(a), Math.sin(e), Math.cos(e) * Math.cos(a)];
}

// The light in glTF world: { pos, dir } with dir the way its light travels. p: the panel's light parameters;
// target: the point it circles (and a spot aims at); orbit: the camera's, when the light moves with the camera.
export function placeLight(p, target, orbit) {
  const az = p.az + (p.withCamera && orbit ? orbit.az : 0);
  const el = Math.max(-89, Math.min(89, p.el + (p.withCamera && orbit ? orbit.el : 0)));
  const d = dirOf(az, el);
  return { pos: target.map((x, j) => x + p.dist * d[j]), dir: d.map(x => -x) };
}

const sub = (a, b) => a.map((x, j) => x - b[j]), dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit = a => { const l = Math.hypot(...a) || 1; return a.map(x => x / l); };

// The shadow map's view of the body: a perspective from a point or spot light at `pos` (sun: dir, an orthographic
// view along it) onto the sphere (centre, radius) around the posed body. All in the b2crunner frame; up: world up.
// Returns { vp (column-major 4x4), eye, fwd, ortho, texel }; the map holds the depth along fwd from eye. texel: the
// map's width in metres (sun), or that over the depth (point, spot).
export function shadowFrustum({ pos, dir }, centre, radius, up) {
  const sun = !pos, fwd = unit(sun ? dir : sub(centre, pos)), d = sun ? 0 : Math.hypot(...sub(centre, pos));
  const eye = sun ? centre.map((c, j) => c - fwd[j] * (radius + 1)) : pos;
  const s = unit(cross(fwd, Math.abs(dot(fwd, up)) > 0.99 ? [1, 0, 0] : up)), u = cross(s, fwd);
  // view (rows s, u, -fwd), then the projection
  const V = [s, u, fwd.map(x => -x)], t = V.map(r => -dot(r, eye));
  let P, texel;
  if (sun) {
    texel = 2 * radius;
    const n = 0.01, f = 2 * radius + 2;
    P = [[1 / radius, 0, 0, 0], [0, 1 / radius, 0, 0], [0, 0, -2 / (f - n), -(f + n) / (f - n)], [0, 0, 0, 1]];
  } else {
    // the light outside the sphere: just the sphere; inside it: a wide view of what lies ahead
    const half = d > radius * 1.02 ? Math.asin(radius / d) : 75 * Math.PI / 180;
    const n = Math.max(d - radius, 0.02), f = d + radius, c = 1 / Math.tan(half);
    texel = 2 * Math.tan(half);
    P = [[c, 0, 0, 0], [0, c, 0, 0], [0, 0, -(f + n) / (f - n), -2 * f * n / (f - n)], [0, 0, -1, 0]];
  }
  const M = [V[0].concat(t[0]), V[1].concat(t[1]), V[2].concat(t[2]), [0, 0, 0, 1]];
  const vp = new Float32Array(16);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) vp[c * 4 + r] = P[r].reduce((acc, x, k) => acc + x * M[k][c], 0);
  return { vp, eye, fwd, ortho: sun, texel };
}
