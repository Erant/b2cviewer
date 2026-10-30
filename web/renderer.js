// WebGL2 splat renderer. The vertex shader poses each splat from the cage (b2ctrain cage.cu pose_bound and the
// stretch fade), evaluates its SH colour in the triangle's canonical frame, and projects it as b2ctrain's rasteriser
// does (EWA, 0.3 px blur, alpha from the 1/255 contour).
import { SPLAT_TEXELS, FACE_TEXELS } from './rig.js';

export const MAX_LIGHTS = 4;

const VS = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;

layout(location = 0) in vec2 aCorner;
layout(location = 1) in uint aIndex;

uniform sampler2D uSplat, uSH, uFace;
uniform highp sampler2D uShadow;   // light 0's shadow map: linear depth (m) of the posed cage along uLFwd
uniform ivec3 uRows;      // items per texture row: splat, sh, face
uniform int uShPer, uDegree;
uniform vec2 uFade;       // stretch fade start, end (0 = off)
uniform vec2 uCrease;     // crease fade: radius (0 = off), width in metres; opacity x 0 at crease value <= radius, x 1 at radius + width
uniform mat3 uViewR;      // world -> camera (OpenCV: x right, y down, z ahead)
uniform vec3 uViewT, uCamPos;
uniform vec4 uIntr;       // fx, fy, cx, cy (pixels)
uniform vec4 uLim;        // EWA Jacobian clamp: pos x, pos y, neg x, neg y
uniform vec2 uViewport;
uniform int uLightView;   // relighting (light.js): 0 off, 1 lit, 2 normals, 3 lighting on grey
uniform float uNormalMix; // how far flat splats trust their own shortest axis over the cage normal
uniform mat3 uWR;         // b2crunner frame -> glTF world (the normals view's colours)
uniform vec3 uSky, uGround, uUp;   // hemisphere ambient (already times its intensity), world up (b2crunner frame)
uniform vec4 uShade;      // wrap, exposure, specular strength, shininess
uniform ivec3 uShadeOpt;  // linear (gamma 2.2) lighting, soft shoulder, relight the DC only (the view-dependent rest as captured)
uniform int uNLights;
uniform vec4 uLPos[${MAX_LIGHTS}];    // xyz position (w = 1) or direction the light travels (w = 0)
uniform vec4 uLCol[${MAX_LIGHTS}];    // rgb colour x intensity, range (0 = none)
uniform vec4 uLSpot[${MAX_LIGHTS}];   // xyz spot axis, cos(outer cone) (-2 = no cone)
uniform float uLSoft[${MAX_LIGHTS}];  // cos(inner cone)
uniform int uShadowOn;
uniform mat4 uLVP;        // b2crunner frame -> light 0's shadow clip space
uniform vec3 uLEye, uLFwd;
uniform vec4 uShadowP;    // bias (m), softness (texels), strength, map size
uniform vec2 uShadowT;    // a texel's width in metres: x (sun), or x times the depth (point, spot); y 1 for the sun

out vec4 vColor;
out vec3 vConic;
out vec2 vD;

vec4 fetch(sampler2D t, int idx, int per, int perRow, int j) {
  return texelFetch(t, ivec2((idx % perRow) * per + j, idx / perRow), 0);
}
vec4 qmul(vec4 a, vec4 b) {   // (w, x, y, z) Hamilton product
  return vec4(a.x * b.x - a.y * b.y - a.z * b.z - a.w * b.w,
              a.x * b.y + a.y * b.x + a.z * b.w - a.w * b.z,
              a.x * b.z - a.y * b.w + a.z * b.x + a.w * b.y,
              a.x * b.w + a.y * b.z - a.z * b.y + a.w * b.x);
}
mat3 qmat(vec4 q) {           // columns of the rotation matrix of a (w, x, y, z) quaternion
  q = normalize(q);
  float w = q.x, x = q.y, y = q.z, z = q.w;
  return mat3(1. - 2. * (y * y + z * z), 2. * (x * y + w * z), 2. * (x * z - w * y),
              2. * (x * y - w * z), 1. - 2. * (x * x + z * z), 2. * (y * z + w * x),
              2. * (x * z + w * y), 2. * (y * z - w * x), 1. - 2. * (x * x + y * y));
}
float tanh_(float x) { float t = exp(-2. * abs(x)); return sign(x) * (1. - t) / (1. + t); }
vec3 tanh3(vec3 x) { return vec3(tanh_(x.x), tanh_(x.y), tanh_(x.z)); }

// Light 0's visibility at p (normal n, towards the light l): soft (12 taps on a disc) comparison against the cage's
// depth map; 1 outside the map. The bias grows with the slope of the surface to the light across the taps' disc (a
// surface at a grazing angle would shadow itself), and the lookup moves off the surface along n by two texels.
float shadow(vec3 p, vec3 n, vec3 l) {
  float tex = uShadowT.x * (uShadowT.y > 0.5 ? 1. : max(dot(p - uLEye, uLFwd), 0.01));
  float cs = clamp(abs(dot(n, l)), 0.05, 1.), slope = min((uShadowP.y + 1.) * tex * sqrt(1. - cs * cs) / cs, 0.1);
  p += n * 2. * tex;
  vec4 c = uLVP * vec4(p, 1.);
  vec2 uv = c.xy / c.w * 0.5 + 0.5;
  if (c.w <= 0. || any(lessThan(uv, vec2(0.))) || any(greaterThan(uv, vec2(1.)))) return 1.;
  float d = dot(p - uLEye, uLFwd) - uShadowP.x - slope, sz = uShadowP.w, lit = 0.;
  vec2 px = uv * sz;
  for (int k = 0; k < 12; k++) {
    float a = float(k) * 2.39996, r = sqrt((float(k) + 0.5) / 12.) * uShadowP.y;   // golden-angle spiral
    ivec2 t = clamp(ivec2(px + r * vec2(cos(a), sin(a))), ivec2(0), ivec2(sz - 1.));
    lit += d <= texelFetch(uShadow, t, 0).r ? 1. : 0.;
  }
  return mix(1., lit / 12., uShadowP.z);
}

// Relit colour of a splat whose captured colour (base) is taken as albedo x an even light of 1. The shadow is looked
// up at the splat's anchor on its cage triangle (normal na): a splat a little inside or outside the cage takes the
// shadow of the surface it belongs to instead of being shadowed by it.
vec3 relight(vec3 base, vec3 pos, vec3 n, vec3 anchor, vec3 na) {
  vec3 alb = uShadeOpt.x == 1 ? pow(max(base, 0.), vec3(2.2)) : base;
  vec3 E = mix(uGround, uSky, dot(n, uUp) * 0.5 + 0.5), spec = vec3(0.);
  vec3 v = normalize(uCamPos - pos);
  for (int j = 0; j < ${MAX_LIGHTS}; j++) {
    if (j >= uNLights) break;
    vec3 l; float att = 1.;
    if (uLPos[j].w > 0.) {
      vec3 dl = uLPos[j].xyz - pos; float d2 = max(dot(dl, dl), 1e-4);
      l = dl * inversesqrt(d2);
      att = 1. / d2;
      if (uLCol[j].w > 0.) { float r = d2 / (uLCol[j].w * uLCol[j].w); att *= clamp(1. - r * r, 0., 1.); }   // smooth range window
      if (uLSpot[j].w > -1.5) att *= smoothstep(uLSpot[j].w, uLSoft[j], dot(-l, uLSpot[j].xyz));
    } else l = -uLPos[j].xyz;
    float nl = dot(n, l), w = uShade.x;
    if (j == 0 && uShadowOn == 1 && nl > -w) att *= shadow(anchor, na, l);
    E += uLCol[j].rgb * att * max((nl + w) / (1. + w), 0.);
    if (uShade.z > 0. && nl > 0.) spec += uLCol[j].rgb * att * uShade.z * pow(max(dot(n, normalize(l + v)), 0.), uShade.w);
  }
  vec3 c = (alb * E + spec) * uShade.y;
  if (uShadeOpt.y == 1) {   // soft shoulder above 0.8: keeps the range below, rolls highlights off towards 1
    vec3 k = vec3(0.8);
    c = mix(c, k + 0.2 * (1. - exp(-(c - k) / 0.2)), step(k, c));
  }
  return uShadeOpt.x == 1 ? pow(max(c, 0.), vec3(1. / 2.2)) : c;
}

void cull() { gl_Position = vec4(0., 0., 2., 1.); }

void main() {
  int i = int(aIndex);
  vec4 t0 = fetch(uSplat, i, ${SPLAT_TEXELS}, uRows.x, 0), t1 = fetch(uSplat, i, ${SPLAT_TEXELS}, uRows.x, 1);
  vec4 t2 = fetch(uSplat, i, ${SPLAT_TEXELS}, uRows.x, 2);
  vec3 pos = t0.xyz; float op = t0.w; vec4 q = t1; vec3 ls = t2.xyz; int f = int(t2.w);
  vec4 shf = vec4(1., 0., 0., 0.);
  vec3 nCage = vec3(0.), anchor = pos;
  if (f >= 0) {
    vec4 t3 = fetch(uSplat, i, ${SPLAT_TEXELS}, uRows.x, 3), t4 = fetch(uSplat, i, ${SPLAT_TEXELS}, uRows.x, 4);
    float b1 = t3.w, b2 = t4.x, w0 = 1. - b1 - b2;
    vec4 A = fetch(uFace, f, ${FACE_TEXELS}, uRows.z, 0), B = fetch(uFace, f, ${FACE_TEXELS}, uRows.z, 1), C = fetch(uFace, f, ${FACE_TEXELS}, uRows.z, 2);
    vec4 R0 = fetch(uFace, f, ${FACE_TEXELS}, uRows.z, 3), R1 = fetch(uFace, f, ${FACE_TEXELS}, uRows.z, 4), R2 = fetch(uFace, f, ${FACE_TEXELS}, uRows.z, 5);
    vec4 dq = fetch(uFace, f, ${FACE_TEXELS}, uRows.z, 6);
    float k = R0.w, k0 = R1.w, ratio = R2.w;
    vec3 off = t3.xyz;
    anchor = A.xyz * w0 + B.xyz * b1 + C.xyz * b2;
    pos = anchor + vec3(dot(R0.xyz, off), dot(R1.xyz, off), dot(R2.xyz, off)) * k;
    if (uFade.x > 0. && uFade.y > uFade.x) {   // stretch fade: opacity -> 0 as the triangle grows from start to end
      float fa = clamp((uFade.y - ratio) / (uFade.y - uFade.x), 0., 1.);
      if (fa < 1.) { float sg = clamp(fa / (1. + exp(-op)), 1e-6, 1. - 1e-6); op = log(sg / (1. - sg)); }
    }
    q = qmul(dq, q);
    ls += (k0 > 0. && k > 0.) ? log(k / k0) : 0.;
    shf = dq;
    if (uLightView > 0) {
      vec3 na = fetch(uFace, f, ${FACE_TEXELS}, uRows.z, 7).xyz, nb = fetch(uFace, f, ${FACE_TEXELS}, uRows.z, 8).xyz, nc = fetch(uFace, f, ${FACE_TEXELS}, uRows.z, 9).xyz;
      nCage = na * w0 + nb * b1 + nc * b2;
      nCage = dot(nCage, nCage) > 1e-8 ? normalize(nCage) : vec3(R0.z, R1.z, R2.z);
    }
  }

  // projection (splat_math.cuh project_one)
  vec3 mc = uViewR * pos + uViewT;
  if (!(mc.z >= 0.01)) { cull(); return; }
  float opac = 1. / (1. + exp(-op));
  if (uCrease.x > 0.) opac *= clamp((fetch(uSplat, i, ${SPLAT_TEXELS}, uRows.x, 4).y - uCrease.x) / max(uCrease.y, 1e-4), 0., 1.);   // crease.js
  if (!(opac >= 1. / 255.)) { cull(); return; }
  mat3 M = uViewR * qmat(q) * mat3(exp(ls.x), 0., 0., 0., exp(ls.y), 0., 0., 0., exp(ls.z));
  float iz = 1. / mc.z, rx = mc.x * iz, ry = mc.y * iz;
  float cxr = clamp(rx, uLim.z, uLim.x), cyr = clamp(ry, uLim.w, uLim.y);
  float dx = uIntr.x * iz, dy = uIntr.y * iz;
  vec3 v0 = vec3(dx * M[0].x - dx * cxr * M[0].z, dx * M[1].x - dx * cxr * M[1].z, dx * M[2].x - dx * cxr * M[2].z);
  vec3 v1 = vec3(dy * M[0].y - dy * cyr * M[0].z, dy * M[1].y - dy * cyr * M[1].z, dy * M[2].y - dy * cyr * M[2].z);
  float c00 = dot(v0, v0) + 0.3, c01 = dot(v0, v1), c11 = dot(v1, v1) + 0.3;
  float det = c00 * c11 - c01 * c01;
  if (!(det > 0.)) { cull(); return; }
  vConic = vec3(c11, -c01, c00) / det;
  vec2 mean = vec2(uIntr.x * rx + uIntr.z, uIntr.y * ry + uIntr.w);
  float power = log(opac * 255.);
  vec2 ext = sqrt(2. * power * vec2(c00, c11));
  if (mean.x + ext.x <= 0. || mean.x - ext.x >= uViewport.x || mean.y + ext.y <= 0. || mean.y - ext.y >= uViewport.y) { cull(); return; }

  // colour: SH in the triangle's canonical frame (the view direction turned back by the frame rotation)
  vec3 d = normalize(pos - uCamPos);
  if (f >= 0) d = transpose(qmat(shf)) * d;
  float c[48];
  for (int j = 0; j < 12; j++) {
    if (j >= uShPer) break;
    vec4 s = fetch(uSH, i, uShPer, uRows.y, j);
    c[j * 4] = s.x; c[j * 4 + 1] = s.y; c[j * 4 + 2] = s.z; c[j * 4 + 3] = s.w;
  }
  #define SH(k) vec3(c[(k) * 3], c[(k) * 3 + 1], c[(k) * 3 + 2])
  vec3 col = SH(0) * 0.28209479;
  if (uDegree >= 1) {
    col += SH(1) * (-0.4886025 * d.y) + SH(2) * (0.4886025 * d.z) + SH(3) * (-0.4886025 * d.x);
    if (uDegree >= 2) {
      float z2 = d.z * d.z, f0b = -1.0925485 * d.z, f1a = 0.54627424;
      float fc1 = d.x * d.x - d.y * d.y, fs1 = 2. * d.x * d.y;
      col += SH(4) * (f1a * fs1) + SH(5) * (f0b * d.y) + SH(6) * (0.9461747 * z2 - 0.31539157) + SH(7) * (f0b * d.x) + SH(8) * (f1a * fc1);
      if (uDegree >= 3) {
        float f0c = -2.285229 * z2 + 0.4570458, f1b = 1.4453057 * d.z, f2a = -0.5900436;
        float fc2 = d.x * fc1 - d.y * fs1, fs2 = d.x * fs1 + d.y * fc1;
        col += SH(9) * (f2a * fs2) + SH(10) * (f1b * fs1) + SH(11) * (f0c * d.y) + SH(12) * (d.z * (1.8658817 * z2 - 1.119529))
             + SH(13) * (f0c * d.x) + SH(14) * (f1b * fc1) + SH(15) * (f2a * fc2);
      }
    }
  }
  col += 0.5;
  if (uLightView > 0) {
    // the normal: the cage's (smooth, posed), turned towards the splat's own shortest axis as far as the splat is flat
    mat3 Q = qmat(q);
    int ax = ls.x <= ls.y && ls.x <= ls.z ? 0 : ls.y <= ls.z ? 1 : 2;
    vec3 nS = Q[ax];
    float lmin = ls[ax], lmid = ax == 0 ? min(ls.y, ls.z) : ax == 1 ? min(ls.x, ls.z) : min(ls.x, ls.y);
    float flat_ = 1. - exp(lmin - lmid);
    vec3 n;
    if (f >= 0) n = normalize(mix(nCage, nS * sign(dot(nS, nCage) + 1e-6), flat_ * uNormalMix));
    else n = nS * sign(dot(nS, uCamPos - pos) + 1e-6);   // no cage: the splat's own axis, facing the camera (the visible side)
    // a splat that is seen cannot face away: the cage has surfaces that do (inside the mouth, stray triangles), and
    // without shadows a light behind would reach them. Reflect the normal into the visible half (rims keep their side).
    vec3 vd = normalize(uCamPos - pos);
    float nv = dot(n, vd);
    if (nv < 0.) n -= 2. * nv * vd;
    if (uLightView == 2) col = uWR * n * 0.5 + 0.5;
    else {
      vec3 na = f >= 0 ? nCage : n;
      if (uLightView == 3) col = relight(vec3(0.5), pos, n, anchor, na);
      else if (uShadeOpt.z == 1) { vec3 dc = max(SH(0) * 0.28209479 + 0.5, 0.); col = relight(dc, pos, n, anchor, na) + (col - dc); }
      else col = relight(col, pos, n, anchor, na);
    }
  }
  vColor = vec4(max(col, 0.), opac);   // raster_fwd.cu composites max(colour, 0)

  vD = aCorner * ext;
  vec2 px = mean + vD;
  gl_Position = vec4(px.x / uViewport.x * 2. - 1., 1. - px.y / uViewport.y * 2., 0., 1.);
}`;

const FS = `#version 300 es
precision highp float;
in vec4 vColor;
in vec3 vConic;
in vec2 vD;
out vec4 fragColor;
void main() {
  float sigma = 0.5 * (vConic.x * vD.x * vD.x + vConic.z * vD.y * vD.y) + vConic.y * vD.x * vD.y;
  if (sigma < 0.) discard;
  float a = min(0.99, vColor.a * exp(-sigma));
  if (a < 1. / 255.) discard;
  fragColor = vec4(vColor.rgb * a, a);
}`;

// The posed cage (from the face texture: corners a, b, c of face gl_VertexID / 3) into light 0's shadow map, as the
// linear depth along the light's axis.
const SHADOW_VS = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
uniform sampler2D uFace;
uniform int uRowsF;
uniform mat4 uLVP;
uniform vec3 uLEye, uLFwd;
out float vDepth;
void main() {
  int f = gl_VertexID / 3, k = gl_VertexID % 3;
  vec3 p = texelFetch(uFace, ivec2((f % uRowsF) * ${FACE_TEXELS} + k, f / uRowsF), 0).xyz;
  vDepth = dot(p - uLEye, uLFwd);
  gl_Position = uLVP * vec4(p, 1.);
}`;
const SHADOW_FS = `#version 300 es
precision highp float;
in float vDepth;
out vec4 fragColor;
void main() { fragColor = vec4(vDepth, 0., 0., 1.); }`;
export const SHADOW_SIZE = 2048;

// Resolve the float framebuffer onto the canvas (clamped, over the background).
const BLIT_VS = `#version 300 es
const vec2 P[3] = vec2[3](vec2(-1., -1.), vec2(3., -1.), vec2(-1., 3.));
void main() { gl_Position = vec4(P[gl_VertexID], 0., 1.); }`;
const BLIT_FS = `#version 300 es
precision highp float;
uniform sampler2D uImg;
uniform vec3 uBg;
out vec4 fragColor;
void main() {
  vec4 c = texelFetch(uImg, ivec2(gl_FragCoord.xy), 0);
  fragColor = vec4(clamp(c.rgb + (1. - c.a) * uBg, 0., 1.), 1.);
}`;

function compile(gl, vs, fs) {
  const p = gl.createProgram();
  for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error('shader: ' + gl.getShaderInfoLog(s));
    gl.attachShader(p, s);
  }
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error('program: ' + gl.getProgramInfoLog(p));
  const u = {};
  for (let k = 0, n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS); k < n; k++) {
    const name = gl.getActiveUniform(p, k).name;
    u[name] = gl.getUniformLocation(p, name);
  }
  return { p, u };
}

export class Renderer {
  constructor(canvas) {
    const gl = canvas.getContext('webgl2', { antialias: false, alpha: false, preserveDrawingBuffer: true });
    if (!gl) throw new Error('WebGL2 is not available');
    if (!gl.getExtension('EXT_color_buffer_float')) throw new Error('EXT_color_buffer_float is not available');
    this.gl = gl; this.canvas = canvas;
    this.maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    this.prog = compile(gl, VS, FS);
    this.blit = compile(gl, BLIT_VS, BLIT_FS);
    this.shadowProg = compile(gl, SHADOW_VS, SHADOW_FS);
    this.shadowMap = null;
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    const quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    this.orderBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.orderBuf);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribIPointer(1, 1, gl.UNSIGNED_INT, 0, 0);
    gl.vertexAttribDivisor(1, 1);
    gl.bindVertexArray(null);
    this.tex = {};
    this.count = 0;
    this.fbo = null; this.fboSize = [0, 0];
  }

  // A float texture holding `n` items of `per` RGBA texels each; returns items per row.
  _texture(name, data, n, per, unit) {
    const gl = this.gl;
    const perRow = Math.max(1, Math.min(Math.floor(this.maxTex / per), 4096));
    const w = perRow * per, h = Math.max(1, Math.ceil(n / perRow));
    if (h > this.maxTex) throw new Error(`${name}: ${n} items do not fit a ${this.maxTex}px texture`);
    let t = this.tex[name];
    if (!t || t.w !== w || t.h !== h) {
      if (t) gl.deleteTexture(t.t);
      t = { t: gl.createTexture(), w, h, perRow, unit };
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, t.t);
      gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, w, h);
      for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
      this.tex[name] = t;
    }
    // rows are item-major (an item never straddles two rows): the flat item layout maps straight onto the texture,
    // full rows first, then the partial last row
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, t.t);
    const full = Math.floor(n / perRow), rest = n - full * perRow;
    if (full) gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, w, full, gl.RGBA, gl.FLOAT, data.subarray(0, w * full * 4));
    if (rest) gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, full, rest * per, 1, gl.RGBA, gl.FLOAT, data.subarray(w * full * 4, (w * full + rest * per) * 4));
    return perRow;
  }

  setSplats(splatTex, n) { this.n = n; this.rows = this.rows || [1, 1, 1]; this.rows[0] = this._texture('splat', splatTex, n, SPLAT_TEXELS, 0); }
  setSH(sh, n, degree) { this.shPer = sh.per; this.degree = degree; this.rows[1] = this._texture('sh', sh.data, n, sh.per, 1); }
  setFace(face, nf) { this.nf = nf; this.rows[2] = this._texture('face', face, nf, FACE_TEXELS, 2); }

  // Light 0's shadow map of the posed cage. sh: { vp (column-major 4x4), eye, fwd }
  _shadowPass(sh) {
    const gl = this.gl, N = SHADOW_SIZE;
    if (!this.shadowMap) {
      const t = gl.createTexture();
      gl.activeTexture(gl.TEXTURE5);
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texStorage2D(gl.TEXTURE_2D, 1, gl.R32F, N, N);
      for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
      const depth = gl.createRenderbuffer();
      gl.bindRenderbuffer(gl.RENDERBUFFER, depth);
      gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, N, N);
      const fbo = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depth);
      this.shadowMap = { t, fbo };
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.shadowMap.fbo);
    gl.viewport(0, 0, N, N);
    gl.clearBufferfv(gl.COLOR, 0, [1e9, 0, 0, 1]);
    gl.clearBufferfi(gl.DEPTH_STENCIL, 0, 1, 0);
    const { p, u } = this.shadowProg;
    gl.useProgram(p);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, this.tex.face.t);
    gl.uniform1i(u.uFace, 2); gl.uniform1i(u.uRowsF, this.rows[2]);
    gl.uniformMatrix4fv(u.uLVP, false, sh.vp);
    gl.uniform3f(u.uLEye, ...sh.eye); gl.uniform3f(u.uLFwd, ...sh.fwd);
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LESS);
    gl.bindVertexArray(this.vao);   // attributes unused: the corners come from gl_VertexID
    gl.drawArrays(gl.TRIANGLES, 0, this.nf * 3);
    gl.bindVertexArray(null);
    gl.disable(gl.DEPTH_TEST);
  }
  setOrder(order) {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.orderBuf);
    gl.bufferData(gl.ARRAY_BUFFER, order, gl.DYNAMIC_DRAW);
    this.count = order.length;
  }

  _target(W, H) {
    const gl = this.gl;
    if (this.fbo && this.fboSize[0] === W && this.fboSize[1] === H) return;
    if (this.fbo) { gl.deleteFramebuffer(this.fbo); gl.deleteTexture(this.fboTex); }
    this.fboTex = gl.createTexture();
    gl.activeTexture(gl.TEXTURE4);
    gl.bindTexture(gl.TEXTURE_2D, this.fboTex);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA16F, W, H);
    for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
    this.fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.fboTex, 0);
    this.fboSize = [W, H];
  }

  // cam: { R (row-major world -> camera, OpenCV), t, pos, fx, fy, cx, cy }; opts: { fade, crease, creaseFade, degree, bg }
  render(cam, opts) {
    const gl = this.gl, W = this.canvas.width, H = this.canvas.height;
    const shadowOn = !!(opts.light?.view && opts.light.shadow && this.count && this.tex.face && this.nf);
    if (shadowOn) this._shadowPass(opts.light.shadow);
    this._target(W, H);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.viewport(0, 0, W, H);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (this.count && this.tex.face) {
      const { p, u } = this.prog;
      gl.useProgram(p);
      gl.uniform1i(u.uSplat, 0); gl.uniform1i(u.uSH, 1); gl.uniform1i(u.uFace, 2);
      for (const t of Object.values(this.tex)) { gl.activeTexture(gl.TEXTURE0 + t.unit); gl.bindTexture(gl.TEXTURE_2D, t.t); }
      gl.uniform3i(u.uRows, ...this.rows);
      gl.uniform1i(u.uShPer, this.shPer);
      gl.uniform1i(u.uDegree, Math.min(opts.degree ?? 3, this.degree));
      gl.uniform2f(u.uFade, ...(opts.fade || [0, 0]));
      gl.uniform2f(u.uCrease, opts.crease || 0, opts.creaseFade ?? 0.02);
      const R = cam.R;   // GLSL mat3 is column-major
      gl.uniformMatrix3fv(u.uViewR, false, [R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]]);
      gl.uniform3f(u.uViewT, ...cam.t);
      gl.uniform3f(u.uCamPos, ...cam.pos);
      gl.uniform4f(u.uIntr, cam.fx, cam.fy, cam.cx, cam.cy);
      gl.uniform4f(u.uLim, (1.15 * W - cam.cx) / cam.fx, (1.15 * H - cam.cy) / cam.fy, (-0.15 * W - cam.cx) / cam.fx, (-0.15 * H - cam.cy) / cam.fy);
      gl.uniform2f(u.uViewport, W, H);
      const L = opts.light || { view: 0 };
      gl.uniform1i(u.uLightView, L.view);
      if (L.view) {
        gl.uniform1f(u.uNormalMix, L.normalMix);
        gl.uniformMatrix3fv(u.uWR, false, L.WR);
        gl.uniform3f(u.uSky, ...L.sky); gl.uniform3f(u.uGround, ...L.ground); gl.uniform3f(u.uUp, ...L.up);
        gl.uniform4f(u.uShade, L.wrap, L.exposure, L.spec, L.shininess);
        gl.uniform3i(u.uShadeOpt, L.linear ? 1 : 0, L.shoulder ? 1 : 0, L.dcOnly ? 1 : 0);
        const ls = L.lights.slice(0, MAX_LIGHTS), pad = (k, m) => { const a = new Float32Array(MAX_LIGHTS * m); ls.forEach((x, j) => a.set(x[k], j * m)); return a; };
        gl.uniform1i(u.uNLights, ls.length);
        gl.uniform4fv(u['uLPos[0]'], pad('pos', 4)); gl.uniform4fv(u['uLCol[0]'], pad('col', 4));
        gl.uniform4fv(u['uLSpot[0]'], pad('spot', 4)); gl.uniform1fv(u['uLSoft[0]'], pad('soft', 1));
        gl.uniform1i(u.uShadowOn, shadowOn && ls.length ? 1 : 0);
        if (shadowOn) {
          const sh = L.shadow;
          gl.activeTexture(gl.TEXTURE5); gl.bindTexture(gl.TEXTURE_2D, this.shadowMap.t); gl.uniform1i(u.uShadow, 5);
          gl.uniformMatrix4fv(u.uLVP, false, sh.vp);
          gl.uniform3f(u.uLEye, ...sh.eye); gl.uniform3f(u.uLFwd, ...sh.fwd);
          gl.uniform4f(u.uShadowP, sh.bias, sh.soft, sh.strength, SHADOW_SIZE);
          gl.uniform2f(u.uShadowT, sh.texel / SHADOW_SIZE, sh.ortho ? 1 : 0);
        }
      }
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);   // premultiplied, back to front
      gl.bindVertexArray(this.vao);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.count);
      gl.bindVertexArray(null);
      gl.disable(gl.BLEND);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    const { p, u } = this.blit;
    gl.useProgram(p);
    gl.activeTexture(gl.TEXTURE4);
    gl.bindTexture(gl.TEXTURE_2D, this.fboTex);
    gl.uniform1i(u.uImg, 4);
    gl.uniform3f(u.uBg, ...opts.bg);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
}
