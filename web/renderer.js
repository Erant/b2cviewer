// WebGL2 splat renderer. The vertex shader poses each splat from the cage (b2ctrain cage.cu pose_bound and the
// stretch fade), evaluates its SH colour in the triangle's canonical frame, and projects it as b2ctrain's rasteriser
// does (EWA, 0.3 px blur, alpha from the 1/255 contour).
import { SPLAT_TEXELS, FACE_TEXELS } from './rig.js';

const VS = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;

layout(location = 0) in vec2 aCorner;
layout(location = 1) in uint aIndex;

uniform sampler2D uSplat, uSH, uFace;
uniform ivec3 uRows;      // items per texture row: splat, sh, face
uniform int uShPer, uDegree;
uniform vec2 uFade;       // stretch fade start, end (0 = off)
uniform mat3 uViewR;      // world -> camera (OpenCV: x right, y down, z ahead)
uniform vec3 uViewT, uCamPos;
uniform vec4 uIntr;       // fx, fy, cx, cy (pixels)
uniform vec4 uLim;        // EWA Jacobian clamp: pos x, pos y, neg x, neg y
uniform vec2 uViewport;

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

void cull() { gl_Position = vec4(0., 0., 2., 1.); }

void main() {
  int i = int(aIndex);
  vec4 t0 = fetch(uSplat, i, ${SPLAT_TEXELS}, uRows.x, 0), t1 = fetch(uSplat, i, ${SPLAT_TEXELS}, uRows.x, 1);
  vec4 t2 = fetch(uSplat, i, ${SPLAT_TEXELS}, uRows.x, 2);
  vec3 pos = t0.xyz; float op = t0.w; vec4 q = t1; vec3 ls = t2.xyz; int f = int(t2.w);
  vec4 shf = vec4(1., 0., 0., 0.);
  if (f >= 0) {
    vec4 t3 = fetch(uSplat, i, ${SPLAT_TEXELS}, uRows.x, 3), t4 = fetch(uSplat, i, ${SPLAT_TEXELS}, uRows.x, 4);
    float b1 = t3.w, b2 = t4.x, w0 = 1. - b1 - b2;
    vec4 A = fetch(uFace, f, ${FACE_TEXELS}, uRows.z, 0), B = fetch(uFace, f, ${FACE_TEXELS}, uRows.z, 1), C = fetch(uFace, f, ${FACE_TEXELS}, uRows.z, 2);
    vec4 R0 = fetch(uFace, f, ${FACE_TEXELS}, uRows.z, 3), R1 = fetch(uFace, f, ${FACE_TEXELS}, uRows.z, 4), R2 = fetch(uFace, f, ${FACE_TEXELS}, uRows.z, 5);
    vec4 dq = fetch(uFace, f, ${FACE_TEXELS}, uRows.z, 6);
    float k = R0.w, k0 = R1.w, ratio = R2.w;
    vec3 off = t3.xyz;
    pos = A.xyz * w0 + B.xyz * b1 + C.xyz * b2 + vec3(dot(R0.xyz, off), dot(R1.xyz, off), dot(R2.xyz, off)) * k;
    if (uFade.x > 0. && uFade.y > uFade.x) {   // stretch fade: opacity -> 0 as the triangle grows from start to end
      float fa = clamp((uFade.y - ratio) / (uFade.y - uFade.x), 0., 1.);
      if (fa < 1.) { float sg = clamp(fa / (1. + exp(-op)), 1e-6, 1. - 1e-6); op = log(sg / (1. - sg)); }
    }
    q = qmul(dq, q);
    ls += (k0 > 0. && k > 0.) ? log(k / k0) : 0.;
    shf = dq;
  }

  // projection (splat_math.cuh project_one)
  vec3 mc = uViewR * pos + uViewT;
  if (!(mc.z >= 0.01)) { cull(); return; }
  float opac = 1. / (1. + exp(-op));
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
  setFace(face, nf) { this.rows[2] = this._texture('face', face, nf, FACE_TEXELS, 2); }
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

  // cam: { R (row-major world -> camera, OpenCV), t, pos, fx, fy, cx, cy }; opts: { fade, degree, bg }
  render(cam, opts) {
    const gl = this.gl, W = this.canvas.width, H = this.canvas.height;
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
      const R = cam.R;   // GLSL mat3 is column-major
      gl.uniformMatrix3fv(u.uViewR, false, [R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]]);
      gl.uniform3f(u.uViewT, ...cam.t);
      gl.uniform3f(u.uCamPos, ...cam.pos);
      gl.uniform4f(u.uIntr, cam.fx, cam.fy, cam.cx, cam.cy);
      gl.uniform4f(u.uLim, (1.15 * W - cam.cx) / cam.fx, (1.15 * H - cam.cy) / cam.fy, (-0.15 * W - cam.cx) / cam.fx, (-0.15 * H - cam.cy) / cam.fy);
      gl.uniform2f(u.uViewport, W, H);
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
