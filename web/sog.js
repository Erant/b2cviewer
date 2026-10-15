// Reader for the web forms of the b2c files: ZIP archives of WebP textures and a few side files.
//   readZip       the entries of a ZIP (stored or deflated; no ZIP64), deflate through DecompressionStream
//   parseSog      a PlayCanvas SOG v2 splat (meta.json + WebP textures) in the renderer's form, and, when meta.json has
//                 a `b2c` key (b2cgltf's web subject), the binding textures and the rig.glb beside it
// Plain SOG files (b2csite's splats, splat-transform output) carry no rig: they are shown unposed.

const SH_PER_BAND = [0, 3, 8, 15];

export const isZip = buf => buf.byteLength >= 4 && new DataView(buf).getUint32(0, true) === 0x04034b50;

// name -> Uint8Array for every file entry, read through the central directory.
export async function readZip(buf) {
  const dv = new DataView(buf), u8 = new Uint8Array(buf);
  let eocd = -1;
  for (let o = buf.byteLength - 22; o >= Math.max(0, buf.byteLength - 22 - 65535); o--)
    if (dv.getUint32(o, true) === 0x06054b50) { eocd = o; break; }
  if (eocd < 0) throw new Error('not a ZIP archive (no end of central directory)');
  const count = dv.getUint16(eocd + 10, true);
  let o = dv.getUint32(eocd + 16, true);
  if (o === 0xffffffff || count === 0xffff) throw new Error('ZIP64 archives are not supported');
  const out = new Map();
  for (let k = 0; k < count; k++) {
    if (dv.getUint32(o, true) !== 0x02014b50) throw new Error('corrupt ZIP central directory');
    const method = dv.getUint16(o + 10, true), csize = dv.getUint32(o + 20, true), size = dv.getUint32(o + 24, true);
    const nlen = dv.getUint16(o + 28, true), xlen = dv.getUint16(o + 30, true), clen = dv.getUint16(o + 32, true);
    const local = dv.getUint32(o + 42, true), name = new TextDecoder().decode(u8.subarray(o + 46, o + 46 + nlen));
    o += 46 + nlen + xlen + clen;
    if (name.endsWith('/')) continue;
    const start = local + 30 + dv.getUint16(local + 26, true) + dv.getUint16(local + 28, true), raw = u8.subarray(start, start + csize);
    if (method === 0) out.set(name, raw);
    else if (method === 8) out.set(name, await inflate(raw, size));
    else throw new Error(`${name}: ZIP compression method ${method} is not supported`);
  }
  return out;
}

async function inflate(raw, size) {
  const stream = new Blob([raw]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
  const data = new Uint8Array(await new Response(stream).arrayBuffer());
  if (data.length !== size) throw new Error('ZIP entry inflated to the wrong size');
  return data;
}

// WebP -> { width, height, data: RGBA bytes } exactly as stored. The data textures put values in every channel,
// alpha included, so the decode must not premultiply: a WebGL2 upload with premultiplication and colour conversion
// off, read back from a framebuffer (as PlayCanvas reads SOG textures).
let gl = null;
async function decodeWebpGL(bytes) {
  const bmp = await createImageBitmap(new Blob([bytes], { type: 'image/webp' }), { premultiplyAlpha: 'none', colorSpaceConversion: 'none' });
  gl ??= new OffscreenCanvas(1, 1).getContext('webgl2');
  if (!gl) throw new Error('reading a .sog needs WebGL2 in a worker (OffscreenCanvas)');
  const { width, height } = bmp, tex = gl.createTexture(), fb = gl.createFramebuffer();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
  gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, bmp);
  bmp.close();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  const data = new Uint8Array(width * height * 4);
  gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, data);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.deleteFramebuffer(fb); gl.deleteTexture(tex);
  return { width, height, data };
}

// The splat of a SOG v2 archive (files: readZip's map) in the renderer's form: pos, logit opacity, w x y z rotation,
// log scales, SH coefficient-major per splat (DC first). decode: WebP bytes -> { width, height, data }.
export async function parseSog(files, decode = decodeWebpGL) {
  const metaBytes = files.get('meta.json');
  if (!metaBytes) throw new Error('not a SOG file (no meta.json)');
  const meta = JSON.parse(new TextDecoder().decode(metaBytes));
  if (meta.version !== 2) throw new Error(`SOG version ${meta.version} is not supported (this viewer reads 2)`);
  const n = meta.count;
  const img = async name => {
    const b = files.get(name);
    if (!b) throw new Error(`the SOG names ${name}, which it does not contain`);
    const im = await decode(b);
    if (im.width * im.height < n && name !== meta.shN?.files?.[0]) throw new Error(`${name} is ${im.width}x${im.height}, too small for ${n} splats`);
    return im;
  };
  const [ml, mu, qt, sc, s0] = await Promise.all([img(meta.means.files[0]), img(meta.means.files[1]), img(meta.quats.files[0]),
    img(meta.scales.files[0]), img(meta.sh0.files[0])]);
  const bands = meta.shN ? meta.shN.bands ?? 0 : 0, coeffs = SH_PER_BAND[bands], K = 1 + coeffs;
  const [cent, lab] = bands ? await Promise.all([img(meta.shN.files[0]), img(meta.shN.files[1])]) : [null, null];

  const pos = new Float32Array(n * 3), op = new Float32Array(n), quat = new Float32Array(n * 4), ls = new Float32Array(n * 3);
  const sh = new Float32Array(n * K * 3);
  const { mins, maxs } = meta.means, scb = meta.scales.codebook, ccb = meta.sh0.codebook, ncb = meta.shN?.codebook;
  const unlog = t => Math.sign(t) * (Math.exp(Math.abs(t)) - 1);
  for (let i = 0; i < n; i++) {
    const p = i * 4;
    for (let k = 0; k < 3; k++) {
      const t = mins[k] + (maxs[k] - mins[k]) * (((mu.data[p + k] << 8) | ml.data[p + k]) / 65535);
      pos[i * 3 + k] = unlog(t);
      ls[i * 3 + k] = scb[sc.data[p + k]];
      sh[i * K * 3 + k] = ccb[s0.data[p + k]];
    }
    const o = Math.min(Math.max(s0.data[p + 3] / 255, 1e-7), 1 - 1e-7);
    op[i] = Math.log(o / (1 - o));
    // smallest-three: three stored components, the fourth (largest) recovered; alpha 252 + its index in x y z w
    const a = (qt.data[p] / 255 - 0.5) * Math.SQRT2, b = (qt.data[p + 1] / 255 - 0.5) * Math.SQRT2, c = (qt.data[p + 2] / 255 - 0.5) * Math.SQRT2;
    const d = Math.sqrt(Math.max(0, 1 - (a * a + b * b + c * c)));
    let x, y, z, w;
    switch (qt.data[p + 3] - 252) {
      case 0: [x, y, z, w] = [a, b, c, d]; break;
      case 1: [x, y, z, w] = [d, b, c, a]; break;
      case 2: [x, y, z, w] = [b, d, c, a]; break;
      case 3: [x, y, z, w] = [b, c, d, a]; break;
      default: throw new Error(`quats.webp: bad mode ${qt.data[p + 3]} at splat ${i}`);
    }
    quat[i * 4] = w; quat[i * 4 + 1] = x; quat[i * 4 + 2] = y; quat[i * 4 + 3] = z;
    if (bands) {   // palette entry: 64 per centroid row, `coeffs` texels each, channel = colour
      const e = lab.data[p] | (lab.data[p + 1] << 8), u = (e % 64) * coeffs, v = Math.floor(e / 64);
      for (let k = 0; k < coeffs; k++) for (let ch = 0; ch < 3; ch++)
        sh[(i * K + 1 + k) * 3 + ch] = ncb[cent.data[((u + k) + v * cent.width) * 4 + ch]];
    }
  }
  const splat = { n, degree: bands, K, pos, op, quat, ls, sh };
  if (!meta.b2c) return { meta, splat, binding: null, rig: null };

  // b2cgltf's web subject: the binding in the SOG's splat order, and the rig (skeleton, skin, cage) as a .glb
  const B = meta.b2c;
  if (B.version !== 1) throw new Error(`b2c web subject version ${B.version} is not supported (this viewer reads 1)`);
  const [bf, bl, bu] = await Promise.all([img(B.binding.face.file), img(B.binding.offset.files[0]), img(B.binding.offset.files[1])]);
  const face = new Int32Array(n), bary = new Float32Array(n * 2), offset = new Float32Array(n * 3);
  const { mins: omin, maxs: omax } = B.binding.offset;
  for (let i = 0; i < n; i++) {
    const p = i * 4, f = bf.data[p] | (bf.data[p + 1] << 8);
    face[i] = f === 0xffff ? -1 : f;
    bary[i * 2] = bf.data[p + 2] / 255; bary[i * 2 + 1] = bf.data[p + 3] / 255;
    for (let k = 0; k < 3; k++) offset[i * 3 + k] = omin[k] + (omax[k] - omin[k]) * (((bu.data[p + k] << 8) | bl.data[p + k]) / 65535);
  }
  const rig = files.get(B.rig);
  if (!rig) throw new Error(`the SOG names its rig ${B.rig}, which it does not contain`);
  return { meta, splat, binding: { face, bary, offset, posing: B.binding.posing }, rig: rig.slice().buffer };
}
