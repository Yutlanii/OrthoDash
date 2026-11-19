// Visor de nube de puntos en WebGL2 (dibujado por la GPU a través de ANGLE/Direct3D).
// Los puntos se suben una vez como uint16 cuantizados (6 B) + RGB (3 B).
// Con color RGB se dibujan los puntos tal cual (como Metashape); con color por altura, un
// "eye-dome lighting" (EDL) suave en una segunda pasada realza el relieve sin normales.
"use strict";

const Nube3D = (() => {
  const VS = `#version 300 es
  in vec3 aPos; in vec3 aCol;
  uniform mat4 uMVP; uniform vec3 uMin; uniform vec3 uRango; uniform float uTam;
  uniform int uModo; uniform vec2 uZ; uniform sampler2D uPaleta;
  out vec3 vCol;
  void main() {
    vec3 p = uMin + aPos * uRango;
    gl_Position = uMVP * vec4(p, 1.0);
    gl_PointSize = uTam;
    if (uModo == 1) vCol = texture(uPaleta, vec2(clamp((p.z - uZ.x) / (uZ.y - uZ.x), 0.0, 1.0), 0.5)).rgb;
    else vCol = aCol;
  }`;
  const FS = `#version 300 es
  precision highp float; in vec3 vCol; out vec4 o;
  void main() { o = vec4(vCol, 1.0); }`;
  const VS_Q = `#version 300 es
  out vec2 vUv; void main() { vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2); vUv = p; gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }`;
  const FS_EDL = `#version 300 es
  precision highp float; in vec2 vUv; out vec4 o;
  uniform sampler2D uColor; uniform sampler2D uProf; uniform vec2 uPx; uniform float uNear; uniform float uFar; uniform float uFuerza; uniform vec3 uFondo;
  float lin(float d) { float z = d * 2.0 - 1.0; return (2.0 * uNear * uFar) / (uFar + uNear - z * (uFar - uNear)); }
  void main() {
    float d = texture(uProf, vUv).r;
    if (d >= 1.0) { o = vec4(uFondo, 1.0); return; }
    vec3 c = texture(uColor, vUv).rgb;
    if (uFuerza <= 0.0) { o = vec4(c, 1.0); return; }   // color puro (como Metashape)
    float l = log2(lin(d));
    float s = 0.0;
    vec2 dirs[8] = vec2[](vec2(1,0), vec2(-1,0), vec2(0,1), vec2(0,-1), vec2(.707,.707), vec2(-.707,.707), vec2(.707,-.707), vec2(-.707,-.707));
    for (int i = 0; i < 8; i++) {
      float dn = texture(uProf, vUv + dirs[i] * uPx * 1.6).r;
      float ln = dn >= 1.0 ? l + 3.0 : log2(lin(dn));
      s += max(0.0, l - ln);
    }
    float sh = exp(-s / 8.0 * 300.0 * uFuerza);
    o = vec4(c * sh, 1.0);
  }`;

  // paleta de altura: la misma (sobria) que el mapa de elevación
  const PALETA = [[0, [38, 70, 83]], [.25, [42, 116, 110]], [.5, [138, 160, 110]], [.72, [214, 190, 140]], [.88, [168, 120, 84]], [1, [240, 236, 228]]];

  function compilar(gl, vs, fs) {
    const p = gl.createProgram();
    for (const [tipo, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
      const s = gl.createShader(tipo); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      gl.attachShader(p, s);
    }
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    const u = {}; const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) { const a = gl.getActiveUniform(p, i); u[a.name] = gl.getUniformLocation(p, a.name); }
    return { p, u };
  }

  // --- álgebra mínima (matrices column-major) ---
  function perspectiva(fov, asp, n, f) {
    const t = 1 / Math.tan(fov / 2), r = 1 / (n - f);
    return [t / asp, 0, 0, 0, 0, t, 0, 0, 0, 0, (n + f) * r, -1, 0, 0, 2 * n * f * r, 0];
  }
  function mirar(ojo, obj, arriba) {
    const z = norm(sub(ojo, obj)), x = norm(cruz(arriba, z)), y = cruz(z, x);
    return [x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -punto(x, ojo), -punto(y, ojo), -punto(z, ojo), 1];
  }
  function mult(a, b) {
    const o = new Array(16);
    for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
      let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k]; o[c * 4 + r] = s;
    }
    return o;
  }
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const cruz = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const punto = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const norm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

  class Visor {
    constructor(canvas) {
      this.c = canvas;
      const gl = this.gl = canvas.getContext("webgl2", { antialias: false, powerPreference: "high-performance", preserveDrawingBuffer: false });
      if (!gl) throw new Error("WebGL2 no disponible");
      this.prog = compilar(gl, VS, FS);
      this.edl = compilar(gl, VS_Q, FS_EDL);
      this.tam = 1.5; this.modo = 0; this.n = 0; this.fondo = [0.08, 0.09, 0.1];
      this.yaw = -Math.PI / 2.4; this.pitch = 0.75; this.dist = 100; this.obj = [0, 0, 0];
      this._paleta();
      this._eventos();
      this.pendiente = false;
      new ResizeObserver(() => this.redibujar()).observe(canvas);
    }

    _paleta() {
      const gl = this.gl, px = new Uint8Array(256 * 4);
      for (let i = 0; i < 256; i++) {
        const t = i / 255; let k = 0; while (k < PALETA.length - 2 && t > PALETA[k + 1][0]) k++;
        const [t0, c0] = PALETA[k], [t1, c1] = PALETA[k + 1], f = (t - t0) / (t1 - t0);
        for (let j = 0; j < 3; j++) px[i * 4 + j] = c0[j] + (c1[j] - c0[j]) * f;
        px[i * 4 + 3] = 255;
      }
      this.texPal = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, this.texPal);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, px);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    }

    cargar(meta, buf) {
      const gl = this.gl, n = meta.puntos;
      const pos = new Uint16Array(buf, 0, n * 3), col = new Uint8Array(buf, n * 6, n * 3);
      // centro de la caja en el origen (precisión float32 en la GPU)
      const lado = meta.escala.map((e) => e * 65535);
      this.rango = lado; this.min = [-lado[0] / 2, -lado[1] / 2, -lado[2] / 2];
      this.centroReal = meta.min.map((m, i) => m + lado[i] / 2 + meta.origen[i]);
      // rango de altura robusto para la paleta (muestra)
      const zs = []; for (let i = 0; i < n; i += Math.max(1, (n / 20000) | 0)) zs.push(pos[i * 3 + 2]);
      zs.sort((a, b) => a - b);
      const q = (p) => this.min[2] + zs[Math.floor(p * (zs.length - 1))] / 65535 * lado[2];
      this.z = [q(0.02), q(0.98)];
      this.vao = gl.createVertexArray(); gl.bindVertexArray(this.vao);
      const b1 = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b1); gl.bufferData(gl.ARRAY_BUFFER, pos, gl.STATIC_DRAW);
      const lp = gl.getAttribLocation(this.prog.p, "aPos"); gl.enableVertexAttribArray(lp); gl.vertexAttribPointer(lp, 3, gl.UNSIGNED_SHORT, true, 0, 0);
      const b2 = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b2); gl.bufferData(gl.ARRAY_BUFFER, col, gl.STATIC_DRAW);
      const lc = gl.getAttribLocation(this.prog.p, "aCol"); gl.enableVertexAttribArray(lc); gl.vertexAttribPointer(lc, 3, gl.UNSIGNED_BYTE, true, 0, 0);
      gl.bindVertexArray(null);
      this.n = n;
      this.vistaInicial();
    }

    acercar(f) { this.dist = Math.max(0.5, this.dist * f); this.redibujar(); }

    vistaInicial() {
      if (!this.rango) return;
      this.obj = [0, 0, 0];
      this.dist = Math.max(this.rango[0], this.rango[1]) * 0.95;
      this.yaw = -Math.PI / 2.4; this.pitch = 0.75;
      this.redibujar();
    }

    _fbo(w, h) {
      const gl = this.gl;
      if (this.fb && this.fbw === w && this.fbh === h) return;
      this.fbw = w; this.fbh = h;
      if (this.fb) { gl.deleteFramebuffer(this.fb); gl.deleteTexture(this.tc); gl.deleteTexture(this.td); }
      this.tc = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, this.tc);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      this.td = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, this.td);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.DEPTH_COMPONENT32F, w, h, 0, gl.DEPTH_COMPONENT, gl.FLOAT, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      this.fb = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, this.fb);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.tc, 0);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, this.td, 0);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }

    redibujar() {
      if (this.pendiente) return;
      this.pendiente = true;
      requestAnimationFrame(() => { this.pendiente = false; this._dibujar(); });
    }

    _dibujar() {
      const gl = this.gl, c = this.c, dpr = window.devicePixelRatio || 1;
      const w = Math.max(1, Math.round(c.clientWidth * dpr)), h = Math.max(1, Math.round(c.clientHeight * dpr));
      if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
      if (!this.n || c.clientWidth === 0) return;
      this._fbo(w, h);
      const lado = Math.max(...this.rango);
      const near = Math.max(this.dist * 0.01, lado * 1e-4), far = this.dist + lado * 3;
      const cp = Math.cos(this.pitch);
      const ojo = [this.obj[0] + this.dist * cp * Math.cos(this.yaw), this.obj[1] + this.dist * cp * Math.sin(this.yaw), this.obj[2] + this.dist * Math.sin(this.pitch)];
      const mvp = mult(perspectiva(0.8, w / h, near, far), mirar(ojo, this.obj, [0, 0, 1]));
      this.ojo = ojo;

      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fb);
      gl.viewport(0, 0, w, h);
      gl.clearColor(0, 0, 0, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);
      const P = this.prog; gl.useProgram(P.p);
      gl.uniformMatrix4fv(P.u.uMVP, false, mvp);
      gl.uniform3fv(P.u.uMin, this.min); gl.uniform3fv(P.u.uRango, this.rango);
      gl.uniform1f(P.u.uTam, this.tam * dpr); gl.uniform1i(P.u.uModo, this.modo); gl.uniform2fv(P.u.uZ, this.z);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.texPal); gl.uniform1i(P.u.uPaleta, 0);
      gl.bindVertexArray(this.vao);
      gl.drawArrays(gl.POINTS, 0, this.n);
      gl.bindVertexArray(null);

      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.disable(gl.DEPTH_TEST);
      const E = this.edl; gl.useProgram(E.p);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.tc); gl.uniform1i(E.u.uColor, 0);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.td); gl.uniform1i(E.u.uProf, 1);
      gl.uniform2f(E.u.uPx, 1 / w, 1 / h); gl.uniform1f(E.u.uNear, near); gl.uniform1f(E.u.uFar, far);
      gl.uniform1f(E.u.uFuerza, this.modo === 1 ? 0.45 : 0.0);   // EDL suave sólo en el color por altura
      gl.uniform3fv(E.u.uFondo, this.fondo);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    _eventos() {
      const c = this.c; let arr = null;
      c.addEventListener("contextmenu", (e) => e.preventDefault());
      c.addEventListener("pointerdown", (e) => { arr = { x: e.clientX, y: e.clientY, b: e.button, s: e.shiftKey }; c.setPointerCapture(e.pointerId); });
      c.addEventListener("pointerup", () => { arr = null; });
      c.addEventListener("pointermove", (e) => {
        if (!arr) return;
        const dx = e.clientX - arr.x, dy = e.clientY - arr.y; arr.x = e.clientX; arr.y = e.clientY;
        if (arr.b === 2 || arr.b === 1 || arr.s) {      // mover el objetivo en el plano de la vista
          const k = this.dist * 0.0016;
          const der = [-Math.sin(this.yaw), Math.cos(this.yaw), 0];
          const ade = [-Math.cos(this.yaw), -Math.sin(this.yaw), 0];
          for (let i = 0; i < 3; i++) this.obj[i] += (-dx * der[i] + dy * ade[i]) * k;
        } else {
          this.yaw -= dx * 0.006;
          this.pitch = Math.min(1.55, Math.max(0.05, this.pitch + dy * 0.006));
        }
        this.redibujar();
      });
      c.addEventListener("wheel", (e) => {
        e.preventDefault();
        this.dist = Math.max(0.5, this.dist * Math.exp(e.deltaY * 0.0012));
        this.redibujar();
      }, { passive: false });
      c.addEventListener("dblclick", () => this.vistaInicial());
    }
  }

  return { Visor };
})();
