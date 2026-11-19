"""Nube densa en GPU (CUDA vía CuPy): mismo algoritmo que densa.py, todo en la tarjeta gráfica.

Por par de imágenes: rectificación (con distorsión) -> census 9x7 -> volumen de costo ->
Semi-Global Matching de 8 direcciones -> ganador + subpíxel + unicidad -> verificación
izquierda-derecha -> filtro de manchas (componentes conexas) -> triangulación y proyección al
mapa de profundidad de la cámara. Después, por cámara: media de los pares que coinciden + filtro
guiado por la foto (suaviza superficies, respeta bordes). Luego fusión multivista: cada punto se
promedia con las cámaras vecinas que lo confirman, lo que funde en una sola superficie las capas que
dejan pequeños desfases entre vistas (medido: ruido en suelo 20 -> 4,5-6,7 cm). Por último, fusión
en vóxeles con confianza, también en GPU.
"""
from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from . import densa
from .alineacion import Vista, k_y_distorsion

warnings.filterwarnings("ignore", message="CUDA path could not be detected")
import cupy as cp  # noqa: E402
from cupyx.scipy import ndimage as ndi  # noqa: E402

P1, P2 = 10, 120          # penalizaciones SGM para costo census 9x7 (0..63)
UNICIDAD = 10             # % como uniquenessRatio de SGBM
COSTO_FUERA = 64

_CODIGO = r"""
extern "C" {

// Remuestreo bilineal con modelo OpenCV: píxel destino -> rayo (Kd^-1) -> R^T -> distorsión -> Ks
// p: Rt[9], fxd, fyd, cxd, cyd, fxs, fys, cxs, cys, k1, k2, p1, p2, k3
__global__ void remap(const unsigned char* src, int sw, int sh, int ch, unsigned char* dst, int dw, int dh,
                      const float* p) {
    int u = blockIdx.x * blockDim.x + threadIdx.x, v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= dw || v >= dh) return;
    float x = (u - p[11]) / p[9], y = (v - p[12]) / p[10];
    float X = p[0] * x + p[1] * y + p[2], Y = p[3] * x + p[4] * y + p[5], Z = p[6] * x + p[7] * y + p[8];
    int o = (v * dw + u) * ch;
    if (Z <= 1e-6f) { for (int c = 0; c < ch; c++) dst[o + c] = 0; return; }
    x = X / Z; y = Y / Z;
    float r2 = x * x + y * y, rad = 1.f + r2 * (p[17] + r2 * (p[18] + r2 * p[21]));
    float xd = x * rad + 2.f * p[19] * x * y + p[20] * (r2 + 2.f * x * x);
    float yd = y * rad + p[19] * (r2 + 2.f * y * y) + 2.f * p[20] * x * y;
    float su = p[13] * xd + p[15], sv = p[14] * yd + p[16];
    int x0 = (int)floorf(su), y0 = (int)floorf(sv);
    if (x0 < 0 || y0 < 0 || x0 + 1 >= sw || y0 + 1 >= sh) { for (int c = 0; c < ch; c++) dst[o + c] = 0; return; }
    float ax = su - x0, ay = sv - y0;
    for (int c = 0; c < ch; c++) {
        const unsigned char* s = src + (y0 * sw + x0) * ch + c;
        float val = (1 - ay) * ((1 - ax) * s[0] + ax * s[ch]) + ay * ((1 - ax) * s[sw * ch] + ax * s[sw * ch + ch]);
        dst[o + c] = (unsigned char)fminf(255.f, fmaxf(1.f, val + 0.5f));  // 0 queda reservado para "fuera"
    }
}

__global__ void census(const unsigned char* img, unsigned long long* out, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    unsigned char c = img[y * W + x];
    unsigned long long b = 0;
    for (int dy = -3; dy <= 3; dy++)
        for (int dx = -4; dx <= 4; dx++) {
            if (dx == 0 && dy == 0) continue;
            int xx = min(max(x + dx, 0), W - 1), yy = min(max(y + dy, 0), H - 1);
            b = (b << 1) | (img[yy * W + xx] < c ? 1ull : 0ull);
        }
    out[y * W + x] = b;
}

// Agregación SGM a lo largo de una dirección. Un bloque por línea de recorrido, un hilo por disparidad.
// El costo census (hamming(census_izq(x), census_der(x - d))) se calcula al vuelo: no se guarda
// el volumen de costo, lo que ahorra 1/3 de la memoria.
__device__ __forceinline__ int costo(const unsigned long long* cl, const unsigned long long* cr,
                                     const unsigned char* il, const unsigned char* ir, int W, int x, int y, int d) {
    int xr = x - d;
    if (xr < 0 || il[y * W + x] == 0 || ir[y * W + xr] == 0) return 64;
    return __popcll(cl[y * W + x] ^ cr[y * W + xr]);
}

__global__ void agregar(const unsigned long long* cl, const unsigned long long* cr, const unsigned char* il,
                        const unsigned char* ir, unsigned short* S, int W, int H, int D, int dx, int dy,
                        int P1, int P2) {
    // Doble búfer para L y para los mínimos por warp: una sola __syncthreads por paso.
    extern __shared__ int sh[];
    int nt = blockDim.x, nw = nt >> 5;
    int* Lb = sh;              // [2][nt]
    int* wm = sh + 2 * nt;     // [2][32]
    int d = threadIdx.x, lane = d & 31, warp = d >> 5;
    int linea = blockIdx.x;
    int x, y;
    // punto de partida de la línea según la dirección
    if (dy == 0) { y = linea; x = dx > 0 ? 0 : W - 1; }
    else if (dx == 0) { x = linea; y = dy > 0 ? 0 : H - 1; }
    else {  // diagonales: las líneas arrancan en el borde superior/inferior y en el lateral
        if (linea < W) { x = linea; y = dy > 0 ? 0 : H - 1; }
        else { x = dx > 0 ? 0 : W - 1; y = dy > 0 ? (linea - W + 1) : (linea - W); }
    }
    const int GRANDE = 1 << 28;
    int minp = 0, cur = 0;
    bool primero = true;
    while (x >= 0 && x < W && y >= 0 && y < H) {
        const int* Lp = Lb + cur * nt;
        int* Ln = Lb + (cur ^ 1) * nt;
        int L = GRANDE;
        if (d < D) {
            int c = costo(cl, cr, il, ir, W, x, y, d);
            if (primero) L = c;
            else {
                int best = Lp[d];
                if (d > 0) best = min(best, Lp[d - 1] + P1);
                if (d < D - 1) best = min(best, Lp[d + 1] + P1);
                best = min(best, minp + P2);
                L = c + best - minp;
            }
        }
        Ln[d] = L;
        int v = L;
        for (int o = 16; o > 0; o >>= 1) v = min(v, __shfl_xor_sync(0xffffffffu, v, o));
        if (lane == 0) wm[cur * 32 + warp] = v;
        __syncthreads();
        minp = GRANDE;
        for (int i = 0; i < nw; i++) minp = min(minp, wm[cur * 32 + i]);
        if (d < D) {
            size_t idx = ((size_t)y * W + x) * D + d;
            S[idx] += (unsigned short)L;
        }
        primero = false;
        cur ^= 1;
        x += dx; y += dy;
    }
}

// Ganador con subpíxel (parábola) y prueba de unicidad; también disparidad de la imagen derecha.
// Un warp por píxel: las lecturas del volumen son contiguas.
__global__ void ganador(const unsigned short* S, float* disp, int* dispR, int W, int H, int D, int unic) {
    int lane = threadIdx.x & 31;
    int x = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), y = blockIdx.y;
    if (x >= W) return;
    const unsigned short* s = S + ((size_t)y * W + x) * D;
    unsigned int b = 0xffffffffu;
    for (int d = lane; d < D; d += 32) b = min(b, ((unsigned int)s[d] << 16) | (unsigned int)d);
    for (int o = 16; o > 0; o >>= 1) b = min(b, __shfl_xor_sync(0xffffffffu, b, o));
    int bd = b & 0xffff, best = b >> 16;
    bool malo = false;
    for (int d = lane; d < D; d += 32)
        if (abs(d - bd) > 1 && s[d] * (100 - unic) < best * 100) malo = true;
    malo = __any_sync(0xffffffffu, malo);
    unsigned int br = 0xffffffffu;
    for (int d = lane; d < D && x + d < W; d += 32)
        br = min(br, ((unsigned int)S[((size_t)y * W + x + d) * D + d] << 16) | (unsigned int)d);
    for (int o = 16; o > 0; o >>= 1) br = min(br, __shfl_xor_sync(0xffffffffu, br, o));
    if (lane != 0) return;
    float r = -1.f;
    if (!malo && bd > 0 && bd < D - 1) {
        float a = s[bd - 1], bb = s[bd], c = s[bd + 1];
        float den = a + c - 2.f * bb;
        r = bd + (den > 0 ? 0.5f * (a - c) / den : 0.f);
    } else if (!malo) r = (float)bd;
    disp[y * W + x] = r;
    dispR[y * W + x] = br == 0xffffffffu ? -1 : (int)(br & 0xffff);
}

// Fusión multivista: cada píxel de la cámara i se proyecta en la cámara j; si la profundidad de j en
// ese píxel coincide (tolerancia relativa), se suma el punto de j sobre el mismo rayo de j.
// q: fx, fy, cx, cy, R[9], t[3] (cam_from_world) de i y luego de j
__global__ void fusion_vista(const float* Di, int w, int h, const float* Dj, int wj, int hj, const float* q,
                             float tol, float* acc, float* cnt) {
    int u = blockIdx.x * blockDim.x + threadIdx.x, v = blockIdx.y;
    if (u >= w) return;
    float z = Di[v * w + u];
    if (isnan(z)) return;
    const float* a = q; const float* b = q + 16;
    float xc = (u - a[2]) / a[0] * z - a[13], yc = (v - a[3]) / a[1] * z - a[14], zc = z - a[15];
    float X = a[4] * xc + a[7] * yc + a[10] * zc;        // R^T (xc - t)
    float Y = a[5] * xc + a[8] * yc + a[11] * zc;
    float Z = a[6] * xc + a[9] * yc + a[12] * zc;
    float xj = b[4] * X + b[5] * Y + b[6] * Z + b[13];
    float yj = b[7] * X + b[8] * Y + b[9] * Z + b[14];
    float zj = b[10] * X + b[11] * Y + b[12] * Z + b[15];
    if (zj <= 0) return;
    int uj = (int)lrintf(b[0] * xj / zj + b[2]), vj = (int)lrintf(b[1] * yj / zj + b[3]);
    if (uj < 0 || vj < 0 || uj >= wj || vj >= hj) return;
    float dj = Dj[vj * wj + uj];
    if (isnan(dj) || fabsf(dj - zj) > tol * dj) return;
    float s = dj / zj;                                   // mismo rayo de j, a la profundidad de j
    float px = xj * s - b[13], py = yj * s - b[14], pz = zj * s - b[15];
    int k = v * w + u;
    acc[3 * k]     += b[4] * px + b[7] * py + b[10] * pz;
    acc[3 * k + 1] += b[5] * px + b[8] * py + b[11] * pz;
    acc[3 * k + 2] += b[6] * px + b[9] * py + b[12] * pz;
    cnt[k] += 1.f;
}

__global__ void izq_der(float* disp, const int* dispR, const unsigned char* il, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y;
    if (x >= W) return;
    int i = y * W + x;
    float d = disp[i];
    if (d < 0 || il[i] == 0) { disp[i] = -1.f; return; }
    int xr = (int)lrintf(x - d);
    if (xr < 0 || xr >= W || abs(dispR[y * W + xr] - (int)lrintf(d)) > 1) disp[i] = -1.f;
}

// Disparidad rectificada -> punto 3D -> mapa de profundidad (sin distorsión) de la cámara i (z mínimo)
// q: f, B, dmin, cxn, cyn, R1[9] (cámara->rectificado), fx, fy, cx, cy
__global__ void proyectar(const float* disp, int W, int H, const float* q, int* prof, int w, int h) {
    int u = blockIdx.x * blockDim.x + threadIdx.x, v = blockIdx.y;
    if (u >= W) return;
    float d = disp[v * W + u];
    if (d < 0) return;
    d += q[2];
    if (d <= 0.5f) return;
    float f = q[0], Z = f * q[1] / d;
    float X = (u - q[3]) * Z / f, Y = (v - q[4]) * Z / f;
    const float* R = q + 5;   // Xc = R^T Xrect
    float xc = R[0] * X + R[3] * Y + R[6] * Z;
    float yc = R[1] * X + R[4] * Y + R[7] * Z;
    float zc = R[2] * X + R[5] * Y + R[8] * Z;
    if (zc <= 0) return;
    int pu = (int)lrintf(q[14] * xc / zc + q[16]), pv = (int)lrintf(q[15] * yc / zc + q[17]);
    if (pu < 0 || pv < 0 || pu >= w || pv >= h) return;
    atomicMin(prof + pv * w + pu, __float_as_int(zc));
}

// Rellena huecos de 1 px: media 3x3 si al menos 5 vecinos tienen dato (como densa.py)
__global__ void rellenar(const float* a, float* b, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float v = a[y * w + x];
    if (!isnan(v)) { b[y * w + x] = v; return; }
    float s = 0; int n = 0;
    for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {
        int xx = x + i, yy = y + j;
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
        float t = a[yy * w + xx];
        if (!isnan(t)) { s += t; n++; }
    }
    b[y * w + x] = n >= 5 ? s / n : nanf("");
}
}
"""

_mod = None


def _k(nombre):
    global _mod
    if _mod is None:
        _mod = cp.RawModule(code=_CODIGO, options=("-std=c++14", "--use_fast_math"))
    return _mod.get_function(nombre)


def _grid2(w, h, b=(16, 16)):
    return ((w + b[0] - 1) // b[0], (h + b[1] - 1) // b[1]), b


def _remap(src, K_src, D_src, Rt, K_dst, W, H):
    """src (h,w[,c]) uint8 en GPU -> imagen (H,W[,c]) en la rejilla destino."""
    ch = 1 if src.ndim == 2 else src.shape[2]
    out = cp.empty((H, W) if ch == 1 else (H, W, ch), cp.uint8)
    D = np.zeros(5); D[:len(D_src)] = np.ravel(D_src)[:5]
    p = cp.asarray(np.r_[np.ravel(Rt), K_dst[0, 0], K_dst[1, 1], K_dst[0, 2], K_dst[1, 2],
                         K_src[0, 0], K_src[1, 1], K_src[0, 2], K_src[1, 2], D[0], D[1], D[2], D[3], D[4]],
                   cp.float32)
    g, b = _grid2(W, H)
    _k("remap")(g, b, (src, cp.int32(src.shape[1]), cp.int32(src.shape[0]), cp.int32(ch), out,
                       cp.int32(W), cp.int32(H), p))
    return out


DIRECCIONES = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]


class _Volumenes:
    """Búferes de costo y agregación reutilizados entre pares.

    Cada par tiene un tamaño (H, W, D) distinto; si se pidiera memoria nueva cada vez, la caché de
    CuPy se llenaría de bloques no reutilizables hasta agotar la VRAM, y en Windows el controlador
    desborda entonces a la RAM del sistema (medido: 547 s en vez de ~20 s).
    """
    def __init__(self):
        self.n = 0
        self.S = None

    def pedir(self, H, W, D):
        n = H * W * D
        if n > self.n:
            self.S = None
            cp.get_default_memory_pool().free_all_blocks()
            self.n = int(n * 1.15)
            self.S = cp.empty(self.n, cp.uint16)
        S = self.S[:n].reshape(H, W, D)
        S.fill(0)
        return S


_vol = _Volumenes()


def _sgm_franja(ca, cb, a, b, D, caminos):
    H, W = a.shape
    S = _vol.pedir(H, W, D)
    nt = 1 << max(5, (D - 1).bit_length())
    for dx, dy in DIRECCIONES[:caminos]:
        lineas = H if dy == 0 else (W if dx == 0 else W + H - 1)
        _k("agregar")((lineas,), (nt,), (ca, cb, a, b, S, cp.int32(W), cp.int32(H), cp.int32(D), cp.int32(dx),
                                         cp.int32(dy), cp.int32(P1), cp.int32(P2)), shared_mem=(2 * nt + 64) * 4)
    disp = cp.empty((H, W), cp.float32); dispR = cp.empty((H, W), cp.int32)
    _k("ganador")(((W + 3) // 4, H), (128,), (S, disp, dispR, cp.int32(W), cp.int32(H), cp.int32(D), cp.int32(UNICIDAD)))
    _k("izq_der")(((W + 127) // 128, H), (128,), (disp, dispR, a, cp.int32(W), cp.int32(H)))
    return disp


SOLAPE = 64   # filas de solape entre franjas (los caminos verticales/diagonales "arrancan" ahí)


def _sgm(a, b, D, caminos=8):
    """Disparidad (float32, -1 = inválida) de a respecto a b, ambas rectificadas (H, W) uint8.

    Si el volumen de agregación (H*W*D*2 bytes) no cabe en la VRAM libre, se procesa en franjas
    horizontales con solape; las líneas epipolares son horizontales, así que sólo los caminos
    verticales/diagonales se acortan en los bordes de franja.
    """
    H, W = a.shape
    ca = cp.empty((H, W), cp.uint64); cb = cp.empty((H, W), cp.uint64)
    g, bl = _grid2(W, H)
    _k("census")(g, bl, (a, ca, cp.int32(W), cp.int32(H)))
    _k("census")(g, bl, (b, cb, cp.int32(W), cp.int32(H)))
    filas = max(4 * SOLAPE, int(_presupuesto["sgm"] // (W * D * 2)))
    if filas >= H:
        return _sgm_franja(ca, cb, a, b, D, caminos)
    disp = cp.full((H, W), -1, cp.float32)
    paso = filas - 2 * SOLAPE
    for y0 in range(0, H, paso):
        lo, hi = max(0, y0 - SOLAPE), min(H, y0 + paso + SOLAPE)
        d = _sgm_franja(ca[lo:hi], cb[lo:hi], a[lo:hi], b[lo:hi], D, caminos)
        fin = min(H, y0 + paso)
        disp[y0:fin] = d[y0 - lo:fin - lo]
    return disp


def _manchas(d: np.ndarray, speckle: int) -> np.ndarray:
    """Filtro de componentes conexas (como speckleWindowSize/speckleRange=2 de SGBM), en CPU: es
    una pasada lineal con cv2.filterSpeckles, y se ejecuta en hilos mientras la GPU sigue con el
    siguiente par."""
    if speckle > 0:
        d16 = np.where(d >= 0, np.round(d * 16), -16).astype(np.int16)
        cv2.filterSpeckles(d16, -16, speckle, 32)
        d[d16 < 0] = -1
    return d


def _disparidad_par(vi, vj, gi, gj, ei, ej, todas, caminos):
    """(geometría, disparidad en RAM) del par, o None."""
    h, w = gi.shape
    g = densa.geometria_par(vi, vj, ei, ej, w, h, todas)
    if g is None:
        return None
    a = _remap(gi, g.Ki, g.Di, g.Rri.T, g.Kn_i, g.W, g.H)
    b = _remap(gj, g.Kj, g.Dj, g.Rrj.T, g.Kn_j, g.W, g.H)
    return g, cp.asnumpy(_sgm(a, b, g.num, caminos))


def _a_profundidad(g, disp, Kud, w, h):
    """Disparidad rectificada -> mapa de profundidad en la rejilla sin distorsión de la cámara i."""
    prof = cp.full((h, w), 0x7F7FFFFF, cp.int32)   # FLT_MAX como entero
    q = cp.asarray(np.r_[g.f, g.B, g.dmin, g.Kn_i[0, 2], g.Kn_i[1, 2], np.ravel(g.Rri),
                         Kud[0, 0], Kud[1, 1], Kud[0, 2], Kud[1, 2]], cp.float32)
    _k("proyectar")(((g.W + 127) // 128, g.H), (128,), (cp.asarray(disp), cp.int32(g.W), cp.int32(g.H), q, prof,
                                                        cp.int32(w), cp.int32(h)))
    z = prof.view(cp.float32)
    z[prof == 0x7F7FFFFF] = cp.nan
    out = cp.empty_like(z)
    gr, bl = _grid2(w, h)
    _k("rellenar")(gr, bl, (z, out, cp.int32(w), cp.int32(h)))
    return out


_presupuesto = {"total": 2e9, "sgm": 1e9, "puntos": 1e7}


def limitar_memoria() -> int:
    """Ajusta el uso de VRAM a la memoria realmente libre (otros programas también la usan).

    El desborde a la RAM del sistema que hace Windows es 5-10x más lento que trabajar por partes.
    """
    from .gpu import vram_libre_mb
    libre = vram_libre_mb() * 1024 ** 2
    total = max(768 * 1024 ** 2, 0.8 * libre)
    cp.get_default_memory_pool().set_limit(size=int(total))
    _presupuesto.update(total=total, sgm=0.45 * total, puntos=total / 250)   # medido: la fusión usa ~200 B/punto
    return int(libre // 1024 ** 2)


def fusionar_por_bloques(X, C, I, n_vistas, vox, minc):
    """densa.fusionar en GPU por franjas de X alineadas a la celda del filtro de aislados (8 vóxeles),
    así cada vóxel y cada celda caen enteros en una franja: el resultado es idéntico al de una sola
    pasada, pero cabe en la VRAM libre."""
    celda = vox * 8
    n_bloques = max(1, int(np.ceil(len(X) / _presupuesto["puntos"])))
    if n_bloques == 1:
        return densa.fusionar(cp.asarray(X), cp.asarray(C), cp.asarray(I), n_vistas, vox, minc, cp)
    col = np.floor(X[:, 0] / celda).astype(np.int64)
    lo, hi = col.min(), col.max() + 1
    cortes = np.quantile(col, np.linspace(0, 1, n_bloques + 1)).astype(np.int64)
    cortes[0], cortes[-1] = lo, hi
    bloque = np.searchsorted(cortes[1:-1], col, side="right").astype(np.int16)
    del col
    orden = np.argsort(bloque, kind="stable")
    limites = np.searchsorted(bloque[orden], np.arange(n_bloques + 1))
    partes = []
    for b in range(n_bloques):
        sel = orden[limites[b]:limites[b + 1]]
        if len(sel):
            partes.append(densa.fusionar(cp.asarray(X[sel]), cp.asarray(C[sel]), cp.asarray(I[sel]), n_vistas, vox, minc, cp))
            cp.get_default_memory_pool().free_all_blocks()
    return tuple(np.concatenate([p[k] for p in partes]) for k in range(3))


TOLERANCIA_PARES = 0.012
TOLERANCIA_FUSION = 0.006   # fracción de la profundidad para fusionar vistas (~30 cm a 50 m)
VECINAS_FUSION = 14         # cámaras vecinas con las que se promedia cada punto (medido: 8 -> 6,4 cm
                            # de ruido en suelo, 14 -> 4,5 cm, mismo tiempo)   # fracción de la profundidad para que un par "coincida" con la mediana
RADIO_GUIADO = 4           # radio del filtro guiado (px a 1400 px de lado)
EPS_GUIADO = 1e-3          # regularización: menor = respeta más los bordes de la imagen


def _filtro_guiado(p, valido, guia, r, eps):
    """Filtro guiado (He et al.) del mapa de profundidad con la foto como guía, ignorando los
    píxeles sin dato (convolución normalizada). Suaviza el ruido de profundidad en superficies
    continuas y respeta los bordes visibles en la imagen; es lo que Metashape aplica a sus mapas de
    profundidad (librería guided-filter en su binario)."""
    m = valido.astype(cp.float32)
    caja = lambda x: ndi.uniform_filter(x, 2 * r + 1, mode="nearest")
    n = cp.maximum(caja(m), 1e-6)
    p0 = cp.where(valido, p, 0).astype(cp.float32)
    ref = cp.sum(p0) / cp.maximum(cp.sum(m), 1)          # se filtra p - ref: mejor precisión float32
    pc = cp.where(valido, p0 - ref, 0)
    mI = caja(guia * m) / n
    mp = caja(pc) / n
    cIp = caja(guia * pc) / n
    vI = caja(guia * guia * m) / n - mI * mI
    a = (cIp - mI * mp) / (vI + eps)
    b = mp - a * mI
    q = (caja(a * m) / n) * guia + caja(b * m) / n
    return cp.where(valido, q + ref, p)


def disponible() -> bool:
    try:
        _k("census")
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def densificar(vistas: list[Vista], calidad: dict, log=print, progreso=None):
    """Igual que densa.densificar, en GPU. Devuelve (xyz, rgb, confianza, info)."""
    lado = calidad["lado_denso"]
    n_vec = calidad["vecinos_densos"]
    speckle = int(densa.FILTRO_SPECKLE[calidad["filtro"]] * (lado / 1400) ** 2)
    libre = limitar_memoria()
    log(f"  Cargando {len(vistas)} imágenes a {lado}px en la GPU ({libre / 1024:.1f} GB de VRAM libres)...")
    cache = densa._Cache(vistas, lado)
    gris = {k: cp.asarray(v) for k, v in cache.gris.items()}
    todas = np.concatenate([v.puntos for v in vistas if len(v.puntos)]) if any(len(v.puntos) for v in vistas) else np.zeros((0, 3))
    if len(todas) > 20000:
        todas = todas[np.random.default_rng(1).choice(len(todas), 20000, replace=False)]
    with ThreadPoolExecutor(max_workers=calidad.get("hilos", 8)) as ex:
        vecinas = list(ex.map(lambda i: densa._vecinas(i, vistas, n_vec), range(len(vistas))))
        vecinas_fusion = list(ex.map(lambda i: densa._vecinas(i, vistas, VECINAS_FUSION), range(len(vistas))))

    caminos = calidad.get("caminos_sgm", 8)
    nubes, colores, ids = [], [], []

    mapas_cam: dict[int, object] = {}     # mapa de profundidad final por cámara (NaN = sin dato)
    k_cam: dict[int, np.ndarray] = {}
    en_gpu = len(vistas) * lado * lado * 4 < 0.3 * _presupuesto["total"]

    def terminar(i, pendientes):
        """Combina los mapas de los pares de la cámara i (media de los que coinciden + filtro guiado)."""
        vi = vistas[i]
        esc = cache.escala[vi.nombre]
        Ki, Di = k_y_distorsion(vi.camara, esc)
        Kud = Ki.copy()
        h, w = cache.gris[vi.nombre].shape
        mapas = [_a_profundidad(g, fut.result(), Kud, w, h) for g, fut in pendientes]
        if not mapas:
            return
        pila = cp.stack(mapas)
        med = cp.nanmedian(pila, axis=0)
        de_acuerdo = cp.abs(pila - med[None]) <= TOLERANCIA_PARES * med[None]
        acuerdo = cp.sum(de_acuerdo, axis=0)
        minimo = 2 if len(mapas) >= 2 else 1
        zmin, zmax, _ = densa._rango_profundidad(vi, todas)
        valido = cp.isfinite(med) & (acuerdo >= minimo) & (med > zmin * 0.7) & (med < zmax * 1.3)
        # media de los pares que coinciden (no la mediana): con n pares el ruido baja ~1/sqrt(n)
        prof = cp.sum(cp.where(de_acuerdo, pila, 0), axis=0) / cp.maximum(acuerdo, 1)
        guia = _remap(cp.asarray(cache.gris[vi.nombre]), Ki, Di, np.eye(3), Kud, w, h).astype(cp.float32) / 255.0
        prof = _filtro_guiado(prof, valido, guia, max(2, round(RADIO_GUIADO * w / 1400)), EPS_GUIADO)
        prof = cp.where(valido, prof, cp.nan).astype(cp.float32)
        mapas_cam[i] = prof if en_gpu else cp.asnumpy(prof)
        k_cam[i] = Kud

    def parametros(i):
        K, v = k_cam[i], vistas[i]
        return np.r_[K[0, 0], K[1, 1], K[0, 2], K[1, 2], np.ravel(v.R), v.t]

    def puntos(i):
        """Fusión multivista de la cámara i con sus vecinas y puntos resultantes (a la RAM)."""
        vi = vistas[i]
        Di_ = cp.asarray(mapas_cam[i])
        h, w = Di_.shape
        acc = cp.zeros((h, w, 3), cp.float32); cnt = cp.zeros((h, w), cp.float32)
        for j in vecinas_fusion[i]:
            if j not in mapas_cam:
                continue
            Dj_ = cp.asarray(mapas_cam[j])
            q = cp.asarray(np.r_[parametros(i), parametros(j)], cp.float32)
            _k("fusion_vista")(((w + 127) // 128, h), (128,), (Di_, cp.int32(w), cp.int32(h), Dj_, cp.int32(Dj_.shape[1]),
                                                             cp.int32(Dj_.shape[0]), q, cp.float32(TOLERANCIA_FUSION), acc, cnt))
        Kud = k_cam[i]
        vv, uu = cp.nonzero(cp.isfinite(Di_))
        z = Di_[vv, uu]
        Xc = cp.stack([(uu - Kud[0, 2]) / Kud[0, 0] * z, (vv - Kud[1, 2]) / Kud[1, 1] * z, z], 1)
        Xw = (Xc - cp.asarray(vi.t, cp.float32)) @ cp.asarray(vi.R, cp.float32)
        n = cnt[vv, uu]
        Xw = (Xw + acc[vv, uu]) / (1 + n)[:, None]       # promedio con las vistas que lo confirman
        Ki, Di = k_y_distorsion(vi.camara, cache.escala[vi.nombre])
        rgb_ud = _remap(cp.asarray(np.ascontiguousarray(cache.rgb[vi.nombre])), Ki, Di, np.eye(3), Kud, w, h)
        nubes.append(cp.asnumpy(Xw.astype(cp.float32))); colores.append(cp.asnumpy(rgb_ud[vv, uu]))
        ids.append(np.full(len(z), i, np.int32))

    # Tubería: mientras la CPU filtra las manchas de la cámara i, la GPU calcula los pares de la i+1.
    anterior = None
    with ThreadPoolExecutor(max_workers=max(2, n_vec)) as cpu:
        for i, vi in enumerate(vistas):
            if progreso:
                progreso(0.85 * i / len(vistas))
            pendientes = []
            esc = cache.escala[vi.nombre]
            for j in vecinas[i]:
                vj = vistas[j]
                r = _disparidad_par(vi, vj, gris[vi.nombre], gris[vj.nombre], esc, cache.escala[vj.nombre], todas, caminos)
                if r is not None:
                    pendientes.append((r[0], cpu.submit(_manchas, r[1], speckle)))
            if anterior is not None:
                terminar(*anterior)
            anterior = (i, pendientes)
        if anterior is not None:
            terminar(*anterior)
    del gris
    _vol.__init__()
    cp.get_default_memory_pool().free_all_blocks()
    for i in sorted(mapas_cam):
        puntos(i)
        if progreso:
            progreso(0.85 + 0.07 * i / len(vistas))
    mapas_cam.clear()
    if not nubes:
        raise RuntimeError("No se pudo generar la nube densa (sin pares válidos).")
    cp.get_default_memory_pool().free_all_blocks()
    X = np.concatenate(nubes); C = np.concatenate(colores); I = np.concatenate(ids)
    del nubes, colores, ids
    log(f"  {len(X):,} puntos crudos de {len(vistas)} mapas de profundidad; fusionando en GPU...")
    vox = densa.tamano_voxel(vistas, todas, cache.escala, calidad)
    minc = calidad.get("confianza_minima", 2)
    xyz, rgb_, conf = fusionar_por_bloques(X, C, I, len(vistas), vox, minc)
    del X, C, I
    cp.get_default_memory_pool().free_all_blocks()
    if progreso:
        progreso(1.0)
    log(f"  Nube densa: {len(xyz):,} puntos (vóxel {vox * 100:.1f} cm, confianza ≥ {minc})")
    return xyz, rgb_, conf, {"puntos": int(len(xyz)), "voxel_m": vox}
