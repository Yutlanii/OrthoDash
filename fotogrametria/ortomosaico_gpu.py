"""Ortomosaico en GPU (CUDA vía CuPy): mismo algoritmo que ortomosaico.py.

Por bloque, en la tarjeta: rejilla gruesa sobre la superficie del DEM -> proyección en cada foto
candidata (con distorsión) -> foto "más nadir" por píxel -> coordenadas finas (bilineal) ->
muestreo de la foto -> fusión multibanda con pirámides laplacianas. La CPU sólo decodifica las
fotos (por adelantado, en hilos) y otro hilo escribe el GeoTIFF mientras la GPU sigue.
"""
from __future__ import annotations

import queue
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import rasterio
from rasterio.enums import ColorInterp, Resampling
from rasterio.windows import Window

from . import color, radiometria
from .alineacion import Vista, k_y_distorsion
from .dem import Raster
from .metadata import leer_imagen, leer_imagen_reducida
from .ortomosaico import CacheImagenes, _affine, _huella_xy, superficie_proyeccion, vistas_reducidas

warnings.filterwarnings("ignore", message="CUDA path could not be detected")
import cupy as cp  # noqa: E402
from cupyx.scipy import ndimage as ndi  # noqa: E402

_CODIGO = r"""
extern "C" {
// Proyección de la rejilla (N puntos del mundo) en una cámara; uv en convención OpenCV y puntuación
// "nadir" (rayo vertical + cercanía al centro), inf si cae fuera de la foto.
// c: R[9], t[3], fx, fy, cx, cy, k1, k2, p1, p2, k3, w, h, Cx, Cy, Cz
__global__ void proyectar(const float* P, int n, const float* c, float* uv, float* score) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float X = P[3 * i], Y = P[3 * i + 1], Z = P[3 * i + 2];
    float xc = c[0] * X + c[1] * Y + c[2] * Z + c[9];
    float yc = c[3] * X + c[4] * Y + c[5] * Z + c[10];
    float zc = c[6] * X + c[7] * Y + c[8] * Z + c[11];
    float s = __int_as_float(0x7f800000);
    float u = -1e6f, v = -1e6f;
    if (zc > 1e-6f) {
        float x = xc / zc, y = yc / zc, r2 = x * x + y * y;
        float rad = 1.f + r2 * (c[16] + r2 * (c[17] + r2 * c[20]));
        float xd = x * rad + 2.f * c[18] * x * y + c[19] * (r2 + 2.f * x * x);
        float yd = y * rad + c[18] * (r2 + 2.f * y * y) + 2.f * c[19] * x * y;
        u = c[12] * xd + c[14]; v = c[13] * yd + c[15];
        float w = c[21], h = c[22];
        if (u >= 0 && v >= 0 && u <= w - 1 && v <= h - 1) {
            float dx = c[23] - X, dy = c[24] - Y, dz = c[25] - Z;
            float cosT = dz * rsqrtf(dx * dx + dy * dy + dz * dz);
            float ru = (u + 0.5f) / w - 0.5f, rv = (v + 0.5f) / h - 0.5f;
            s = (1.f - cosT) + 0.25f * (ru * ru + rv * rv);
        }
    }
    uv[2 * i] = u; uv[2 * i + 1] = v; score[i] = s;
}

// Interpolación bilineal de una rejilla gruesa (hc, wc, 2) a la fina (hh, ww): posición = píxel / paso
__global__ void ampliar_uv(const float* uvc, int hc, int wc, float paso, float* uvf, int hh, int ww) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= ww || y >= hh) return;
    float gx = x / paso, gy = y / paso;
    int x0 = min((int)gx, wc - 2), y0 = min((int)gy, hc - 2);
    float ax = gx - x0, ay = gy - y0;
    for (int k = 0; k < 2; k++) {
        float a = uvc[(y0 * wc + x0) * 2 + k], b = uvc[(y0 * wc + x0 + 1) * 2 + k];
        float c = uvc[((y0 + 1) * wc + x0) * 2 + k], d = uvc[((y0 + 1) * wc + x0 + 1) * 2 + k];
        uvf[(y * ww + x) * 2 + k] = (1 - ay) * ((1 - ax) * a + ax * b) + ay * ((1 - ax) * c + ax * d);
    }
}

// Muestreo bilineal de un recorte de la foto (borde replicado); ok = dentro de la foto completa.
__global__ void muestrear(const float* img, int iw, int ih, int ch, int ox, int oy, int W, int H,
                          const float* uvf, int n, float* out, float* ok) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float u = uvf[2 * i], v = uvf[2 * i + 1];
    ok[i] = (u >= 0 && v >= 0 && u <= W - 1 && v <= H - 1) ? 1.f : 0.f;
    float x = fminf(fmaxf(u - ox, 0.f), iw - 1.001f), y = fminf(fmaxf(v - oy, 0.f), ih - 1.001f);
    int x0 = (int)x, y0 = (int)y;
    int x1 = min(x0 + 1, iw - 1), y1 = min(y0 + 1, ih - 1);
    float ax = x - x0, ay = y - y0;
    for (int c = 0; c < ch; c++) {
        float a = img[(y0 * iw + x0) * ch + c], b = img[(y0 * iw + x1) * ch + c];
        float cc = img[(y1 * iw + x0) * ch + c], d = img[(y1 * iw + x1) * ch + c];
        out[i * ch + c] = (1 - ay) * ((1 - ax) * a + ax * b) + ay * ((1 - ax) * cc + ax * d);
    }
}
}
"""

_mod = None


def _k(nombre):
    global _mod
    if _mod is None:
        _mod = cp.RawModule(code=_CODIGO, options=("-std=c++14", "--use_fast_math"))
    return _mod.get_function(nombre)


_G5 = cp.asarray(np.array([1, 4, 6, 4, 1], np.float32) / 16)


def _desenfocar(a, k):
    a = ndi.convolve1d(a, k, axis=0, mode="mirror")
    return ndi.convolve1d(a, k, axis=1, mode="mirror")


def _bajar(a):          # cv2.pyrDown
    return _desenfocar(a, _G5)[::2, ::2]


def _subir(a, forma):   # cv2.pyrUp(dstsize=forma)
    up = cp.zeros(forma[:2] + a.shape[2:], a.dtype)
    up[::2, ::2] = a[:(forma[0] + 1) // 2, :(forma[1] + 1) // 2]
    return _desenfocar(up, _G5 * 2)


def _piramide_laplaciana(img, niveles):
    g = [img]
    for _ in range(niveles):
        g.append(_bajar(g[-1]))
    lap = [g[k] - _subir(g[k + 1], g[k].shape) for k in range(niveles)]
    lap.append(g[-1])
    return lap


def _piramide_gauss(m, niveles):
    g = [m]
    for _ in range(niveles):
        g.append(_bajar(g[-1]))
    return g


def _colapsar(lap):
    img = lap[-1]
    for k in range(len(lap) - 2, -1, -1):
        img = _subir(img, lap[k].shape) + lap[k]
    return img


def _muestrear_dem(dem_gpu, mx, my):
    """Bilineal (borde replicado) de la superficie del DEM en coordenadas de celda."""
    return ndi.map_coordinates(dem_gpu, cp.stack([my.ravel(), mx.ravel()]), order=1, mode="nearest").reshape(my.shape)


def _parametros_camara(v: Vista) -> cp.ndarray:
    K, D = k_y_distorsion(v.camara, 1.0)   # convención OpenCV: listo para muestrear
    return cp.asarray(np.r_[np.ravel(v.R), v.t, K[0, 0], K[1, 1], K[0, 2], K[1, 2], D[:5],
                            v.camara.width, v.camara.height, v.centro], cp.float32)


def construir_ortomosaico(vistas: list[Vista], dem: Raster, res: float, ruta_tif, origen, crs,
                          multiespectral: bool = False, nombres_bandas=None, log=print, progreso=None,
                          hilos: int = 6, max_ram_gb: float = 8.0, reduccion: int = 1,
                          ganancias: dict | None = None) -> dict:
    """reduccion > 1: fotos a 1/reduccion (modo express). En multiespectral se reduce después de
    la corrección radiométrica, cuyo viñeteo es por píxel de la imagen completa."""
    if reduccion > 1:
        vistas = vistas_reducidas(vistas, reduccion)
    from .densa_gpu import limitar_memoria
    limitar_memoria()
    H_dem, W_dem = dem.forma
    W = int(np.ceil(W_dem * dem.res / res))
    H = int(np.ceil(H_dem * dem.res / res))
    log(f"  Ortomosaico {W}x{H} px a {res * 100:.2f} cm/pixel ({len(vistas)} imágenes, GPU)")

    dem_valido = np.isfinite(dem.datos)
    sup_gpu = cp.asarray(superficie_proyeccion(dem))
    valido_gpu = cp.asarray(dem_valido)
    zmin, zmax = float(np.nanmin(dem.datos)), float(np.nanmax(dem.datos))
    huellas = [_huella_xy(v, zmin, zmax) for v in vistas]
    params = [_parametros_camara(v) for v in vistas]

    def cargar(v: Vista):
        img = leer_imagen(v.foto.ruta) if reduccion == 1 or multiespectral else leer_imagen_reducida(v.foto.ruta, reduccion)
        if multiespectral:
            img = radiometria.corregir(img, v.foto)
            if img.ndim == 3 and img.shape[2] > 1:   # una banda por archivo; si no, se promedia
                img = img.mean(2, keepdims=True)
            if reduccion > 1:
                img = cv2.resize(img, (v.camara.width, v.camara.height), interpolation=cv2.INTER_AREA)
                img = img[..., None] if img.ndim == 2 else img
            return img
        if img.dtype != np.uint8:
            f = img.astype(np.float32)
            lo, hi = np.percentile(f, (0.1, 99.9))
            img = np.clip((f - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        if img.shape[2] == 1:
            img = np.repeat(img, 3, 2)
        img = np.ascontiguousarray(img[..., :3])
        return color.aplicar(img, ganancias.get(v.nombre)) if ganancias else img   # exposición igualada

    cache = CacheImagenes(int(max_ram_gb * 1024 ** 3), cargar)
    nb = 1 if multiespectral else 3
    T, M, niveles = 1024, 96, 5
    bloques = [(r, c) for r in range(0, H, T) for c in range(0, W, T)]
    paso = max(1, min(8, int(round(dem.res / res))))

    def geometria(rc):
        r0, c0 = rc
        r1, c1 = min(r0 + T, H), min(c0 + T, W)
        R0, C0, R1, C1 = r0 - M, c0 - M, r1 + M, c1 + M
        bx0, bx1 = dem.x0 + (C0 + 0.5) * res, dem.x0 + (C1 - 0.5) * res
        by1, by0 = dem.y0 - (R0 + 0.5) * res, dem.y0 - (R1 - 0.5) * res
        cand = [k for k, (a, b, c, d) in enumerate(huellas) if a <= bx1 and b >= bx0 and c <= by1 and d >= by0]
        return (r0, c0, r1, c1, R0, C0, R1, C1), cand

    # Decodificación por adelantado de las fotos de los bloques siguientes (la CPU no espera a la GPU)
    previas = ThreadPoolExecutor(max_workers=max(2, min(hilos, 8)))
    pedidas: set[int] = set()
    geo_bloques = [geometria(rc) for rc in bloques]

    def adelantar(i):
        for j in range(i, min(i + 3, len(bloques))):
            for k in geo_bloques[j][1]:
                if k not in pedidas:
                    pedidas.add(k)
                    previas.submit(cache.obtener, vistas[k])

    def bloque(i):
        (r0, c0, r1, c1, R0, C0, R1, C1), cand = geo_bloques[i]
        if not cand:
            return None
        hh, ww = R1 - R0, C1 - C0
        hc, wc = (hh - 1) // paso + 2, (ww - 1) // paso + 2
        cols = C0 + cp.arange(wc, dtype=cp.float32) * paso
        rows = R0 + cp.arange(hc, dtype=cp.float32) * paso
        xs = dem.x0 + (cols + 0.5) * res
        ys = dem.y0 - (rows + 0.5) * res
        MX, MY = cp.meshgrid((xs - dem.x0) / dem.res - 0.5, (dem.y0 - ys) / dem.res - 0.5)
        Z = _muestrear_dem(sup_gpu, MX, MY)
        XX, YY = cp.meshgrid(xs, ys)
        P = cp.ascontiguousarray(cp.stack([XX.ravel(), YY.ravel(), Z.ravel()], 1).astype(cp.float32))
        # cobertura del DEM a resolución fina (vecino más cercano)
        cf = cp.clip(cp.rint((cp.arange(C0, C1) + 0.5) * res / dem.res - 0.5).astype(cp.int64), -1, W_dem)
        rf = cp.clip(cp.rint((cp.arange(R0, R1) + 0.5) * res / dem.res - 0.5).astype(cp.int64), -1, H_dem)
        dentro = ((rf >= 0) & (rf < H_dem))[:, None] & ((cf >= 0) & (cf < W_dem))[None, :]
        cub = dentro & valido_gpu[cp.clip(rf, 0, H_dem - 1)][:, cp.clip(cf, 0, W_dem - 1)]
        if not bool(cub[M:M + (r1 - r0), M:M + (c1 - c0)].any()):
            return None
        n = P.shape[0]
        uvs = cp.empty((len(cand), n, 2), cp.float32)
        scores = cp.empty((len(cand), n), cp.float32)
        for j, k in enumerate(cand):
            _k("proyectar")(((n + 255) // 256,), (256,), (P, cp.int32(n), params[k], uvs[j], scores[j]))
        mejor = cp.argmin(scores, axis=0)
        etiqueta_c = cp.where(cp.isfinite(scores.min(0)), mejor, -1).reshape(hc, wc)
        iy = cp.clip(cp.rint(cp.arange(hh) / paso).astype(cp.int64), 0, hc - 1)
        ix = cp.clip(cp.rint(cp.arange(ww) / paso).astype(cp.int64), 0, wc - 1)
        etiqueta = etiqueta_c[iy][:, ix]
        etiqueta[~cub] = -1
        usadas = [int(j) for j in cp.unique(etiqueta).get() if j >= 0]
        if not usadas:
            return None
        acum = peso = None
        bl = ((ww + 15) // 16, (hh + 15) // 16)
        for j in usadas:
            v = vistas[cand[j]]
            uvf = cp.empty((hh, ww, 2), cp.float32)
            _k("ampliar_uv")(bl, (16, 16), (uvs[j].reshape(hc, wc, 2), cp.int32(hc), cp.int32(wc), cp.float32(paso),
                                            uvf, cp.int32(hh), cp.int32(ww)))
            # recorte de la foto que cubre el bloque (sube a la GPU sólo lo necesario)
            img = cache.obtener(v)
            ih, iw = img.shape[:2]
            u, vv = uvf[..., 0], uvf[..., 1]
            dentro_img = (u >= 0) & (vv >= 0) & (u <= iw - 1) & (vv <= ih - 1)
            if not bool(dentro_img.any()):
                continue
            lim = cp.stack([u[dentro_img].min(), u[dentro_img].max(), vv[dentro_img].min(), vv[dentro_img].max()]).get()
            x0, x1 = max(0, int(lim[0]) - 2), min(iw, int(lim[1]) + 3)
            y0, y1 = max(0, int(lim[2]) - 2), min(ih, int(lim[3]) + 3)
            rec = cp.asarray(np.ascontiguousarray(img[y0:y1, x0:x1])).astype(cp.float32)
            if rec.ndim == 2:
                rec = rec[..., None]
            ch = rec.shape[2]
            muestra = cp.empty((hh, ww, ch), cp.float32)
            ok = cp.empty((hh, ww), cp.float32)
            npx = hh * ww
            _k("muestrear")(((npx + 255) // 256,), (256,), (rec, cp.int32(x1 - x0), cp.int32(y1 - y0), cp.int32(ch),
                                                         cp.int32(x0), cp.int32(y0), cp.int32(iw), cp.int32(ih),
                                                         uvf, cp.int32(npx), muestra, ok))
            m = ((etiqueta == j) & (ok > 0)).astype(cp.float32)
            lap = _piramide_laplaciana(muestra, niveles)
            gm = _piramide_gauss(m, niveles)
            gv = _piramide_gauss(ok, niveles)
            gm = [g * (vv_ > 0.995) for g, vv_ in zip(gm, gv)]   # nunca mezclar fuera de la imagen
            if acum is None:
                acum = [l * g[..., None] for l, g in zip(lap, gm)]
                peso = gm
            else:
                for q in range(niveles + 1):
                    acum[q] += lap[q] * gm[q][..., None]
                    peso[q] += gm[q]
        if acum is None:
            return None
        mosaico = _colapsar([a / cp.maximum(p, 1e-6)[..., None] for a, p in zip(acum, peso)])
        mos = mosaico[M:M + (r1 - r0), M:M + (c1 - c0)]
        alfa = etiqueta[M:M + (r1 - r0), M:M + (c1 - c0)] >= 0
        if multiespectral:
            out = cp.where(alfa[..., None], mos, cp.nan).astype(cp.float32)
        else:
            out = cp.clip(mos + 0.5, 0, 255).astype(cp.uint8)
            out[~alfa] = 0
        return (r0, c0), cp.asnumpy(cp.moveaxis(out, 2, 0)), cp.asnumpy(alfa)

    perfil = dict(driver="GTiff", width=W, height=H, crs=crs, transform=_affine(dem, res, origen),
                  tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER", compress="deflate",
                  num_threads="all_cpus")
    if multiespectral:
        perfil.update(count=nb, dtype="float32", nodata=np.nan, predictor=3)
    else:
        perfil.update(count=4, dtype="uint8", photometric="RGB", predictor=2)

    # Escritura en otro hilo: la GPU no espera a la compresión del GeoTIFF
    cola: queue.Queue = queue.Queue(maxsize=4)
    error: list[BaseException] = []

    def escritor(dst):
        try:
            while (item := cola.get()) is not None:
                (r0, c0), datos, alfa = item
                win = Window(c0, r0, alfa.shape[1], alfa.shape[0])
                if multiespectral:
                    dst.write(datos, window=win)
                else:
                    dst.write(datos, indexes=[1, 2, 3], window=win)
                    dst.write((alfa * 255).astype(np.uint8), 4, window=win)
        except BaseException as e:   # se informa en el hilo principal
            error.append(e)

    with rasterio.Env(GDAL_NUM_THREADS="ALL_CPUS"), rasterio.open(ruta_tif, "w", **perfil) as dst:
        if not multiespectral:
            dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha]
        hilo = threading.Thread(target=escritor, args=(dst,), daemon=True)
        hilo.start()
        try:
            for i in range(len(bloques)):
                adelantar(i)
                r = bloque(i)
                if r is not None:
                    cola.put(r)
                if error:
                    raise error[0]
                if progreso:
                    progreso(0.85 * (i + 1) / len(bloques))
        finally:
            cola.put(None)
            hilo.join()
            previas.shutdown(wait=False, cancel_futures=True)
        if error:
            raise error[0]
        if nombres_bandas:
            for i, nombre in enumerate(nombres_bandas, 1):
                dst.set_band_description(i, nombre)
        log("  Generando vistas previas internas (overviews)...")
        dst.build_overviews([2, 4, 8, 16, 32, 64], Resampling.average)
        dst.update_tags(ns="rio_overview", resampling="average")
    del sup_gpu, valido_gpu
    cp.get_default_memory_pool().free_all_blocks()
    if progreso:
        progreso(1.0)
    return {"ancho_px": W, "alto_px": H, "resolucion_m": res}
