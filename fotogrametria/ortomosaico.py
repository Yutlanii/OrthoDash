"""Paso 6: ortomosaico.

Equivalente a Metashape buildOrthomosaic(surface=DEM, blending=Mosaic, hole_filling=True):
  * Cada pixel del terreno (X, Y, Z del DEM) se proyecta en las fotos que lo ven.
  * Se elige la foto más "nadir" (rayo más vertical y más cercano al centro de la imagen).
  * Fusión multibanda (pirámide laplaciana): las altas frecuencias se mezclan sólo en la línea de
    corte y las bajas en una franja más ancha => costuras invisibles sin "fantasmas".
Procesa por bloques en paralelo, así funciona con vuelos grandes sin agotar memoria.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

from .alineacion import Vista
from .dem import Raster, _rellenar_piramide
from .metadata import leer_imagen, leer_imagen_reducida
from . import color, radiometria


class CacheImagenes:
    def __init__(self, max_bytes: int, cargar):
        self.max = max_bytes
        self.cargar = cargar
        self.datos: OrderedDict[str, np.ndarray] = OrderedDict()
        self.bytes = 0
        self.lock = threading.Lock()
        self.cargando: dict[str, threading.Event] = {}

    def obtener(self, v: Vista) -> np.ndarray:
        while True:
            with self.lock:
                if v.nombre in self.datos:
                    self.datos.move_to_end(v.nombre)
                    return self.datos[v.nombre]
                ev = self.cargando.get(v.nombre)
                if ev is None:
                    ev = threading.Event()
                    self.cargando[v.nombre] = ev
                    soy_cargador = True
                else:
                    soy_cargador = False
            if not soy_cargador:
                ev.wait()
                continue
            img = self.cargar(v)
            with self.lock:
                self.datos[v.nombre] = img
                self.bytes += img.nbytes
                while self.bytes > self.max and len(self.datos) > 1:
                    _, viejo = self.datos.popitem(last=False)
                    self.bytes -= viejo.nbytes
                del self.cargando[v.nombre]
            ev.set()
            return img


def _huella_xy(v: Vista, zmin: float, zmax: float) -> tuple[float, float, float, float]:
    w, h = v.camara.width, v.camara.height
    px = np.array([[0, 0], [w, 0], [w, h], [0, h], [w / 2, 0], [w, h / 2], [w / 2, h], [0, h / 2]], float)
    rayos = np.c_[np.asarray(v.camara.cam_from_img(px)), np.ones(len(px))]
    d = rayos @ v.R
    c = v.centro
    pts = []
    for z in (zmin, zmax):
        ok = d[:, 2] < -1e-6
        s = (z - c[2]) / d[ok, 2]
        pts.append(c[:2] + d[ok, :2] * s[:, None])
    P = np.concatenate(pts)
    return P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max()


def _piramide_laplaciana(img: np.ndarray, niveles: int):
    g = [img]
    for _ in range(niveles):
        b = cv2.pyrDown(g[-1])
        g.append(b[..., None] if b.ndim == 2 else b)
    lap = []
    for k in range(niveles):
        arriba = cv2.pyrUp(g[k + 1], dstsize=(g[k].shape[1], g[k].shape[0]))
        lap.append(g[k] - arriba.reshape(g[k].shape))
    lap.append(g[-1])
    return lap


def _piramide_gauss(m: np.ndarray, niveles: int):
    g = [m]
    for _ in range(niveles):
        g.append(cv2.pyrDown(g[-1]))
    return g


def _colapsar(lap):
    img = lap[-1]
    for k in range(len(lap) - 2, -1, -1):
        h, w = lap[k].shape[:2]
        arriba = cv2.pyrUp(img, dstsize=(w, h))
        img = arriba.reshape(lap[k].shape) + lap[k]
    return img


def _rellenar_invalidos(img: np.ndarray, valido: np.ndarray) -> np.ndarray:
    if valido.all():
        return img
    out = np.empty_like(img)
    for b in range(img.shape[2]):
        out[..., b] = _rellenar_piramide(img[..., b], valido)
    return out


SUAVIZADO_M = 1.0   # radio (m) de suavizado de la superficie de proyección (medido: 0,1 m da remolinos en
                    # la vegetación; 1 m queda limpio, como Metashape, sin coste de tiempo)


def superficie_proyeccion(dem: Raster, suavizado_m: float | None = None) -> np.ndarray:
    """Superficie sobre la que se proyectan las fotos: el DEM rellenado y suavizado.

    En copas de árboles el DSM es muy irregular; proyectar sobre él deforma la textura (remolinos)
    y hace saltar la foto elegida. Una superficie más suave da un ortomosaico limpio, como el de
    Metashape, a costa de un leve desplazamiento de los objetos altos (no del suelo)."""
    valido = np.isfinite(dem.datos)
    lleno = _rellenar_piramide(np.where(valido, dem.datos, 0), valido).astype(np.float32)
    sigma = max(0.5, (SUAVIZADO_M if suavizado_m is None else suavizado_m) / dem.res)
    return cv2.GaussianBlur(cv2.medianBlur(lleno, 5), (0, 0), sigma)


def vistas_reducidas(vistas: list[Vista], factor: int) -> list[Vista]:
    """Las mismas vistas con la cámara reescalada a las fotos leídas a 1/factor
    (leer_imagen_reducida da ceil(lado / factor))."""
    import dataclasses
    import pycolmap
    out = []
    for v in vistas:
        c = pycolmap.Camera.create_from_model_name(0, v.camara.model_name, 1.0, v.camara.width, v.camara.height)
        c.params = v.camara.params
        c.rescale(-(-v.camara.width // factor), -(-v.camara.height // factor))
        out.append(dataclasses.replace(v, camara=c))
    return out


def construir_ortomosaico(vistas: list[Vista], dem: Raster, res: float, ruta_tif, origen, crs,
                          multiespectral: bool = False, nombres_bandas=None, log=print, progreso=None,
                          hilos: int = 6, max_ram_gb: float = 8.0, reduccion: int = 1,
                          ganancias: dict | None = None) -> dict:
    """reduccion > 1: fotos a 1/reduccion (modo express). En multiespectral se reduce después de
    la corrección radiométrica, cuyo viñeteo es por píxel de la imagen completa."""
    if reduccion > 1:
        vistas = vistas_reducidas(vistas, reduccion)
    H_dem, W_dem = dem.forma
    ancho_m, alto_m = W_dem * dem.res, H_dem * dem.res
    W = int(np.ceil(ancho_m / res))
    H = int(np.ceil(alto_m / res))
    log(f"  Ortomosaico {W}x{H} px a {res * 100:.2f} cm/pixel ({len(vistas)} imágenes)")

    dem_valido = np.isfinite(dem.datos)
    dem_superficie = superficie_proyeccion(dem)
    zmin, zmax = float(np.nanmin(dem.datos)), float(np.nanmax(dem.datos))
    huellas = [_huella_xy(v, zmin, zmax) for v in vistas]

    # Imágenes completas (RGB uint8 o reflectancia float32)
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

    T, M = 1024, 96
    niveles = 5
    bloques = [(r, c) for r in range(0, H, T) for c in range(0, W, T)]

    paso = max(1, min(8, int(round(dem.res / res))))  # la proyección es suave a escala del DEM

    def bloque(rc):
        r0, c0 = rc
        r1, c1 = min(r0 + T, H), min(c0 + T, W)
        R0, C0, R1, C1 = r0 - M, c0 - M, r1 + M, c1 + M
        hh, ww = R1 - R0, C1 - C0
        bx0, bx1 = dem.x0 + (C0 + 0.5) * res, dem.x0 + (C1 - 0.5) * res
        by1, by0 = dem.y0 - (R0 + 0.5) * res, dem.y0 - (R1 - 0.5) * res
        cand = [k for k, (a, b, c, d) in enumerate(huellas) if a <= bx1 and b >= bx0 and c <= by1 and d >= by0]
        if not cand:
            return rc, None
        # Rejilla gruesa (cada `paso` pixeles) donde se proyecta; luego se interpola exacto (bilineal)
        hc, wc = (hh - 1) // paso + 2, (ww - 1) // paso + 2
        cols = C0 + np.arange(wc) * paso
        rows = R0 + np.arange(hc) * paso
        xs = dem.x0 + (cols + 0.5) * res
        ys = dem.y0 - (rows + 0.5) * res
        MX, MY = np.meshgrid(((xs - dem.x0) / dem.res - 0.5).astype(np.float32), ((dem.y0 - ys) / dem.res - 0.5).astype(np.float32))
        Z = cv2.remap(dem_superficie, MX, MY, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        XX, YY = np.meshgrid(xs, ys)
        P = np.stack([XX.ravel(), YY.ravel(), Z.ravel().astype(np.float64)], 1)
        # mapas de la rejilla fina -> gruesa
        fy, fx = np.mgrid[0:hh, 0:ww].astype(np.float32)
        fx /= paso; fy /= paso
        # cobertura del DEM a resolución fina
        mxf = ((dem.x0 + (np.arange(C0, C1) + 0.5) * res - dem.x0) / dem.res - 0.5).astype(np.float32)
        myf = ((dem.y0 - (dem.y0 - (np.arange(R0, R1) + 0.5) * res)) / dem.res - 0.5).astype(np.float32)
        MXf, MYf = np.meshgrid(mxf, myf)
        cub = cv2.remap(dem_valido.astype(np.uint8), MXf, MYf, cv2.INTER_NEAREST, borderValue=0).astype(bool)
        if not cub[M:M + (r1 - r0), M:M + (c1 - c0)].any():
            return rc, None
        mejor = np.full(hc * wc, np.inf, np.float32)
        etiqueta_c = np.full(hc * wc, -1, np.int32)
        uvs = {}
        for k in cand:
            v = vistas[k]
            uv, z = v.proyectar(P)
            w_img, h_img = v.camara.width, v.camara.height
            ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] <= w_img - 1) & (uv[:, 1] <= h_img - 1)
            if not ok.any():
                continue
            # puntuación: rayo vertical + cercanía al centro de la imagen (cámara "más nadir")
            d = v.centro[None, :] - P
            cos_t = d[:, 2] / np.linalg.norm(d, axis=1)
            rr = ((uv[:, 0] / w_img - 0.5) ** 2 + (uv[:, 1] / h_img - 0.5) ** 2)
            score = np.where(ok, (1 - cos_t) + 0.25 * rr, np.inf).astype(np.float32)
            mk = score < mejor
            mejor = np.where(mk, score, mejor)
            etiqueta_c = np.where(mk, k, etiqueta_c)
            uvs[k] = (uv - 0.5).reshape(hc, wc, 2).astype(np.float32)   # COLMAP -> OpenCV (centro en 0)
        etiqueta = cv2.remap(etiqueta_c.reshape(hc, wc).astype(np.float32), fx, fy, cv2.INTER_NEAREST).astype(np.int32)
        etiqueta[~cub] = -1
        usadas = [k for k in np.unique(etiqueta) if k >= 0]
        if not usadas:
            return rc, None
        acum = peso = None
        for k in usadas:
            uvf = cv2.remap(uvs[k], fx, fy, cv2.INTER_LINEAR)
            v = vistas[k]
            ok = (uvf[..., 0] >= 0) & (uvf[..., 1] >= 0) & (uvf[..., 0] <= v.camara.width - 1) & (uvf[..., 1] <= v.camara.height - 1)
            img = cache.obtener(v)
            muestra = cv2.remap(img, uvf[..., 0], uvf[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            if muestra.ndim == 2:
                muestra = muestra[..., None]
            muestra = muestra.astype(np.float32)
            m = ((etiqueta == k) & ok).astype(np.float32)
            lap = _piramide_laplaciana(muestra, niveles)
            gm = _piramide_gauss(m, niveles)
            gv = _piramide_gauss(ok.astype(np.float32), niveles)
            for n in range(niveles + 1):
                gm[n] = gm[n] * (gv[n] > 0.995)  # nunca mezclar fuera de la imagen
            if acum is None:
                acum = [l * g[..., None] for l, g in zip(lap, gm)]
                peso = [g.copy() for g in gm]
            else:
                for n in range(niveles + 1):
                    acum[n] += lap[n] * gm[n][..., None]
                    peso[n] += gm[n]
        res_lap = [a / np.maximum(p, 1e-6)[..., None] for a, p in zip(acum, peso)]
        mosaico = _colapsar(res_lap)
        interior = (slice(M, M + (r1 - r0)), slice(M, M + (c1 - c0)))
        return rc, (mosaico[interior], etiqueta[interior] >= 0)

    perfil = dict(driver="GTiff", width=W, height=H, crs=crs, transform=_affine(dem, res, origen),
                  tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER", compress="deflate",
                  num_threads="all_cpus")
    if multiespectral:
        perfil.update(count=nb, dtype="float32", nodata=np.nan, predictor=3)
    else:
        perfil.update(count=4, dtype="uint8", photometric="RGB", predictor=2)
    hechos = 0
    with rasterio.Env(GDAL_NUM_THREADS="ALL_CPUS"), rasterio.open(ruta_tif, "w", **perfil) as dst:
        if not multiespectral:
            from rasterio.enums import ColorInterp
            dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha]
        with ThreadPoolExecutor(max_workers=hilos) as ex:
            for rc, res_b in ex.map(bloque, bloques):
                hechos += 1
                if progreso:
                    progreso(0.92 * hechos / len(bloques))
                if res_b is None:
                    continue
                mos, alfa = res_b
                r0, c0 = rc
                h, w = alfa.shape
                win = Window(c0, r0, w, h)
                if multiespectral:
                    out = np.where(alfa[..., None], mos, np.nan).astype(np.float32)
                    dst.write(np.moveaxis(out, 2, 0), window=win)
                else:
                    out = np.clip(mos + 0.5, 0, 255).astype(np.uint8)
                    out[~alfa] = 0
                    dst.write(np.moveaxis(out, 2, 0), indexes=[1, 2, 3], window=win)
                    dst.write((alfa * 255).astype(np.uint8), 4, window=win)
        if nombres_bandas:
            for i, n in enumerate(nombres_bandas, 1):
                dst.set_band_description(i, n)
        log("  Generando vistas previas internas (overviews)...")
        dst.build_overviews([2, 4, 8, 16, 32, 64], Resampling.average)
        dst.update_tags(ns="rio_overview", resampling="average")
    if progreso:
        progreso(1.0)
    return {"ancho_px": W, "alto_px": H, "resolucion_m": res}


def _affine(dem: Raster, res: float, origen):
    from affine import Affine
    return Affine(res, 0, dem.x0 + origen[0], 0, -res, dem.y0 + origen[1])
