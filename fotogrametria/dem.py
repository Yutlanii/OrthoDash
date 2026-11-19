"""Paso 5: modelo digital de elevación (DSM) a partir de la nube densa.

Como en Metashape (buildDem, source=PointCloud, interpolation=Enabled): rasteriza la nube y
rellena por interpolación todas las zonas vistas por al menos una imagen.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .alineacion import Vista


@dataclass
class Raster:
    datos: np.ndarray          # (H, W) float32, NaN = sin dato
    x0: float                  # esquina superior izquierda (local)
    y0: float
    res: float                 # tamaño de pixel (m)

    @property
    def forma(self):
        return self.datos.shape

    def transform_affine(self, origen):
        from affine import Affine
        return Affine(self.res, 0, self.x0 + origen[0], 0, -self.res, self.y0 + origen[1])


def mascara_cobertura(vistas: list[Vista], x0, y0, res, forma, z_suelo: float) -> np.ndarray:
    """Zona del terreno vista por al menos una imagen (huellas proyectadas a la altura del suelo)."""
    m = np.zeros(forma, np.uint8)
    for v in vistas:
        w, h = v.camara.width, v.camara.height
        bordes = []
        for t in np.linspace(0, 1, 9)[:-1]:
            bordes += [(t * w, 0), (w, t * h), (w - t * w, h), (0, h - t * h)]
        px = np.array(bordes, float)
        rayos = np.c_[np.asarray(v.camara.cam_from_img(px)), np.ones(len(px))]
        d = rayos @ v.R  # dirección en mundo
        c = v.centro
        ok = d[:, 2] < -1e-6
        if ok.sum() < 3:
            continue
        s = (z_suelo - c[2]) / d[ok, 2]
        P = c + d[ok] * s[:, None]
        col = (P[:, 0] - x0) / res
        fil = (y0 - P[:, 1]) / res
        cv2.fillPoly(m, [np.round(np.c_[col, fil]).astype(np.int32)], 1)
    return m.astype(bool)


def _rellenar_piramide(z: np.ndarray, valido: np.ndarray) -> np.ndarray:
    """Interpolación "push-pull": rellena huecos de forma suave y rápida."""
    if valido.all():
        return z
    niveles = []
    zc, wc = np.where(valido, z, 0).astype(np.float32), valido.astype(np.float32)
    while min(zc.shape) > 2:
        niveles.append((zc, wc))
        h, w = zc.shape
        hp, wp = (h + 1) // 2 * 2, (w + 1) // 2 * 2
        zp = np.zeros((hp, wp), np.float32); wp_ = np.zeros((hp, wp), np.float32)
        zp[:h, :w] = zc * wc; wp_[:h, :w] = wc
        zs = zp.reshape(hp // 2, 2, wp // 2, 2).sum((1, 3))
        ws = wp_.reshape(hp // 2, 2, wp // 2, 2).sum((1, 3))
        zc = np.where(ws > 0, zs / np.maximum(ws, 1e-9), 0)
        wc = np.minimum(ws, 1.0)
    est = zc
    for zc_n, wc_n in reversed(niveles):
        h, w = zc_n.shape
        arriba = cv2.resize(est, (w, h), interpolation=cv2.INTER_LINEAR)
        est = wc_n * zc_n + (1 - wc_n) * arriba
    return est


def _region_datos(valido: np.ndarray, res: float, radio_m: float = 10.0) -> np.ndarray:
    H, W = valido.shape
    f = max(1, int(round(0.25 / res)))  # trabajar a ~25 cm
    peq = cv2.resize(valido.astype(np.uint8), (max(1, W // f), max(1, H // f)), interpolation=cv2.INTER_AREA)
    peq = (peq > 0).astype(np.uint8)
    k = max(3, int(radio_m / (res * f)) | 1)
    nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    cerr = cv2.morphologyEx(cv2.copyMakeBorder(peq, k, k, k, k, cv2.BORDER_CONSTANT, value=0), cv2.MORPH_CLOSE, nucleo)
    cerr = cerr[k:-k, k:-k]
    # rellena huecos interiores (todo lo no conectado con el borde exterior)
    inv = cv2.copyMakeBorder((1 - cerr).astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)
    marco = np.zeros((inv.shape[0] + 2, inv.shape[1] + 2), np.uint8)
    cv2.floodFill(inv, marco, (0, 0), 2)
    lleno = cerr.copy()
    lleno[inv[1:-1, 1:-1] == 1] = 1
    # descarta islas y franjas sueltas (bordes vistos por una sola foto): sólo zonas conectadas con
    # al menos el 5 % del área de la principal
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(lleno, connectivity=8)
    if n > 2:
        areas = stats[1:, cv2.CC_STAT_AREA]
        lleno = np.isin(etiquetas, 1 + np.nonzero(areas >= 0.05 * areas.max())[0]).astype(np.uint8)
    return cv2.resize(lleno, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)


def construir_dem(xyz: np.ndarray, vistas: list[Vista], res: float, log=print, radio_m: float = 10.0) -> tuple[Raster, np.ndarray]:
    lo = np.percentile(xyz[:, :2], 0.05, axis=0)
    hi = np.percentile(xyz[:, :2], 99.95, axis=0)
    x0 = np.floor(lo[0] / res) * res
    y0 = np.ceil(hi[1] / res) * res
    W = int(np.ceil((hi[0] - x0) / res)) + 1
    H = int(np.ceil((y0 - lo[1]) / res)) + 1
    log(f"  Rejilla DEM {W}x{H} a {res * 100:.1f} cm/pixel")
    col = np.floor((xyz[:, 0] - x0) / res).astype(np.int64)
    fil = np.floor((y0 - xyz[:, 1]) / res).astype(np.int64)
    ok = (col >= 0) & (col < W) & (fil >= 0) & (fil < H)
    col, fil, z = col[ok], fil[ok], xyz[ok, 2]
    lin = fil * W + col
    orden = np.lexsort((z, lin))
    lin_o = lin[orden]
    ini = np.r_[0, np.nonzero(np.diff(lin_o))[0] + 1]
    cnt = np.diff(np.r_[ini, len(lin_o)])
    med = z[orden[ini + cnt // 2]]          # mediana por celda: robusta a ruido
    dem = np.full(H * W, np.nan, np.float32)
    dem[lin_o[ini]] = med
    dem = dem.reshape(H, W)
    valido = np.isfinite(dem)

    # Filtro de picos: celdas que difieren mucho de su vecindario
    relleno = _rellenar_piramide(dem, valido)
    suave = cv2.medianBlur(relleno.astype(np.float32), 5)
    pico = valido & (np.abs(dem - suave) > max(res * 20, 0.5))
    valido &= ~pico

    z_suelo = float(np.percentile(xyz[:, 2], 5))
    cobertura = mascara_cobertura(vistas, x0, y0, res, (H, W), z_suelo)
    # Zona con datos: cierre morfológico (~10 m) + relleno de huecos interiores. Así se interpola todo
    # lo visto por las cámaras dentro del bloque (como "interpolation=Enabled") sin extrapolar fuera.
    cobertura &= _region_datos(valido, res, radio_m)
    final = _rellenar_piramide(np.where(valido, dem, 0), valido)
    final = cv2.GaussianBlur(final, (3, 3), 0.7)
    final = np.where(valido, np.where(np.abs(final - dem) < res * 4, dem, final), final)
    final = np.where(cobertura, final, np.nan).astype(np.float32)
    porc = 100 * valido.sum() / max(cobertura.sum(), 1)
    log(f"  Celdas con dato directo: {porc:.0f}% del área cubierta (resto interpolado)")
    return Raster(final, x0, y0, res), cobertura
