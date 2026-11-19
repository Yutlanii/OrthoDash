"""Paso 4: nube de puntos densa.

Misma arquitectura que describe Metashape (buildDepthMaps + buildPointCloud):
  1. Para cada cámara se eligen las mejores vecinas (traslape + ángulo de intersección).
  2. Mapa de profundidad por par: rectificación estéreo + Semi-Global Matching con verificación
     izquierda-derecha y filtro de componentes conexas (equivalente a los modos de filtrado
     Mild / Moderate / Aggressive).
  3. Los mapas por par se combinan en un mapa de profundidad por cámara (mediana + consistencia).
  4. Los mapas combinados se convierten en nubes parciales y se fusionan en un vóxel global;
     el número de cámaras que contribuyen se guarda como "confianza".
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np

from .alineacion import Vista, k_y_distorsion
from .metadata import leer_imagen

FILTRO_SPECKLE = {"suave": 50, "moderado": 120, "agresivo": 250}


class _Cache:
    def __init__(self, vistas: list[Vista], lado: int):
        self.escala = {}
        self.gris = {}
        self.rgb = {}
        self.lado = lado

        def cargar(v: Vista):
            esc = lado / max(v.foto.ancho, v.foto.alto)
            img = leer_imagen(v.foto.ruta, escala=esc)
            if img.dtype != np.uint8:
                f = img.astype(np.float32)
                lo, hi = np.percentile(f, (0.5, 99.5))
                img = np.clip((f - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
            if img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)
            img = img[..., :3]
            gris = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            return v.nombre, esc, gris, img

        with ThreadPoolExecutor(max_workers=8) as ex:
            for nombre, esc, g, c in ex.map(cargar, vistas):
                self.escala[nombre], self.gris[nombre], self.rgb[nombre] = esc, g, c


def _rango_profundidad(v: Vista, todas: np.ndarray) -> tuple[float, float, float]:
    P = v.puntos if len(v.puntos) >= 20 else todas
    z = v.a_camara(P)[:, 2]
    z = z[z > 0]
    if len(z) == 0:
        return 1.0, 1000.0, 100.0
    return float(np.percentile(z, 1)), float(np.percentile(z, 99)), float(np.median(z))


def _vecinas(i: int, vistas: list[Vista], n: int) -> list[int]:
    vi = vistas[i]
    P = vi.puntos
    if len(P) > 400:
        P = P[np.random.default_rng(0).choice(len(P), 400, replace=False)]
    zmed = np.median(vi.a_camara(P)[:, 2]) if len(P) else 50.0
    cand = []
    for j, vj in enumerate(vistas):
        if j == i or vj.sensor != vi.sensor:
            continue
        b = np.linalg.norm(vi.centro - vj.centro)
        r = b / max(zmed, 1e-6)
        if r < 0.05 or r > 0.5:
            continue
        if len(P):
            uv, z = vj.proyectar(P)
            dentro = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < vj.camara.width) & (uv[:, 1] < vj.camara.height)
            traslape = dentro.mean()
        else:
            traslape = 0.5
        if traslape < 0.3:
            continue
        # relación base/altura ideal ~0.15 (≈9°): buena precisión, poca oclusión y rango de búsqueda corto
        peso_angulo = np.exp(-((r - 0.15) / 0.12) ** 2)
        # misma orientación (evita pares con rotación fuerte entre pasadas)
        cos_rot = (np.trace(vi.R @ vj.R.T) - 1) / 2
        cand.append((traslape * peso_angulo * (0.5 + 0.5 * max(cos_rot, 0)), j))
    cand.sort(reverse=True)
    return [j for _, j in cand[:n]]


@dataclass
class GeometriaPar:
    Ki: np.ndarray; Di: np.ndarray; Kj: np.ndarray; Dj: np.ndarray
    Rri: np.ndarray; Rrj: np.ndarray        # cámara -> marco rectificado
    Kn_i: np.ndarray; Kn_j: np.ndarray      # intrínsecos rectificados (j desplazada dmin)
    W: int; H: int                          # tamaño rectificado
    f: float; B: float; dmin: int; num: int  # disparidad buscada: [0, num) + dmin


def geometria_par(vi: Vista, vj: Vista, ei: float, ej: float, w: int, h: int, todas: np.ndarray) -> GeometriaPar | None:
    """Rectificación propia: el eje X rectificado es siempre la línea base (Ci -> Cj), así las líneas
    epipolares son horizontales sin importar la dirección de vuelo, y la disparidad es positiva.
    El punto principal de vj se desplaza la disparidad mínima esperada para que el SGM sólo busque
    en el rango útil (menos tiempo y sin perder la franja izquierda de la imagen).
    """
    Ki, Di = k_y_distorsion(vi.camara, ei)
    Kj, Dj = k_y_distorsion(vj.camara, ej)
    base = vj.centro - vi.centro
    B = float(np.linalg.norm(base))
    if B < 1e-6:
        return None
    ex = base / B
    zm = vi.R[2] + vj.R[2]
    ey = np.cross(zm, ex); ey /= np.linalg.norm(ey)
    ez = np.cross(ex, ey)
    Rn = np.stack([ex, ey, ez])            # mundo -> rectificado
    Rri, Rrj = Rn @ vi.R.T, Rn @ vj.R.T    # cámara -> rectificado

    # Puntos de enlace en el marco rectificado => disparidad esperada (d = f B / Z)
    f = float(Ki[0, 0] + Ki[1, 1]) / 2
    P = vi.puntos if len(vi.puntos) >= 20 else todas
    Xr = vi.a_camara(P) @ Rri.T
    Xr = Xr[Xr[:, 2] > 0]
    if len(Xr) < 10:
        return None
    d_esp = f * B / Xr[:, 2]
    dmin = int(np.floor(np.percentile(d_esp, 1) * 0.85)) - 8
    dmax = float(np.percentile(d_esp, 99) * 1.15) + 8
    num = int(np.ceil((dmax - dmin) / 16.0) * 16)
    if num > 512 or num <= 0:
        return None

    # Extensión de la imagen i en el marco rectificado
    bordes = np.array([[0, 0], [w, 0], [w, h], [0, h], [w / 2, 0], [w, h / 2], [w / 2, h], [0, h / 2]], float)
    rayos = np.c_[(bordes[:, 0] - Ki[0, 2]) / Ki[0, 0], (bordes[:, 1] - Ki[1, 2]) / Ki[1, 1], np.ones(len(bordes))]
    r = rayos @ Rri.T
    px = f * r[:, :2] / r[:, 2:3]
    x0, y0 = px.min(0)
    x1, y1 = px.max(0)
    W, H = int(np.ceil(x1 - x0)), int(np.ceil(y1 - y0))
    lim = 2.2 * max(w, h)
    if W > lim or H > lim:
        return None
    Kn_i = np.array([[f, 0, -x0], [0, f, -y0], [0, 0, 1.0]])
    Kn_j = Kn_i.copy(); Kn_j[0, 2] += dmin    # desplaza j: disparidad buscada = d - dmin
    return GeometriaPar(Ki, Di, Kj, Dj, Rri, Rrj, Kn_i, Kn_j, W, H, f, B, dmin, num)


def _profundidad_par(vi: Vista, vj: Vista, cache: _Cache, Kud: np.ndarray, speckle: int, todas: np.ndarray):
    """Mapa de profundidad (en la rejilla sin distorsión de vi) a partir del par (vi, vj)."""
    ei, ej = cache.escala[vi.nombre], cache.escala[vj.nombre]
    gi, gj = cache.gris[vi.nombre], cache.gris[vj.nombre]
    h, w = gi.shape
    g = geometria_par(vi, vj, ei, ej, w, h, todas)
    if g is None:
        return None
    Ki, Di, Kj, Dj, Rri, Rrj = g.Ki, g.Di, g.Kj, g.Dj, g.Rri, g.Rrj
    Kn_i, Kn_j, W, H, f, B, dmin, num = g.Kn_i, g.Kn_j, g.W, g.H, g.f, g.B, g.dmin, g.num
    m1 = cv2.initUndistortRectifyMap(Ki, Di, Rri, Kn_i, (W, H), cv2.CV_32FC1)
    m2 = cv2.initUndistortRectifyMap(Kj, Dj, Rrj, Kn_j, (W, H), cv2.CV_32FC1)
    a = cv2.remap(gi, m1[0], m1[1], cv2.INTER_LINEAR, borderValue=0)
    b = cv2.remap(gj, m2[0], m2[1], cv2.INTER_LINEAR, borderValue=0)
    bs = 5
    sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=num, blockSize=bs,
                                 P1=8 * bs * bs, P2=32 * bs * bs, disp12MaxDiff=1,
                                 uniquenessRatio=10, speckleWindowSize=speckle, speckleRange=2,
                                 preFilterCap=63, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disp = sgbm.compute(a, b).astype(np.float32) / 16.0
    disp[(disp < 0) | (a == 0)] = np.nan
    disp = disp + dmin
    P1 = Kn_i
    R1 = Rri
    vv, uu = np.nonzero(np.isfinite(disp) & (disp > 0.5))
    if len(uu) == 0:
        return None
    dd = disp[vv, uu]
    Z = f * B / dd
    cx, cy = P1[0, 2], P1[1, 2]
    X = (uu - cx) * Z / f
    Y = (vv - cy) * Z / f
    Xrect = np.stack([X, Y, Z], 1)
    Xc = Xrect @ R1  # R1^T aplicado a filas
    ok = Xc[:, 2] > 0
    Xc = Xc[ok]
    u = Kud[0, 0] * Xc[:, 0] / Xc[:, 2] + Kud[0, 2]
    v = Kud[1, 1] * Xc[:, 1] / Xc[:, 2] + Kud[1, 2]
    ui, vi_ = np.round(u).astype(np.int64), np.round(v).astype(np.int64)
    ok = (ui >= 0) & (vi_ >= 0) & (ui < w) & (vi_ < h)
    ui, vi_, z = ui[ok], vi_[ok], Xc[ok, 2]
    prof = np.full((h, w), np.inf, np.float32)
    orden = np.argsort(-z)
    prof[vi_[orden], ui[orden]] = z[orden]
    prof[~np.isfinite(prof)] = np.nan
    # rellena huecos de 1 px dejados por el remuestreo
    m = np.isnan(prof)
    if m.any():
        tmp = np.where(m, 0, prof)
        cnt = cv2.blur((~m).astype(np.float32), (3, 3))
        avg = cv2.blur(tmp, (3, 3)) / np.maximum(cnt, 1e-6)
        relleno = m & (cnt > 0.5)
        prof[relleno] = avg[relleno]
    return prof


def densificar(vistas: list[Vista], calidad: dict, log=print, progreso=None):
    """Devuelve (xyz float64 Nx3, rgb uint8 Nx3, confianza uint8 N, info dict)."""
    lado = calidad["lado_denso"]
    n_vec = calidad["vecinos_densos"]
    speckle = int(FILTRO_SPECKLE[calidad["filtro"]] * (lado / 1400) ** 2)
    log(f"  Cargando {len(vistas)} imágenes a {lado}px...")
    cache = _Cache(vistas, lado)
    todas = np.concatenate([v.puntos for v in vistas if len(v.puntos)]) if any(len(v.puntos) for v in vistas) else np.zeros((0, 3))
    if len(todas) > 20000:
        todas = todas[np.random.default_rng(1).choice(len(todas), 20000, replace=False)]

    def procesar(i: int):
        vi = vistas[i]
        vec = _vecinas(i, vistas, n_vec)
        if not vec:
            return None
        esc = cache.escala[vi.nombre]
        Ki, Di = k_y_distorsion(vi.camara, esc)
        Kud = Ki.copy()
        mapas = []
        for j in vec:
            try:
                d = _profundidad_par(vi, vistas[j], cache, Kud, speckle, todas)
            except cv2.error:
                d = None
            if d is not None:
                mapas.append(d)
        if not mapas:
            return None
        pila = np.stack(mapas)
        with np.errstate(all="ignore"):
            med = np.nanmedian(pila, axis=0)
            acuerdo = np.sum(np.abs(pila - med[None]) <= 0.012 * med[None], axis=0)
        minimo = 2 if len(mapas) >= 2 else 1
        valido = np.isfinite(med) & (acuerdo >= minimo)
        # filtro por rango de profundidad plausible
        zmin, zmax, _ = _rango_profundidad(vi, todas)
        valido &= (med > zmin * 0.7) & (med < zmax * 1.3)
        vv, uu = np.nonzero(valido)
        z = med[vv, uu].astype(np.float64)
        x = (uu - Kud[0, 2]) / Kud[0, 0] * z
        y = (vv - Kud[1, 2]) / Kud[1, 1] * z
        Xc = np.stack([x, y, z], 1)
        Xw = (Xc - vi.t) @ vi.R  # R^T (Xc - t)
        rgb_ud = cv2.undistort(cache.rgb[vi.nombre], Ki, Di, None, Kud)
        col = rgb_ud[vv, uu]
        return Xw.astype(np.float64), col, len(mapas)

    principales = list(range(len(vistas)))
    nubes, colores, ids = [], [], []
    hechos = 0
    with ThreadPoolExecutor(max_workers=calidad.get("hilos", 8)) as ex:
        for k, r in enumerate(ex.map(procesar, principales)):
            hechos += 1
            if r is not None:
                nubes.append(r[0]); colores.append(r[1]); ids.append(np.full(len(r[0]), k, np.int32))
            if progreso:
                progreso(0.85 * hechos / len(principales))
    if not nubes:
        raise RuntimeError("No se pudo generar la nube densa (sin pares válidos).")
    X = np.concatenate(nubes); C = np.concatenate(colores); I = np.concatenate(ids)
    del nubes, colores, ids
    log(f"  {len(X):,} puntos crudos de {len(vistas)} mapas de profundidad; fusionando...")

    vox = tamano_voxel(vistas, todas, cache.escala, calidad)
    minc = calidad.get("confianza_minima", 2)
    xyz, rgb, conf = fusionar(X, C, I, len(vistas), vox, minc, np)
    if progreso:
        progreso(1.0)
    log(f"  Nube densa: {len(xyz):,} puntos (vóxel {vox * 100:.1f} cm, confianza ≥ {minc})")
    return xyz, rgb, conf, {"puntos": int(len(xyz)), "voxel_m": vox}


def tamano_voxel(vistas, todas, escalas, calidad) -> float:
    """Tamaño de vóxel ≈ GSD de los mapas de profundidad."""
    zs = [_rango_profundidad(v, todas)[2] for v in vistas[:50]]
    f_esc = np.median([v.camara.mean_focal_length() * escalas[v.nombre] for v in vistas[:50]])
    return float(np.median(zs) / f_esc) * calidad.get("factor_voxel", 1.0)


def fusionar(X, C, I, n_vistas: int, vox: float, minc: int, xp):
    """Fusión en vóxeles con confianza = cámaras distintas por vóxel. xp = numpy o cupy.

    Devuelve arreglos de NumPy (xyz float64, rgb uint8, conf uint8).
    """
    clave = xp.floor(X / vox).astype(xp.int64)
    clave -= clave.min(0)
    dims = clave.max(0) + 1
    lineal = (clave[:, 0] * dims[1] + clave[:, 1]) * dims[2] + clave[:, 2]
    del clave
    uniq, inv, cuenta = xp.unique(lineal, return_inverse=True, return_counts=True)
    del lineal
    inv = inv.ravel()
    n = len(uniq)
    xyz = xp.stack([xp.bincount(inv, X[:, k].astype(xp.float64), n) for k in range(3)], 1) / cuenta[:, None]
    rgb = xp.stack([xp.bincount(inv, C[:, k].astype(xp.float64), n) for k in range(3)], 1) / cuenta[:, None]
    par = xp.unique(inv.astype(xp.int64) * (n_vistas + 1) + I)
    conf = xp.bincount((par // (n_vistas + 1)).astype(xp.int64), minlength=n)
    conf = xp.minimum(conf, 255).astype(xp.uint8)
    keep = conf >= minc
    xyz, rgb, conf = xyz[keep], rgb[keep].astype(xp.uint8), conf[keep]
    xyz, rgb, conf = _quitar_aislados(xyz, rgb, conf, vox, xp)
    if xp is not np:
        xyz, rgb, conf = xp.asnumpy(xyz), xp.asnumpy(rgb), xp.asnumpy(conf)
    return xyz, rgb, conf


def _quitar_aislados(xyz, rgb, conf, vox, xp=np):
    """Elimina puntos muy alejados de la superficie local (ruido por encima/debajo)."""
    celda = vox * 8
    ij = xp.floor(xyz[:, :2] / celda).astype(xp.int64)
    ij -= ij.min(0)
    lin = ij[:, 0] * (ij[:, 1].max() + 1) + ij[:, 1]
    orden = xp.lexsort(xp.stack([xyz[:, 2], lin.astype(xp.float64)]))
    lin_o = lin[orden]
    corte = xp.concatenate([xp.zeros(1, bool), lin_o[1:] != lin_o[:-1]])
    ini = xp.nonzero(corte)[0]
    ini = xp.concatenate([xp.zeros(1, ini.dtype), ini])
    cnt = xp.diff(xp.concatenate([ini, xp.asarray([len(lin_o)], ini.dtype)]))
    zo = xyz[orden, 2]
    med = zo[ini + cnt // 2]
    q1 = zo[ini + cnt // 4]
    q3 = zo[ini + (3 * cnt) // 4]
    grupo = xp.cumsum(corte)
    iqr = xp.maximum(q3 - q1, vox * 4)
    ok_o = xp.abs(zo - med[grupo]) <= 3.0 * iqr[grupo] + vox * 10
    ok = xp.empty_like(ok_o)
    ok[orden] = ok_o
    return xyz[ok], rgb[ok], conf[ok]
