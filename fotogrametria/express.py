"""Modo express: ortomosaico en segundos, sin nube densa.

Idea (como el "fast orthophoto" de ODM o Pix4Dreact, pero sin rehacer emparejamientos): tras una
alineación ligera, las cámaras ya están fijas y la base de datos de COLMAP guarda todas las
coincidencias verificadas entre pares de fotos. Triangularlas con las poses finales da decenas de
miles de puntos del terreno en menos de un segundo, suficientes para una superficie de proyección
suavizada (DEM aproximado) sobre la que se ortorrectifica.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pycolmap

from .alineacion import Vista

ERROR_ANGULAR_MAX = 0.004   # rad (~2-3 px a 1400 px): descarta coincidencias que no se cortan
ANGULO_MIN = np.radians(2)  # intersección mínima entre rayos (evita puntos sin profundidad fiable)


def triangular_coincidencias(db_path: Path, vistas: list[Vista], escalas: dict[str, float],
                             dir_proxies: Path | None = None):
    """Puntos 3D (mundo local) y colores de todas las coincidencias verificadas, con las poses fijas.

    Triangulación por punto medio entre los dos rayos de cada coincidencia, vectorizada por par.
    Devuelve (xyz float64 Nx3, rgb uint8 Nx3)."""
    por_nombre = {v.nombre: v for v in vistas}
    db = pycolmap.Database.open(str(db_path))
    try:
        nombre_de = {im.image_id: im.name for im in db.read_all_images()}
        kp = {}
        pares, geos = db.read_two_view_geometries()
        def rayos(iid):
            if iid not in kp:
                nombre = nombre_de[iid]
                v = por_nombre.get(nombre)
                if v is None:
                    kp[iid] = None
                else:
                    uv = db.read_keypoints(iid)[:, :2].astype(np.float64) / escalas[nombre]   # a resolución completa
                    d = np.c_[np.asarray(v.camara.cam_from_img(uv)), np.ones(len(uv))] @ v.R   # direcciones en el mundo
                    kp[iid] = (d / np.linalg.norm(d, axis=1, keepdims=True), v.centro, uv * escalas[nombre], nombre)
            return kp[iid]
        xyz, cols = [], []
        for par, g in zip(pares, geos):
            m = g.inlier_matches
            if len(m) < 8:
                continue
            i1, i2 = pycolmap.pair_id_to_image_pair(par)
            a, b = rayos(i1), rayos(i2)
            if a is None or b is None:
                continue
            d1, d2 = a[0][m[:, 0]], b[0][m[:, 1]]
            w0 = a[1] - b[1]
            bb = np.sum(d1 * d2, 1); dd = d1 @ w0; ee = d2 @ w0
            den = 1 - bb ** 2
            ok = den > np.sin(ANGULO_MIN) ** 2
            s = (bb * ee - dd) / np.where(ok, den, 1); t = (ee - bb * dd) / np.where(ok, den, 1)
            p1 = a[1] + s[:, None] * d1; p2 = b[1] + t[:, None] * d2
            sep = np.linalg.norm(p1 - p2, axis=1)
            ok &= (s > 0) & (t > 0) & (sep < ERROR_ANGULAR_MAX * np.minimum(s, t))
            if not ok.any():
                continue
            xyz.append(((p1 + p2) / 2)[ok])
            cols.append(np.c_[a[2][m[ok, 0]], np.full(ok.sum(), i1)])
    finally:
        db.close()
    if not xyz:
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
    xyz = np.concatenate(xyz)
    rgb = _colores(np.concatenate(cols), nombre_de, dir_proxies)
    return xyz, rgb


def _colores(uv_id: np.ndarray, nombre_de: dict, dir_proxies: Path | None) -> np.ndarray:
    """Color de cada punto en su foto (proxy de la alineación); gris si no hay proxies."""
    rgb = np.full((len(uv_id), 3), 160, np.uint8)
    if dir_proxies is None:
        return rgb
    ids = uv_id[:, 2].astype(np.int64)
    for iid in np.unique(ids):
        ruta = Path(dir_proxies) / nombre_de[int(iid)]
        img = cv2.imdecode(np.fromfile(str(ruta), np.uint8), cv2.IMREAD_COLOR) if ruta.exists() else None
        if img is None:
            continue
        sel = ids == iid
        u = np.clip(uv_id[sel, 0] - 0.5, 0, img.shape[1] - 1).astype(int)
        v = np.clip(uv_id[sel, 1] - 0.5, 0, img.shape[0] - 1).astype(int)
        rgb[sel] = img[v, u][:, ::-1]
    return rgb
