"""Compensación de exposición entre fotos (equivalente a la calibración de color de Metashape).

Cámaras con exposición automática (p. ej. Parrot Sequoia) cambian el brillo de una foto a otra; la
fusión multibanda del ortomosaico suaviza las costuras pero deja parches de distinto brillo. Aquí se
estima una ganancia por foto y canal con las coincidencias ya verificadas de la alineación: en cada
par se compara el color de los mismos puntos en ambas fotos (mediana del cociente) y se resuelve por
mínimos cuadrados log(g_j) - log(g_i) = cociente, regularizado hacia ganancia 1 (evita la deriva),
y se aplica sólo la diferencia de cada foto con sus vecinas (los parches), no las tendencias por zonas.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr

UMBRAL_DISPERSION = np.log(1.05)   # medido: Sequoia con parches ±6 % (se corrige); DJI Air 2S ±4 %, cuyo
                                   # brillo varía por la escena (bosque vs cantera) y no debe tocarse
REGULARIZACION = 1.0               # peso de "ganancia 1" por foto, relativo al de un par medio
MIN_COINCIDENCIAS = 30


def ganancias_exposicion(db_path: Path, dir_proxies: Path, nombres: list[str], log=print) -> dict[str, np.ndarray] | None:
    """{nombre: ganancia RGB (multiplicar la foto)} o None si no hace falta corregir."""
    db_path, dir_proxies = Path(db_path), Path(dir_proxies)
    if not db_path.exists() or not dir_proxies.exists():
        return None
    validos = set(nombres)
    db = pycolmap.Database.open(str(db_path))
    try:
        nombre_de = {im.image_id: im.name for im in db.read_all_images() if im.name in validos}
        kp = {iid: db.read_keypoints(iid)[:, :2] for iid in nombre_de}
        pares, geos = db.read_two_view_geometries()
    finally:
        db.close()

    def leer(iid):
        img = cv2.imdecode(np.fromfile(str(dir_proxies / nombre_de[iid]), np.uint8), cv2.IMREAD_COLOR)
        return iid, None if img is None or img.ndim != 3 else cv2.blur(img, (7, 7))   # color local, sin ruido
    with ThreadPoolExecutor(max_workers=8) as ex:
        imgs = {iid: im for iid, im in ex.map(leer, list(nombre_de)) if im is not None}
    indice = {iid: k for k, iid in enumerate(sorted(imgs))}
    if len(indice) < 3:
        return None

    filas, cols, vals, rhs, pesos = [], [], [], [], []
    for par, g in zip(pares, geos):
        i, j = pycolmap.pair_id_to_image_pair(par)
        m = g.inlier_matches
        if i not in indice or j not in indice or len(m) < MIN_COINCIDENCIAS:
            continue
        ci, cj = _muestrear(imgs[i], kp[i][m[:, 0]]), _muestrear(imgs[j], kp[j][m[:, 1]])
        ok = np.all((ci > 20) & (ci < 235) & (cj > 20) & (cj < 235), 1)      # sin saturación ni negro
        if ok.sum() < MIN_COINCIDENCIAS:
            continue
        r = np.median(np.log(cj[ok]) - np.log(ci[ok]), axis=0)               # log g_j - log g_i (BGR)
        e = len(filas) // 2
        filas += [e, e]; cols += [indice[j], indice[i]]; vals += [1.0, -1.0]
        rhs.append(r); pesos.append(np.sqrt(ok.sum()))
    if not rhs:
        return None
    n_ec, n = len(rhs), len(indice)
    w = np.array(pesos)
    # Regularización hacia ganancia 1 (como el compensador de OpenCV): sin ella, los pequeños sesgos
    # de cada par (el brillo cambia con el ángulo de visión) se acumulan a lo largo del bloque y una
    # esquina acaba quemada (medido: ±49 % de deriva en un vuelo de 302 fotos).
    lam = REGULARIZACION * float(np.mean(w))
    filas_r = list(range(n_ec, n_ec + n)); cols_r = list(range(n))
    A = coo_matrix((np.r_[np.array(vals) * np.repeat(w, 2), np.full(n, lam)],
                    (filas + filas_r, cols + cols_r)), shape=(n_ec + n, n)).tocsr()
    b = np.vstack([np.array(rhs) * w[:, None], np.zeros((n, 3))])
    log_g = np.stack([lsqr(A, b[:, c], atol=1e-10, btol=1e-10)[0] for c in range(3)], 1)
    # Sólo se corrige lo que difiere cada foto de sus vecinas (lo que forma parches); las tendencias por
    # zonas se conservan: la exposición automática aclara con razón una zona de bosque frente a una
    # cantera clara (medido: corregirlo todo oscurecía el bosque entero).
    vecinas = [[k] for k in range(n)]
    for e in range(n_ec):
        a_, b_ = cols[2 * e], cols[2 * e + 1]
        vecinas[a_].append(b_); vecinas[b_].append(a_)
    log_g = log_g - np.array([log_g[v].mean(0) for v in vecinas])
    dispersion = float(log_g.std(0).max())
    log(f"  Variación de exposición entre vecinas: ±{(np.exp(dispersion) - 1) * 100:.1f} %")
    if dispersion < UMBRAL_DISPERSION:
        return None
    log(f"  Exposición variable entre fotos (±{(np.exp(dispersion) - 1) * 100:.0f} %): se igualan los colores")
    inv = {k: iid for iid, k in indice.items()}
    return {nombre_de[inv[k]]: np.exp(-log_g[k, ::-1]).astype(np.float32) for k in range(n)}   # BGR -> RGB


def _muestrear(img: np.ndarray, uv: np.ndarray) -> np.ndarray:
    u = np.clip(np.round(uv[:, 0] - 0.5), 0, img.shape[1] - 1).astype(int)   # COLMAP -> píxel
    v = np.clip(np.round(uv[:, 1] - 0.5), 0, img.shape[0] - 1).astype(int)
    return img[v, u].astype(np.float32)


def aplicar(img: np.ndarray, ganancia: np.ndarray | None) -> np.ndarray:
    """Aplica la ganancia RGB a una foto de 8 bits (en su sitio si es posible)."""
    if ganancia is None or img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] < 3:
        return img
    lut = np.clip(np.arange(256)[:, None] * ganancia[None, :3], 0, 255).astype(np.uint8)   # 256x3
    out = np.empty_like(img)
    for c in range(3):
        out[..., c] = lut[img[..., c], c]
    return out
