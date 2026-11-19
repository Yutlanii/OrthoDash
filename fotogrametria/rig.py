"""Cámaras multiespectrales como sistema multicámara (rig), como el "multi-camera system" de Metashape.

En un dron multiespectral (DJI Mavic 3M / P4M, MicaSense, Sequoia) todas las cámaras disparan a la vez
y están fijas entre sí. En lugar de alinear cada foto de cada banda (5x más fotos, y emparejar NIR con
RGB es difícil), se alinea sólo la cámara principal y cada banda hereda la pose de su disparo con una
rotación fija entre cámaras, medida una sola vez:

  1. En ~16 disparos se emparejan las fotos de todas las cámaras entre sí (SIFT sobre intensidad y sobre
     magnitud de gradiente: entre bandas el contraste puede invertirse, p. ej. rojo vs borde rojo).
  2. Para cada par de cámaras se juntan las coincidencias de todos los disparos y se estima la rotación
     entre ambas (RANSAC sobre rayos sin distorsión, con la calibración de cada cámara). La separación
     entre lentes (~cm) es despreciable a la altura de vuelo, así que basta la rotación.
  3. A cada banda se llega desde la principal por el camino de pares con más coincidencias.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pycolmap

from .alineacion import Vista, k_y_distorsion
from .metadata import Foto, focal_pixeles, leer_imagen, leer_imagen_reducida

LADO = 1296                # lado mayor al que se reducen las fotos para emparejar
MUESTRAS = 16              # disparos usados para medir el rig (sobran: 3 parámetros por banda, miles de puntos)
MIN_INLIERS = 40           # coincidencias mínimas para aceptar una rotación entre cámaras
UMBRAL_PX = 2.5            # error máximo (px a LADO) para contar una coincidencia como válida


def camara_completa(f: Foto) -> pycolmap.Camera:
    """Modelo de cámara a resolución completa: calibración de fábrica si existe, si no focal EXIF."""
    if f.calibracion:
        fx, fy, cx, cy, k1, k2, p1, p2, k3 = f.calibracion
        c = pycolmap.Camera.create_from_model_name(0, "FULL_OPENCV", fx, f.ancho, f.alto)
        c.params = [fx, fy, cx, cy, k1, k2, p1, p2, k3, 0, 0, 0]
        return c
    fp = focal_pixeles(f) or 1.2 * max(f.ancho, f.alto)
    return pycolmap.Camera.create_from_model_name(0, "RADIAL", fp, f.ancho, f.alto)


def es_rig(sensores: dict[str, list[Foto]], principal: str) -> bool:
    """Las demás cámaras disparan con la principal (comparten disparo en >= 80 % de sus fotos)."""
    otras = [s for s in sensores if s != principal]
    if not otras:
        return False
    claves = {f.clave_captura for f in sensores[principal]}
    return all(np.mean([f.clave_captura in claves for f in sensores[s]]) >= 0.8 for s in otras)


def _rasgos(f: Foto):
    """Keypoints (px a resolución completa, convención COLMAP) y descriptores de intensidad y gradiente."""
    reduccion = max([1] + [r for r in (2, 4, 8) if max(f.ancho, f.alto) / r >= LADO])
    img = leer_imagen_reducida(f.ruta, reduccion) if (reduccion > 1 and f.es_rgb) else leer_imagen(f.ruta)
    g = (img[..., 1] if img.shape[2] >= 3 else img[..., 0]).astype(np.float32)   # canal verde si es RGB
    e0 = img.shape[1] / f.ancho                     # escala de la imagen leída respecto a la completa
    e = LADO / max(g.shape)
    g = cv2.resize(g, None, fx=e, fy=e, interpolation=cv2.INTER_AREA)
    e = e * e0
    def a8(x):
        lo, hi = np.percentile(x, (1, 99.5))
        return np.clip((x - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    suave = cv2.GaussianBlur(g, (0, 0), 1.0)
    grad = np.hypot(cv2.Sobel(suave, cv2.CV_32F, 1, 0), cv2.Sobel(suave, cv2.CV_32F, 0, 1))
    sift = cv2.SIFT_create(2500)
    salida = []
    for imagen in (a8(g), a8(grad)):
        kp, de = sift.detectAndCompute(imagen, None)
        uv = np.array([k.pt for k in kp], np.float64).reshape(-1, 2)
        salida.append(((uv + 0.5) / e, de))       # a resolución completa (COLMAP: centro del píxel en 0.5)
    return salida


def _rotacion(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """R que minimiza |b - R a| (Kabsch) para rayos unitarios Nx3."""
    U, _, Vt = np.linalg.svd(a.T @ b)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def _ransac_rotacion(a: np.ndarray, b: np.ndarray, umbral: float, iters: int = 1000):
    """Hipótesis sobre una muestra (rápido aunque haya cientos de miles de coincidencias) y
    refinamiento con todas."""
    rng = np.random.default_rng(0)
    m = rng.choice(len(a), min(len(a), 5000), replace=False)
    am, bm = a[m], b[m]
    mejor = None
    for _ in range(iters):
        idx = rng.choice(len(am), 3, replace=False)
        R = _rotacion(am[idx], bm[idx])
        ok = np.linalg.norm(bm - am @ R.T, axis=1) < umbral
        if mejor is None or ok.sum() > mejor.sum():
            mejor = ok
    R = _rotacion(am[mejor], bm[mejor])
    # refinamiento con todas, con umbral decreciente: las coincidencias falsas cerca del umbral sesgan
    # la rotación (medido: ~1 px de banda con un solo paso a 5 px)
    for f in (1.0, 0.7, 0.4):
        ok = np.linalg.norm(b - a @ R.T, axis=1) < umbral * f
        if ok.sum() < 3:
            break
        R = _rotacion(a[ok], b[ok])
    ok = np.linalg.norm(b - a @ R.T, axis=1) < umbral
    return R, int(ok.sum())


def orientar_bandas(vistas_principal: list[Vista], sensores: dict[str, list[Foto]], principal: str,
                    log=print, hilos: int = 8) -> list[Vista]:
    """Vistas de las demás cámaras del rig, con la pose del disparo de la principal y su rotación fija."""
    por_disparo = {v.foto.clave_captura: v for v in vistas_principal}
    otras = [s for s in sensores if s != principal]
    camaras = {s: camara_completa(sensores[s][0]) for s in sensores}
    camaras[principal] = vistas_principal[0].camara          # la principal ya está calibrada por el ajuste
    fotos_de = {s: {f.clave_captura: f for f in sensores[s]} for s in sensores}

    # 1) disparos de muestra con todas las cámaras y la principal orientada
    comunes = [k for k in por_disparo if all(k in fotos_de[s] for s in otras)]
    if not comunes:
        raise RuntimeError("Las bandas no comparten disparos con la cámara principal.")
    muestra = [comunes[int(i)] for i in np.linspace(0, len(comunes) - 1, min(MUESTRAS, len(comunes)))]
    hilos_cv = cv2.getNumThreads()
    cv2.setNumThreads(1)       # el paralelismo va por foto: OpenCV multihilo dentro de cada hilo sólo compite
    tareas = [(k, s) for k in muestra for s in sensores]
    fotos_tarea = [por_disparo[k].foto if s == principal else fotos_de[s][k] for k, s in tareas]
    with ThreadPoolExecutor(max_workers=hilos) as ex:
        rasgos = dict(zip(tareas, ex.map(_rasgos, fotos_tarea)))

    # 2) rotación entre cada par de cámaras con las coincidencias de todos los disparos de muestra
    kd = {s: k_y_distorsion(c, 1.0) for s, c in camaras.items()}
    def rayos(s, uv):   # vectorizado (cam_from_img de pycolmap es iterativo por punto: 41 s en 1 M de puntos)
        K, D = kd[s]
        n = cv2.undistortPoints((uv - 0.5).reshape(-1, 1, 2).astype(np.float64), K, D).reshape(-1, 2)   # COLMAP -> OpenCV
        d = np.c_[n, np.ones(len(n))]
        return d / np.linalg.norm(d, axis=1, keepdims=True)
    lista = [principal] + otras
    pares = {}

    def emparejar(tarea):
        a, b, k, t = tarea
        (ua, da), (ub, db) = rasgos[(k, a)][t], rasgos[(k, b)][t]
        if da is None or db is None or len(da) < 8 or len(db) < 8:
            return tarea, None
        m = [x for x, y in cv2.BFMatcher().knnMatch(da, db, k=2) if x.distance < 0.8 * y.distance]
        return tarea, (ua[[x.queryIdx for x in m]], ub[[x.trainIdx for x in m]]) if m else None
    tareas_e = [(a, b, k, t) for i, a in enumerate(lista) for b in lista[i + 1:] for k in muestra for t in range(2)]
    coincidencias: dict[tuple, list] = {}
    with ThreadPoolExecutor(max_workers=hilos) as ex:   # intensidad y gradiente, en paralelo
        for (a, b, _, _), r in ex.map(emparejar, tareas_e):
            if r is not None:
                coincidencias.setdefault((a, b), []).append(r)
    for i, a in enumerate(lista):
        for b in lista[i + 1:]:
            ra = [x[0] for x in coincidencias.get((a, b), [])]
            rb = [x[1] for x in coincidencias.get((a, b), [])]
            if not ra:
                continue
            A, B = rayos(a, np.concatenate(ra)), rayos(b, np.concatenate(rb))
            if len(A) < 3 * MIN_INLIERS:
                continue
            f_ang = UMBRAL_PX * max(sensores[a][0].ancho, sensores[a][0].alto) / LADO / camaras[a].mean_focal_length()
            R, n = _ransac_rotacion(A, B, f_ang)
            if n >= MIN_INLIERS:
                pares[(a, b)] = (R, n)              # rayo_b = R rayo_a
                pares[(b, a)] = (R.T, n)

    cv2.setNumThreads(hilos_cv)

    # 3) camino de más confianza (el eslabón más débil, el mayor posible) desde la principal
    rot = {principal: np.eye(3)}
    confianza = {principal: np.inf}
    pendientes = set(otras)
    while pendientes:
        mejor = None
        for (a, b), (R, n) in pares.items():
            if a in rot and b in pendientes:
                c = min(confianza[a], n)
                if mejor is None or c > mejor[0]:
                    mejor = (c, a, b, R)
        if mejor is None:
            break
        c, a, b, R = mejor
        rot[b] = R @ rot[a]
        confianza[b] = c
        pendientes.discard(b)
    for s in pendientes:
        log(f"  [aviso] No se pudo medir la orientación de {s} respecto a la cámara principal; se omite")

    # 4) cada foto de banda: pose del disparo de la principal + rotación fija de su cámara
    vistas = []
    for s in otras:
        if s not in rot:
            continue
        ang = np.degrees(np.arccos(np.clip((np.trace(rot[s]) - 1) / 2, -1, 1)))
        log(f"  Rig: {s} orientada respecto a la principal ({ang:.2f}°, {int(confianza[s])} coincidencias)")
        carpeta = "rig_" + sensores[s][0].banda
        for k, f in fotos_de[s].items():
            vp = por_disparo.get(k)
            if vp is None:
                continue
            R = rot[s] @ vp.R
            vistas.append(Vista(nombre=f"{carpeta}/{f.ruta.stem}.jpg", foto=f, camara=camaras[s],
                                R=R, t=-R @ vp.centro, sensor=s, puntos=vp.puntos))
    return vistas
