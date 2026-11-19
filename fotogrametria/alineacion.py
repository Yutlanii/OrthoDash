"""Pasos 2 y 3: alinear imágenes (SfM) y optimizar cámaras (ajuste de haces con GPS).

Equivalente a Metashape:
  matchPhotos(generic_preselection + reference_preselection, keypoint_limit, tiepoint_limit)
  alignCameras()
  optimizeCameras()  -> ajuste de haces con las posiciones GPS como observaciones (precisión 10 m)
"""
from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from .geo import MarcoLocal
from .metadata import leer_imagen_reducida, Foto, focal_pixeles, leer_imagen


@dataclass
class Vista:
    """Cámara orientada lista para densificar y ortorrectificar (resolución completa)."""
    nombre: str
    foto: Foto
    camara: pycolmap.Camera      # intrínsecos a resolución completa
    R: np.ndarray                # cam_from_world rotación 3x3
    t: np.ndarray                # cam_from_world traslación (3,)
    sensor: str
    puntos: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # puntos de enlace visibles (mundo)

    @property
    def centro(self) -> np.ndarray:
        return -self.R.T @ self.t

    def a_camara(self, X: np.ndarray) -> np.ndarray:
        return X @ self.R.T + self.t

    def proyectar(self, X: np.ndarray):
        """Proyecta puntos del mundo (N,3) a pixeles (N,2) con distorsión. Devuelve (uv, profundidad).

        uv sigue la convención de COLMAP: el centro del primer píxel es (0.5, 0.5). Para muestrear con
        OpenCV (centro en 0) hay que restar 0.5.
        """
        Xc = self.a_camara(X)
        return proyectar_camara(self.camara, Xc), Xc[:, 2]


def proyectar_camara(cam: pycolmap.Camera, Xc: np.ndarray) -> np.ndarray:
    """Proyección vectorizada (modelos PINHOLE/SIMPLE_RADIAL/RADIAL/OPENCV/FULL_OPENCV sin k4-k6) en numpy."""
    z = Xc[:, 2]
    zz = np.where(np.abs(z) < 1e-9, 1e-9, z)
    x = Xc[:, 0] / zz
    y = Xc[:, 1] / zz
    p = cam.params
    m = cam.model_name
    if m in ("OPENCV", "FULL_OPENCV"):
        fx, fy, cx, cy, k1, k2, p1, p2 = p[:8]
        k3 = p[8] if m == "FULL_OPENCV" else 0.0   # k4-k6 = 0 (calibración de fábrica fija)
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
        xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    elif m == "SIMPLE_RADIAL":
        f, cx, cy, k = p
        fx = fy = f
        rad = 1 + k * (x * x + y * y)
        xd, yd = x * rad, y * rad
    elif m == "RADIAL":
        f, cx, cy, k1, k2 = p
        fx = fy = f
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 * r2
        xd, yd = x * rad, y * rad
    elif m in ("PINHOLE", "SIMPLE_PINHOLE"):
        if m == "PINHOLE":
            fx, fy, cx, cy = p
        else:
            f, cx, cy = p
            fx = fy = f
        xd, yd = x, y
    else:  # modelo no vectorizado: usar pycolmap
        return np.asarray(cam.img_from_cam(Xc.astype(np.float64), check_cheirality=False))
    return np.stack([fx * xd + cx, fy * yd + cy], axis=1)


def k_y_distorsion(cam: pycolmap.Camera, escala: float = 1.0):
    """Matriz K y coeficientes de distorsión estilo OpenCV (centro del primer píxel en 0) para la
    imagen escalada por `escala`. COLMAP pone ese centro en 0.5: cx_opencv = cx_colmap * escala - 0.5."""
    p = cam.params
    m = cam.model_name
    if m == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = p
        d = np.array([k1, k2, p1, p2, 0.0])
    elif m == "FULL_OPENCV":   # k4-k6 = 0: igual al modelo OpenCV de 5 coeficientes
        fx, fy, cx, cy, k1, k2, p1, p2, k3 = p[:9]
        d = np.array([k1, k2, p1, p2, k3])
    elif m == "SIMPLE_RADIAL":
        fx = fy = p[0]; cx, cy = p[1], p[2]; d = np.array([p[3], 0, 0, 0, 0.0])
    elif m == "RADIAL":
        fx = fy = p[0]; cx, cy = p[1], p[2]; d = np.array([p[3], p[4], 0, 0, 0.0])
    elif m == "PINHOLE":
        fx, fy, cx, cy = p; d = np.zeros(5)
    else:
        fx = fy = p[0]; cx, cy = p[1], p[2]; d = np.zeros(5)
    K = np.array([[fx * escala, 0, cx * escala - 0.5],
                  [0, fy * escala, cy * escala - 0.5],
                  [0, 0, 1.0]])
    return K, d


# ----------------------------------------------------------------------------------------------
def _escribir_proxy(foto: Foto, destino: Path, lado_max: int, reducida: bool = False) -> float:
    if reducida:   # decodificación JPEG ya reducida (modo express): mucho menos trabajo por foto
        W, H = foto.ancho, foto.alto
        factor = max([1] + [f for f in (2, 4, 8) if max(W, H) / f >= lado_max])
        img = leer_imagen_reducida(foto.ruta, factor) if factor > 1 else leer_imagen(foto.ruta)
        esc = min(1.0, lado_max / max(W, H))
        h, w = H, W
    else:
        img = leer_imagen(foto.ruta)
        h, w = img.shape[:2]
        esc = min(1.0, lado_max / max(h, w))
    if esc < 1.0 and img.shape[:2] != (round(h * esc), round(w * esc)):
        img = cv2.resize(img, (round(w * esc), round(h * esc)), interpolation=cv2.INTER_AREA)
        if img.ndim == 2:
            img = img[:, :, None]
    if img.dtype != np.uint8 or img.shape[2] == 1:
        # multiespectral/16 bits: estiramiento por percentiles para que SIFT encuentre textura
        g = img[..., 0].astype(np.float32) if img.shape[2] == 1 else cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY)
        lo, hi = np.percentile(g, (0.5, 99.5))
        g = np.clip((g - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)
        out = g
    else:
        out = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2BGR)
    destino.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 95])
    buf.tofile(str(destino))
    return esc


def _nombre_sensor(i: int) -> str:
    return f"sensor{i:02d}"


def alinear(fotos: list[Foto], sensores: dict[str, list[Foto]], principal: str, marco: MarcoLocal | None,
            trabajo: Path, calidad: dict, log=print, progreso=None) -> tuple[list[Vista], dict]:
    """Ejecuta el SfM completo y devuelve las vistas orientadas en el marco local UTM."""
    trabajo = Path(trabajo)
    dir_img = trabajo / "proxies"
    db_path = trabajo / "database.db"
    if db_path.exists():
        db_path.unlink()
    if dir_img.exists():
        shutil.rmtree(dir_img)

    # 1) Proxies (una subcarpeta por sensor => un modelo de cámara por sensor, como los grupos de
    #    calibración de Metashape)
    lado = calidad["lado_alineacion"]
    if calidad.get("proxies_reducidos"):
        # lado exacto de la escala DCT del JPEG: la foto sale de la decodificación sin redimensionar
        mayor = max(sensores[principal][0].ancho, sensores[principal][0].alto)
        lado = calidad["lado_alineacion"] = -(-mayor // max([1] + [r for r in (2, 4, 8) if mayor / r >= lado]))
    nombres_sensor = {s: _nombre_sensor(i) for i, s in enumerate(sensores)}
    tareas = []
    for s, lista in sensores.items():
        for f in lista:
            tareas.append((f, dir_img / nombres_sensor[s] / (f.ruta.stem + ".jpg")))
    log(f"  Preparando {len(tareas)} imágenes (lado máx. {lado}px)...")
    escalas: dict[str, float] = {}
    reducida = bool(calidad.get("proxies_reducidos", False))
    with ThreadPoolExecutor(max_workers=16 if reducida else 8) as ex:
        for k, ((f, d), esc) in enumerate(zip(tareas, ex.map(lambda a: _escribir_proxy(*a, lado, reducida), tareas))):
            escalas[f"{d.parent.name}/{d.name}"] = esc
            if progreso:
                progreso(0.15 * (k + 1) / len(tareas))
    foto_de = {f"{nombres_sensor[s]}/{f.ruta.stem}.jpg": f for s, lista in sensores.items() for f in lista}

    # 2) Base de datos + intrínsecos iniciales desde EXIF
    lector = pycolmap.ImageReaderOptions()
    lector.camera_model = "RADIAL"  # f, cx, cy, k1, k2 (focal única, como Metashape)
    pycolmap.Database.open(db_path).close()  # crea la base vacía
    pycolmap.import_images(db_path, dir_img, camera_mode=pycolmap.CameraMode.PER_FOLDER, options=lector)
    db = pycolmap.Database.open(db_path)
    try:
        imagenes = db.read_all_images()
        cam_de_img = {im.camera_id: im.name for im in imagenes}
        for cam in db.read_all_cameras():
            f = foto_de.get(cam_de_img.get(cam.camera_id, ""))
            fp = focal_pixeles(f) if f else None
            esc = escalas.get(cam_de_img.get(cam.camera_id, ""), 1.0)
            if f is not None and f.calibracion:
                # calibración de fábrica (Sequoia, DJI DewarpData): FULL_OPENCV con k4-k6 = 0, que es el
                # modelo OpenCV de 5 coeficientes. Queda fija en los ajustes: en terreno plano con fotos
                # verticales la focal no es observable (se confunde con la altura; medido: derivaba de
                # 3595 a ~6000 px y deformaba el bloque, 17 m de error frente al GPS)
                fx, fy, cx, cy, k1, k2, p1, p2, k3 = f.calibracion
                nueva = pycolmap.Camera.create_from_model_name(cam.camera_id, "FULL_OPENCV", fx * esc, cam.width, cam.height)
                nueva.params = [fx * esc, fy * esc, cx * esc, cy * esc, k1, k2, p1, p2, k3, 0, 0, 0]
                nueva.has_prior_focal_length = True
                db.update_camera(nueva)
                calidad["calibracion_fija"] = True
            elif fp:
                cam.params = [fp * esc, cam.width / 2, cam.height / 2, 0, 0]
                cam.has_prior_focal_length = True
                db.update_camera(cam)
        # Posiciones GPS como "pose priors" (referencia de Metashape, precisión 10 m por defecto)
        n_prior = 0
        if marco is not None:
            db.clear_pose_priors()
            sigma_h, sigma_v = calidad["sigma_gps_h"], calidad["sigma_gps_v"]
            cov = np.diag([sigma_h ** 2, sigma_h ** 2, sigma_v ** 2])
            for im in imagenes:
                f = foto_de[im.name]
                if f.lat is None or f.alt is None:
                    continue
                pos = marco.a_local(f.lat, f.lon, f.alt)
                prior = pycolmap.PosePrior()
                prior.position = pos
                prior.position_covariance = cov
                prior.coordinate_system = pycolmap.PosePriorCoordinateSystem.CARTESIAN
                prior.corr_data_id = im.data_id
                db.write_pose_prior(prior)
                n_prior += 1
    finally:
        db.close()
    log(f"  {len(imagenes)} imágenes en base de datos, {n_prior} con GPS")

    # 3-4) Puntos clave (SIFT) y emparejamiento: preselección por referencia (GPS) o genérica
    if n_prior >= 3:
        huella = _huella_estimada(fotos, principal, sensores)
        # sin altura conocida, sin límite de distancia: sólo cuentan las vecinas más cercanas
        modo_par = ("espacial", float(max(huella * 1.2, 5.0)) if huella else 1e5)
    else:
        modo_par = ("exhaustivo" if len(fotos) <= 200 else "secuencial", 0.0)
    info_gpu = calidad.get("gpu")
    if info_gpu is not None and info_gpu.colmap is None:
        info_gpu = None
    if info_gpu is not None:
        try:
            _sift_gpu(info_gpu, db_path, dir_img, lado, calidad, modo_par, log, progreso)
        except RuntimeError as e:
            if calidad.get("gpu_obligatoria"):
                raise
            log(f"  [aviso] La GPU falló; se continúa con la CPU. Detalle: {str(e).splitlines()[0]}")
            info_gpu = None
    if info_gpu is None:
        _sift_cpu(db_path, dir_img, lado, calidad, modo_par, log, progreso)
    if progreso:
        progreso(0.65)

    # 5) Reconstrucción de cámaras (alignCameras). Primero SfM global (GLOMAP): promedia rotaciones y
    #    posiciones de todo el bloque a la vez y es ~2x más rápido que el incremental con la misma
    #    precisión (medido: 42 s vs 85 s, 0,765 vs 0,772 px). Si deja imágenes fuera, se usa el
    #    incremental con GPS, más robusto en bloques débiles.
    salida = trabajo / "sfm"
    if salida.exists():
        shutil.rmtree(salida)
    salida.mkdir(parents=True)
    log("  Reconstruyendo cámaras (SfM global)...")
    glob_ = pycolmap.GlobalPipelineOptions()
    glob_.multiple_models = True
    glob_.min_model_size = 3
    glob_.mapper.global_positioning.use_gpu = False  # Ceres de COLMAP no trae CUDA
    glob_.mapper.bundle_adjustment.ceres.use_gpu = False
    if calidad.get("calibracion_fija"):
        glob_.mapper.bundle_adjustment.refine_extra_params = False
        glob_.mapper.bundle_adjustment.refine_focal_length = False
    if calidad.get("sfm_ligero"):
        # modo express: pocas trazas por vista bastan para las poses (el ajuste final con GPS las
        # refina); medido en 309 fotos: 50 s en vez de 97 s con la misma reproyección
        glob_.mapper.track_required_tracks_per_view = 100
        glob_.mapper.skip_retriangulation = True
        glob_.mapper.ba_num_iterations = 1
    modelos = pycolmap.global_mapping(db_path, dir_img, salida, options=glob_)
    rec = max(modelos.values(), key=lambda r: r.num_reg_images()) if modelos else None
    if rec is None or rec.num_reg_images() < 0.95 * len(imagenes):
        n = rec.num_reg_images() if rec else 0
        log(f"  SfM global alineó {n}/{len(imagenes)}; se reintenta con SfM incremental...")
        shutil.rmtree(salida)
        salida.mkdir(parents=True)
        mapa = pycolmap.IncrementalPipelineOptions()
        mapa.use_prior_position = n_prior >= 3
        mapa.use_robust_loss_on_prior_position = True
        mapa.ba_refine_principal_point = False
        mapa.ba_refine_extra_params = not calidad.get("calibracion_fija", False)
        mapa.ba_refine_focal_length = not calidad.get("calibracion_fija", False)
        mapa.multiple_models = True
        mapa.min_model_size = 3
        modelos = pycolmap.incremental_mapping(db_path, dir_img, salida, options=mapa)
        if not modelos:
            raise RuntimeError("No se pudieron alinear las imágenes (poca textura o poco traslape).")
        inc = max(modelos.values(), key=lambda r: r.num_reg_images())
        if rec is None or inc.num_reg_images() >= rec.num_reg_images():
            rec = inc
    log(f"  Alineadas {rec.num_reg_images()}/{len(imagenes)} imágenes; {rec.num_points3D()} puntos de enlace")
    if progreso:
        progreso(0.85)

    # 6) Optimizar cámaras: Sim3 a GPS + ajuste de haces con observaciones GPS
    stats = {"imagenes": len(imagenes), "alineadas": rec.num_reg_images(), "puntos_enlace": rec.num_points3D()}
    if n_prior >= 3:
        rec = _georreferenciar(rec, foto_de, marco, calidad, log)
        res = _residuos_gps(rec, foto_de, marco, calidad.get("gps_corregido"))
        stats["error_gps_m"] = res
        log(f"  Error de cámaras vs GPS (RMSE): XY {res['xy']:.2f} m, Z {res['z']:.2f} m")
    stats["error_reproyeccion_px"] = float(rec.compute_mean_reprojection_error())
    stats["longitud_media_track"] = float(rec.compute_mean_track_length())
    log(f"  Error medio de reproyección: {stats['error_reproyeccion_px']:.3f} px (a {lado}px)")
    rec.write(salida)

    vistas = _a_vistas(rec, foto_de, nombres_sensor, escalas)
    vistas = _completar_rig(vistas, fotos, sensores, nombres_sensor, foto_de, log)
    if progreso:
        progreso(1.0)
    return vistas, stats


def _log_emparejamiento(modo_par, calidad, log):
    tipo, radio = modo_par
    if tipo == "espacial":
        log(f"  Emparejando por posición GPS (radio {radio:.0f} m, máx {calidad['vecinos_emparejamiento']} vecinos)...")
    else:
        log(f"  Sin GPS: emparejamiento {tipo}...")


def _sift_cpu(db_path, dir_img, lado, calidad, modo_par, log, progreso):
    ext = pycolmap.FeatureExtractionOptions()
    ext.max_image_size = lado
    ext.sift.max_num_features = calidad["puntos_clave"]
    ext.sift.estimate_affine_shape = False
    log(f"  Detectando puntos clave (límite {calidad['puntos_clave']})...")
    pycolmap.extract_features(db_path, dir_img, extraction_options=ext)
    if progreso:
        progreso(0.40)
    mopt = pycolmap.FeatureMatchingOptions()
    mopt.max_num_matches = calidad["puntos_enlace"]
    mopt.guided_matching = bool(calidad.get("emparejamiento_guiado", False))
    mopt.num_threads = max(2, min(8, (os.cpu_count() or 4)))  # limita memoria del emparejador CPU
    _log_emparejamiento(modo_par, calidad, log)
    tipo, radio = modo_par
    if tipo == "espacial":
        popt = pycolmap.SpatialPairingOptions()
        popt.ignore_z = True
        popt.max_num_neighbors = calidad["vecinos_emparejamiento"]
        popt.max_distance = radio
        pycolmap.match_spatial(db_path, matching_options=mopt, pairing_options=popt)
    elif tipo == "exhaustivo":
        pycolmap.match_exhaustive(db_path, matching_options=mopt)
    else:
        sopt = pycolmap.SequentialPairingOptions()
        sopt.overlap = 25
        sopt.loop_detection = False
        pycolmap.match_sequential(db_path, matching_options=mopt, pairing_options=sopt)


def _sift_gpu(info_gpu, db_path, dir_img, lado, calidad, modo_par, log, progreso):
    """Lo mismo que _sift_cpu, con el COLMAP-CUDA de la misma versión sobre la misma base de datos."""
    from .gpu import ejecutar_colmap, uso_ajeno

    # Windows reparte la GPU por turnos: si otro programa la ocupa mucho, SiftGPU (miles de
    # lanzamientos cortos) va hasta 10x más lento (medido: 4 s/imagen con un editor 3D abierto, vs
    # 0,4 s) y la CPU es más rápida. El emparejamiento sigue en GPU: es menos sensible.
    ocupada = sum(u for _, u in uso_ajeno())
    if ocupada >= 30:
        log(f"  La GPU está ocupada por otros programas ({ocupada}%): puntos clave con todos los núcleos de la CPU")
        ext = pycolmap.FeatureExtractionOptions()
        ext.max_image_size = lado
        ext.sift.max_num_features = calidad["puntos_clave"]
        ext.sift.estimate_affine_shape = False
        pycolmap.extract_features(db_path, dir_img, extraction_options=ext)
        _emparejar_gpu(info_gpu, db_path, calidad, calidad["puntos_clave"], modo_par, log, progreso)
        return

    # SiftGPU da menos coincidencias por par que el SIFT de CPU con el mismo límite (medido: ~300 vs
    # ~540): se detectan 1,5x puntos, que en GPU cuestan pocos segundos.
    puntos = int(calidad["puntos_clave"] * 1.5)
    log(f"  Detectando puntos clave en GPU (límite {puntos})...")
    ejecutar_colmap(info_gpu, [
        "feature_extractor", "--database_path", str(db_path), "--image_path", str(dir_img),
        "--ImageReader.camera_model", "RADIAL", "--ImageReader.single_camera_per_folder", "1",
        "--FeatureExtraction.use_gpu", "1", "--FeatureExtraction.max_image_size", str(lado),
        "--SiftExtraction.max_num_features", str(puntos),
        "--SiftExtraction.estimate_affine_shape", "0"], log)
    _emparejar_gpu(info_gpu, db_path, calidad, puntos, modo_par, log, progreso)


def _emparejar_gpu(info_gpu, db_path, calidad, puntos, modo_par, log, progreso):
    from .gpu import ejecutar_colmap

    if progreso:
        progreso(0.40)
    _log_emparejamiento(modo_par, calidad, log)
    tipo, radio = modo_par
    # El emparejador GPU descarta los puntos que excedan max_num_matches; las orientaciones SIFT
    # múltiples dan más puntos que el límite, así que se deja margen para usarlos todos.
    comun = ["--database_path", str(db_path), "--FeatureMatching.use_gpu", "1",
             "--FeatureMatching.max_num_matches", str(2 * puntos),
             "--FeatureMatching.guided_matching", "1" if calidad.get("emparejamiento_guiado") else "0"]
    if tipo == "espacial":
        ejecutar_colmap(info_gpu, ["spatial_matcher", *comun, "--SpatialMatching.ignore_z", "1",
                                   "--SpatialMatching.max_num_neighbors", str(calidad["vecinos_emparejamiento"]),
                                   "--SpatialMatching.max_distance", f"{radio:.3f}"], log)
    elif tipo == "exhaustivo":
        ejecutar_colmap(info_gpu, ["exhaustive_matcher", *comun], log)
    else:
        ejecutar_colmap(info_gpu, ["sequential_matcher", *comun, "--SequentialMatching.overlap", "25",
                                   "--SequentialMatching.loop_detection", "0"], log)


def _huella_estimada(fotos, principal, sensores) -> float | None:
    """Ancho de terreno que cubre una foto, o None si no se conoce la altura sobre el suelo (sin
    RelativeAltitude en el XMP, p. ej. Parrot): suponer una altura podría dejar fuera pares válidos."""
    f = sensores[principal][0]
    alturas = [x.alt_relativa for x in sensores[principal] if x.alt_relativa]
    if not alturas:
        return None
    fp = focal_pixeles(f) or 1.2 * max(f.ancho, f.alto)
    return float(np.median(alturas) * max(f.ancho, f.alto) / fp)


def _pose(img) -> tuple[np.ndarray, np.ndarray]:
    cfw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
    M = np.asarray(cfw.matrix())
    return M[:, :3].copy(), M[:, 3].copy()


def _centros(rec, foto_de, marco, gps_corregido: dict | None = None):
    nombres, C, G = [], [], []
    for im in rec.images.values():
        if not im.has_pose:
            continue
        f = foto_de[im.name]
        if f.lat is None or f.alt is None:
            continue
        R, t = _pose(im)
        nombres.append(im.name)
        C.append(-R.T @ t)
        G.append(gps_corregido[im.name] if gps_corregido and im.name in gps_corregido else marco.a_local(f.lat, f.lon, f.alt))
    return nombres, np.array(C), np.array(G)


def _retraso_gps(rec, foto_de, marco, log) -> dict | None:
    """Detecta y corrige un retraso del GPS respecto al disparo (p. ej. Parrot Sequoia).

    Con el bloque ya alineado al GPS, si las cámaras quedan sistemáticamente por delante (o detrás)
    de su posición GPS en la dirección de vuelo, el GPS se registró con retraso. Se corrige cada
    posición en proporción a su velocidad (desplazamiento entre fotos consecutivas). Medido en un vuelo
    Sequoia: +6,9 m con 13,4 m entre fotos (~0,5 intervalos). Sin efecto si no es significativo."""
    nombres, C, G = _centros(rec, foto_de, marco)
    if len(nombres) < 10:
        return None
    vel = np.zeros_like(G)
    for carpeta in {n.split("/")[0] for n in nombres}:      # cada sensor es una secuencia propia
        idx = sorted((i for i, n in enumerate(nombres) if n.startswith(carpeta + "/")),
                     key=lambda i: (str(foto_de[nombres[i]].fecha or ""), nombres[i]))
        if len(idx) >= 3:
            vel[idx] = np.gradient(G[idx], axis=0)
    paso = np.linalg.norm(vel[:, :2], axis=1)
    mov = paso > 0.5
    if mov.sum() < 10:
        return None
    a_lo_largo = np.sum((C - G)[:, :2] * vel[:, :2], 1) / np.maximum(paso, 1e-6)
    al = a_lo_largo[mov]
    media, error = float(np.median(al)), float(al.std() / np.sqrt(len(al)))
    if abs(media) < 1.0 or abs(media) < 4 * error:
        return None
    k = float(np.clip(np.median(al / paso[mov]), -1.5, 1.5))
    log(f"  Retraso del GPS detectado: {media:+.1f} m en la dirección de vuelo (~{k:.2f} intervalos entre fotos); corregido")
    return {n: g + k * v for n, g, v in zip(nombres, G, vel)}


def _georreferenciar(rec, foto_de, marco, calidad, log):
    nombres, C, G = _centros(rec, foto_de, marco)
    ransac = pycolmap.RANSACOptions()
    ransac.max_error = float(calidad["sigma_gps_h"]) * 1.5
    sim = pycolmap.align_reconstruction_to_locations(rec, nombres, G, 3, ransac)
    if sim is not None:
        rec.transform(sim)
    corregido = _retraso_gps(rec, foto_de, marco, log)
    if corregido:
        calidad["gps_corregido"] = corregido
        nombres, C, G = _centros(rec, foto_de, marco, corregido)
        sim = pycolmap.align_reconstruction_to_locations(rec, nombres, G, 3, ransac)
        if sim is not None:
            rec.transform(sim)
    # Ajuste de haces con observaciones GPS (optimizeCameras). El punto principal queda fijo: en vuelos
    # verticales sin puntos de control se confunde con la inclinación de las cámaras (medido: liberarlo
    # mueve el mapa ~2 m con el mismo ajuste al GPS).
    try:
        ba = pycolmap.BundleAdjustmentOptions()
        ba.refine_principal_point = False
        ba.refine_extra_params = not calidad.get("calibracion_fija", False)
        ba.refine_focal_length = not calidad.get("calibracion_fija", False)
        ba.print_summary = False
        prior_opt = pycolmap.PosePriorBundleAdjustmentOptions()
        prior_opt.alignment_ransac.max_error = float(calidad["sigma_gps_h"]) * 1.5
        cfg = pycolmap.BundleAdjustmentConfig()
        for iid in rec.reg_image_ids():
            cfg.add_image(iid)
        priors = []
        db_priors = {}
        sigma_h, sigma_v = calidad["sigma_gps_h"], calidad["sigma_gps_v"]
        for im in rec.images.values():
            if not im.has_pose:
                continue
            f = foto_de[im.name]
            if f.lat is None:
                continue
            p = pycolmap.PosePrior()
            p.position = corregido[im.name] if corregido and im.name in corregido else marco.a_local(f.lat, f.lon, f.alt)
            p.position_covariance = np.diag([sigma_h ** 2, sigma_h ** 2, sigma_v ** 2])
            p.coordinate_system = pycolmap.PosePriorCoordinateSystem.CARTESIAN
            p.corr_data_id = im.data_id
            priors.append(p)
        ajustador = pycolmap.create_pose_prior_bundle_adjuster(ba, prior_opt, cfg, priors, rec)
        ajustador.solve()
        log("  Ajuste de haces con GPS completado")
    except Exception as e:  # versión de pycolmap sin BA con priors: BA normal + re-alineación
        log(f"  (BA con GPS no disponible: {e}); usando BA estándar")
        ba = pycolmap.BundleAdjustmentOptions()
        ba.refine_principal_point = False
        ba.refine_extra_params = not calidad.get("calibracion_fija", False)
        ba.refine_focal_length = not calidad.get("calibracion_fija", False)
        pycolmap.bundle_adjustment(rec, ba)
        nombres, C, G = _centros(rec, foto_de, marco, calidad.get("gps_corregido"))
        sim = pycolmap.align_reconstruction_to_locations(rec, nombres, G, 3, ransac)
        if sim is not None:
            rec.transform(sim)
    return rec


def _residuos_gps(rec, foto_de, marco, gps_corregido: dict | None = None) -> dict:
    _, C, G = _centros(rec, foto_de, marco, gps_corregido)
    d = C - G
    return {"xy": float(np.sqrt(np.mean(np.sum(d[:, :2] ** 2, 1)))), "z": float(np.sqrt(np.mean(d[:, 2] ** 2)))}


def _a_vistas(rec, foto_de, nombres_sensor, escalas) -> list[Vista]:
    inv_sensor = {v: k for k, v in nombres_sensor.items()}
    xyz = {pid: p.xyz for pid, p in rec.points3D.items()}
    vistas = []
    for im in rec.images.values():
        if not im.has_pose:
            continue
        f = foto_de[im.name]
        cam = pycolmap.Camera(rec.cameras[im.camera_id].todict()) if hasattr(pycolmap.Camera, "__init__") else rec.cameras[im.camera_id]
        try:
            cam = pycolmap.Camera.create_from_model_name(0, cam.model_name, 1.0, cam.width, cam.height)
            cam.params = rec.cameras[im.camera_id].params
        except Exception:
            cam = rec.cameras[im.camera_id]
        cam.rescale(f.ancho, f.alto)
        R, t = _pose(im)
        pts = np.array([xyz[p.point3D_id] for p in im.points2D if p.has_point3D()]) if im.num_points3D else np.zeros((0, 3))
        vistas.append(Vista(nombre=im.name, foto=f, camara=cam, R=R, t=t, sensor=inv_sensor[im.name.split("/")[0]], puntos=pts))
    return vistas


def _completar_rig(vistas, fotos, sensores, nombres_sensor, foto_de, log) -> list[Vista]:
    """Bandas multiespectrales que no se alinearon: heredan la pose del disparo (rig) de otra banda."""
    if len(sensores) <= 1:
        return vistas
    # referencia del mismo disparo, preferentemente del mismo tipo de sensor: una banda multiespectral
    # comparte casi la misma orientación que las demás bandas del rig, no la de la cámara RGB
    por_captura: dict[tuple[str, str], Vista] = {}
    for v in vistas:
        por_captura.setdefault((v.foto.clave_captura, v.foto.tipo), v)
        por_captura.setdefault((v.foto.clave_captura, "*"), v)
    alineadas = {v.nombre for v in vistas}
    camaras_sensor = {v.sensor: v.camara for v in vistas}
    nuevas = 0
    for s, lista in sensores.items():
        for f in lista:
            nombre = f"{nombres_sensor[s]}/{f.ruta.stem}.jpg"
            if nombre in alineadas:
                continue
            ref = por_captura.get((f.clave_captura, f.tipo)) or por_captura.get((f.clave_captura, "*"))
            if ref is None:
                continue
            cam = camaras_sensor.get(s)
            if cam is None:
                fp = focal_pixeles(f) or 1.2 * max(f.ancho, f.alto)
                cam = pycolmap.Camera.create_from_model_name(0, "RADIAL", fp, f.ancho, f.alto)
            vistas.append(Vista(nombre=nombre, foto=f, camara=cam, R=ref.R.copy(), t=ref.t.copy(), sensor=s, puntos=ref.puntos))
            nuevas += 1
    if nuevas:
        log(f"  {nuevas} imágenes de bandas completadas con la pose del disparo (rig multicámara)")
    return vistas
