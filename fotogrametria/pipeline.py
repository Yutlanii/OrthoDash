"""Flujo completo automático: imágenes -> ortomosaico, DEM y nube de puntos.

Todos los parámetros se deducen de los datos (GSD, traslape, número de imágenes, sensores, RAM).
El usuario sólo elige la carpeta (y opcionalmente la calidad).
"""
from __future__ import annotations

import csv
import json
import os
import pickle
import shutil
import time
from pathlib import Path

import numpy as np

from . import alineacion, color, densa, dem as dem_mod, exportar, express, geo, gpu as gpu_mod, indices, metadata, ortomosaico, radiometria, rig

CALIDADES = {
    #            alineación                     | nube densa             | ortomosaico
    "rapida": dict(lado_alineacion=2000, puntos_clave=6000, puntos_enlace=4096, vecinos_emparejamiento=15,
                   lado_denso=1000, vecinos_densos=3, filtro="agresivo", factor_orto=2.0, caminos_sgm=4),
    "normal": dict(lado_alineacion=2800, puntos_clave=8000, puntos_enlace=6000, vecinos_emparejamiento=20,
                   lado_denso=1400, vecinos_densos=4, filtro="agresivo", factor_orto=1.0, caminos_sgm=4),
    "alta":   dict(lado_alineacion=4000, puntos_clave=12000, puntos_enlace=8192, vecinos_emparejamiento=25,
                   lado_denso=2000, vecinos_densos=5, filtro="moderado", factor_orto=1.0, caminos_sgm=8,
                   emparejamiento_guiado=True),
    # Sólo ortomosaico, en segundos: alineación ligera (fotos ~1000-1400 px decodificadas reducidas, SfM
    # global con pocas trazas), sin nube densa (superficie de las coincidencias ya verificadas) y
    # ortomosaico desde fotos leídas a media resolución. Ver express.py.
    "express": dict(lado_alineacion=1000, puntos_clave=2000, puntos_enlace=2000, vecinos_emparejamiento=8,
                    lado_denso=1000, vecinos_densos=3, filtro="agresivo", factor_orto=2.0, caminos_sgm=4,
                    express=True, proxies_reducidos=True, sfm_ligero=True, reduccion_orto=2, res_dem_m=0.5),
}

ETAPAS = [  # (clave, nombre visible, peso en la barra de progreso)
    ("carga", "Cargando imágenes", 2),
    ("alineacion", "Alineando imágenes y optimizando cámaras", 40),
    ("densa", "Creando nube de puntos densa", 22),
    ("dem", "Creando modelo de elevación (DEM)", 4),
    ("orto", "Creando ortomosaico", 27),
    ("exportar", "Guardando resultados", 5),
]


def _ram_gb() -> float:
    try:
        import ctypes

        class M(ctypes.Structure):
            _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                        ("avail", ctypes.c_ulonglong), ("a", ctypes.c_ulonglong), ("b", ctypes.c_ulonglong),
                        ("c", ctypes.c_ulonglong), ("d", ctypes.c_ulonglong), ("e", ctypes.c_ulonglong)]
        m = M(); m.l = ctypes.sizeof(M)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.avail / 1024 ** 3
    except Exception:
        return 8.0


class Progreso:
    def __init__(self, cb=None):
        self.cb = cb
        self.total = sum(p for _, _, p in ETAPAS)
        self.base = 0.0
        self.actual = None

    def etapa(self, clave):
        acum = 0
        for k, nombre, p in ETAPAS:
            if k == clave:
                self.base, self.peso, self.actual = acum, p, nombre
                break
            acum += p
        self(0.0)

    def __call__(self, frac: float):
        if self.cb:
            self.cb(self.actual, min(1.0, (self.base + self.peso * frac) / self.total), min(1.0, frac))


def procesar(carpeta, salida=None, calidad: str = "normal", log=print, progreso_cb=None, reanudar: bool = False,
             gpu: str = "auto") -> dict:
    """gpu: "auto" (usar la tarjeta gráfica si hay), "si" (exigirla) o "no" (sólo CPU)."""
    carpeta = Path(carpeta)
    salida = Path(salida) if salida else carpeta.parent / f"{carpeta.name} - Resultados"
    salida.mkdir(parents=True, exist_ok=True)
    trabajo = salida / "_trabajo"
    trabajo.mkdir(exist_ok=True)
    if not reanudar:   # sin informe, una carpeta a medio sobrescribir no se abre como si estuviera completa
        for viejo in ("informe.json", "informe.html"):
            (salida / viejo).unlink(missing_ok=True)
    cal = dict(CALIDADES[calidad])
    cal.update(sigma_gps_h=5.0, sigma_gps_v=10.0, hilos=max(2, (os.cpu_count() or 4)), confianza_minima=1)
    prog = Progreso(progreso_cb)
    tiempos: dict[str, float] = {}
    t_total = time.time()

    # 1) Cargar imágenes ------------------------------------------------------------------------
    t = time.time(); prog.etapa("carga")
    log(f"1/6 Cargando imágenes de: {carpeta}")
    cal["gpu"] = gpu_mod.resolver(gpu, log)
    cal["gpu_obligatoria"] = gpu == "si"
    fotos = metadata.escanear_carpeta(carpeta, prog, log)
    if len(fotos) < 3:
        raise RuntimeError(f"Se necesitan al menos 3 imágenes (se encontraron {len(fotos)} legibles).")
    lev = metadata.analizar(fotos, log)
    fotos, sensores, principal = lev.fotos, lev.sensores, lev.principal
    for s, l in sensores.items():
        log(f"  Sensor {s}: {len(l)} imágenes{'  (principal)' if s == principal else ''}")
    if len(sensores[principal]) < 3:
        raise RuntimeError("Se necesitan al menos 3 imágenes de la cámara principal.")
    con_gps = [f for f in fotos if f.lat is not None and f.alt is not None]
    marco = geo.crear_marco([f.lat for f in con_gps], [f.lon for f in con_gps], [f.alt for f in con_gps]) if len(con_gps) >= 3 else None
    if marco:
        log(f"  Sistema de coordenadas: WGS 84 / UTM (EPSG:{marco.epsg})")
    else:
        log("  [aviso] Las imágenes no tienen GPS: el resultado no estará georreferenciado ni a escala real.")
    gsd_exif = metadata.gsd_estimado(sensores[principal][0])
    tiempos["Carga"] = time.time() - t

    # 2-3) Alinear + optimizar cámaras ----------------------------------------------------------
    t = time.time(); prog.etapa("alineacion")
    log("2/6 Alineando imágenes y optimizando cámaras")
    cache_al = trabajo / "alineacion.pkl"
    if reanudar and cache_al.exists():
        vistas, st_al = pickle.load(open(cache_al, "rb"))
        log(f"  (reutilizando alineación previa: {len(vistas)} cámaras)")
    else:
        if len(sensores) > 1 and rig.es_rig(sensores, principal):
            # sistema multicámara (Mavic 3M, P4M, MicaSense, Sequoia): se alinea sólo la cámara principal
            # y cada banda hereda la pose de su disparo con la rotación fija entre cámaras (ver rig.py)
            log(f"  Cámara multiespectral sincronizada: se alinean sólo las {len(sensores[principal])} fotos de la "
                f"cámara principal y las bandas se orientan como rig")
            vistas, st_al = alineacion.alinear(sensores[principal], {principal: sensores[principal]}, principal,
                                               marco, trabajo, cal, log, prog)
            vistas = vistas + rig.orientar_bandas(vistas, sensores, principal, log, cal["hilos"])
        else:
            vistas, st_al = alineacion.alinear(fotos, sensores, principal, marco, trabajo, cal, log, prog)
        pickle.dump((vistas, st_al), open(cache_al, "wb"))
    tiempos["Alineación + optimización"] = time.time() - t
    vp = [v for v in vistas if v.sensor == principal]
    if len(vp) < 3:
        raise RuntimeError("Muy pocas imágenes alineadas del sensor principal.")
    # igualación de exposición entre fotos RGB (sólo si varía; ver color.py)
    ganancias = None if not sensores[principal][0].es_rgb else         color.ganancias_exposicion(trabajo / "database.db", trabajo / "proxies", [v.nombre for v in vp], log)

    # 4) Nube densa -----------------------------------------------------------------------------
    t = time.time(); prog.etapa("densa")
    if cal.get("express"):
        log("3/6 Express: superficie a partir de las coincidencias ya verificadas (sin nube densa)")
        escalas = {v.nombre: min(1.0, cal["lado_alineacion"] / max(v.foto.ancho, v.foto.alto)) for v in vp}
        xyz, rgb = express.triangular_coincidencias(trabajo / "database.db", vp, escalas, trabajo / "proxies")
        if len(xyz) < 100:
            raise RuntimeError("Muy pocos puntos para el modo express; usa la calidad Rápida.")
        conf = np.ones(len(xyz), np.uint8)
        st_d = {"puntos": int(len(xyz)), "voxel_m": cal["res_dem_m"]}
        log(f"  {len(xyz):,} puntos del terreno")
        prog(1.0)
    else:
        log("3/6 Creando nube de puntos densa")
        xyz, rgb, conf, st_d = _con_gpu(cal, log, "nube densa",
                                        lambda: _modulo_gpu("densa_gpu").densificar(vp, cal, log, prog),
                                        lambda: densa.densificar(vp, cal, log, prog))
    tiempos["Nube densa"] = time.time() - t

    # 5) DEM -----------------------------------------------------------------------------------
    t = time.time(); prog.etapa("dem")
    log("4/6 Creando modelo digital de elevación")
    res_dem = st_d["voxel_m"] * 1.0
    # express: puntos dispersos, la zona con datos es irregular; un cierre mayor evita huecos en el borde
    dem, cobertura = dem_mod.construir_dem(xyz, vp, res_dem, log, radio_m=30.0 if cal.get("express") else 10.0)
    prog(1.0)
    tiempos["DEM"] = time.time() - t

    # GSD real (altura sobre el DEM / focal en pixeles)
    zsuelo = float(np.nanmedian(dem.datos))
    gsd = float(np.median([(v.centro[2] - zsuelo) / v.camara.mean_focal_length() for v in vp]))
    log(f"  GSD medido: {gsd * 100:.2f} cm/pixel" + (f" (EXIF: {gsd_exif * 100:.2f})" if gsd_exif else ""))

    # 6) Ortomosaico ---------------------------------------------------------------------------
    t = time.time(); prog.etapa("orto")
    log("5/6 Creando ortomosaico")
    crs = marco.crs if marco else None
    origen = marco.origen if marco else np.zeros(3)
    res_orto = gsd * cal["factor_orto"]
    area = np.isfinite(dem.datos).sum() * dem.res ** 2
    max_px = 1.5e9
    if area / res_orto ** 2 > max_px:
        res_orto = float(np.sqrt(area / max_px))
        log(f"  (área grande: resolución ajustada a {res_orto * 100:.2f} cm para limitar el tamaño)")
    ram = _ram_gb()
    resultados = {}
    hilos_orto = int(min(cal["hilos"], max(2, ram // 3)))
    es_ms = not sensores[principal][0].es_rgb
    sensores_ms = [s for s in sensores if not sensores[s][0].es_rgb]
    if not es_ms:
        ruta_orto = salida / "ortomosaico.tif"
        info_o = _orto(cal, log)(vp, dem, res_orto, ruta_orto, origen, crs, log=log,
                                 progreso=lambda f: prog(f * (0.6 if sensores_ms else 1.0)),
                                 hilos=hilos_orto, max_ram_gb=max(2.0, ram * 0.4),
                                 reduccion=cal.get("reduccion_orto", 1), ganancias=ganancias)
        resultados["ortomosaico"] = ruta_orto
    if sensores_ms:
        try:
            resultados.update(_orto_multiespectral(vistas, sensores, sensores_ms, dem, gsd, cal, salida, trabajo,
                                                   origen, crs, log, prog, hilos_orto, ram, es_ms))
        except Exception as e:
            if es_ms:   # sin RGB, el multiespectral es el resultado principal
                raise
            # con RGB, un problema en las bandas no debe costar el ortomosaico RGB, el DEM y la nube
            log(f"  [aviso] No se pudo generar el ortomosaico multiespectral ({type(e).__name__}: {e}); "
                "se entregan los demás resultados")
        if es_ms:
            info_o = {"resolucion_m": res_orto}
        if "ortomosaico_multiespectral" in resultados:
            ruta = salida / "ortomosaico_todas_las_bandas.tif"
            k = indices.apilar_bandas(resultados.get("ortomosaico"), resultados["ortomosaico_multiespectral"], ruta,
                                      **exportar.compresion(cal))
            resultados["ortomosaico_todas_las_bandas"] = ruta
            log(f"  Imagen con todas las bandas en 8 bits guardada (escala de reflectancia {k:.4g})")
    tiempos["Ortomosaico"] = time.time() - t

    # 7) Exportar ------------------------------------------------------------------------------
    t = time.time(); prog.etapa("exportar")
    log("6/6 Guardando nube de puntos, DEM e informe")
    if cal.get("express"):   # nube dispersa: pequeña, sólo LAZ
        resultados["nube_dispersa"] = exportar.exportar_nube(salida / "nube_dispersa.laz", xyz, rgb, conf, origen, crs)
    else:
        resultados["nube_densa"] = exportar.exportar_nube(salida / "nube_densa.laz", xyz, rgb, conf, origen, crs)
        prog(0.4)
        exportar.exportar_ply(salida / "nube_densa.ply", xyz, rgb, origen)
    exportar.nube_visor(trabajo / "visor", xyz, rgb, origen)
    exportar.exportar_dem(salida / "dem.tif", dem, origen, crs)
    resultados["dem"] = salida / "dem.tif"
    prog(0.7)
    previas = {}
    exportar.vista_previa_dem(salida / "dem_vista_previa.png", dem)
    previas["Modelo digital de elevación (DSM)"] = "dem_vista_previa.png"
    if "ortomosaico" in resultados:
        exportar.vista_previa_orto(resultados["ortomosaico"], salida / "ortomosaico_vista_previa.jpg")
        previas = {"Ortomosaico": "ortomosaico_vista_previa.jpg", **previas}
    if "ndvi_png" in resultados:
        previas["NDVI"] = resultados["ndvi_png"].name
    _camaras_csv(salida / "camaras.csv", vistas, marco)
    tiempos["Exportación"] = time.time() - t
    tiempos["TOTAL"] = time.time() - t_total

    resumen = {
        "Imágenes": len(fotos),
        "Cámaras alineadas": f"{st_al['alineadas']} / {st_al['imagenes']}",
        "Puntos de enlace": st_al["puntos_enlace"],
        "Error de reproyección (px)": round(st_al["error_reproyeccion_px"], 3),
        "Error cámaras vs GPS XY / Z (m)": (f"{st_al['error_gps_m']['xy']:.2f} / {st_al['error_gps_m']['z']:.2f}" if "error_gps_m" in st_al else "sin GPS"),
        "Puntos nube densa": f"{st_d['puntos']:,}",
        "Resolución DEM (cm)": round(dem.res * 100, 2),
        "Resolución ortomosaico (cm)": round(info_o["resolucion_m"] * 100, 2),
        "GSD (cm)": round(gsd * 100, 2),
        "Área (ha)": round(area / 1e4, 3),
        "Sistema de coordenadas": f"EPSG:{marco.epsg}" if marco else "local (sin GPS)",
        "Calidad": calidad,
        "Procesador": f"GPU {cal['gpu'].nombre}" if cal["gpu"] else "CPU",
    }
    datos = {"resumen": resumen, "tiempos": tiempos, "alineacion": st_al, "densa": st_d,
             "indices": resultados.get("indices_valores", {}),
             "resultados": {k: str(v) for k, v in resultados.items() if k != "indices_valores"}}
    (salida / "informe.json").write_text(json.dumps(datos, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    exportar.informe_html(salida / "informe.html", datos, previas)
    resultados["informe"] = salida / "informe.html"
    prog(1.0)
    log(f"Listo en {tiempos['TOTAL'] / 60:.1f} min. Resultados en: {salida}")
    if not reanudar:
        shutil.rmtree(trabajo / "proxies", ignore_errors=True)
    return {"salida": salida, **resultados, "resumen": resumen}


def _orto_multiespectral(vistas, sensores, sensores_ms, dem, gsd_principal, cal, salida, trabajo, origen, crs,
                         log, prog, hilos, ram, es_principal):
    import rasterio

    orden = sorted(sensores_ms, key=lambda s: (sensores[s][0].longitud_onda or 1e9, sensores[s][0].banda))
    nombres = [sensores[s][0].banda for s in orden]
    log(f"  Bandas multiespectrales: {', '.join(nombres)}")
    zsuelo = float(np.nanmedian(dem.datos))
    temporales = []
    res_ms = None
    for i, s in enumerate(orden):
        vs = [v for v in vistas if v.sensor == s]
        if not vs:
            log(f"  [aviso] banda {sensores[s][0].banda} sin cámaras orientadas; se omite")
            continue
        if res_ms is None:
            res_ms = float(np.median([(v.centro[2] - zsuelo) / v.camara.mean_focal_length() for v in vs])) * cal["factor_orto"]
        radiometria.preparar([v.foto for v in vs])
        tmp = trabajo / f"banda_{i}.tif"
        base, peso = (0.6, 0.4) if not es_principal else (0.0, 1.0)
        _orto(cal, log)(vs, dem, res_ms, tmp, origen, crs, multiespectral=True,
                                          nombres_bandas=[sensores[s][0].banda], log=log,
                                          progreso=lambda f, i=i: prog(base + peso * (i + f) / len(orden)),
                                          hilos=hilos, max_ram_gb=max(2.0, ram * 0.4),
                                          reduccion=cal.get("reduccion_orto", 1))
        temporales.append((sensores[s][0].banda, tmp))
    ruta = salida / "ortomosaico_multiespectral.tif"
    with rasterio.open(temporales[0][1]) as src:
        perfil = src.profile.copy()
    perfil.update(count=len(temporales), num_threads="all_cpus", **exportar.compresion(cal))
    with rasterio.Env(GDAL_NUM_THREADS="ALL_CPUS"), rasterio.open(ruta, "w", **perfil) as dst:
        for k, (n, tmp) in enumerate(temporales, 1):
            with rasterio.open(tmp) as src:
                for _, win in src.block_windows(1):
                    dst.write(src.read(1, window=win), k, window=win)
            dst.set_band_description(k, n)
        from rasterio.enums import Resampling
        dst.build_overviews([2, 4, 8, 16, 32], Resampling.average)
    out = {"ortomosaico_multiespectral": ruta}
    nombres_ind = indices.calcular_indices(ruta, salida / "indices_vegetacion.tif", salida / "ndvi_vista_previa.png",
                                           **exportar.compresion(cal))
    if nombres_ind:
        out["indices"] = salida / "indices_vegetacion.tif"
        out["indices_valores"] = indices.informe_valores(out["indices"])
        if "NDVI" in nombres_ind:
            out["ndvi_png"] = salida / "ndvi_vista_previa.png"
        for n, v in out["indices_valores"].items():
            log(f"  {n}: mediana {v['mediana']:.2f} (p5 {v['p5']:.2f}, p95 {v['p95']:.2f})")
    else:
        log("  [aviso] Faltan bandas para calcular índices de vegetación (hacen falta NIR y Red, Green o RedEdge)")
    limitacion = radiometria.limitacion(sensores[orden[0]][0])
    if limitacion:
        log(f"  [aviso] Radiometría: {limitacion}. Para valores exactos usa un panel de calibración.")
    return out


def _modulo_gpu(nombre):
    import importlib
    return importlib.import_module(f".{nombre}", __package__)


def _usa_cupy(cal) -> bool:
    return cal.get("gpu") is not None and cal["gpu"].cupy


def _con_gpu(cal, log, etapa, en_gpu, en_cpu):
    """Ejecuta la etapa en GPU si se puede; en modo automático, si la GPU falla se repite en CPU."""
    if not _usa_cupy(cal):
        return en_cpu()
    try:
        return en_gpu()
    except Exception as e:   # memoria insuficiente, controlador, compilación...
        if cal.get("gpu_obligatoria"):
            raise
        log(f"  [aviso] La GPU falló en {etapa} ({type(e).__name__}: {str(e).splitlines()[0][:120]}); se usa la CPU")
        return en_cpu()


def _orto(cal, log):
    """construir_ortomosaico de GPU o CPU con la misma firma (y respaldo en CPU)."""
    def f(*a, **k):
        return _con_gpu(cal, log, "ortomosaico",
                        lambda: _modulo_gpu("ortomosaico_gpu").construir_ortomosaico(*a, **k),
                        lambda: ortomosaico.construir_ortomosaico(*a, **k))
    return f


def _camaras_csv(ruta: Path, vistas, marco) -> None:
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["imagen", "sensor", "este", "norte", "altura", "lat", "lon"])
        for v in vistas:
            c = v.centro
            if marco:
                lat, lon, _ = marco.a_wgs84(c)
                g = c + marco.origen
            else:
                lat = lon = ""
                g = c
            w.writerow([v.foto.nombre, v.sensor, f"{g[0]:.3f}", f"{g[1]:.3f}", f"{g[2]:.3f}", lat, lon])
