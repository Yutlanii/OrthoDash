"""Lectura de metadatos de imágenes de dron (EXIF + XMP) y análisis del levantamiento.

Este módulo es el único que interpreta las convenciones de cada fabricante; el resto del programa
usa sus conclusiones (tipo de sensor, banda normalizada, disparo, sensor principal, calibración).

Soporta (ver pruebas/test_camaras.py, que simula los metadatos de cada una):
  * RGB: DJI Phantom/Mavic/Air/Mini, Phantom 4 RTK, Matrice; Sony, Hasselblad, senseFly... (JPG/TIF)
  * Multiespectral de un canal por archivo: DJI P4 Multispectral y Mavic 3M, MicaSense RedEdge/Altum/
    RedEdge-P, Parrot Sequoia
  * Mezcla RGB + multiespectral en la misma carpeta (Mavic 3M, P4M, Sequoia)
  * Cámaras con varias lentes (DJI M30T/H20T/M3T: gran angular, zoom, térmica): se usa la gran angular
  * Bandas térmicas (Altum LWIR, DJI _T): se ignoran (otra física, otra resolución)
"""
from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import ExifTags, Image

Image.MAX_IMAGE_PIXELS = None

EXTENSIONES = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}

# Sufijos de banda en nombres de archivo: DJI M3M (_MS_G...), DJI RGB (_D, _W, _V, _Z, _T),
# Sequoia (_GRE, _RED, _REG, _NIR, _RGB) y MicaSense (_1 ... _10). Sólo para agrupar por disparo cuando
# faltan hora o GPS, y para deducir la banda si el XMP no la trae.
_SUFIJO_BANDA = re.compile(r"(_MS)?_(G|R|RE|NIR|D|W|V|Z|T|GRE|RED|REG|RGB|[1-9]|10)$", re.IGNORECASE)

# Nombre normalizado de banda -> longitud de onda central típica (nm) si el XMP no la trae
BANDAS = {"Blue": 475, "Green": 560, "Red": 668, "RedEdge": 717, "NIR": 842, "Panchro": 634, "LWIR": 11000}
_ALIAS_BANDA = {
    "blue": "Blue", "b": "Blue", "azul": "Blue",
    "green": "Green", "g": "Green", "gre": "Green", "verde": "Green",
    "red": "Red", "r": "Red", "rojo": "Red",
    "rededge": "RedEdge", "re": "RedEdge", "reg": "RedEdge", "redege": "RedEdge", "edge": "RedEdge",
    "nir": "NIR", "nearinfrared": "NIR", "nearir": "NIR", "infrarrojo": "NIR", "ir": "NIR",
    "panchro": "Panchro", "panchromatic": "Panchro", "pan": "Panchro",
    "lwir": "LWIR", "thermal": "LWIR", "longwaveinfrared": "LWIR", "termica": "LWIR", "t": "LWIR",
}


def normalizar_banda(nombre: str | None) -> str | None:
    """Nombre canónico de banda (Blue, Green, Red, RedEdge, NIR, Panchro, LWIR) o None si no se reconoce."""
    if not nombre:
        return None
    clave = re.sub(r"[^a-z]", "", nombre.lower())
    return _ALIAS_BANDA.get(clave)


@dataclass
class Foto:
    ruta: Path
    nombre: str
    ancho: int
    alto: int
    canales: int
    bits: int
    fabricante: str = ""
    modelo: str = ""
    banda: str = "RGB"            # "RGB" o banda normalizada (Blue, Green, Red, RedEdge, NIR, Panchro, LWIR)
    tipo: str = "rgb"             # "rgb", "multiespectral" o "termica"
    lente: str = ""               # DJI multi-lente: WideCamera, ZoomCamera, InfraredCamera...
    longitud_onda: float = 0.0    # nm (0 para RGB)
    lat: float | None = None
    lon: float | None = None
    alt: float | None = None      # altitud absoluta (m)
    alt_relativa: float | None = None
    focal_mm: float | None = None
    focal_35mm: float | None = None
    fecha: datetime | None = None
    yaw: float | None = None
    pitch: float | None = None
    clave_captura: str = ""       # agrupa las bandas tomadas en el mismo disparo
    radiometria: dict = field(default_factory=dict)
    # calibración de fábrica (px, convención OpenCV salvo cx, cy de COLMAP): fx, fy, cx, cy, k1, k2, p1, p2, k3
    calibracion: tuple | None = None
    px_por_mm: float | None = None    # tamaño de píxel del sensor (EXIF FocalPlaneXResolution)

    @property
    def es_rgb(self) -> bool:
        return self.tipo == "rgb"

    @property
    def sensor(self) -> str:
        return f"{self.modelo or 'camara'}|{self.banda}|{self.ancho}x{self.alto}"


def _racional(v) -> float:
    try:
        return float(v)
    except Exception:
        try:
            return float(v[0]) / float(v[1])
        except Exception:
            return float("nan")


def _dms(valores, ref) -> float | None:
    try:
        d, m, s = (_racional(x) for x in valores)
        v = d + m / 60.0 + s / 3600.0
        return -v if ref in ("S", "W") else v
    except Exception:
        return None


def _xmp(img: Image.Image, ruta: Path) -> str:
    xmp = img.info.get("xmp") or img.info.get("XML:com.adobe.xmp") or b""
    if not xmp:
        # TIFF: etiqueta 700 (XMP)
        try:
            xmp = img.tag_v2.get(700, b"")  # type: ignore[attr-defined]
        except Exception:
            xmp = b""
    if not xmp:
        # Búsqueda directa en los primeros bytes del archivo
        with open(ruta, "rb") as f:
            cab = f.read(256 * 1024)
        i = cab.find(b"<x:xmpmeta")
        if i >= 0:
            j = cab.find(b"</x:xmpmeta>", i)
            xmp = cab[i : j + 12] if j > 0 else b""
    if isinstance(xmp, bytes):
        xmp = xmp.decode("utf-8", "ignore")
    return xmp or ""


def _xmp_valor(xmp: str, clave: str) -> str | None:
    """Busca `prefijo:clave="valor"` o `<prefijo:clave>valor</prefijo:clave>`."""
    m = re.search(r'[\w-]+:' + clave + r'="([^"]*)"', xmp)
    if m:
        return m.group(1).strip()
    m = re.search(r"<[\w-]+:" + clave + r">\s*(?:<rdf:Seq>\s*<rdf:li>)?([^<]*)<", xmp)
    return m.group(1).strip() if m else None


def _xmp_lista(xmp: str, clave: str) -> str | None:
    """Como _xmp_valor, pero conserva todos los elementos de una lista rdf:Seq, separados por comas."""
    m = re.search(r"<[\w-]+:" + clave + r">\s*<rdf:Seq>(.*?)</rdf:Seq>", xmp, re.S)
    if m:
        return ",".join(re.findall(r"<rdf:li>\s*([^<]*?)\s*</rdf:li>", m.group(1)))
    return _xmp_valor(xmp, clave)


def _texto_exif(v) -> str:
    """Valor EXIF numérico o lista de ellos ((4800, 4800, 4800, 4800) en MicaSense), como texto."""
    return ",".join(str(_racional(x)) for x in v) if isinstance(v, (tuple, list)) else str(_racional(v))


def _modelo_radiometrico(fabricante: str, r: dict) -> str:
    """El fabricante decide el significado de los campos: Sentera, por ejemplo, también escribe
    polinomios de viñeteo, pero con los coeficientes en otro orden."""
    marca = fabricante.lower()
    if "dji" in marca:
        return "dji"
    if "micasense" in marca and len(r.get("RadiometricCalibration", "").split(",")) >= 3:
        return "micasense"
    if "parrot" in marca and len(r.get("SensorModel", "").split(",")) >= 3:
        return "sequoia"
    return "generico"


def _calibracion_fabrica(xmp: str, tags: dict, ancho: int, alto: int) -> tuple | None:
    """Calibración interna que algunos fabricantes graban en el XMP, como hace Pix4D al leerla:
      * Parrot Sequoia: Camera:PerspectiveFocalLength y Camera:PrincipalPoint (mm desde la esquina) y
        Camera:PerspectiveDistortion = R1, R2, R3, T1, T2 (mismo modelo que OpenCV k1, k2, k3, p1, p2).
      * DJI (M3E, M30, P4 RTK...): drone-dji:DewarpData = "fecha;fx,fy,cx,cy,k1,k2,p1,p2,k3", con cx, cy
        como desplazamiento desde el centro de la imagen.
    Devuelve (fx, fy, cx, cy, k1, k2, p1, p2, k3) en píxeles, o None."""
    try:
        foco, pp, dist = (_xmp_valor(xmp, k) for k in ("PerspectiveFocalLength", "PrincipalPoint", "PerspectiveDistortion"))
        res = tags.get("FocalPlaneXResolution")
        if foco and pp and dist and res and int(tags.get("FocalPlaneResolutionUnit", 0) or 0) == 4:   # px por mm
            pxmm = _racional(res)
            px, py = (float(v) * pxmm for v in pp.split(","))
            r1, r2, r3, t1, t2 = (float(v) for v in dist.split(","))
            f = float(foco) * pxmm
            return (f, f, px, py, r1, r2, t1, t2, r3)
        dw = _xmp_valor(xmp, "DewarpData")
        if dw and ";" in dw:
            fx, fy, dx, dy, k1, k2, p1, p2, k3 = (float(v) for v in dw.split(";", 1)[1].split(","))
            return (fx, fy, ancho / 2 + dx, alto / 2 + dy, k1, k2, p1, p2, k3)
    except (ValueError, TypeError):
        pass
    return None


def _xmp_float(xmp: str, clave: str) -> float | None:
    v = _xmp_valor(xmp, clave)
    try:
        return float(v) if v not in (None, "") else None
    except ValueError:
        return None


def leer_foto(ruta: Path) -> Foto:
    with Image.open(ruta) as img:
        ancho, alto = img.size
        modo = img.mode
        canales = len(img.getbands())
        bits = 16 if modo in ("I;16", "I;16B", "I;16L", "I") else (32 if modo == "F" else 8)
        exif = img.getexif()
        tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
        try:
            sub = exif.get_ifd(0x8769)
            tags.update({ExifTags.TAGS.get(k, k): v for k, v in sub.items()})
        except Exception:
            pass
        try:
            gps = {ExifTags.GPSTAGS.get(k, k): v for k, v in exif.get_ifd(0x8825).items()}
        except Exception:
            gps = {}
        xmp = _xmp(img, ruta)

    f = Foto(ruta=ruta, nombre=ruta.name, ancho=ancho, alto=alto, canales=canales, bits=bits)
    f.fabricante = str(tags.get("Make", "") or "").strip("\x00 ")
    f.modelo = str(tags.get("Model", "") or "").strip("\x00 ")
    f.focal_mm = _racional(tags["FocalLength"]) if "FocalLength" in tags else None
    f.focal_35mm = _racional(tags["FocalLengthIn35mmFilm"]) if tags.get("FocalLengthIn35mmFilm") else None
    f.calibracion = _calibracion_fabrica(xmp, tags, ancho, alto)
    unidad = {2: 25.4, 3: 10.0, 4: 1.0}.get(int(tags.get("FocalPlaneResolutionUnit", 0) or 0))   # pulgada, cm, mm
    if unidad and tags.get("FocalPlaneXResolution"):
        v = _racional(tags["FocalPlaneXResolution"]) / unidad
        f.px_por_mm = v if np.isfinite(v) and v > 10 else None

    fecha = tags.get("DateTimeOriginal") or tags.get("DateTime")
    if fecha:
        try:
            f.fecha = datetime.strptime(str(fecha).strip("\x00 "), "%Y:%m:%d %H:%M:%S")
            sub = str(tags.get("SubsecTimeOriginal", "") or "").strip("\x00 ")
            if sub.isdigit():
                f.fecha = f.fecha.replace(microsecond=int(float("0." + sub) * 1e6))
        except ValueError:
            pass

    # Posición: XMP de DJI es más precisa que EXIF
    f.lat = _xmp_float(xmp, "GpsLatitude") or _xmp_float(xmp, "Latitude")
    f.lon = _xmp_float(xmp, "GpsLongitude") or _xmp_float(xmp, "GpsLongtitude") or _xmp_float(xmp, "Longitude")
    f.alt = _xmp_float(xmp, "AbsoluteAltitude")
    if (f.lat is None or f.lon is None) and "GPSLatitude" in gps and "GPSLongitude" in gps:
        f.lat = _dms(gps["GPSLatitude"], gps.get("GPSLatitudeRef", "N"))
        f.lon = _dms(gps["GPSLongitude"], gps.get("GPSLongitudeRef", "E"))
    if f.alt is None and "GPSAltitude" in gps:
        f.alt = _racional(gps["GPSAltitude"])
        if gps.get("GPSAltitudeRef") in (1, b"\x01"):
            f.alt = -f.alt
    # posición incompleta o sin señal (algunas cámaras escriben 0,0): se trata como sin GPS
    if f.lat is None or f.lon is None or not (np.isfinite(f.lat) and np.isfinite(f.lon)) or (f.lat == 0 and f.lon == 0):
        f.lat = f.lon = None
    if f.alt is not None and not np.isfinite(f.alt):
        f.alt = None
    f.alt_relativa = _xmp_float(xmp, "RelativeAltitude")
    f.yaw = _xmp_float(xmp, "GimbalYawDegree") or _xmp_float(xmp, "Yaw")
    f.pitch = _xmp_float(xmp, "GimbalPitchDegree") or _xmp_float(xmp, "Pitch")

    # Tipo de sensor y banda espectral
    f.lente = _xmp_valor(xmp, "ImageSource") or ""
    banda_xmp = _xmp_valor(xmp, "BandName")
    onda = _xmp_float(xmp, "CentralWavelength")
    sufijo = _SUFIJO_BANDA.search(ruta.stem)
    if "infrared" in f.lente.lower() or (sufijo and sufijo.group(2).upper() == "T" and "dji" in f.fabricante.lower()):
        # DJI térmica (_T, R-JPEG de 3 canales en falso color): no es una foto de color del terreno
        f.tipo, f.banda = "termica", "LWIR"
    elif canales >= 3:
        # Una imagen de 3+ canales es siempre de color: la Parrot Sequoia, p. ej., lista "Red, Green,
        # Blue" en BandName de su cámara RGB (y sólo se leería "Red"). Las bandas son de 1 canal.
        f.tipo, f.banda = "rgb", "RGB"
    else:
        banda = normalizar_banda(banda_xmp) or (normalizar_banda(sufijo.group(2)) if sufijo else None)
        if banda is None:   # banda desconocida: se conserva el nombre (o el índice MicaSense) tal cual
            banda = (banda_xmp or (f"Banda{sufijo.group(2)}" if sufijo else "Banda1")).replace(" ", "")
        f.banda = banda
        f.longitud_onda = onda or float(BANDAS.get(banda, 0))
        f.tipo = "termica" if banda == "LWIR" or f.longitud_onda > 3000 else "multiespectral"

    # Radiometría (multiespectral): se usa para normalizar a reflectancia relativa
    if f.tipo == "multiespectral":
        r = {}
        for clave in ("BlackLevel", "SensorGain", "SensorGainAdjustment", "ExposureTime", "Irradiance",
                      "VignettingData", "VignettingPolynomial", "VignettingCenter", "CalibratedOpticalCenterX",
                      "CalibratedOpticalCenterY", "RadiometricCalibration", "SensorModel",
                      "VignettingPolynomial2D", "VignettingPolynomial2DName", "HorizontalIrradiance",
                      "DirectIrradiance", "ScatteredIrradiance", "SpectralIrradiance", "SolarElevation",
                      "IrradianceScaleToSIUnits"):
            v = _xmp_lista(xmp, clave)
            if v:
                r[clave] = v
        for clave in ("ExposureTime", "BlackLevel", "FNumber"):
            if clave not in r and clave in tags:
                r[clave] = _texto_exif(tags[clave])
        iso = tags.get("ISOSpeedRatings") or tags.get("PhotographicSensitivity") or tags.get("ISOSpeed")
        if iso:
            r["ISO"] = _texto_exif(iso)
        r["modelo"] = _modelo_radiometrico(f.fabricante, r)
        f.radiometria = r

    f.clave_captura = _SUFIJO_BANDA.sub("", ruta.stem)   # provisional; analizar() agrupa por hora y GPS
    return f


def escanear_carpeta(carpeta: Path, progreso=None, log=print) -> list[Foto]:
    # se ignoran resultados de procesamientos anteriores (ortomosaicos, DEM y proxies son TIF/JPG)
    def es_resultado(p: Path) -> bool:
        return any(d.name == "_trabajo" or d.name.endswith(" - Resultados") for d in p.parents)
    rutas = sorted(p for p in Path(carpeta).rglob("*")
                   if p.suffix.lower() in EXTENSIONES and p.is_file() and not es_resultado(p.relative_to(carpeta)))
    def leer(r):
        try:
            return leer_foto(r)
        except Exception as e:  # imagen corrupta o no soportada: se ignora y se informa
            log(f"  [aviso] no se pudo leer {r.name}: {e}")
            return None
    fotos: list[Foto] = []
    with ThreadPoolExecutor(max_workers=16) as ex:   # casi todo es espera de disco
        for i, f in enumerate(ex.map(leer, rutas)):
            if f is not None:
                fotos.append(f)
            if progreso:
                progreso((i + 1) / max(1, len(rutas)))
    return fotos


@dataclass
class Levantamiento:
    """Conclusiones sobre las fotos de una carpeta, para todo el flujo."""
    fotos: list[Foto]                       # fotos que se procesan
    sensores: dict[str, list[Foto]]         # sólo los sensores usados
    principal: str                          # guía alineación, nube densa, DEM y ortomosaico
    ignoradas: list[tuple[str, int, str]]   # (sensor, nº de fotos, motivo)


def analizar(fotos: list[Foto], log=print) -> Levantamiento:
    """Decide qué sensores se usan y cuál es el principal, y agrupa las fotos por disparo.

    * Las térmicas se ignoran (otra física; además la R-JPEG de DJI es un falso color).
    * Con varias cámaras RGB, la principal es la de más fotos y, a igualdad, la de mayor campo de
      visión; un zoom (campo de visión < 60 % del principal) se ignora: cubre poco y alinea mal.
    * Bandas multiespectrales con menos de 3 fotos se ignoran.
    """
    grupos = agrupar_sensores(fotos)
    ignoradas: list[tuple[str, int, str]] = []
    for k in [k for k, v in grupos.items() if v[0].tipo == "termica"]:
        ignoradas.append((k, len(grupos.pop(k)), "cámara térmica"))
    if not grupos:
        raise RuntimeError("No hay fotos RGB ni multiespectrales utilizables (sólo térmicas o ninguna).")
    principal = elegir_sensor_principal(grupos)
    fov_p = campo_vision(grupos[principal][0])
    for k in list(grupos):
        g0 = grupos[k][0]
        if k == principal:
            continue
        if g0.es_rgb and fov_p and campo_vision(g0) and campo_vision(g0) < 0.6 * fov_p:
            ignoradas.append((k, len(grupos.pop(k)), "zoom (campo de visión mucho menor que el principal)"))
        elif len(grupos[k]) < 3:
            ignoradas.append((k, len(grupos.pop(k)), "muy pocas fotos"))
    usadas = [f for g in grupos.values() for f in g]
    agrupar_disparos(usadas)
    for k, n, motivo in ignoradas:
        log(f"  Se ignoran {n} fotos de {k}: {motivo}")
    return Levantamiento(usadas, grupos, principal, ignoradas)


def campo_vision(f: Foto) -> float | None:
    """Ancho del campo de visión (rad, aprox.) para comparar cámaras; None si no se conoce la focal."""
    fp = focal_pixeles(f)
    return float(2 * np.arctan(max(f.ancho, f.alto) / (2 * fp))) if fp else None


def agrupar_disparos(fotos: list[Foto]) -> None:
    """Asigna `clave_captura` común a las fotos de un mismo disparo (las bandas de un rig).

    Universal: mismo instante (±0,5 s con subsegundos, ±1 s sin ellos), misma posición (±3 m) y como
    mucho una foto por sensor. Si faltan hora o GPS se usa el nombre sin sufijo de banda (convenciones
    de DJI, MicaSense y Sequoia)."""
    for f in fotos:
        if f.fecha is None:
            f.clave_captura = "nombre:" + _SUFIJO_BANDA.sub("", f.ruta.stem)
    con_hora = [f for f in fotos if f.fecha is not None]
    if not con_hora:
        return
    tol = 0.5 if any(f.fecha.microsecond for f in con_hora) else 1.0
    actual: dict[str, Foto] = {}
    n = 0
    for f in sorted(con_hora, key=lambda x: (x.fecha, x.nombre)):
        ref = next(iter(actual.values()), None)
        nueva = (ref is None or f.sensor in actual or abs((f.fecha - ref.fecha).total_seconds()) > tol
                 or _distancia_m(f, ref) > 3.0)
        if nueva:
            n += 1
            actual = {}
        actual[f.sensor] = f
        f.clave_captura = f"disparo{n:06d}"


def _distancia_m(a: Foto, b: Foto) -> float:
    if None in (a.lat, a.lon, b.lat, b.lon):
        return 0.0
    dy = (a.lat - b.lat) * 110540.0
    dx = (a.lon - b.lon) * 111320.0 * np.cos(np.radians(a.lat))
    return float(np.hypot(dx, dy))


def agrupar_sensores(fotos: list[Foto]) -> dict[str, list[Foto]]:
    grupos: dict[str, list[Foto]] = {}
    for f in fotos:
        grupos.setdefault(f.sensor, []).append(f)
    return grupos


def elegir_sensor_principal(grupos: dict[str, list[Foto]]) -> str:
    """Sensor que guía alineación, nube densa y DEM.

    RGB si existe (más resolución). Si sólo hay multiespectral, se usa la banda más nítida típica
    (verde → roja → la de menor longitud de onda, como hace Metashape con el sensor maestro).
    """
    rgb = [k for k, v in grupos.items() if v[0].es_rgb]
    if rgb:   # más fotos; a igualdad, mayor campo de visión (gran angular antes que zoom); luego resolución
        return max(rgb, key=lambda k: (len(grupos[k]), campo_vision(grupos[k][0]) or 0.0,
                                       grupos[k][0].ancho * grupos[k][0].alto))
    for preferida in ("Panchro", "Green", "Red", "RedEdge", "NIR"):   # pancromática: más resolución
        candidatos = [k for k, v in grupos.items() if v[0].banda == preferida]
        if candidatos:
            return max(candidatos, key=lambda k: len(grupos[k]))
    return min(grupos, key=lambda k: grupos[k][0].longitud_onda or 1e9)


def gsd_estimado(f: Foto) -> float | None:
    """Tamaño de pixel en suelo (m) aproximado a partir de EXIF."""
    if not (f.focal_mm and f.alt_relativa):
        return None
    if f.focal_35mm:
        # ancho del sensor equivalente: 36mm * focal/focal35
        ancho_sensor = 36.0 * f.focal_mm / f.focal_35mm
    else:
        return None
    pixel = ancho_sensor / max(f.ancho, f.alto)
    return float(f.alt_relativa * pixel / f.focal_mm)


def focal_pixeles(f: Foto) -> float | None:
    """Focal en píxeles: calibración de fábrica > focal equivalente a 35 mm > tamaño de píxel del sensor."""
    if f.calibracion:
        return float(f.calibracion[0])
    if f.focal_mm and f.focal_35mm:
        ancho_sensor = 36.0 * f.focal_mm / f.focal_35mm
        return f.focal_mm / ancho_sensor * max(f.ancho, f.alto)
    if f.focal_mm and f.px_por_mm:
        return float(f.focal_mm * f.px_por_mm)
    return None


def leer_imagen(ruta: Path, escala: float = 1.0) -> np.ndarray:
    """Lee una imagen conservando su profundidad (uint8/uint16/float). HxWxC."""
    import cv2

    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)   # etiquetas TIFF propias de DJI/MicaSense
    datos = np.fromfile(str(ruta), dtype=np.uint8)
    img = cv2.imdecode(datos, cv2.IMREAD_UNCHANGED | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        with Image.open(ruta) as im:
            img = np.array(im)
    else:
        if img.ndim == 3 and img.shape[2] >= 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB if img.shape[2] == 4 else cv2.COLOR_BGR2RGB)
    if img.ndim == 2:
        img = img[:, :, None]
    if escala != 1.0:
        h, w = img.shape[:2]
        img = cv2.resize(img, (max(1, round(w * escala)), max(1, round(h * escala))), interpolation=cv2.INTER_AREA)
        if img.ndim == 2:
            img = img[:, :, None]
    return img


def leer_imagen_reducida(ruta: Path, factor: int) -> np.ndarray:
    """Lee la imagen a 1/factor de su tamaño. Con JPEG y factor 2, 4 u 8 la reducción ocurre al
    decodificar (escala DCT), más rápido que decodificar entera y reducir; el tamaño resultante es
    ceil(lado / factor). Otros formatos: lectura normal y reducción. HxWxC."""
    import cv2

    flags = {2: cv2.IMREAD_REDUCED_COLOR_2, 4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8}
    if factor in flags and Path(ruta).suffix.lower() in (".jpg", ".jpeg"):
        img = cv2.imdecode(np.fromfile(str(ruta), dtype=np.uint8), flags[factor] | cv2.IMREAD_IGNORE_ORIENTATION)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = leer_imagen(ruta)
    h, w = img.shape[:2]
    img = cv2.resize(img, (-(-w // factor), -(-h // factor)), interpolation=cv2.INTER_AREA)
    return img[:, :, None] if img.ndim == 2 else img


def ruta_os(p: Path) -> str:
    return os.fspath(p)
