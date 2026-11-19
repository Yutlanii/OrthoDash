"""Banco de pruebas de cámaras: fotos sintéticas con los metadatos de cada fabricante.

Cada perfil reproduce lo que el programa necesita interpretar de una cámara real (nombres de archivo,
canales, bits, EXIF, XMP con los campos y formatos de su fabricante) y comprueba las conclusiones de
metadata.analizar(): sensor principal, bandas usadas, sensores ignorados, agrupación por disparo,
calibración y focal. Después pasa una foto de cada sensor usado por los lectores de imagen del flujo
(proxy de alineación, lectura reducida del ortomosaico RGB, corrección radiométrica multiespectral).

Ejecutar:  .venv\\Scripts\\python.exe -m pruebas.test_camaras
Añade aquí un perfil cada vez que llegue una cámara nueva o un caso que falló.
"""
from __future__ import annotations

import sys
import tempfile
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fotogrametria import alineacion, metadata, radiometria, rig  # noqa: E402

N_DISPAROS = 4
T0 = datetime(2026, 2, 21, 10, 0, 0)


# ---------------------------------------------------------------------------------- generación
def _exif(marca, modelo, t: datetime, lat, lon, alt, focal=None, focal35=None, px_mm=None, subseg=True):
    e = Image.Exif()
    e[271], e[272], e[306] = marca, modelo, t.strftime("%Y:%m:%d %H:%M:%S")
    sub = e.get_ifd(0x8769)
    sub[36867] = t.strftime("%Y:%m:%d %H:%M:%S")
    if subseg:
        sub[37521] = f"{t.microsecond // 1000:03d}"
    sub[33434] = 0.001
    if focal:
        sub[37386] = focal
    if focal35:
        sub[41989] = focal35
    if px_mm:
        sub[41486], sub[41487], sub[41488] = px_mm, px_mm, 4
    if lat is not None:
        g = e.get_ifd(0x8825)
        g[1], g[3] = ("N" if lat >= 0 else "S"), ("E" if lon >= 0 else "W")
        dms = lambda v: (float(int(abs(v))), float(int(abs(v) * 60 % 60)), abs(v) * 3600 % 60)
        g[2], g[4], g[5], g[6] = dms(lat), dms(lon), 0, float(alt)
    return e.tobytes()


def _xmp(**campos) -> bytes:
    """Atributos 'prefijo:Clave' y listas rdf:Seq (valor tuple) como los escriben los fabricantes."""
    attrs = " ".join(f'{k.replace("__", ":")}="{v}"' for k, v in campos.items() if not isinstance(v, tuple))
    listas = "".join(f'<{k.replace("__", ":")}><rdf:Seq>{"".join(f"<rdf:li>{x}</rdf:li>" for x in v)}</rdf:Seq>'
                     f'</{k.replace("__", ":")}>' for k, v in campos.items() if isinstance(v, tuple))
    return (f'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            f'<rdf:Description {attrs}>{listas}</rdf:Description></rdf:RDF></x:xmpmeta>').encode()


def _foto(ruta: Path, w, h, canales, bits, exif: bytes, xmp: bytes | None):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    forma = (h, w) if canales == 1 else (h, w, canales)
    img = Image.fromarray(np.zeros(forma, np.uint16 if bits == 16 else np.uint8))
    if ruta.suffix.lower() in (".tif", ".tiff"):   # el XMP va dentro del EXIF (etiqueta 700), como en los TIFF reales
        e = Image.Exif(); e.load(exif)
        if xmp:
            e[700] = xmp
        img.save(ruta, exif=e.tobytes())
    else:
        img.save(ruta, exif=exif, **({"xmp": xmp} if xmp else {}))


def _posicion(i):   # disparos separados ~14 m hacia el este
    return 20.18 + 0.0, -104.78 + i * 0.000135, 1000.0 + i * 0.1


# ---------------------------------------------------------------------------------- perfiles
def dji_mavic3(d):
    for i in range(N_DISPAROS):
        lat, lon, alt = _posicion(i); t = T0 + timedelta(seconds=2 * i)
        _foto(d / f"DJI_{i:04d}.JPG", 5280, 3956, 3, 8, _exif("DJI", "L2D-20c", t, lat, lon, alt, 12.29, 24),
              _xmp(drone__dji__GpsLatitude=lat, drone__dji__GpsLongitude=lon, drone__dji__AbsoluteAltitude=alt,
                   drone__dji__RelativeAltitude="+80.0", drone__dji__GimbalPitchDegree="-90.0"))
    return dict(principal=("rgb", "RGB"), bandas=set(), ignorados=set(), disparos=N_DISPAROS)


def dji_p4rtk(d):
    for i in range(N_DISPAROS):
        lat, lon, alt = _posicion(i); t = T0 + timedelta(seconds=2 * i)
        _foto(d / f"DJI_{i:04d}.JPG", 5472, 3648, 3, 8, _exif("DJI", "FC6310R", t, lat, lon, alt, 8.8, 24),
              _xmp(drone__dji__GpsLatitude=lat, drone__dji__GpsLongtitude=lon, drone__dji__AbsoluteAltitude=alt,
                   drone__dji__RelativeAltitude="+60.0",
                   drone__dji__DewarpData="2020-01-01;3678.38,3677.59,-10.05,6.83,-0.2659,0.1101,0.00015,-0.00016,-0.0281"))
    return dict(principal=("rgb", "RGB"), bandas=set(), ignorados=set(), disparos=N_DISPAROS,
                calibracion=(3678.38, 5472 / 2 - 10.05))


def dji_p4m(d):   # DJI_xxx0.JPG (RGB) + DJI_xxx1..5.TIF; sin sufijo de banda en el nombre
    for i in range(N_DISPAROS):
        lat, lon, alt = _posicion(i); t = T0 + timedelta(seconds=2 * i, milliseconds=100)
        base = 10 * (i + 1)
        _foto(d / f"DJI_{base:04d}.JPG", 1600, 1300, 3, 8, _exif("DJI", "FC6360", t, lat, lon, alt, 5.74, 40),
              _xmp(drone__dji__GpsLatitude=lat, drone__dji__GpsLongitude=lon, drone__dji__AbsoluteAltitude=alt,
                   drone__dji__RelativeAltitude="+50.0", Camera__BandName="RGB"))
        for k, (banda, nm) in enumerate([("Blue", 450), ("Green", 560), ("Red", 650), ("RedEdge", 730), ("NIR", 840)], 1):
            _foto(d / f"DJI_{base + k:04d}.TIF", 1600, 1300, 1, 16, _exif("DJI", "FC6360", t, lat, lon, alt, 5.74, 40),
                  _xmp(drone__dji__GpsLatitude=lat, drone__dji__GpsLongitude=lon, drone__dji__AbsoluteAltitude=alt,
                       drone__dji__RelativeAltitude="+50.0", Camera__BandName=banda, Camera__CentralWavelength=nm,
                       drone__dji__BlackLevel="4096", drone__dji__SensorGain="1.0", drone__dji__Irradiance="500"))
    return dict(principal=("rgb", "RGB"), bandas={"Blue", "Green", "Red", "RedEdge", "NIR"}, ignorados=set(),
                disparos=N_DISPAROS, completos=6)


def dji_mavic3m(d):   # _D.JPG + _MS_G/_MS_R/_MS_RE/_MS_NIR.TIF
    for i in range(N_DISPAROS):
        lat, lon, alt = _posicion(i); t = T0 + timedelta(seconds=2 * i, milliseconds=300)
        kw = dict(drone__dji__GpsLatitude=lat, drone__dji__GpsLongitude=lon, drone__dji__AbsoluteAltitude=alt,
                  drone__dji__RelativeAltitude="+60.0")
        _foto(d / f"DJI_2026_{i:04d}_D.JPG", 5280, 3956, 3, 8, _exif("DJI", "M3M", t, lat, lon, alt, 12.29, 24), _xmp(**kw))
        for suf, banda in (("G", "Green"), ("R", "Red"), ("RE", "RedEdge"), ("NIR", "NIR")):
            _foto(d / f"DJI_2026_{i:04d}_MS_{suf}.TIF", 2592, 1944, 1, 16, _exif("DJI", "M3M", t, lat, lon, alt, 4.34, 25),
                  _xmp(**kw, Camera__BandName=banda))
    return dict(principal=("rgb", "RGB"), bandas={"Green", "Red", "RedEdge", "NIR"}, ignorados=set(),
                disparos=N_DISPAROS, completos=5)


def micasense_rededge(d, altum=False, panchro=False):
    bandas = [("Blue", 475), ("Green", 560), ("Red", 668), ("NIR", 842), ("Red edge", 717)]
    if altum:
        bandas.append(("LWIR", 11000))
    if panchro:
        bandas.append(("Panchro", 634))
    for i in range(N_DISPAROS):
        lat, lon, alt = _posicion(i); t = T0 + timedelta(seconds=2 * i, milliseconds=500)
        for k, (banda, nm) in enumerate(bandas, 1):
            w, h = (160, 120) if banda == "LWIR" else ((2464, 2056) if banda == "Panchro" else (1280, 960))
            _foto(d / f"000SET/000/IMG_{i:04d}_{k}.tif", w, h, 1, 16,
                  _exif("MicaSense", "RedEdge-P" if panchro else ("Altum" if altum else "RedEdge-M"), t, lat, lon, alt, 5.4, None, 266.67),
                  _xmp(Camera__BandName=banda, Camera__CentralWavelength=nm, Camera__RigCameraIndex=k - 1,
                       Camera__BlackLevel="4800", Camera__Irradiance="1.2"))
    usadas = {"Blue", "Green", "Red", "NIR", "RedEdge"} | ({"Panchro"} if panchro else set())
    return dict(principal=("multiespectral", "Panchro" if panchro else "Green"), bandas=usadas,
                ignorados={"térmica"} if altum else set(), disparos=N_DISPAROS, completos=len(usadas))


def sequoia(d):   # _RGB.JPG con BandName en lista + _GRE/_RED/_REG/_NIR.TIF del mismo disparo
    for i in range(N_DISPAROS):
        lat, lon, alt = _posicion(i); t = T0 + timedelta(seconds=2 * i)
        cal = dict(Camera__PerspectiveFocalLength="4.8176", Camera__PrincipalPoint="3.0987,2.2942",
                   Camera__PerspectiveDistortion="0.1627,-0.4082,0.3225,-0.0008,0.0009")
        _foto(d / f"IMG_180419_{i:06d}_{i:04d}_RGB.JPG", 4608, 3456, 3, 8,
              _exif("Parrot", "Sequoia", t, lat, lon, alt, 4.88, 28, 746.27),
              _xmp(Camera__BandName=("Red", "Green", "Blue"), **cal))
        for suf, banda in (("GRE", "Green"), ("RED", "Red"), ("REG", "Red edge"), ("NIR", "NIR")):
            _foto(d / f"IMG_180419_{i:06d}_{i:04d}_{suf}.TIF", 1280, 960, 1, 16,
                  _exif("Parrot", "Sequoia", t + timedelta(milliseconds=40), lat, lon, alt, 3.98, None, 266.67),
                  _xmp(Camera__BandName=banda))
    return dict(principal=("rgb", "RGB"), bandas={"Green", "Red", "RedEdge", "NIR"}, ignorados=set(),
                disparos=N_DISPAROS, completos=5, calibracion=(4.8176 * 746.27, 3.0987 * 746.27))


def dji_m30t(d):   # gran angular _W, zoom _Z y térmica _T del mismo disparo
    for i in range(N_DISPAROS):
        lat, lon, alt = _posicion(i); t = T0 + timedelta(seconds=2 * i)
        kw = dict(drone__dji__GpsLatitude=lat, drone__dji__GpsLongitude=lon, drone__dji__AbsoluteAltitude=alt,
                  drone__dji__RelativeAltitude="+80.0")
        _foto(d / f"DJI_{i:04d}_W.JPG", 4000, 3000, 3, 8, _exif("DJI", "M30T", t, lat, lon, alt, 4.5, 24),
              _xmp(**kw, drone__dji__ImageSource="WideCamera"))
        _foto(d / f"DJI_{i:04d}_Z.JPG", 8000, 6000, 3, 8, _exif("DJI", "M30T", t, lat, lon, alt, 21.0, 113),
              _xmp(**kw, drone__dji__ImageSource="ZoomCamera"))
        _foto(d / f"DJI_{i:04d}_T.JPG", 640, 512, 3, 8, _exif("DJI", "M30T", t, lat, lon, alt, 9.1, 40),
              _xmp(**kw, drone__dji__ImageSource="InfraredCamera"))
    return dict(principal=("rgb", "RGB"), bandas=set(), ignorados={"térmica", "zoom"}, disparos=N_DISPAROS,
                principal_ancho=4000)


def sony_sin_xmp(d):   # sólo EXIF; sin focal equivalente a 35 mm (focal por tamaño de píxel)
    for i in range(N_DISPAROS):
        lat, lon, alt = _posicion(i); t = T0 + timedelta(seconds=3 * i)
        _foto(d / f"DSC{i:05d}.JPG", 7952, 5304, 3, 8, _exif("SONY", "ILCE-7RM4", t, lat, lon, alt, 35.0, None, 222.2, subseg=False), None)
    return dict(principal=("rgb", "RGB"), bandas=set(), ignorados=set(), disparos=N_DISPAROS, focal=35.0 * 222.2)


def sin_gps(d):
    for i in range(N_DISPAROS):
        t = T0 + timedelta(seconds=2 * i)
        _foto(d / f"IMG_{i:04d}.jpg", 4000, 3000, 3, 8, _exif("GoPro", "HERO", t, None, None, None, 3.0, 16), None)
    return dict(principal=("rgb", "RGB"), bandas=set(), ignorados=set(), disparos=N_DISPAROS, sin_gps=True)


def con_resultados_dentro(d):   # carpeta que contiene un procesamiento anterior: no debe leerlo
    r = dji_mavic3(d)
    _foto(d / "vuelo - Resultados" / "dem.tif", 500, 500, 1, 16, _exif("x", "x", T0, None, None, None), None)
    _foto(d / "vuelo - Resultados" / "_trabajo" / "proxies" / "sensor00" / "a.jpg", 300, 200, 3, 8, _exif("x", "x", T0, None, None, None), None)
    return r


def gps_incompleto(d):   # una foto con latitud sin longitud y otra con 0,0: no deben romper la lectura
    r = dji_mavic3(d)
    e = Image.Exif(); e[271], e[272] = "DJI", "L2D-20c"; e.get_ifd(0x8825)[2] = (20.0, 1.0, 1.0)
    _foto(d / "DJI_9998.JPG", 5280, 3956, 3, 8, e.tobytes(), None)
    _foto(d / "DJI_9999.JPG", 5280, 3956, 3, 8, _exif("DJI", "L2D-20c", T0 + timedelta(seconds=99), 0.0, 0.0, 0.0, 12.29, 24), None)
    r["extra_sin_gps"] = 2
    return r


PERFILES = {
    "DJI Mavic 3 (RGB)": dji_mavic3,
    "DJI Phantom 4 RTK (DewarpData)": dji_p4rtk,
    "DJI P4 Multispectral": dji_p4m,
    "DJI Mavic 3M": dji_mavic3m,
    "MicaSense RedEdge-MX": micasense_rededge,
    "MicaSense Altum (térmica)": lambda d: micasense_rededge(d, altum=True),
    "MicaSense RedEdge-P (pancromática)": lambda d: micasense_rededge(d, panchro=True),
    "Parrot Sequoia RGB + MS": sequoia,
    "DJI M30T (gran angular + zoom + térmica)": dji_m30t,
    "Sony sin XMP ni focal 35 mm": sony_sin_xmp,
    "Cámara sin GPS": sin_gps,
    "Carpeta con resultados anteriores": con_resultados_dentro,
    "GPS incompleto o 0,0": gps_incompleto,
}


# ---------------------------------------------------------------------------------- comprobación
def comprobar(nombre, crear, raiz: Path) -> list[str]:
    d = raiz / nombre.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
    esperado = crear(d)
    errores = []
    fotos = metadata.escanear_carpeta(d, log=lambda m: errores.append(f"lectura: {m}"))
    lev = metadata.analizar(fotos, log=lambda m: None)
    p0 = lev.sensores[lev.principal][0]
    if (p0.tipo, p0.banda) != esperado["principal"]:
        errores.append(f"principal {(p0.tipo, p0.banda)} != {esperado['principal']}")
    if "principal_ancho" in esperado and p0.ancho != esperado["principal_ancho"]:
        errores.append(f"principal de {p0.ancho} px de ancho, se esperaba {esperado['principal_ancho']}")
    bandas = {g[0].banda for g in lev.sensores.values() if g[0].tipo == "multiespectral"}
    if bandas != esperado["bandas"]:
        errores.append(f"bandas {sorted(bandas)} != {sorted(esperado['bandas'])}")
    motivos = " ".join(m for _, _, m in lev.ignoradas)
    for clave in esperado["ignorados"]:
        if clave not in motivos:
            errores.append(f"no se ignoró: {clave} (ignorados: {lev.ignoradas})")
    if not esperado["ignorados"] and lev.ignoradas:
        errores.append(f"ignorados inesperados: {lev.ignoradas}")
    con_gps = [f for f in lev.fotos if f.lat is not None]
    disparos = {f.clave_captura for f in con_gps} if con_gps else {f.clave_captura for f in lev.fotos}
    if len(disparos) != esperado["disparos"]:
        errores.append(f"{len(disparos)} disparos != {esperado['disparos']}")
    if "completos" in esperado:
        por = {}
        for f in lev.fotos:
            por.setdefault(f.clave_captura, set()).add(f.sensor)
        incompletos = [k for k, v in por.items() if len(v) not in (esperado["completos"], 1)]
        if incompletos:
            errores.append(f"disparos con bandas mezcladas: {incompletos[:3]}")
    # cámaras multiespectrales sincronizadas: se alinea sólo la principal y las bandas van como rig
    debe_rig = bool(esperado["bandas"])
    if len(lev.sensores) > 1 and rig.es_rig(lev.sensores, lev.principal) != debe_rig:
        errores.append(f"rig detectado = {not debe_rig}, se esperaba {debe_rig}")
    if "calibracion" in esperado:
        c = p0.calibracion
        if not c or abs(c[0] - esperado["calibracion"][0]) > 1 or abs(c[2] - esperado["calibracion"][1]) > 1:
            errores.append(f"calibración {c} != {esperado['calibracion']}")
    if "focal" in esperado and abs((metadata.focal_pixeles(p0) or 0) - esperado["focal"]) > 1:
        errores.append(f"focal {metadata.focal_pixeles(p0)} != {esperado['focal']}")
    if esperado.get("sin_gps") and any(f.lat is not None for f in lev.fotos):
        errores.append("se inventó GPS")
    if "extra_sin_gps" in esperado and sum(f.lat is None for f in lev.fotos) != esperado["extra_sin_gps"]:
        errores.append(f"fotos sin GPS: {sum(f.lat is None for f in lev.fotos)} != {esperado['extra_sin_gps']}")
    if any("_trabajo" in str(f.ruta) or "Resultados" in str(f.ruta) for f in lev.fotos):
        errores.append("leyó archivos de un procesamiento anterior")
    # lectores de imagen del flujo, con una foto de cada sensor usado
    for s, g in lev.sensores.items():
        f = g[0]
        try:
            alineacion._escribir_proxy(f, raiz / "_proxy" / (f.ruta.stem + ".jpg"), 1400, f.es_rgb)
            if f.es_rgb:
                img = metadata.leer_imagen_reducida(f.ruta, 2)
                assert img.ndim == 3 and img.shape[2] >= 3, f"RGB con forma {img.shape}"
            else:
                img = radiometria.corregir(metadata.leer_imagen(f.ruta), f)
                assert img.ndim == 3 and img.shape[2] == 1, f"banda con forma {img.shape}"
        except Exception as e:
            errores.append(f"lectura de {s}: {type(e).__name__}: {e}")
    return errores


def main() -> int:
    fallos = 0
    with tempfile.TemporaryDirectory() as tmp:
        for nombre, crear in PERFILES.items():
            try:
                errores = comprobar(nombre, crear, Path(tmp))
            except Exception:
                errores = [traceback.format_exc()]
            fallos += bool(errores)
            print(("OK    " if not errores else "FALLA ") + nombre)
            for e in errores:
                print("        " + e.replace("\n", "\n        "))
    print(f"\n{len(PERFILES) - fallos}/{len(PERFILES)} perfiles correctos")
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
