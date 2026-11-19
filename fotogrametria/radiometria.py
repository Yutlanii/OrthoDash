"""Corrección radiométrica para imágenes multiespectrales.

metadata.py decide qué modelo corresponde a cada foto (radiometria["modelo"]). Cada fabricante tiene
su fórmula y sus unidades:
  * DJI (P4 Multispectral, Mavic 3M): negro, viñeteo (el polinomio MULTIPLICA), ganancia y exposición,
    y sensor de sol con el ajuste de ganancia entre bandas. Da reflectancia relativa.
  * MicaSense (RedEdge, Altum): fórmula oficial con a1, a2, a3, viñeteo (el polinomio DIVIDE) y
    gradiente de filas: L = V R (p - negro) a1 / (2^bits ganancia t). Con el sensor DLS da reflectancia:
    pi L / E, con E en W/m2/nm.
  * Parrot Sequoia: I = f^2 (p - B) / (A t ISO + C) y viñeteo 2D. El Sunshine va en unidades propias de
    cada banda: sólo corrige cambios de luz durante el vuelo.
  * Otras cámaras: negro, exposición e ISO. No se aplican polinomios cuyo significado se desconoce.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from .metadata import Foto


def _floats(texto: str | None) -> list[float]:
    if not texto:
        return []
    out = []
    for p in texto.replace(";", ",").replace(" ", ",").split(","):
        try:
            out.append(float(p))
        except ValueError:
            pass
    return out


def _uno(r: dict, clave: str, defecto: float = 0.0) -> float:
    v = _floats(r.get(clave))
    return v[0] if v else defecto


@lru_cache(maxsize=16)
def _mapa_radial(h: int, w: int, cx: float, cy: float, coef: tuple) -> np.ndarray:
    """1 + k1 r + k2 r^2 + ... Igual para todas las fotos de una banda: se calcula una vez (antes se
    recalculaba por foto: 1236 veces en un vuelo de Mavic 3M)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    poly = np.ones_like(rr)
    potencia = np.ones_like(rr)
    for k in coef:
        potencia = potencia * rr
        poly = poly + k * potencia
    poly.flags.writeable = False
    return poly


@lru_cache(maxsize=16)
def _mapa_2d(h: int, w: int, coef: tuple, exponentes: tuple) -> np.ndarray:
    """Suma de c x^i y^j con x, y normalizados a 0-1 (Sequoia, algunas MicaSense)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    xx, yy = xx / w, yy / h
    poly = np.zeros_like(xx)
    for n, c in enumerate(coef):
        poly += c * xx ** exponentes[2 * n] * yy ** exponentes[2 * n + 1]
    salida = poly.astype(np.float32)
    salida.flags.writeable = False
    return salida


def _tiempo_s(r: dict) -> float:
    t = _uno(r, "ExposureTime")
    return t / 1e6 if t > 1.0 else t     # DJI guarda microsegundos en XMP


def _dji(x: np.ndarray, r: dict) -> np.ndarray:
    negro = _floats(r.get("BlackLevel"))
    if negro:
        x = x - float(np.mean(negro))
    h, w = x.shape[:2]
    coef = _floats(r.get("VignettingData")) or _floats(r.get("VignettingPolynomial"))
    if coef:
        cx = _floats(r.get("CalibratedOpticalCenterX")) or [w / 2]
        cy = _floats(r.get("CalibratedOpticalCenterY")) or [h / 2]
        cen = _floats(r.get("VignettingCenter"))
        if len(cen) >= 2:
            cx, cy = [cen[0]], [cen[1]]
        x = x * _mapa_radial(h, w, float(cx[0]), float(cy[0]), tuple(coef))[..., None]
    t = _tiempo_s(r)
    g = _uno(r, "SensorGain") or _uno(r, "ISO", 100.0) / 100.0
    if t > 0:
        x = x / (max(g, 1e-6) * t)
    irr = _uno(r, "Irradiance")
    if irr > 0:
        x = x * _uno(r, "SensorGainAdjustment", 1.0) / irr
    return x


def _irradiancia_dls(r: dict) -> float:
    """Irradiancia horizontal en W/m2/nm. Se recompone de directa y difusa porque en algunos firmwares
    la etiqueta horizontal salía mal calculada mientras las componentes eran correctas."""
    escala = _uno(r, "IrradianceScaleToSIUnits", 0.01 if "HorizontalIrradiance" in r else 1.0)
    directa, difusa = _uno(r, "DirectIrradiance"), _uno(r, "ScatteredIrradiance")
    if directa > 0 and "SolarElevation" in r:
        return (directa * math.sin(_uno(r, "SolarElevation")) + difusa) * escala
    return (_uno(r, "HorizontalIrradiance") or _uno(r, "SpectralIrradiance") or _uno(r, "Irradiance")) * escala


def _vineteo_2d(x: np.ndarray, r: dict) -> np.ndarray:
    coef, exp = _floats(r.get("VignettingPolynomial2D")), _floats(r.get("VignettingPolynomial2DName"))
    if coef and len(exp) == 2 * len(coef):
        h, w = x.shape[:2]
        return x / _mapa_2d(h, w, tuple(coef), tuple(exp))[..., None]
    return x


def _micasense(x: np.ndarray, r: dict, bits: int) -> np.ndarray:
    a1, a2, a3 = _floats(r["RadiometricCalibration"])[:3]
    t = _tiempo_s(r)
    h, w = x.shape[:2]
    x = x - _uno(r, "BlackLevel")
    centro, coef = _floats(r.get("VignettingCenter")), _floats(r.get("VignettingPolynomial"))
    if len(centro) >= 2 and coef:
        x = x / _mapa_radial(h, w, centro[0], centro[1], tuple(coef))[..., None]
    else:
        x = _vineteo_2d(x, r)
    filas = np.arange(h, dtype=np.float32).reshape(h, 1, 1)
    x = np.maximum(x / (1.0 + a2 * filas / t - a3 * filas), 0)
    x = x * a1 / (2.0 ** bits * max(_uno(r, "ISO", 100.0) / 100.0, 1e-6) * t)
    e = _irradiancia_dls(r)
    return x * math.pi / e if e > 0 else x


def _luz_relativa(r: dict) -> float:
    """Sensores de luz sin unidades comunes entre bandas: corrige sólo la variación durante el vuelo."""
    e, referencia = _uno(r, "Irradiance"), _uno(r, "IrradianciaReferencia")
    return referencia / e if e > 0 and referencia > 0 else 1.0


def _sequoia(x: np.ndarray, r: dict) -> np.ndarray:
    a, b, c = _floats(r["SensorModel"])[:3]
    f = _uno(r, "FNumber", 2.2)
    x = _vineteo_2d(f * f * (x - b) / (a * _tiempo_s(r) * _uno(r, "ISO", 100.0) + c), r)
    return x * _luz_relativa(r)


def _generico(x: np.ndarray, r: dict) -> np.ndarray:
    x = x - _uno(r, "BlackLevel")
    t = _tiempo_s(r)
    return x / (max(_uno(r, "ISO", 100.0) / 100.0, 1e-6) * t) * _luz_relativa(r) if t > 0 else x


def preparar(fotos: list[Foto]) -> None:
    """Guarda en las fotos de una banda la irradiancia mediana del vuelo, referencia de _luz_relativa."""
    valores = [v for f in fotos if (v := _uno(f.radiometria, "Irradiance")) > 0]
    if valores:
        mediana = str(float(np.median(valores)))
        for f in fotos:
            f.radiometria["IrradianciaReferencia"] = mediana


def corregir(img: np.ndarray, foto: Foto) -> np.ndarray:
    """img: HxWx1 (uint16/uint8). Devuelve float32 HxWx1."""
    r = foto.radiometria
    x = img.astype(np.float32)
    modelo = r.get("modelo")
    if modelo == "dji":
        x = _dji(x, r)
    elif modelo == "micasense":
        x = _micasense(x, r, foto.bits)
    elif modelo == "sequoia":
        x = _sequoia(x, r)
    else:
        x = _generico(x, r)
    return np.maximum(x, 0).astype(np.float32)


def limitacion(foto: Foto) -> str | None:
    """Qué no se puede garantizar con esta cámara; None si la escala entre bandas está calibrada."""
    r = foto.radiometria
    modelo = r.get("modelo")
    if modelo == "dji" and r.get("SensorGainAdjustment"):
        return None
    if modelo == "micasense":
        if _irradiancia_dls(r) > 0:
            return None
        return "esta MicaSense no trae datos del sensor de luz DLS: los valores son radiancia, no reflectancia"
    if modelo == "sequoia":
        return ("en Sequoia el sensor de sol no da una escala común entre bandas: los índices sirven para "
                "comparar zonas del vuelo, no como valores absolutos")
    return "no hay un modelo radiométrico calibrado para esta cámara: los índices son sólo orientativos"
