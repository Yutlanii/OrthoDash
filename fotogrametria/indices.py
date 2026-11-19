"""Índices de vegetación e imagen con todas las bandas en 8 bits (agricultura de precisión).

Los índices se calculan siempre sobre la reflectancia relativa en float32, nunca sobre los 8 bits.
Sólo se incluyen índices de cociente, que no cambian si todas las bandas se multiplican por la misma
constante y por tanto son válidos con reflectancia relativa. SAVI, OSAVI o EVI llevan términos
absolutos (L = 0,5 o 0,16) y sin panel de calibración darían valores engañosos.

La imagen de 8 bits usa una sola escala lineal para todas las bandas multiespectrales: estirar cada
banda por su cuenta se vería mejor "en crudo" pero rompería la proporción NIR/rojo y cualquier índice
calculado después en un SIG.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

FILAS = 1024   # filas por tramo al recorrer los ortomosaicos


def _nd(a, b):
    return (a - b) / (a + b)


# nombre -> (bandas necesarias, fórmula, descripción)
INDICES = {
    "NDVI": (("NIR", "Red"), lambda b: _nd(b["NIR"], b["Red"]), "Vigor general de la vegetación"),
    "GNDVI": (("NIR", "Green"), lambda b: _nd(b["NIR"], b["Green"]), "Clorofila; se satura menos que el NDVI"),
    "NDRE": (("NIR", "RedEdge"), lambda b: _nd(b["NIR"], b["RedEdge"]), "Nitrógeno y clorofila en cultivos densos"),
    "LCI": (("NIR", "RedEdge", "Red"), lambda b: (b["NIR"] - b["RedEdge"]) / (b["NIR"] + b["Red"]), "Clorofila foliar"),
}
_PALETA = [(0.0, (165, 0, 38)), (0.5, (255, 255, 191)), (1.0, (0, 104, 55))]    # rojo -> amarillo -> verde


def _tramos(alto: int, ancho: int):
    for r in range(0, alto, FILAS):
        yield Window(0, r, ancho, min(FILAS, alto - r))


def calcular_indices(ruta_ms: Path, ruta_salida: Path, ruta_png: Path, **opciones) -> list[str]:
    """Escribe un GeoTIFF float32 con una banda por índice posible y la vista previa del NDVI."""
    with rasterio.open(ruta_ms) as ms:
        posicion = {n: i + 1 for i, n in enumerate(ms.descriptions) if n}
        nombres = [n for n, (req, _, _) in INDICES.items() if all(b in posicion for b in req)]
        if not nombres:
            return []
        usadas = sorted({b for n in nombres for b in INDICES[n][0]})
        perfil = ms.profile.copy()
        perfil.update(count=len(nombres), dtype="float32", nodata=np.nan, predictor=3, num_threads="all_cpus", **opciones)
        with rasterio.Env(GDAL_NUM_THREADS="ALL_CPUS"), rasterio.open(ruta_salida, "w", **perfil) as dst:
            for i, n in enumerate(nombres, 1):
                dst.set_band_description(i, n)
                dst.update_tags(i, descripcion=INDICES[n][2])
            for win in _tramos(ms.height, ms.width):
                b = {u: ms.read(posicion[u], window=win).astype(np.float32) for u in usadas}
                with np.errstate(all="ignore"):
                    for i, n in enumerate(nombres, 1):
                        v = INDICES[n][1](b)
                        dst.write(np.where(np.isfinite(v), np.clip(v, -1, 1), np.nan).astype(np.float32), i, window=win)
            dst.build_overviews([2, 4, 8, 16], Resampling.nearest)
    if "NDVI" in nombres:
        _previa(ruta_salida, nombres.index("NDVI") + 1, ruta_png)
    return nombres


def _previa(ruta: Path, banda: int, ruta_png: Path, lado: int = 2000) -> None:
    with rasterio.open(ruta) as src:
        e = min(1.0, lado / max(src.width, src.height))
        v = src.read(banda, out_shape=(max(1, int(src.height * e)), max(1, int(src.width * e))),
                     resampling=Resampling.nearest)
    t = np.linspace(0, 1, 256)
    lut = np.stack([np.interp(t, [p for p, _ in _PALETA], [c[k] for _, c in _PALETA]) for k in (2, 1, 0)], 1).astype(np.uint8)
    color = lut[(np.clip((np.nan_to_num(v, nan=-0.2) + 0.2) / 1.2, 0, 1) * 255).astype(np.uint8)]
    color[~np.isfinite(v)] = 255
    ok, buf = cv2.imencode(".png", color)
    buf.tofile(str(ruta_png))


def _escala_comun(ms) -> float:
    """Factor lineal que lleva el percentil 99,9 de la banda más brillante a 250."""
    e = max(1, max(ms.width, ms.height) // 1024)
    d = ms.read(out_shape=(ms.count, max(1, ms.height // e), max(1, ms.width // e)), resampling=Resampling.nearest)
    v = d[np.isfinite(d)]
    return 250.0 / max(float(np.percentile(v, 99.9)), 1e-12) if v.size else 1.0


def apilar_bandas(ruta_rgb: Path | None, ruta_ms: Path, ruta_salida: Path, **opciones) -> float:
    """Una sola imagen uint8 con todas las bandas, en la rejilla del multiespectral (0 = sin dato).

    R, G y B conservan sus valores; las bandas multiespectrales usan una escala lineal común, guardada
    en la etiqueta ESCALA_REFLECTANCIA (reflectancia relativa = valor / escala). Devuelve la escala."""
    with rasterio.open(ruta_ms) as ms:
        k = _escala_comun(ms)
        n_rgb = 3 if ruta_rgb else 0
        perfil = ms.profile.copy()
        perfil.update(count=n_rgb + ms.count, dtype="uint8", nodata=0, predictor=2, num_threads="all_cpus",
                      BIGTIFF="IF_SAFER", **opciones)
        nombres = (["RGB_Rojo", "RGB_Verde", "RGB_Azul"] if ruta_rgb else []) + \
                  [ms.descriptions[i] or f"Banda{i + 1}" for i in range(ms.count)]
        rgb = rasterio.open(ruta_rgb) if ruta_rgb else None
        vrt = WarpedVRT(rgb, crs=ms.crs, transform=ms.transform, width=ms.width, height=ms.height,
                        resampling=Resampling.average) if rgb else None
        try:
            with rasterio.Env(GDAL_NUM_THREADS="ALL_CPUS"), rasterio.open(ruta_salida, "w", **perfil) as dst:
                dst.update_tags(ESCALA_REFLECTANCIA=f"{k:.8g}", FORMULA="reflectancia_relativa = valor / ESCALA_REFLECTANCIA")
                for i, n in enumerate(nombres, 1):
                    dst.set_band_description(i, n)
                for win in _tramos(ms.height, ms.width):
                    d = ms.read(window=win)
                    valido = np.all(np.isfinite(d), axis=0)
                    m = np.clip(np.nan_to_num(d, nan=0.0) * k + 0.5, 1, 255).astype(np.uint8)
                    m[:, ~valido] = 0
                    if vrt is not None:
                        c = vrt.read([1, 2, 3, 4], window=win)
                        visible = np.clip(c[:3], 1, 255)
                        visible[:, (c[3] == 0) | ~valido] = 0
                        m = np.concatenate([visible, m])
                    dst.write(m, window=win)
                dst.build_overviews([2, 4, 8, 16], Resampling.average)
        finally:
            if vrt is not None:
                vrt.close(); rgb.close()
    return k


def informe_valores(ruta_indices: Path) -> dict[str, dict[str, float]]:
    """Mediana y percentiles 5 y 95 de cada índice, para el informe y para detectar valores absurdos."""
    salida = {}
    with rasterio.open(ruta_indices) as src:
        e = max(1, max(src.width, src.height) // 1024)
        for i in range(1, src.count + 1):
            v = src.read(i, out_shape=(max(1, src.height // e), max(1, src.width // e)), resampling=Resampling.nearest)
            v = v[np.isfinite(v)]
            if v.size:
                p = np.percentile(v, [5, 50, 95])
                salida[src.descriptions[i - 1]] = {"p5": round(float(p[0]), 3), "mediana": round(float(p[1]), 3), "p95": round(float(p[2]), 3)}
    return salida
