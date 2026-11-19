"""Paso 7: exportar nube de puntos (LAZ/PLY), DEM (GeoTIFF), vistas previas e informe."""
from __future__ import annotations

import html
import json
from pathlib import Path

import cv2
import laspy
import numpy as np
import rasterio
from rasterio.enums import Resampling

from .dem import Raster


def exportar_nube(ruta: Path, xyz: np.ndarray, rgb: np.ndarray, conf: np.ndarray, origen, crs) -> Path:
    ruta = Path(ruta)
    cab = laspy.LasHeader(point_format=7, version="1.4")
    g = xyz + origen
    cab.offsets = np.floor(g.min(0))
    cab.scales = np.array([0.001, 0.001, 0.001])
    cab.add_extra_dim(laspy.ExtraBytesParams(name="confianza", type=np.uint8, description="camaras que ven el punto"))
    try:
        cab.add_crs(crs)
    except Exception:
        pass
    las = laspy.LasData(cab)
    las.x, las.y, las.z = g[:, 0], g[:, 1], g[:, 2]
    rgb16 = rgb.astype(np.uint16) * 257
    las.red, las.green, las.blue = rgb16[:, 0], rgb16[:, 1], rgb16[:, 2]
    las.confianza = conf
    try:
        las.write(str(ruta), laz_backend=laspy.LazBackend.LazrsParallel)
    except Exception:
        ruta = ruta.with_suffix(".las")
        las.write(str(ruta))
    return ruta


MAX_PUNTOS_VISOR = 12_000_000


def nube_visor(carpeta: Path, xyz: np.ndarray, rgb: np.ndarray, origen, max_puntos: int = MAX_PUNTOS_VISOR) -> None:
    """Nube compacta para el visor 3D (WebGL): posiciones uint16 cuantizadas en la caja de la nube
    (precisión = lado / 65535, ~2 mm por cada 130 m) + RGB, 9 bytes por punto. Si hay más de
    `max_puntos` se toma una muestra uniforme: es sólo para revisar el resultado."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    if len(xyz) > max_puntos:
        # muestreo de Bernoulli: lineal (choice sin reemplazo permuta todo el arreglo)
        sel = np.random.default_rng(0).random(len(xyz)) < max_puntos / len(xyz)
        xyz, rgb = xyz[sel], rgb[sel]
    lo, hi = xyz.min(0), xyz.max(0)
    esc = np.maximum(hi - lo, 1e-6) / 65535.0
    q = np.round((xyz - lo) / esc).astype(np.uint16)
    with open(carpeta / "nube.bin", "wb") as f:
        f.write(q.tobytes())
        f.write(np.ascontiguousarray(rgb, np.uint8).tobytes())
    meta = {"puntos": int(len(xyz)), "min": lo.tolist(), "escala": esc.tolist(),
            "origen": [float(x) for x in origen]}
    (carpeta / "nube.json").write_text(json.dumps(meta), encoding="utf-8")


def nube_visor_desde_las(ruta_las: Path, carpeta: Path) -> None:
    """Para resultados de versiones anteriores sin nube de visor: la genera desde el LAZ."""
    las = laspy.read(str(ruta_las))
    xyz = np.stack([las.x, las.y, las.z], 1)
    origen = np.floor(xyz.min(0))
    rgb = (np.stack([las.red, las.green, las.blue], 1) // 257).astype(np.uint8)
    nube_visor(carpeta, xyz - origen, rgb, origen)


def exportar_ply(ruta: Path, xyz: np.ndarray, rgb: np.ndarray, origen) -> None:
    """PLY binario con coordenadas locales (+ origen en comentario), útil para CloudCompare/Meshlab."""
    n = len(xyz)
    cab = (
        "ply\nformat binary_little_endian 1.0\n"
        f"comment origen_utm {origen[0]:.3f} {origen[1]:.3f} {origen[2]:.3f}\n"
        f"element vertex {n}\nproperty float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    reg = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    reg["x"], reg["y"], reg["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    reg["r"], reg["g"], reg["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(ruta, "wb") as f:
        f.write(cab.encode("ascii"))
        f.write(reg.tobytes())


def exportar_dem(ruta: Path, dem: Raster, origen, crs) -> None:
    datos = np.where(np.isfinite(dem.datos), dem.datos + origen[2], -32767).astype(np.float32)
    perfil = dict(driver="GTiff", width=datos.shape[1], height=datos.shape[0], count=1, dtype="float32",
                  crs=crs, transform=dem.transform_affine(origen), nodata=-32767, tiled=True,
                  blockxsize=512, blockysize=512, compress="deflate", predictor=3, BIGTIFF="IF_SAFER")
    with rasterio.open(ruta, "w", **perfil) as dst:
        dst.write(datos, 1)
        dst.set_band_description(1, "Elevacion (m)")
        dst.build_overviews([2, 4, 8, 16], Resampling.average)


def _guardar_png(ruta: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(Path(ruta).suffix, img)
    buf.tofile(str(ruta))


def vista_previa_dem(ruta: Path, dem: Raster, lado: int = 1600) -> None:
    z = dem.datos
    valido = np.isfinite(z)
    if not valido.any():
        return
    esc = min(1.0, lado / max(z.shape))
    zz = np.where(valido, z, np.nanmedian(z))
    if esc < 1:
        zz = cv2.resize(zz, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA)
        valido = cv2.resize(valido.astype(np.uint8), (zz.shape[1], zz.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    lo, hi = np.percentile(zz[valido], (1, 99))
    norm = np.clip((zz - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    # sombreado
    r = dem.res / esc
    gy, gx = np.gradient(zz, r)
    pend = np.pi / 2 - np.arctan(np.hypot(gx, gy))
    asp = np.arctan2(-gx, gy)
    az, alt = np.radians(315), np.radians(45)
    som = np.sin(alt) * np.sin(pend) + np.cos(alt) * np.cos(pend) * np.cos(az - asp)
    som = np.clip(som, 0, 1)[..., None]
    out = (color * (0.45 + 0.55 * som)).astype(np.uint8)
    out[~valido] = 255
    _guardar_png(ruta, out)


def vista_previa_orto(ruta_tif: Path, ruta_jpg: Path, lado: int = 2000) -> None:
    with rasterio.open(ruta_tif) as src:
        esc = min(1.0, lado / max(src.width, src.height))
        w, h = max(1, int(src.width * esc)), max(1, int(src.height * esc))
        datos = src.read(out_shape=(src.count, h, w), resampling=Resampling.average)
    if datos.dtype == np.uint8:
        rgb = np.moveaxis(datos[:3], 0, 2)
        alfa = datos[3] if datos.shape[0] > 3 else np.full((h, w), 255, np.uint8)
        img = np.where(alfa[..., None] > 0, rgb, 255).astype(np.uint8)
        _guardar_png(ruta_jpg, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        b = datos[0]
        v = np.isfinite(b)
        lo, hi = np.percentile(b[v], (1, 99)) if v.any() else (0, 1)
        g = np.clip((np.nan_to_num(b) - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)
        g[~v] = 255
        _guardar_png(ruta_jpg, g)


def compresion(cal: dict) -> dict:
    """Opciones de GeoTIFF: en modo express, deflate rápido (archivo algo mayor, mucho menos tiempo)."""
    return {"zlevel": 1} if cal.get("express") else {}


def informe_html(ruta: Path, datos: dict, imagenes: dict[str, str]) -> None:
    filas = "".join(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>" for k, v in datos["resumen"].items())
    tiempos = "".join(f"<tr><th>{html.escape(k)}</th><td>{v:.1f} s</td></tr>" for k, v in datos["tiempos"].items())
    figs = "".join(f"<figure><img src='{html.escape(src)}'><figcaption>{html.escape(t)}</figcaption></figure>" for t, src in imagenes.items())
    doc = f"""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8"><title>Informe de procesamiento</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222}}
table{{border-collapse:collapse;margin:12px 0}}th,td{{border:1px solid #ddd;padding:6px 10px;text-align:left}}
th{{background:#f4f6f8}}figure{{margin:16px 0}}img{{max-width:100%;border:1px solid #ccc}}h1{{font-size:22px}}</style></head>
<body><h1>Informe de fotogrametría</h1><h2>Resumen</h2><table>{filas}</table>
<h2>Tiempos</h2><table>{tiempos}</table><h2>Resultados</h2>{figs}
<details><summary>Datos completos (JSON)</summary><pre>{html.escape(json.dumps(datos, indent=2, ensure_ascii=False, default=str))}</pre></details>
</body></html>"""
    Path(ruta).write_text(doc, encoding="utf-8")
