"""Servidor local (sólo 127.0.0.1) del visor: interfaz web, teselas de mapa y nube de puntos.

Las teselas se cortan al vuelo de los GeoTIFF: GDAL elige la vista previa interna (overview)
adecuada a cada zoom, así una tesela de 256 px se lee en milisegundos aunque el ortomosaico tenga
cientos de megapíxeles. Las teselas recientes quedan en memoria.
"""
from __future__ import annotations

import json
import math
import secrets
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

WEB = Path(__file__).resolve().parent / "web"
TESELA = 256
TIPOS = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".js": "text/javascript; charset=utf-8", ".woff2": "font/woff2", ".svg": "image/svg+xml",
         ".json": "application/json", ".png": "image/png", ".webp": "image/webp"}

# Paletas sobrias (sin neón): relieve terroso y NDVI rojo-ocre-verde
_PAL_DEM = [(0.0, (38, 70, 83)), (0.25, (42, 116, 110)), (0.5, (138, 160, 110)),
            (0.72, (214, 190, 140)), (0.88, (168, 120, 84)), (1.0, (240, 236, 228))]
_PAL_NDVI = [(0.0, (150, 60, 45)), (0.35, (205, 150, 80)), (0.55, (225, 210, 140)),
             (0.75, (120, 165, 80)), (1.0, (35, 105, 55))]


def _lut(paleta) -> np.ndarray:
    t = np.linspace(0, 1, 256)
    pos = [p for p, _ in paleta]
    return np.stack([np.interp(t, pos, [c[k] for _, c in paleta]) for k in range(3)], 1).astype(np.uint8)


LUT_DEM, LUT_NDVI = _lut(_PAL_DEM), _lut(_PAL_NDVI)
RANGO_INDICE = {"NDVI": (0.0, 0.9), "GNDVI": (0.0, 0.9), "NDRE": (0.0, 0.6), "LCI": (0.0, 0.7)}   # fijos: comparables entre vuelos


class Capa:
    """Un GeoTIFF servido como teselas en la rejilla común del visor."""

    def __init__(self, nombre: str, ruta: Path, banda: int = 1, tipo: str | None = None):
        self.nombre, self.ruta, self.banda = nombre, ruta, banda
        self.tipo = tipo or nombre
        self._local = threading.local()
        with rasterio.open(ruta) as src:
            self.ancho, self.alto, self.bandas = src.width, src.height, src.count
            self.t = src.transform
            self.crs = src.crs
            self.nodata = src.nodata
            self.res = abs(self.t.a)
            b = src.bounds
            self.limites = (b.left, b.bottom, b.right, b.top)
            self.rango = None
            self.rangos: list[tuple[float, float]] = []
            self.bandas_color = [1]
            if self.tipo == "multi":
                # falso color de cultivos: NIR, Red, Green -> R, G, B (salida BGR: Green, Red, NIR)
                pos = {n: i + 1 for i, n in enumerate(src.descriptions) if n}
                self.bandas_color = [pos[n] for n in ("Green", "Red", "NIR")] if all(n in pos for n in ("Green", "Red", "NIR")) \
                    else list(range(1, min(3, src.count) + 1))
            if self.tipo in ("dem", "multi"):
                # rango de color global (con una vista reducida: rápido); igual en todas las teselas
                fac = max(1, int(max(src.width, src.height) / 1024))
                forma = (max(1, src.height // fac), max(1, src.width // fac))
                for b in self.bandas_color:
                    d = src.read(b, out_shape=forma, resampling=Resampling.nearest, masked=True)
                    d = d.astype(np.float32).filled(np.nan)
                    v = d[np.isfinite(d)]
                    self.rangos.append(tuple(float(x) for x in np.percentile(v, (1, 99))) if v.size else (0.0, 1.0))
                self.rango = self.rangos[0]
            elif self.tipo == "indice":
                self.rango = RANGO_INDICE.get(nombre, (0.0, 0.9))

    def _src(self):
        src = getattr(self._local, "src", None)
        if src is None:
            src = self._local.src = rasterio.open(self.ruta)
        return src

    def leer(self, e0, n1, tam_m, n_px, bandas, pad=0, remuestreo=Resampling.bilinear):
        """Lee la caja [e0, e0+tam_m] x [n1-tam_m, n1] a n_px (+pad) píxeles. Devuelve (datos, máscara) o None."""
        src = self._src()
        px = tam_m / n_px
        e0p, n1p = e0 - pad * px, n1 + pad * px
        n_tot = n_px + 2 * pad
        c0 = (e0p - self.t.c) / self.t.a
        r0 = (n1p - self.t.f) / self.t.e
        esc = (n_tot * px) / self.res          # píxeles del raster por tesela
        c1, r1 = c0 + esc, r0 + esc
        cc0, rr0 = max(0.0, c0), max(0.0, r0)
        cc1, rr1 = min(float(self.ancho), c1), min(float(self.alto), r1)
        if cc1 <= cc0 or rr1 <= rr0:
            return None
        f = n_tot / esc                         # píxeles de salida por píxel del raster
        x0, y0 = int(round((cc0 - c0) * f)), int(round((rr0 - r0) * f))
        x1, y1 = int(round((cc1 - c0) * f)), int(round((rr1 - r0) * f))
        if x1 <= x0 or y1 <= y0:
            return None
        win = Window(cc0, rr0, cc1 - cc0, rr1 - rr0)
        datos = src.read(bandas, window=win, out_shape=(len(bandas), y1 - y0, x1 - x0), resampling=remuestreo)
        lienzo = np.zeros((len(bandas), n_tot, n_tot), datos.dtype)
        if np.issubdtype(datos.dtype, np.floating):
            lienzo[:] = np.nan
        lienzo[:, y0:y1, x0:x1] = datos
        return lienzo


class Visor:
    """Estado del visor: carpeta de resultados abierta, capas y rejilla de teselas."""

    def __init__(self):
        self.lock = threading.Lock()
        self.capas: dict[str, Capa] = {}
        self.carpeta: Path | None = None
        self.rejilla = None
        self.cache: OrderedDict[tuple, bytes] = OrderedDict()
        self.version = 0

    def abrir(self, carpeta: Path) -> dict:
        carpeta = Path(carpeta)
        capas = {}
        for nombre, archivo in (("orto", "ortomosaico.tif"), ("multi", "ortomosaico_multiespectral.tif"), ("dem", "dem.tif")):
            if (carpeta / archivo).exists():
                capas[nombre] = Capa(nombre, carpeta / archivo)
        if (carpeta / "indices_vegetacion.tif").exists():
            with rasterio.open(carpeta / "indices_vegetacion.tif") as src:
                nombres = list(src.descriptions)
            for i, n in enumerate(nombres, 1):
                capas[n] = Capa(n, carpeta / "indices_vegetacion.tif", banda=i, tipo="indice")
        elif (carpeta / "ndvi.tif").exists():     # resultados de versiones anteriores
            capas["NDVI"] = Capa("NDVI", carpeta / "ndvi.tif", tipo="indice")
        if not capas:
            raise FileNotFoundError("La carpeta no contiene resultados (ortomosaico.tif / dem.tif).")
        # rejilla común: zoom 0 = toda la zona en una tesela
        l = [min(c.limites[0] for c in capas.values()), min(c.limites[1] for c in capas.values()),
             max(c.limites[2] for c in capas.values()), max(c.limites[3] for c in capas.values())]
        lado = max(l[2] - l[0], l[3] - l[1])
        res0 = lado / TESELA
        res_min = min(c.res for c in capas.values())
        zmax = max(1, math.ceil(math.log2(res0 / res_min)) + 1)
        with self.lock:
            self.capas, self.carpeta = capas, carpeta
            self.rejilla = {"e0": l[0], "n1": l[3], "res0": res0, "zmax": zmax}
            self.cache.clear()
            self.version += 1
        crs = next(iter(capas.values())).crs
        info = {
            "carpeta": str(carpeta), "version": self.version, "rejilla": self.rejilla, "limites": l,
            "epsg": crs.to_epsg() if crs else None,
            "capas": {n: {"tipo": c.tipo, "res": c.res, "ancho": c.ancho, "alto": c.alto, "rango": c.rango, "limites": c.limites}
                      for n, c in capas.items()},
            "nube": any((carpeta / n).exists() for n in ("nube_densa.laz", "nube_densa.las", "nube_dispersa.laz")),
            "informe": (carpeta / "informe.html").exists(),
            "archivos": [{"nombre": f.name, "mb": round(f.stat().st_size / 1e6, 1)} for f in sorted(carpeta.iterdir())
                         if f.is_file() and f.suffix.lower() in (".tif", ".laz", ".las", ".ply", ".csv", ".html")],
        }
        if (carpeta / "informe.json").exists():
            try:
                datos = json.loads((carpeta / "informe.json").read_text(encoding="utf-8"))
                info["resumen"], info["indices"] = datos.get("resumen", {}), datos.get("indices", {})
            except (OSError, ValueError):
                pass
        return info

    # ------------------------------------------------------------------------------------------
    def tesela(self, capa: str, z: int, x: int, y: int) -> bytes | None:
        clave = (self.version, capa, z, x, y)
        with self.lock:
            if clave in self.cache:
                self.cache.move_to_end(clave)
                return self.cache[clave]
            c, g = self.capas.get(capa), self.rejilla
        if c is None or g is None:
            return None
        tam = g["res0"] * TESELA / 2 ** z
        e0, n1 = g["e0"] + x * tam, g["n1"] - y * tam
        img = self._dibujar(c, e0, n1, tam, tam / TESELA)
        datos = None
        if img is not None:
            ok, buf = cv2.imencode(".webp", img, [cv2.IMWRITE_WEBP_QUALITY, 88])
            datos = buf.tobytes() if ok else None
        with self.lock:
            self.cache[clave] = datos
            while len(self.cache) > 3000:
                self.cache.popitem(last=False)
        return datos

    def _dibujar(self, c: Capa, e0, n1, tam, px):
        if c.tipo == "orto":
            d = c.leer(e0, n1, tam, TESELA, [1, 2, 3, 4] if c.bandas >= 4 else list(range(1, c.bandas + 1)))
            if d is None:
                return None
            rgb = np.moveaxis(d[:3], 0, 2)
            alfa = d[3] if c.bandas >= 4 else np.full(rgb.shape[:2], 255, np.uint8)
            if not alfa.any():
                return None
            return np.dstack([rgb[..., ::-1], alfa])
        if c.tipo == "multi":
            d = c.leer(e0, n1, tam, TESELA, c.bandas_color)
            if d is None:
                return None
            d = d.astype(np.float32)
            v = np.isfinite(d[0])
            if not v.any():
                return None
            out = np.zeros((TESELA, TESELA, 4), np.uint8)
            for k in range(d.shape[0]):     # salida BGR
                lo, hi = c.rangos[k]
                out[..., k] = np.clip((np.nan_to_num(d[k]) - lo) / max(hi - lo, 1e-9) * 255, 0, 255)
            out[..., 3] = v * 255
            return out
        # dem / índice: color + sombreado (el DEM se lee con 1 px de margen para las pendientes)
        pad = 1 if c.tipo == "dem" else 0
        d = c.leer(e0, n1, tam, TESELA, [c.banda], pad=pad)
        if d is None:
            return None
        z = d[0].astype(np.float32)
        if c.nodata is not None and np.isfinite(c.nodata):
            z[z == c.nodata] = np.nan
        valido = np.isfinite(z)
        if not valido[pad:pad + TESELA, pad:pad + TESELA].any():
            return None
        lo, hi = c.rango or (np.nanmin(z), np.nanmax(z))
        t = np.clip((np.nan_to_num(z, nan=lo) - lo) / max(hi - lo, 1e-9), 0, 1)
        rgb = (LUT_DEM if c.tipo == "dem" else LUT_NDVI)[(t * 255).astype(np.uint8)].astype(np.float32)
        if c.tipo == "dem":
            zz = np.where(valido, z, np.nanmedian(z))
            gy, gx = np.gradient(zz, px)
            pend = np.pi / 2 - np.arctan(np.hypot(gx, gy))
            asp = np.arctan2(-gx, gy)
            az, alt = np.radians(315), np.radians(40)
            som = np.clip(np.sin(alt) * np.sin(pend) + np.cos(alt) * np.cos(pend) * np.cos(az - asp), 0, 1)
            rgb *= (0.35 + 0.65 * som)[..., None]
            rgb, valido = rgb[1:-1, 1:-1], valido[1:-1, 1:-1]
        out = np.dstack([rgb[..., ::-1].clip(0, 255).astype(np.uint8), (valido * 255).astype(np.uint8)])
        return out

    def valor(self, e: float, n: float) -> dict:
        """Valores de las capas en una coordenada (para la lectura bajo el cursor)."""
        out = {}
        for nombre, c in self.capas.items():
            if c.tipo not in ("dem", "indice"):
                continue
            col, fil = ~c.t * (e, n)
            if 0 <= col < c.ancho and 0 <= fil < c.alto:
                v = float(c._src().read(c.banda, window=Window(int(col), int(fil), 1, 1))[0, 0])
                if np.isfinite(v) and v != c.nodata:
                    out[nombre] = v
        return out

    # ------------------------------------------------------------------------------------------
    def nube(self) -> tuple[bytes, dict] | None:
        """Nube para el visor 3D: posiciones uint16 cuantizadas + color RGB (9 bytes/punto)."""
        if self.carpeta is None:
            return None
        d = self.carpeta / "_trabajo" / "visor"
        bin_, meta = d / "nube.bin", d / "nube.json"
        if not bin_.exists():
            from ..exportar import nube_visor_desde_las
            las = next((p for p in (self.carpeta / "nube_densa.laz", self.carpeta / "nube_densa.las",
                                     self.carpeta / "nube_dispersa.laz") if p.exists()), None)
            if las is None:
                return None
            nube_visor_desde_las(las, d)
        return bin_.read_bytes(), json.loads(meta.read_text(encoding="utf-8"))


def _manejador(visor: Visor, token: str):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _enviar(self, codigo, cuerpo=b"", tipo="application/octet-stream", cache=True):
            self.send_response(codigo)
            self.send_header("Content-Type", tipo)
            self.send_header("Content-Length", str(len(cuerpo)))
            self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
            self.end_headers()
            if cuerpo:
                self.wfile.write(cuerpo)

        def do_GET(self):
            u = urlparse(self.path)
            partes = [p for p in u.path.split("/") if p]
            # ruta secreta: ninguna otra página del equipo puede leer los resultados
            if not partes or partes[0] != token:
                return self._enviar(404)
            partes = partes[1:]
            try:
                if not partes:
                    partes = ["index.html"]
                if partes[0] == "t" and len(partes) == 6:          # /t/<version>/<capa>/<z>/<x>/<y>
                    capa, z, x, y = partes[2], int(partes[3]), int(partes[4]), int(partes[5].split(".")[0])
                    datos = visor.tesela(capa, z, x, y)
                    return self._enviar(200, datos, "image/webp") if datos else self._enviar(204)
                if partes[0] == "valor":
                    q = parse_qs(u.query)
                    v = visor.valor(float(q["e"][0]), float(q["n"][0]))
                    return self._enviar(200, json.dumps(v).encode(), "application/json", cache=False)
                if partes[0] == "nube.json" or partes[0] == "nube.bin":
                    r = visor.nube()
                    if r is None:
                        return self._enviar(404)
                    if partes[0] == "nube.json":
                        return self._enviar(200, json.dumps(r[1]).encode(), "application/json", cache=False)
                    return self._enviar(200, r[0], "application/octet-stream", cache=False)
                if partes[0] == "informe" and visor.carpeta is not None:
                    ruta = (visor.carpeta / "/".join(partes[1:] or ["informe.html"])).resolve()
                    if visor.carpeta.resolve() in ruta.parents and ruta.is_file():
                        return self._enviar(200, ruta.read_bytes(), TIPOS.get(ruta.suffix, "application/octet-stream"), cache=False)
                    return self._enviar(404)
                ruta = (WEB / "/".join(partes)).resolve()
                if WEB.resolve() in ruta.parents and ruta.is_file():
                    return self._enviar(200, ruta.read_bytes(), TIPOS.get(ruta.suffix, "application/octet-stream"), cache=False)
                return self._enviar(404)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            except Exception as e:  # el visor nunca debe tumbar la aplicación
                try:
                    self._enviar(500, str(e).encode(), "text/plain")
                except OSError:
                    pass

    return H


def iniciar(visor: Visor) -> tuple[ThreadingHTTPServer, str]:
    token = secrets.token_urlsafe(16)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _manejador(visor, token))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/{token}/"
