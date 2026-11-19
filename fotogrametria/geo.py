"""Georreferenciación: WGS84 -> UTM (métrico) con un origen local para precisión numérica."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyproj import CRS, Transformer


@dataclass
class MarcoLocal:
    epsg: int
    origen: np.ndarray  # (E0, N0, Z0) en metros UTM

    @property
    def crs(self) -> CRS:
        return CRS.from_epsg(self.epsg)

    def a_local(self, lat, lon, alt) -> np.ndarray:
        t = Transformer.from_crs(4326, self.epsg, always_xy=True)
        e, n = t.transform(np.asarray(lon, float), np.asarray(lat, float))
        return np.stack([e, n, np.asarray(alt, float)], axis=-1) - self.origen

    def a_wgs84(self, xyz_local: np.ndarray) -> np.ndarray:
        t = Transformer.from_crs(self.epsg, 4326, always_xy=True)
        g = np.asarray(xyz_local, float) + self.origen
        lon, lat = t.transform(g[..., 0], g[..., 1])
        return np.stack([lat, lon, g[..., 2]], axis=-1)


def epsg_utm(lat: float, lon: float) -> int:
    zona = int((lon + 180.0) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zona


def crear_marco(lats, lons, alts) -> MarcoLocal:
    lats, lons, alts = (np.asarray(v, float) for v in (lats, lons, alts))
    epsg = epsg_utm(float(np.median(lats)), float(np.median(lons)))
    t = Transformer.from_crs(4326, epsg, always_xy=True)
    e, n = t.transform(lons, lats)
    origen = np.array([np.round(np.median(e), -2), np.round(np.median(n), -2), 0.0])
    return MarcoLocal(epsg=epsg, origen=origen)
