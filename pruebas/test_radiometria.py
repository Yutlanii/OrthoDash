"""Radiometría por fabricante, con valores calculados a mano sobre metadatos de cada cámara.

Ejecutar:  .venv\\Scripts\\python.exe -m pruebas.test_radiometria
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fotogrametria import metadata, radiometria  # noqa: E402
from fotogrametria.metadata import Foto  # noqa: E402


def _foto(fabricante, r, bits=16):
    r = dict(r, modelo=metadata._modelo_radiometrico(fabricante, r))
    return Foto(ruta=Path("x.tif"), nombre="x.tif", ancho=4, alto=4, canales=1, bits=bits, fabricante=fabricante,
                tipo="multiespectral", banda="Red", radiometria=r)


def _cerca(a, b, tol=1e-5):
    assert abs(a - b) <= tol * max(abs(b), 1e-12), f"{a} != {b}"


def lista_xmp():
    xmp = "<C:V>\n <rdf:Seq>\n  <rdf:li>1.5</rdf:li>\n  <rdf:li>-2e-07</rdf:li>\n </rdf:Seq>\n</C:V>"
    assert metadata._xmp_lista(xmp, "V") == "1.5,-2e-07"
    assert metadata._xmp_lista('C:V="1,2,3"', "V") == "1,2,3"
    assert metadata._texto_exif((4800, 4800)) == "4800.0,4800.0"


def micasense_sin_viñeteo():
    f = _foto("MicaSense", {"RadiometricCalibration": "0.0004,0,0", "BlackLevel": "4800,4800", "ExposureTime": "0.001",
                            "ISO": "200", "DLS": "", "HorizontalIrradiance": "100", "DirectIrradiance": "80",
                            "ScatteredIrradiance": "20", "SolarElevation": str(math.pi / 2)})
    x = radiometria.corregir(np.full((4, 4, 1), 14800, np.uint16), f)
    radiancia = 10000 * 0.0004 / (2 ** 16 * 2 * 0.001)
    _cerca(float(x[0, 0, 0]), radiancia * math.pi / 1.0)     # (80 + 20) * 0.01 W/m2/nm
    assert radiometria.limitacion(f) is None


def micasense_viñeteo_divide_y_gradiente():
    r = {"RadiometricCalibration": "1,0.0001,0", "ExposureTime": "0.01", "ISO": "100",
         "VignettingCenter": "0,0", "VignettingPolynomial": "0.1"}
    x = radiometria.corregir(np.full((4, 4, 1), 65536 // 2, np.int32), _foto("MicaSense", r))
    _cerca(float(x[0, 0, 0]), 50.0)
    _cerca(float(x[0, 3, 0]), 50.0 / (1 + 0.1 * 3), 1e-4)     # radial: se divide, no se multiplica
    _cerca(float(x[3, 0, 0]), 50.0 / (1 + 0.1 * 3) / (1 + 0.0001 * 3 / 0.01), 1e-4)


def sequoia():
    r = {"SensorModel": "2,10,3", "ExposureTime": "0.5", "ISO": "100", "FNumber": "2.2", "Irradiance": "20",
         "VignettingPolynomial2D": "2", "VignettingPolynomial2DName": "0,0"}
    f = _foto("Parrot", r)
    g = _foto("Parrot", dict(r, Irradiance="10"))
    radiometria.preparar([f, g])
    x = radiometria.corregir(np.full((4, 4, 1), 113, np.uint16), f)
    esperado = 2.2 ** 2 * 103 / (2 * 0.5 * 100 + 3) / 2 * (15 / 20)
    _cerca(float(x[0, 0, 0]), esperado)
    assert "Sequoia" in radiometria.limitacion(f)


def otras_camaras_no_aplican_polinomios():
    f = _foto("Sentera", {"BlackLevel": "100", "ExposureTime": "0.01", "ISO": "100",
                          "VignettingPolynomial": "5", "VignettingCenter": "0,0"})
    x = radiometria.corregir(np.full((4, 4, 1), 200, np.uint16), f)
    _cerca(float(x[3, 3, 0]), 10000.0)
    assert radiometria.limitacion(f) is not None


def dji_viñeteo_multiplica():
    f = _foto("DJI", {"BlackLevel": "100", "VignettingData": "0.1", "CalibratedOpticalCenterX": "0",
                      "CalibratedOpticalCenterY": "0", "ExposureTime": "1000", "SensorGain": "2",
                      "Irradiance": "4", "SensorGainAdjustment": "2"})
    x = radiometria.corregir(np.full((4, 4, 1), 200, np.uint16), f)
    _cerca(float(x[0, 3, 0]), 100 * 1.3 / (2 * 0.001) * 2 / 4, 1e-4)
    assert radiometria.limitacion(f) is None


PRUEBAS = [lista_xmp, micasense_sin_viñeteo, micasense_viñeteo_divide_y_gradiente, sequoia,
           otras_camaras_no_aplican_polinomios, dji_viñeteo_multiplica]

if __name__ == "__main__":
    fallos = 0
    for p in PRUEBAS:
        try:
            p()
            print(f"OK    {p.__name__}")
        except Exception as e:
            fallos += 1
            print(f"FALLO {p.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(PRUEBAS) - fallos}/{len(PRUEBAS)} correctas")
    sys.exit(1 if fallos else 0)
