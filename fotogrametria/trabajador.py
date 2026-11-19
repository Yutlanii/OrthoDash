"""Proceso aparte que ejecuta el procesamiento para la ventana.

Vive fuera de la interfaz porque buena parte del trabajo es código nativo (pycolmap, CUDA) que no se
puede interrumpir desde Python: suspender o terminar el proceso funciona en cualquier etapa.

Arranca por adelantado y carga las bibliotecas antes de recibir el pedido, para que iniciar no cueste
tiempo. Recibe un JSON por stdin y responde con líneas MARCA + JSON por stdout. El resto de stdout y
stderr (COLMAP, caídas de código nativo) va tal cual al registro.
"""
from __future__ import annotations

import faulthandler
import json
import sys
import time
import traceback

MARCA = "@@FOTOGRAMETRIA@@"


def _enviar(**datos) -> None:
    print(MARCA + json.dumps(datos, default=str), flush=True)


def main() -> None:
    faulthandler.enable()     # una caída de código nativo deja su pila en stderr, y por tanto en el registro
    from . import gpu
    from .pipeline import _modulo_gpu, procesar
    if gpu.detectar().cupy:
        _modulo_gpu("densa_gpu"), _modulo_gpu("ortomosaico_gpu")
    linea = sys.stdin.readline()
    if not linea:
        return
    pedido = json.loads(linea)
    ultimo = [0.0]

    def progreso(etapa, total, local):
        ahora = time.time()
        if ahora - ultimo[0] > 0.15 or local >= 1.0:
            ultimo[0] = ahora
            _enviar(tipo="progreso", etapa=etapa, total=total, local=local)

    try:
        r = procesar(pedido["carpeta"], None, pedido["calidad"], log=lambda m: _enviar(tipo="log", texto=m),
                     progreso_cb=progreso, gpu=pedido["gpu"])
        _enviar(tipo="fin", salida=r["salida"], resumen=r["resumen"])
    except Exception as e:
        _enviar(tipo="error", mensaje=str(e), detalle=traceback.format_exc())


if __name__ == "__main__":
    main()
