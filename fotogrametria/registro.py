"""Registro de cada procesamiento, para que quien prueba el programa pueda enviarlo si algo falla.

Se guarda sólo en este equipo y nunca se envía solo. Incluye el equipo, las opciones, los mensajes del
proceso y los errores, también los de COLMAP y las caídas de código nativo. Contiene rutas de carpetas.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

CONSERVAR = 20      # registros que se mantienen
_SIN_VENTANA = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def carpeta() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    ruta = Path(base) / "OrthoDash" / "registros"
    ruta.mkdir(parents=True, exist_ok=True)
    return ruta


def ultimo() -> Path | None:
    registros = sorted(carpeta().glob("*.log"))
    return registros[-1] if registros else None


def _equipo() -> list[str]:
    lineas = [f"Windows: {platform.platform()}", f"Python: {sys.version.split()[0]}",
              f"Procesador: {platform.processor()} ({os.cpu_count()} hilos)"]
    try:
        import psutil
        lineas.append(f"RAM: {psutil.virtual_memory().total / 1024 ** 3:.0f} GB")
    except ImportError:
        pass
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=15, creationflags=_SIN_VENTANA)
        lineas.append(f"GPU: {r.stdout.strip() or 'no detectada'}")
    except (OSError, subprocess.SubprocessError):
        lineas.append("GPU: no detectada")
    return lineas


class Registro:
    def __init__(self, carpeta_fotos: str, calidad: str, gpu: str):
        self.ruta = carpeta() / f"{datetime.now():%Y%m%d-%H%M%S}.log"
        self._cerrojo = threading.Lock()
        self._archivo = open(self.ruta, "w", encoding="utf-8", errors="replace", buffering=1)
        for linea in [f"Inicio: {datetime.now():%Y-%m-%d %H:%M:%S}", *_equipo(), f"Fotos: {carpeta_fotos}",
                      f"Calidad: {calidad}", f"Procesador elegido: {gpu}", ""]:
            self._archivo.write(linea + "\n")
        self._limpiar()

    def escribir(self, texto: str) -> None:
        with self._cerrojo:
            if not self._archivo.closed:
                self._archivo.write(f"{datetime.now():%H:%M:%S} {texto}\n")

    def cerrar(self, resultado: str) -> None:
        self.escribir(f"=== {resultado} ===")
        with self._cerrojo:
            self._archivo.close()

    def _limpiar(self) -> None:
        for vieja in sorted(carpeta().glob("*.log"))[:-CONSERVAR]:
            try:
                vieja.unlink()
            except OSError:
                pass
