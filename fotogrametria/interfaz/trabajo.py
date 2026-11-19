"""Control del proceso de trabajo: iniciar, pausar, reanudar, cancelar y registrar."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

from .. import gpu as gpu_mod
from ..registro import Registro
from ..trabajador import MARCA

RAIZ = Path(__file__).resolve().parent.parent.parent
_SIN_VENTANA = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class Trabajo:
    """al_evento recibe los diccionarios del trabajador y, al terminar, uno de tipo fin, error o cancelado."""

    def __init__(self, al_evento):
        self._al_evento = al_evento
        self._cerrojo = threading.Lock()
        self._proceso: subprocess.Popen | None = None
        self._lector: threading.Thread | None = None
        self._registro: Registro | None = None
        self._salida: Path | None = None
        self._terminal = False     # el trabajador ya informó de cómo acabó
        self._cancelado = False
        self.pausado = False

    # --- proceso listo de antemano -------------------------------------------------------------
    def preparar(self) -> None:
        with self._cerrojo:
            if self._proceso is None or self._proceso.poll() is not None:
                self._lanzar()

    def _lanzar(self) -> None:
        entorno = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", FOTOGRAMETRIA_PADRE=str(os.getpid()))
        self._terminal = self._cancelado = self.pausado = False
        self._proceso = p = subprocess.Popen(
            [sys.executable, "-m", "fotogrametria.trabajador"], cwd=RAIZ, env=entorno, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
            creationflags=_SIN_VENTANA)
        if os.name == "nt":
            gpu_mod._atar_a_esta_app(p.pid)
        self._lector = threading.Thread(target=self._leer, args=(p,), daemon=True)
        self._lector.start()

    def reiniciar(self) -> None:
        """Descarta el proceso preparado (por ejemplo, si se instalaron componentes GPU)."""
        with self._cerrojo:
            p, self._proceso = self._proceso, None
        if p is not None:
            self._terminar(p)
        self.preparar()

    # --- órdenes -------------------------------------------------------------------------------
    def iniciar(self, carpeta: str, salida: str, calidad: str, gpu: str) -> None:
        with self._cerrojo:
            if self._proceso is None or self._proceso.poll() is not None:
                self._lanzar()
            self._salida = Path(salida)
            self._registro = Registro(carpeta, calidad, gpu)
            self._proceso.stdin.write(json.dumps({"carpeta": carpeta, "calidad": calidad, "gpu": gpu}) + "\n")
            self._proceso.stdin.flush()

    def _familia(self) -> list[psutil.Process]:
        try:
            yo = psutil.Process(self._proceso.pid)
            return [yo, *yo.children(recursive=True)]
        except (psutil.NoSuchProcess, AttributeError):
            return []

    def pausar(self) -> bool:
        with self._cerrojo:
            suspendidos = set()     # Windows acumula suspensiones: cada proceso se suspende una sola vez
            for _ in range(2):      # el segundo paso alcanza a un hijo recién creado durante el primero
                for p in self._familia():
                    if p.pid in suspendidos:
                        continue
                    suspendidos.add(p.pid)
                    try:
                        p.suspend()
                    except psutil.Error:
                        pass
            self.pausado = bool(self._familia())
            if self.pausado and self._registro:
                self._registro.escribir("[pausado por el usuario]")
            return self.pausado

    def reanudar(self) -> bool:
        with self._cerrojo:
            for p in self._familia():
                try:
                    p.resume()
                except psutil.Error:
                    pass
            self.pausado = False
            if self._registro:
                self._registro.escribir("[reanudado por el usuario]")
            return True

    def cancelar(self) -> None:
        with self._cerrojo:
            self._cancelado = True
            if self._proceso is not None:
                self._terminar(self._proceso)

    def _terminar(self, proceso: subprocess.Popen) -> None:
        try:
            familia = psutil.Process(proceso.pid)
            for p in [*familia.children(recursive=True), familia]:
                try:
                    p.kill()
                except psutil.Error:
                    pass
        except psutil.NoSuchProcess:
            pass

    # --- lectura -------------------------------------------------------------------------------
    def _leer(self, proceso: subprocess.Popen) -> None:
        for linea in proceso.stdout:
            linea = linea.rstrip("\r\n")
            if not linea.startswith(MARCA):
                if self._registro:
                    self._registro.escribir(linea)
                continue
            evento = json.loads(linea[len(MARCA):])
            if evento["tipo"] == "log" and self._registro:
                self._registro.escribir(evento["texto"])
            elif evento["tipo"] == "error" and self._registro:
                self._registro.escribir(f"ERROR: {evento['mensaje']}\n{evento['detalle']}")
            if evento["tipo"] in ("fin", "error"):
                self._terminal = evento
            else:
                self._al_evento(evento)
        codigo = proceso.wait()
        with self._cerrojo:
            if proceso is not self._proceso:
                return      # proceso preparado que se descartó
            registro, self._registro = self._registro, None
            evento = self._cierre(codigo)
            salida = self._salida
            cancelado = self._cancelado
            self._proceso = None
        if registro is None:
            return          # se cerró antes de recibir trabajo
        registro.cerrar({"fin": "Terminado", "error": "Error", "cancelado": "Cancelado por el usuario"}[evento["tipo"]])
        if cancelado and salida is not None:
            _limpiar(salida)
        evento["registro"] = str(registro.ruta)
        self._al_evento(evento)
        self.preparar()

    def _cierre(self, codigo: int) -> dict:
        if self._cancelado:
            return {"tipo": "cancelado"}
        if self._terminal:
            return self._terminal
        mensaje = f"El procesamiento se cerró inesperadamente (código {codigo})."
        if self._registro:
            self._registro.escribir(f"ERROR: {mensaje}")
        return {"tipo": "error", "mensaje": mensaje, "detalle": ""}


def _limpiar(salida: Path) -> None:
    """Quita los temporales de un trabajo cancelado. Windows libera los archivos poco después de terminar."""
    trabajo = salida / "_trabajo"
    for _ in range(10):
        shutil.rmtree(trabajo, ignore_errors=True)
        if not trabajo.exists():
            break
        time.sleep(0.3)
    try:
        salida.rmdir()      # sólo si quedó vacía
    except OSError:
        pass
