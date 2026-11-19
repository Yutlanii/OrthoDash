"""Ventana principal: interfaz web (WebView2 de Windows) + servidor local del visor."""
from __future__ import annotations

import ctypes
import json
import os
import shutil
import threading
import traceback
from pathlib import Path

import webview

from .. import gpu as gpu_mod
from .. import registro
from ..pipeline import CALIDADES, ETAPAS
from .servidor import Visor, iniciar
from .trabajo import Trabajo

NOMBRE = "OrthoDash"
ICONO = Path(__file__).resolve().parent / "web" / "logo.ico"
EXT_IMAGEN = {".jpg", ".jpeg", ".tif", ".tiff", ".png", ".dng"}


class Api:
    """Métodos que la interfaz llama con `pywebview.api.<método>()` (devuelven JSON)."""

    def __init__(self, visor: Visor):
        self._visor = visor
        self._ventana = None
        self._trabajando = False
        self._trabajo = Trabajo(self._evento_trabajo)

    # --- eventos hacia la interfaz -----------------------------------------------------------
    def _evento(self, **datos):
        if self._ventana is not None:
            self._ventana.evaluate_js(f"window.app && app.evento({json.dumps(datos, default=str)})")

    # --- consultas ----------------------------------------------------------------------------
    def estado(self):
        return {"etapas": [nombre for _, nombre, _ in ETAPAS], "calidades": list(CALIDADES)}

    def estado_gpu(self):
        info = gpu_mod.detectar()
        return {"nvidia": info.nvidia, "nombre": info.nombre, "memoria_mb": info.memoria_mb,
                "colmap": info.colmap is not None, "cupy": info.cupy, "utilizable": info.utilizable,
                "completo": info.completo, "libre_mb": gpu_mod.vram_libre_mb() if info.nvidia else 0,
                "ajenos": gpu_mod.uso_ajeno() if info.nvidia else []}

    def elegir_carpeta(self):
        r = self._ventana.create_file_dialog(webview.FileDialog.FOLDER)
        if not r:
            return None
        return self.describir_carpeta(r[0])

    def describir_carpeta(self, ruta):
        p = Path(ruta)
        imgs = [f for f in p.iterdir() if f.suffix.lower() in EXT_IMAGEN and f.is_file()]
        return {"ruta": str(p), "nombre": p.name, "imagenes": len(imgs),
                "megabytes": round(sum(f.stat().st_size for f in imgs) / 1e6, 1),
                "resultados": str(p.parent / f"{p.name} - Resultados")}

    def analizar_carpeta(self, ruta):
        """Detecta la cámara en segundo plano; el resultado llega con el evento "analisis"."""
        def tarea():
            from .. import metadata
            try:
                fotos = metadata.escanear_carpeta(Path(ruta), log=lambda m: None)
                lev = metadata.analizar(fotos, log=lambda m: None)
                p0 = lev.sensores[lev.principal][0]
                bandas = [g[0].banda for g in lev.sensores.values() if g[0].tipo == "multiespectral"]
                self._evento(tipo="analisis", ruta=ruta, camara=p0.modelo or p0.fabricante or "Cámara desconocida",
                             disparos=len(lev.sensores[lev.principal]), rgb=p0.es_rgb, bandas=sorted(bandas),
                             gps=sum(f.lat is not None for f in lev.fotos), total=len(lev.fotos),
                             ignoradas=[m for _, _, m in lev.ignoradas])
            except Exception as e:
                self._evento(tipo="analisis", ruta=ruta, error=str(e))
        threading.Thread(target=tarea, daemon=True).start()

    def abrir_resultados(self, ruta=None):
        if ruta is None:
            r = self._ventana.create_file_dialog(webview.FileDialog.FOLDER)
            if not r:
                return None
            ruta = r[0]
        try:
            return self._visor.abrir(Path(ruta))
        except Exception as e:
            return {"error": str(e)}

    def abrir_carpeta(self, ruta):
        if ruta and Path(ruta).exists():
            os.startfile(ruta)

    def abrir_archivo(self, ruta):
        if ruta and Path(ruta).exists():
            os.startfile(ruta)

    # --- procesamiento ------------------------------------------------------------------------
    def iniciar(self, carpeta, calidad="normal", gpu="auto"):
        if self._trabajando:
            return False
        self._trabajando = True
        salida = self.describir_carpeta(carpeta)["resultados"]
        try:
            self._trabajo.iniciar(carpeta, salida, calidad, gpu)
        except Exception as e:
            self._trabajando = False
            self._evento(tipo="error", mensaje=f"No se pudo iniciar el procesamiento: {e}", detalle=traceback.format_exc())
            return False
        return True

    def pausar(self):
        return self._trabajo.pausar()

    def reanudar(self):
        return self._trabajo.reanudar()

    def cancelar(self):
        self._trabajo.cancelar()

    def _evento_trabajo(self, e):
        if e["tipo"] in ("fin", "error", "cancelado"):
            self._trabajando = False
        if e["tipo"] == "fin":
            try:
                e["info"] = self._visor.abrir(Path(e["salida"]))
            except Exception as err:
                e = {"tipo": "error", "mensaje": str(err), "detalle": traceback.format_exc(), "registro": e.get("registro")}
        self._evento(**e)

    def guardar_registro(self):
        """Guarda una copia del último registro donde elija el usuario, para poder enviarla."""
        origen = registro.ultimo()
        if origen is None:
            return {"error": "Todavía no hay ningún registro."}
        r = self._ventana.create_file_dialog(webview.FileDialog.SAVE, directory=str(Path.home() / "Desktop"),
                                             save_filename=f"registro-{origen.name}")
        if not r:
            return None
        destino = r if isinstance(r, str) else r[0]
        shutil.copyfile(origen, destino)
        return {"ruta": destino}

    def instalar_gpu(self):
        if self._trabajando:
            return False
        self._trabajando = True

        def tarea():
            try:
                gpu_mod.instalar(log=lambda m: self._evento(tipo="log", texto=m))
                gpu_mod.detectar.cache_clear()
                self._trabajo.reiniciar()
                self._evento(tipo="gpu", estado=self.estado_gpu())
            except Exception as e:
                self._evento(tipo="error", mensaje=f"No se pudo instalar el componente GPU: {e}", detalle="")
            finally:
                self._trabajando = False
        threading.Thread(target=tarea, daemon=True).start()
        return True


def _identidad_windows() -> None:
    """Sin esto Windows agrupa la ventana en la barra de tareas con Python y muestra su icono."""
    if os.name == "nt":
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(NOMBRE)


def _poner_icono() -> None:
    """pywebview sólo acepta icono en Linux; en Windows se asigna a la ventana ya creada."""
    if os.name != "nt":
        return
    u32 = ctypes.windll.user32
    u32.FindWindowW.restype = u32.LoadImageW.restype = ctypes.c_void_p
    u32.LoadImageW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
    u32.SendMessageW.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_void_p]
    ventana = u32.FindWindowW(None, NOMBRE)
    for tipo, lado in ((1, 32), (0, 16)):      # WM_SETICON: grande y pequeño
        icono = u32.LoadImageW(None, str(ICONO), 1, lado, lado, 0x10)      # IMAGE_ICON, LR_LOADFROMFILE
        if ventana and icono:
            u32.SendMessageW(ventana, 0x0080, tipo, icono)


def main():
    _identidad_windows()
    visor = Visor()
    _, url = iniciar(visor)
    api = Api(visor)
    ventana = webview.create_window(NOMBRE, url, js_api=api, width=1440, height=900,
                                    min_size=(1100, 700), background_color="#FFFFFF")
    api._ventana = ventana
    ventana.events.shown += _poner_icono
    threading.Thread(target=api._trabajo.preparar, daemon=True).start()
    webview.start(gui="edgechromium", debug=bool(os.environ.get("FOTOGRAMETRIA_DEBUG")))


if __name__ == "__main__":
    main()
