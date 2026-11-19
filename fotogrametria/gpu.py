"""Aceleración por tarjeta gráfica (NVIDIA CUDA).

Dos vías independientes:
  * COLMAP-CUDA (ejecutable oficial, misma versión que pycolmap, misma base de datos): detección
    SIFT y emparejamiento. pycolmap para Windows no trae CUDA.
  * CuPy (kernels CUDA propios): nube densa (SGM) y ortomosaico.
El ajuste de haces sigue en CPU: el Ceres de COLMAP para Windows no trae CUDA.

Modos: "auto" (usa la GPU en lo que esté disponible), "si" (exige GPU), "no" (sólo CPU).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pycolmap

RAIZ = Path(__file__).resolve().parent.parent
DIR_COLMAP = RAIZ / "herramientas" / "colmap-cuda"
URL_COLMAP = (f"https://github.com/colmap/colmap/releases/download/{pycolmap.__version__}/"
              "colmap-x64-windows-cuda.zip")
MODOS = ("auto", "si", "no")
_SIN_VENTANA = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass
class InfoGPU:
    nombre: str | None = None
    memoria_mb: int = 0
    colmap: Path | None = None   # SIFT + emparejamiento
    cupy: bool = False           # nube densa + ortomosaico

    @property
    def nvidia(self) -> bool:
        return self.nombre is not None

    @property
    def utilizable(self) -> bool:
        return self.nvidia and (self.colmap is not None or self.cupy)

    @property
    def completo(self) -> bool:
        return self.nvidia and self.colmap is not None and self.cupy

    def describir(self) -> str:
        if not self.nvidia:
            return "no se encontró tarjeta NVIDIA"
        if not self.utilizable:
            return f"{self.nombre} (faltan los componentes GPU: ejecuta Instalar.bat)"
        faltan = [n for n, ok in (("alineación", self.colmap), ("nube densa y ortomosaico", self.cupy)) if not ok]
        extra = f"; sin GPU para {', '.join(faltan)}" if faltan else ""
        return f"{self.nombre}, {self.memoria_mb / 1024:.0f} GB{extra}"


def _buscar_colmap() -> Path | None:
    for c in (DIR_COLMAP / "bin" / "colmap.exe", DIR_COLMAP / "colmap.exe"):
        if c.exists():
            return c
    hall = list(DIR_COLMAP.glob("**/bin/colmap.exe")) if DIR_COLMAP.exists() else []
    return hall[0] if hall else None


@lru_cache(maxsize=1)
def detectar() -> InfoGPU:
    info = InfoGPU(colmap=_buscar_colmap())
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=15, creationflags=_SIN_VENTANA)
        if r.returncode == 0 and r.stdout.strip():
            nombre, mem = r.stdout.strip().splitlines()[0].rsplit(",", 1)
            info.nombre, info.memoria_mb = nombre.strip(), int(float(mem))
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    if info.nvidia:
        info.cupy = _cupy_funciona()
    return info


def _cupy_funciona() -> bool:
    """CuPy importa, ve la GPU y compila un kernel (NVRTC compatible con la arquitectura)."""
    try:
        import warnings
        warnings.filterwarnings("ignore", message="CUDA path could not be detected")
        import cupy as cp
        k = cp.RawKernel(r'extern "C" __global__ void f(int* a){a[threadIdx.x]=threadIdx.x;}', "f")
        a = cp.zeros(32, cp.int32)
        k((1,), (32,), (a,))
        return int(a[31]) == 31
    except Exception:
        return False


def prioridad_maxima() -> str:
    """Sube la prioridad del proceso en la CPU y en el planificador de GPU de Windows.

    Así el procesamiento gana tiempo de GPU frente a otros programas abiertos (editores 3D, fondos
    animados, navegadores) sin bloquear el sistema: se usa "alta", no "tiempo real".
    """
    if os.name != "nt":
        return "normal"
    return _prioridad_alta(os.getpid())


def _prioridad_alta(pid: int) -> str:
    import ctypes
    k32 = ctypes.windll.kernel32
    k32.OpenProcess.restype = ctypes.c_void_p
    k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    k32.CloseHandle.argtypes = [ctypes.c_void_p]
    proc = k32.OpenProcess(0x0200 | 0x0400, False, pid)  # SET_INFORMATION | QUERY_INFORMATION
    if not proc:
        return "normal"
    try:
        k32.SetPriorityClass(proc, 0x00000080)                       # HIGH_PRIORITY_CLASS
        f = ctypes.windll.gdi32.D3DKMTSetProcessSchedulingPriorityClass
        f.argtypes, f.restype = [ctypes.c_void_p, ctypes.c_int], ctypes.c_long
        if f(proc, 4) == 0:      # D3DKMT_SCHEDULINGPRIORITYCLASS_HIGH -> STATUS_SUCCESS
            return "alta (CPU y GPU)"
    except (AttributeError, OSError):
        pass
    finally:
        k32.CloseHandle(proc)
    return "alta (sólo CPU)"


def uso_ajeno() -> list[tuple[str, int]]:
    """Programas que están usando la GPU ahora (nombre, % de núcleos), de mayor a menor.

    Windows reparte la GPU por turnos: un editor 3D o un juego abierto puede hacer que los pasos con
    muchos lanzamientos cortos (SIFT) vayan 5-10x más lentos.
    """
    try:
        r = subprocess.run(["nvidia-smi", "pmon", "-c", "1", "-s", "u"], capture_output=True, text=True,
                           timeout=20, creationflags=_SIN_VENTANA)
    except (OSError, subprocess.SubprocessError):
        return []
    propios = _pids_propios()   # esta aplicación y sus hijos (p. ej. el visor WebView2)
    uso: dict[str, int] = {}
    for ln in r.stdout.splitlines():
        c = ln.split()
        if ln.startswith("#") or len(c) < 5 or not c[3].isdigit() or (c[1].isdigit() and int(c[1]) in propios):
            continue
        nombre = " ".join(c[9:]) if len(c) > 9 else c[-1]
        uso[nombre] = max(uso.get(nombre, 0), int(c[3]))
    return sorted(((n, u) for n, u in uso.items() if u >= 5), key=lambda x: -x[1])


def _pids_propios() -> set[int]:
    try:
        import psutil
        # el procesamiento corre en un proceso hijo de la ventana: el visor WebView2 también es propio
        yo = psutil.Process(int(os.environ.get("FOTOGRAMETRIA_PADRE", os.getpid())))
        return {os.getpid(), yo.pid, *(h.pid for h in yo.children(recursive=True))}
    except Exception:
        return {os.getpid()}


def vram_libre_mb() -> int:
    """VRAM realmente libre según el controlador (nvidia-smi).

    En Windows (WDDM) cudaMemGetInfo informa casi toda la memoria como libre aunque otros programas
    la usen; si se confía en ese dato, CUDA desborda a la RAM del sistema y todo va 5-10x más lento.
    """
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=15, creationflags=_SIN_VENTANA)
        total, usada = (int(float(x)) for x in r.stdout.strip().splitlines()[0].split(","))
        return max(0, total - usada)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return 2048


def resolver(modo: str, log=print) -> InfoGPU | None:
    """Devuelve la GPU a usar o None (CPU). En modo "si" falla si no se puede usar."""
    if modo not in MODOS:
        raise ValueError(f"modo GPU desconocido: {modo!r} (usa {', '.join(MODOS)})")
    if modo == "no":
        log(f"  Procesador: CPU (GPU desactivada) — prioridad {prioridad_maxima()}")
        return None
    info = detectar()
    if info.utilizable:
        log(f"  Procesador: GPU {info.describir()} — prioridad {prioridad_maxima()}")
        ajenos = uso_ajeno()
        if ajenos:
            lista = ", ".join(f"{n} {u}%" for n, u in ajenos[:3])
            log(f"  [aviso] Otros programas están usando la tarjeta gráfica ({lista}). "
                "Ciérralos para que el proceso sea más rápido.")
        return info
    if modo == "si":
        raise RuntimeError(f"Se pidió usar la GPU pero no está disponible: {info.describir()}.")
    log(f"  Procesador: CPU ({info.describir()}) — prioridad {prioridad_maxima()}")
    return None


_trabajo_hijos = None


def _atar_a_esta_app(pid: int) -> None:
    """Mete el proceso hijo en un Job Object con KILL_ON_JOB_CLOSE: si esta aplicación se cierra (o
    se cae) a mitad del proceso, Windows termina también COLMAP. Sin esto quedaba huérfano ocupando
    la GPU al 98 % (visto al cerrar la ventana durante el emparejamiento)."""
    global _trabajo_hijos
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.windll.kernel32
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    if _trabajo_hijos is None:
        class Basica(ctypes.Structure):
            _fields_ = [("a", ctypes.c_int64), ("b", ctypes.c_int64), ("LimitFlags", wintypes.DWORD),
                        ("c", ctypes.c_size_t), ("d", ctypes.c_size_t), ("e", wintypes.DWORD),
                        ("f", ctypes.c_size_t), ("g", wintypes.DWORD), ("h", wintypes.DWORD)]

        class Extendida(ctypes.Structure):
            _fields_ = [("Basic", Basica), ("Io", ctypes.c_uint64 * 6), ("i", ctypes.c_size_t),
                        ("j", ctypes.c_size_t), ("k", ctypes.c_size_t), ("l", ctypes.c_size_t)]
        job = k32.CreateJobObjectW(None, None)
        info = Extendida()
        info.Basic.LimitFlags = 0x2000                      # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))   # extendida
        _trabajo_hijos = job                                # Windows lo cierra al terminar este proceso
    h = k32.OpenProcess(0x0100 | 0x0001, False, pid)        # PROCESS_SET_QUOTA | PROCESS_TERMINATE
    if h:
        k32.AssignProcessToJobObject(_trabajo_hijos, h)
        k32.CloseHandle(h)


def ejecutar_colmap(info: InfoGPU, args: list[str], log=print) -> None:
    """Ejecuta un comando de COLMAP-CUDA; muestra sólo los errores."""
    entorno = dict(os.environ)
    entorno["PATH"] = str(info.colmap.parent) + os.pathsep + str(info.colmap.parent.parent / "lib") + os.pathsep + entorno.get("PATH", "")
    p = subprocess.Popen([str(info.colmap), *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         errors="replace", env=entorno, creationflags=_SIN_VENTANA)
    if os.name == "nt":
        _atar_a_esta_app(p.pid)
        _prioridad_alta(p.pid)   # también en el planificador de GPU, como el proceso principal
    salida, _ = p.communicate()
    if p.returncode != 0:
        cola = "\n".join(salida.strip().splitlines()[-15:])
        raise RuntimeError(f"COLMAP-CUDA falló ({args[0]}, código {p.returncode}):\n{cola}")


# ----------------------------------------------------------------------------------------------
# Instalación del componente GPU (≈400 MB). Descarga por partes en paralelo y reanudable, porque
# las conexiones lentas o inestables cortan las descargas grandes.

# Se usa curl.exe (incluido en Windows 10/11) porque usa los certificados del sistema; el Python de
# algunos equipos no los reconoce.
def _curl(*args: str, timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["curl.exe", "-sS", "-L", "--connect-timeout", "30", *args], capture_output=True,
                          text=True, errors="replace", timeout=timeout, creationflags=_SIN_VENTANA)


def _tamano(url: str) -> int:
    error = ""
    for _ in range(20):
        try:
            r = _curl("-f", "-I", url, timeout=120)
        except subprocess.TimeoutExpired:
            continue
        largos = [ln.split(":", 1)[1] for ln in r.stdout.splitlines() if ln.lower().startswith("content-length:")]
        if r.returncode == 0 and largos and int(largos[-1]) > 0:
            return int(largos[-1])
        error = r.stderr.strip()
    raise RuntimeError(f"No se pudo acceder a {url}: {error}")


def _descargar(url: str, destino: Path, partes: int = 16, log=print) -> None:
    total = _tamano(url)
    trozos = destino.with_suffix(".partes")
    trozos.mkdir(parents=True, exist_ok=True)
    paso = -(-total // partes)
    rangos = [(i, i * paso, min(total, (i + 1) * paso) - 1) for i in range(partes)]
    errores: list[str] = []

    def bajar(i, ini, fin):
        ruta = trozos / f"{i:03d}"
        for _ in range(100):
            ya = ruta.stat().st_size if ruta.exists() else 0
            if ini + ya > fin:
                return
            with open(ruta, "ab") as f:
                # URL original en cada intento: la redirección de GitHub es una URL firmada que caduca
                subprocess.run(["curl.exe", "-sSf", "-L", "--connect-timeout", "30", "--speed-limit", "1024",
                                "--speed-time", "60", "-r", f"{ini + ya}-{fin}", url], stdout=f,
                               stderr=subprocess.DEVNULL, creationflags=_SIN_VENTANA)
        errores.append(f"parte {i}")

    def hechos():
        return sum((trozos / f"{i:03d}").stat().st_size for i in range(partes) if (trozos / f"{i:03d}").exists())

    hilos = [threading.Thread(target=bajar, args=r, daemon=True) for r in rangos]
    for h in hilos:
        h.start()
    ultimo = -1
    while any(h.is_alive() for h in hilos):
        for h in hilos:
            h.join(timeout=5)
        pct = int(100 * hechos() / total)
        if pct // 10 != ultimo:
            ultimo = pct // 10
            log(f"  descargado {pct}% de {total / 1e6:.0f} MB")
    if errores:
        raise RuntimeError(f"No se pudo completar la descarga ({', '.join(errores)}). Vuelve a intentarlo.")
    with open(destino, "wb") as out:
        for i in range(partes):
            with open(trozos / f"{i:03d}", "rb") as f:
                shutil.copyfileobj(f, out)
    if destino.stat().st_size != total:
        raise RuntimeError("Descarga incompleta; vuelve a intentarlo.")
    shutil.rmtree(trozos)


def instalar(log=print) -> InfoGPU:
    """Descarga COLMAP-CUDA si hay una NVIDIA y aún no está instalado."""
    detectar.cache_clear()
    info = detectar()
    if not info.nvidia:
        log("No hay tarjeta NVIDIA: se usará la CPU.")
        return info
    if not info.cupy:
        # CUDA para Python (nube densa y ortomosaico). NVRTC >= 12.8 compila para todas las
        # generaciones actuales, incluida Blackwell (RTX 50).
        log(f"Tarjeta {info.nombre} encontrada. Instalando CUDA para Python (CuPy)...")
        import sys
        r = subprocess.run([sys.executable, "-m", "pip", "install", "--retries", "10", "--timeout", "120",
                            "cupy-cuda12x", "nvidia-cuda-nvrtc-cu12>=12.8", "nvidia-cuda-runtime-cu12>=12.8"],
                           creationflags=_SIN_VENTANA)
        if r.returncode != 0:
            log("  [aviso] No se pudo instalar CuPy; la nube densa y el ortomosaico usarán la CPU.")
        detectar.cache_clear()
        info = detectar()
    if info.colmap:
        log(f"Componentes GPU instalados ({info.describir()}).")
        return info
    log(f"Descargando componente GPU de alineación (COLMAP {pycolmap.__version__} CUDA)...")
    zip_ = DIR_COLMAP.with_suffix(".zip")
    _descargar(URL_COLMAP, zip_, log=log)
    log("  Descomprimiendo...")
    tmp = DIR_COLMAP.with_name(DIR_COLMAP.name + "_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(zip_) as z:
        z.extractall(tmp)
    hijos = list(tmp.iterdir())
    shutil.rmtree(DIR_COLMAP, ignore_errors=True)
    (hijos[0] if len(hijos) == 1 and hijos[0].is_dir() else tmp).rename(DIR_COLMAP)
    shutil.rmtree(tmp, ignore_errors=True)
    zip_.unlink()
    for prueba in DIR_COLMAP.glob("bin/*_test.exe"):  # pruebas unitarias de COLMAP (~190 MB)
        prueba.unlink()
    detectar.cache_clear()
    info = detectar()
    log("Componente GPU instalado." if info.colmap else "[aviso] No se encontró colmap.exe tras instalar.")
    return info


if __name__ == "__main__":
    instalar()
