"""Uso:  python -m fotogrametria CARPETA_IMAGENES [--salida CARPETA] [--calidad express|rapida|normal|alta] [--gpu auto|si|no]
       python -m fotogrametria --instalar-gpu

Sin argumentos abre la ventana gráfica.
"""
import argparse
import sys
import time


def main():
    if sys.argv[1:] == ["--instalar-gpu"]:
        from .gpu import instalar
        instalar()
        return
    if len(sys.argv) == 1:
        from .interfaz.app import main as interfaz
        interfaz()
        return
    ap = argparse.ArgumentParser(prog="fotogrametria", description="Imágenes de dron -> ortomosaico automático")
    ap.add_argument("carpeta", help="Carpeta con las imágenes (RGB y/o multiespectrales)")
    ap.add_argument("--salida", help="Carpeta de resultados (por defecto: '<carpeta> - Resultados')")
    ap.add_argument("--calidad", default="normal", choices=["express", "rapida", "normal", "alta"],
                    help="express = sólo ortomosaico, lo más rápido posible")
    ap.add_argument("--gpu", default="auto", choices=["auto", "si", "no"],
                    help="Tarjeta gráfica NVIDIA: auto = usarla si está disponible (por defecto)")
    ap.add_argument("--reanudar", action="store_true", help="Reutiliza la alineación de una ejecución previa")
    a = ap.parse_args()

    from .pipeline import procesar

    ultimo = [0.0]

    def prog(etapa, frac, _local=None):
        if time.time() - ultimo[0] > 5 or frac >= 1:
            ultimo[0] = time.time()
            print(f"    [{frac * 100:5.1f}%] {etapa}", flush=True)

    def log(m):
        print(m, flush=True)

    r = procesar(a.carpeta, a.salida, a.calidad, log=log, progreso_cb=prog, reanudar=a.reanudar, gpu=a.gpu)
    print("\nResumen:")
    for k, v in r["resumen"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
