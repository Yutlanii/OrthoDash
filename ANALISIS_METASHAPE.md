# Análisis de Agisoft Metashape Pro y su flujo de fotogrametría

Este documento resume lo que se obtuvo de la instalación local
`C:\Program Files\Agisoft\Metashape Pro` y cómo se trasladó a este proyecto.

## Fuentes analizadas

| Fuente | Qué aporta |
|---|---|
| `metashape.exe` (108 MB, monolítico, Qt5 + Python 3.9 embebido) | 233 648 cadenas ASCII extraídas: nombres de tareas internas, parámetros de cada etapa y docstrings de la API Python embebida |
| `help/metashape.chm` (manual descompilado con `hh.exe -decompile`) | Descripción oficial de cada algoritmo y de sus parámetros por defecto |
| `licenses/*` | Librerías de terceros que revelan la tecnología usada internamente |

La API Python no viene en un `.pyd` aparte: está compilada dentro de `metashape.exe`.
No se copió código de Metashape; el proyecto reimplementa el flujo con código propio y
librerías abiertas.

## Librerías internas detectadas (carpeta `licenses`)

* **cholmod**: solucionador de matrices dispersas → ajuste de haces (*bundle adjustment*).
* **guided-filter**: filtrado de mapas de profundidad que respeta bordes.
* **opencv**: visión por computadora (rectificación, remuestreo).
* **poisson**: reconstrucción de mallas (Screened Poisson).
* **laszip**: exportación LAS/LAZ.
* **geographiclib**: geodesia y sistemas de coordenadas.
* **vulkan / vulkanmemoryallocator / spirvreflect**: cálculo en GPU por Vulkan (no CUDA).
* **libtiff / openjpeg / libwebp / zstd**: formatos ráster de salida.

## Tareas internas y sus parámetros (cadenas `Tarea/parámetro` del ejecutable)

```
MatchPhotos/   downscale, generic_preselection, reference_preselection, reference_preselection_mode,
               keypoint_limit, tiepoint_limit, filter_mask, mask_tiepoints, filter_stationary_points,
               ntrees, neighbour_checks, generic_preselection_limit/level/reduce,
               reference_preselection_neighbors, filter_matches, binary_features,
               guided_matching_neighbors, filter_weak_points
AlignCameras/  adaptive_fitting, hierarchical_threshold
OptimizeCameras/ adaptive_fitting, corrections_order, corrections_reduced
BuildDepthMaps/ downscale, filter_mode, max_neighbors, point_threshold, optimize_range,
               enable_hierarchical, dog_filter_threshold, pm_enable, pm_max_neighbors,
               pm_subsample_neighbors, max_gpu_multiplier, image_cache_size
BuildPointCloud/ point_colors, point_confidence, max_neighbors, density
BuildOrthomosaic/ hole_filling_depth, expansion_limit, seamlines_downscale_level
```

Lo que se deduce de estos nombres:

* `ntrees` y `neighbour_checks`: el emparejamiento usa árboles k-d aproximados (FLANN).
* `binary_features`: existe un modo alternativo con descriptores binarios.
* `pm_*`: los mapas de profundidad tienen un modo PatchMatch además del modo por defecto.
* `enable_hierarchical`: la estimación de profundidad es jerárquica (de grueso a fino).
* `dog_filter_threshold`: los puntos clave son extremos de diferencia de gaussianas, es decir, tipo SIFT.
* `optimize_range`: el rango de búsqueda de profundidad se acota con los puntos de enlace.

Ejemplos de uso embebidos en el binario:

```python
chunk.matchPhotos(downscale=1, generic_preselection=True, reference_preselection=False)
chunk.alignCameras()
chunk.buildDepthMaps(downscale=4, filter_mode=Metashape.AggressiveFiltering)
```

## Algoritmo por etapa (según el manual) y cómo se reimplementó

| Etapa Metashape | Cómo funciona en Metashape | Implementación en este proyecto |
|---|---|---|
| **Add Photos** | Lee EXIF/XMP y crea un grupo de calibración por cámara física. En multiespectral, cada banda es un sensor de un sistema multicámara; el sensor maestro es el de menor longitud de onda. | `metadata.py`: lee EXIF/XMP (GPS, gimbal, banda, longitud de onda, radiometría). Agrupa por sensor y elige sensor principal (RGB o la banda más nítida). |
| **Align Photos** | Accuracy High = imagen original; Medium = ¼ de píxeles. *Key point limit* 40 000 y *tie point limit* 4 000 a 10 000. Preselección genérica (a baja resolución) o por referencia (GPS). Aerotriangulación con ajuste de haces basado en ecuaciones de colinealidad. | `alineacion.py`: SIFT con límite de puntos según la calidad. Emparejamiento por vecindad GPS (*reference preselection*) o exhaustivo si no hay GPS. SfM incremental con *pose priors* GPS. |
| **Optimize Cameras** | Ajusta f, cx, cy, k1 a k3 y p1, p2 junto con las posiciones GPS (precisión de cámara por defecto: 10 m). | Alineación Sim3 a GPS con RANSAC y luego ajuste de haces con observaciones GPS (σ 5 m horizontal y 10 m vertical, pérdida robusta). Modelo OPENCV (f, cx, cy, k1, k2, p1, p2). |
| **Build Depth Maps** | Mapas de profundidad por pares de imágenes que se combinan en un mapa por cámara. El filtro de ruido por componentes conexas tiene tres modos: Mild, Moderate y Aggressive (este último recomendado para vuelos aéreos). | `densa.py`: rectificación estéreo y Semi-Global Matching con verificación izquierda-derecha. El filtro *speckle* por componentes conexas tiene los mismos tres modos. Mapa combinado por mediana con consistencia de al menos 2 pares. |
| **Build Point Cloud** | Nubes parciales por cámara fusionadas con filtrado en zonas de traslape. Guarda la "confianza" (número de mapas que aportan a cada punto). | Fusión en vóxeles del tamaño del GSD denso. Confianza = cámaras distintas por vóxel; se filtra con confianza ≥ 2 y se eliminan valores atípicos por celda (IQR). |
| **Build DEM** | Rasteriza la nube. Con *interpolation Enabled* (el modo por defecto) rellena todo lo visto por al menos una imagen. | `dem.py`: mediana por celda, filtro de picos e interpolación *push-pull* restringida a las huellas de las cámaras. |
| **Build Orthomosaic** | Superficie DEM. Blending *Mosaic*: separa la imagen en bandas de frecuencia; la frecuencia más alta se mezcla sólo en la línea de corte y las más bajas en zonas más anchas. En modo *Disabled* toma la cámara más perpendicular a la superficie. Hole filling activado. | `ortomosaico.py`: proyecta cada píxel del DEM a las fotos y elige la más nadir y centrada. Fusión multibanda por pirámide laplaciana (5 niveles), por bloques en paralelo. |
| **Calibrate Reflectance** | Paneles y/o sensor de sol. Viñeteo como polinomio bivariado de grado 3 (V = exp(Σ cᵢⱼ xⁱ yʲ)), con los coeficientes tomados del XMP. | `radiometria.py`: nivel de negro, viñeteo del XMP, exposición y ganancia, y sensor de sol, lo que da reflectancia relativa. Calcula NDVI si hay bandas Red y NIR. |
| **Export** | GeoTIFF, LAS/LAZ y otros. | GeoTIFF con *overviews* (ortomosaico RGBA o multibanda float32, DEM float32), LAZ 1.4 con color y confianza, PLY, CSV de cámaras e informe HTML. |

## Diferencias y mejoras respecto a Metashape

* **Un solo paso**: no hay que guardar el proyecto `.psx`, ajustar la región ni elegir parámetros en cada diálogo. Todo se calcula a partir de los datos (GSD, traslape, RAM y sensores).
* **Valores por defecto adecuados para dron** (los que el manual recomienda para vuelos aéreos): filtro agresivo, DEM desde la nube con interpolación y blending Mosaic.
* **Multiespectral automático**: detecta las bandas, completa la pose de las bandas no alineadas usando el *rig* (disparo común), genera el ortomosaico multibanda y el NDVI.
* **Informe de calidad automático**: error de reproyección, error frente a GPS, GSD y tiempos por etapa.
* **Limitaciones actuales frente a Metashape**:
  * no hay GCP ni puntos de control manuales;
  * no hay clasificación de terreno (sólo DSM, no DTM);
  * la densificación corre en CPU (pycolmap para Windows no incluye CUDA).
