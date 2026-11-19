# OrthoDash

Convierte las fotos de un vuelo de dron en un **ortomosaico georreferenciado**. También genera
el **modelo digital de elevación (DEM)** y la **nube de puntos densa**. Todo se hace con un solo
botón y sin configurar parámetros.

Funciona con imágenes **RGB**, **multiespectrales** (DJI P4M y Mavic 3M, MicaSense, Parrot
Sequoia) o **ambas a la vez**.

## Uso

1. La primera vez, haz doble clic en **`Instalar.bat`**. Necesita Python 3.10 a 3.13.
2. Haz doble clic en **`OrthoDash`** (o en `Ejecutar.bat`).
3. Pulsa **Elegir carpeta de fotos**, selecciona la carpeta del vuelo, elige la calidad y pulsa **Iniciar**.
   Al elegir la carpeta, el programa muestra la cámara detectada y el tiempo estimado de cada calidad.
4. Al terminar, el resultado aparece en el visor de la derecha:
   * **Ortomosaico**, **Elevación** (relieve sombreado con leyenda) y **NDVI** o **Multiespectral** cuando
     corresponde, en un mapa con zoom fluido. Bajo el cursor se leen las coordenadas UTM, latitud/longitud y la altura.
   * **Nube 3D**: hasta 12 millones de puntos, con color real o por altura y sombreado EDL.
     Arrastrar gira, clic derecho mueve, la rueda acerca y doble clic vuelve a la vista inicial.
   * Los botones **Informe** y **Abrir carpeta** de la pestaña Resultados abren el informe y la carpeta de resultados.
   * El botón del sol o la luna cambia entre tema claro y oscuro.

Los resultados se guardan junto a la carpeta de fotos, en `«carpeta» - Resultados`. Con
**Abrir resultados** se puede volver a ver cualquier procesamiento previo.

### Desde la terminal

```bat
.venv\Scripts\python.exe -m fotogrametria "D:\Vuelos\Parcela1"
.venv\Scripts\python.exe -m fotogrametria "D:\Vuelos\Parcela1" --calidad alta --salida "D:\Resultados\Parcela1"
```

Las calidades disponibles son `express`, `rapida`, `normal` (Estándar, la predeterminada) y `alta`.

### Modo Express (solo ortomosaico)

Es para cuando se necesita el ortomosaico en el momento, por ejemplo en campo.

| Vuelo | Fotos | Express |
|---|---|---|
| DJI Air 2S | 59 | 12 s |
| Parrot Sequoia | 302 | 1 min |
| DJI Mavic 3M (RGB + 4 bandas, NDVI e imagen de 7 bandas) | 309 disparos, 1545 fotos | 3,6 min |

* **Alineación ligera:** fotos de 1000-1400 px decodificadas ya reducidas desde el JPEG, unos 3000 puntos
  por foto, 8 vecinas y SfM global con pocas trazas. Después, el mismo ajuste con GPS de los demás modos.
* **Sin nube densa:** se triangulan las coincidencias ya verificadas entre fotos, con las cámaras fijas.
* **DEM aproximado** (0,5 m) a partir de esos puntos, como superficie de proyección.
* **Ortomosaico** en GPU a 2 veces el GSD, con las fotos y las bandas a media resolución, y compresión rápida.

Se obtienen el ortomosaico, un DEM aproximado y la nube dispersa (`nube_dispersa.laz`). Tiene menos
resolución que los otros modos, y en zonas con mucho relieve o árboles altos puede haber estiramientos
cerca de los bordes.
Con `--gpu auto|si|no` se elige el procesador (ver abajo).

## Tarjeta gráfica (GPU)

Si el equipo tiene una tarjeta **NVIDIA**, el programa la usa en todas las etapas donde es posible:

| Etapa | En la GPU |
|---|---|
| Detección de puntos SIFT y emparejamiento | COLMAP oficial con CUDA (misma versión que pycolmap) |
| Nube densa | Kernels CUDA propios: rectificación, census, Semi-Global Matching, verificación izquierda-derecha, triangulación y fusión en vóxeles |
| Ortomosaico | Kernels CUDA propios: proyección sobre el DEM, selección de la foto más nadir, muestreo y fusión multibanda |
| Visor 3D | WebGL2 (la GPU dibuja la nube a través del motor nativo del navegador) |

El ajuste de haces y la reconstrucción de cámaras siguen en la CPU, porque el COLMAP para Windows no
trae CUDA en esa parte. Aun así, ahora se usa SfM global (GLOMAP), que tarda la mitad que el incremental.
Al procesar, el programa también sube su prioridad en la CPU y en el planificador de GPU de Windows.

**Programas abiertos.** Windows reparte la GPU entre todos los programas. Un editor 3D (Unity, Blender),
un juego o un fondo de pantalla animado le quitan tiempo y memoria. La ventana avisa cuáles la están
usando. Si la GPU está muy ocupada, la detección SIFT pasa a la CPU, que en ese caso es más rápida.
Además, el procesamiento se ajusta a la memoria de vídeo realmente libre.

| Opción en la ventana | En la terminal | Qué hace |
|---|---|---|
| Automático (GPU si hay) | `--gpu auto` (predeterminado) | Usa la GPU si está disponible; si no hay o falla, continúa con la CPU |
| Tarjeta gráfica (GPU) | `--gpu si` | Exige la GPU y da error si no se puede usar |
| Solo CPU | `--gpu no` | No usa la GPU |

`Instalar.bat` instala los componentes GPU solo si detecta una NVIDIA: CuPy con NVRTC 12.8 o superior
(compatible con todas las series actuales, incluida la RTX 50) y COLMAP con CUDA (~400 MB). También se
pueden instalar después con el botón **Instalar** de la ventana o con `python -m fotogrametria --instalar-gpu`.
Necesita un controlador NVIDIA actualizado.

## Resultados

| Archivo | Contenido |
|---|---|
| `ortomosaico.tif` | Ortomosaico RGB + alfa, GeoTIFF en UTM con *overviews*; se abre en QGIS, ArcGIS o Google Earth Pro |
| `ortomosaico_todas_las_bandas.tif` | **Una sola imagen en 8 bits con todas las bandas**: R, G y B de la cámara RGB y las multiespectrales, en la misma rejilla. 0 = sin dato |
| `indices_vegetacion.tif` | NDVI, GNDVI, NDRE y LCI, una banda por índice (float32), calculados sobre la reflectancia |
| `ortomosaico_multiespectral.tif` | Bandas multiespectrales en reflectancia relativa (float32), para análisis |
| `ndvi_vista_previa.png` | Imagen del NDVI lista para compartir |
| `dem.tif` | Modelo digital de superficie (DSM) en metros |
| `nube_densa.laz` / `nube_densa.ply` | Nube de puntos densa con color; el LAZ incluye la "confianza" de cada punto |
| `camaras.csv` | Posición final de cada cámara |
| `informe.html` | Resumen de calidad, tiempos y vistas previas |

## Cómo funciona

El flujo sigue los pasos de Metashape (ver [ANALISIS_METASHAPE.md](ANALISIS_METASHAPE.md)):

1. **Cargar imágenes** (`metadata.py`): lee EXIF/XMP (GPS, cámara, banda espectral, radiometría).
2. **Alinear imágenes** (`alineacion.py`): SIFT, emparejamiento guiado por GPS y SfM global. Si el global deja imágenes fuera, se recurre al SfM incremental.
3. **Optimizar cámaras** (`alineacion.py`): georreferenciación Sim3 y ajuste de haces con observaciones GPS.
4. **Nube densa** (`densa.py`, en GPU `densa_gpu.py`): mapas de profundidad por pares rectificados con Semi-Global Matching, combinados por cámara y fusionados con un valor de confianza.
5. **DEM** (`dem.py`): rasterización con mediana por celda, filtro de picos e interpolación de huecos.
6. **Ortomosaico** (`ortomosaico.py`, en GPU `ortomosaico_gpu.py`): proyección sobre el DEM, selección de la foto más nadir y fusión multibanda de costuras.
7. **Guardar** (`exportar.py`): GeoTIFF, LAZ, PLY e informe.

Todos los parámetros se calculan automáticamente a partir de los datos: GSD, traslape, número
de imágenes, sensores y RAM disponible.

## Rendimiento medido

Prueba real con 59 fotos DJI Air 2S de 20 MP, vuelo a 50 m y 1,7 ha, en un Ryzen 7 5700X con 48 GB de
RAM y una RTX 5060 Ti de 8 GB. Calidad normal; mismo código en las dos columnas.

| Etapa | Solo CPU | Con GPU |
|---|---|---|
| Alineación y optimización de cámaras | 325 s | 110 s |
| Nube densa (~19-22 M puntos) | 96 s | 27 s |
| DEM | 8 s | 9 s |
| Ortomosaico (≈1,9 cm/píxel) | 31 s | 10 s |
| Exportación | 7 s | 14 s* |
| **Total** | **7,8 min** | **2,8 min** |

En las dos pruebas se alinearon 59/59 cámaras, con 0,75-0,76 px de error de reproyección y 0,17 m (XY)
de diferencia entre las cámaras y el GPS.

La prueba con GPU se hizo con el editor de Unity abierto, que usaba entre el 40 y el 50 % de la tarjeta.
Por eso la detección SIFT se hizo en la CPU (unos 60 s). Con la GPU libre, SIFT y emparejamiento tardan
unos 35 s en total. Además, la nube densa en GPU recupera algo más de superficie en los bordes del vuelo.

\* Incluye la nube para el visor 3D. Desde entonces se aceleró esa parte y ahora tarda unos 3 s.

En calidad **rápida**, el proceso completo con GPU tardó **1,0 min**.

**Medición posterior, con la GPU libre** (sin otros programas usándola) y con la fusión multivista de la
nube densa: en calidad normal, el proceso completo tardó **1,5 min**. En el mismo vuelo, Metashape en
calidad *Medium* tardó unos 4,2 min. La fusión multivista reduce el ruido de la nube en suelo desnudo de
20 cm a unos 5-7 cm; en la misma métrica, la nube de Metashape da unos 2 cm.

## Limitaciones

* Sin puntos de control (GCP): la precisión absoluta es la del GPS del dron, que suele ser de 1 a 3 m.
* El DEM es de superficie (DSM), con árboles y edificios; no se genera modelo de terreno (DTM).
* En multiespectral la salida es reflectancia **relativa**: consistente entre imágenes y bandas, válida para NDVI. Para reflectancia absoluta hace falta un panel de calibración.
* La aceleración por GPU solo funciona con tarjetas NVIDIA. Con otras tarjetas, o sin tarjeta, todo funciona con la CPU.
* El ajuste de haces se hace en la CPU.

## Cámaras soportadas

`fotogrametria/metadata.py` es el único lugar que interpreta las convenciones de cada fabricante.
Decide, para todo el flujo, el tipo de cada foto (RGB, banda multiespectral o térmica), la banda
normalizada (*Blue, Green, Red, RedEdge, NIR, Panchro*), qué fotos son del mismo disparo (por hora y
GPS) y qué cámara guía el proceso. El registro indica qué fotos se ignoran y por qué.

| Cámara | Qué se usa |
|---|---|
| DJI Phantom, Mavic, Air, Mini, Matrice (RGB) | RGB; en el Phantom 4 RTK, además, la calibración de fábrica |
| DJI P4 Multispectral, Mavic 3M | RGB principal y bandas B, G, R, RE, NIR |
| DJI M30T, H20T, M3T | Cámara gran angular; se ignoran el zoom y la térmica |
| MicaSense RedEdge, Altum, RedEdge-P | Bandas multiespectrales (la pancromática como principal); se ignora la térmica LWIR |
| Parrot Sequoia | RGB principal con su calibración de fábrica, y bandas G, R, RE, NIR |
| Sony, Hasselblad, senseFly y cámaras solo con EXIF | RGB; focal por tamaño de píxel si falta la de 35 mm |
| Sin GPS | Se procesa en coordenadas locales, sin georreferenciar |

Para comprobar que nada se rompió con un cambio, o al añadir una cámara nueva, ejecuta:

```bat
.venv\Scripts\python.exe -m pruebas.test_camaras
```

Crea fotos sintéticas con los metadatos exactos de cada cámara (nombres, canales, bits, EXIF y XMP) y
verifica la clasificación, la agrupación por disparo, la calibración y la lectura de imágenes.
Si una cámara nueva falla, añade su perfil a `pruebas/test_camaras.py` y corrige `metadata.py`.

Estas pruebas cubren la interpretación de los metadatos, que es donde se originan este tipo de errores.
No garantizan la calidad del resultado con fotos reales, que depende del vuelo, el traslape y la textura.

## Agricultura de precisión

Disponible en todas las calidades cuando hay imágenes multiespectrales.

* **Los índices se calculan sobre la reflectancia en float32**, nunca sobre los 8 bits.
* **Índices incluidos:** NDVI (vigor), GNDVI (clorofila), NDRE (nitrógeno y clorofila) y LCI (clorofila
  foliar). Son cocientes, válidos con reflectancia relativa. SAVI y EVI no se incluyen porque necesitan
  reflectancia absoluta (panel de calibración) y darían valores engañosos.
* **La imagen de 8 bits usa una sola escala lineal para todas las bandas multiespectrales**, guardada en
  la etiqueta `ESCALA_REFLECTANCIA`. Estirar cada banda por separado se vería más vistoso, pero rompería
  la proporción entre bandas. Medido en una Mavic 3M: un NDVI calculado desde los 8 bits difiere 0,004 de
  media del real. Como la escala es común, las bandas oscuras (verde, rojo) tienen valores bajos; en QGIS
  se ven bien con el estiramiento automático. Para combinarlas, usa falso color.
* **Radiometría:** en DJI se aplican nivel de negro, viñeteo, ganancia, exposición, sensor de luz solar y
  ajuste de ganancia entre bandas. MicaSense y Sequoia tienen modelos propios aún no implementados: sus
  índices pueden desviarse, y el registro lo avisa. Para valores absolutos hace falta un panel de calibración.

## Cámaras multiespectrales (sistema multicámara)

Una cámara multiespectral dispara todas sus lentes a la vez: 309 disparos de una Mavic 3M son 1545 fotos
(RGB, G, R, RE y NIR). El programa las trata como un **sistema multicámara**, igual que Metashape, y no
como fotos independientes (`rig.py`):

1. Se alinean solo las fotos de la cámara principal (la RGB; sin RGB, la pancromática o la verde).
2. En 16 disparos se mide una sola vez la rotación fija entre cámaras, con la calibración de fábrica de
   cada una. A cada banda se llega por el camino de pares con más coincidencias, porque emparejar
   directamente NIR y RGB es difícil.
3. Cada foto de banda hereda la pose de su disparo más esa rotación.

Medido en un vuelo de Mavic 3M, comparando la medición con dos grupos de disparos distintos: las bandas
multiespectrales coinciden entre sí con 0,07-0,23 px. El bloque multiespectral puede diferir ~1 px de la
RGB (unos 4 cm), lo que no afecta a índices como el NDVI, calculados solo con bandas multiespectrales.
Si las cámaras no disparan juntas, se alinean todas las fotos como antes.

## Cámaras con particularidades (se tratan automáticamente)

* **Calibración de fábrica:** si la foto la trae en sus metadatos (Parrot Sequoia; DJI con
  `DewarpData`), se usa y se mantiene fija. En terreno plano con fotos verticales, la focal no se puede
  estimar y tiende a derivar.
* **Retraso del GPS:** si las cámaras quedan sistemáticamente por delante o por detrás de su GPS en la
  dirección de vuelo, se corrige. Se midió 0,4-0,5 intervalos entre fotos en una Sequoia.
* **Exposición variable:** si el brillo salta de una foto a sus vecinas (parches en el ortomosaico), se
  igualan las ganancias. Las diferencias por zonas, como bosque frente a suelo claro, se respetan.
* **Sin altura relativa** en los metadatos: los pares de fotos se buscan por cercanía GPS, sin límite de
  distancia.

## Consejos para mejores resultados

* Vuela con un traslape de **70-80 % frontal y 60-70 % lateral**.
* Usa luz homogénea; evita sombras cambiantes por nubes.
* En multiespectral, incluye las imágenes del sensor de sol. Si tienes fotos del panel de calibración, déjalas fuera de la carpeta.

## Pausar, cancelar y registro de errores

Durante el proceso puedes **pausar y reanudar** (no gasta CPU ni GPU, pero la memoria sigue reservada y el avance se pierde si cierras el programa) o **cancelar**, que borra los temporales.

Cada procesamiento guarda un registro en `%LOCALAPPDATA%\OrthoDash\registros` (se conservan los últimos 20). Nunca se envía solo. Si algo falla, pulsa **Guardar registro de errores**; también está en **Avanzado > Guardar el último registro**. Incluye el equipo, las opciones, los mensajes y las rutas de las carpetas.

## Radiometría multiespectral

| Cámara | Resultado | Estado |
|---|---|---|
| DJI Mavic 3M, P4 Multispectral | Reflectancia relativa | Mavic 3M verificado con un vuelo real |
| MicaSense RedEdge, Altum | Reflectancia con sensor DLS | Fórmula verificada con fotos reales de Altum |
| Parrot Sequoia | Valores comparables dentro del vuelo | Sin verificar con archivos reales |
| Otras cámaras | Sólo negro, exposición e ISO | Índices orientativos |
