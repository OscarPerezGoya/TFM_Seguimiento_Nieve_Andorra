# Evolución de la Presencia de Nieve en el Suelo en Andorra

## Introducción
Trabajo Final de Máster del Master de Ciencia de Datos de la Universidad Abierta de Cataluña, titulado "Evolución de la Presencia de Nieve en el Suelo en Andorra" redactado por Óscar Pérez Goya en diciembre de 2025.

El trabajo consiste en la elaboración de una cartografía en la que se clasifica, para el territorio andorrano la superficie en las categorías siguientes: valores nulos/nieve/no nieve/nubes remanentes, durante el periodo invernal a lo largo de una serie de años (desde abril de 2013 hasta junio de 2025). 

La información satelital se obtiene de los sonsores instalados en los satélites Sentinel-2 y la familia de satélites Landsat. El acceso a los datos se ha realizado mediante la plataforma de Google Earth Engine.

## Estructura del repositorio 
El respositorio se estructura de la siguiente forma:

- codigo/: notebook de Google Colab con el código generado 
- data/: datasets generados en formato csv que contienen la información de los satélites disponibles para el periodo de estudio, la información de procesado de cada una de las imágenes junto con las principales estadísticas asociadas a cada imagen
- mapas_nieve/: Esta carpeta incluye las imágenes resultantes clasificadas (valores nulos/nieve/no nieve/nubes remanentes) en formato GeoTiff.
- snow_coverage/: Incluye el archivo a sustituir para lanzar el código de "Ínidice de nieve con Deep Learning SonwCoverage" (make_prediction.py) y el resultado (mapa de nieve/no nieve) resultante para la zona de estudio. 

## Herramientas utilizadas
El código se ha generado en Google Colab utilizando las librerías que se incluyen en el notebook facilitado. 

Los datos satelitales así como el preprocesado se ha realizado con la herramienta de Google Earth Engine. Además, los datos obtenidos (imágenes y datasets) se han guardado en Drive de Google personal.

## Instrucciones de uso

Para el correcto funcionamiento del código es necedario conocer que:

- el código necesita que el usuario esté autenticado en la plataforma de Google Earth Engine para poder acceder y procesar los datos satelitales. Las rutas de guardado de los datos (datasets y mapas resultantes) son de Google Drive y pueden ser modificadas en el código para guardar los resultados en otros directorios.
- Para el lanzamiento del código del apartado "Ínidice de nieve con Deep Learning SonwCoverage", se recomienda el uso de GPU. Además, se debe sustiuir el archivo   make_prediction.py que se instala del repositorio facilitado por los  autores del algoritmo SnowCoverage por el archivo make_prediction.py que se incluye en la carpeta snow_coverage_DL.

## Resultados obtenidos:

Los resultados obtenidos son:
- la serie de mapas clasificados sobre Andorra con los valores nulos/nieve/no nieve/nubes remanentes) en formato GeoTiff
- el dataset de imágenes satelitales disponibles para el periodo invernal y la serie de años tratada: Andorra_Sentinel_Landsat_Invierno.csv
- los datasets de registro de las imágenes obtenidas (un dataset por cada temporada invernal): Tabla_Nubes_Snowline_<Temporada>.csv
- el dataset con la información estadística de cada imagen satelital procesada: estadisticas_mapas_nieve.csv
- el dataset con la información estadística agregada por temporadas :  estadisticas_agregadas.csv
- Dashboard que permite la visualizacion de los mapas clasificados y diversas gráficas con los valores obtenidos.

## Licencia

Véase archivo LICENSE.md 






