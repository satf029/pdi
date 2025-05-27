
README - Análisis Morfológico de Células

Autores: Samuel Torres y Edis Fernandez

Este programa realiza un análisis morfológico sobre una imagen binaria de células. A través de técnicas de reconstrucción morfológica, logra separar los distintos tipos de células en base a sus características estructurales. A continuación, se describe cada parte del proceso.

---------------------------------------------------------------------------------------------------

1. Importación de librerías:
   - numpy: operaciones numéricas.
   - scipy.ndimage.binary_dilation: dilatación para la reconstrucción morfológica.
   - skimage.measure.label, regionprops: para etiquetar y analizar regiones.
   - cv2 (OpenCV): carga, procesamiento y visualización de imágenes.

---------------------------------------------------------------------------------------------------

2. Función `generate_marker`:
   - Crea un marcador tomando los bordes de la imagen invertida.
   - Este marcador es necesario para iniciar la reconstrucción morfológica desde los bordes.

3. Función `morphological_reconstruction`:
   - Realiza una reconstrucción morfológica usando dilatación binaria iterativa.
   - Rellena objetos conectados al marcador, sin exceder la máscara original.

---------------------------------------------------------------------------------------------------

4. Parte 1: Eliminación de objetos conectados al borde.
   - Se invierte la imagen binaria para resaltar las células como objetos blancos.
   - Se genera un marcador desde el borde y se reconstruye la imagen.
   - Se resta esta reconstrucción para obtener las células completamente internas (Imagen A).

---------------------------------------------------------------------------------------------------

5. Parte 2: Aislamiento de núcleos celulares.
   - Inversión de Imagen A.
   - Reconstrucción desde el borde para extraer siluetas (citoplasmas).
   - Se usa XOR con la imagen invertida para obtener solo los núcleos (Imagen B).

---------------------------------------------------------------------------------------------------

6. Parte 3: Reconstrucción de citoplasmas desde los núcleos.
   - Imagen B se invierte para usar como marcador.
   - Se reconstruyen los citoplasmas completos dentro de los núcleos (Imagen C).

---------------------------------------------------------------------------------------------------

7. Parte 4: Extracción del citoplasma exclusivamente.
   - Se realiza una operación XOR entre Imagen C e Imagen A.
   - Se invierte para dejar solo el citoplasma sin el núcleo (Imagen D).

---------------------------------------------------------------------------------------------------

8. Parte 5: Relleno del núcleo para destacar estructuras internas.
   - Se vuelve a reconstruir Imagen C invertida.
   - Se rellena para aislar los núcleos (Imagen E).

---------------------------------------------------------------------------------------------------

9. Parte 6: Aislamiento del núcleo de tipo 4 (núcleo blanco con citoplasma negro).
   - Se reconstruye primero con Imagen E invertida y luego con Imagen C invertida.
   - El resultado es Imagen F, con solo los núcleos Tipo 4.

---------------------------------------------------------------------------------------------------

10. Parte 7: Diferencia entre citoplasmas y núcleos.
   - XOR entre Imagen A e Imagen D y luego con Imagen F.
   - El resultado es Imagen G, con los núcleos blancos sobre citoplasmas negros.

---------------------------------------------------------------------------------------------------

11. Parte 8: Clasificación de células.
   - Se etiquetan las regiones de Imagen G.
   - Se identifican los huecos (núcleos) dentro de cada célula.
   - Si el hueco es grande (>15 px), se clasifica como Tipo 3, si no, como Tipo 2.

   Resultado:
   - Se muestran la cantidad de células de cada tipo.
   - Se visualiza una imagen final con las células tipo 2 en gris (100) y tipo 3 más claras (200).

---------------------------------------------------------------------------------------------------

12. Visualización:
   - Se muestran todas las etapas del proceso para validación visual.

Opcionalmente se pueden guardar todas las imágenes intermedias como archivos JPG.

---------------------------------------------------------------------------------------------------

Requisitos:
- Python 3.x
- OpenCV
- SciPy
- scikit-image
- NumPy

