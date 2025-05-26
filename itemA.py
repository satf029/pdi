# -*- coding: utf-8 -*-
"""
@author: Samuel Torres y Edis Fernandez
"""

import numpy as np
from scipy.ndimage import binary_dilation
import cv2


def generate_marker(binary_image):
    inverted_image =cv2.bitwise_not(binary_image)
    # Crear la imagen marcador
    marker = np.zeros_like(binary_image)
    marker[0, :] = inverted_image[0, :]
    marker[-1, :] = inverted_image[-1, :]
    marker[:, 0] = inverted_image[:, 0]
    marker[:, -1] = inverted_image[:, -1]

    return marker


def morphological_reconstruction(marker, mask):
    if marker.shape != mask.shape:
        raise ValueError("Las dimensiones de marker y mask deben ser iguales.")

    reconstructed = marker.copy()

    while True:
        previous = reconstructed.copy()
        dilated = binary_dilation(reconstructed, structure=np.ones((3, 3)))
        reconstructed = dilated & mask
        if np.array_equal(reconstructed, previous):
            break

    return reconstructed

mask = cv2.imread('pdi.bmp', 0)
# Convertir las imágenes a binarias
_, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

#Invertimos la imagen para hacer que el objeto sea 255
image_inverted = cv2.bitwise_not(mask_bin)

#generamos el marcador, que rellenara la imagen desde los bordes
marker = generate_marker(image_inverted)
# Llamar a la función de reconstrucción morfológica
#rellena la imagen desde el marcador, y retorna los objetos recuperados gracias al relleno
reconstructed = morphological_reconstruction(marker,image_inverted)

#como reconstructed tiene las celulas que estan en los bordes
# almacenamos en la variable imagen_A para que solo tenga las celulas que no estan en
# los bordes
imagen_A =image_inverted - (reconstructed*255)
#invertimos para volver a la escala de la imagen original
imagen_A= cv2.bitwise_not(imagen_A)
##################################################################
# TEMA 2
#####################################################################
#nuevamente hacemos que las celulas esten pintados con blanco
# es decir, hacemos que el objeto este con 255
imagen_A_inverted = cv2.bitwise_not(imagen_A)
# generamos el marcador que rellenara desde los bordes
marker_b = generate_marker(imagen_A_inverted)
# Llama a la funcion de reconstruccion morfologica
# rellena la imagen_A ( fondo blanco y celulas negras) desde los bordes para obtener las
# siluetas de las celulas
reconstructed_2 = morphological_reconstruction(marker_b,imagen_A)
reconstructed_2 = (reconstructed_2 * 255).astype(np.uint8)
# como la diferencia de la imagen de reconstructed con la imagen_A_invertida es
# que reconstructed solo tiene la siluetas en blanco y el fondo en negro
#no contiene el citoplasma de la celula
# Por eso se aplica xor para que pinte en blanco los nucleos de las celulas
imagen_B = cv2.bitwise_xor(imagen_A_inverted, reconstructed_2)

#parte 3
# Invertir imagen B (núcleos) para que estén en blanco como semillas
# utilizaremos la imagen B como marcador
imagen_B_inv = cv2.bitwise_not(imagen_B)

# Reconstrucción: expandimos desde núcleos dentro de citoplasmas
reconstructed_C = morphological_reconstruction(imagen_B_inv, imagen_A_inverted)

# Convertir a imagen visible
imagen_C_binaria = (reconstructed_C * 255).astype(np.uint8)

# Invertimos para que los objetos sean negros sobre fondo blanco
imagen_C = cv2.bitwise_not(imagen_C_binaria)

#parte 4

imagen_D = cv2.bitwise_xor(imagen_C, imagen_A)
imagen_D = cv2.bitwise_not(imagen_D)

#parte 5
# hacemos que los objetos a analizar sean 255
imagen_C_inverted = cv2.bitwise_not(imagen_C)
#generamos el marcador de donde se va a rellenar
marker_e = generate_marker(imagen_C_inverted)
#reconstuimos desde el marcador
#obtenemos de nuevo solo las siluetas pero con fondo blanco y siluetas negras
reconstructed_e = morphological_reconstruction(marker_e, imagen_C)
#usamos la imagen obtenida (fondo blanco y celulas negras) como mascara
# para rellenar en blanco todo el citoplasma de la celula en blanco
reconstructed_e = morphological_reconstruction(reconstructed_e, imagen_C_inverted)
# reconstruccion con fondo negro con celulas blancas y el citoplasma completo en negro
# solo el nucleo difiere en el color de imagen_E_inv y de la imagen_C_invertida
imagen_E_inv = cv2.bitwise_xor(reconstructed_e*255, imagen_C_inverted)
imagen_E = cv2.bitwise_not(imagen_E_inv)




# Mostrar las imágenes
cv2.imshow('Original', mask_bin)
cv2.imshow('Imagen A', imagen_A)
cv2.imshow('Imagen B', imagen_B)
cv2.imshow('Imagen C', imagen_C)
cv2.imshow('Imagen D', imagen_D)
cv2.imshow('Imagen E', imagen_E)



cv2.waitKey(0)
cv2.destroyAllWindows()
