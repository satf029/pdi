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

image_inverted = cv2.bitwise_not(mask_bin)
marker = generate_marker(image_inverted)
# Llamar a la función de reconstrucción morfológica
reconstructed = morphological_reconstruction(marker,image_inverted)
imagen_A =image_inverted - (reconstructed*255)
imagen_A= cv2.bitwise_not(imagen_A)
# Convertir la imagen reconstruida a formato 8-bit para visualización
# reconstructed = (reconstructed * 255).astype(np.uint8)
imagen_A_inverted = cv2.bitwise_not(imagen_A)
marker_b = generate_marker(imagen_A_inverted)

reconstructed_2 = morphological_reconstruction(marker_b,imagen_A)
reconstructed_2 = (reconstructed_2 * 255).astype(np.uint8)
imagen_B = cv2.bitwise_xor(imagen_A_inverted, reconstructed_2)

#parte 3
# Invertir imagen B (núcleos) para que estén en blanco como semillas
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
imagen_C_inverted = cv2.bitwise_not(imagen_C)
marker_e = generate_marker(imagen_C_inverted)
reconstructed_e = morphological_reconstruction(marker_e, imagen_C)

reconstructed_e = morphological_reconstruction(reconstructed_e, imagen_C_inverted)
imagen_E_inv = reconstructed_e * 255
imagen_ = cv2.bitwise_xor(imagen_E_inv, imagen_C)
imagen_E = cv2.bitwise_not(reconstructed_e*255)

# print(np.unique(marker))
# print(imagen_A)
# print(np.unique(reconstructed_2))


cv2.imshow('C',imagen_C)
cv2.imshow('C invertida', imagen_C_inverted)
cv2.imshow('E invertida', imagen_E_inv)
cv2.imshow('imagen_', imagen_)




# Mostrar las imágenes
# cv2.imshow('Original', mask_bin)
# cv2.imshow('Imagen A', imagen_A)
# cv2.imshow('Imagen B', imagen_B)
# cv2.imshow('Imagen C', imagen_C)
# cv2.imshow('Imagen D', imagen_D)
# cv2.imshow('Imagen E', imagen_E)



cv2.waitKey(0)
cv2.destroyAllWindows()
