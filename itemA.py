# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 11:46:12 2024

@author: jvazquez
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
print(np.unique(marker_b))
reconstructed_2 = morphological_reconstruction(marker_b,imagen_A)
reconstructed_2 = (reconstructed_2 * 255).astype(np.uint8)
reconstructed_inv = cv2.bitwise_not(reconstructed)
imagen_B = cv2.bitwise_xor(imagen_A_inverted, reconstructed_2)
# Mostrar las imágenes
cv2.imshow('Original', mask_bin)
cv2.imshow('Imagen A', imagen_A)
cv2.imshow('Imagen B', imagen_B)
cv2.waitKey(0)
cv2.destroyAllWindows()
