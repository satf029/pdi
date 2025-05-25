# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 11:46:12 2024

@author: jvazquez
"""

import numpy as np
from scipy.ndimage import binary_dilation
import cv2


def generate_marker(binary_image):
    inverted_image = 1-binary_image
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
_, mask_bin = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

image_inverted = 1-mask_bin
marker = generate_marker(image_inverted)
# Llamar a la función de reconstrucción morfológica
reconstructed = morphological_reconstruction(marker,1-mask_bin)
imagen_A =(1-mask_bin) - reconstructed
imagen_A= 1-imagen_A
# Convertir la imagen reconstruida a formato 8-bit para visualización
# reconstructed = (reconstructed * 255).astype(np.uint8)
imagen_A_invertida = 1-imagen_A
marker_b = generate_marker(imagen_A_invertida)
reconstructed_2 = morphological_reconstruction(marker_b,imagen_A_invertida)

# Mostrar las imágenes
cv2.imshow('Original', mask_bin*255)
cv2.imshow('Imagen A', imagen_A*255)
cv2.imshow('Imagen Invertida',reconstructed_2*255)
cv2.waitKey(0)
cv2.destroyAllWindows()
