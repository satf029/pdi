# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 11:46:12 2024

@author: jvazquez
"""

import numpy as np
from scipy.ndimage import binary_dilation
import cv2


def generate_marker(binary_image):
    # Invertir la imagen binaria
    inverted_image = cv2.bitwise_not(binary_image)

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
marker = generate_marker(mask_bin)
# Llamar a la función de reconstrucción morfológica
reconstructed = morphological_reconstruction(marker,1-mask_bin)
cleaned =(1-mask_bin) - reconstructed
cleaned= 1-cleaned
# Convertir la imagen reconstruida a formato 8-bit para visualización
reconstructed = (reconstructed * 255).astype(np.uint8)


# Mostrar las imágenes
cv2.imshow('Marcador', marker)
cv2.imshow('Imagen', image_inverted*255)
cv2.imshow('Reconstruccion',cleaned*255)
cv2.waitKey(0)
cv2.destroyAllWindows()
