import pydicom
from PIL import Image
import cv2
import numpy as np

def preprocess_image(array):
    """
    Preprocesa una imagen para normalizarla y ajustarla al formato requerido.

    Parámetros:
        array (numpy.ndarray): Arreglo numpy que representa la imagen.

    Retorna:
        numpy.ndarray: Imagen preprocesada con dimensiones ajustadas y normalizada.
    """
    array = cv2.resize(array, (512, 512))
    array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    array = clahe.apply(array) / 255
    return np.expand_dims(np.expand_dims(array, axis=-1), axis=0)
