import pydicom
from PIL import Image
import cv2
import numpy as np

def read_image_file(filepath):
    """
    Lee un archivo de imagen y devuelve su representación RGB y un objeto PIL Image.

    Parámetros:
        filepath (str): Ruta al archivo de imagen (soporta .dcm u otros formatos de imagen).

    Retorna:
        tuple: Una tupla que contiene la imagen RGB como un arreglo numpy y un objeto PIL Image.
    """
    if filepath.lower().endswith(".dcm"):
        dicom = pydicom.dcmread(filepath)
        img_array = dicom.pixel_array
        img_rgb = cv2.cvtColor(
            (img_array / np.max(img_array) * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB
        )
    else:
        img_rgb = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

    return img_rgb, Image.fromarray(img_rgb)