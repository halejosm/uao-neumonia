import pydicom
from PIL import Image
import cv2
import numpy as np

def read_dicom_file(path):
    img = pydicom.dcmread(path)
    img_array = img.pixel_array
    img_array_normalized = (img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255.0
    img_array_normalized = np.uint8(img_array_normalized)
    img_RGB = cv2.cvtColor(img_array_normalized, cv2.COLOR_GRAY2RGB)
    return img_RGB, Image.fromarray(img_array)

def read_jpg_file(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Error al cargar la imagen: {path}. Verifica que el archivo sea válido.")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb, Image.fromarray(img_rgb)
