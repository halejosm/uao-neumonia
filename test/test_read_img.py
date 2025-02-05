from unittest.mock import MagicMock
import numpy as np

# Importar la función que se va a realizar el test unitario
from src.data.read_img import read_image_file

def test_read_image_file_dcm(mocker):
    """
    Prueba unitaria para la función read_image_file con archivos DICOM.
    
    Se mockea la función pydicom.dcmread para simular la lectura de un archivo DICOM
    y se verifica que la salida sea la esperada en formato RGB y PIL.
    """
    # Mock para el resultado de pydicom.dcmread
    mock_dicom = MagicMock()

    # Mockear el atributo pixel_array
    mock_dicom.pixel_array = np.array([[0, 1], [2, 3]])

    # Mockear la función pydicom.dcmread
    mocker.patch('pydicom.dcmread', return_value=mock_dicom)

    # Invocar la función que se va a probar
    resultado_img_rgb, resultado_img_pil = read_image_file('imagen_1.dcm')

    esperada_img_rgb = np.array([
        [[0, 0, 0], [85, 85, 85]],
        [[170, 170, 170], [255, 255, 255]]
    ])

    mensaje_array = "Las imágenes RGB de resultado y esperada no son iguales"

    # Verificar si las imágenes son iguales
    assert np.array_equal(resultado_img_rgb, esperada_img_rgb), mensaje_array
    
    mensaje_size = (
        f"El size correcto es: (2, 2) y el size de la prueba es: {resultado_img_pil.size}"
    )

    # El size correcto debe ser (2, 2)
    assert resultado_img_pil.size == (2, 2), mensaje_size

def test_read_image_file_jpg(mocker):
    """
    Prueba unitaria para la función read_image_file con archivos JPG.
    
    Se mockea la función cv2.imread para simular la lectura de un archivo JPG
    y se verifica que la salida sea la esperada en formato RGB y PIL.
    """
    # Generar un array aleatorio para mockear el resultado de cv2.imread
    array = np.array([
        [[0, 0, 0], [255, 255, 255]],
        [[255, 255, 255], [255, 255, 255]]
    ], dtype=np.uint8)

    # Mockear la función cv2.imread
    mocker.patch('cv2.imread', return_value=array)

    # Invocar la función que se va a probar
    resultado_img_rgb, resultado_img_pil = read_image_file('imagen_2.jpg')

    esperada_img_rgb = np.array([
        [[0, 0, 0], [255, 255, 255]],
        [[255, 255, 255], [255, 255, 255]]
    ], dtype=np.uint8)

    mensaje_array = "Las imágenes RGB de resultado y esperada no son iguales"

    # Verificar si las imágenes son iguales
    assert np.array_equal(resultado_img_rgb, esperada_img_rgb), mensaje_array

    mensaje_size = (
        f"El size correcto es: (2, 2) y el size de la prueba es: {resultado_img_pil.size}"
    )

    # El size correcto debe ser (2, 2)
    assert resultado_img_pil.size == (2, 2), mensaje_size
