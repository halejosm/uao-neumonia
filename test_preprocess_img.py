import pytest
import numpy as np

# Importar la funcion que se va realizar el test unitario
from preprocess_img import preprocess_image

def test_preprocess_image():
    # Generar un array aleatorio para probar preprocess_image
    array = np.random.randint(0, 256, (256, 256, 3), dtype = np.uint8)

    # Invocar la funcion que se va a hacer la prueba unitaria
    resultado = preprocess_image(array)

    mensaje_shape = f"El shape correcto es: (1, 512, 512, 1) y el shape de la prueba es: {resultado.shape}"

    # El shape correcto debe ser (1, 512, 512, 1)
    assert resultado.shape == (1, 512, 512, 1), mensaje_shape

    mensaje_rango = f"El rango de valores correcto es entre 0.0 y 1.0, val min de la prueba es: {np.min(resultado)} y el val max es: {np.max(resultado)}"

    # El rango de valores del resultado debe ser entre 0 y 1
    assert np.min(resultado) >= 0.0 and np.max(resultado) <= 1.0, mensaje_rango