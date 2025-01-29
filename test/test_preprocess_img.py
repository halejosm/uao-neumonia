import numpy as np

# Importar la función que se va a realizar el test unitario
from src.data.preprocess_img import preprocess_image

def test_preprocess_image():
    """
    Prueba unitaria para la función preprocess_image.
    
    Verifica que la salida tenga el shape correcto y que los valores estén
    dentro del rango esperado.
    """
    # Generar un array aleatorio para probar preprocess_image
    array = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

    # Invocar la función que se va a probar
    resultado = preprocess_image(array)

    # Verificar que la forma de la salida sea la esperada
    shape_correcto = (1, 512, 512, 1)
    mensaje_shape = (
        f"El shape correcto es: {shape_correcto} y el shape de la prueba es: {resultado.shape}"
    )
    assert resultado.shape == shape_correcto, mensaje_shape

    # Verificar que los valores estén en el rango esperado
    valor_min = np.min(resultado)
    valor_max = np.max(resultado)
    mensaje_rango = (
        f"El rango de valores correcto es entre 0.0 y 1.0, "
        f"val min de la prueba es: {valor_min} y el val max es: {valor_max}"
    )
    assert 0.0 <= valor_min <= 1.0 and 0.0 <= valor_max <= 1.0, mensaje_rango
