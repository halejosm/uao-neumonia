import tensorflow as tf

# Bloque separado para cargar el modelo
def model_fun():
    """
    Carga un modelo preentrenado de TensorFlow guardado en un archivo .h5.

    Retorna:
        tensorflow.keras.Model: El modelo cargado.
    """
    return tf.keras.models.load_model('conv_MLP_84.h5')