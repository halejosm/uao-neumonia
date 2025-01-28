import numpy as np
from load_model import model_fun
from grad_cam import generate_grad_cam
from preprocess_img import preprocess_image

def predict_image(array):
    """
    Realiza la predicción de una imagen usando un modelo preentrenado y genera un mapa de calor Grad-CAM.

    Parámetros:
        array (numpy.ndarray): Imagen de entrada en forma de arreglo numpy.

    Retorna:
        tuple: Clase predicha, probabilidad de la predicción, y el mapa de calor Grad-CAM.
    """
    img = preprocess_image(array)
    model = model_fun()
    preds = model.predict(img)
    label_map = {0: "bacteriana", 1: "normal", 2: "viral"}
    prediction = np.argmax(preds[0])
    probability = np.max(preds[0]) * 100
    heatmap = generate_grad_cam(model, array)
    return label_map[prediction], probability, heatmap