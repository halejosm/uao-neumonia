import numpy as np
from load_model import model_fun
from grad_cam import generate_grad_cam
from preprocess_img import preprocess_image

def predict_image(array):
    img = preprocess_image(array)                           # Llamar la función pre-procesamiento de imagen: retorna imagen 
    model = model_fun()                                     # Llamar la función para carga modelo y prediccíon: retorna la clase predicción y probabilidad.
    preds = model.predict(img)
    label_map = {0: "bacteriana", 1: "normal", 2: "viral"}
    prediction = np.argmax(preds[0])
    probability = np.max(preds[0]) * 100
    heatmap = generate_grad_cam(model, array)               # Llama la función para generar el Grad-CAM: devuelve una imagen con un mapa de calor superpuesto.
    return label_map[prediction], probability, heatmap