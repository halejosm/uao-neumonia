#from preprocess_img import preprocess
#from load_model import load_model
#from grad_cam import generate_grad_cam
#
#def predict(array):
#    # Preprocesar la imagen
#    batch_array_img = preprocess(array)
#    # Cargar el modelo y obtener predicción
#    model = load_model()
#    prediction = model.predict(batch_array_img)
#    class_id = prediction.argmax()
#    probability = prediction.max() * 100
#    label = ["bacteriana", "normal", "viral"][class_id]
#    # Generar el mapa de calor Grad-CAM
#    heatmap = generate_grad_cam(array, model)
#    return label, probability, heatmap

import numpy as np
from load_model import model_fun
from grad_cam import generate_grad_cam
from preprocess_img import preprocess_image

def predict(array):
    img = preprocess_image(array)                           #call function to pre-process image: it returns image in batch format
    model = model_fun()                                     #call function to load model and predict: it returns predicted class and probability
    preds = model.predict(img)
    label_map = {0: "bacteriana", 1: "normal", 2: "viral"}
    prediction = np.argmax(preds[0])
    probability = np.max(preds[0]) * 100
    heatmap = generate_grad_cam(array)                      #call function to generate Grad-CAM: it returns an image with a superimposed heatmap
    return label_map[prediction], probability, heatmap