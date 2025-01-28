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
from preprocess_img import preprocess
from grad_cam import grad_cam

def predict(array):
    #   1. call function to pre-process image: it returns image in batch format
    batch_array_img = preprocess(array)
    #   2. call function to load model and predict: it returns predicted class and probability
    model = model_fun()
    #model = tf.keras.models.load_model('conv_MLP_84.h5')
    # model_cnn = tf.keras.models.load_model('conv_MLP_84.h5')
    prediction = np.argmax(model.predict(batch_array_img))
    proba = np.max(model.predict(batch_array_img)) * 100
    label = ""
    if prediction == 0:
        label = "bacteriana"
    if prediction == 1:
        label = "normal"
    if prediction == 2:
        label = "viral"
    #   3. call function to generate Grad-CAM: it returns an image with a superimposed heatmap
    heatmap = grad_cam(array)
    return (label, proba, heatmap)