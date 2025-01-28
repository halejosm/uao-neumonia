import numpy as np
from load_model import model_fun
from grad_cam import generate_grad_cam
from preprocess_img import preprocess_image

def predict_image(array):
    img = preprocess_image(array)                           #call function to pre-process image: it returns image in batch format
    model = model_fun()                                     #call function to load model and predict: it returns predicted class and probability
    preds = model.predict(img)
    label_map = {0: "bacteriana", 1: "normal", 2: "viral"}
    prediction = np.argmax(preds[0])
    probability = np.max(preds[0]) * 100
    heatmap = generate_grad_cam(model, array)                      #call function to generate Grad-CAM: it returns an image with a superimposed heatmap
    return label_map[prediction], probability, heatmap