from preprocess_img import preprocess
from load_model import load_model
from grad_cam import generate_grad_cam

def predict(array):
    # Preprocesar la imagen
    batch_array_img = preprocess(array)
    # Cargar el modelo y obtener predicción
    model = load_model()
    prediction = model.predict(batch_array_img)
    class_id = prediction.argmax()
    probability = prediction.max() * 100
    label = ["bacteriana", "normal", "viral"][class_id]
    # Generar el mapa de calor Grad-CAM
    heatmap = generate_grad_cam(array, model)
    return label, probability, heatmap
