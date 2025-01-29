import numpy as np
import cv2
from tensorflow.keras import backend as K  # type: ignore
from src.data.preprocess_img import preprocess_image

def generate_grad_cam(model, array):
    """
    Genera un mapa de Grad-CAM para la imagen proporcionada.

    Args:
        model: El modelo utilizado para generar el mapa de Grad-CAM.
        array: La imagen de entrada en formato de matriz numpy.

    Returns:
        Imagen con el mapa de calor superpuesto.
    """
    img = preprocess_image(array)
    preds = model.predict(img)
    argmax = np.argmax(preds[0])
    output = model.output[:, argmax]
    last_conv_layer = model.get_layer("conv10_thisone")
    grads = K.gradients(output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate(img)

    for filters in range(conv_layer_output_value.shape[-1]):
        conv_layer_output_value[:, :, filters] *= pooled_grads_value[filters]

    # Creando mapa de calor
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    heatmap = cv2.resize((heatmap * 255).astype(np.uint8), (512, 512))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img2 = cv2.resize(array, (512, 512))
    hif = 0.8
    transparency = heatmap * hif
    transparency = transparency.astype(np.uint8)
    superimposed_img = cv2.add(transparency, img2)
    superimposed_img = superimposed_img.astype(np.uint8)

    return superimposed_img[:, :, ::-1]
