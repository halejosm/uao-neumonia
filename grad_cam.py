import numpy as np
import cv2
from detector_neumonia import preprocess
from load_model import model_fun
from tensorflow.keras import backend as K # type: ignore

def generate_grad_cam(model, array):
    img = preprocess(array)
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

    # creating the heatmap
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
