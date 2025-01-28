import tensorflow as tf

def model_fun():
    return tf.keras.models.load_model('conv_MLP_84.h5')