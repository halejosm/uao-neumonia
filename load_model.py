import tensorflow as tf

def load_model():
    model = tf.keras.models.load_model('conv_MLP_84.h5')
    return model
