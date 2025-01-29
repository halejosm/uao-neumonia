import tensorflow as tf

class Predictor:
    def __init__(self):
        self.image_array = None
        self.label = ""
        self.probability = 0.0
        self.heatmap = None
