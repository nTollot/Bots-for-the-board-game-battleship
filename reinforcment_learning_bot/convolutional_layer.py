import tensorflow as tf
from tensorflow import keras
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.activations import relu


class ConvolutionalLayer(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv2d = Conv2D(
            filters=256,
            kernel_size=3,
            padding="same",
            name="conv2d")
        self.batch_normalisation = BatchNormalization()

    def call(self, input):
        layer = self.conv2d(input)
        layer = self.batch_normalisation(layer)
        layer = relu(layer)
        return layer
