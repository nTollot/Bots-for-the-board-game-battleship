import tensorflow as tf
from tensorflow import keras
from keras.layers import Add
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.activations import relu


class ResidualLayer(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv2d_1 = Conv2D(
            filters=256,
            kernel_size=3,
            padding="same",
            name="conv2d/1")
        self.conv2d_2 = Conv2D(
            filters=256,
            kernel_size=3,
            padding="same",
            name="conv2d/2")
        self.batch_normalisation_1 = BatchNormalization()
        self.batch_normalisation_2 = BatchNormalization()

    def call(self, input):
        layer = self.conv2d_1(input)
        layer = self.batch_normalisation_1(layer)
        layer = relu(layer)
        layer = self.conv2d_2(input)
        layer = self.batch_normalisation_2(layer)
        layer = Add()([layer, input])
        layer = relu(layer)
        return layer
