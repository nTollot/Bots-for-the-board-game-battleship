import tensorflow as tf
from tensorflow import keras
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.activations import relu
from keras.activations import tanh


class ValueHead(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv2d = Conv2D(
            filters=1,
            kernel_size=1,
            padding="same",
            name="conv2d")
        self.batch_normalisation = BatchNormalization()
        self.dense_1 = Dense(units=256, name="dense/1")
        self.dense_2 = Dense(units=1, name="dense/2")

    def call(self, input):
        layer = self.conv2d(input)
        layer = self.batch_normalisation(layer)
        layer = Flatten()(layer)
        layer = relu(layer)
        layer = self.dense_1(layer)
        layer = relu(layer)
        layer = self.dense_2(layer)
        layer = tanh(layer)
        return layer
