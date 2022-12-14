from numpy import inf
import tensorflow as tf
from tensorflow import keras
from keras.layers import Add
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Multiply
from keras.activations import relu
from keras.activations import softmax


class PolicyHead(keras.Model):
    def __init__(self, n_actions, **kwargs):
        super().__init__(**kwargs)
        self.conv2d = Conv2D(
            filters=2,
            kernel_size=1,
            padding="same",
            name="conv2d")
        self.batch_normalisation = BatchNormalization()
        self.dense = Dense(units=n_actions, name="dense")

    def call(self, input):
        value_input, mask_input = input
        layer = self.conv2d(value_input)
        layer = self.batch_normalisation(layer)
        layer = Flatten()(layer)
        layer = relu(layer)
        layer = self.dense(layer)
        inf_mask = -relu(inf*(1-2*mask_input))
        output = Multiply()([layer, mask_input])
        output = Add()([output, inf_mask])
        output = softmax(output)
        return output
