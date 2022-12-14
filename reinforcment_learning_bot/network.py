import tensorflow as tf
from tensorflow import keras

from convolutional_layer import ConvolutionalLayer
from residual_layer import ResidualLayer
from value_head import ValueHead
from policy_head import PolicyHead


class RLModel(keras.Model):
    def __init__(self, n_actions: int, depth: int = 40, **kwargs):
        super().__init__(**kwargs)
        self.depth = depth
        self.convolutional_layer = ConvolutionalLayer()
        self.residual_layers = []
        for _ in range(self.depth):
            self.residual_layers.append(ResidualLayer())
        self.value_head = ValueHead()
        self.policy_head = PolicyHead(n_actions)

    def call(self, input):
        value_input, mask_input = input
        layer = self.convolutional_layer(value_input)
        for i in range(self.depth):
            layer = self.residual_layers[i](layer)
        value_output = self.value_head(layer)
        policy_output = self.policy_head([layer, mask_input])
        return value_output, policy_output
