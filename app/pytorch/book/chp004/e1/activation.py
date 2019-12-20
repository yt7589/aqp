from __future__ import print_function, division
import math
import numpy as np
import copy
from layer import Layer
import layer_common as allc

class Activation(Layer):
    """A layer that applies an activation operation to the input.

    Parameters:
    -----------
    name: string
        The name of the activation function that will be used.
    """

    def __init__(self, name):
        self.activation_name = name
        self.activation_func = allc.activation_functions[name]()
        self.trainable = True
        self.input_shape = None

    def layer_name(self):
        return "Activation (%s)" % (self.activation_func.__class__.__name__)

    def forward_pass(self, X, Y=None, training=True):
        self.layer_input = X
        a = self.activation_func(X)
        return a, a

    def backward_pass(self, accum_grad):
        return accum_grad * self.activation_func.gradient(self.layer_input)

    def output_shape(self):
        return self.input_shape