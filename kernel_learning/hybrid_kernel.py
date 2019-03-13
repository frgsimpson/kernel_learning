# Copyright (C) PROWLER.io 2018
#
# Licensed under the Apache License, Version 2.0

from typing import Tuple

import numpy as np
import tensorflow as tf

from gpflow.params import Parameter, ParamList
from gpflow.transforms import positive
from gpflow.kernels import Kernel
from gpflow import settings, transforms
from gpflow.decors import params_as_tensors

from kernel_learning.primitive_kernels import load_default_basis_kernels

DEFAULT_N_LAYERS = 3
HYBRID_VARIANCE = 1.0


class HybridKernel(Kernel):
    """ A kernel comprised of a non-linear combination of kernels. """

    def __init__(self, input_dim, basis_kernels=None,  n_layers=DEFAULT_N_LAYERS, active_dims=None, name=None):
        """
        Initialise the structure of the neural network in a similar manner to Kernel.
        Additional argument is basis_kernels which is a list of kernels which will be combined to form the hybrid kernel.
        Due to the network architecture, which halves the breadth of the network after each layer, the number of basis
        kernels must be 2^n_layers, or a multiple thereof.
        """

        if basis_kernels is None:
            basis_kernels = load_default_basis_kernels()

        if not all(isinstance(k, Kernel) for k in basis_kernels):
            raise TypeError("Invalid kernel instances")

        super().__init__(input_dim, active_dims, name=name)

        self.kernel_list = basis_kernels
        self.kernel_parameters = ParamList(basis_kernels)

        self.n_basis_kernels = len(basis_kernels)
        self.n_layers = n_layers

        self.variance = Parameter(value=HYBRID_VARIANCE, transform=transforms.positive,
                                  dtype=settings.float_type, trainable=False)

        self._define_topology()
        self._initialise_weights_and_biases()


    @params_as_tensors
    def K(self, X, X2=None) -> tf.Variable:
        """
        Calculates the kernel matrix K(X, X2) by passing the basis kernels through a neural network.

        :param X tensor: N x 1 TensorFlow variable representing the coordinates to be evaluated
        :param X2 tensor: M x 1 TensorFlow variable representing the coordinates to be evaluated
        :return tensor: N x M kernel matrix
        """

        if X2 is None:
            X2 = X

        # If X x X2 is M x N then kernel_stack is M x N x L where L is length of kernel
        kernel_stack = self._build_hybrid_kernel(X, X2)
        output_shape = tf.shape(kernel_stack)[:2]

        # Ensures rank 2 for faster network propagation, so kernel_stack reshaped to (MxN) X L
        kernel_stack = tf.reshape(kernel_stack, [-1, self.n_basis_kernels])
        hybrid_output = self.forward_pass(kernel_stack)

        # Output shape is (MxN) x 1, so need to reshape to desired M x N
        hybrid_output = tf.reshape(hybrid_output, output_shape)

        return self.variance * hybrid_output

    @params_as_tensors
    def Kdiag(self, X) -> tf.Variable:
        """
        Calculates the diagonal components of the kernel matrix.

        :param tensor X: : N x 1 TensorFlow variable representing the coordinates to be evaluated
        :return:
        """

        kernel_stack = self._build_hybrid_kernel_diag(X)
        hybrid_output = self.forward_pass(kernel_stack)

        return self.variance * hybrid_output

    @params_as_tensors
    def forward_pass(self, kernel_stack) -> tf.Variable:
        """
        Propagates the kernel stack through the layers of the network. For now only testing with linear layers.

        :param kernel_stack:
        :return:
        """

        for layer_number in range(self.n_layers):
            kernel_stack = self.linear_layer(kernel_stack, layer_number)
            if (layer_number + 1) < self.n_layers:  # Skip product on final layer
                kernel_stack = self.product_layer(kernel_stack)

        return tf.squeeze(kernel_stack, axis=-1)

    @params_as_tensors
    def linear_layer(self, kernel_stack, layer_number) -> tf.Variable:
        """ Generate a linear combination of the kernels

        :param kernel_stack: Rank 3 tensor
        :param int layer_number: The network layer we wish to propagate through
        :return:
        """

        weights = self.weights[layer_number]
        biases = self.biases[layer_number]

        return tf.matmul(kernel_stack, weights) + biases

    @params_as_tensors
    def product_layer(self, kernel_stack, step_size=2) -> tf.Variable:
        """ Multiplies adjacent kernels with each other.

        :param kernel_stack: Rank 2 tensor
        :param int layer_number: The network layer we wish to propagate through
        :return:
        """

        kernel_stack = tf.reshape(kernel_stack, [tf.shape(kernel_stack)[0], -1, step_size])
        product_stack = tf.reduce_prod(kernel_stack, -1)

        return product_stack

    def get_layer_dims(self, layer_number) -> Tuple[int, int]:
        """ Returns the dimensions required for a given weight matrix. Currently fully connected layers maintain their
         original dimension, while product layers reduce the size of each layer by a factor of two.

        :param int layer_number:
        :return:
        """

        input_dims = self.topology[layer_number]
        output_dims = 1 if layer_number == self.n_layers - 1 else input_dims

        return input_dims, output_dims

    @params_as_tensors
    def _build_hybrid_kernel(self, X, X2=None) -> tf.Variable:
        """  Converts a list of kernels into a single rank 3 tensor
        :param X: N x 1 tensor representing the coordinates to be evaluated
        :param X2: M x 1 tensor representing the coordinates to be evaluated
        :return:  N x M x L tensor representing a collection of kernels
        """

        kx = [k.K(X, X2) for k in self.kernel_list]
        hybrid_kernel = tf.stack(kx, axis=2)

        return hybrid_kernel

    @params_as_tensors
    def _build_hybrid_kernel_diag(self, X):
        """  Converts a list of kernel diagonals into a single rank 2 tensor
        :param X: N x 1 tensor representing the coordinates to be evaluated
        :return:  N x L tensor representing a collection of kernels
        """

        kx = [k.Kdiag(X) for k in self.kernel_list]
        hybrid_kernel = tf.stack(kx, axis=1)

        return hybrid_kernel

    @params_as_tensors
    def _initialise_weights_and_biases(self):
        """ Construct parameters of the neural network. """

        weights = []
        biases = []

        for layer_number in range(self.n_layers):

            weights_name, bias_name = self.get_param_names(layer_number)
            input_dims, output_dims = self.get_layer_dims(layer_number)

            min_w = 1. / (2 * input_dims) / 2.0
            max_w = 3. / (2 * input_dims) / 2.0

            initial_weights = np.random.uniform(low=min_w, high=max_w, size=[input_dims, output_dims]).astype(
                settings.float_type)

            layer_weights = Parameter(initial_weights, transform=positive, name=weights_name)
            layer_biases = Parameter(0.01 * np.ones([output_dims], dtype=settings.float_type),
                                   transform=positive, name=bias_name)

            weights.append(layer_weights)
            biases.append(layer_biases)

        self.weights = ParamList(weights)
        self.biases = ParamList(biases)

    @params_as_tensors
    def _define_topology(self):
        """ Determine size of each layer, shrinks by factor of two due to product layer. """

        min_basis_kernels = 2 ** self.n_layers
        assert self.n_basis_kernels % min_basis_kernels == 0, 'Invalid number of basis kernels, must be a ' \
                                                              'multiple of 2^n_layers'

        topology = np.ones((self.n_layers + 1))
        topology[0] = self.n_basis_kernels

        for layer_number in range(1, self.n_layers):
            topology[layer_number] = topology[layer_number - 1] // 2

        self.topology = topology.astype(int)

    @staticmethod
    def get_param_names(layer_number):
        """ Defines the naming convention of the weights and biases for a given layer of the network.

        :param layer_number:
        :return: str, str  Names of the weights and bias parameters
        """

        weights_name = 'weights' + str(layer_number)
        bias_name = 'bias' + str(layer_number)

        return weights_name, bias_name

    def dump_weights(self, layer_number=0):
        """ Show the value of the weights """

        print("Kernel weights: ", self.weights[layer_number].read_value())
        print("Kernel biases: ", self.biases[layer_number].read_value())
