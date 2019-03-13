# Copyright (C) PROWLER.io 2018
#
# Licensed under the Apache License, Version 2.0

import gpflow
import matplotlib.pyplot as plt
import numpy as np
import pytest
import tensorflow as tf

from kernel_learning.hybrid_kernel import HybridKernel
from kernel_learning.primitive_kernels import load_default_basis_kernels

plt.style.use('ggplot')

DATA_TYPE = 'airline'  # mauna, random, airline
NORMALISE_DATA = True
N_NETWORK_LAYERS = 3  # Default 3
LEARNING_RATE = 1e-2  # Default 1e-3
ITERATIONS = 200000  # Default 200000, runs in 20 seconds

np.random.seed(42)


@pytest.mark.usefixtures("reset_default_tf_graph")
def test_hybrid_kernel(with_tf_session):
    """ Check hybrid kernel """

    with with_tf_session as session:

        kernel_list = load_default_basis_kernels(1.0)
        hybrid_kernel = HybridKernel(input_dim=1, basis_kernels=kernel_list, n_layers=N_NETWORK_LAYERS)

        rng = np.random.RandomState(1)
        X = tf.placeholder(gpflow.settings.float_type)
        X_data = rng.randn(3, 1).astype(gpflow.settings.float_type)

        hybrid_kernel.compile()
        result = session.run(hybrid_kernel.K(X), feed_dict={X: X_data})
        assert(type(result) == np.ndarray), 'Kernel output is not a ndarray'
        total = np.sum(result.flatten())
        assert(not np.isnan(total)), 'Kernel is generating nans'
