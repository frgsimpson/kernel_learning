# Copyright (C) PROWLER.io 2018
#
# Licensed under the Apache License, Version 2.0

# Script to reproduce Figure 3 from Sun et al.
import numpy as np
import gpflow as gf

from kernel_learning.data import load_data, plot_model_performance, median_distance_local, DEFAULT_RANDOM_SEED
from kernel_learning.hybrid_kernel import HybridKernel
from kernel_learning.primitive_kernels import load_default_basis_kernels
from kernel_learning.utils import PrintAction, run_adam

import matplotlib.pyplot as plt
plt.style.use('ggplot')

DATA_TYPE = 'airline'  # mauna, random, airline, sinusoid
NORMALISE_DATA = True
PREDICT_EXTRAPOLATION = True
N_NETWORK_LAYERS = 3  # Default 3
LEARNING_RATE = 1e-2  # Default 1e-3
ITERATIONS = 3000000  # Default 200000, runs in 20 seconds
N_DATA_SAMPLES = 128
EXTRAPOLATION_FRACTION = 1.0  # How far to extrapolate relative to original data

x_raw, y_raw = load_data(DATA_TYPE, normalise_data=False, seed=DEFAULT_RANDOM_SEED, n_samples=N_DATA_SAMPLES)
n_extended_samples = int(len(x_raw) * (1 + EXTRAPOLATION_FRACTION))
x_raw_extended, y_raw_extended = load_data(DATA_TYPE, normalise_data=False, seed=DEFAULT_RANDOM_SEED, n_samples=n_extended_samples)
x_data, y_data = load_data(DATA_TYPE, normalise_data=True, seed=DEFAULT_RANDOM_SEED, n_samples=N_DATA_SAMPLES)

data_lengthscale = median_distance_local(x_data).astype('float32')
dx = x_data[1] - x_data[0]

if PREDICT_EXTRAPOLATION:
    n_extend = int(len(x_data) * EXTRAPOLATION_FRACTION)
    x_extended = x_data[0:n_extend] - np.min(x_data) + np.max(x_data)
    x_predict = np.concatenate([x_data, x_extended])
else:
    x_predict = x_data

# Define kernel network
input_dims = x_data.shape[1]
kernel_list = load_default_basis_kernels(data_lengthscale, input_dims)
hybrid_kernel = HybridKernel(input_dim=1, basis_kernels=kernel_list, n_layers=N_NETWORK_LAYERS)
hybrid_kernel.dump_weights()

# Define model
gpr_model = gf.models.GPR(x_data, y_data, kern=hybrid_kernel)

# Optimise parameters
run_adam(gpr_model, LEARNING_RATE, ITERATIONS, callback=PrintAction(gpr_model, 'GPR with Adam'))
hybrid_kernel.dump_weights()

# Predict
mean, var = gpr_model.predict_y(x_predict)

# Invert normalisation
mean = mean * y_raw.std() + y_raw.mean()
var = var * (y_raw.std() ** 2)

x_raw_test = x_predict * x_raw.std() + x_raw.mean()

# Visualise
plot_model_performance(x_raw_extended, y_raw_extended, x_raw_test, mean, var)

xmin = x_raw_test.min() - 0.1
xmax = x_raw_test.max()
ymin = y_raw_extended.min() - 3
ymax = y_raw_extended.max() + 3

if DATA_TYPE == 'airline':
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    fig = plt.gcf()
    fig.set_size_inches(14, 6)

plt.show()
