# Copyright (C) PROWLER.io 2018
#
# Licensed under the Apache License, Version 2.0

import os
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

DEFAULT_N_SAMPLES = 12
DEFAULT_RANDOM_SEED = 42
OMEGA_1 = 12
OMEGA_2 = 25
AMPLITUDE_1 = 1.0
AMPLITUDE_2 = 0.66
RELATIVE_NOISE_AMPLITUDE = 1e-2
MOCK_GAUSSIAN_PARAMS = [[0.001], [1.25, 0.25, 2.0]]  # Constant amplitude, then mean, beta, and amplitude of gaussian

LOCAL_PATH = os.path.dirname(os.path.realpath(__file__))
AIRLINE_DATAFILE = LOCAL_PATH + '/data/airline_passengers.txt'
MAUNA_DATAFILE = LOCAL_PATH + '/data/maunaloa_full.txt'


def load_train_test_data(data_name, seed=None, n_samples=DEFAULT_N_SAMPLES, extrap_fracion=1):
    """ Returns normalised training data and unnormalised test data. """

    x, y, norm_constants = load_data(data_name, True, seed, n_samples, return_normalisation=True)
    n_extend = int(len(x) * (1 + extrap_fracion))
    n_extra = n_extend - len(x)

    x_raw_test, y_raw_test = load_data(data_name, False, seed, n_extend)

    delta_x = x[1] - x[0]
    x_extended = x[:n_extra] - np.min(x) + np.max(x) + delta_x
    x_predict = np.concatenate([x, x_extended])

    return x, y, x_predict, x_raw_test, y_raw_test, norm_constants


def load_2d_train_test_data(data_name, n_samples=DEFAULT_N_SAMPLES, extrap_fracion=1,  seed=None):
    """ Returns normalised training data and unnormalised test data. """

    x, y, norm_constants = load_data(data_name, True, seed, n_samples, return_normalisation=True)
    n_extend = int(len(x[1]) * (1 + extrap_fracion))
    x_raw_test, y_raw_test = load_data(data_name, False, seed, n_extend)

    x_mu = norm_constants[0]
    x_sigma = norm_constants[1]

    x_predict = (x_raw_test - x_mu) / x_sigma

    return x, y, x_predict, x_raw_test, y_raw_test, norm_constants


def load_data(data_name, normalise_data=True, seed=None, n_samples=DEFAULT_N_SAMPLES, trend_gradient=None,
              return_normalisation=False):
    """  Prepare normalised numpy arrays of NxD with N samples and D dimensions.

    :param str data_name:
    :param bool normalise_data:
    :param int seed:
    :param int n_samples: Synthetic datasets have optional number of samples
    :param float trend_gradient: Option to impart a linear gradient in y(x)
    :return:
    """

    if seed:
        np.random.seed(seed)

    if data_name == 'airline':
        x, y = load_airline_data(n_samples)
    elif data_name == 'random':
        x, y = load_random_data(n_samples)
    elif data_name == 'mauna':
        x, y = load_maunaloa_data(n_samples)
    elif data_name == 'sinusoid':
        x, y = load_sinusoidal_data(n_samples)
    elif data_name == 'pattern':
        x, y = load_pattern_data(n_samples)
    elif data_name == 'symmetric_pattern':
        x, y = load_pattern_data(n_samples, theta=0.0)
    elif data_name == 'vertical_lines':
        x, y = load_asymmetric_pattern_data(n_samples, theta=0.0)
    elif data_name == 'tilted_lines':
        x, y = load_asymmetric_pattern_data(n_samples)
    else:
        raise NotImplementedError('Unknown data requested: ', data_name)

    # Ensure arrays are 2D
    if x.ndim == 1:
        x = x[:, None]

    if y.ndim == 1:
        y = y[:, None]

    if trend_gradient:
        y += trend_gradient * x

    if return_normalisation:
        norm_constants = get_norm_constants(x, y)

    if normalise_data:
        x = normalise(x)
        y = normalise(y)

    if return_normalisation:
        return x, y, norm_constants
    else:
        return x, y


def get_norm_constants(x, y):

    mu_x = np.mean(x)
    sigma_x = np.std(x)
    mu_y = np.mean(y)
    sigma_y = np.std(y)
    norm_constants = [mu_x, sigma_x, mu_y, sigma_y]

    return norm_constants


def normalise(x: np.ndarray) -> np.ndarray:
    """ Rescale data to zero mean and unit variance.

    :param ndarray x:
    :return: ndarray: Normalised along
    """

    x = (x - np.mean(x)) / x.std()

    return x - np.mean(x)


def load_sinusoidal_data(n_samples: int, linearly_spaced_x: bool=True)-> Tuple[np.ndarray, np.ndarray]:
    """ Returns noisy data based on linear combination of sin and cos. """

    if linearly_spaced_x:
        dx = 0.03
        xmin = 0.0
        xmax = dx * (n_samples - 1)
        x = np.linspace(xmin, xmax, n_samples)
    else:
        x = np.random.rand(n_samples, 1)

    mean_offset = 3.0
    y = AMPLITUDE_1 * np.sin(OMEGA_1 * x) + AMPLITUDE_2 * np.cos(OMEGA_2 * x)

    # Add noise
    noise_amplitude = RELATIVE_NOISE_AMPLITUDE * y.std()
    noise = np.random.randn(n_samples) * noise_amplitude
    y += noise

    # Displace mean
    y += mean_offset

    return x, y


def load_random_data(n_samples: int)-> Tuple[np.ndarray, np.ndarray]:

    x_data = np.random.rand(n_samples, 1)
    y_data = np.sin(x_data*6) + np.random.randn(*x_data.shape) * 0.001

    return x_data, y_data


def load_airline_data(n_samples: int)-> Tuple[np.ndarray, np.ndarray]:
    """X is a vector giving the time step, and y is the total number of international
    airline passengers, in thousands. Each element corresponds to one
    month, and it goes from Jan. 1949 through Dec. 1960."""

    y_values = np.loadtxt(AIRLINE_DATAFILE).flatten().astype(float)[0:n_samples]
    x_values = 1949 + 1/12 * np.arange(y_values.size).astype(float)
    return x_values, y_values


def load_maunaloa_data(n_samples: int, load_interpolated_data: bool=True)-> Tuple[np.ndarray, np.ndarray]:
    """ Extracts """

    data = np.loadtxt(MAUNA_DATAFILE)

    # Columns from full text file are: 0 year; 1 month; 2 numerical date; 3 average; 4 interpolation; 5 trend
    y_column = 4 if load_interpolated_data else 3
    y = data[:, y_column].flatten()

    year = data[:, 0].flatten()
    month = data[:, 1].flatten()

    x = year + (month - 0.5) / 12.0

    return x[0:n_samples], y[0:n_samples]


def load_pattern_data(n_samples: int, theta=0.3, noise_amplitude=1e-4)-> Tuple[np.ndarray, np.ndarray]:
    """ 2D function defined in Sun et al, 1806.04326, but added missing np.abs() """

    xmin = - (n_samples / 5)  # Must increase with n_samples to allow extrapolation
    xmax = (n_samples / 5)

    x1 = np.linspace(xmin, xmax, n_samples)
    x2 = np.linspace(xmin, xmax, n_samples)

    x_grid1, x_grid2 = np.meshgrid(x2, x1)

    x = np.array([x_grid1, x_grid2])

    # Rotate the coordinate system
    z_grid1 = x_grid1 * np.cos(theta) - x_grid2 * np.sin(theta)
    z_grid2 = x_grid2 * np.cos(theta) + x_grid1 * np.sin(theta)

    y = (np.cos(2 * z_grid1) + np.cos(2 * z_grid2))

    noise = np.random.randn(n_samples**2) * noise_amplitude
    y += noise.reshape(y.shape)

    return x, y


def load_asymmetric_pattern_data(n_samples: int, theta=0.2, noise_amplitude=1e-4)-> Tuple[np.ndarray, np.ndarray]:
    """ 2D function defined in Sun et al, 1806.04326, but added missing np.abs() """

    xmin = - (n_samples / 5)  # Must increase with n_samples to allow extrapolation
    xmax = (n_samples / 5)

    x1 = np.linspace(xmin, xmax, n_samples)
    x2 = np.linspace(xmin, xmax, n_samples)

    x_grid1, x_grid2 = np.meshgrid(x2, x1)

    x = np.array([x_grid1, x_grid2])

    # Rotate the coordinate system
    z_grid1 = x_grid1 * np.cos(theta) - x_grid2 * np.sin(theta)
    z_grid2 = x_grid2 * np.cos(theta) + x_grid1 * np.sin(theta)

    y = (np.cos(2 * z_grid1))

    noise = np.random.randn(n_samples**2) * noise_amplitude
    y += noise.reshape(y.shape)

    return x, y


def plot_model_performance(x_data, y_data, x_test, mean, var, n_sigma=1, norm_constants=None):
    """  Visualise the data and the prediction as a confidence interval.

    :param x_data: Actual x values
    :param y_data: Actual y values
    :param x_test: X values to make predicitons
    :param mean: The mean of the prediction
    :param var: The variance of the prediction
    :param n_sigma: How many standard deviations are represented by the shaded area
    """

    if norm_constants is not None:
        mu_y = norm_constants[2]
        sigma_y = norm_constants[3]
        mean = mean * sigma_y + mu_y
        var *= sigma_y ** 2

    plt.plot(x_data, y_data, 'kx', mew=2)
    line, = plt.plot(x_test, mean, lw=2)
    plt.fill_between(x_test[:, 0], mean[:, 0] - n_sigma * np.sqrt(var[:, 0]), mean[:, 0] + n_sigma * np.sqrt(var[:, 0]), color=line.get_color(), alpha=0.2)


def plot_2d_model_performance(x_raw, y_raw, x_predict, mean_prediction, var, plot_residuals=True, show_colorbar=True):

    n_subplots = 3 if plot_residuals else 2

    fig, ax = plt.subplots(1, n_subplots)
    im = ax[0].imshow(y_raw, cmap=plt.get_cmap('bone'))
    clim = im.properties()['clim']
    ax[0].set_title('Truth')

    ax[1].imshow(mean_prediction, cmap=plt.get_cmap('bone'), clim=clim)
    ax[1].set_title('Prediction')

    if plot_residuals:
        residuals = mean_prediction - y_raw
        residual_image = ax[2].imshow(residuals, cmap=plt.get_cmap('hot'))
        ax[2].set_title('Residuals')

    if show_colorbar:
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.3, 0.05, 0.35])
        fig.colorbar(residual_image, cax=cbar_ax)

    plt.show()


def plot_2d_samples(samples, do_log=False):
    """ Illustrate a list of images. """

    n_subplots = len(samples)
    fig, ax = plt.subplots(1, n_subplots)

    for i in range(n_subplots):
        img = np.log(samples[i]) if do_log else samples[i]
        ax[i].imshow(img, cmap=plt.get_cmap('bone'))

    plt.show()


def median_distance_local(x: np.ndarray) -> np.ndarray:
    """  Get the median of distances between x, x will be subsampled if very large

    :param x: shape of [n, d]
    """
    if x.shape[0] > 10000:
        permutation = np.random.permutation(x.shape[0])
        x = x[permutation[:10000]]
    x_col = np.expand_dims(x, 1)
    x_row = np.expand_dims(x, 0)
    dis_a = np.abs(x_col - x_row)
    dis_a = np.reshape(dis_a, [-1, dis_a.shape[-1]])
    return np.median(dis_a, 0) * (x.shape[1] ** 0.5)
