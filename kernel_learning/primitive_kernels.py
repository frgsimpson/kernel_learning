# Copyright (C) PROWLER.io 2018
#
# Licensed under the Apache License, Version 2.0

# Defines the kernels to be incorporated in the input layer
from typing import List

import gpflow.kernels as gk

KERNEL_DICT = dict(
    White=gk.White,
    Constant=gk.Constant,
    ExpQuad=gk.RBF,
    RBF=gk.RBF,
    Matern12=gk.Matern12,
    Matern32=gk.Matern32,
    Matern52=gk.Matern52,
    Cosine=gk.Cosine,
    ArcCosine=gk.ArcCosine,
    Linear=gk.Linear,
    Periodic=gk.Periodic,
    RatQuad=gk.RationalQuadratic,
)


def load_default_basis_kernels(ls=1.0, input_dims=1) -> List:
    """  Define list of kernels to be used as input layer for the kernel network.
    Matches setup in Neural-Kernel-Network-master

    :param float ls: Initial lengthscale
    :return: List of kernels
    """

    kernel_params_list = [
    {'name': 'Linear', 'params': {'input_dim': input_dims, 'name': 'k0'}},
    {'name': 'Periodic', 'params': {'input_dim': input_dims, 'period': ls, 'lengthscales': ls, 'name': 'k1'}},
    {'name': 'ExpQuad', 'params': {'input_dim': input_dims, 'lengthscales': ls / 4.0, 'name': 'k2'}},
    {'name': 'RatQuad', 'params': {'input_dim': input_dims, 'alpha': 0.2, 'lengthscales': ls * 2.0, 'name': 'k3'}},
    {'name': 'Linear', 'params': {'input_dim': input_dims, 'name': 'k4'}},
    {'name': 'RatQuad', 'params': {'input_dim': input_dims, 'alpha': 0.1, 'lengthscales': ls, 'name': 'k5'}},
    {'name': 'ExpQuad', 'params': {'input_dim': input_dims, 'lengthscales': ls, 'name': 'k6'}},
    {'name': 'Periodic',
     'params': {'input_dim': input_dims, 'period': ls / 4.0, 'lengthscales': ls / 4.0, 'name': 'k7'}}]

    kernel_list = []
    for kernel_param_dict in kernel_params_list:
        kernel_name = str(kernel_param_dict['name'])
        kernel_params = kernel_param_dict['params']
        kernel = KERNEL_DICT[kernel_name]
        initialised_kernel = kernel(**kernel_params)

        kernel_list.append(initialised_kernel)

    return kernel_list
