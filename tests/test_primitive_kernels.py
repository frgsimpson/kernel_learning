# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
import pytest
from gpflow.kernels import Kernel

from kernel_learning import primitive_kernels as pk


@pytest.mark.usefixtures("reset_default_tf_graph")
def test_load_default_basis_kernels(with_tf_session):

    with with_tf_session as session:

        kernel_list = pk.load_default_basis_kernels()

        for basis_kernel in kernel_list:
            assert(isinstance(basis_kernel, Kernel))
