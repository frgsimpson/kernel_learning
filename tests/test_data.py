# Copyright (C) PROWLER.io 2018
#
# Licensed under the Apache License, Version 2.0

import numpy as np

from kernel_learning import data

DATA_TYPES = ['airline', 'mauna', 'random']


def test_load_data():
    for data_type in DATA_TYPES:
        x_raw, y_raw = data.load_data(data_type, normalise_data=False)
        assert isinstance(x_raw, np.ndarray)
        assert isinstance(y_raw, np.ndarray)


def test_normalise():
    x = np.random.normal(loc=10.00, scale=1.8, size=10)

    normalised_x = data.normalise(x)
    mean_x = normalised_x.mean()
    std_x = normalised_x.std()

    assert np.isclose(mean_x, 0.0)
    assert np.isclose(std_x, 1.0)

