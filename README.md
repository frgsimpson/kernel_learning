# Kernel Learning

This package is a [GPflow](http://github.com/GPflow) implementation of , originally presented in a paper entitled ['Differentiable Compositional Kernel Learning for Gaussian Processes' by Sun et al.](https://arxiv.org/abs/1806.04326).


## What does GPflow do?

GPflow implements modern Gaussian process inference for composable kernels and likelihoods. The [online user manual (develop)](http://gpflow.readthedocs.io/en/develop/)/[(master)](http://gpflow.readthedocs.io/en/master/) contains more details. The interface follows on from [GPy](http://github.com/sheffieldml/gpy), and the docs have further [discussion of the comparison](http://gpflow.readthedocs.io/en/develop/intro.html#what-s-the-difference-between-gpy-and-gpflow).

GPflow uses [TensorFlow](http://www.tensorflow.org) for running computations, which allows fast execution on GPUs, and uses Python 3.5 or above.

## Install

The package can be installed by cloning the repository and running
```
pip install -r requirements.txt
python setup.py install
```
in the root folder.

## Demo

Once installed, you should be able to run the script '/demo/run_hybrid_kernels.py', which should generate the figure below.


![Alt text](./demo/img.jpg?raw=true "Extrapolation of airline passenger data.")

