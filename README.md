# Kernel Learning

Gaussian Processes offer a versatile tool for modelling the underlying behaviour of partially observed systems. However it can be challenging to find an appropriate choice of kernel for a given problem. One approach is to construct a kernel by combining a broad variety of simpler kernels, via a sequence of layers which resembles a neural network.  The technique was originally presented in a paper entitled ['Differentiable Compositional Kernel Learning for Gaussian Processes' by Sun et al](https://arxiv.org/abs/1806.04326). The implementation in this repository is designed to be used as part of the [GPflow](http://github.com/GPflow) package.

## Install

The package can be installed by cloning the repository and running the following commands from the root folder:
```
pip install -r requirements.txt
python setup.py install
```

## Demo

Once installed, you should be able to run the script '/demo/run_hybrid_kernels.py', which produces the figure below.


![Alt text](./demo/airline.png?raw=true "Extrapolation of airline passenger data.")
