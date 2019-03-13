from setuptools import find_packages, setup

requirements = [
    'numpy>=1.10.0',
]

setup(name='kernel_learning',
      version='0.1',
      author="Fergus Simpson",
      author_email="fergus@prowler.io",
      description=("GPflow implementation of 'Differentiable kernel learning'"),
      license="Apache License 2.0",
      keywords="machine-learning",
      url="https://github.com/frgsimpson",
      python_requires=">=3.5",
      packages=find_packages(include=["kernel_learning",
                                      "kernel_learning.*"]),
      install_requires=requirements,
      include_package_data=True,
      classifiers=[
          'License :: OSI Approved :: Apache Software License',
          'Natural Language :: English',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Artificial Intelligence'
      ])
