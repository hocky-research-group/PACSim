# Ionic Colloids

Install and setup [Git Large File Storage](https://git-lfs.com) (used for the large files in the Literature directory.)

## Dependencies

If you only want to use the openmm part of this package, you can use Python 3.12 (or older).

If you want to use the Hoomd part of this package, use Python 3.10 (or older) because that is the latest Python version 
that is supported by Hoomd < 3 .

We recommend installing Python and the required packages using 
[Anaconda](https://www.anaconda.com/products/distribution). 

The following packages are required:

- jupyterlab >= 4.1 
- matplotlib >= 3.8
- numpy >= 1.26
- openmm >= 8.0
- pytest >= 7.4
- gsd >= 3.2
- hoomd == 2.9.7

## Installation

Clone the repository and install the package in editable mode in your virtual environment using pip:

```bash
pip install -e .
```

Note that this attempts to install the requirements with pip, if you did not install them yourself before. However, 
because hoomd is not available on PyPI, you need to install it manually (or via conda).
