# Ionic Colloids

Install and setup [Git Large File Storage](https://git-lfs.com) (used for the large files in the Literature directory.)

## Dependencies

If you only want to use the openmm part of this package, you can use Python 3.12 (or any older version >= 3.7).

If you want to use the Hoomd part of this package, use Python 3.10 (or any older version >= 3.7) because that is the 
latest Python version that is supported by Hoomd < 3.

We recommend installing Python and the required packages using 
[Anaconda](https://www.anaconda.com/products/distribution). 

The following packages are required:

- jupyterlab >= 4.1 
- matplotlib >= 3.8
- numpy >= 1.26
- openmm >= 8.0
- pytest >= 7.4
- gsd >= 3.2
- pyyaml >= 6.0
- tqdm >= 4.65
- pandas >= 2.2
- hoomd == 2.9.7

## Installation

Clone the repository and install the package in editable mode in your virtual environment using pip:

```bash
pip install -e .
```

Note that this attempts to install the requirements with pip, if you did not install them yourself before. However, 
because hoomd is not available on PyPI, you need to install it manually (or via conda).

## Usage

The installation process creates two executables `colloids-run` and `colloids-resume`. You might have to add the 
directory where pip installs executables to your PATH environment variable in order to access these executables.

### colloids-run

The `colloids-run` executable is used to run simulations. It expects a configuration file in yaml format as the only 
positional argument:

```bash
colloids-run run.yaml
```

An exemplary configuration file called `example.yaml` can be created with the command 
`colloids-run --example`. Another exemplary configuration file is provided in `colloids/run.yaml`.

### colloids-resume
A simulation that is run with the `colloids-run` executable creates checkpoints in periodic intervals. One can resume a 
simulation from a checkpoint using the `colloids-resume` executable. It expects the original configuration file (because
the checkpoint file only stores the positions and velocities of the particles in an OpenMM context), the checkpoint 
file, and the number of time steps that should be run (the corresponding value in the configuration file is ignored). 
For example, use the following command to continue a simulation for 100000 time steps:

```bash
colloids-resume run.yaml checkpoint.chk 100000
```
